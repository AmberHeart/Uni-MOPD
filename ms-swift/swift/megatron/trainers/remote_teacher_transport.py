# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, MOPD contributors.
"""Bounded HTTP control/binary BF16 transport for an independent teacher worker.

HTTP handler threads never run CUDA or distributed collectives. The process
which owns the Megatron teacher calls `serve`, and every `score` callback is
executed serially in that caller. The training client can prefetch the next
chunk through `StreamingTeacherWindow` while the student runs its current
forward/backward. Network inputs are JSON and safetensors, never pickle.
"""
from dataclasses import asdict, dataclass, field
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import queue
import re
import threading
import time
from typing import Callable, Dict, List, Mapping, Optional, Tuple
import urllib.error
import urllib.parse
import urllib.request
import uuid

from .remote_teacher_protocol import (
    PROTOCOL_VERSION, HiddenResult, ScoreRequest, TeacherIdentity, WindowIdentity,
    decode_hidden_response, encode_hidden_response, require,
)


class TeacherRPCError(RuntimeError):
    pass


@dataclass
class _Work:
    request: ScoreRequest
    done: threading.Event = field(default_factory=threading.Event)
    payload: Optional[bytes] = None
    error: Optional[str] = None


class TeacherRPCService:
    """One teacher-bank lease, one ordered GPU forward, bounded queued work."""

    def __init__(
        self,
        score: Callable[[ScoreRequest], Tuple[List[HiddenResult], Dict[str, float]]],
        teachers: Mapping[str, TeacherIdentity],
        *,
        auth_token: str,
        max_request_bytes: int = 64 * 1024**2,
        max_response_bytes: int = 4 * 1024**3,
        timeout_seconds: float = 600,
        on_shutdown: Callable[[], None] = None,
        on_idle: Callable[[], None] = None,
        idle_interval_seconds: Optional[float] = None,
    ):
        require(isinstance(auth_token, str) and re.fullmatch(r'[A-Za-z0-9_-]{16,128}', auth_token) is not None,
                'Teacher service needs a private single-line token')
        require(bool(teachers) and all(key == teacher.key for key, teacher in teachers.items()),
                'Invalid teacher bank identities')
        require(type(max_request_bytes) is int and max_request_bytes > 0
                and type(max_response_bytes) is int and max_response_bytes > 0
                and math.isfinite(timeout_seconds) and timeout_seconds > 0,
                'Invalid teacher RPC resource limits')
        self.score = score
        self.teachers = dict(teachers)
        self.auth_token = auth_token
        self.session_id = uuid.uuid4().hex
        self.max_request_bytes = max_request_bytes
        self.max_response_bytes = max_response_bytes
        self.timeout_seconds = timeout_seconds
        self.on_shutdown = on_shutdown
        self.on_idle = on_idle
        self.idle_interval_seconds = (
            min(30., timeout_seconds / 4) if idle_interval_seconds is None else idle_interval_seconds)
        require(math.isfinite(self.idle_interval_seconds)
                and 0 < self.idle_interval_seconds < timeout_seconds, 'Invalid teacher idle-heartbeat interval')
        self.lock = threading.Lock()
        self.jobs = queue.Queue(maxsize=1)
        self.window = None
        self.next_chunk = 0
        self.delivered_samples = 0
        self.seen_samples = set()
        self.busy = False
        self.broken = False
        self.stopping = False
        self.server = None
        self.http_thread = None

    def start_http(self, host: str, port: int) -> int:
        require(self.server is None, 'Teacher HTTP service was already started')
        owner = self

        class Server(ThreadingHTTPServer):
            def get_request(self):
                connection, address = super().get_request()
                connection.settimeout(owner.timeout_seconds)
                return connection, address

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass  # Never log Authorization headers or potentially large inputs.

            def _reply(self, status, payload, content_type='application/json'):
                if content_type == 'application/json':
                    payload = json.dumps(payload, allow_nan=False).encode()
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _authorized(self):
                expected = 'Bearer ' + owner.auth_token
                if not hmac.compare_digest(self.headers.get('Authorization', ''), expected):
                    self._reply(401, {'error': 'Unauthorized teacher request'})
                    return False
                return True

            def do_GET(self):
                if not self._authorized():
                    return
                if self.path != '/health':
                    self._reply(404, {'error': 'Unknown endpoint'})
                    return
                with owner.lock:
                    ready = not owner.broken and not owner.stopping
                    self._reply(200 if ready else 503, {
                        'protocol_version': PROTOCOL_VERSION, 'session_id': owner.session_id,
                        'ready': ready,
                        'clock_unix_seconds': time.time(),
                        'teachers': {key: asdict(value) for key, value in owner.teachers.items()},
                    })

            def do_POST(self):
                if not self._authorized():
                    return
                try:
                    length = int(self.headers.get('Content-Length', '-1'))
                    require(0 <= length <= owner.max_request_bytes, 'Teacher request exceeds its byte limit')
                    raw = self.rfile.read(length)
                    require(len(raw) == length, 'Incomplete teacher request')
                    body = json.loads(raw)
                    result = owner._handle(self.path, body)
                    if isinstance(result, bytes):
                        self._reply(200, result, 'application/octet-stream')
                    else:
                        self._reply(200, result)
                except (ValueError, KeyError, TypeError) as error:
                    self._reply(400, {'error': str(error)})
                except TeacherRPCError as error:
                    self._reply(503, {'error': str(error)})

        self.server = Server((host, port), Handler)
        self.server.daemon_threads = True
        self.http_thread = threading.Thread(
            target=self.server.serve_forever, kwargs={'poll_interval': .05},
            name='mopd-teacher-http', daemon=True)
        self.http_thread.start()
        return self.server.server_address[1]

    def _check_window(self, body):
        require(body['session_id'] == self.session_id, 'Teacher server session changed')
        require(self.window is not None and WindowIdentity(**body['window']) == self.window,
                'Wrong teacher window or policy version')

    def _handle(self, path, body):
        if path == '/open':
            with self.lock:
                if self.broken or self.stopping:
                    raise TeacherRPCError('Teacher worker is broken; restart the owned runtime')
                require(body['session_id'] == self.session_id, 'Teacher server session changed')
                require(self.window is None and not self.busy, 'Previous teacher window is not closed')
                require(body['teachers'] == {key: teacher.fingerprint for key, teacher in self.teachers.items()},
                        'Teacher bank does not match the training heads')
                self.window = WindowIdentity(**body['window'])
                self.next_chunk = self.delivered_samples = 0
                self.seen_samples = set()
                return {'status': 'opened', 'session_id': self.session_id}
        if path == '/score':
            request = ScoreRequest.from_dict(body['request'])
            with self.lock:
                self._check_window({'session_id': body['session_id'], 'window': asdict(request.window)})
                if self.broken:
                    raise TeacherRPCError('Teacher worker is broken')
                require(not self.busy and request.chunk_index == self.next_chunk,
                        'Teacher chunks must execute once, in order')
                ids = {binding.sample_id for binding in request.bindings}
                require(not ids & self.seen_samples and len(self.seen_samples) + len(ids) <= self.window.samples,
                        'Repeated or excess teacher sample occurrences')
                require(all(binding.teacher_key in self.teachers for binding in request.bindings),
                        'Unknown routed teacher')
                self.seen_samples.update(ids)
                self.busy = True
                work = _Work(request)
                self.jobs.put_nowait(work)
            if not work.done.wait(self.timeout_seconds):
                with self.lock:
                    self.broken = True
                raise TeacherRPCError('Teacher forward timed out; no new window is allowed')
            if work.error:
                raise TeacherRPCError(work.error)
            return work.payload
        if path in ('/close', '/abort'):
            with self.lock:
                self._check_window(body)
                require(not self.busy, 'Settle the in-flight teacher forward before closing/aborting')
                if path == '/close':
                    require(not self.broken and self.next_chunk == self.window.chunks
                            and self.delivered_samples == self.window.samples,
                            'Teacher window is incomplete; optimizer update is forbidden')
                self.window = None
                return {'status': 'closed' if path == '/close' else 'aborted'}
        if path == '/shutdown':
            with self.lock:
                require(body['session_id'] == self.session_id and self.window is None and not self.busy,
                        'Cannot shut down an active teacher window')
                require(not self.stopping, 'Teacher is already stopping')
                self.stopping = True
                self.jobs.put_nowait(None)
            return {'status': 'stopping'}
        raise ValueError('Unknown teacher RPC endpoint')

    def serve(self):
        """Run on the native teacher owner thread, AFTER all GPU ranks initialize."""
        require(self.server is not None, 'Start HTTP control before serving GPU work')
        try:
            while True:
                try:
                    work = self.jobs.get(timeout=self.idle_interval_seconds if self.on_idle is not None else None)
                except queue.Empty:
                    # The native owner thread participates in the heartbeat.
                    # Never issue GPU/collective work from an HTTP/timer thread.
                    self.on_idle()
                    continue
                if work is None:
                    if self.on_shutdown is not None:
                        self.on_shutdown()
                    return
                try:
                    results, metrics = self.score(work.request)
                    require([result.binding for result in results] == work.request.bindings,
                            'Teacher forward returned different sample bindings')
                    for result in results:
                        identity = self.teachers[result.binding.teacher_key]
                        require(result.teacher_fingerprint == identity.fingerprint,
                                'Teacher forward used a different checkpoint/head')
                        require(tuple(result.hidden.shape) == (len(result.binding.loss_positions), identity.hidden_size),
                                'Teacher forward returned invalid hidden dimensions')
                    payload = encode_hidden_response(
                        work.request.window, work.request.chunk_index, results,
                        session_id=self.session_id, max_bytes=self.max_response_bytes, metrics=metrics)
                    with self.lock:
                        work.payload = payload
                        self.next_chunk += 1
                        self.delivered_samples += len(results)
                except Exception as error:
                    with self.lock:
                        self.broken = True
                        work.error = f'Teacher chunk {work.request.chunk_index} failed: {type(error).__name__}: {error}'
                finally:
                    with self.lock:
                        self.busy = False
                    work.done.set()
        finally:
            self.server.shutdown()
            self.server.server_close()
            self.http_thread.join(timeout=5)


class TeacherHTTPClient:
    def __init__(
        self, base_url: str, teachers: Optional[Mapping[str, TeacherIdentity]], *,
        auth_token: str, timeout_seconds: float = 600, max_response_bytes: int = 4 * 1024**3,
    ):
        parsed = urllib.parse.urlsplit(base_url)
        require(parsed.scheme in ('http', 'https') and parsed.hostname is not None
                and parsed.username is None and parsed.password is None
                and not parsed.query and not parsed.fragment, 'Invalid teacher endpoint')
        require(math.isfinite(timeout_seconds) and timeout_seconds > 0
                and type(max_response_bytes) is int and max_response_bytes > 0, 'Invalid teacher client limits')
        require(isinstance(auth_token, str) and re.fullmatch(r'[A-Za-z0-9_-]{16,128}', auth_token) is not None,
                'Teacher client needs a private single-line token')
        self.base_url = base_url.rstrip('/')
        self.teachers = None
        self.validated = False
        self.auth_token = auth_token
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes
        # This RPC stays inside the training job's own network, so it must bypass any
        # HTTP proxy configured in the environment.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        before = time.time()
        hello = self._request('/health')
        after = time.time()
        require(hello['protocol_version'] == PROTOCOL_VERSION and hello['ready'], 'Teacher server is not ready')
        actual = {key: TeacherIdentity.from_dict(value) for key, value in hello['teachers'].items()}
        self.teachers = actual
        if teachers is not None:
            self.bind_teachers(teachers)
        self.session_id = hello['session_id']
        # A bounded clock-offset interval, not an assumed synchronized clock.
        # Used only to interpret cross-node diagnostic timelines.
        clock = hello['clock_unix_seconds']
        require(type(clock) in (int, float) and math.isfinite(clock) and after >= before,
                'Teacher clock observation is invalid')
        self.clock_offset_bounds = (clock - after, clock - before)
        self.window = None

    def bind_teachers(self, expected):
        require(dict(expected) == self.teachers, 'Teacher server/head/tokenizer identities disagree')
        self.validated = True

    def _request(self, path, value=None, *, binary=False):
        data = None if value is None else json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
        request = urllib.request.Request(
            self.base_url + path, data=data,
            headers={'Authorization': 'Bearer ' + self.auth_token, 'Content-Type': 'application/json'})
        limit = self.max_response_bytes if binary else 16 * 1024**2
        try:
            with self.opener.open(request, timeout=self.timeout_seconds) as response:
                length = int(response.headers.get('Content-Length', '-1'))
                require(0 <= length <= limit, 'Teacher response exceeds its declared byte limit')
                payload = response.read(length)
                require(len(payload) == length, 'Teacher response was truncated')
        except urllib.error.HTTPError as error:
            message = error.read(65536).decode('utf-8', errors='replace')
            raise TeacherRPCError(f'Teacher RPC {path} returned HTTP {error.code}: {message}') from error
        return payload if binary else json.loads(payload)

    def open(self, window: WindowIdentity):
        require(self.validated, 'Validate the local teacher heads before opening a window')
        require(self.window is None, 'Client already owns a teacher window')
        result = self._request('/open', {
            'session_id': self.session_id, 'window': asdict(window),
            'teachers': {key: value.fingerprint for key, value in self.teachers.items()},
        })
        require(result['status'] == 'opened', 'Teacher window was not acknowledged')
        self.window = window

    def score(self, request: ScoreRequest):
        require(self.window == request.window, 'Client request does not belong to its active window')
        payload = self._request('/score', {
            'session_id': self.session_id, 'request': request.to_dict(),
        }, binary=True)
        return decode_hidden_response(
            payload, request.window, request.chunk_index, request.bindings, self.teachers,
            session_id=self.session_id, max_bytes=self.max_response_bytes)

    def close(self):
        require(self.window is not None, 'No active teacher window')
        result = self._request('/close', {'session_id': self.session_id, 'window': asdict(self.window)})
        require(result['status'] == 'closed', 'Teacher window close was not acknowledged')
        self.window = None

    def abort(self):
        if self.window is None:
            return
        result = self._request('/abort', {'session_id': self.session_id, 'window': asdict(self.window)})
        require(result['status'] == 'aborted', 'Teacher abort was not acknowledged')
        self.window = None

    def shutdown(self):
        require(self.window is None, 'Finish/abort the current window before stopping its teacher')
        result = self._request('/shutdown', {'session_id': self.session_id})
        require(result['status'] == 'stopping', 'Teacher shutdown was not acknowledged')
