# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, MOPD contributors.
"""CPU-only, current-policy complete-response buffering.

This is not token streaming and never invokes a model, collective or optimizer.
The producer transfers ownership of each JSON response. Queue credits include
the consumer's current lease until its backward is complete, not only items
still waiting in the queue.
"""
from collections import deque
from dataclasses import dataclass, field
import json
import math
from multiprocessing.connection import wait as wait_connections
import sys
import threading
import time
from typing import Any, Mapping, Tuple


COMPLETION_STREAM_VERSION = 1
MAX_COMPLETION_FRAME_BYTES = 128 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_storage_bytes(value):
    """Count the owned Python JSON object graph, including shared strings once."""
    seen = set()

    def visit(item):
        if id(item) in seen:
            return 0
        seen.add(id(item))
        size = sys.getsizeof(item)
        if isinstance(item, dict):
            return size + sum(visit(key) + visit(part) for key, part in item.items())
        if isinstance(item, (list, tuple)):
            return size + sum(visit(part) for part in item)
        require(item is None or type(item) in (bool, int, float, str),
                'Completion queues accept JSON values, not tensors or image objects')
        return size

    return visit(value)


def completion_timing(output, *, elapsed_seconds, first_output_seconds):
    """Keep engine-clock fields separate from local delivery-clock observations."""
    metrics = getattr(output, 'metrics', None)
    native = {}
    for key in ('arrival_time', 'first_scheduled_time', 'first_token_time', 'finished_time'):
        value = getattr(metrics, key, None)
        native[key] = float(value) if value is not None and math.isfinite(value) else None
    first, finished = native['first_token_time'], native['finished_time']
    return {
        'worker_finished_observed_s': elapsed_seconds,
        'worker_first_nonempty_output_s': first_output_seconds,
        'engine_clock': native,
        'engine_decode_s': finished - first if first is not None and finished is not None else None,
    }


def window_header(window):
    return {key: getattr(window, key) for key in ('run_id', 'window_id', 'policy_step')}


def encode_completion_frame(frame):
    value = json.dumps(frame, allow_nan=False, separators=(',', ':')).encode('utf-8') + b'\n'
    require(len(value) <= MAX_COMPLETION_FRAME_BYTES, 'Complete-response frame exceeds its wire byte limit')
    return value


def iter_completion_frames(chunks):
    """Parse bounded NDJSON without letting iter_lines build an unlimited row."""
    pending = bytearray()
    for chunk in chunks:
        pending.extend(chunk)
        while True:
            end = pending.find(b'\n')
            if end < 0:
                require(len(pending) < MAX_COMPLETION_FRAME_BYTES, 'Oversized incomplete completion frame')
                break
            require(end + 1 <= MAX_COMPLETION_FRAME_BYTES, 'Oversized completion frame')
            line = bytes(pending[:end])
            del pending[:end + 1]
            require(bool(line), 'Empty completion frame')
            frame = json.loads(line)
            require(isinstance(frame, dict), 'Completion frame must be an object')
            yield frame
    require(not pending, 'Truncated final completion frame')


class CompletionPipeRelay:
    """Read only IPC on the HTTP side; native workers own all inference/R3 work."""

    def __init__(self, connections, occurrence_groups, header, *, timeout_seconds):
        require(len(connections) == len(occurrence_groups) and connections,
                'Completion relay requires one occurrence partition per inference worker')
        require(math.isfinite(timeout_seconds) and timeout_seconds > 0, 'Invalid completion relay timeout')
        ids = [item for group in occurrence_groups for item in group]
        require(ids and len(set(ids)) == len(ids), 'Completion relay needs unique selected occurrences')
        self.connections = list(connections)
        self.groups = list(occurrence_groups)
        self.header = header
        self.timeout_seconds = timeout_seconds
        self.complete = False

    def __iter__(self):
        started = time.monotonic()
        active = dict(enumerate(self.connections))
        seen = [set() for _ in self.connections]
        total = sum(map(len, self.groups))
        yield encode_completion_frame({
            'event': 'start', 'protocol_version': COMPLETION_STREAM_VERSION,
            'window': self.header, 'samples': total,
        })
        while active:
            # A yielded frame can be held by downstream backpressure while the
            # student is computing. Bound waiting for worker progress, not the
            # total optimizer-window duration (which can exceed this timeout).
            ready = wait_connections(list(active.values()), timeout=self.timeout_seconds)
            if not ready:
                raise TimeoutError('No complete-response worker progress before the timeout')
            for rank, connection in list(active.items()):
                if connection not in ready:
                    continue
                message = connection.recv()
                require(isinstance(message, dict), 'Invalid complete-response worker message')
                if message.get('event') == 'error':
                    raise RuntimeError(f'Complete-response worker {rank} failed: {message["error"]}')
                if message.get('event') == 'end':
                    # Empty DP workers still execute one native placeholder
                    # request so the vLLM DP engine can progress symmetrically.
                    expected = max(1, len(self.groups[rank]))
                    require(message['samples'] == len(seen[rank]) == expected,
                            'Inference worker ended with missing completions')
                    del active[rank]
                    continue
                require(message.get('event') == 'completion', 'Unknown completion worker event')
                index = message['index']
                require(type(index) is int and 0 <= index < max(1, len(self.groups[rank]))
                        and index not in seen[rank], 'Repeated or unselected inference-worker completion')
                seen[rank].add(index)
                if not self.groups[rank]:
                    continue
                yield encode_completion_frame({
                    'event': 'completion', 'window': self.header,
                    'occurrence_id': self.groups[rank][index],
                    'output': message['output'], 'timing': message['timing'],
                    'server_delivery_s': time.monotonic() - started,
                })
        self.complete = True
        yield encode_completion_frame({
            'event': 'end', 'window': self.header, 'samples': total,
            'server_elapsed_s': time.monotonic() - started,
        })


@dataclass(frozen=True)
class CompletionWindow:
    run_id: str
    window_id: str
    policy_step: int
    occurrence_ids: Tuple[str, ...]

    def __post_init__(self):
        require(isinstance(self.run_id, str) and self.run_id
                and isinstance(self.window_id, str) and self.window_id, 'Missing completion-window identity')
        require(type(self.policy_step) is int and self.policy_step >= 0, 'Invalid completion policy version')
        require(isinstance(self.occurrence_ids, tuple) and self.occurrence_ids
                and all(isinstance(item, str) and item for item in self.occurrence_ids)
                and len(set(self.occurrence_ids)) == len(self.occurrence_ids),
                'Select every unique sample occurrence before generating the window')


@dataclass(frozen=True)
class CompletedSample:
    window: CompletionWindow
    occurrence_id: str
    output: Mapping[str, Any]
    prompt_tokens: int
    response_tokens: int
    received_seconds: float
    timing: Mapping[str, Any] = field(default_factory=dict)

    @property
    def tokens(self):
        return self.prompt_tokens + self.response_tokens


@dataclass(frozen=True)
class CompletionLease:
    index: int
    samples: Tuple[CompletedSample, ...]
    storage_bytes: int
    tokens: int


class CompletionQueue:
    """Bounded whole-response queue with one backward-owned consumer lease.

    A pressure-limited chunk can be smaller than the requested maximum, but
    always contains a complete number of `multiple` samples. The training
    adapter must use its teacher-RPC/DP block size as that multiple. No samples
    are dropped and queue completion does not perform a gradient update.
    """

    def __init__(self, window, *, max_samples, max_tokens, max_bytes, timeout_seconds):
        require(isinstance(window, CompletionWindow), 'Missing selected completion window')
        require(all(type(value) is int and value > 0 for value in (max_samples, max_tokens, max_bytes)),
                'Completion queue sample/token/byte limits must be positive integers')
        require(math.isfinite(timeout_seconds) and timeout_seconds > 0, 'Invalid completion queue timeout')
        self.window = window
        self.expected = set(window.occurrence_ids)
        self.max_samples, self.max_tokens, self.max_bytes = max_samples, max_tokens, max_bytes
        self.timeout_seconds = timeout_seconds
        self.condition = threading.Condition()
        self.ready = deque()
        self.received = set()
        self.released = set()
        self.lease = None
        self.producer_done = False
        self.finished = False
        self.failure = None
        self.waiting_producers = 0
        self.buffered_samples = self.buffered_tokens = self.buffered_bytes = 0
        self.peak_samples = self.peak_tokens = self.peak_bytes = 0
        self.producer_wait_seconds = self.consumer_wait_seconds = 0.
        self.next_chunk = 0
        self.started = time.monotonic()
        self.events = []

    def elapsed(self):
        return time.monotonic() - self.started

    def _check(self):
        if self.failure is not None:
            raise RuntimeError('Completion window failed; no optimizer update is allowed') from self.failure
        require(not self.finished, 'Completion window was already finalized')

    def _wait(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Completion queue timed out; check producer progress and chunk resource limits')
        self.condition.wait(remaining)
        self._check()

    def fail(self, error):
        with self.condition:
            if self.failure is None:
                self.failure = error
            self.condition.notify_all()

    def put(self, sample):
        """Transfer ownership; the producer must not mutate output after this call."""
        try:
            require(isinstance(sample, CompletedSample) and sample.window == self.window,
                    'Completion belongs to another window or policy version')
            require(sample.occurrence_id in self.expected, 'Unselected completion occurrence')
            require(type(sample.prompt_tokens) is int and sample.prompt_tokens > 0
                    and type(sample.response_tokens) is int and sample.response_tokens >= 0
                    and math.isfinite(sample.received_seconds) and sample.received_seconds >= 0,
                    'Invalid complete-response token counts or arrival time')
            require(isinstance(sample.output, dict), 'A completion must retain its complete JSON output')
            size = json_storage_bytes({'output': sample.output, 'timing': sample.timing})
            require(size <= self.max_bytes and sample.tokens <= self.max_tokens,
                    'One complete response exceeds the queue resource limits')
            with self.condition:
                self._check()
                require(not self.producer_done and sample.occurrence_id not in self.received,
                        'Duplicate completion or response after producer completion')
                deadline = time.monotonic() + self.timeout_seconds
                waited = None
                try:
                    while (self.buffered_samples + 1 > self.max_samples
                           or self.buffered_tokens + sample.tokens > self.max_tokens
                           or self.buffered_bytes + size > self.max_bytes):
                        if waited is None:
                            waited = time.monotonic()
                            self.waiting_producers += 1
                            self.condition.notify_all()
                        self._wait(deadline)
                    # A different server may have supplied the same occurrence
                    # while this producer was waiting for credits.
                    require(not self.producer_done and sample.occurrence_id not in self.received,
                            'Duplicate completion or response after producer completion')
                finally:
                    if waited is not None:
                        self.waiting_producers -= 1
                        self.producer_wait_seconds += time.monotonic() - waited
                self.ready.append((sample, size))
                self.received.add(sample.occurrence_id)
                self.buffered_samples += 1
                self.buffered_tokens += sample.tokens
                self.buffered_bytes += size
                self.peak_samples = max(self.peak_samples, self.buffered_samples)
                self.peak_tokens = max(self.peak_tokens, self.buffered_tokens)
                self.peak_bytes = max(self.peak_bytes, self.buffered_bytes)
                self.events.append({
                    'event': 'completion_ready', 'occurrence_id': sample.occurrence_id,
                    'received_s': sample.received_seconds, 'admitted_s': self.elapsed(),
                    'prompt_tokens': sample.prompt_tokens, 'response_tokens': sample.response_tokens,
                    'timing': sample.timing,
                })
                self.condition.notify_all()
        except BaseException as error:
            self.fail(error)
            raise

    def end_production(self):
        try:
            with self.condition:
                self._check()
                require(not self.producer_done and self.received == self.expected,
                        'Producer finished without exactly the selected window occurrences')
                self.producer_done = True
                self.events.append({'event': 'producer_finished', 'elapsed_s': self.elapsed()})
                self.condition.notify_all()
        except BaseException as error:
            self.fail(error)
            raise

    def acquire(self, max_samples, *, multiple=1):
        require(type(max_samples) is int and type(multiple) is int
                and 0 < multiple <= max_samples <= self.max_samples
                and max_samples % multiple == len(self.expected) % multiple == 0,
                'Consumer chunks must fit queue credits and contain complete DP/teacher blocks')
        started = time.monotonic()
        try:
            with self.condition:
                self._check()
                require(self.lease is None, 'Release the previous chunk only after its backward')
                deadline = started + self.timeout_seconds
                while True:
                    available = min(max_samples, len(self.ready)) // multiple * multiple
                    if available and (available == max_samples or self.producer_done or self.waiting_producers):
                        break
                    if self.producer_done:
                        require(not self.ready, 'Incomplete final DP/teacher block')
                        raise StopIteration
                    self._wait(deadline)
                entries = [self.ready.popleft() for _ in range(available)]
                samples = tuple(sample for sample, _ in entries)
                self.lease = CompletionLease(
                    self.next_chunk, samples, sum(size for _, size in entries), sum(s.tokens for s in samples))
                self.next_chunk += 1
                self.consumer_wait_seconds += time.monotonic() - started
                self.events.append({
                    'event': 'chunk_acquired', 'chunk': self.lease.index, 'elapsed_s': self.elapsed(),
                    'occurrence_ids': [sample.occurrence_id for sample in samples],
                    'tokens': self.lease.tokens,
                })
                # Credits still include the current lease.
                return self.lease
        except StopIteration:
            raise
        except BaseException as error:
            self.fail(error)
            raise

    def release(self, lease):
        try:
            with self.condition:
                self._check()
                require(lease is self.lease and lease is not None, 'Wrong or duplicate completion chunk release')
                ids = {sample.occurrence_id for sample in lease.samples}
                require(not ids & self.released, 'Repeated sample consumption')
                self.released.update(ids)
                self.buffered_samples -= len(lease.samples)
                self.buffered_tokens -= lease.tokens
                self.buffered_bytes -= lease.storage_bytes
                self.events.append({'event': 'chunk_released', 'chunk': lease.index, 'elapsed_s': self.elapsed()})
                self.lease = None
                self.condition.notify_all()
        except BaseException as error:
            self.fail(error)
            raise

    def finish(self):
        """Check the full native update boundary; never update or normalize gradients."""
        with self.condition:
            self._check()
            require(self.producer_done and self.lease is None and not self.ready
                    and self.received == self.released == self.expected,
                    'Cannot update before every selected response has completed backward')
            require(self.buffered_samples == self.buffered_tokens == self.buffered_bytes == 0,
                    'Completion credits leaked across the optimizer boundary')
            self.finished = True
            return {
                'samples': len(self.released), 'chunks': self.next_chunk,
                'queue_peak_samples': self.peak_samples, 'queue_peak_tokens': self.peak_tokens,
                'queue_peak_bytes': self.peak_bytes, 'producer_backpressure_s': self.producer_wait_seconds,
                'consumer_wait_s': self.consumer_wait_seconds, 'events': list(self.events),
            }

    def abort(self, error):
        self.fail(error)
        with self.condition:
            self.ready.clear()
            self.lease = None
            self.buffered_samples = self.buffered_tokens = self.buffered_bytes = 0
            self.condition.notify_all()


class CompletionProducer:
    """One CPU-only HTTP reader per vLLM server; never a trainer/CUDA thread."""

    def __init__(self, sessions, base_urls, requests, request_config, queue):
        require(len(sessions) == len(base_urls) > 0 and len(requests) == len(queue.window.occurrence_ids),
                'Completion producer topology or selected occurrence count differs')
        self.queue = queue
        self.lock = threading.Lock()
        self.cancelled = threading.Event()
        self.responses = {}
        self.threads = []
        self.completed_servers = 0
        self.server_count = len(sessions)
        self.header = window_header(queue.window)
        per_server = (len(requests) + len(sessions) - 1) // len(sessions)
        for index, (session, url) in enumerate(zip(sessions, base_urls)):
            offset = index * per_server
            ids = queue.window.occurrence_ids[offset:offset + per_server]
            body = {'window': self.header, 'occurrence_ids': list(ids),
                    'infer_requests': requests[offset:offset + per_server],
                    'request_config': request_config, 'timeout_seconds': queue.timeout_seconds}
            thread = threading.Thread(
                target=self._read, args=(index, session, url, body), name=f'mopd-completions-{index}', daemon=True)
            self.threads.append(thread)
        for thread in self.threads:
            thread.start()

    def _read(self, index, session, url, body):
        response = None
        try:
            ids = set(body['occurrence_ids'])
            seen, started, ended = set(), False, False
            if ids:
                response = session.post(
                    f'{url}/infer_completed/', json=body, stream=True,
                    timeout=(min(30., self.queue.timeout_seconds), self.queue.timeout_seconds))
                with self.lock:
                    self.responses[index] = response
                require(response.status_code == 200,
                        f'Complete-response server {index} rejected the window: HTTP {response.status_code}')
                frames = iter_completion_frames(response.iter_content(chunk_size=64 * 1024))
                for frame in frames:
                    require(not self.cancelled.is_set(), 'Completion producer was cancelled')
                    require(frame.get('window') == self.header and not ended,
                            'Completion frame belongs to another window or follows its end')
                    event = frame.get('event')
                    if event == 'error':
                        raise RuntimeError(f'Complete-response server {index}: {frame["error"]}')
                    if event == 'start':
                        require(not started and frame.get('protocol_version') == COMPLETION_STREAM_VERSION
                                and frame['samples'] == len(ids), 'Wrong complete-response stream handshake')
                        started = True
                        continue
                    require(started, 'Completion arrived before its stream handshake')
                    if event == 'end':
                        require(frame['samples'] == len(ids) and seen == ids,
                                'Complete-response server ended without all its selected occurrences')
                        ended = True
                        continue
                    require(event == 'completion', 'Unknown complete-response event')
                    occurrence = frame['occurrence_id']
                    require(occurrence in ids and occurrence not in seen,
                            'Duplicate or foreign complete-response occurrence')
                    output = frame['output']
                    usage = output['response']['usage']
                    choices = output['response']['choices']
                    require(len(choices) == 1 and isinstance(choices[0].get('token_ids'), list)
                            and len(choices[0]['token_ids']) == usage['completion_tokens'],
                            'Complete-response delivery lost generated token identity')
                    self.queue.put(CompletedSample(
                        self.queue.window, occurrence, output, usage['prompt_tokens'], usage['completion_tokens'],
                        self.queue.elapsed(), frame['timing']))
                    seen.add(occurrence)
                require(started and ended, 'Truncated complete-response stream has no confirmed end')
            with self.lock:
                self.completed_servers += 1
                if self.completed_servers == self.server_count:
                    self.queue.end_production()
        except BaseException as error:
            self.queue.fail(error)
            self.cancelled.set()
        finally:
            if response is not None:
                response.close()
            with self.lock:
                self.responses.pop(index, None)

    def finish(self):
        """Join every reader before allowing the native whole-window update."""
        deadline = time.monotonic() + self.queue.timeout_seconds
        for thread in self.threads:
            thread.join(max(0., deadline - time.monotonic()))
        require(not any(thread.is_alive() for thread in self.threads), 'Completion HTTP readers did not settle')
        return self.queue.finish()

    def abort(self, error):
        self.cancelled.set()
        self.queue.abort(error)
        with self.lock:
            responses = list(self.responses.values())
        for response in responses:
            response.close()
