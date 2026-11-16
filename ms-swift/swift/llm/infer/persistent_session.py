# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0

"""Experiment ownership for a retained, single-turn vLLM rollout engine."""
import re
from fastapi import Request
from starlette.responses import JSONResponse


SESSION_PROTOCOL_VERSION = 1
SESSION_HEADER = 'X-MOPD-Session'
PUBLIC_PATHS = {'/health/', '/session_status/', '/begin_session/',
                '/get_world_size/', '/get_engine_type/'}
READ_ONLY_PATHS = PUBLIC_PATHS - {'/begin_session/'}


def install_session_middleware(app, owner):
    if owner._persistent_session is None:
        return

    @app.middleware('http')
    async def experiment_owner(request: Request, call_next):
        session = owner._persistent_session
        if session is None:
            return await call_next(request)
        if request.url.path in READ_ONLY_PATHS:
            return await call_next(request)
        # Recheck AFTER locking: a queued old-client request cannot run under
        # the next experiment. Stream bodies use the existing completion guard.
        async with owner._persistent_http_lock:
            if (request.url.path not in PUBLIC_PATHS
                    and not session.authorize(request.headers.get(SESSION_HEADER))):
                return JSONResponse({'detail': 'Missing or stale experiment session'}, status_code=409)
            return await call_next(request)


class PersistentRolloutSession:
    def __init__(self, identity):
        if not re.fullmatch(r'[0-9a-f]{64}', identity):
            raise ValueError('Persistent vLLM requires a SHA256 runtime compatibility identity')
        self.identity = identity
        self.session_id = None
        self.ready = False
        self.received_weights = False
        self.completed = set()

    def begin(self, session_id, identity):
        if identity != self.identity:
            raise ValueError('Persistent vLLM model/code/layout identity mismatch; cold restart required')
        if not isinstance(session_id, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,160}', session_id):
            raise ValueError('Invalid experiment session identifier')
        if self.session_id is not None or session_id in self.completed:
            raise ValueError('An experiment is active or this session identifier has already been used')
        self.session_id = session_id
        self.ready = False
        self.received_weights = False

    def authorize(self, session_id):
        return self.session_id is not None and session_id == self.session_id

    def end(self):
        self.completed.add(self.session_id)
        self.session_id = None
        self.ready = False
        self.received_weights = False

    def status(self):
        return {'session_protocol': SESSION_PROTOCOL_VERSION, 'identity': self.identity,
                'session_id': self.session_id, 'ready': self.ready,
                'state': 'idle' if self.session_id is None else 'active'}
