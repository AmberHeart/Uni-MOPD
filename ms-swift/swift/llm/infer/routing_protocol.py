# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, MOPD contributors.
"""Compact, JSON-safe expert IDs; no logits, gate weights, or private APIs."""
import base64

import numpy as np

ROUTING_PROTOCOL_VERSION = 1
ROUTING_INFO_KEY = 'mopd_routing'


def encode_routing_trace(token_ids, expert_ids, num_experts):
    tokens = np.asarray(token_ids, dtype='<i4')
    routes = np.asarray(expert_ids)
    if tokens.ndim != 1 or routes.ndim != 3 or len(tokens) != len(routes):
        raise ValueError('Routing trace must pair [tokens] IDs with [tokens, layers, topk] experts.')
    if not 0 < num_experts < 65535:
        raise ValueError('The compact routing protocol supports fewer than 65535 experts.')
    if routes.size and (routes.min() < 0 or routes.max() >= num_experts):
        raise ValueError('Cannot export missing or out-of-bank expert IDs.')
    dtype = np.dtype('u1' if num_experts < 256 else '<u2')
    routes = np.ascontiguousarray(routes, dtype=dtype)
    return {
        'version': ROUTING_PROTOCOL_VERSION,
        'shape': list(routes.shape),
        'num_experts': int(num_experts),
        'dtype': dtype.str,
        'tokens': base64.b64encode(tokens.tobytes()).decode('ascii'),
        'experts': base64.b64encode(routes.tobytes()).decode('ascii'),
    }


def decode_routing_trace(payload, *, max_tokens):
    if payload.get('version') != ROUTING_PROTOCOL_VERSION:
        raise ValueError('Unsupported routing protocol version.')
    shape = payload.get('shape')
    if (not isinstance(shape, list) or len(shape) != 3
            or any(type(value) is not int or value < 0 for value in shape)
            or not 0 < shape[0] <= max_tokens or not 0 < shape[1] <= 256):
        raise ValueError(f'Invalid routing shape: {shape!r}.')
    experts = payload.get('num_experts')
    if type(experts) is not int or not 0 < shape[2] <= experts < 65535:
        raise ValueError('Invalid routing top-k/expert bank.')
    expected_dtype = np.dtype('u1' if experts < 256 else '<u2')
    if payload.get('dtype') != expected_dtype.str:
        raise ValueError('Routing dtype does not match the declared expert bank.')
    tokens = base64.b64decode(payload['tokens'], validate=True)
    routes = base64.b64decode(payload['experts'], validate=True)
    if len(tokens) != shape[0] * 4 or len(routes) != int(np.prod(shape)) * expected_dtype.itemsize:
        raise ValueError('Routing payload byte length does not match its shape.')
    token_ids = np.frombuffer(tokens, dtype='<i4').copy()
    expert_ids = np.frombuffer(routes, dtype=expected_dtype).reshape(shape).copy()
    if expert_ids.max() >= experts:
        raise ValueError('Captured expert ID is outside the declared bank.')
    return {'token_ids': token_ids, 'expert_ids': expert_ids, 'num_experts': experts}


def bind_routing_trace(payload, prompt_token_ids, completion_token_ids, *, max_tokens):
    """Bind a worker capture to the returned prediction-input prefix."""
    if (not isinstance(prompt_token_ids, (list, tuple)) or not prompt_token_ids
            or not isinstance(completion_token_ids, (list, tuple)) or not completion_token_ids):
        raise ValueError('R3 binding requires returned prompt and completion token IDs.')
    returned = list(prompt_token_ids) + list(completion_token_ids)
    if any(type(token) is not int or not 0 <= token < 2**31 for token in returned):
        raise ValueError('R3 binding requires nonnegative int32 token IDs.')
    if len(returned) > max_tokens:
        raise ValueError('Returned R3 sequence exceeds the configured model length.')
    decoded = decode_routing_trace(payload, max_tokens=max_tokens)
    required = len(returned) - 1
    captured = len(decoded['token_ids'])
    if captured < required:
        raise ValueError(f'R3 capture is incomplete: captured={captured}, required={required}.')
    compared = min(captured, len(returned))
    if not np.array_equal(decoded['token_ids'][:compared], returned[:compared]):
        raise ValueError('R3 captured token IDs differ from the returned prompt/completion.')
    if captured == compared:
        return payload
    return encode_routing_trace(
        decoded['token_ids'][:compared], decoded['expert_ids'][:compared], decoded['num_experts'])


class RoutingSlotStore:
    """CPU routes beside physical KV slots, plus completed-request snapshots.

    Prefix-cache hits reuse the *same KV slots*, including multimodal cache
    identity. Finished routes are snapshotted before the next scheduler step
    can recycle those slots. Nothing is keyed only by prompt text/token IDs.
    """

    def __init__(self, num_slots, num_layers, topk, num_experts, block_size):
        self.num_experts = num_experts
        self.block_size = block_size
        self.dtype = np.dtype('u1' if num_experts < 256 else '<u2')
        self.missing = np.iinfo(self.dtype).max
        self.routes = np.full((num_slots, num_layers, topk), self.missing, dtype=self.dtype)
        self.tokens = np.full(num_slots, -1, dtype='<i4')
        self.active = {}
        self.completed = {}
        self.consumed = set()

    def reset(self):
        self.routes.fill(self.missing)
        self.tokens.fill(-1)
        self.active.clear()
        self.completed.clear()
        self.consumed.clear()

    def start_requests(self, request_ids):
        for request_id in request_ids:
            self.active.pop(request_id, None)
            self.completed.pop(request_id, None)
            self.consumed.discard(request_id)

    def finish_requests(self, request_ids):
        for request_id in request_ids:
            if request_id in self.active and request_id not in self.consumed:
                self.completed[request_id] = self._snapshot(request_id)
            self.active.pop(request_id, None)
            self.consumed.discard(request_id)

    def suspend_requests(self, scheduled_ids):
        # A preempted/unscheduled request can lose its KV allocation. Never
        # later snapshot those addresses after another request reused them.
        # Resumption installs a fresh block table in write_step; cached full
        # prefix blocks themselves retain their valid slot-bound routes.
        for request_id in set(self.active).difference(scheduled_ids):
            self.active.pop(request_id)
            self.consumed.discard(request_id)

    def write_step(self, slots, token_ids, expert_ids, views):
        slots = np.asarray(slots, dtype=np.int64)
        tokens = np.asarray(token_ids, dtype='<i4')
        routes = np.asarray(expert_ids)
        if (slots.ndim != 1 or tokens.shape != slots.shape
                or routes.shape != (len(slots), *self.routes.shape[1:])):
            raise ValueError('Scheduled tokens, KV slots and captured routes are misaligned.')
        if len(slots) and (slots.min() < 0 or slots.max() >= len(self.tokens)):
            raise ValueError('Routing capture requires valid local KV slots.')
        if routes.size and (routes.min() < 0 or routes.max() >= self.num_experts):
            raise ValueError('A scheduled token/layer did not capture valid expert IDs.')
        self.routes[slots] = routes
        self.tokens[slots] = tokens
        # Replace lengths on preemption/resume; max(old, new) could expose
        # routes from the old allocation before its prefix was recomputed.
        for request_id, blocks, length in views:
            self.active[request_id] = (np.asarray(blocks, dtype=np.int64).copy(), int(length))

    def _snapshot(self, request_id):
        blocks, length = self.active[request_id]
        positions = np.arange(length, dtype=np.int64)
        if length <= 0 or (length - 1) // self.block_size >= len(blocks):
            raise ValueError('Request routing length exceeds its actual KV block table.')
        slots = blocks[positions // self.block_size] * self.block_size + positions % self.block_size
        tokens, routes = self.tokens[slots].copy(), self.routes[slots].copy()
        if (tokens < 0).any() or (routes == self.missing).any():
            raise ValueError('A cached prefix has no corresponding token/routing capture.')
        # Keep completion snapshots compact; base64 encoding belongs to the
        # export RPC, not the next latency-sensitive model execution step.
        return tokens, routes

    def pop(self, request_ids):
        result = {}
        for request_id in request_ids:
            if request_id in self.completed:
                tokens, routes = self.completed.pop(request_id)
            elif request_id in self.active:
                tokens, routes = self._snapshot(request_id)
                self.consumed.add(request_id)
            else:
                continue
            result[request_id] = encode_routing_trace(tokens, routes, self.num_experts)
        return result
