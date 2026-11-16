# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for compact R3 route transport and returned-prefix binding."""
import base64

import numpy as np
import pytest

from tests.mopd.source_test_utils import load_module


@pytest.fixture(scope='module')
def protocol():
    return load_module(
        'ms-swift/swift/llm/infer/routing_protocol.py',
        'unimopd_test_routing_protocol',
    )


@pytest.mark.parametrize('extra_rows', [0, 1, 3])
def test_returned_prefix_binding_keeps_routes_and_trims_only_unreturned_tail(protocol, extra_rows):
    prompt, completion = [10, 11, 12], [13, 14, 15]
    required = (prompt + completion)[:-1]
    captured = (prompt + completion + [16, 17])[:len(required) + extra_rows]
    routes = (np.arange(len(captured) * 4 * 2) % 128).reshape(len(captured), 4, 2)
    payload = protocol.encode_routing_trace(captured, routes, 128)

    bound = protocol.bind_routing_trace(payload, prompt, completion, max_tokens=32)
    decoded = protocol.decode_routing_trace(bound, max_tokens=len(prompt + completion))
    kept = min(len(captured), len(prompt + completion))
    np.testing.assert_array_equal(decoded['token_ids'], (prompt + completion)[:kept])
    np.testing.assert_array_equal(decoded['expert_ids'], routes[:kept])
    if extra_rows <= 1:
        assert bound is payload


@pytest.mark.parametrize(
    'change,match',
    [
        ('prompt', 'differ'),
        ('completion', 'differ'),
        ('terminal', 'differ'),
        ('missing', 'incomplete'),
        ('budget', 'model length'),
        ('empty_prompt', 'requires returned'),
        ('empty_completion', 'requires returned'),
        ('invalid_token', 'int32'),
    ],
)
def test_prefix_binding_rejects_changed_or_incomplete_sequences(protocol, change, match):
    prompt, completion, limit = [10, 11], [12, 13], 16
    captured = [10, 11, 12, 13, 14]
    if change == 'prompt':
        prompt[0] = 99
    elif change == 'completion':
        completion[0] = 99
    elif change == 'terminal':
        completion[-1] = 99
    elif change == 'missing':
        captured = captured[:2]
    elif change == 'budget':
        limit = 3
    elif change == 'empty_prompt':
        prompt = []
    elif change == 'empty_completion':
        completion = []
    elif change == 'invalid_token':
        prompt[0] = True
    payload = protocol.encode_routing_trace(
        captured,
        np.zeros((len(captured), 2, 2), dtype=np.uint8),
        128,
    )
    with pytest.raises(ValueError, match=match):
        protocol.bind_routing_trace(payload, prompt, completion, max_tokens=limit)


@pytest.mark.parametrize('num_experts,dtype', [(128, 'u1'), (300, '<u2')])
def test_trace_roundtrip_uses_compact_integer_storage(protocol, num_experts, dtype):
    token_ids = np.array([1, 151655, 7], dtype=np.int32)
    routes = (np.arange(3 * 4 * 2) % num_experts).reshape(3, 4, 2)
    encoded = protocol.encode_routing_trace(token_ids, routes, num_experts)
    restored = protocol.decode_routing_trace(encoded, max_tokens=3)
    np.testing.assert_array_equal(restored['token_ids'], token_ids)
    np.testing.assert_array_equal(restored['expert_ids'], routes)
    assert len(base64.b64decode(encoded['experts'])) == routes.size * np.dtype(dtype).itemsize


@pytest.mark.parametrize(
    'change',
    [
        {'version': 7},
        {'shape': [1, 2]},
        {'shape': [1000, 4, 2]},
        {'dtype': '<i8'},
        {'experts': 'AA=='},
        {'tokens': '*invalid*'},
    ],
)
def test_malformed_payloads_are_rejected(protocol, change):
    trace = protocol.encode_routing_trace(
        np.arange(3),
        np.zeros((3, 4, 2), dtype=np.uint8),
        128,
    )
    with pytest.raises((ValueError, base64.binascii.Error)):
        protocol.decode_routing_trace({**trace, **change}, max_tokens=10)


def test_completed_snapshot_survives_kv_slot_reuse(protocol):
    store = protocol.RoutingSlotStore(16, 2, 2, 128, 4)
    first = np.array([[[1, 2], [3, 4]]] * 3, dtype=np.uint8)
    store.write_step([4, 5, 6], [10, 11, 12], first, [('old', [1], 3)])
    store.finish_requests(['old'])
    replacement = np.repeat(first[:1] + 5, 4, axis=0)
    store.write_step([4, 5, 6, 7], [20, 21, 22, 23], replacement, [('new', [1], 4)])
    restored = protocol.decode_routing_trace(store.pop(['old'])['old'], max_tokens=3)
    np.testing.assert_array_equal(restored['token_ids'], [10, 11, 12])
    np.testing.assert_array_equal(restored['expert_ids'], first)


def test_preemption_discards_stale_slot_ownership(protocol):
    store = protocol.RoutingSlotStore(8, 1, 2, 128, 4)
    store.write_step([0, 1], [10, 11], np.zeros((2, 1, 2)), [('old', [0], 2)])
    store.suspend_requests({'new'})
    store.write_step([0, 1], [20, 21], np.ones((2, 1, 2)), [('new', [0], 2)])
    store.finish_requests(['old'])
    assert store.pop(['old']) == {}
    current = protocol.decode_routing_trace(store.pop(['new'])['new'], max_tokens=2)
    np.testing.assert_array_equal(current['token_ids'], [20, 21])
