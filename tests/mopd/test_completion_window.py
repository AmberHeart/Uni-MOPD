# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract for complete-response microbatch encoding."""
from types import SimpleNamespace

from tests.mopd.source_test_utils import load_symbols


def test_encode_tags_only_fields_consumed_by_forward_step():
    encode = load_symbols(
        'ms-swift/swift/megatron/trainers/completion_window.py',
        ['_encode'],
        parent='IncrementalRolloutWindow',
    )['_encode']
    trainer = SimpleNamespace(_encode_batch=lambda raw: {'input_ids': [[1, 2, 3]], 'labels': [[-100, 2, 3]]})
    window = SimpleNamespace(trainer=trainer, data_source='rollout', policy_step=7)
    raw = [{'domain': 'StemPuzzle', 'finish_reason': 'stop', 'rollout_logprobs': [-0.1, -0.2]}]

    batch = encode(window, raw, '0')

    assert batch == {
        'input_ids': [[1, 2, 3]],
        'labels': [[-100, 2, 3]],
        'data_source': 'rollout',
        '_rollout_weight_step': 7,
        'teacher_idx': '0',
    }
