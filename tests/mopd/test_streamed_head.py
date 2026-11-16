# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
"""Checkpointed teacher-head chunks preserve the full-vocabulary objective."""
import types

import pytest
import torch

from tests.mopd.source_test_utils import gkd_symbols, load_module, load_vocab_parallel


class NativeShapeHead(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.2)
        self.bias = None
        self.sequence_parallel = True

    def forward(self, hidden, *, weight=None, runtime_gather_output=None):
        actual = self.weight if weight is None else weight
        return hidden @ actual.T, None


def implementation(*, loss_type='jsd', temperature=1.0):
    mpu = types.SimpleNamespace(
        get_tensor_model_parallel_world_size=lambda: 1,
        get_tensor_model_parallel_rank=lambda: 0,
        get_tensor_model_parallel_group=lambda: None,
        get_data_parallel_group=lambda: None,
        get_context_parallel_world_size=lambda: 1,
    )
    vocab = load_vocab_parallel(mpu)
    symbols = gkd_symbols(
        ['_align_vocab_size', 'generalized_jsd_loss'],
        {
            **{
                name: getattr(vocab, name)
                for name in dir(vocab)
                if name.startswith('vocab_parallel_')
            },
            'mpu': mpu,
            'get_args': lambda: types.SimpleNamespace(
                context_parallel_size=1,
                padded_vocab_size=23,
            ),
        },
    )
    criterion = types.SimpleNamespace(
        loss_temperature=temperature,
        teacher_label_smoothing=0.0,
        gkd_loss_type=loss_type,
        gkd_todi_beta=1.0,
    )
    for name in ('_align_vocab_size', 'generalized_jsd_loss'):
        setattr(criterion, name, types.MethodType(symbols[name], criterion))
    stream = load_module(
        'ms-swift/swift/megatron/trainers/streamed_distillation.py',
        f'unimopd_test_streamed_head_{loss_type}_{temperature}',
    )
    return stream, criterion


@pytest.mark.parametrize('loss_type,beta', [('jsd', 0.0), ('jsd', 1.0), ('todi', 1.0)])
@pytest.mark.parametrize('chunk_size', [1, 3, 512])
@pytest.mark.parametrize('checkpoint_chunks', [False, True])
def test_streamed_and_dense_heads_match_value_and_gradients(
    loss_type,
    beta,
    chunk_size,
    checkpoint_chunks,
):
    torch.manual_seed(1904)
    labels = torch.tensor([[-100, 1, 2, -100, 4, 5, 6, -100, 8]])
    selected = labels.ne(-100).reshape(-1).nonzero().reshape(-1)
    teacher_hidden = torch.randn(1, labels.numel(), 7)
    teacher = NativeShapeHead(7, 23).requires_grad_(False)
    student = NativeShapeHead(7, 23)
    hidden = torch.randn_like(teacher_hidden, requires_grad=True)
    stream, criterion = implementation(loss_type=loss_type)

    student_full = student(hidden.transpose(0, 1))[0].transpose(0, 1)
    teacher_full = teacher(teacher_hidden.transpose(0, 1))[0].transpose(0, 1)
    expected, expected_count = criterion.generalized_jsd_loss(
        student_full,
        teacher_full,
        labels,
        beta=beta,
        return_sum_count=True,
    )
    expected.backward()
    expected_hidden = hidden.grad.clone()
    expected_weight = student.weight.grad.clone()

    student.weight.grad = None
    actual_hidden = hidden.detach().clone().requires_grad_()
    target = stream.snapshot_teacher_head(
        types.SimpleNamespace(
            share_embeddings_and_output_weights=False,
            output_layer=teacher,
        ),
        teacher_hidden[:, selected],
    )
    actual, actual_count, _ = stream.streamed_head_loss(
        actual_hidden[:, selected],
        target,
        labels,
        student,
        None,
        criterion.generalized_jsd_loss,
        beta=beta,
        chunk_size=chunk_size,
        collect_diagnostics=False,
        checkpoint_chunks=checkpoint_chunks,
    )
    actual.backward()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_hidden.grad, expected_hidden, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(student.weight.grad, expected_weight, atol=1e-6, rtol=1e-5)
    assert actual_count == expected_count
    assert teacher.weight.grad is None
    assert student.sequence_parallel is True
    assert teacher.sequence_parallel is True


def test_snapshot_retains_current_frozen_weight_storage():
    stream, _ = implementation()
    teacher = NativeShapeHead(4, 23).requires_grad_(False)
    original = teacher.weight.detach().clone()
    target = stream.snapshot_teacher_head(
        types.SimpleNamespace(
            share_embeddings_and_output_weights=False,
            output_layer=teacher,
        ),
        torch.randn(1, 5, 4),
    )
    assert target.weight.data_ptr() == teacher.weight.data_ptr()
    teacher.weight.data = torch.full_like(teacher.weight, 99.0)
    assert torch.equal(target.weight, original)
    assert not torch.equal(target.weight, teacher.weight)


def test_empty_supervision_keeps_a_graph_connected_zero():
    stream, criterion = implementation()
    labels = torch.full((1, 3), -100)
    student = NativeShapeHead(4, 23)
    teacher = NativeShapeHead(4, 23).requires_grad_(False)
    hidden = torch.randn(1, 1, 4, requires_grad=True)
    target = stream.snapshot_teacher_head(
        types.SimpleNamespace(
            share_embeddings_and_output_weights=False,
            output_layer=teacher,
        ),
        torch.randn_like(hidden),
    )
    loss, count, _ = stream.streamed_head_loss(
        hidden,
        target,
        labels,
        student,
        None,
        criterion.generalized_jsd_loss,
        beta=1.0,
        chunk_size=2,
        collect_diagnostics=False,
    )
    loss.backward()
    assert loss == 0 and count == 0
    assert not hidden.grad.any()
