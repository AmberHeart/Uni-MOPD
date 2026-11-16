# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
"""Full-vocabulary FKL, RKL and ToDi use the published objective definitions."""
import math
import types

import pytest
import torch

from tests.mopd.source_test_utils import gkd_symbols, load_symbols, load_vocab_parallel


def trainer_for(kind='jsd', *, beta=1.0, temperature=1.0, sharpness=1.0):
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
                padded_vocab_size=17,
            ),
        },
    )
    trainer = types.SimpleNamespace(
        beta=beta,
        loss_temperature=temperature,
        teacher_label_smoothing=0.0,
        gkd_loss_type=kind,
        gkd_todi_beta=sharpness,
    )
    for name in ('_align_vocab_size', 'generalized_jsd_loss'):
        setattr(trainer, name, types.MethodType(symbols[name], trainer))
    return trainer


def inputs(dtype=torch.float32):
    generator = torch.Generator().manual_seed(619)
    student = torch.randn(2, 5, 17, generator=generator, dtype=dtype)
    teacher = torch.randn(2, 5, 17, generator=generator, dtype=dtype)
    labels = torch.tensor([[-100, 1, 4, -100, 3], [-100, 9, 5, 7, 6]])
    return student, teacher, labels


def todi_reference(log_student, log_teacher, sharpness):
    delta = log_teacher.detach() - log_student
    teacher_weight = torch.sigmoid(sharpness * delta).detach()
    student_weight = torch.sigmoid(-sharpness * delta).detach()
    return (
        teacher_weight * log_teacher.exp() * delta
        + student_weight * log_student.exp() * (-delta)
    ).sum(-1)


@pytest.mark.parametrize(
    'kind,beta',
    [('jsd', 0.0), ('jsd', 1.0), ('todi', 1.0)],
)
@pytest.mark.parametrize('temperature', [0.7, 1.0, 1.5])
@pytest.mark.parametrize('chunk_size', [1, 3, 512])
def test_objective_value_mask_temperature_and_gradient(kind, beta, temperature, chunk_size):
    trainer = trainer_for(kind, beta=beta, temperature=temperature)
    student, teacher, labels = inputs()
    student.requires_grad_()
    actual = trainer.generalized_jsd_loss(
        student,
        teacher,
        labels,
        beta=beta,
        chunk_size=chunk_size,
    )
    actual.backward()

    mask = labels != -100
    reference = student.detach().clone().requires_grad_()
    log_student = (reference[mask] / temperature).log_softmax(-1)
    log_teacher = (teacher[mask] / temperature).log_softmax(-1)
    if kind == 'todi':
        values = todi_reference(log_student, log_teacher, 1.0)
    elif beta == 0:
        values = (log_teacher.exp() * (log_teacher - log_student)).sum(-1)
    else:
        values = (log_student.exp() * (log_student - log_teacher)).sum(-1)
    expected = values.mean()
    expected.backward()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(student.grad, reference.grad, atol=2e-5, rtol=2e-4)
    assert not student.grad[~mask].any()


@pytest.mark.parametrize('kind,beta', [('jsd', 0.0), ('jsd', 1.0), ('todi', 1.0)])
def test_identical_teacher_and_student_are_stationary(kind, beta):
    trainer = trainer_for(kind, beta=beta)
    student, _, labels = inputs()
    student.requires_grad_()
    loss, count = trainer.generalized_jsd_loss(
        student,
        student.detach().clone(),
        labels,
        beta=beta,
        return_sum_count=True,
    )
    loss.backward()
    assert loss == 0
    assert count == (labels != -100).sum()
    assert student.grad.abs().max() < 1e-7


def test_response_only_rows_match_the_dense_rkl():
    trainer = trainer_for('jsd', beta=1.0)
    student, teacher, labels = inputs()
    mask = labels != -100
    dense = trainer.generalized_jsd_loss(student, teacher, labels, beta=1.0)
    selected = trainer.generalized_jsd_loss(
        student[mask].reshape(-1, 1, student.shape[-1]),
        teacher[mask].reshape(-1, 1, teacher.shape[-1]),
        labels,
        beta=1.0,
        student_logits_pre_masked=True,
        teacher_logits_pre_masked=True,
    )
    torch.testing.assert_close(selected, dense)


@pytest.mark.parametrize(
    'changes',
    [
        {'gkd_loss_type': 'unsupported'},
        {'gkd_todi_beta': 0.0},
        {'gkd_todi_beta': float('nan')},
        {'beta': -0.1},
        {'beta': 1.1},
        {'beta': float('inf')},
        {'rlhf_type': 'dpo', 'gkd_loss_type': 'todi'},
    ],
)
def test_invalid_objective_configuration_fails_before_training(changes):
    validate = load_symbols(
        'ms-swift/swift/megatron/argument/rlhf_args.py',
        ['_validate_gkd_objective'],
        parent='MegatronRLHFArguments',
        namespace={'math': math},
    )['_validate_gkd_objective']
    args = types.SimpleNamespace(
        rlhf_type='gkd',
        gkd_loss_type='jsd',
        gkd_todi_beta=1.0,
        beta=1.0,
    )
    for key, value in changes.items():
        setattr(args, key, value)
    with pytest.raises(ValueError):
        validate(args)
