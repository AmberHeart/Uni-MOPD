# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

from unimopd.merge.objectives import (
    calibration_objective,
    sequence_weighted_objective,
    token_objectives,
)


def test_full_vocab_objectives_match_definitions():
    student = torch.tensor([[2.0, -1.0, 0.5], [0.1, 0.4, -0.2]], requires_grad=True)
    teacher = torch.tensor([[1.0, 0.0, -0.5], [-0.3, 0.7, 0.2]])
    labels = torch.tensor([0, 2])
    values = token_objectives(student, teacher, labels, temperature=0.7)
    s = F.log_softmax(student / 0.7, dim=-1)
    t = F.log_softmax(teacher / 0.7, dim=-1)
    rkl_terms = s.exp() * (s - t)
    fkl_terms = t.exp() * (t - s)
    weight = torch.sigmoid(t - s).detach()
    torch.testing.assert_close(values['rkl'], rkl_terms.sum(-1))
    torch.testing.assert_close(values['fkl'], fkl_terms.sum(-1))
    torch.testing.assert_close(
        values['todi'], (weight * fkl_terms + (1 - weight) * rkl_terms).sum(-1))
    torch.testing.assert_close(
        values['response_ce'], F.cross_entropy(student, labels, reduction='none'))
    values['todi'].sum().backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_sequence_means_are_not_token_weighted():
    token_values = torch.tensor([2.0, 4.0, 10.0])
    offsets = torch.tensor([0, 2, 3])
    weights = torch.tensor([0.75, 0.25])
    actual = sequence_weighted_objective(token_values, offsets, weights)
    assert actual == 0.75 * 3 + 0.25 * 10


def test_identical_distributions_have_zero_kl():
    logits = torch.randn(5, 11)
    labels = torch.arange(5)
    offsets = torch.tensor([0, 2, 5])
    weights = torch.tensor([0.5, 0.5])
    for objective in ('rkl', 'fkl', 'todi'):
        value = calibration_objective(
            logits, logits.detach(), labels, offsets, weights,
            objective=objective)
        torch.testing.assert_close(value, torch.tensor(0.0), atol=2e-7, rtol=0)
