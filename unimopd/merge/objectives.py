# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0

"""Full-vocabulary response objectives used for merge calibration."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

OBJECTIVES = ('rkl', 'fkl', 'todi', 'response_ce')


def token_objectives(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return per-token RKL, FKL, ToDi, and response CE.

    RKL is ``KL(student || teacher)`` and FKL is
    ``KL(teacher || student)``. The KL terms follow the paper convention and
    do not apply a temperature-squared multiplier. Response CE is evaluated on
    the untempered student logits. ToDi uses the beta=1 detached token weights
    from the calibration experiments.
    """
    if (student_logits.ndim != 2 or student_logits.shape != teacher_logits.shape
            or labels.shape != student_logits.shape[:1]):
        raise ValueError(
            'student and teacher logits must share [response_tokens, full_vocab], '
            'with one aligned label per token.')
    if (not math.isfinite(temperature) or temperature <= 0
            or isinstance(temperature, bool)):
        raise ValueError('temperature must be finite and positive.')
    if teacher_logits.requires_grad:
        raise ValueError('teacher logits must be frozen.')
    if labels.dtype != torch.long:
        raise ValueError('labels must use torch.long token IDs.')
    if labels.numel() and (labels.min() < 0 or labels.max() >= student_logits.shape[-1]):
        raise ValueError('A response label is outside the full vocabulary.')

    student = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    reverse_terms = student.exp() * (student - teacher)
    forward_terms = teacher.exp() * (teacher - student)
    detached_weight = torch.sigmoid(teacher - student).detach()
    return {
        'rkl': reverse_terms.sum(dim=-1),
        'fkl': forward_terms.sum(dim=-1),
        'todi': (
            detached_weight * forward_terms
            + (1 - detached_weight) * reverse_terms
        ).sum(dim=-1),
        'response_ce': F.cross_entropy(
            student_logits.float(), labels, reduction='none'),
    }


def sequence_weighted_objective(
    token_values: torch.Tensor,
    sequence_offsets: torch.Tensor,
    sequence_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute a weighted mean of per-sequence token means."""
    if token_values.ndim != 1:
        raise ValueError('token_values must be one-dimensional.')
    if (sequence_offsets.ndim != 1 or sequence_offsets.dtype != torch.long
            or len(sequence_offsets) < 2):
        raise ValueError('sequence_offsets must be a one-dimensional long tensor.')
    if (sequence_offsets[0] != 0 or sequence_offsets[-1] != len(token_values)
            or bool((sequence_offsets[1:] <= sequence_offsets[:-1]).any())):
        raise ValueError('sequence_offsets must strictly partition all response tokens.')
    count = len(sequence_offsets) - 1
    if sequence_weights.shape != (count,):
        raise ValueError('sequence_weights must contain one value per sequence.')
    weights = sequence_weights.to(device=token_values.device, dtype=torch.float32)
    if (not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any())
            or not torch.isclose(weights.sum(), weights.new_tensor(1.0), atol=1e-6, rtol=0)):
        raise ValueError('sequence_weights must be positive, finite, and sum to one.')
    means = torch.stack([
        token_values[int(left):int(right)].mean()
        for left, right in zip(sequence_offsets[:-1], sequence_offsets[1:])
    ])
    return (weights * means).sum()


def calibration_objective(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    sequence_offsets: torch.Tensor,
    sequence_weights: torch.Tensor,
    *,
    objective: str,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Evaluate one named response objective."""
    if objective not in OBJECTIVES:
        raise ValueError(f'objective must be one of {OBJECTIVES}, got {objective!r}.')
    values = token_objectives(
        student_logits, teacher_logits, labels, temperature=temperature)
    return sequence_weighted_objective(
        values[objective], sequence_offsets, sequence_weights)
