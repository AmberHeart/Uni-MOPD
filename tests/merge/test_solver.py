# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch

from unimopd.merge.calibrate import ARTIFACT_FORMAT, calibrate
from unimopd.merge.solver import (
    project_simplex,
    projected_gradient_residual,
    solve,
)


class Quadratic:
    def __init__(self, target):
        self.target = torch.tensor(target, dtype=torch.float32)

    def __call__(self, alpha, need_gradient):
        difference = alpha - self.target
        value = 0.5 * torch.dot(difference, difference)
        return float(value), difference if need_gradient else None


def test_simplex_projection():
    actual = project_simplex([-1.0, 0.2, 3.0, 0.4])
    assert bool((actual >= 0).all())
    torch.testing.assert_close(actual.sum(), torch.tensor(1.0))
    assert torch.equal(project_simplex(actual), actual)
    assert projected_gradient_residual([0.25] * 4, [0.0] * 4) == 0
    with pytest.raises(ValueError, match='finite'):
        project_simplex([0.0, float('nan')])


@pytest.mark.parametrize('method,learning_rate', [('pgd', 1.0), ('adam', 0.1)])
def test_solvers_reduce_objective(method, learning_rate):
    result = solve(
        Quadratic([0.05, 0.15, 0.3, 0.2, 0.3]),
        [0.2] * 5,
        method=method,
        steps=50,
        learning_rate=learning_rate,
        tolerance=1e-5,
    )
    assert result['best']['value'] < result['history'][0]['value']
    assert abs(sum(result['best']['alpha']) - 1) < 2e-7
    assert min(result['best']['alpha']) >= 0


def test_artifact_calibration_and_json_result(tmp_path):
    artifact = tmp_path / 'artifact.pt'
    teacher = torch.tensor([
        [3.0, 0.0],
        [0.0, 3.0],
        [3.0, 0.0],
        [0.0, 3.0],
    ])
    base = torch.zeros_like(teacher)
    deltas = torch.stack([
        torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]),
        torch.tensor([[0.0, 2.0], [2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]),
    ], dim=1)
    torch.save({
        'format': ARTIFACT_FORMAT,
        'coefficient_names': ['aligned', 'opposed'],
        'student_base_logits': base,
        'student_logit_deltas': deltas,
        'teacher_logits': teacher,
        'labels': torch.tensor([0, 1, 0, 1]),
        'sequence_offsets': torch.tensor([0, 2, 4]),
        'sequence_weights': torch.tensor([0.5, 0.5]),
    }, artifact)
    result = calibrate(
        artifact,
        objective='fkl',
        method='pgd',
        steps=8,
        learning_rate=1.0,
    )
    assert result['best']['alpha'][0] > 0.9
    json.dumps(result, allow_nan=False)
