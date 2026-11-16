# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0

"""Calibrate merge coefficients from a response-logit artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import torch

from .objectives import OBJECTIVES, calibration_objective
from .solver import solve

ARTIFACT_FORMAT = 'unimopd_response_logit_calibration_v1'


class LogitArtifactOracle:
    """Differentiable CPU oracle for an affine response-logit artifact."""

    def __init__(self, payload: dict, *, objective: str, temperature: float):
        required = {
            'format', 'coefficient_names', 'student_base_logits',
            'student_logit_deltas', 'teacher_logits', 'labels',
            'sequence_offsets', 'sequence_weights',
        }
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise ValueError(f'Calibration artifact requires fields {sorted(required)}.')
        if payload['format'] != ARTIFACT_FORMAT:
            raise ValueError(f'Unsupported calibration artifact format: {payload["format"]!r}.')
        names = payload['coefficient_names']
        if (not isinstance(names, list) or not names
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)):
            raise ValueError('coefficient_names must be unique nonempty strings.')
        self.coefficient_names = names
        self.base = self._tensor(payload['student_base_logits'], 'student_base_logits')
        self.deltas = self._tensor(payload['student_logit_deltas'], 'student_logit_deltas')
        self.teacher = self._tensor(payload['teacher_logits'], 'teacher_logits').detach()
        self.labels = torch.as_tensor(payload['labels'], dtype=torch.long, device='cpu')
        self.offsets = torch.as_tensor(
            payload['sequence_offsets'], dtype=torch.long, device='cpu')
        self.weights = self._tensor(payload['sequence_weights'], 'sequence_weights')
        self.objective = objective
        self.temperature = float(temperature)
        if (self.base.ndim != 2 or self.teacher.shape != self.base.shape
                or self.deltas.shape != (self.base.shape[0], len(names), self.base.shape[1])
                or self.labels.shape != self.base.shape[:1]):
            raise ValueError('Calibration artifact logit/delta/label shapes are inconsistent.')
        # Validate the sequence contract before starting a solver.
        calibration_objective(
            self.base,
            self.teacher,
            self.labels,
            self.offsets,
            self.weights,
            objective=self.objective,
            temperature=self.temperature,
        )

    @staticmethod
    def _tensor(value, name):
        tensor = torch.as_tensor(value, device='cpu')
        if not torch.is_floating_point(tensor) or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f'{name} must be a finite floating-point tensor.')
        return tensor.float()

    def __call__(self, alpha: torch.Tensor, need_gradient: bool):
        if alpha.shape != (len(self.coefficient_names),):
            raise ValueError('alpha length differs from coefficient_names.')
        coefficient = alpha.detach().clone().requires_grad_(need_gradient)
        student = self.base + torch.einsum('k,nkv->nv', coefficient, self.deltas)
        value = calibration_objective(
            student,
            self.teacher,
            self.labels,
            self.offsets,
            self.weights,
            objective=self.objective,
            temperature=self.temperature,
        )
        gradient = (
            torch.autograd.grad(value, coefficient)[0].detach()
            if need_gradient else None
        )
        return float(value.detach()), gradient


def load_artifact(path: str | Path) -> dict:
    """Load a tensor-only artifact without enabling arbitrary pickle objects."""
    return torch.load(Path(path), map_location='cpu', weights_only=True)


def calibrate(
    artifact: str | Path,
    *,
    objective: str,
    method: str,
    steps: int,
    learning_rate: float,
    temperature: float = 1.0,
    initial_alpha: list[float] | None = None,
) -> dict:
    """Run coefficient calibration and return a JSON-serializable result."""
    oracle = LogitArtifactOracle(
        load_artifact(artifact), objective=objective, temperature=temperature)
    if initial_alpha is None:
        initial_alpha = [1 / len(oracle.coefficient_names)] * len(oracle.coefficient_names)
    result = solve(
        oracle,
        initial_alpha,
        method=method,
        steps=steps,
        learning_rate=learning_rate,
    )
    result.update({
        'artifact_format': ARTIFACT_FORMAT,
        'objective': objective,
        'temperature': temperature,
        'coefficient_names': oracle.coefficient_names,
    })
    return result


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--objective', choices=OBJECTIVES, required=True)
    parser.add_argument('--solver', choices=('pgd', 'adam'), default='pgd')
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--initial-alpha', nargs='*', type=float)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    result = calibrate(
        args.artifact,
        objective=args.objective,
        method=args.solver,
        steps=args.steps,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        initial_alpha=args.initial_alpha or None,
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite calibration output: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
