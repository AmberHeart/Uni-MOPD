# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0

"""Projected simplex solvers for merge coefficients."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch


def project_simplex(values: Sequence[float] | torch.Tensor) -> torch.Tensor:
    """Project a finite vector onto ``alpha >= 0, sum(alpha) = 1`` in FP32."""
    vector = torch.as_tensor(values, dtype=torch.float64, device='cpu').flatten()
    if vector.numel() == 0 or not bool(torch.isfinite(vector).all()):
        raise ValueError('Coefficient vector must contain finite values.')
    ordered, _ = torch.sort(vector, descending=True)
    cumulative = torch.cumsum(ordered, dim=0) - 1
    indices = torch.arange(1, len(vector) + 1, dtype=vector.dtype)
    active = ordered - cumulative / indices > 0
    rho = int(torch.nonzero(active, as_tuple=False)[-1])
    threshold = cumulative[rho] / (rho + 1)
    projected = torch.clamp(vector - threshold, min=0).to(torch.float32)
    # Keep the serialized FP32 point on the simplex after rounding.
    correction = projected.new_tensor(1.0) - projected.sum()
    projected[int(torch.argmax(projected))] += correction
    if bool((projected < 0).any()) or not torch.isclose(
            projected.sum(), projected.new_tensor(1.0), atol=2e-7, rtol=0):
        raise RuntimeError('FP32 simplex projection failed.')
    return projected


def projected_gradient_residual(
    alpha: Sequence[float] | torch.Tensor,
    gradient: Sequence[float] | torch.Tensor,
    *,
    step_size: float = 1.0,
) -> float:
    """Return the norm of the projected-gradient mapping."""
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError('step_size must be finite and positive.')
    point = project_simplex(alpha)
    grad = torch.as_tensor(gradient, dtype=torch.float32, device='cpu')
    if grad.shape != point.shape or not bool(torch.isfinite(grad).all()):
        raise ValueError('gradient must be finite and match alpha.')
    projected = project_simplex(point - step_size * grad)
    return float(torch.linalg.vector_norm((point - projected) / step_size))


def _oracle_result(oracle: Callable, alpha: torch.Tensor, *, need_gradient: bool):
    answer = oracle(alpha.detach().clone(), need_gradient)
    if not isinstance(answer, tuple) or len(answer) != 2:
        raise ValueError('oracle must return (value, gradient_or_none).')
    value, gradient = answer
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('oracle returned a nonfinite value.')
    if need_gradient:
        gradient = torch.as_tensor(gradient, dtype=torch.float32, device='cpu')
        if gradient.shape != alpha.shape or not bool(torch.isfinite(gradient).all()):
            raise ValueError('oracle gradient must be finite and match alpha.')
    elif gradient is not None:
        raise ValueError('value-only oracle returned an unexpected gradient.')
    return value, gradient


def solve(
    oracle: Callable,
    initial_alpha: Sequence[float] | torch.Tensor,
    *,
    method: str,
    steps: int,
    learning_rate: float,
    tolerance: float = 1e-7,
    armijo_c: float = 1e-4,
    backtrack: float = 0.5,
    max_trials: int = 8,
    adam_betas: tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
) -> dict:
    """Minimize an oracle on the probability simplex with PGD or Adam."""
    if method not in {'pgd', 'adam'}:
        raise ValueError('method must be pgd or adam.')
    if type(steps) is not int or steps < 0:
        raise ValueError('steps must be a nonnegative integer.')
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate must be finite and positive.')
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('tolerance must be finite and nonnegative.')
    if not 0 < armijo_c < 1 or not 0 < backtrack < 1:
        raise ValueError('Invalid Armijo backtracking settings.')
    if type(max_trials) is not int or max_trials <= 0:
        raise ValueError('max_trials must be a positive integer.')

    alpha = project_simplex(initial_alpha)
    parameter = torch.nn.Parameter(alpha.clone())
    optimizer = (
        torch.optim.Adam(
            [parameter],
            lr=learning_rate,
            betas=adam_betas,
            eps=adam_eps,
            weight_decay=0,
        )
        if method == 'adam' else None
    )
    history = []
    stop_reason = 'max_steps'

    for iteration in range(steps + 1):
        value, gradient = _oracle_result(oracle, alpha, need_gradient=True)
        residual = projected_gradient_residual(alpha, gradient)
        history.append({
            'iteration': iteration,
            'alpha': alpha.tolist(),
            'value': value,
            'gradient': gradient.tolist(),
            'projected_gradient_residual': residual,
        })
        if residual <= tolerance:
            stop_reason = 'stationary'
            break
        if iteration == steps:
            break

        if method == 'adam':
            assert optimizer is not None
            with torch.no_grad():
                parameter.copy_(alpha)
            optimizer.zero_grad(set_to_none=True)
            parameter.grad = gradient.clone()
            optimizer.step()
            alpha = project_simplex(parameter.detach())
            continue

        accepted = False
        step_size = learning_rate
        for _ in range(max_trials):
            candidate = project_simplex(alpha - step_size * gradient)
            candidate_value, _ = _oracle_result(
                oracle, candidate, need_gradient=False)
            displacement = candidate - alpha
            armijo_bound = value + armijo_c * float(torch.dot(gradient, displacement))
            if candidate_value <= armijo_bound:
                alpha = candidate
                accepted = True
                break
            step_size *= backtrack
        if not accepted:
            stop_reason = 'line_search_failed'
            break

    best = min(history, key=lambda point: (point['value'], point['iteration']))
    return {
        'method': method,
        'initial_alpha': history[0]['alpha'],
        'final_alpha': history[-1]['alpha'],
        'best': best,
        'stop_reason': stop_reason,
        'history': history,
    }
