# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic, without-replacement allocation from a multi-domain JSONL pool."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


POLICIES = ('uniform', 'gap', 'gap_sqrt_tokens')
UNIMOPD_DOMAINS = (
    'StemPuzzle',
    'GeneralVQA',
    'TextChartUnder',
    'guiagent',
    'EmbodUnde',
)


def _rank(seed: int, stage: str, query_id: str) -> bytes:
    return hashlib.sha256(f'{seed}:{stage}:{query_id}'.encode()).digest()


def _query_id(row: Mapping) -> str:
    value = row.get('id', row.get('query_id'))
    if not isinstance(value, (str, int)) or str(value) == '':
        raise ValueError('Each row needs a nonempty string or integer id/query_id.')
    return str(value)


def _weights(
    domains: Sequence[str],
    policy: str,
    signals: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, float]:
    if policy not in POLICIES:
        raise ValueError(f'Unknown allocation policy {policy!r}; choose from {POLICIES}.')
    if policy == 'uniform':
        return dict.fromkeys(domains, 1.0)
    if signals is None or set(signals) != set(domains):
        raise ValueError('Signal policies require exactly one signal entry per domain.')

    weights = {}
    for domain in domains:
        signal = signals[domain]
        gap = signal.get('gap')
        if (isinstance(gap, bool) or not isinstance(gap, (int, float))
                or not math.isfinite(gap) or gap < 0):
            raise ValueError(f'{domain!r}.gap must be finite and nonnegative.')
        weight = float(gap)
        if policy == 'gap_sqrt_tokens':
            tokens = signal.get('mean_tokens')
            if (isinstance(tokens, bool) or not isinstance(tokens, (int, float))
                    or not math.isfinite(tokens) or tokens <= 0):
                raise ValueError(f'{domain!r}.mean_tokens must be finite and positive.')
            weight /= math.sqrt(tokens)
        weights[domain] = weight
    if not any(weights.values()):
        raise ValueError('At least one domain must have a positive allocation weight.')
    return weights


def compute_quotas(
    total: int,
    domains: Sequence[str],
    *,
    policy: str = 'uniform',
    signals: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, int]:
    """Allocate an integer total with the deterministic largest-remainder rule."""
    if type(total) is not int or total <= 0:
        raise ValueError('total must be a positive integer.')
    domains = tuple(domains)
    if not domains or any(not isinstance(domain, str) or not domain for domain in domains):
        raise ValueError('domains must contain nonempty strings.')
    if len(domains) != len(set(domains)):
        raise ValueError('domains must be unique.')

    weights = _weights(domains, policy, signals)
    weight_sum = sum(weights.values())
    exact = {domain: total * weights[domain] / weight_sum for domain in domains}
    quotas = {domain: math.floor(exact[domain]) for domain in domains}
    remainder = total - sum(quotas.values())
    order = sorted(domains, key=lambda domain: (-(exact[domain] - quotas[domain]), domain))
    for domain in order[:remainder]:
        quotas[domain] += 1
    return quotas


def allocate_records(
    records: Iterable[Mapping],
    total: int,
    *,
    domains: Sequence[str] | None = None,
    policy: str = 'uniform',
    signals: Mapping[str, Mapping[str, float]] | None = None,
    selection_seed: int = 20261015,
    order_seed: int = 42,
) -> tuple[list[dict], dict]:
    """Select within each domain without replacement, then globally hash-shuffle."""
    rows = [dict(row) for row in records]
    if not rows:
        raise ValueError('The input pool is empty.')
    ids = [_query_id(row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Query IDs must be globally unique.')
    for row in rows:
        if not isinstance(row.get('domain'), str) or not row['domain']:
            raise ValueError('Each row needs a nonempty domain field.')

    available = Counter(row['domain'] for row in rows)
    domains = UNIMOPD_DOMAINS if domains is None else tuple(domains)
    unexpected = sorted(set(available) - set(domains))
    if unexpected:
        raise ValueError(f'Input contains domains outside --domains: {unexpected}.')
    quotas = compute_quotas(total, domains, policy=policy, signals=signals)
    short = {domain: (quotas[domain], available[domain])
             for domain in domains if available[domain] < quotas[domain]}
    if short:
        raise ValueError(f'Insufficient without-replacement capacity: {short}.')

    selected = []
    for domain in domains:
        candidates = [row for row in rows if row['domain'] == domain]
        candidates.sort(key=lambda row: (_rank(selection_seed, 'selection', _query_id(row)), _query_id(row)))
        selected.extend(candidates[:quotas[domain]])
    selected.sort(
        key=lambda row: (_rank(order_seed, 'global-order', _query_id(row)), _query_id(row)))
    manifest = {
        'policy': policy,
        'selection_seed': int(selection_seed),
        'order_seed': int(order_seed),
        'total': len(selected),
        'domains': list(domains),
        'quotas': quotas,
        'available': {domain: available[domain] for domain in domains},
        'without_replacement': True,
        'global_order': 'sha256(order_seed:global-order:query_id)',
        'selection_order': 'sha256(selection_seed:selection:query_id), independently within each domain',
    }
    if signals is not None:
        manifest['signals'] = {domain: dict(signals[domain]) for domain in domains}
    return selected, manifest


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f'{path}:{line_number} is not a JSON object.')
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Multi-domain JSONL query pool.')
    parser.add_argument('--output', type=Path, required=True, help='Fresh output JSONL path.')
    parser.add_argument('--manifest', type=Path, help='Defaults to <output>.manifest.json.')
    parser.add_argument('--total', type=int, required=True)
    parser.add_argument(
        '--domains',
        nargs='+',
        default=list(UNIMOPD_DOMAINS),
        help='Ordered domain set; defaults to the five Uni-MOPD training keys.',
    )
    parser.add_argument('--policy', choices=POLICIES, default='uniform')
    parser.add_argument('--signals', type=Path, help='JSON mapping used by gap policies.')
    parser.add_argument('--selection-seed', type=int, default=20261015)
    parser.add_argument('--order-seed', type=int, default=42)
    args = parser.parse_args()

    signals = json.loads(args.signals.read_text()) if args.signals else None
    selected, manifest = allocate_records(
        _read_jsonl(args.input),
        args.total,
        domains=args.domains,
        policy=args.policy,
        signals=signals,
        selection_seed=args.selection_seed,
        order_seed=args.order_seed,
    )
    _write_jsonl(args.output, selected)
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + '.manifest.json')
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    main()
