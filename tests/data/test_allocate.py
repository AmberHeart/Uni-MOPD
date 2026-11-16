# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
import hashlib

import pytest

from unimopd.data.allocate import UNIMOPD_DOMAINS, allocate_records, compute_quotas


DOMAINS = UNIMOPD_DOMAINS


def pool(size=8):
    return [
        {'id': f'{domain}-{index}', 'domain': domain, 'messages': [{'role': 'user', 'content': str(index)}]}
        for domain in DOMAINS
        for index in range(size)
    ]


def test_uniform_is_without_replacement_and_deterministic():
    first, manifest = allocate_records(pool(), 25, domains=DOMAINS, selection_seed=7, order_seed=11)
    second, _ = allocate_records(reversed(pool()), 25, domains=DOMAINS, selection_seed=7, order_seed=11)

    assert first == second
    assert manifest['quotas'] == dict.fromkeys(DOMAINS, 5)
    assert len({row['id'] for row in first}) == 25
    assert {domain: sum(row['domain'] == domain for row in first) for domain in DOMAINS} == manifest['quotas']
    assert manifest['selection_order'] == (
        'sha256(selection_seed:selection:query_id), independently within each domain')
    assert manifest['global_order'] == 'sha256(order_seed:global-order:query_id)'


def test_hash_ranking_matches_the_frozen_source_recipe():
    rows = pool()
    selected, _ = allocate_records(rows, 25, selection_seed=7, order_seed=11)
    expected = []
    for domain in DOMAINS:
        candidates = [row for row in rows if row['domain'] == domain]
        candidates.sort(key=lambda row: (
            hashlib.sha256(f'7:selection:{row["id"]}'.encode()).digest(), row['id']))
        expected.extend(candidates[:5])
    expected.sort(key=lambda row: (
        hashlib.sha256(f'11:global-order:{row["id"]}'.encode()).digest(), row['id']))
    assert selected == expected


def test_selection_and_global_order_seeds_are_independent():
    baseline, _ = allocate_records(pool(), 20, domains=DOMAINS, selection_seed=7, order_seed=11)
    reordered, _ = allocate_records(pool(), 20, domains=DOMAINS, selection_seed=7, order_seed=12)
    reselected, _ = allocate_records(pool(), 20, domains=DOMAINS, selection_seed=8, order_seed=11)

    assert {row['id'] for row in baseline} == {row['id'] for row in reordered}
    assert [row['id'] for row in baseline] != [row['id'] for row in reordered]
    assert {row['id'] for row in baseline} != {row['id'] for row in reselected}


def test_gap_and_gap_sqrt_token_quotas():
    signals = {
        'StemPuzzle': {'gap': 4, 'mean_tokens': 16},
        'GeneralVQA': {'gap': 3, 'mean_tokens': 9},
        'TextChartUnder': {'gap': 2, 'mean_tokens': 4},
        'guiagent': {'gap': 1, 'mean_tokens': 1},
        'EmbodUnde': {'gap': 0, 'mean_tokens': 1},
    }
    assert compute_quotas(20, DOMAINS, policy='gap', signals=signals) == {
        'StemPuzzle': 8, 'GeneralVQA': 6, 'TextChartUnder': 4, 'guiagent': 2, 'EmbodUnde': 0}
    assert compute_quotas(20, DOMAINS, policy='gap_sqrt_tokens', signals=signals) == {
        'StemPuzzle': 5, 'GeneralVQA': 5, 'TextChartUnder': 5, 'guiagent': 5, 'EmbodUnde': 0}


def test_capacity_and_identity_fail_loudly():
    with pytest.raises(ValueError, match='capacity'):
        allocate_records(pool(size=1), 10, domains=DOMAINS)
    duplicate = pool()
    duplicate[-1]['id'] = duplicate[0]['id']
    with pytest.raises(ValueError, match='globally unique'):
        allocate_records(duplicate, 10, domains=DOMAINS)
