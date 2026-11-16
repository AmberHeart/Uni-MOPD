# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for domain-to-teacher routing and single-teacher microbatching."""
import types
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

from tests.mopd.source_test_utils import gkd_symbols

ROOT = Path(__file__).resolve().parents[2]


def published_recipe():
    text = (ROOT / 'configs' / 'train' / 'unimopd_uniform_rkl.yaml').read_text()
    return yaml.safe_load(text.replace('${oc.env:', '${'))


def published_mapping():
    """The five domain keys the released recipe actually routes on."""
    return published_recipe()['domain_teacher_mapping']


def published_normalization():
    """The loss normalization the released recipe trains with."""
    return published_recipe()['mopd_loss_normalization']


DOMAINS = published_mapping()
STEM, VQA, DOC, GUI, EMB = sorted(DOMAINS, key=lambda domain: DOMAINS[domain])


BOUND_SYMBOLS = ('_get_teacher_idx_for_sample', '_group_batch_by_teacher')
# Declared ``@staticmethod`` in production; the loader strips the decorator, so
# they are used unbound here rather than attached to the trainer.
STATIC_SYMBOLS = ('_build_domain_base_weights', '_compute_domain_loss_scales')


def trainer_for(mapping=None, *, multi_teacher=True):
    symbols = gkd_symbols(list(BOUND_SYMBOLS + STATIC_SYMBOLS), {'defaultdict': defaultdict})
    trainer = types.SimpleNamespace(
        multi_teacher_mode=multi_teacher,
        domain_teacher_mapping=DOMAINS if mapping is None else mapping,
    )
    for name in BOUND_SYMBOLS:
        setattr(trainer, name, types.MethodType(symbols[name], trainer))
    for name in STATIC_SYMBOLS:
        setattr(trainer, name, symbols[name])
    return trainer


def samples(*domains):
    return [{'domain': domain, 'id': index} for index, domain in enumerate(domains)]


def test_the_recipe_routes_five_domains_to_five_distinct_teachers():
    assert sorted(DOMAINS.values()) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize('domain,expected', sorted(DOMAINS.items()))
def test_every_published_domain_routes_to_its_own_teacher(domain, expected):
    assert trainer_for()._get_teacher_idx_for_sample({'domain': domain}) == expected


def test_single_teacher_mode_ignores_the_domain_tag():
    trainer = trainer_for(multi_teacher=False)
    assert trainer._get_teacher_idx_for_sample({'domain': GUI}) == 0
    assert trainer._get_teacher_idx_for_sample({}) == 0


@pytest.mark.parametrize('sample,match', [
    ({'id': 0}, 'requires every sample to have a "domain" field'),
    ({'domain': 'audio'}, 'not in domain_teacher_mapping'),
])
def test_unroutable_samples_fail_loudly(sample, match):
    with pytest.raises(ValueError, match=match):
        trainer_for()._get_teacher_idx_for_sample(sample)


def test_each_microbatch_carries_exactly_one_teacher():
    batch = samples(GUI, STEM, VQA, STEM, GUI, VQA)
    microbatches = trainer_for()._group_batch_by_teacher(batch, num_microbatches=3)

    assert [teacher for _, teacher in microbatches] == [0, 1, 3]
    for rows, teacher in microbatches:
        assert {DOMAINS[row['domain']] for row in rows} == {teacher}


def test_grouping_preserves_every_sample_exactly_once():
    batch = samples(EMB, GUI, DOC, GUI, EMB, DOC)
    microbatches = trainer_for()._group_batch_by_teacher(batch, num_microbatches=6)

    regrouped = [row for rows, _ in microbatches for row in rows]
    assert sorted(row['id'] for row in regrouped) == list(range(len(batch)))


def test_domain_histogram_that_cannot_tile_the_grid_is_rejected():
    # Three GUI and one STEM sample cannot split into microbatches of two
    # without padding one teacher's microbatch or dropping a real sample.
    with pytest.raises(ValueError, match='multiple of micro_batch_size'):
        trainer_for()._group_batch_by_teacher(samples(GUI, GUI, GUI, STEM), num_microbatches=2)


def test_single_teacher_mode_keeps_the_plain_even_split():
    batch = samples(GUI, STEM, VQA, EMB)
    microbatches = trainer_for(multi_teacher=False)._group_batch_by_teacher(batch, num_microbatches=2)

    assert [teacher for _, teacher in microbatches] == [0, 0]
    assert [row['id'] for rows, _ in microbatches for row in rows] == [0, 1, 2, 3]


def test_unspecified_domain_weights_are_uniform():
    weights = trainer_for()._build_domain_base_weights(DOMAINS, None)
    assert weights == {domain: 1.0 for domain in DOMAINS}


def test_domains_sharing_a_teacher_keep_separate_weights():
    shared = {GUI: 0, f'{GUI}_mobile': 0}
    weights = trainer_for()._build_domain_base_weights(shared, {GUI: 2.0})
    assert weights == {GUI: 2.0, f'{GUI}_mobile': 1.0}


def test_equal_domains_share_the_step_gradient_equally():
    trainer = trainer_for()
    counts = {domain: 4.0 for domain in DOMAINS}
    scales = trainer._compute_domain_loss_scales(
        counts, {domain: 40.0 for domain in DOMAINS},
        trainer._build_domain_base_weights(DOMAINS, None),
        'domain_sequence', m_global=20.0)

    # scale_d = M_global * q_d / B_d with q_d = 1/5 for five equal domains.
    assert scales == pytest.approx({domain: 20.0 * 0.2 / 4.0 for domain in DOMAINS})


def test_absent_domains_renormalize_over_the_domains_present():
    trainer = trainer_for()
    present = {STEM: 2.0, GUI: 2.0}
    scales = trainer._compute_domain_loss_scales(
        present, {STEM: 20.0, GUI: 20.0},
        trainer._build_domain_base_weights(DOMAINS, None),
        'domain_sequence', m_global=4.0)

    assert set(scales) == {STEM, GUI}
    assert scales == pytest.approx({STEM: 4.0 * 0.5 / 2.0, GUI: 4.0 * 0.5 / 2.0})


def test_token_mode_divides_by_tokens_and_sequence_mode_by_sequences():
    trainer = trainer_for()
    weights = trainer._build_domain_base_weights(DOMAINS, None)
    counts = {domain: 4.0 for domain in DOMAINS}
    tokens = {STEM: 100.0, VQA: 40.0, DOC: 40.0, GUI: 10.0, EMB: 40.0}

    by_token = trainer._compute_domain_loss_scales(counts, tokens, weights, 'domain_token', 20.0)
    by_sequence = trainer._compute_domain_loss_scales(counts, tokens, weights, 'domain_sequence', 20.0)

    assert by_token == pytest.approx({d: 20.0 * 0.2 / tokens[d] for d in DOMAINS})
    assert by_sequence == pytest.approx({d: 20.0 * 0.2 / 4.0 for d in DOMAINS})


def test_a_domain_with_samples_but_no_valid_tokens_is_not_scaled():
    trainer = trainer_for()
    tokens = {domain: 40.0 for domain in DOMAINS}
    tokens[GUI] = 0.0
    scales = trainer._compute_domain_loss_scales(
        {domain: 4.0 for domain in DOMAINS}, tokens,
        trainer._build_domain_base_weights(DOMAINS, None),
        'domain_sequence', m_global=16.0)

    assert GUI not in scales
    assert scales == pytest.approx({d: 16.0 * 0.25 / 4.0 for d in DOMAINS if d != GUI})
