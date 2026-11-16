# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract for the optional equal-share domain sampler."""
import types

import pytest
import torch

from tests.mopd.source_test_utils import load_symbols
from tests.mopd.test_domain_routing import published_mapping

DOMAINS = published_mapping()


def sampler_class():
    return load_symbols(
        'ms-swift/swift/megatron/trainers/utils.py',
        ['DomainBalancedRandomSampler'],
        namespace={
            'Optional': types.SimpleNamespace,
            'Dict': dict,
            'RandomSeedDataset': type('RandomSeedDataset', (), {}),
        },
    )['DomainBalancedRandomSampler']


class Dataset:
    """Column-accessible dataset, matching the shape Megatron hands the sampler."""

    def __init__(self, domains):
        self._domains = list(domains)

    def __len__(self):
        return len(self._domains)

    def __getitem__(self, key):
        if key == 'domain':
            return self._domains
        return {'domain': self._domains[key]}


def build(domains, *, micro_batch_size=1, dp_rank=0, dp_size=1, consumed=0, shuffle=True):
    dataset = Dataset(domains)
    return sampler_class()(
        dataset,
        total_samples=len(dataset),
        consumed_samples=consumed,
        micro_batch_size=micro_batch_size,
        data_parallel_rank=dp_rank,
        data_parallel_size=dp_size,
        data_sharding=True,
        shuffle=shuffle,
        domain_teacher_mapping=DOMAINS,
    )


def even_domains(per_domain):
    return [domain for domain in DOMAINS for _ in range(per_domain)]


def drain(sampler, dataset_domains):
    return [dataset_domains[idx] for batch in sampler for idx in batch]


def test_every_domain_run_covers_all_five_in_canonical_order():
    domains = even_domains(4)
    sampler = build(domains)
    emitted = drain(sampler, domains)

    assert len(emitted) == 20
    order = sorted(DOMAINS, key=lambda d: DOMAINS[d])
    for start in range(0, len(emitted), len(order)):
        assert emitted[start:start + len(order)] == order


def test_each_query_is_emitted_at_most_once_per_epoch():
    domains = even_domains(4)
    sampler = build(domains)
    indices = [idx for batch in sampler for idx in batch]

    assert len(indices) == len(set(indices))


def test_data_parallel_ranks_receive_disjoint_shares():
    domains = even_domains(8)
    shares = [set(idx for batch in build(domains, dp_rank=rank, dp_size=4) for idx in batch)
              for rank in range(4)]

    # 8 queries per domain over 4 ranks: 2 per domain per rank, 5 domains.
    assert all(len(share) == 10 for share in shares)
    assert set.intersection(*shares) == set()
    assert len(set.union(*shares)) == 40


def test_the_stream_is_deterministic_for_a_fixed_epoch():
    domains = even_domains(4)
    first = [idx for batch in build(domains) for idx in batch]
    second = [idx for batch in build(domains) for idx in batch]

    assert first == second


def test_unshuffled_order_follows_the_dataset():
    domains = even_domains(2)
    sampler = build(domains, shuffle=False)
    indices = [idx for batch in sampler for idx in batch]

    # Canonical bucket order, first element of each bucket, then the second.
    assert indices == [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]


def test_resume_skips_exactly_the_queries_already_consumed():
    domains = even_domains(4)
    full = [idx for batch in build(domains) for idx in batch]
    resumed = [idx for batch in build(domains, consumed=5) for idx in batch]

    assert resumed == full[5:]


def test_microbatches_are_whole_and_advance_consumed_samples_by_the_global_step():
    domains = even_domains(5)
    sampler = build(domains, micro_batch_size=5, dp_size=1)
    batches = list(sampler)

    assert all(len(batch) == 5 for batch in batches)
    assert sampler.consumed_samples == len(batches) * 5
    assert sampler.samples_per_epoch == 25


@pytest.mark.parametrize('domains,match', [
    (even_domains(4)[:-1] + ['audio'], 'not in domain_teacher_mapping'),
    # Every mapped domain needs queries; one missing domain cannot be balanced.
    ([d for d in DOMAINS if DOMAINS[d] != 3] * 4, 'cannot domain-balance'),
])
def test_unusable_domain_columns_fail_loudly(domains, match):
    with pytest.raises(ValueError, match=match):
        build(domains)


def test_a_bucket_smaller_than_the_dp_size_fails_loudly():
    with pytest.raises(ValueError, match='per-rank effective length is 0'):
        build(even_domains(2), dp_size=4)


def test_a_microbatch_that_would_drop_a_partial_tail_fails_loudly():
    with pytest.raises(ValueError, match='not divisible by micro_batch_size'):
        build(even_domains(3), micro_batch_size=2)


def test_group_by_length_is_rejected_as_incompatible():
    with pytest.raises(ValueError, match='mutually exclusive'):
        domains = even_domains(4)
        sampler_class()(
            Dataset(domains),
            total_samples=len(domains),
            consumed_samples=0,
            micro_batch_size=1,
            data_parallel_rank=0,
            data_parallel_size=1,
            data_sharding=True,
            shuffle=True,
            group_by_length=True,
            domain_teacher_mapping=DOMAINS,
        )


def test_an_empty_mapping_is_rejected():
    domains = even_domains(4)
    with pytest.raises(ValueError, match='non-empty domain_teacher_mapping'):
        sampler_class()(
            Dataset(domains),
            total_samples=len(domains),
            consumed_samples=0,
            micro_batch_size=1,
            data_parallel_rank=0,
            data_parallel_size=1,
            data_sharding=True,
            domain_teacher_mapping={},
        )


def test_row_access_datasets_are_supported():
    class RowOnly(Dataset):

        def __getitem__(self, key):
            if key == 'domain':
                raise TypeError('column access unsupported')
            return {'domain': self._domains[key]}

    domains = even_domains(4)
    sampler = sampler_class()(
        RowOnly(domains),
        total_samples=len(domains),
        consumed_samples=0,
        micro_batch_size=1,
        data_parallel_rank=0,
        data_parallel_size=1,
        data_sharding=True,
        domain_teacher_mapping=DOMAINS,
    )

    assert sampler.domains == domains
    assert len([idx for batch in sampler for idx in batch]) == 20


def test_torch_generator_shuffle_is_seeded_by_epoch_not_by_rank():
    domains = even_domains(8)
    # Both ranks must see the same bucket permutation, otherwise contiguous
    # DP slicing would not be disjoint.
    first, second = (build(domains, dp_rank=rank, dp_size=2) for rank in range(2))
    list(first), list(second)

    assert first.epoch == second.epoch == 0
    assert torch.equal(torch.tensor(first._buckets), torch.tensor(second._buckets))
