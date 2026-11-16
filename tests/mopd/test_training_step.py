# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end MOPD step on a tiny student and five mock teachers.

Composes the production query sampler, domain routing, single-teacher
microbatching, the full-vocabulary reverse-KL head and the domain loss scales
into an optimizer loop small enough to run on CPU. This is the whole training
contract short of Megatron's distributed plumbing.
"""
import types
from collections import defaultdict

import pytest
import torch

from tests.mopd.source_test_utils import gkd_symbols, load_module, load_symbols, load_vocab_parallel
from tests.mopd.test_domain_routing import published_mapping, published_normalization

DOMAINS = published_mapping()
PUBLISHED_MODE = published_normalization()
HIDDEN, VOCAB, SEQ = 8, 23, 6


class Head(torch.nn.Module):
    """Output head in Megatron's native (sequence, batch, hidden) convention."""

    def __init__(self, generator=None):
        super().__init__()
        weight = torch.randn(VOCAB, HIDDEN, generator=generator) * 0.2
        self.weight = torch.nn.Parameter(weight)
        self.bias = None
        self.sequence_parallel = False

    def forward(self, hidden, *, weight=None, runtime_gather_output=None):
        return hidden @ (self.weight if weight is None else weight).T, None


def mpu_stub():
    return types.SimpleNamespace(
        get_tensor_model_parallel_world_size=lambda: 1,
        get_tensor_model_parallel_rank=lambda: 0,
        get_tensor_model_parallel_group=lambda: None,
        get_data_parallel_group=lambda: None,
        get_context_parallel_world_size=lambda: 1,
    )


@pytest.fixture(scope='module')
def stack():
    """Production symbols for one MOPD step, loaded without CUDA or Megatron."""
    mpu = mpu_stub()
    vocab = load_vocab_parallel(mpu)
    criterion = types.SimpleNamespace(
        loss_temperature=1.0,
        teacher_label_smoothing=0.0,
        gkd_loss_type='jsd',
        gkd_todi_beta=1.0,
        multi_teacher_mode=True,
        domain_teacher_mapping=DOMAINS,
    )
    symbols = gkd_symbols(
        [
            '_align_vocab_size',
            'generalized_jsd_loss',
            '_get_teacher_idx_for_sample',
            '_group_batch_by_teacher',
            '_build_domain_base_weights',
            '_compute_domain_loss_scales',
        ],
        {
            **{name: getattr(vocab, name) for name in dir(vocab) if name.startswith('vocab_parallel_')},
            'mpu': mpu,
            'defaultdict': defaultdict,
            'get_args': lambda: types.SimpleNamespace(context_parallel_size=1, padded_vocab_size=VOCAB),
        },
    )
    for name in ('_align_vocab_size', 'generalized_jsd_loss', '_get_teacher_idx_for_sample',
                 '_group_batch_by_teacher'):
        setattr(criterion, name, types.MethodType(symbols[name], criterion))
    for name in ('_build_domain_base_weights', '_compute_domain_loss_scales'):
        setattr(criterion, name, symbols[name])
    sampler = load_symbols(
        'ms-swift/swift/megatron/trainers/utils.py',
        ['DomainBalancedRandomSampler'],
        namespace={
            'Optional': types.SimpleNamespace,
            'Dict': dict,
            'RandomSeedDataset': type('RandomSeedDataset', (), {}),
        },
    )['DomainBalancedRandomSampler']
    stream = load_module('ms-swift/swift/megatron/trainers/streamed_distillation.py',
                         'unimopd_test_training_step_stream')
    return types.SimpleNamespace(trainer=criterion, sampler=sampler, stream=stream)


class Queries:
    """One query per row, tagged with the domain that selects its teacher."""

    def __init__(self, per_domain):
        self.rows = [{'domain': domain, 'id': index}
                     for index, domain in enumerate(d for d in DOMAINS for _ in range(per_domain))]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if key == 'domain':
            return [row['domain'] for row in self.rows]
        return self.rows[key]


def rollout(query, generator):
    """Stand in for the student's on-policy sample: hidden states plus a response mask.

    Response lengths vary, as they do in a real rollout. That is what separates
    global-token normalization from per-sequence averaging: with equal-length
    responses the two are numerically identical and nothing is pinned down.
    """
    prompt = 1 + int(torch.randint(0, SEQ - 2, (1, ), generator=generator))
    labels = torch.full((1, SEQ), -100)
    labels[0, prompt:] = torch.randint(0, VOCAB, (SEQ - prompt, ), generator=generator)
    return {
        'hidden': torch.randn(1, SEQ, HIDDEN, generator=generator),
        'labels': labels,
        'domain': query['domain'],
    }


def routed_loss(stack, student_head, teacher_head, sample):
    """Full-vocabulary RKL of one microbatch against its routed teacher."""
    labels = sample['labels']
    selected = labels.ne(-100).reshape(-1).nonzero().reshape(-1)
    # The teacher reads detached hidden states; the student's copy carries the graph.
    teacher_hidden = sample['hidden'].detach()
    hidden = teacher_hidden.clone().requires_grad_()
    target = stack.stream.snapshot_teacher_head(
        types.SimpleNamespace(share_embeddings_and_output_weights=False, output_layer=teacher_head),
        teacher_hidden[:, selected],
    )
    loss_sum, count, _ = stack.stream.streamed_head_loss(
        hidden[:, selected],
        target,
        labels,
        student_head,
        None,
        stack.trainer.generalized_jsd_loss,
        beta=1.0,
        chunk_size=4,
        collect_diagnostics=False,
        checkpoint_chunks=False,
    )
    return loss_sum, count


def run_step(stack, student_head, teachers, queries, indices, *, num_microbatches, generator,
             mode=PUBLISHED_MODE):
    """One MOPD update: route, group, accumulate scaled microbatch gradients.

    ``global_token`` is the published recipe: ``loss_func`` returns the
    microbatch ``(loss_sum, num_tokens)`` and Megatron's per-token schedule
    divides the summed gradient by the step's global token count, so every
    response token carries equal weight regardless of which domain it came
    from. The two ``domain_*`` modes are the optional per-domain
    normalizations, which instead scale each microbatch by a factor computed
    from a step-level all-reduce.
    """
    batch = [queries[i] for i in indices]
    microbatches = stack.trainer._group_batch_by_teacher(batch, num_microbatches)

    encoded = [[rollout(query, generator) for query in rows] for rows, _ in microbatches]
    sample_counts, token_counts = defaultdict(float), defaultdict(float)
    for rows in encoded:
        for sample in rows:
            tokens = float(sample['labels'].ne(-100).sum())
            if tokens > 0:
                sample_counts[sample['domain']] += 1.0
                token_counts[sample['domain']] += tokens

    scales = {} if mode == PUBLISHED_MODE else stack.trainer._compute_domain_loss_scales(
        sample_counts, token_counts,
        stack.trainer._build_domain_base_weights(DOMAINS, None),
        mode, m_global=float(num_microbatches))

    used_teachers, reported = [teacher for _, teacher in microbatches], 0.0
    step_tokens = sum(token_counts.values())
    for (rows, teacher_idx), samples in zip(microbatches, encoded):
        for sample in samples:
            loss_sum, count = routed_loss(stack, student_head, teachers[teacher_idx], sample)
            if mode == PUBLISHED_MODE:
                scaled = loss_sum / step_tokens
            else:
                base = loss_sum if mode == 'domain_token' else loss_sum / count
                scaled = base * scales[sample['domain']] / num_microbatches
            scaled.backward()
            reported += float(scaled.detach())
    return types.SimpleNamespace(teachers=used_teachers, loss=reported, scales=dict(scales))


@pytest.fixture
def world(stack):
    generator = torch.Generator().manual_seed(2026)
    student_head = Head(generator)
    teachers = [Head(generator).requires_grad_(False) for _ in DOMAINS]
    queries = Queries(per_domain=4)
    sampler = stack.sampler(
        queries,
        total_samples=len(queries),
        consumed_samples=0,
        micro_batch_size=1,
        data_parallel_rank=0,
        data_parallel_size=1,
        data_sharding=True,
        shuffle=True,
        domain_teacher_mapping=DOMAINS,
    )
    indices = [idx for batch in sampler for idx in batch]
    return types.SimpleNamespace(
        student=student_head, teachers=teachers, queries=queries, indices=indices, generator=generator)


def test_one_step_touches_every_teacher_once_per_domain_share(stack, world):
    step = run_step(stack, world.student, world.teachers, world.queries, world.indices[:10],
                    num_microbatches=10, generator=world.generator)

    # Ten equal-share queries over five domains: each teacher owns two
    # microbatches, and microbatches are emitted grouped by teacher index.
    assert step.teachers == [index for index in sorted(DOMAINS.values()) for _ in range(2)]


def test_gradients_reach_the_student_and_never_the_frozen_teachers(stack, world):
    run_step(stack, world.student, world.teachers, world.queries, world.indices[:5],
             num_microbatches=5, generator=world.generator)

    assert world.student.weight.grad is not None
    assert torch.isfinite(world.student.weight.grad).all()
    assert world.student.weight.grad.abs().sum() > 0
    assert all(teacher.weight.grad is None for teacher in world.teachers)


def test_training_reduces_the_routed_reverse_kl(stack, world):
    optimizer = torch.optim.AdamW([world.student.weight], lr=3e-3)

    def epoch_loss():
        total = 0.0
        generator = torch.Generator().manual_seed(77)
        for index in world.indices[:10]:
            sample = rollout(world.queries[index], generator)
            teacher = world.teachers[DOMAINS[sample['domain']]]
            loss_sum, count = routed_loss(stack, world.student, teacher, sample)
            total += float(loss_sum.detach()) / float(count)
        return total / 10

    before = epoch_loss()
    for step in range(8):
        optimizer.zero_grad(set_to_none=True)
        window = world.indices[step % 2 * 10:step % 2 * 10 + 10]
        run_step(stack, world.student, world.teachers, world.queries, window,
                 num_microbatches=10, generator=world.generator)
        optimizer.step()
    after = epoch_loss()

    assert after < before, (before, after)


def test_a_student_initialized_as_its_teacher_is_already_stationary(stack, world):
    student = Head()
    teachers = [Head().requires_grad_(False) for _ in DOMAINS]
    for teacher in teachers:
        teacher.weight.data.copy_(student.weight.data)

    run_step(stack, student, teachers, world.queries, world.indices[:5],
             num_microbatches=5, generator=world.generator)

    torch.testing.assert_close(student.weight.grad, torch.zeros_like(student.weight.grad),
                               atol=1e-6, rtol=0)


def test_the_published_recipe_is_the_global_token_objective(stack, world):
    """The step gradient equals the closed-form routed RKL over all response tokens.

    Reference is computed independently of the trainer: plain softmax logits
    through an explicit reverse-KL expression, summed over every response token
    in the step and divided once by the step's token count. This is what
    ``global_token`` means, and it pins both the objective and the scale.
    """
    assert PUBLISHED_MODE == 'global_token'
    student = Head(torch.Generator().manual_seed(31))
    teachers = [Head(torch.Generator().manual_seed(40 + index)).requires_grad_(False)
                for index in range(len(DOMAINS))]
    indices = world.indices[:10]

    # The step groups the batch by teacher before sampling rollouts, so the
    # reference must consume the generator in that same grouped order.
    grouped = stack.trainer._group_batch_by_teacher(
        [world.queries[i] for i in indices], 10)
    samples = []
    generator = torch.Generator().manual_seed(808)
    for rows, _ in grouped:
        for query in rows:
            samples.append(rollout(query, generator))

    total_tokens = sum(float(s['labels'].ne(-100).sum()) for s in samples)
    reference = torch.zeros((), dtype=torch.float64)
    for sample in samples:
        rows = sample['labels'].ne(-100).reshape(-1)
        hidden = sample['hidden'].detach().reshape(-1, HIDDEN)[rows]
        teacher = teachers[DOMAINS[sample['domain']]]
        student_logits = (hidden @ student.weight.T).double()
        teacher_logits = (hidden @ teacher.weight.T).double()
        log_student = torch.log_softmax(student_logits, dim=-1)
        log_teacher = torch.log_softmax(teacher_logits, dim=-1)
        # Reverse KL: expectation under the student of log(student) - log(teacher).
        reference = reference + (log_student.exp() * (log_student - log_teacher)).sum()
    reference = reference / total_tokens
    reference.backward()
    expected_grad = student.weight.grad.clone()

    student.weight.grad = None
    step = run_step(stack, student, teachers, world.queries, indices,
                    num_microbatches=10, generator=torch.Generator().manual_seed(808))

    assert step.scales == {}
    assert step.loss == pytest.approx(float(reference.detach()), rel=1e-5)
    torch.testing.assert_close(student.weight.grad, expected_grad, atol=1e-5, rtol=1e-4)


def test_both_domain_normalization_modes_run_and_scale_differently(stack, world):
    token = run_step(stack, Head(), world.teachers, world.queries, world.indices[:10],
                     num_microbatches=10, generator=torch.Generator().manual_seed(5),
                     mode='domain_token')
    sequence = run_step(stack, Head(), world.teachers, world.queries, world.indices[:10],
                        num_microbatches=10, generator=torch.Generator().manual_seed(5),
                        mode='domain_sequence')

    assert set(token.scales) == set(sequence.scales) == set(DOMAINS)
    # Token mode divides by response tokens, sequence mode by sequences.
    assert all(token.scales[d] < sequence.scales[d] for d in DOMAINS)


def test_an_unroutable_query_stops_the_step(stack, world):
    with pytest.raises(ValueError, match='not in domain_teacher_mapping'):
        stack.trainer._group_batch_by_teacher([{'domain': 'audio'}], num_microbatches=1)
