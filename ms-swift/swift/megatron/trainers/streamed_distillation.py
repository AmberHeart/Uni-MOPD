# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, MOPD contributors.
"""Streamed teacher heads and checkpointed full-vocabulary distillation.

Keep the student's native whole-response GEMM in forward/backward. Changing
its token dimension changes BF16 dgrad rounding. Checkpoint teacher projection
and criterion per chunk; retain one BF16 student-logit buffer, not full teacher
logits or the full-response FP32 loss intermediates.
"""
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint


@dataclass
class TeacherHeadTarget:
    hidden_states: torch.Tensor  # [1, selected_tokens, hidden]
    head: torch.nn.Module
    weight: torch.Tensor


def head_weight(model):
    return (model.shared_embedding_or_output_weight()
            if model.share_embeddings_and_output_weights else model.output_layer.weight)


def snapshot_teacher_head(model, hidden_states):
    weight = head_weight(model)
    if weight is None or weight.requires_grad or hidden_states.requires_grad:
        raise ValueError('Streamed teacher targets require a frozen native head and detached hidden states.')
    if getattr(model.output_layer, 'bias', None) is not None:
        raise ValueError('The streamed causal LM-head path requires a bias-free native head.')
    # Teacher models are not DDP-wrapped. Their offloader replaces parameter.data
    # with CPU storage; a detached tensor keeps THIS forward's GPU head storage
    # alive without retaining the rest of that teacher or cloning [V,H].
    return TeacherHeadTarget(hidden_states.detach(), model.output_layer, weight.detach())


@contextmanager
def selected_head_context(head):
    original = head.sequence_parallel
    head.sequence_parallel = False
    try:
        yield
    finally:
        head.sequence_parallel = original


def project_selected_hidden(head, hidden, weight=None):
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError('Streamed heads expect selected [1, tokens, hidden] states.')
    if getattr(head, 'bias', None) is not None:
        raise ValueError('Streamed heads require a bias-free native output layer.')
    # GPTModel already gathered SP and selected the rows. Native non-SP
    # ColumnParallelLinear retains its TP dgrad reduction; its explicit
    # gather's backward must therefore remain scatter-only.
    with selected_head_context(head):
        logits, _ = head(hidden.transpose(0, 1).contiguous(), weight=weight, runtime_gather_output=False)
    return logits.transpose(0, 1).contiguous()


@torch.no_grad()
def merge_jsd_statistics(chunks):
    chunks = [chunk for chunk in chunks if chunk]
    if not chunks:
        return {}
    result = {}
    for key in chunks[0]:
        if key in ('kl_values', 'reward_values'):
            result[key] = torch.cat([chunk[key] for chunk in chunks])
        else:
            value = chunks[0][key].clone()
            for chunk in chunks[1:]:
                value = value + chunk[key]
            result[key] = value
    return result


@torch.no_grad()
def finalize_jsd_statistics(state, loss_sum, count, *, teacher_is_log_probs=False):
    if not state:
        return {}
    s = state
    result = {
        'kl_teacher': s['kl_teacher_sum'] / count,
        'kl_student': s['kl_student_sum'] / count,
        'kl_forward': s['kl_forward_sum'] / count,
        'kl_reverse': s['kl_reverse_sum'] / count,
        'jsd_std': (s['jsd_sq_sum'] / count - (loss_sum / count).detach() ** 2).clamp(min=0).sqrt(),
        'kl_at_agree': s['kl_agree_sum'] / s['n_agree'].clamp(min=1),
        'kl_at_disagree': s['kl_disagree_sum'] / s['n_disagree'].clamp(min=1),
        'top1_agreement': s['agreement_sum'] / s['n_total'].clamp(min=1),
        'teacher_maxprob_at_disagree': s['teacher_maxprob_disagree_sum'] / s['n_disagree'].clamp(min=1),
        'student_maxprob_at_disagree': s['student_maxprob_disagree_sum'] / s['n_disagree'].clamp(min=1),
        'student_entropy': s['s_entropy_sum'] / count,
        'teacher_entropy': s['t_entropy_sum'] / count,
        'student_logit_norm': (s['s_logit_sq_sum'] / count).sqrt(),
    }
    if not teacher_is_log_probs:
        result['teacher_logit_norm'] = (s['t_logit_sq_sum'] / count).sqrt()
    forward = result['kl_forward']
    result['kl_asymmetry'] = torch.where(
        forward > 1e-8, result['kl_reverse'] / forward, forward.new_ones(()))
    result['kl_disagree_ratio'] = result['kl_at_disagree'] / result['kl_at_agree'].clamp(min=1e-6)
    entropy = result['teacher_entropy']
    result['entropy_ratio'] = torch.where(
        entropy > 1e-8, result['student_entropy'] / entropy, entropy.new_ones(()))
    # Raw sufficient statistics for the post-aggregation rebuild of the stratified
    # sum/count diagnostics in MegatronGKDTrainer.train_step (see _SUM_RATIOS). The
    # ratios above are per-microbatch and get mean-averaged by Megatron's scalar branch,
    # so the trainer re-forms them from these sums/counts and pops these keys -- they are
    # never logged. Mirrors the non-streamed path in generalized_jsd_loss.
    for _raw in ('kl_agree_sum', 'kl_disagree_sum', 'n_agree', 'n_disagree', 'n_total',
                 'agreement_sum', 'teacher_maxprob_disagree_sum',
                 'student_maxprob_disagree_sum',
                 'jsd_sq_sum', 's_logit_sq_sum', 't_logit_sq_sum'):
        value = s.get(_raw)
        if value is not None:
            result[f'_mopd_raw/{_raw}'] = value
    # Pooled RMS/std recombination inputs (see _POOLED_NORMS / _POOLED_STDS). ``count`` is
    # the valid-token count for this microbatch and ``loss_sum`` its accumulated JSD sum.
    result['_mopd_raw/nv'] = count
    result['_mopd_raw/jsd_sum'] = loss_sum.detach()
    values = s['kl_values']
    if values.numel():
        for name, percentile in (('kl_p50', .5), ('kl_p90', .9), ('kl_p99', .99)):
            result[name] = values.quantile(percentile)
        result['kl_max'] = values.max()
    result.update({key: value for key, value in s.items() if key.startswith('mopd/')})
    if 'reward_values' in s and s['reward_values'].numel():
        for name, percentile in (('p10', .1), ('p50', .5), ('p90', .9), ('p99', .99)):
            result[f'mopd/reward_{name}'] = s['reward_values'].quantile(percentile)
    return result


def streamed_head_loss(
    student_hidden, teacher, labels, student_head, student_weight, criterion, *,
    beta, chunk_size, collect_diagnostics, checkpoint_chunks=True,
):
    """Return one graph-connected (loss sum, token count, diagnostics) tuple.

    ``criterion`` is the trainer's existing generalized_jsd_loss; this function
    never implements a second KL formula. It receives already-selected hidden
    rows; ``labels`` retain the original full-sequence geometry.
    """
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError('Streamed head chunk size must be a positive integer.')
    selected_labels = labels[labels != -100].reshape(1, -1)
    valid_count = selected_labels.numel()
    rows = max(valid_count, 1)
    if (student_hidden.ndim != 3 or student_hidden.shape[:2] != (1, rows)
            or teacher.hidden_states.shape[:2] != (1, rows)):
        raise ValueError('Selected student/teacher hidden rows do not match the supervised labels.')
    if student_hidden.dtype != teacher.hidden_states.dtype or teacher.hidden_states.dtype != teacher.weight.dtype:
        raise ValueError('Streamed hidden states and teacher weights must retain their native dtype.')
    if not valid_count:
        selected_labels = labels.new_full((1, 1), -100)

    statistics = []
    total_loss = None
    # Native full-response student projection/backward must keep the same GEMM
    # shape as the response-only baseline: chunking this head leaves logits and
    # loss exact but changes the token dimension of the backward GEMM, and BF16
    # dgrad rounding depends on that shape. Teacher projection is frozen and can
    # still be streamed.
    student_logits = project_selected_hidden(student_head, student_hidden, student_weight)

    def make_chunk():
        def compute(s_logits, teacher_chunk, weight, chunk_labels):
            with torch.no_grad():
                t_logits = project_selected_hidden(teacher.head, teacher_chunk, weight)
            raw = {} if collect_diagnostics else None
            loss_sum, _ = criterion(
                s_logits, t_logits, chunk_labels, beta=beta,
                student_logits_pre_masked=True, teacher_logits_pre_masked=True,
                return_sum_count=True, raw_diagnostics=raw)
            return loss_sum, raw or {}
        return compute

    for start in range(0, rows, chunk_size):
        end = min(start + chunk_size, rows)
        inputs = (student_logits[:, start:end], teacher.hidden_states[:, start:end],
                  teacher.weight, selected_labels[:, start:end])
        compute = make_chunk()
        if checkpoint_chunks and torch.is_grad_enabled():
            loss_sum, raw = checkpoint(compute, *inputs, use_reentrant=False)
        else:
            loss_sum, raw = compute(*inputs)
        total_loss = loss_sum if total_loss is None else total_loss + loss_sum
        statistics.append(raw)

    count = student_hidden.new_tensor(float(valid_count), dtype=torch.float32)
    diagnostics = finalize_jsd_statistics(merge_jsd_statistics(statistics), total_loss, count)
    return total_loss, count, diagnostics
