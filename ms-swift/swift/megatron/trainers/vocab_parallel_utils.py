# Copyright (c) Alibaba, Inc. and its affiliates.
"""Vocabulary-parallel utilities for Tensor Parallelism.

This module provides utilities for computing log_softmax, entropy, KL divergence,
and other operations across vocab-parallel sharded tensors in Tensor Parallelism (TP).

When using TP, the vocabulary dimension is sharded across TP ranks. These utilities
correctly handle the distributed computation by:
1. Finding global max via all_reduce (for numerical stability)
2. Computing sum of exp via all_reduce (for normalization)
3. All-reducing partial sums for final results
"""

import math
from typing import Optional, Tuple

import torch
from megatron.core import mpu
from torch.autograd.function import once_differentiable


class _VocabParallelLogSoftmax(torch.autograd.Function):
    """Vocab-parallel log_softmax with a cross-shard-correct backward.

    The stable forward is the standard two-collective idiom: global max
    (MAX-reduce, detached — a pure numerical shift), local exp, global exp-sum
    (SUM-reduce, detached — its autograd kernel would replicate the local
    exp-gradient instead of reducing the full-vocab normalization term), then
    ``z - max - log(sum)``.

    The true ``d log_softmax(z)_i / d z_j = δ_ij - softmax(z)_j``, where the
    softmax denominator spans the FULL vocabulary (all TP shards). A plain
    autograd pass through the collectives cannot express that, so the backward
    computes the correction explicitly:
    ``grad_i = grad_out_i - softmax_i * sum_j grad_out_j``,
    where ``sum_j grad_out_j`` is all-reduced across the TP group (the real
    cross-shard reduction the plain kernels were missing). ``softmax`` here is
    the full-vocab softmax, computed from the saved logits and the detached
    global max/sum.
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, tp_group, tp_size: int) -> torch.Tensor:
        logits_max = logits.max(dim=-1, keepdim=True)[0]
        torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
        logits_max = logits_max.detach()

        exp_logits = torch.exp(logits - logits_max)
        sum_exp = exp_logits.sum(dim=-1, keepdim=True)
        torch.distributed.all_reduce(sum_exp, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        sum_exp = sum_exp.detach()

        log_softmax = logits - logits_max - torch.log(sum_exp)
        # Save the LOCAL softmax slice (== full softmax on this shard, since the
        # max/sum are global) for the backward correction.
        ctx.save_for_backward(exp_logits / sum_exp)
        ctx.tp_group = tp_group
        ctx.tp_size = tp_size
        return log_softmax

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (softmax_local,) = ctx.saved_tensors
        # Per-position sum of incoming gradients over the FULL vocab.
        grad_sum = grad_output.sum(dim=-1, keepdim=True)
        if ctx.tp_size > 1:
            torch.distributed.all_reduce(
                grad_sum, op=torch.distributed.ReduceOp.SUM, group=ctx.tp_group)
        grad_input = grad_output - grad_sum * softmax_local
        return grad_input, None, None


def vocab_parallel_log_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Compute log_softmax across vocab-parallel sharded logits.

    When using Tensor Parallelism, vocab is sharded across TP ranks.
    This function correctly computes log_softmax by:
    1. Finding global max via all_reduce
    2. Computing sum of exp via all_reduce
    3. Computing log_softmax using the global statistics

    Args:
        logits: Logits tensor [..., partition_vocab_size]

    Returns:
        log_softmax tensor [..., partition_vocab_size]
    """
    tp_size = mpu.get_tensor_model_parallel_world_size()

    if tp_size == 1:
        return torch.nn.functional.log_softmax(logits, dim=-1)

    tp_group = mpu.get_tensor_model_parallel_group()

    return _VocabParallelLogSoftmax.apply(logits, tp_group, tp_size)


def vocab_parallel_entropy(log_probs: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
    """Compute entropy from pre-computed vocab-parallel sharded log probabilities.

    When using Tensor Parallelism, vocab is sharded across TP ranks.
    This function correctly computes entropy by:
    1. Computing partial entropy = -sum(exp(log_p) * log_p) on each rank's partition
    2. All-reducing the partial entropies to get the global sum.

    Entropy is computed in chunks to reduce memory usage.

    Args:
        log_probs: Pre-computed log probabilities tensor [..., partition_vocab_size]
        chunk_size: Number of tokens to process per chunk (default: 512)

    Returns:
        Entropy tensor [...] (scalar per position)
    """
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_size = mpu.get_tensor_model_parallel_world_size()

    # Flatten all but the last dimension for chunked processing
    original_shape = log_probs.shape[:-1]
    vocab_size = log_probs.shape[-1]
    log_probs_flat = log_probs.view(-1, vocab_size)  # [total_tokens, partition_vocab_size]
    total_tokens = log_probs_flat.shape[0]

    entropies_list = []
    for start_idx in range(0, total_tokens, chunk_size):
        end_idx = min(start_idx + chunk_size, total_tokens)
        log_probs_chunk = log_probs_flat[start_idx:end_idx]  # [chunk_size, partition_vocab_size]

        # Compute partial entropy on this rank's vocab partition
        # entropy = -sum(p * log_p) = -sum(exp(log_p) * log_p)
        probs = torch.exp(log_probs_chunk)
        partial_entropy = -(probs * log_probs_chunk).sum(dim=-1)  # [chunk_size]

        # All-reduce to get global entropy if using TP
        if tp_size > 1:
            torch.distributed.all_reduce(partial_entropy, op=torch.distributed.ReduceOp.SUM, group=tp_group)

        entropies_list.append(partial_entropy)

    # Concatenate all chunks and reshape back
    entropies = torch.cat(entropies_list, dim=0)
    entropies = entropies.view(original_shape)

    return entropies


def vocab_parallel_top1_indices(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Global argmax token ids for student and teacher under vocab-parallel sharding.

    Each TP rank holds only ``partition_vocab_size`` of the vocabulary, so a per-rank
    ``argmax`` returns the best id *within that shard*, which is not the global argmax
    unless this rank happens to own it. Comparing those per-shard winners answers a
    different question than "do student and teacher agree on the top token", and the
    error does NOT wash out: measured over 2-way sharding, the shard-local rate is
    biased UPWARD (true 0.1338 -> 0.1582 for distinct distributions), because a shard
    offers fewer competing ids so chance agreement is easier. It is not a collapse
    toward ``1/tp_size`` -- for identical distributions the shard-local rate is a
    correct 1.0. This resolves the true global argmax by all-gathering the per-shard
    (max value, global id) pairs and picking the owning rank per token.

    Args:
        student_logits: Student logits/log-probs [num_tokens, partition_vocab_size]
        teacher_logits: Teacher logits/log-probs [num_tokens, partition_vocab_size]

    Returns:
        ``(student_top1, teacher_top1)``, each [num_tokens] of GLOBAL vocab ids.
        Monotone transforms (e.g. log_softmax) do not change the result, so raw logits
        and log-probs are interchangeable inputs.
    """
    tp_size = mpu.get_tensor_model_parallel_world_size()

    if tp_size == 1:
        return student_logits.argmax(dim=-1), teacher_logits.argmax(dim=-1)

    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_group = mpu.get_tensor_model_parallel_group()
    partition_size = student_logits.shape[-1]

    s_max_val, s_local_idx = student_logits.max(dim=-1)
    t_max_val, t_local_idx = teacher_logits.max(dim=-1)

    s_global_idx = s_local_idx + tp_rank * partition_size
    t_global_idx = t_local_idx + tp_rank * partition_size

    n = s_max_val.numel()
    packed_vals = torch.cat([s_max_val.contiguous(), t_max_val.contiguous()])
    packed_idxs = torch.cat([s_global_idx.contiguous(), t_global_idx.contiguous()])

    all_vals = torch.empty(tp_size * 2 * n, device=packed_vals.device, dtype=packed_vals.dtype)
    all_idxs = torch.empty(tp_size * 2 * n, device=packed_idxs.device, dtype=packed_idxs.dtype)

    torch.distributed.all_gather_into_tensor(all_vals, packed_vals, group=tp_group)
    torch.distributed.all_gather_into_tensor(all_idxs, packed_idxs, group=tp_group)

    all_vals = all_vals.view(tp_size, 2, n)
    all_idxs = all_idxs.view(tp_size, 2, n)

    s_best = all_vals[:, 0, :].argmax(dim=0)
    t_best = all_vals[:, 1, :].argmax(dim=0)

    s_top1 = all_idxs[:, 0, :].gather(0, s_best.unsqueeze(0)).squeeze(0)
    t_top1 = all_idxs[:, 1, :].gather(0, t_best.unsqueeze(0)).squeeze(0)

    return s_top1, t_top1


def vocab_parallel_top1_agreement(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    """Fraction of positions where the GLOBAL argmax of student and teacher agree.

    Thin wrapper over :func:`vocab_parallel_top1_indices`; see that function for why the
    global resolution is required under TP.

    Returns:
        Scalar tensor: agreement rate in [0, 1]
    """
    s_top1, t_top1 = vocab_parallel_top1_indices(student_logits, teacher_logits)
    return (s_top1 == t_top1).float().mean()


def vocab_parallel_kl_div(input_log_probs: torch.Tensor, target_log_probs: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence for vocab-parallel sharded log probabilities.

    KL(target || input) = sum(target_prob * (target_log_prob - input_log_prob))
                        = sum(exp(target_log_prob) * (target_log_prob - input_log_prob))

    Since both log_probs are sharded across TP, we compute the partial sum
    on each rank and then all_reduce to get the global sum.

    Args:
        input_log_probs: Input log probabilities [..., partition_vocab_size]
        target_log_probs: Target log probabilities [..., partition_vocab_size]

    Returns:
        KL divergence per position [...], already reduced across TP
    """
    tp_group = mpu.get_tensor_model_parallel_group()

    # Compute partial KL on this rank's vocab partition
    target_probs = torch.exp(target_log_probs)
    partial_kl = (target_probs * (target_log_probs - input_log_probs)).sum(dim=-1)

    if mpu.get_tensor_model_parallel_world_size() > 1:
        tp_group = mpu.get_tensor_model_parallel_group()
        torch.distributed.all_reduce(partial_kl, op=torch.distributed.ReduceOp.SUM, group=tp_group)

    return partial_kl


class _VocabParallelReverseKL(torch.autograd.Function):
    """First-order logit VJP without the cancelling ``+1`` log-probability term."""

    @staticmethod
    def forward(ctx, student_logits, teacher_log_probs, tp_group, tp_size):
        # Custom Function.forward runs without recording the intermediate
        # log_softmax graph; its generic VJP is deliberately not used here.
        if tp_size == 1:
            student_log_probs = torch.nn.functional.log_softmax(student_logits, dim=-1)
        else:
            student_log_probs = _VocabParallelLogSoftmax.apply(student_logits, tp_group, tp_size)
        delta = student_log_probs - teacher_log_probs
        loss = (student_log_probs.exp() * delta).sum(dim=-1)
        if tp_size > 1:
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        # Recompute vocabulary-sized probabilities/gaps in backward. The small
        # loss copy also permits callers to scale the returned loss in-place.
        ctx.save_for_backward(student_log_probs, teacher_log_probs, loss.clone())
        ctx.mark_non_differentiable(student_log_probs)
        ctx.set_materialize_grads(False)
        return loss, student_log_probs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss, _grad_log_probs):
        if grad_loss is None:
            return None, None, None, None
        student_log_probs, teacher_log_probs, loss = ctx.saved_tensors
        probability = student_log_probs.exp()
        scale = grad_loss.unsqueeze(-1)
        grad_student = None
        if ctx.needs_input_grad[0]:
            delta = student_log_probs - teacher_log_probs
            # KL and log_softmax's separate VJPs otherwise cancel two rounded
            # probabilities: p*(delta+1) - r*sum(p*(delta+1)).
            # Centering first gives exact zeros when the log-probabilities match.
            grad_student = scale * probability * (delta - loss.unsqueeze(-1))
        grad_teacher = -scale * probability if ctx.needs_input_grad[1] else None
        # The loss is replicated on TP ranks. Do not all-reduce grad_loss or
        # multiply by TP size: the full-vocabulary center was reduced in forward.
        return grad_student, grad_teacher, None, None


def vocab_parallel_reverse_kl_div(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return full-vocab KL(student || teacher) and detached student log-probs.

    Inputs have identical LOCAL vocabulary slices. The teacher log-probabilities
    are already GLOBALLY normalized; the student logits are normalized here.
    Mask selection and temperature scaling belong to the caller. Floating-point
    precision is preserved (GKD supplies FP32, reference tests may use FP64).

    The per-token loss is replicated across TP ranks and requires the same
    upstream loss weights on each TP rank. Its student-logit backward is
    ``p * (log_p - log_q - KL)``, so equal log-probabilities give exactly zero
    without checking/clipping small gradients or skipping an optimizer update.
    The second return value is diagnostic-only and not differentiable.
    Teacher log-probability gradients are retained if explicitly requested;
    normal GKD teachers remain frozen. Higher-order derivatives are unsupported.
    """
    if student_logits.shape != teacher_log_probs.shape:
        raise ValueError('Reverse KL needs matching student/teacher vocabulary slices.')
    return _VocabParallelReverseKL.apply(
        student_logits, teacher_log_probs,
        mpu.get_tensor_model_parallel_group(), mpu.get_tensor_model_parallel_world_size())


class _VocabParallelToDi(torch.autograd.Function):
    """ToDi with detached weights and a replicated-loss, identity-backward sum.

    Recompute weights in backward instead of retaining another set of full-vocab
    weights/products for every response chunk until the model backward starts.
    Inputs are LOCAL slices of GLOBALLY normalized log probabilities.
    """

    @staticmethod
    def forward(ctx, student_log_probs, teacher_log_probs, beta, tp_group, tp_size):
        delta = teacher_log_probs - student_log_probs
        scaled_delta = beta * delta
        forward_mass = torch.sigmoid(scaled_delta) * teacher_log_probs.exp()
        # sigmoid(-x) avoids the cancellation in 1 - sigmoid(x).
        reverse_mass = torch.sigmoid(-scaled_delta) * student_log_probs.exp()
        partial = ((forward_mass - reverse_mass) * delta).sum(dim=-1)
        if tp_size > 1:
            torch.distributed.all_reduce(partial, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        ctx.save_for_backward(student_log_probs, teacher_log_probs)
        ctx.beta = beta
        return partial

    @staticmethod
    def backward(ctx, grad_output):
        student_log_probs, teacher_log_probs = ctx.saved_tensors
        delta = teacher_log_probs - student_log_probs
        scaled_delta = ctx.beta * delta
        forward_mass = torch.sigmoid(scaled_delta).detach() * teacher_log_probs.exp()
        reverse_mass = torch.sigmoid(-scaled_delta).detach() * student_log_probs.exp()
        # dL/dlog(S) = -alpha*T + (1-alpha)*S*(1 + log(S)-log(T)).
        # Do NOT sum grad_output across TP: each rank owns the same scalar loss.
        # vocab_parallel_log_softmax supplies the cross-shard normalization VJP.
        grad_student = reverse_mass * (1 - delta) - forward_mass
        return grad_output.unsqueeze(-1) * grad_student, None, None, None, None


def vocab_parallel_todi_div(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Token-wise FKL/RKL weighting from ToDi (arXiv:2505.16297v2, Eq. 11).

    The teacher and sigmoid weights are stopped; student probability and
    log-probability gradients remain live. At beta=1 the *value* equals FKL+RKL,
    but the stopped-weight gradient does not. No extra 1/2 or temperature factor
    is applied. The caller retains response masking and sequence/token reduction.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('ToDi beta must be finite and strictly positive.')
    return _VocabParallelToDi.apply(
        student_log_probs, teacher_log_probs.detach(), beta,
        mpu.get_tensor_model_parallel_group(), mpu.get_tensor_model_parallel_world_size())


def vocab_parallel_gather_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    log_probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather log probabilities for target labels from vocab-parallel logits.

    When using TP, each rank only has a partition of the vocabulary. This function:
    1. Computes log_softmax with vocab-parallel support (if log_probs not provided)
    2. Gathers log probs for target tokens
    3. All-reduces to combine results from all TP ranks

    Args:
        logits: Logits tensor [batch, seq, partition_vocab_size]
        labels: Token labels [batch, seq], -100 for masked positions
        log_probs: Pre-computed log_softmax (optional, to avoid recomputation)

    Returns:
        per_token_logps: [batch, seq] log probabilities for target tokens
    """
    tp_size = mpu.get_tensor_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()

    # Compute log_probs if not provided
    if log_probs is None:
        log_probs = vocab_parallel_log_softmax(logits)

    # Get the local vocab range for this TP rank
    partition_vocab_size = log_probs.shape[-1]
    if tp_size > 1:
        vocab_start = tp_rank * partition_vocab_size
        vocab_end = vocab_start + partition_vocab_size
        # Check which labels fall within this TP rank's vocab partition
        local_labels = labels - vocab_start
        # Mask for labels within local vocab range
        in_range_mask = (labels >= vocab_start) & (labels < vocab_end)
        # Clamp local_labels to valid range for gather
        local_labels = local_labels.clamp(min=0, max=partition_vocab_size - 1)
    else:
        local_labels = labels
        in_range_mask = torch.ones_like(labels, dtype=torch.bool)
        local_labels = local_labels.clamp(min=0)

    # Gather log probs for target tokens
    gathered_logps = torch.gather(log_probs, dim=-1, index=local_labels.unsqueeze(-1)).squeeze(-1)

    # For TP: only the rank that owns the target token has the correct log prob
    # Other ranks have incorrect values, so we need to zero them out
    if tp_size > 1:
        gathered_logps = gathered_logps * in_range_mask.float()
        # All-reduce to sum contributions from all ranks
        # (only one rank has non-zero value for each token)
        torch.distributed.all_reduce(
            gathered_logps, op=torch.distributed.ReduceOp.SUM, group=mpu.get_tensor_model_parallel_group())

    # Apply loss mask (labels == -100 are masked)
    loss_mask = labels != -100
    per_token_logps = gathered_logps * loss_mask.float()

    return per_token_logps


def compute_logps_and_entropy_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    compute_entropy: bool = False,
    entropy_chunk_size: int = 512,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute per-token log probabilities and optionally entropy from logits.

    This is a unified method that efficiently computes both logps and entropy
    in a single pass when both are needed, sharing the log_softmax computation.

    Note: In Megatron, labels are already shifted (via torch.roll in get_batch_on_this_tp_rank),
    so logits and labels are already aligned. No additional shift is needed here.

    Args:
        logits: Logits tensor [batch, seq, partition_vocab_size] or [1, total_tokens, partition_vocab_size]
        labels: Token labels [batch, seq] or [1, total_tokens], -100 for masked positions
        compute_entropy: Whether to compute entropy (default: False)
        entropy_chunk_size: Chunk size for entropy computation (default: 512)

    Returns:
        Tuple of:
            - per_token_logps: [batch, seq] or [1, total_tokens] log probabilities for target tokens
            - per_token_entropy: Same shape as per_token_logps, or None if compute_entropy=False
    """
    # Compute log_softmax (shared for both logps and entropy)
    log_probs = vocab_parallel_log_softmax(logits)

    # Gather logps for target tokens
    per_token_logps = vocab_parallel_gather_logps(logits, labels, log_probs=log_probs)

    # Compute entropy if requested (reuse log_probs to avoid redundant computation)
    per_token_entropy = None
    if compute_entropy:
        per_token_entropy = vocab_parallel_entropy(log_probs, chunk_size=entropy_chunk_size)

    return per_token_logps, per_token_entropy
