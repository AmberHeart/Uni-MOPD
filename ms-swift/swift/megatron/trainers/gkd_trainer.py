# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime
import hashlib
import math
import os
import queue
import random
import re
import threading
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from enum import Enum
from functools import partial
from typing import Dict, List, Optional

import json
import torch
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.training import get_args, get_model, get_timers, get_wandb_writer
from megatron.training.utils import unwrap_model
from megatron.training.utils import is_last_rank as megatron_is_last_rank
from transformers import AutoConfig, AutoTokenizer

from swift.llm import Template, get_model_info_meta, to_device
from swift.utils import get_logger
from ..argument import MegatronArguments
from ..model import get_megatron_model_meta
from ..utils import convert_hf_config, forward_step_helper, get_padding_to
from .rlhf_mixin import MegatronRLHFTrainer
from .rollout_mixin import MegatronRolloutMixin
from .utils import (MegatronPretrainingRandomSampler, get_swift_datasets_provider, load_megatron_model_to_gpu,
                    offload_megatron_model_to_cpu)
from .vocab_parallel_utils import (vocab_parallel_entropy, vocab_parallel_kl_div, vocab_parallel_log_softmax,
                                   vocab_parallel_reverse_kl_div, vocab_parallel_todi_div,
                                   vocab_parallel_top1_indices)

logger = get_logger()


def _weight_lag(rollout_weight_step: int, train_step: int) -> int:
    if rollout_weight_step is None or rollout_weight_step < 0:
        return -1
    return int(train_step) - int(rollout_weight_step)


def _validate_weight_lag(lag: int, *, overlap_enabled: bool) -> None:
    if lag < 0:
        return
    limit = 1 if overlap_enabled else 0
    if lag > limit:
        raise RuntimeError(
            f'Rollout weight lag {lag} exceeds the on-policy limit {limit}; '
            'stop before consuming stale completions.')


def _env_or_arg(env_name: str, arg_value, cast):
    """Return ``cast(os.environ[env_name])`` if the env var is set non-empty, else ``arg_value``.

    An environment override takes precedence over the argument value. A malformed
    value falls back to ``arg_value``.
    """
    raw = os.environ.get(env_name)
    if raw is None or raw.strip() == '':
        return arg_value
    try:
        return cast(raw.strip())
    except (TypeError, ValueError):
        logger.warning(f'Ignoring malformed env override {env_name}={raw!r}; using {arg_value!r}.')
        return arg_value




# Divergence ratios that MUST be formed after the microbatch aggregation, never before.
# ``(ratio_key, numerator_key, denominator_key, denominator_floor)``.
#
# Ratios are not averageable across microbatches. Compute them from the
# already-aggregated numerator and denominator so every logged row is
# self-consistent.
#
# The components are already averaged correctly by the time ``train_step`` sees them, so
# re-deriving the ratios here costs NO collective (metric paths keep `skip_metric_dp_allreduce`
# and no key is added/removed, so rank symmetry is untouched). SCOPE -- what this does and
# does not buy: the result is a ratio of microbatch means. It is not the
# token-weighted global ratio (that would
# need the per-class SUMS and COUNTS, which the metric dict does not carry; the
# cross-step accumulation in Megatron's training loop would also need those).
# Self-check any ratio row with ``logged_ratio == num/den``.
#
# Do NOT "fix" this by returning a 2-element [num, den] tensor from ``loss_func``: that hits
# training.py's numel()==2 branch, which adds a per-key DP all-reduce into the metric path --
# the very collective class that deadlocks against MoE EP/TP all-to-all.
_DIVERGENCE_RATIOS = (
    ('divergence/kl_asymmetry', 'divergence/kl_reverse', 'divergence/kl_forward', 1e-8),
    ('divergence/kl_disagree_ratio', 'divergence/kl_at_disagree', 'divergence/kl_at_agree', 1e-6),
    ('divergence/entropy_ratio', 'divergence/student_entropy', 'divergence/teacher_entropy', 1e-8),
)


def recompute_divergence_ratios(loss_dict: Dict[str, torch.Tensor]) -> None:
    """Re-derive the ratio diagnostics in ``loss_dict`` from its aggregated components.

    In-place, on scalars the native path has already reduced. Missing keys are skipped
    (diagnostics are collected on an interval, so a non-collecting step carries none).
    """
    for ratio_key, num_key, den_key, floor in _DIVERGENCE_RATIOS:
        num, den = loss_dict.get(num_key), loss_dict.get(den_key)
        if num is None or den is None or ratio_key not in loss_dict:
            continue
        denominator = torch.as_tensor(den, dtype=torch.float32)
        numerator = torch.as_tensor(num, dtype=torch.float32).to(denominator.device)
        # Same near-zero guard each ratio used pre-aggregation, applied once post-aggregation.
        loss_dict[ratio_key] = torch.where(denominator > floor,
                                           numerator / denominator.clamp(min=floor),
                                           torch.ones_like(denominator))


# Stratified components that ``loss_func`` produced as a per-microbatch ``sum/count`` and
# that the native path then averaged -- so the logged value was a mean-of-ratios. Each
# entry is ``(output_key, numerator_sum_key, denominator_count_key)``.
#
# These cannot be re-formed from the ratio alone, so ``loss_func`` also emits the raw
# statistics under ``_mopd_raw/`` and this helper divides them once, after aggregation.
# The reconstruction is exact for the pooled statistics because the per-rank microbatch
# count ``M`` is the same divisor for every key: ``mean_i(sum_i) / mean_i(count_i)``
# equals ``Σ_i sum_i / Σ_i count_i``. That identity is what lets a MEAN-only aggregation
# (the native ``numel()==1`` branch, which adds no collective) still yield
# ``sum/count``. The raw keys are popped here so they never reach the log.
#
# Scope, as with ``_DIVERGENCE_RATIOS``: this is a rank-local reconstruction, not a
# DP-global one (``skip_metric_dp_allreduce`` stays on). It removes the mean-of-ratios
# error; it does not turn these into cross-DP token-weighted values.
_SUM_RATIOS = (
    ('divergence/kl_at_agree', '_mopd_raw/kl_agree_sum', '_mopd_raw/n_agree'),
    ('divergence/kl_at_disagree', '_mopd_raw/kl_disagree_sum', '_mopd_raw/n_disagree'),
    ('divergence/top1_agreement', '_mopd_raw/agreement_sum', '_mopd_raw/n_total'),
    ('divergence/teacher_maxprob_at_disagree', '_mopd_raw/teacher_maxprob_disagree_sum',
     '_mopd_raw/n_disagree'),
    ('divergence/student_maxprob_at_disagree', '_mopd_raw/student_maxprob_disagree_sum',
     '_mopd_raw/n_disagree'),
)

# Namespace for sufficient statistics that exist only to feed the recompute helpers.
_RAW_STAT_PREFIX = '_mopd_raw/'

# Pooled nonlinear diagnostics: a per-microbatch ``sqrt`` aggregated by a plain mean is a
# mean-of-nonlinear-function, not the pooled value. Recombining needs the SUM OF SQUARES
# (and, for a std, the SUM), which the metric dict does not carry -- so loss_func exports
# them under ``_mopd_raw/`` and these entries recombine once after aggregation. Same
# MEAN-only identity as _SUM_RATIOS: ``mean(sq)/mean(nv) == Σsq/Σnv``, so a dict averaged
# by the native path still yields the pooled statistic.
#
# ``efficiency/student_logit_norm`` / ``efficiency/teacher_logit_norm`` are RMS of the
# logits: pooled = sqrt(Σsq/Σn).
_POOLED_NORMS = (
    ('efficiency/student_logit_norm', '_mopd_raw/s_logit_sq_sum'),
    ('efficiency/teacher_logit_norm', '_mopd_raw/t_logit_sq_sum'),
)
# ``loss/jsd_per_token_std`` is the pooled standard deviation of per-token JSD:
# sqrt(Σx²/Σn − (Σx/Σn)²).
_POOLED_STDS = (
    ('loss/jsd_per_token_std', '_mopd_raw/jsd_sq_sum', '_mopd_raw/jsd_sum'),
)
_NV_RAW_KEY = '_mopd_raw/nv'


def recompute_pooled_stats(loss_dict: Dict[str, torch.Tensor]) -> None:
    """Recombine the pooled RMS/std diagnostics from their aggregated sufficient statistics.

    In-place. Skips when the shared token count is absent or zero (a step with no valid
    tokens has no defined pooled statistic).
    """
    nv = loss_dict.get(_NV_RAW_KEY)
    if nv is None:
        return
    denominator = torch.as_tensor(nv, dtype=torch.float32)
    if float(denominator) <= 0:
        return
    denominator = denominator.clamp(min=1.0)
    for out_key, sq_key in _POOLED_NORMS:
        sq = loss_dict.get(sq_key)
        if sq is None or out_key not in loss_dict:
            continue
        mean_sq = torch.as_tensor(sq, dtype=torch.float32).to(denominator.device) / denominator
        loss_dict[out_key] = mean_sq.clamp(min=0).sqrt()
    for out_key, sq_key, sum_key in _POOLED_STDS:
        sq, total = loss_dict.get(sq_key), loss_dict.get(sum_key)
        if sq is None or total is None or out_key not in loss_dict:
            continue
        mean_sq = torch.as_tensor(sq, dtype=torch.float32).to(denominator.device) / denominator
        mean_x = torch.as_tensor(total, dtype=torch.float32).to(denominator.device) / denominator
        loss_dict[out_key] = (mean_sq - mean_x ** 2).clamp(min=0).sqrt()


def recompute_sum_ratios(loss_dict: Dict[str, torch.Tensor]) -> None:
    """Re-form ``sum/count`` diagnostics from their aggregated sufficient statistics.

    Mutates ``loss_dict`` in place and drops every raw statistic it consumed, so those
    never appear in the log or in ``total_loss_dict``.
    """
    consumed = set()
    for out_key, num_key, den_key in _SUM_RATIOS:
        consumed.update((num_key, den_key))
        num, den = loss_dict.get(num_key), loss_dict.get(den_key)
        if num is None or den is None or out_key not in loss_dict:
            continue
        denominator = torch.as_tensor(den, dtype=torch.float32)
        numerator = torch.as_tensor(num, dtype=torch.float32).to(denominator.device)
        # A zero count means the class is empty for this step: report 0.0, matching the
        # pre-aggregation ``sum / clamp(count, min=1)`` behaviour (which also yielded 0).
        loss_dict[out_key] = torch.where(denominator > 0,
                                         numerator / denominator.clamp(min=1.0),
                                         torch.zeros_like(denominator))
    for key in consumed:
        loss_dict.pop(key, None)


def recompute_logged_diagnostics(loss_dict: Dict[str, torch.Tensor]) -> None:
    """Single entry point: fix every aggregation-dependent diagnostic, then drop raw stats.

    ORDER IS LOAD-BEARING. The stratified ``sum/count`` ratios run first so
    ``kl_at_agree``/``kl_at_disagree`` are correct before ``kl_disagree_ratio`` (their
    ratio) is re-derived. ``recompute_pooled_stats`` needs the raw sufficient statistics,
    so it must precede the final sweep. Only then are the ``_mopd_raw/`` keys removed, so
    they never reach the log or ``total_loss_dict``.
    """
    recompute_sum_ratios(loss_dict)
    recompute_pooled_stats(loss_dict)
    recompute_divergence_ratios(loss_dict)
    for key in [key for key in loss_dict if key.startswith(_RAW_STAT_PREFIX)]:
        loss_dict.pop(key, None)


class DataSource(str, Enum):
    """Data source for GKD training."""
    DATASET = 'dataset'  # Offline: use responses from dataset
    STUDENT = 'student'  # On-policy: use student-generated responses
    TEACHER = 'teacher'  # Sequential KD: use teacher-generated responses


class MegatronGKDTrainer(MegatronRolloutMixin, MegatronRLHFTrainer):

    _VISION_SPECIAL_TOKENS = re.compile(
        r'<\|vision_start\|>|<\|vision_end\|>|<\|image_pad\|>|<\|video_pad\|>')

    def __init__(self, args: MegatronArguments, template, **kwargs):
        self.vllm_client = kwargs.pop('vllm_client', None)
        self._remote_teacher_actor = None
        self._local_completion_actor = None
        super().__init__(args, template)

        # GKD-specific parameters
        self.beta = args.beta  # JSD interpolation coefficient
        self.gkd_loss_type = args.gkd_loss_type
        self.gkd_todi_beta = args.gkd_todi_beta
        self.temperature = args.temperature
        self.loss_temperature = args.gkd_loss_temperature
        self.mopd_response_only_logits = args.mopd_response_only_logits
        self.mopd_head_chunk_size = args.mopd_head_chunk_size
        logger.info(f'GKD loss_temperature={self.loss_temperature}, rollout_temperature={self.temperature}, '
                    f'response_only_logits={self.mopd_response_only_logits}')
        logger.info(f'GKD loss_type={self.gkd_loss_type}, beta={self.beta}, todi_beta={self.gkd_todi_beta}'
                    + (' (legacy beta unused for ToDi)' if self.gkd_loss_type == 'todi' else ''))
        self.lmbda = args.lmbda  # On-policy probability
        self.seq_kd = args.seq_kd  # Sequential KD: use teacher-generated responses
        self.offload_teacher_model = args.offload_teacher_model  # Offload teacher to CPU
        self.lazy_teacher_logits = getattr(args, 'lazy_teacher_logits', True)  # True = per-microbatch (save memory), False = pre-compute all
        self.overlap_rollout_training = getattr(args, 'overlap_rollout_training', False)  # True = rollout one mb then train it (interleave)
        self.sft_alpha = getattr(args, 'sft_alpha', 0.0)  # Weight for SFT loss
        # Opt-in per-token loss (reuses Megatron's native calculate_per_token_loss path):
        # loss_func returns a 3-tuple (loss_sum_local, num_tokens_local, metric) so
        # finalize_model_grads scales gradients by 1/global_total_tokens -- long sequences
        # get proportional weight instead of the legacy per-microbatch mean. Default OFF ->
        # byte-identical to the per-sequence path.
        self.calculate_per_token_loss = bool(getattr(args, 'calculate_per_token_loss', False))
        self.use_opsd = getattr(args, 'use_opsd', False)  # OPSD: prepend reference solution to teacher prompt (use with lazy_teacher_logits)
        self.use_sft_loss = getattr(args, 'use_sft_loss', False)  # Use SFT loss
        self.teacher_label_smoothing = float(getattr(args, 'teacher_label_smoothing', 0.0))  # T' = (1-α)T + α/V
        # Teacher body reuse. When enabled with MOPD routing
        # is active AND offload_teacher_model=True, keep the teacher body resident across
        # CONSECUTIVE same-teacher microbatches — only offload/load when teacher_idx
        # changes. Cuts body loads from num_microbatches to num_distinct_teachers_this_step.
        # State: `self._resident_teacher_idx` tracks which teacher is currently on GPU
        # (None = none resident). Cleaned up at the start of `_replace_data_iterator` so
        # no teacher body is left resident when the next step's rollout runs.
        self.teacher_body_reuse = bool(getattr(args, 'teacher_body_reuse', False))
        self.mopd_teacher_cpu_cache = bool(getattr(args, 'mopd_teacher_cpu_cache', False))
        self._teacher_cpu_weights = {}
        self._resident_teacher_idx: Optional[int] = None
        # Keep metric reduction separate from the MoE collective schedule.
        self.skip_metric_dp_allreduce = bool(getattr(args, 'skip_metric_dp_allreduce', False))
        # GKD needs a teacher: either the single/primary `teacher_model`, or the
        # MOPD domain-routing list `teacher_models`. The single-teacher path keeps
        # its original requirement; only the routing path relaxes it.
        assert (args.teacher_model is not None
                or getattr(args, 'teacher_models', None)), \
            'Teacher model path is required for GKD training (set teacher_model or teacher_models)'
        self.use_vllm = getattr(args, 'use_vllm', False)

        # Multi-teacher configuration. Two independent modes:
        #   - MOPD domain routing (args.teacher_models set): N teachers, each sample routed
        #     to one teacher by its `domain` field via domain_teacher_mapping. Standard beta KL.
        #   - Two-teacher probability fusion (args.teacher_model_2 set): fuse into one target.
        # When neither is set the single-teacher path is byte-identical.
        self.multi_teacher_mode = getattr(args, 'teacher_models', None) is not None
        if self.multi_teacher_mode:
            self.teacher_model_paths = list(args.teacher_models)
            mapping = getattr(args, 'domain_teacher_mapping', None)
            self.domain_teacher_mapping = json.loads(mapping) if isinstance(mapping, str) else mapping
            self.num_teachers = len(self.teacher_model_paths)
            self.is_multi_teacher = self.num_teachers > 1
            logger.info(f'MOPD domain-routing mode: {self.num_teachers} teachers, '
                        f'mapping={self.domain_teacher_mapping}')
        else:
            self.domain_teacher_mapping = None
            # Two-teacher probability-space ensemble (opt-in via teacher_model_2).
            # Ordered list of teacher paths; index 0 is the primary teacher. When only
            # one teacher is configured the entire fusion machinery is dormant and the
            # single-teacher path is byte-identical.
            self.teacher_model_paths = [args.teacher_model]
            teacher_model_2 = getattr(args, 'teacher_model_2', None)
            if teacher_model_2 is not None:
                self.teacher_model_paths.append(teacher_model_2)
            self.num_teachers = len(self.teacher_model_paths)
            self.is_multi_teacher = self.num_teachers > 1
        self.teacher_fusion = getattr(args, 'teacher_fusion', 'prob_avg')
        # Normalized once during argument validation (_init_gkd_ensemble); equal weights
        # for the single-teacher case where the value is unused.
        fusion_weights = getattr(args, 'teacher_fusion_weights', None)
        if fusion_weights is None:
            fusion_weights = [1.0 / self.num_teachers] * self.num_teachers
        self.teacher_fusion_weights = list(fusion_weights)

        # ------------------------------------------------------------------
        # MOPD loss normalization (default None = legacy). The two ALIAS modes
        # ('sequence'/'global_token') were already folded into calculate_per_token_loss at
        # argument-validation time, so the trainer only needs to recognize the two DOMAIN
        # modes here. When a domain mode is active, _replace_data_iterator stamps every
        # microbatch with a scalar `mopd_loss_scale` (from ONE fixed-shape DP all-reduce of
        # per-DOMAIN sample/token counts) and loss_func multiplies the microbatch loss by it.
        # Presence/weights/renormalization are per DOMAIN (not per teacher_idx): domains that
        # share a teacher must NOT be collapsed before presence is known, else an absent
        # shared domain's weight would still count. The fixed domain index is
        # sorted(domain_teacher_mapping.keys()) -> position, identical on every rank so the
        # collective stays rank-symmetric. See _compute_domain_loss_scales for the verified
        # Megatron scaling chain. CP==1, MBS==1, sft_alpha==0, calculate_per_token_loss False,
        # and no overlap are all enforced at arg time. Default None -> the code below never
        # runs and the loss path is byte-identical to the pre-change trainer.
        self.mopd_loss_normalization = getattr(args, 'mopd_loss_normalization', None)
        self.mopd_domain_loss_mode = self.mopd_loss_normalization in ('domain_sequence', 'domain_token')
        self._mopd_domain_base_weights: Optional[Dict[str, float]] = None
        # Fixed rank-symmetric domain index used by loss normalization.
        self._mopd_domain_order: List[str] = sorted((self.domain_teacher_mapping or {}).keys())
        self._mopd_domain_to_idx: Dict[str, int] = {d: i for i, d in enumerate(self._mopd_domain_order)}
        if self.mopd_domain_loss_mode:
            dlw = getattr(args, 'mopd_domain_loss_weights', None)
            dlw = json.loads(dlw) if isinstance(dlw, str) else dlw
            self._mopd_domain_base_weights = self._build_domain_base_weights(
                self.domain_teacher_mapping or {}, dlw)
            logger.info(f'MOPD loss normalization ENABLED: mode={self.mopd_loss_normalization}, '
                        f'per-domain base weights={self._mopd_domain_base_weights}.')

        # Get device for data processing
        self.device = torch.cuda.current_device()
        # Initialize vLLM rollout engine if on-policy generation is enabled
        self._init_rollout_engine()

        # Teacher models will be loaded in setup_model_and_optimizer
        # Using the same parallel parameters (PP/TP/CP/EP) as student model.
        # `self.teacher_models` is the primary teacher's chunk list (one entry per VP
        # stage) and stays the canonical single-teacher handle. For the two-teacher
        # ensemble, `self.teacher_groups[teacher_idx]` holds each teacher's chunk list
        # and `self.teacher_models is self.teacher_groups[0]`.
        self.teacher_models = []
        self.teacher_groups: List[list] = []
        # teacher_idx -> whether that teacher is MoE (populated in _load_teacher_model).
        self._teacher_is_moe: Dict[int, bool] = {}

        # Teacher model config for temporary args override during forward
        # When teacher and student have different architecture, we need to override args temporarily.
        # `self._teacher_megatron_config` mirrors `self.teacher_megatron_configs[0]` for the
        # single-teacher path; the per-teacher list backs the ensemble forwards.
        self._teacher_megatron_config: Optional[Dict] = None  # Will be set in _load_teacher_model
        self.teacher_megatron_configs: List[Optional[Dict]] = []
        # Index selecting which teacher's config _teacher_args_context applies. Set transiently
        # by the fused-logits helper while iterating teachers; defaults to the primary teacher.
        self._active_teacher_idx: int = 0

        # Truncation strategy for handling sequences that exceed max_length
        self.truncation_strategy = args.truncation_strategy
        self.max_completion_length = args.max_completion_length

        # Resample iterator will be initialized lazily
        self.resample_data_iterator = None
        self._train_valid_test_dataset_provider = None
        self._resample_train_dataset = None
        self._resample_route_indices = None
        self._teacher_dummy_batch = None
        self._coordination_dummy_encoded = None
        self._scheduled_num_microbatches = None

        # Encode-resample transient-retry config (truncation_strategy='delete' path). Args are
        # the primary source (declared in megatron_args.py); environment variables
        # may override them. See resample_encode_failed_inputs for classification.
        self.mopd_resample_retries = max(0, int(_env_or_arg(
            'MOPD_RESAMPLE_RETRIES', getattr(args, 'mopd_resample_retries', 0), int)))
        self.mopd_resample_backoff_base_s = max(0.0, float(_env_or_arg(
            'MOPD_RESAMPLE_BACKOFF_BASE_S', getattr(args, 'mopd_resample_backoff_base_s', 0.5), float)))
        self.mopd_resample_retry_budget_s = max(0.0, float(_env_or_arg(
            'MOPD_RESAMPLE_RETRY_BUDGET_S', getattr(args, 'mopd_resample_retry_budget_s', 60.0), float)))
        # Rank-LOCAL counter of encode failures keyed by (domain, exception-type-name). Involves
        # NO collective communication and does NOT touch the fixed-shape metric tensor that the
        # DP all_reduce in `_log_metrics` reduces. Summarized once at end of train() per rank.
        # defaultdict(int) is imported at module top.
        self._resample_failure_counts = defaultdict(int)

        # Diagnostic metrics interval (0 = every step, N = every N steps + step 1)
        self._diagnostics_interval = int(getattr(args, 'diagnostics_interval', 1) or 1)

    def train(self, train_dataset, val_dataset, data_collator):
        """Override train to initialize resample iterator for truncation_strategy='delete'."""
        needs_resample = (self.truncation_strategy == 'delete')
        if needs_resample:
            self._train_valid_test_dataset_provider = get_swift_datasets_provider(train_dataset, val_dataset)
            self._train_valid_test_dataset_provider.is_distributed = True
            self._resample_train_dataset = train_dataset
            self._resample_route_indices = None
            self._data_collator = data_collator

        try:
            super().train(train_dataset, val_dataset, data_collator)
        finally:
            self._log_resample_failure_summary()

    def _log_resample_failure_summary(self):
        """Emit ONE end-of-run line summarizing encode-resample failures on THIS rank.

        Rank-local read of ``self._resample_failure_counts`` (defaultdict keyed by
        ``(domain, exception-type-name)``). Performs NO collective communication and touches NO
        metric tensor — the values are per-rank and printed as-is. The writer-rank slice is
        representative; aggregating across ranks would require a separate fixed-shape
        collective, which this diagnostic deliberately avoids. Silent when nothing failed.
        """
        counts = getattr(self, '_resample_failure_counts', None)
        if not counts:
            return
        total = sum(counts.values())
        try:
            rank = mpu.get_data_parallel_rank()
        except Exception:
            rank = int(os.environ.get('RANK', '0'))
        parts = ', '.join(f'{domain!r}/{exc}={n}'
                          for (domain, exc), n in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))))
        logger.warning(
            f'[resample-failure-summary] dp_rank={rank}: {total} encode-failure event(s) this run '
            f'(rank-local, not DP-reduced). By (domain, exception): {parts}')

    def setup_model_and_optimizer(self, model_provider_func, model_type, *_args, **kwargs):
        """Setup model and optimizer, including teacher model(s).

        Teacher models use the same parallel parameters (PP/TP/CP/EP) as the student model.
        For the two-teacher ensemble, each teacher is loaded sequentially into its own
        chunk list; `self.teacher_models` always aliases the primary teacher (index 0).
        """
        if getattr(self.args, 'mopd_teacher_endpoint', None):
            # No teacher backbones on the student nodes. Keep their frozen
            # native vocabulary heads only; the independent server owns forward.
            result = super().setup_model_and_optimizer(model_provider_func, model_type, *_args, **kwargs)
            from .remote_teacher_actor import RemoteTeacherActor
            self._remote_teacher_actor = RemoteTeacherActor(self)
            return result
        for teacher_idx, teacher_model_path in enumerate(self.teacher_model_paths):
            logger.info(f'Loading teacher model [{teacher_idx}] from: {teacher_model_path}')
            self._load_teacher_model(teacher_model_path, model_type, teacher_idx=teacher_idx)

        result = super().setup_model_and_optimizer(model_provider_func, model_type, *_args, **kwargs)
        if self.args.mopd_rollout_completion_chunk_size:
            from .local_completion_actor import LocalCompletionActor
            self._local_completion_actor = LocalCompletionActor(self)
        return result


    @staticmethod
    def _tokenizer_id_signature(tokenizer):
        """Build a deterministic fingerprint of a tokenizer's token-to-id mapping.

        Returns (vocab_size, sorted_special_token_ids_tuple, vocab_hash) where vocab_hash
        is a stable hash over the full token->id mapping. Two tokenizers with the same
        padded vocab size but different id assignments produce different signatures.
        """
        import hashlib
        try:
            vocab = tokenizer.get_vocab()  # dict: token -> id
        except Exception:
            vocab = None
        if vocab is not None:
            # Hash the (token, id) pairs in id order so the fingerprint is mapping-sensitive
            # and order-independent of dict iteration.
            items = sorted(vocab.items(), key=lambda kv: (kv[1], kv[0]))
            h = hashlib.sha256()
            for tok, idx in items:
                h.update(repr((tok, idx)).encode('utf-8'))
            vocab_hash = h.hexdigest()
            vocab_size = len(vocab)
        else:
            vocab_hash = None
            vocab_size = getattr(tokenizer, 'vocab_size', None)
        specials = []
        for attr in ('pad_token_id', 'eos_token_id', 'bos_token_id', 'unk_token_id'):
            specials.append((attr, getattr(tokenizer, attr, None)))
        return vocab_size, tuple(specials), vocab_hash

    def _check_teacher_tokenizer_matches_student(self, teacher_idx, teacher_model_path, teacher_model_dir):
        """Fail-fast if a teacher's tokenizer token-id MAPPING differs from the student's.

        Compares the deterministic token->id mapping and key special-token ids, not only
        the vocab size. Probability-space fusion requires identical token grids; mismatched
        mappings would silently mix probabilities over different tokens.
        """
        student_tokenizer = getattr(self.template, 'tokenizer', None)
        if student_tokenizer is None:
            logger.warning(
                'Cannot verify teacher tokenizer mapping: student tokenizer unavailable on '
                'self.template; skipping the token-id-mapping fail-fast.')
            return
        try:
            teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_dir, trust_remote_code=True)
        except Exception as e:
            raise ValueError(
                f'Two-teacher ensemble: failed to load tokenizer for teacher [{teacher_idx}] '
                f'({teacher_model_path}) to verify token-id mapping: {e}')

        s_size, s_specials, s_hash = self._tokenizer_id_signature(student_tokenizer)
        t_size, t_specials, t_hash = self._tokenizer_id_signature(teacher_tokenizer)

        if s_size != t_size:
            raise ValueError(
                f'Two-teacher ensemble requires identical tokenizer vocab across student and all '
                f'teachers, but teacher [{teacher_idx}] ({teacher_model_path}) has tokenizer vocab '
                f'size {t_size} vs student {s_size}.')
        if s_specials != t_specials:
            raise ValueError(
                f'Two-teacher ensemble requires identical special-token ids across student and all '
                f'teachers, but teacher [{teacher_idx}] ({teacher_model_path}) differs: '
                f'teacher specials={t_specials} vs student specials={s_specials}.')
        if s_hash is not None and t_hash is not None and s_hash != t_hash:
            raise ValueError(
                f'Two-teacher ensemble requires identical token-id MAPPING across student and all '
                f'teachers. Teacher [{teacher_idx}] ({teacher_model_path}) has the same vocab size '
                f'as the student but a different token->id mapping (vocab fingerprint mismatch). '
                f'Probability-space fusion over mismatched token grids is invalid.')

    def _load_teacher_model(self, teacher_model_path: str, model_type: str, teacher_idx: int = 0):
        megatron_args = get_args()
        vp_size = getattr(megatron_args, 'virtual_pipeline_model_parallel_size')
        assert vp_size is None or vp_size == 1, 'GKD currently does not support VPP.'
        # Per-teacher type/revision. MOPD domain-routing mode indexes the teacher_models_type/
        # _revision lists; the two-teacher fusion mode uses teacher_model[_2]_type.
        if self.multi_teacher_mode:
            types = getattr(self.args, 'teacher_models_type', None) or []
            revisions = getattr(self.args, 'teacher_models_revision', None) or []
            tm_type = types[teacher_idx] if teacher_idx < len(types) else None
            tm_revision = revisions[teacher_idx] if teacher_idx < len(revisions) else None
        elif teacher_idx == 0:
            tm_type = getattr(self.args, 'teacher_model_type', None)
            tm_revision = getattr(self.args, 'teacher_model_revision', None)
        else:
            tm_type = getattr(self.args, 'teacher_model_2_type', None)
            tm_revision = getattr(self.args, 'teacher_model_2_revision', None)
        teacher_model_info, _ = get_model_info_meta(
            teacher_model_path,
            model_type=tm_type,
            model_revision=tm_revision,
            use_hf=self.args.use_hf,
            hub_token=self.args.hub_token)
        teacher_model_type = teacher_model_info.model_type

        # Get teacher's HF config and convert to Megatron config
        teacher_config = AutoConfig.from_pretrained(teacher_model_info.model_dir, trust_remote_code=True)
        teacher_megatron_model_meta = get_megatron_model_meta(teacher_model_type)
        if teacher_megatron_model_meta is None:
            raise ValueError(f'Teacher model type "{teacher_model_type}" is not supported in Megatron. '
                             f'Teacher model path: {teacher_model_path}')

        teacher_megatron_config = convert_hf_config(teacher_config)

        # Store teacher config for temporary args override during forward.
        # Primary teacher (idx 0) keeps the canonical single-teacher handle; the
        # per-teacher list backs ensemble forwards.
        self._teacher_megatron_config = teacher_megatron_config
        while len(self.teacher_megatron_configs) <= teacher_idx:
            self.teacher_megatron_configs.append(None)
        self.teacher_megatron_configs[teacher_idx] = teacher_megatron_config

        # Vocab fail-fast (two-teacher): teachers must share the student's padded vocab
        # AND token-id mapping so probability-space fusion mixes distributions over the
        # SAME token grid. The pad+copy alignment trick is only valid pairwise, so
        # heterogeneous vocab is rejected here, before any teacher forward, rather than
        # silently mixing distributions over mismatched grids. Two tokenizers can share a
        # padded vocab size but assign different ids, so we compare the deterministic
        # token-to-id mapping (not only the size).
        if self.is_multi_teacher:
            student_vocab = getattr(megatron_args, 'padded_vocab_size', None)
            teacher_vocab = teacher_megatron_config.get('padded_vocab_size', None)
            if (student_vocab is not None and teacher_vocab is not None
                    and student_vocab != teacher_vocab):
                raise ValueError(
                    f'Multi-teacher distillation requires identical padded vocab across student and '
                    f'all teachers, but teacher [{teacher_idx}] ({teacher_model_path}) has '
                    f'padded_vocab_size={teacher_vocab} vs student padded_vocab_size='
                    f'{student_vocab}. Heterogeneous-vocab teachers are not supported in this '
                    f'version.')
            self._check_teacher_tokenizer_matches_student(teacher_idx, teacher_model_path,
                                                          teacher_model_info.model_dir)

        logger.info(f'Loading teacher model: type={teacher_model_type}, '
                    f'hidden_size={teacher_megatron_config.get("hidden_size")}, '
                    f'num_layers={teacher_megatron_config.get("num_layers")}')

        # Store original student model config from Megatron global args
        # We need to override these with teacher's config temporarily
        essential_keys = {'hf_model_type', 'model_dir'}

        # MoE-related keys that must be explicitly handled for Dense/MoE compatibility.
        # When student is MoE and teacher is Dense (or vice versa), these keys need to be
        # properly reset to ensure correct model architecture creation.
        moe_related_keys = {
            'num_experts',
            'moe_ffn_hidden_size',
            'moe_shared_expert_intermediate_size',
            'moe_router_topk',
            'moe_router_num_groups',
            'moe_router_group_topk',
            'moe_router_pre_softmax',
            'moe_router_score_function',
            'moe_router_bias_update_rate',
            'moe_router_topk_scaling_factor',
            'moe_router_load_balancing_type',
            'moe_router_enable_expert_bias',
            'moe_apply_probs_on_input',
            'moe_layer_freq',
            'moe_grouped_gemm',
            'moe_use_legacy_grouped_gemm',
            'use_shared_expert_gate',
        }

        # EP-related keys that need special handling for Dense/MoE compatibility.
        # Dense models cannot use expert parallelism (EP > 1).
        ep_related_keys = {
            'expert_model_parallel_size',
            'expert_tensor_parallel_size',
        }

        keys_to_override = (set(teacher_megatron_config.keys()) | essential_keys | moe_related_keys | ep_related_keys)

        original_config = {}
        for key in keys_to_override:
            if hasattr(megatron_args, key):
                original_config[key] = getattr(megatron_args, key)

        # Apply teacher config to global Megatron args
        for key, value in teacher_megatron_config.items():
            setattr(megatron_args, key, value)
        megatron_args.hf_model_type = teacher_model_type
        megatron_args.model_dir = teacher_model_info.model_dir

        # Reset MoE-related keys that are not in teacher config to None.
        # This ensures Dense teacher doesn't inherit MoE settings from MoE student,
        # and MoE teacher gets its own settings without interference from Dense student.
        teacher_is_moe = teacher_megatron_config.get('num_experts') is not None
        for key in moe_related_keys:
            if key not in teacher_megatron_config and hasattr(megatron_args, key):
                setattr(megatron_args, key, None)

        if teacher_is_moe:
            MegatronArguments._set_moe_default(megatron_args)
            # Ensure moe_grouped_gemm is True for MoE models to use GroupedMLP,
            # which is required by gpt_bridge weight loading logic.
            if megatron_args.moe_grouped_gemm is None:
                megatron_args.moe_grouped_gemm = True

            # Restore original EP settings if student is Dense.
            # This allows MoE teacher to use EP > 1 even when student is Dense.
            if not self.student_is_moe:
                megatron_args.expert_model_parallel_size = self._original_ep_size
                megatron_args.expert_tensor_parallel_size = self._original_etp_size
        else:
            # Dense teacher cannot use expert parallelism.
            # Reset EP to 1 to avoid "num_moe_experts must be non None to use expert-parallel" error.
            megatron_args.expert_model_parallel_size = 1
            megatron_args.expert_tensor_parallel_size = 1
        converted_backing = ([] if self.offload_teacher_model
                             and getattr(self, 'mopd_teacher_cpu_cache', False)
                             and getattr(megatron_args, 'mopd_model_cache_dir', None) else None)
        try:
            # Use get_model() to create teacher with same parallel config (PP/TP/CP/EP) as student
            # but with teacher's model architecture (hidden_size, num_layers, etc.)
            teacher_models = get_model(teacher_megatron_model_meta.model_provider, model_type, wrap_with_ddp=False)

            # Create bridge for teacher model (for weight loading)
            teacher_bridge = teacher_megatron_model_meta.bridge_cls()

            # Load teacher weights and set to eval mode
            for m in teacher_models:
                unwrapped = unwrap_model(m)
                cache_result = teacher_bridge.load_weights(unwrapped, teacher_model_info.model_dir)
                if converted_backing is not None:
                    from .teacher_cpu_weights import converted_cpu_weights
                    converted_backing.extend(converted_cpu_weights(unwrapped, cache_result))
                unwrapped.requires_grad_(False)
                unwrapped.eval()

            logger.info(f'Teacher model loaded successfully with PP={megatron_args.pipeline_model_parallel_size}, '
                        f'TP={megatron_args.tensor_model_parallel_size}')

        finally:
            # Restore original student config to Megatron global args
            for key, value in original_config.items():
                setattr(megatron_args, key, value)

        # Store this teacher's chunk list. teacher_groups[teacher_idx] holds each teacher;
        # the primary teacher aliases self.teacher_models for the byte-identical single path.
        while len(self.teacher_groups) <= teacher_idx:
            self.teacher_groups.append(None)
        self.teacher_groups[teacher_idx] = teacher_models
        if teacher_idx == 0:
            self.teacher_models = teacher_models
        # Per-teacher architecture, keyed by index. Needed by _assert_ep_teacher_consistent:
        # only an MoE teacher runs the expert all-to-all that can mix weights across
        # teachers, and a dense teacher is forced to EP=1 above regardless of student EP.
        self._teacher_is_moe[teacher_idx] = teacher_is_moe

        # Offload teacher models to CPU if enabled
        if self.offload_teacher_model:
            if getattr(self, 'mopd_teacher_cpu_cache', False):
                # A newly loaded teacher/checkpoint owns a new immutable source.
                self._teacher_cpu_weights.pop(teacher_idx, None)
                self._offload_one_teacher(teacher_idx, cpu_weights=converted_backing)
            else:
                offload_megatron_model_to_cpu(teacher_models)
            logger.info(f'Teacher model [{teacher_idx}] offloaded to CPU to save GPU memory')

    @contextmanager
    def _teacher_args_context(self, teacher_idx: Optional[int] = None):
        """Context manager to temporarily override Megatron args with a teacher's config.

        This is necessary for teacher model forward to use correct hidden_size, num_layers, etc.
        when the teacher has a different architecture than the student. When `teacher_idx` is
        None the active teacher index (`self._active_teacher_idx`, default 0) is used, so the
        single-teacher path is unchanged.
        """
        megatron_args = get_args()

        idx = self._active_teacher_idx if teacher_idx is None else teacher_idx
        if self.teacher_megatron_configs and idx < len(self.teacher_megatron_configs) \
                and self.teacher_megatron_configs[idx] is not None:
            teacher_config = self.teacher_megatron_configs[idx]
        else:
            teacher_config = self._teacher_megatron_config

        # Save original values and override with teacher config
        original_values = {}
        for key, value in teacher_config.items():
            if hasattr(megatron_args, key):
                original_values[key] = getattr(megatron_args, key)
                setattr(megatron_args, key, value)

        try:
            yield
        finally:
            # Restore original values
            for key, value in original_values.items():
                setattr(megatron_args, key, value)

    def _offload_teacher_models(self):
        """Offload all teacher models to CPU to save GPU memory."""
        for index, teacher_models in enumerate(self.teacher_groups):
            if teacher_models:
                if getattr(self, 'mopd_teacher_cpu_cache', False):
                    self._offload_one_teacher(index)
                else:
                    offload_megatron_model_to_cpu(teacher_models)

    def _load_teacher_models_to_gpu(self):
        """Load all teacher models back to GPU."""
        for index, teacher_models in enumerate(self.teacher_groups):
            if teacher_models:
                if getattr(self, 'mopd_teacher_cpu_cache', False):
                    self._load_one_teacher_to_gpu(index)
                else:
                    load_megatron_model_to_gpu(teacher_models, load_grad=False)

    def _offload_one_teacher(self, teacher_idx: int, *, cpu_weights=None):
        """Offload a single teacher's chunks to CPU (sequential ensemble forward)."""
        teacher_models = self.teacher_groups[teacher_idx]
        if teacher_models:
            if getattr(self, 'mopd_teacher_cpu_cache', False):
                from .teacher_cpu_weights import FrozenTeacherCPUWeights
                created = teacher_idx not in self._teacher_cpu_weights
                if created:
                    if cpu_weights is None:
                        cached = FrozenTeacherCPUWeights(teacher_models)
                    else:
                        cached = FrozenTeacherCPUWeights(teacher_models, pin_memory=False, cpu_weights=cpu_weights)
                    self._teacher_cpu_weights[teacher_idx] = cached
                    logger.info(f'Teacher [{teacher_idx}] CPU backing: {cached.backing_kind}, '
                                f'logical_bytes={cached.nbytes}, copied_cpu_bytes={cached.cpu_copy_bytes}')
                cached = self._teacher_cpu_weights[teacher_idx]
                changed = cached.offload()
                return
            offload_megatron_model_to_cpu(teacher_models)

    def _load_one_teacher_to_gpu(self, teacher_idx: int):
        """Load a single teacher's chunks to GPU (sequential ensemble forward)."""
        teacher_models = self.teacher_groups[teacher_idx]
        if teacher_models:
            if getattr(self, 'mopd_teacher_cpu_cache', False):
                cached = self._teacher_cpu_weights[teacher_idx]
                changed = cached.load(self.device)
                return
            load_megatron_model_to_gpu(teacher_models, load_grad=False)

    @contextmanager
    def load_teacher_model_context(self, teacher_idx: Optional[int] = None):
        """Context manager to load teacher models for forward pass and optionally offload after.

        When offload_teacher_model is enabled:
        - Load teacher models to GPU before forward pass
        - Offload teacher models to CPU after forward pass

        If `teacher_idx` is given (MOPD domain routing / multi-teacher), only that teacher's
        chunks are loaded/offloaded — keeps at most one teacher resident. Otherwise all
        teacher groups are loaded (legacy single/two-teacher behavior).

        This saves GPU memory during the training step.
        """
        # The EP teacher-consistency guard runs regardless of offload: routing picks the
        # teacher (and therefore the expert shards) whether or not the other teachers
        # happen to also sit on GPU.
        if teacher_idx is not None:
            self._assert_teacher_uniform_across_ep(teacher_idx)

        if not self.offload_teacher_model:
            yield
            return

        if teacher_idx is not None:
            self._load_one_teacher_to_gpu(teacher_idx)
        else:
            self._load_teacher_models_to_gpu()
        try:
            yield
        finally:
            if teacher_idx is not None:
                self._offload_one_teacher(teacher_idx)
            else:
                self._offload_teacher_models()

    def _assert_teacher_uniform_across_ep(self, teacher_idx: int) -> None:
        """Fail loudly if the ranks of one EP group disagree on the current teacher.

        Offloading keeps at most one teacher resident PER RANK -- a memory property. MoE
        correctness needs a different, stronger property: every rank of an expert-parallel
        group must run the SAME teacher at the same time. With EP>1 the experts of one
        teacher are sharded across ranks and the MoE all-to-all dispatches each token to
        whichever rank owns its target expert, so if rank A is resident on teacher 0 while
        rank B is resident on teacher 1, A's tokens get processed by B's copy of the
        experts. Nothing raises: all teachers share an architecture, so every shape and
        dtype in the all-to-all still matches, and the step silently trains against a
        chimera of two teachers.

        Today's safety rests on ``_group_batch_by_teacher`` sorting each rank's LOCAL
        domain histogram: when the sampler hands every DP rank an identical domain count
        the sorted orders coincide and microbatch slot k is the same teacher everywhere.
        That is an incidental property of the sampler, not an enforced invariant -- an
        uneven domain supply, a sampler change, or a single image-read failure that
        resamples one rank's sample is enough to break it. This assert converts that
        silent corruption into a crash.

        Only an MoE teacher runs that all-to-all, so a mismatch among dense teachers (which
        are forced to EP=1 in ``_load_teacher_model`` regardless of student EP) is benign
        and allowed; an unknown teacher is treated as MoE so the guard fails safe.

        Collective: one 2-element int64 all-gather ((teacher_idx, is_moe)) over the EP
        group, issued unconditionally. Must therefore be reached by all ranks of the group
        for the same microbatch -- both call sites run per microbatch on every train rank.
        """
        if not self.multi_teacher_mode:
            return
        try:
            ep_size = mpu.get_expert_model_parallel_world_size()
        except AssertionError:
            return  # expert parallelism not initialized (non-MoE path)
        if ep_size <= 1:
            return
        ep_group = mpu.get_expert_model_parallel_group()
        # Gather (teacher_idx, is_moe) per rank. `ep_size` above reflects the STUDENT's EP,
        # so it stays >1 even when the routed teachers are dense; a dense teacher is built
        # with EP=1 and runs no expert all-to-all, so a per-rank teacher mismatch among
        # dense teachers cannot mix expert weights and must not abort. The all-gather is
        # unconditional -- deciding rank-locally whether to participate would deadlock the
        # moment one rank selected a dense teacher and another an MoE teacher.
        is_moe = 1 if self._teacher_is_moe.get(teacher_idx, True) else 0
        local = torch.tensor([teacher_idx, is_moe], dtype=torch.int64, device=torch.cuda.current_device())
        gathered = torch.empty(ep_size * 2, dtype=torch.int64, device=local.device)
        torch.distributed.all_gather_into_tensor(gathered, local, group=ep_group)
        gathered = gathered.view(ep_size, 2)
        indices, moe_flags = gathered[:, 0], gathered[:, 1]
        if bool((indices == indices[0]).all()):
            return
        if not bool(moe_flags.any()):
            return  # all selected teachers dense: no expert all-to-all, mismatch is benign
        raise RuntimeError(
            f'MOPD EP consistency violation: ranks of this expert-parallel group are on '
            f'different teachers {indices.tolist()} (this rank: {teacher_idx}), and at least one '
            f'is MoE (per-rank is_moe: {moe_flags.tolist()}). The MoE all-to-all would route '
            f'tokens to another teacher\'s expert shards, silently mixing teacher weights. Make '
            f'the per-step domain histogram identical across DP ranks so teacher-sorted '
            f'microbatch slots align.')

    def _ensure_teacher_resident(self, teacher_idx: int) -> None:
        """Ensure teacher ``teacher_idx`` is resident on the GPU,
        offloading a different resident teacher if necessary. No-op if that teacher is
        already the resident one.

        Callers must be on the `teacher_body_reuse` + `offload_teacher_model` +
        `multi_teacher_mode` code path (guarded at the call site) — this method assumes
        the residency schedule is the source of truth (it does NOT offload after the
        forward; that is the whole point). End-of-step cleanup is done by
        `_offload_resident_teacher_if_any()` at the start of the next
        `_replace_data_iterator`.

        Correctness note: the logic is order-independent. If microbatches are somehow
        NOT teacher-sorted, this still loads the right teacher on every switch (just
        with more swaps — efficiency loss but no correctness loss).
        """
        self._assert_teacher_uniform_across_ep(teacher_idx)
        if self._resident_teacher_idx == teacher_idx:
            return  # already resident — skip both offload and load
        if self._resident_teacher_idx is not None:
            self._offload_one_teacher(self._resident_teacher_idx)
        self._load_one_teacher_to_gpu(teacher_idx)
        self._resident_teacher_idx = teacher_idx

    def _offload_resident_teacher_if_any(self) -> None:
        """End-of-step cleanup for teacher-body reuse: if a teacher body is still resident
        from the previous step's last microbatch, offload it to CPU and reset the
        residency marker. Called at the start of `_replace_data_iterator` so the next
        step's rollout / student forward has clean memory. Safe to call unconditionally
        (no-op when no teacher is resident, e.g. when the reuse flag is off).
        """
        if self._resident_teacher_idx is None:
            return
        self._offload_one_teacher(self._resident_teacher_idx)
        self._resident_teacher_idx = None

    @contextmanager
    def _template_context(self, template: Template, max_length: Optional[int] = None):
        """Context manager to temporarily modify max_length constraint from template."""
        original_max_length = template.max_length
        template.max_length = max_length
        try:
            yield
        finally:
            template.max_length = original_max_length

    def _clean_vision_tokens_in_response(self, batch: List[Dict]) -> None:
        """Remove vision special token text from generated assistant responses.

        During vLLM rollout the model may emit vision special tokens (e.g. <|image_pad|>).
        When the response text is re-tokenized these become real special token ids in input_ids,
        causing get_rope_index to crash because there is no corresponding image_grid_thw/video_grid_thw.
        """
        for data in batch:
            messages = data.get('messages')
            if not messages:
                continue
            last_msg = messages[-1]
            if last_msg.get('role') == 'assistant' and isinstance(last_msg.get('content'), str):
                cleaned = self._VISION_SPECIAL_TOKENS.sub('', last_msg['content'])
                if cleaned != last_msg['content']:
                    last_msg['content'] = cleaned

    def _get_vision_token_ids(self) -> set:
        """Token ids of vision special tokens, resolved once from the tokenizer.

        Empty for non-multimodal templates (tokens absent / map to unk)."""
        cached = getattr(self, '_vision_token_ids_cache', None)
        if cached is not None:
            return cached
        ids = set()
        tokenizer = getattr(self.template, 'tokenizer', None) or getattr(self.template.processor, 'tokenizer', None)
        unk_id = getattr(tokenizer, 'unk_token_id', None)
        for tok in ('<|vision_start|>', '<|vision_end|>', '<|image_pad|>', '<|video_pad|>'):
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != unk_id:
                ids.add(tid)
        self._vision_token_ids_cache = ids
        return ids

    def _clean_vision_tokens_in_response_ids(self, batch: List[Dict]) -> None:
        """Strip vision special token ids from rollout response token ids.

        The text-level cleaner (_clean_vision_tokens_in_response) is bypassed once
        _maybe_replace_response_token overwrites assistant content with raw vLLM token ids.
        Those ids can still contain spurious <|vision_start|>/<|image_pad|> emitted during
        generation, which makes input_ids carry more image regions than image_grid_thw has,
        crashing get_rope_index. Remove them here, keeping response_loss_mask and
        rollout_logprobs aligned at the same positions.
        """
        vision_ids = self._get_vision_token_ids()
        if not vision_ids:
            return

        def filter_seq(ids, mask, logps):
            keep = [i for i, t in enumerate(ids) if t not in vision_ids]
            if len(keep) == len(ids):
                return ids, mask, logps, 0
            removed = len(ids) - len(keep)
            new_ids = [ids[i] for i in keep]
            new_mask = [mask[i] for i in keep] if mask is not None else None
            new_logps = [logps[i] for i in keep] if logps is not None else None
            return new_ids, new_mask, new_logps, removed

        total_removed = 0
        for data in batch:
            token_ids = data.get('response_token_ids')
            if not token_ids:
                continue
            loss_mask = data.get('response_loss_mask')
            logprobs = data.get('rollout_logprobs')
            # Normalize to nested List[List[...]] (mirror replace_assistant_response_with_ids).
            nested = isinstance(token_ids[0], list)
            ids_turns = token_ids if nested else [token_ids]
            mask_turns = loss_mask if (loss_mask and isinstance(loss_mask[0], list)) else (
                [loss_mask] if loss_mask else [None] * len(ids_turns))
            logp_turns = logprobs if (logprobs and isinstance(logprobs[0], list)) else (
                [logprobs] if logprobs else [None] * len(ids_turns))

            new_ids_turns, new_mask_turns, new_logp_turns = [], [], []
            item_removed = 0
            for t, ids_t in enumerate(ids_turns):
                m_t = mask_turns[t] if t < len(mask_turns) else None
                lp_t = logp_turns[t] if t < len(logp_turns) else None
                f_ids, f_mask, f_logps, removed = filter_seq(ids_t, m_t, lp_t)
                item_removed += removed
                new_ids_turns.append(f_ids)
                new_mask_turns.append(f_mask)
                new_logp_turns.append(f_logps)

            if item_removed == 0:
                continue
            total_removed += item_removed
            data['response_token_ids'] = new_ids_turns if nested else new_ids_turns[0]
            if loss_mask is not None:
                data['response_loss_mask'] = new_mask_turns if (
                    loss_mask and isinstance(loss_mask[0], list)) else new_mask_turns[0]
            if logprobs is not None:
                data['rollout_logprobs'] = new_logp_turns if (
                    logprobs and isinstance(logprobs[0], list)) else new_logp_turns[0]

        if total_removed:
            logger.warning_once(
                f'Stripped {total_removed} vision special token id(s) from rollout responses '
                'before re-encoding (student emitted vision tokens in its completion).')

    def _maybe_replace_response_token(self, batch: List[Dict]) -> None:
        """Use vLLM response_token_ids directly to avoid re-tokenize mismatch with rollout_logprobs."""
        from swift.trainers.rlhf_trainer.utils import replace_assistant_response_with_ids

        for data in batch:
            if data.get('response_token_ids'):
                loss_mask = data.get('response_loss_mask')
                data['messages'] = replace_assistant_response_with_ids(
                    data['messages'], data['response_token_ids'], loss_mask)











    def _encode_batch(self, batch: List[Dict], *, megatron_args=None) -> Dict[str, torch.Tensor]:
        """Encode a batch of raw data into model inputs."""
        self._maybe_replace_response_token(batch)
        template = self.template
        # A teacher CPU lookahead encoder uses an owner-thread snapshot.
        # Teacher forwards temporarily mutate the process-global model args.
        args = get_args() if megatron_args is None else megatron_args
        max_length = (getattr(self.args, 'mopd_max_sequence_length', None)
                      or template.max_length + self.max_completion_length)
        with self._template_context(template, max_length=max_length):
            encoded_list = [template.encode(data, return_length=True) for data in batch]
            padding_to = get_padding_to(args)
            # A completion chunk holds many MBS1 inputs. Keep queued pixel/
            # position tensors on CPU; native _prepare_batch moves only the
            # consumed microbatch to CUDA for both teacher and student.
            target_device = 'cpu' if (
                getattr(self.args, 'mopd_teacher_endpoint', None)
                or getattr(self.args, 'mopd_rollout_completion_chunk_size', 0)) else self.device
            encoded_batch = to_device(template.data_collator(encoded_list, padding_to=padding_to), target_device)

        encoded_batch['num_samples'] = len(batch)
        if len(batch) == 1 and batch[0].get('_mopd_generation') is not None:
            # Behavior-logprob diagnostics follow this sample's actual sampling
            # temperature. Full-vocab KD still uses self.loss_temperature.
            encoded_batch['_mopd_rollout_temperature'] = batch[0]['_mopd_generation']['temperature']
        routing_wire = batch[0].get('_mopd_router_replay_wire') if len(batch) == 1 else None
        if routing_wire is not None:
            from swift.llm.infer.routing_protocol import decode_routing_trace
            decoded = decode_routing_trace(routing_wire, max_tokens=encoded_batch['input_ids'].shape[-1])
            encoded_batch['_mopd_router_replay'] = {
                'token_ids': torch.from_numpy(decoded['token_ids']),
                'expert_ids': torch.from_numpy(decoded['expert_ids']),
                'num_experts': decoded['num_experts'],
            }
        return encoded_batch


    @staticmethod
    def _message_to_text(message) -> str:
        content = message.get('content') if isinstance(message, dict) else message
        if content is None:
            return ''
        if isinstance(content, str):
            return content
        return str(content)

    def _extract_rollout_prompt_completion(self, sample: Dict) -> tuple:
        messages = sample.get('messages', []) if isinstance(sample, dict) else []
        prompt_parts = []
        completion = ''
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get('role') or message.get('from')
            text = self._message_to_text(message)
            if role in ('assistant', 'gpt'):
                completion = text
            elif role in ('user', 'human', 'system'):
                prompt_parts.append(text)
        return '\n\n'.join(prompt_parts), completion

    def _postprocess_rollout_outputs(self, inputs, outputs):
        results = super()._postprocess_rollout_outputs(inputs, outputs)
        for result, output in zip(results, outputs):
            # Keep binary route metadata out of human-readable rollout_infos
            # and do not modify the shared RolloutOutput object.
            infos = result.get('rollout_infos', {})
            if 'mopd_routing' in infos:
                result['_mopd_router_replay_wire'] = infos['mopd_routing']
                result['rollout_infos'] = {key: value for key, value in infos.items() if key != 'mopd_routing'}
            usage = getattr(getattr(output, 'response', None), 'usage', None)
            if usage is not None and getattr(usage, 'prompt_tokens', None) is not None:
                result['prompt_length'] = usage.prompt_tokens
        return results

    def _log_student_rollout_samples(self, rollout_batch: List[Dict], step: Optional[int] = None,
                                    *, preselected: bool = False, batch_size: Optional[int] = None) -> None:
        # The complete-response path selects BEFORE completion-order delivery.
        # Only lightweight selected records are gathered; images stay on disk.
        # Use one writer rank independently of the logger configuration.
        if not rollout_batch or not megatron_is_last_rank():
            return
        args = get_args()
        step = int(self._step if step is None else step)
        try:
            from .rollout_samples import sample_indices, sample_record, write_sample_records
            if preselected:
                records = rollout_batch
            else:
                indices = sample_indices(len(rollout_batch), self.args.mopd_rollout_samples_per_step,
                                         self.args.seed, step)
                records = [sample_record(rollout_batch[index], step=step, row_index=index,
                                         batch_size=batch_size or getattr(args, 'global_batch_size', len(rollout_batch)))
                           for index in indices]
                # Legacy batch-at-once/overlap gathers only this DP partition.
                # Do not label its local row indexes as global-window coverage.
                for record in records:
                    record.update(sampling_scope='writer_dp_partition', sampling_population_size=len(rollout_batch))
            if records:
                write_sample_records(os.path.join(args.save, 'generations_media'), step, records)
        except FileExistsError as e:
            logger.error(f'Local rollout samples at step {step} conflict with an existing attempt; '
                         f'the old records were preserved and the new records were not saved: {e}')
        except Exception as e:
            logger.warning(f'Failed to save local student rollout samples at step {step}: {e}')

    def _extract_reference_answers(self, batch: List[Dict]) -> List[str]:
        """Extract reference answer (last assistant message content) from each sample for OPSD."""
        refs = []
        for sample in batch:
            if isinstance(sample, dict) and 'messages' in sample:
                messages = sample['messages']
            elif isinstance(sample, (list, tuple)):
                messages = sample
            else:
                refs.append('')
                continue
            ref = ''
            for m in reversed(messages):
                if isinstance(m, dict) and m.get('role') == 'assistant':
                    c = m.get('content')
                    ref = c.strip() if isinstance(c, str) else ''
                    break
            refs.append(ref)
        return refs

    def _build_opsd_batch(self, rollout_batch: List[Dict], ref_answers: List[str]) -> List[Dict]:
        """Build OPSD samples: prompt + 'Here is a reference solution: <answer>' + student output."""
        OPSD_SUFFIX = '\nHere is a reference solution: '
        if len(ref_answers) != len(rollout_batch):
            raise ValueError('OPSD needs one reference answer per rollout sample.')
        out = []
        for i, sample in enumerate(rollout_batch):
            if 'messages' not in sample:
                raise ValueError(f'Invalid sample: {sample}')
            messages = [dict(message) for message in sample['messages']]
            ref = ref_answers[i]
            # Append reference to the last user message (right before assistant reply)
            user_idx = next((j for j in range(len(messages) - 1, -1, -1)
                             if messages[j].get('role') == 'user'), None)
            if user_idx is None or not isinstance(messages[user_idx].get('content'), str):
                raise ValueError('OPSD needs a text user message before the completion.')
            messages[user_idx]['content'] += OPSD_SUFFIX + ref
            out.append({**sample, 'messages': messages})
        return out

    def _encode_teacher_opsd_batch(self, opsd_batch: List[Dict]) -> tuple:
        """Encode OPSD batch and return (teacher_opsd_batch dict, opsd_completion_start list)."""
        template = self.template
        args = get_args()
        max_length = (getattr(self.args, 'mopd_max_sequence_length', None)
                      or template.max_length + self.max_completion_length)
        with self._template_context(template, max_length=max_length):
            encoded_list = [template.encode(d, return_length=True) for d in opsd_batch]
            opsd_completion_start = []
            for enc in encoded_list:
                labels = enc.get('labels')
                start = next((i for i, v in enumerate(labels) if v != -100), 0)
                opsd_completion_start.append(start)
            padding_to = get_padding_to(args)
            teacher_opsd_batch = to_device(
                template.data_collator(encoded_list, padding_to=padding_to), self.device)
        teacher_opsd_batch['num_samples'] = len(opsd_batch)
        return teacher_opsd_batch, opsd_completion_start

    def _pad_student_batch_to_teacher_length(
        self, encoded_batch: Dict, teacher_opsd_batch: Dict
    ) -> None:
        """Pad student encoded batch to teacher OPSD length so pipeline send/recv seq_length matches (fixes RoPE size mismatch when use_opsd)."""
        if get_args().pipeline_model_parallel_size == 1:
            # PP1 compares targets by causal remap; padding the student is both
            # unnecessary and would invalidate its packed text-position metadata.
            return
        pos_stu = encoded_batch.get('position_ids')
        pos_tea = teacher_opsd_batch.get('position_ids')
        L_stu = pos_stu.shape[-1]
        L_tea = pos_tea.shape[-1]
        if L_tea <= L_stu:
            raise ValueError("Teacher's input_ids is shorter than student's")
        pad_len = L_tea - L_stu
        if hasattr(self.template, 'processor') and getattr(self.template.processor, 'pad_token_id', None) is not None:
            pad_id = self.template.processor.pad_token_id
        elif getattr(self.template, 'tokenizer', None) is not None and getattr(self.template.tokenizer, 'pad_token_id', None) is not None:
            pad_id = self.template.tokenizer.pad_token_id
        start_pos = next((i for i, v in enumerate(encoded_batch['labels'][0]) if v != -100))
        for key in ('position_ids', 'input_ids', 'attention_mask', 'labels'):
            if key not in encoded_batch or encoded_batch[key] is None:
                continue
            t = encoded_batch[key]
            pad_val = 0 if key in ('position_ids', 'attention_mask') else (-100 if key == 'labels' else pad_id)
            pad_val = t.new_full((*t.shape[:-1], pad_len), pad_val, dtype=t.dtype)
            encoded_batch[key] = torch.cat([t, pad_val], dim=-1)

    def _get_random_num(self) -> float:
        """Generate a deterministic random number consistent across all processes.

        Uses an isolated Random instance with seed based on args.seed + step counter

        Returns:
            float: A random number in the range [0.0, 1.0).
        """
        seed = int(getattr(self.args, 'seed', 0))
        seed += int(self._step)
        rng = random.Random(seed)
        return rng.random()

    def _determine_data_source(self) -> DataSource:
        """Determine data source for current step based on GKD algorithm.

        GKD training mode selection logic:
        1. With probability lmbda: On-Policy (student generates)
        2. If seq_kd=True and not on-policy: Sequential KD (teacher generates)
        3. Otherwise: Off-Policy (use dataset responses)

        Returns:
            DataSource enum indicating which source to use.
        """
        random_num = self._get_random_num()

        if random_num < self.lmbda:
            # Mode 1: On-Policy learning, student model generates responses
            if self.use_vllm:
                return DataSource.STUDENT
            else:
                # If vLLM not enabled, fall back to dataset
                logger.warning_once('On-policy mode triggered but use_vllm=False. '
                                    'Falling back to dataset responses. Enable vLLM for on-policy generation.')
                return DataSource.DATASET
        elif self.seq_kd:
            # Mode 2: Sequential KD, teacher model generates responses
            # Note: Teacher generation is not implemented yet, use dataset
            logger.warning_once('seq_kd=True but teacher generation is not implemented in Megatron GKD yet. '
                                'Falling back to dataset responses.')
            return DataSource.DATASET
        else:
            # Mode 3: Off-Policy learning, use dataset responses
            return DataSource.DATASET

    def _init_resample_data_iterator(self):
        """Initialize an independent data iterator for dynamic resampling (lazy initialization).

        Builds a local-only cyclic dataloader from the stored train_dataset, avoiding
        build_train_valid_test_data_iterators which contains torch.distributed.broadcast
        and deadlocks when called mid-training-loop with EP>1.
        """
        from megatron.core import mpu
        args = get_args()

        train_dataset = self._resample_train_dataset
        data_collator = getattr(self, '_data_collator', None)

        batch_sampler = MegatronPretrainingRandomSampler(
            train_dataset,
            total_samples=len(train_dataset),
            consumed_samples=0,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding,
            shuffle=True,
        )
        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=0,
            pin_memory=False,
            collate_fn=data_collator,
        )

        def _cyclic(dl):
            while True:
                for batch in dl:
                    yield batch

        return iter(_cyclic(dataloader))


    def _encode_with_transient_retry(self, data: Dict, sleep_budget_remaining: float):
        """Encode ``data``; on a TRANSIENT I/O failure retry the ORIGINAL sample before giving up.

        Returns ``(success: bool, seconds_slept: float)``. Never raises for an encode failure —
        the caller decides whether to replace. Pure local I/O (``template.encode`` + ``time.sleep``);
        NO collective communication, so it preserves the DP rank-symmetry of the outer function.

        Retry classification. The ORDER is deliberate because ``UnidentifiedImageError`` subclasses
        ``OSError`` in Pillow, so a naive ``except OSError`` would wrongly retry an undecodable image:
          * ``UnidentifiedImageError`` (PIL): the bytes are not a valid image and will never
            decode on a re-read, so it is not retried.
          * ``OSError`` / ``IOError`` (including ``FileNotFoundError``): potentially transient
            filesystem I/O, retried with exponential backoff.
          * Any other exception (``ValueError`` / ``KeyError`` / truncation errors): malformed data
            or a logic error — retrying only burns wall-clock. NOT retried.

        Every failure EVENT (each failed attempt, retried or not) is recorded rank-locally in
        ``self._resample_failure_counts`` keyed by ``(domain, exception-type-name)`` and logged at
        WARNING with the domain, exception type, and retry outcome. Counting events (a sample retried
        twice then succeeding contributes 2) makes the end-of-run summary reflect raw I/O pressure,
        not only unrecoverable samples.
        """
        # Imported here (not module scope) so the method body is self-contained for the CPU-only
        # AST-lift test; PIL is always present in a VL training image, and imports are cached.
        try:
            from PIL import UnidentifiedImageError
        except Exception:  # pragma: no cover - PIL always present for VL; degrade to "no PIL class"
            UnidentifiedImageError = None

        template = self.template
        domain = data.get('domain') if isinstance(data, dict) else None
        max_retries = self.mopd_resample_retries
        seconds_slept = 0.0
        attempt = 0  # number of retries ALREADY performed (0 == original attempt)
        while True:
            try:
                encoded = template.encode(data)
                policy = getattr(self, '_mopd_domain_generation', {}) or {}
                prompt_cap = policy.get(domain, {}).get('max_prompt_tokens')
                if prompt_cap is not None:
                    labels = encoded.get('labels')
                    prompt_length = (len(encoded['input_ids']) if labels is None else next(
                        (i for i, label in enumerate(labels) if label != -100), len(encoded['input_ids'])))
                    if prompt_length > prompt_cap:
                        raise ValueError(
                            f'MOPD prompt budget exceeded: domain={domain!r}, '
                            f'actual={prompt_length}, cap={prompt_cap}; no image/token truncation')
                if attempt > 0:
                    logger.warning(
                        f'Encode succeeded on retry {attempt}/{max_retries} for a sample '
                        f'(domain={domain!r}) after a transient failure; keeping the ORIGINAL sample.')
                return True, seconds_slept
            except Exception as e:
                exc_name = type(e).__name__
                self._resample_failure_counts[(domain, exc_name)] += 1
                is_decode_failure = (UnidentifiedImageError is not None
                                     and isinstance(e, UnidentifiedImageError))
                retryable = isinstance(e, (OSError, IOError)) and not is_decode_failure
                if not retryable or attempt >= max_retries:
                    reason = ('non-retryable exception class' if not retryable
                              else f'retries exhausted ({attempt}/{max_retries})')
                    if getattr(self.args, 'strict', False):
                        raise RuntimeError(
                            f'MOPD strict encode failed for id={data.get("id", data.get("sid"))!r}, '
                            f'domain={domain!r}: {reason}') from e
                    logger.warning(
                        f'Encode failed (domain={domain!r}, exc={exc_name}: {e}); {reason} -> sample '
                        f'will be replaced within the SAME domain and serving-teacher route. retried={attempt}, '
                        f'slept={seconds_slept:.2f}s')
                    return False, seconds_slept
                # Transient and retries remain. Exponential backoff base * 2**attempt, clamped to the
                # remaining per-call sleep budget so a pathological burst cannot add hours.
                remaining = sleep_budget_remaining - seconds_slept
                if remaining <= 0:
                    if getattr(self.args, 'strict', False):
                        raise RuntimeError(
                            f'MOPD strict encode retry budget exhausted for '
                            f'id={data.get("id", data.get("sid"))!r}, domain={domain!r}') from e
                    logger.warning(
                        f'Encode failed (domain={domain!r}, exc={exc_name}: {e}); transient but the '
                        f'per-call retry budget is exhausted -> REPLACING without further retry. '
                        f'retried={attempt}, slept={seconds_slept:.2f}s')
                    return False, seconds_slept
                delay = min(self.mopd_resample_backoff_base_s * (2 ** attempt), remaining)
                logger.warning(
                    f'Encode failed (domain={domain!r}, exc={exc_name}: {e}); transient, retrying the '
                    f'ORIGINAL sample in {delay:.2f}s (attempt {attempt + 1}/{max_retries}).')
                time.sleep(delay)
                seconds_slept += delay
                attempt += 1

    def _resample_route_key(self, sample: Dict) -> tuple:
        return sample.get('domain'), self._get_teacher_idx_for_sample(sample)

    def _replacement_candidates(self, sample: Dict, limit: int):
        """Deterministic route-local draws; no independent unsaved iterator cursor."""
        dataset = self._resample_train_dataset
        if dataset is None:
            raise RuntimeError('Cannot replace a failed sample without the original training dataset.')
        if self._resample_route_indices is None:
            index = defaultdict(list)
            columns = getattr(dataset, 'column_names', ())
            if 'domain' in columns:
                domains = dataset['domain']
                rows = ({'domain': domain} for domain in domains)
            else:
                rows = iter(dataset)
            for i, row in enumerate(rows):
                index[self._resample_route_key(row)].append(i)
            self._resample_route_indices = index
        route = self._resample_route_key(sample)
        pool = self._resample_route_indices.get(route, ())
        if not pool:
            raise RuntimeError(f'No replacement candidates for route={route!r}; '
                               'refusing cross-domain/teacher substitution.')
        sample_key = sample.get('id') or sample.get('sid') or json.dumps(
            sample.get('messages', []), sort_keys=True, ensure_ascii=False)
        seed_material = (
            f'{self.args.seed}:{get_args().consumed_train_samples}:'
            f'{mpu.get_data_parallel_rank()}:{sample_key}')
        rng = random.Random(int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], 'big'))
        for index in rng.sample(pool, min(limit, len(pool))):
            yield dataset[index]

    def resample_encode_failed_inputs(self, inputs: List[Dict], max_resample_rounds: int = 10) -> List[Dict]:
        """Retry originals, then replace only within the same domain AND teacher route.

        The returned order/count are unchanged. No collectives occur here.
        Strict mode raises after in-place retries instead of substituting data.
        """
        result = []
        remaining_budget = self.mopd_resample_retry_budget_s
        for original in inputs:
            success, slept = self._encode_with_transient_retry(original, remaining_budget)
            remaining_budget = max(0.0, remaining_budget - slept)
            if success:
                result.append(original)
                continue
            route = self._resample_route_key(original)
            for replacement in self._replacement_candidates(original, max_resample_rounds):
                if self._resample_route_key(replacement) != route:
                    raise RuntimeError('Replacement index changed domain or serving-teacher identity.')
                success, slept = self._encode_with_transient_retry(replacement, remaining_budget)
                remaining_budget = max(0.0, remaining_budget - slept)
                if success:
                    result.append(replacement)
                    logger.warning(
                        f'MOPD encode replacement: route={route!r} '
                        f'original_id={original.get("id", original.get("sid"))!r} '
                        f'replacement_id={replacement.get("id", replacement.get("sid"))!r}.')
                    break
            else:
                raise RuntimeError(
                    f'Failed to collect a valid replacement within route={route!r} after at most '
                    f'{max_resample_rounds} candidates; refusing cross-domain/teacher substitution.')
        return result

    def _get_num_microbatches(self) -> int:
        """Get the number of microbatches for the current training step."""
        if getattr(self, '_eval_num_microbatches', None) is not None:
            return self._eval_num_microbatches
        from megatron.core.num_microbatches_calculator import get_num_microbatches
        return get_num_microbatches()


    @staticmethod
    def _loss_token_indices(labels: torch.Tensor) -> torch.Tensor:
        """SB offsets in BS loss-mask order, as required by the selected LM head."""
        batch_size, seq_len = labels.shape
        indices = (labels.reshape(-1) != -100).nonzero(as_tuple=False).squeeze(-1)
        if batch_size == 1:
            return indices
        return (indices % seq_len) * batch_size + indices // seq_len

    def _response_head_indices(self, labels: torch.Tensor) -> torch.Tensor:
        if labels is None:
            raise ValueError('mopd_response_only_logits requires supervised-position labels.')
        indices = self._loss_token_indices(labels)
        # A nonempty native GEMM keeps zero-supervision coordination slots on
        # the same collective/backward path. The loss still returns zero tokens.
        return indices if indices.numel() else indices.new_zeros(1)

    def _synchronize_dataset_batch(self, batch: List[Dict]) -> List[Dict]:
        """One authoritative recovered batch per TP/PP/CP replica group."""
        group = self._get_rollout_group()
        return self._gather_rollout_results(
            batch if torch.distributed.get_rank(group=group) == 0 else [])

    def _get_teacher_idx_for_sample(self, sample: Dict) -> int:
        """Map a raw sample's domain to its teacher index."""
        if not self.multi_teacher_mode:
            return 0
        domain = sample.get('domain') if isinstance(sample, dict) else None
        if domain is None:
            raise ValueError(
                'MOPD domain-routing requires every sample to have a "domain" field, but got a '
                f'sample without it. Sample keys: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}')
        if domain not in self.domain_teacher_mapping:
            raise ValueError(
                f'Domain "{domain}" not in domain_teacher_mapping. '
                f'Available: {list(self.domain_teacher_mapping.keys())}')
        return self.domain_teacher_mapping[domain]


    def _group_batch_by_teacher(self, global_batch: List[Dict], num_microbatches: int) -> List[tuple]:
        """Group a global batch by teacher so each microbatch is single-teacher.

        Returns a list of (raw_batch, teacher_idx) microbatches. Non-MOPD modes return the
        plain even split with teacher_idx=0 (byte-identical to the legacy path).

        Correctness contract (MOPD): the grouping must NOT drop real samples or duplicate
        samples into the gradient. That requires each domain's per-step sample count to be a
        multiple of micro_batch_size, and the resulting single-teacher microbatch counts to
        sum to exactly num_microbatches. If the per-step domain histogram cannot tile cleanly,
        fail fast (the data pipeline must supply domain-aligned global batches, e.g. via an
        interleave/sampler that yields whole-microbatch domain runs) rather than silently
        corrupting the effective training data.
        """
        assert num_microbatches > 0, f'num_microbatches must be > 0, got {num_microbatches}'
        assert len(global_batch) > 0 and len(global_batch) % num_microbatches == 0, (
            f'global_batch size ({len(global_batch)}) must be a positive multiple of '
            f'num_microbatches ({num_microbatches}).')
        micro_batch_size = len(global_batch) // num_microbatches
        if not self.multi_teacher_mode:
            return [(global_batch[i * micro_batch_size:(i + 1) * micro_batch_size], 0)
                    for i in range(num_microbatches)]

        groups = defaultdict(list)
        for sample in global_batch:
            groups[self._get_teacher_idx_for_sample(sample)].append(sample)

        # Each domain must contribute whole microbatches (no partial -> no padding/drop).
        bad = {tidx: len(s) for tidx, s in groups.items() if len(s) % micro_batch_size != 0}
        if bad:
            raise ValueError(
                f'MOPD domain routing: each teacher\'s per-step sample count must be a multiple '
                f'of micro_batch_size ({micro_batch_size}) so microbatches are single-teacher '
                f'without dropping or duplicating samples, but got counts {dict(sorted(((t, len(s)) for t, s in groups.items())))} '
                f'(offending: {bad}). Make the global batch domain-aligned (e.g. interleave the '
                f'domain datasets in whole-microbatch runs, or size global_batch so each domain '
                f'is a multiple of micro_batch_size).')

        microbatches: List[tuple] = []
        for teacher_idx, samples in sorted(groups.items()):
            n_full = len(samples) // micro_batch_size
            for i in range(n_full):
                microbatches.append((samples[i * micro_batch_size:(i + 1) * micro_batch_size], teacher_idx))

        if len(microbatches) != num_microbatches:
            raise ValueError(
                f'MOPD domain routing produced {len(microbatches)} single-teacher microbatches '
                f'but the step needs exactly {num_microbatches}. This means the per-step domain '
                f'histogram does not tile the microbatch grid. Align the data pipeline so each '
                f'global batch yields exactly num_microbatches single-teacher microbatches.')
        return microbatches

    @staticmethod
    def _build_domain_base_weights(domain_teacher_mapping: Dict[str, int],
                                   domain_loss_weights: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Per-DOMAIN base weight (keyed by domain NAME, NOT teacher_idx).

        Presence and renormalization are done at DOMAIN granularity so that domains sharing a
        teacher are never collapsed before presence is known — otherwise an absent shared
        domain's weight would still count in the renormalization. Each domain in
        ``domain_teacher_mapping`` gets its configured weight (from ``domain_loss_weights``) or
        1.0 if unspecified; NO summing across domains that share a teacher_idx. ``None`` (or an
        empty dict) yields a uniform weight of 1.0 per domain. These are BASE weights; the
        per-step renormalization over PRESENT domains happens in _compute_domain_loss_scales.
        """
        weights: Dict[str, float] = {}
        for domain in domain_teacher_mapping:
            weights[domain] = 1.0 if domain_loss_weights is None else float(domain_loss_weights.get(domain, 1.0))
        return weights

    @staticmethod
    def _compute_domain_loss_scales(global_sample_counts: Dict[str, float],
                                    global_token_counts: Dict[str, float],
                                    base_weights: Dict[str, float],
                                    mode: str,
                                    m_global: float) -> Dict[str, float]:
        """Per-DOMAIN gradient scale for the two MOPD domain normalization modes.

        Pure function (no collectives): callers pass the ALREADY-GLOBAL per-DOMAIN sample
        counts B_d and valid-token counts N_d (summed across DP). Keys are domain NAMES.
        Returns a dict domain -> scalar scale to multiply that microbatch's gradient-bearing
        base by.

        VERIFIED MEGATRON SCALING CHAIN (non-per-token / 2-tuple return path; CP==1, which
        is asserted for domain modes):
          * loss_func returns ``loss / cp_world_size``;
          * schedules.forward_step_calc_loss multiplies by ``cp_group_size`` (cancels the
            /cp) then divides by ``num_microbatches`` == M_local == GBS/(MBS*DP);
          * backward sums the M_local local microbatches;
          * DDP grad-sync scales by ``1/(DP*CP)`` (=1/DP at CP==1; applied through
            Megatron's ``target_gradient_scaling_factor``).
          Net effective objective:
            L_eff = (1 / M_global) * sum over ALL microbatches of the returned `loss`,
          with M_global = M_local * DP = GBS/MBS (== total #microbatches; == total #samples
          at MBS==1).
        To realize the two targets
            domain_sequence: L = sum_d q_d * (1/B_d) * sum_{i in d}(loss_sum_i / T_i)
            domain_token:    L = sum_d q_d * (sum_{i in d} loss_sum_i) / N_d
        the returned per-microbatch `loss` must be
            domain_sequence: (M_global * q_d / B_d) * mean_i     (mean_i = loss_sum_i / T_i)
            domain_token:    (M_global * q_d / N_d) * loss_sum_i
        i.e. scale_d = M_global * q_d / denom_d with denom = B_d (sequence) or N_d (token),
        applied respectively to the per-microbatch MEAN or the per-microbatch LOSS_SUM. The
        classic failure mode -- dividing by T_i again inside the microbatch -- would collapse
        domain_token into sequence weighting; here loss_sum_i is used raw for domain_token.

        q_d is renormalized over the DOMAINS present this step (>=1 sample AND >=1 valid token
        globally); absent domains never appear and never fail (so small smoke runs where a
        domain is missing still work). B_d already EXCLUDES zero-valid-token samples (counted
        upstream only when T_i > 0), so a mixed valid/zero-token domain is not biased down.
        """
        present = [d for d in base_weights
                   if global_sample_counts.get(d, 0) > 0 and global_token_counts.get(d, 0) > 0]
        wsum = sum(base_weights[d] for d in present)
        scales: Dict[str, float] = {}
        if wsum <= 0:
            return scales
        for d in present:
            q = base_weights[d] / wsum
            denom = global_token_counts[d] if mode == 'domain_token' else global_sample_counts[d]
            if denom > 0:
                scales[d] = float(m_global) * q / float(denom)
        return scales

    def _attach_domain_loss_scales(self, encoded_batches: List[Dict], grouped_microbatches: List[tuple],
                                   num_microbatches: int) -> None:
        """Compute per-microbatch ``mopd_loss_scale`` for the active domain mode.

        Runs once per step inside _replace_data_iterator, AFTER encoding and BEFORE any
        forward. Does ONE fixed-shape ``[2, num_domains]`` SUM all-reduce over the DP group
        (row 0 = per-DOMAIN sample counts, row 1 = per-DOMAIN valid-token counts) —
        rank-symmetric because ``num_domains`` and the domain index (sorted
        domain_teacher_mapping keys) are config-constant, and every DP rank enters this
        unconditionally every step when a domain mode is active (dataset/off-policy steps
        included). Mirrors the per-domain metric collective's device/dtype/rank-symmetry
        discipline.

        ``encoded_batches[i]`` and ``grouped_microbatches[i]`` are parallel (same append
        order); MBS==1 is enforced for domain modes, so each microbatch is exactly one sample
        and its domain is ``raw_batch[0]['domain']``. Domains missing/unmapped cannot reach
        here: ``_group_batch_by_teacher`` already routed every sample through
        ``_get_teacher_idx_for_sample`` (which raises for missing/unmapped domains in
        multi-teacher mode). A sample is counted into B_d only when its valid-token count > 0,
        so a zero-valid-token sample never inflates the domain_sequence denominator; it adds 0
        to N_d, which is a no-op for domain_token.
        """
        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        order = self._mopd_domain_order
        idx = self._mopd_domain_to_idx
        nd = len(order)
        # row 0: per-domain sample count (valid-token samples only); row 1: valid-token count.
        buf = torch.zeros(2, nd, device=device)
        for eb, (raw_batch, _teacher_idx) in zip(encoded_batches, grouped_microbatches):
            domain = raw_batch[0].get('domain') if raw_batch else None
            di = idx.get(domain)
            if di is None:
                continue
            labels = eb.get('labels')
            valid = int((labels != -100).sum().item()) if labels is not None else 0
            if valid > 0:
                buf[0, di] += 1.0
            buf[1, di] += float(valid)
        # Every DP rank contributes the same-shape tensor.
        try:
            dp_group = mpu.get_data_parallel_group()
        except Exception:
            dp_group = None
        if dp_group is not None:
            torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM, group=dp_group)
        global_sample_counts = {order[i]: float(buf[0, i].item()) for i in range(nd)}
        global_token_counts = {order[i]: float(buf[1, i].item()) for i in range(nd)}
        # M_global = M_local * DP = GBS/MBS (the effective global microbatch count the
        # Megatron chain divides by — see _compute_domain_loss_scales). num_microbatches is
        # the DP-LOCAL count and is identical on every rank.
        dp_size = mpu.get_data_parallel_world_size()
        m_global = float(int(num_microbatches) * int(dp_size))
        scales = self._compute_domain_loss_scales(
            global_sample_counts, global_token_counts,
            self._mopd_domain_base_weights or {}, self.mopd_loss_normalization, m_global)
        for eb, (raw_batch, _teacher_idx) in zip(encoded_batches, grouped_microbatches):
            domain = raw_batch[0].get('domain') if raw_batch else None
            eb['mopd_loss_scale'] = float(scales.get(domain, 0.0))

    def _replace_data_iterator(self, data_iterator, model):
        self._scheduled_num_microbatches = None
        # If the previous step's last microbatch
        # left a teacher body resident on GPU, offload it BEFORE the next step's rollout
        # runs (rollout drives vLLM but weight-sync into vLLM and student forward memory
        # both share the device, so we do not want a stray teacher body co-resident with
        # the peak-memory phases). Chosen this location because it is the single
        # deterministic point that runs after the previous step's optimizer step
        # completed AND before ANY next-step GPU work — regardless of whether the run
        # uses overlap_rollout_training or the plain rollout path. Safe no-op when the
        # flag is off or nothing is resident.
        self._offload_resident_teacher_if_any()

        num_microbatches = self._get_num_microbatches()

        # Determine data source once for the entire global batch
        data_source = self._determine_data_source()
        # Collect all micro-batches into a global batch
        global_batch = []
        for mb_i in range(num_microbatches):
            raw_batch = next(data_iterator)

            # Resample for encoding failed data when truncation_strategy is 'delete'
            if self.truncation_strategy == 'delete' and self._train_valid_test_dataset_provider is not None:
                raw_batch = self.resample_encode_failed_inputs(raw_batch)

            global_batch.extend(raw_batch)

        if data_source == DataSource.DATASET:
            # On-policy outputs are already gathered across model-parallel
            # replicas. Off-policy rows need the same authoritative view after
            # rank-local I/O retries/replacements, before teacher/student encode.
            global_batch = self._synchronize_dataset_batch(global_batch)

        if getattr(self.args, 'mopd_rollout_completion_chunk_size', 0):
            if data_source != DataSource.STUDENT:
                raise ValueError('Complete-response streaming requires a student-policy window.')
            stream_actor = (self._remote_teacher_actor
                            or getattr(self, '_local_completion_actor', None))
            if stream_actor is None:
                raise RuntimeError('Complete-response streaming was not initialized with the model.')
            iterator = stream_actor.prepare_completion_window(global_batch, data_source)
            self._step += 1
            return RerunDataIterator(iterator)

        # Overlap mode: vLLM keeps generating in a background thread; training consumes microbatches from a queue
        # as soon as each is ready (vLLM and training run in parallel).
        # Three-tier batching (MOPD-compatible):
        #   global batch  -- one optimizer step (grad accumulation target); size = GBS,
        #                    across ALL dp ranks. This rank owns len(global_batch) =
        #                    num_microbatches samples (MBS=1 -> one sample per microbatch).
        #   mini batch    -- rollout/train interleaving chunk expressed in GLOBAL samples;
        #                    size = overlap_rollout_mini_batch_size. On each rank the
        #                    per-rank chunk size = mini_bs // dp_size. Domains may MIX within
        #                    a chunk (vLLM is domain-agnostic and processes it as one request
        #                    for throughput).
        #   micro batch   -- MICRO_BATCH_SIZE=1, and in MOPD MUST be single-domain (one
        #                    resident teacher per microbatch). With MBS=1 every microbatch
        #                    is trivially single-domain — one sample = one domain.
        # The producer first stably sorts THIS RANK's samples by teacher_idx (rank-local,
        # no cross-rank comms — the sort only reorders which of this rank's own samples
        # this rank trains on, which does NOT change the sample set contributed to the
        # gradient; DP partitioning is fixed upstream in _get_local_rollout_batch). It
        # then walks the sorted per-rank list in steps of per_rank_chunk: roll out the
        # chunk as one vLLM request (mixed domains OK), then queue each rolled-out sample
        # as its own microbatch with its teacher_idx tag. Rolling out chunk-by-chunk keeps
        # training/vLLM overlap; sorting first minimizes teacher body-reuse switches
        # (see _ensure_teacher_resident: switching teachers pays a full PCIe offload+load).
        # In non-MOPD mode _get_teacher_idx_for_sample returns 0 for every sample, so the
        # sort is a no-op and all microbatches carry teacher_idx=0 — byte-identical to
        # the legacy per-sequence overlap path.
        if self.overlap_rollout_training and data_source == DataSource.STUDENT:
            micro_batch_size = len(global_batch) // num_microbatches  # == 1 in this recipe
            # overlap_rollout_mini_batch_size is a GLOBAL sample count (samples out of GBS).
            # Validated in argument post-init: divides global_batch_size and is a positive
            # multiple of micro_batch_size. Default = 128.
            mini_batch_size_global = int(getattr(self.args, 'overlap_rollout_mini_batch_size', 0) or 0)
            dp_size = mpu.get_data_parallel_world_size()
            gbs = len(global_batch) * dp_size  # per-rank len * dp = true global batch size
            if mini_batch_size_global <= 0 or mini_batch_size_global > gbs:
                # Fallback: entire global batch as one mini-batch (i.e. whole per-rank
                # global_batch as a single chunk). Legacy rollout granularity, still
                # correct, and safe if the arg is unset.
                mini_batch_size_global = gbs
            # Defensive asserts (arg post-init already validates the same thing; assert
            # here so the trainer fails fast if arg validation is ever bypassed).
            assert mini_batch_size_global % micro_batch_size == 0, (
                f'overlap_rollout_mini_batch_size ({mini_batch_size_global}) must be a '
                f'multiple of micro_batch_size ({micro_batch_size}).')
            assert gbs % mini_batch_size_global == 0, (
                f'global_batch_size ({gbs}) must be a multiple of '
                f'overlap_rollout_mini_batch_size ({mini_batch_size_global}).')
            assert mini_batch_size_global % dp_size == 0, (
                f'overlap_rollout_mini_batch_size ({mini_batch_size_global}) must be a '
                f'multiple of dp_size ({dp_size}) so each dp rank owns whole per-rank '
                f'chunks.')
            per_rank_chunk = mini_batch_size_global // dp_size
            assert per_rank_chunk >= 1 and len(global_batch) % per_rank_chunk == 0, (
                f'per_rank_chunk ({per_rank_chunk}) must be >=1 and divide per-rank '
                f'global_batch length ({len(global_batch)}).')

            # Rank-local sort by teacher_idx to cluster same-teacher microbatches
            # together, minimizing teacher-body offload/load switches at consumer time.
            # Python's sorted is stable, so the pre-sort order (data pipeline order) is
            # preserved WITHIN each teacher's group. Non-MOPD: teacher_idx=0 for every
            # sample -> sort is a no-op, order preserved exactly.
            global_batch_sorted = sorted(global_batch, key=self._get_teacher_idx_for_sample)

            mb_queue = queue.Queue(maxsize=num_microbatches)
            # Snapshot the current (pre-increment) training step. This is what the
            # produced microbatches are semantically "for": each is tagged with
            # `_rollout_weight_step = rollout_step` and metric logging keys off
            # this value. Consumer forward_step computes train_step from a
            # post-increment self._step (train_step = self._step - 1), so keeping
            # rollout_step at the PRE-increment value preserves the current
            # lag == 0 arithmetic on the overlap path (contract allows <= 1).
            rollout_step = int(self._step)

            # Main-thread lockstep weight sync. The producer thread MUST NEVER
            # call _move_model_to_vllm(): it broadcasts on the EP-PP-GROUP
            # (weight-sync bridge group), and if a second sync fires from a
            # background thread it interleaves with the main thread's training
            # collectives -> asymmetric collective counts across ranks -> NCCL
            # watchdog hang on EP-PP-GROUP.
            self._move_model_to_vllm()

            # Bump self._step and set _last_loaded_step BOTH before starting the
            # producer thread, so the producer observes a consistent pair:
            #
            #   Inside _generate_completions (rollout_mixin.py) the guard is
            #     if self._step != self._last_loaded_step: _move_model_to_vllm()
            #   If the counters were updated after producer.start(), the producer
            #   thread could read self._step == N+1 while _last_loaded_step was
            #   still N, triggering a SECOND, background-thread weight sync racing
            #   with the main thread. Whether it fired would depend on GIL
            #   scheduling, so ranks could disagree on whether to issue the
            #   collective — an asymmetric collective count that hangs the job.
            #
            # With both writes here, the producer thread reads a stable
            # self._step == _last_loaded_step == N+1 -> guard is False, no second
            # sync, no cross-rank asymmetry. Both writes are performed by the main
            # thread before producer.start(), so there is no producer/main race on
            # either counter.
            self._step += 1
            self._last_loaded_step = self._step

            # Sentinel used to propagate a producer-thread failure to the consumer.
            # Without it, any exception in _producer (rollout / gather / encode / assert)
            # would kill the daemon thread silently and leave the consumer blocked
            # forever on mb_queue.get() — a false hang that masks the real error.
            class _ProducerFailed:

                def __init__(self, exc):
                    self.exc = exc

            def _producer():
                try:
                    rollout_log_samples = []
                    total_queued = 0
                    num_chunks = len(global_batch_sorted) // per_rank_chunk
                    for mb_i in range(num_chunks):
                        chunk_start = mb_i * per_rank_chunk
                        chunk_end = chunk_start + per_rank_chunk
                        chunk_raw = global_batch_sorted[chunk_start:chunk_end]
                        # Capture teacher_idx BEFORE rollout on the raw sample. The rolled-out
                        # sample dicts also still carry `domain` (rollout_mixin.py:
                        # _postprocess_rollout_outputs merges outputs into the input dict
                        # in-place, so non-touched keys like `domain` survive), but taking
                        # the tag from the raw sample here is explicit and independent of
                        # rollout-side field handling — and preserves positional
                        # correspondence between chunk_raw and mini_rollouts (which
                        # _gather_rollout_results reconstructs in rank-order across the
                        # rollout group, matching the pre-split chunk order).
                        chunk_teacher_ids = [self._get_teacher_idx_for_sample(s) for s in chunk_raw]
                        # vLLM is domain-agnostic: one rollout request per chunk, domains
                        # may mix. Reuses the same helpers as the non-overlap path.
                        local_chunk = self._get_local_rollout_batch(chunk_raw)
                        local_chunk = self._generate_completions(local_chunk)
                        chunk_rollouts = self._gather_rollout_results(local_chunk)
                        assert len(chunk_rollouts) == len(chunk_raw), (
                            f'rollout gather returned {len(chunk_rollouts)} samples but chunk '
                            f'had {len(chunk_raw)} — order/count invariant broken.')
                        rollout_log_samples.extend([{
                            **sample, 'messages': [dict(message) for message in sample.get('messages', [])]
                        } for sample in chunk_rollouts])
                        # MBS=1: each rolled-out sample becomes its own single-sample
                        # microbatch. No _group_batch_by_teacher call needed — with MBS=1
                        # each microbatch is trivially single-domain, and its teacher_idx
                        # was captured from the raw sample above (positional match).
                        for j, sample in enumerate(chunk_rollouts):
                            teacher_idx = chunk_teacher_ids[j]
                            raw_micro = [sample]
                            encoded_batch_i = self._encode_batch(raw_micro)
                            encoded_batch_i['data_source'] = data_source
                            # OPSD is documented as not-combined-with-MOPD; guard identically
                            # to the non-overlap path.
                            if self.use_opsd and not self.multi_teacher_mode:
                                ref_answers_i = self._extract_reference_answers(raw_micro)
                                opsd_batch_i = self._build_opsd_batch(raw_micro, ref_answers_i)
                                teacher_opsd_batch_i, opsd_completion_start_i = \
                                    self._encode_teacher_opsd_batch(opsd_batch_i)
                                encoded_batch_i['teacher_opsd_batch'] = teacher_opsd_batch_i
                                encoded_batch_i['opsd_completion_start'] = opsd_completion_start_i
                                self._pad_student_batch_to_teacher_length(encoded_batch_i, teacher_opsd_batch_i)
                            if not self.lazy_teacher_logits:
                                encoded_batch_i['teacher_logits'] = self._compute_teacher_logits_for_one_batch(
                                    encoded_batch_i, vp_stage=None)
                            # Tag the rollout-side weight version that produced this
                            # micro-batch so forward_step can compute weight_lag against
                            # the training step that consumes it (lag <= 1).
                            encoded_batch_i['_rollout_weight_step'] = rollout_step
                            # MOPD domain routing: tell forward_step which teacher this
                            # microbatch maps to (0 in non-MOPD mode). Mirrors the
                            # non-overlap path's `encoded_batch['teacher_idx'] = teacher_idx`.
                            encoded_batch_i['teacher_idx'] = teacher_idx
                            mb_queue.put(encoded_batch_i)
                            total_queued += 1
                    # Consumer loops exactly num_microbatches times; producer MUST match.
                    # With MBS=1, len(global_batch) == num_microbatches and per_rank_chunk
                    # divides len(global_batch), so sum over chunks of per_rank_chunk equals
                    # len(global_batch) == num_microbatches.
                    assert total_queued == num_microbatches, (
                        f'overlap producer queued {total_queued} microbatches but consumer '
                        f'expects {num_microbatches}.')
                    self._log_student_rollout_samples(rollout_log_samples, step=rollout_step)
                except BaseException as e:
                    # A producer failure must surface as a real error in the consumer, not
                    # a silent daemon-thread death that hangs the step forever. Push the
                    # exception through the queue so _consumer_iter re-raises it.
                    #   - Partial failure (queued < num_microbatches): the queue has free
                    #     capacity, so the sentinel is delivered and the consumer re-raises.
                    #   - Post-loop failure (all microbatches already queued): the queue is full AND the consumer has
                    #     already drained every valid item and finished — the step itself
                    #     succeeded. put_nowait avoids blocking forever on the full queue;
                    #     dropping the sentinel is correct because there is no consumer left
                    #     to receive it.
                    logger.error(f'overlap rollout producer thread failed: {e!r}')
                    try:
                        mb_queue.put_nowait(_ProducerFailed(e))
                    except queue.Full:
                        pass

            def _consumer_iter():
                for _ in range(num_microbatches):
                    item = mb_queue.get()
                    if isinstance(item, _ProducerFailed):
                        raise RuntimeError(
                            'overlap rollout producer thread failed; re-raising in the '
                            'consumer so the step aborts instead of hanging') from item.exc
                    yield item

            producer = threading.Thread(target=_producer, daemon=True)
            producer.start()
            # NOTE: self._step was already incremented BEFORE producer.start()
            # above, so the producer thread sees a stable counter and does not
            # re-trigger _move_model_to_vllm from the background thread.
            return RerunDataIterator(_consumer_iter())

        # On-policy mode: generate completions for the entire global batch at once
        ref_answers = None
        if data_source == DataSource.STUDENT:
            if self.use_opsd:
                ref_answers = self._extract_reference_answers(global_batch)
            local_batch = self._get_local_rollout_batch(global_batch)
            local_batch = self._generate_completions(local_batch)
            global_batch = self._gather_rollout_results(local_batch)
            self._log_student_rollout_samples(global_batch)
        elif data_source == DataSource.TEACHER:
            logger.warning_once('Teacher mode triggered but teacher generation is not implemented in Megatron GKD yet. '
                                'Falling back to dataset responses.')
        remote = getattr(self, '_remote_teacher_actor', None)
        if remote is not None:
            if data_source != DataSource.STUDENT:
                raise ValueError('Remote teacher streaming requires a current-policy rollout window.')
            global_batch = remote.rebalance(global_batch)

        # Split global batch back into micro-batches for encoding. In MOPD domain-routing
        # mode, group by teacher so each microbatch is single-teacher and tag teacher_idx;
        # otherwise this is the plain even split (teacher_idx 0), byte-identical to legacy.
        encoded_batches = []
        micro_batch_size = len(global_batch) // num_microbatches
        grouped_microbatches = self._group_batch_by_teacher(global_batch, num_microbatches)
        # Snapshot the student weight version visible to vLLM at the moment
        # rollout finished. For the on-policy path the sync happened inside
        # _generate_completions, so `_last_loaded_step` already equals the
        # current training step (lag = 0). For dataset/teacher data sources
        # there is no rollout-side weight version, so emit -1 to signal
        # "not applicable" downstream.
        if data_source == DataSource.STUDENT:
            rollout_weight_step_for_step = int(getattr(self, '_last_loaded_step', -1))
        else:
            rollout_weight_step_for_step = -1
        with remote.encoding_guard() if remote is not None else nullcontext():
            for i, (raw_batch, teacher_idx) in enumerate(grouped_microbatches):
                start_idx = i * micro_batch_size
                end_idx = start_idx + micro_batch_size
                encoded_batch = self._encode_batch(raw_batch)
                # Store data_source for conditional SFT loss in loss_func
                encoded_batch['data_source'] = data_source
                encoded_batch['_rollout_weight_step'] = rollout_weight_step_for_step
                # MOPD domain routing: record which teacher this microbatch maps to.
                encoded_batch['teacher_idx'] = teacher_idx
                # OPSD: build teacher batch with reference solution in prompt (lazy_teacher_logits path only).
                # OPSD ref_answers slicing is positional and only valid for the non-grouped path
                # (MOPD reorders samples by domain); OPSD is not combined with MOPD routing.
                if self.use_opsd and ref_answers is not None and not self.multi_teacher_mode:
                    ref_answers_mb = ref_answers[start_idx:end_idx]
                    opsd_batch = self._build_opsd_batch(raw_batch, ref_answers_mb)
                    teacher_opsd_batch, opsd_completion_start = self._encode_teacher_opsd_batch(opsd_batch)
                    encoded_batch['teacher_opsd_batch'] = teacher_opsd_batch
                    encoded_batch['opsd_completion_start'] = opsd_completion_start
                    self._pad_student_batch_to_teacher_length(encoded_batch, teacher_opsd_batch)
                encoded_batches.append(encoded_batch)

        # MOPD loss-normalization domain modes: stamp each microbatch with its scalar
        # gradient scale via ONE fixed-shape DP all-reduce of per-domain sample/token
        # counts. MUST run here — after all encoded_batches exist, before ANY forward — so
        # the collective completes inside _replace_data_iterator. Gated only on the
        # rank-constant `mopd_domain_loss_mode`, so every DP rank enters it every step
        # (rank-symmetric). Inert (no-op) when the mode is off -> byte-identical legacy path.
        if self.mopd_domain_loss_mode:
            self._attach_domain_loss_scales(encoded_batches, grouped_microbatches, num_microbatches)

        if not self.lazy_teacher_logits:
            self._compute_teacher_logits(encoded_batches)

        if remote is None:
            encoded_batches = self._align_teacher_microbatches(encoded_batches)
            iterator = iter(encoded_batches)
        else:
            iterator = remote.prepare_window(grouped_microbatches, encoded_batches, rollout_weight_step_for_step)
        # Increment step counter (used for deterministic random and weight sync)
        self._step += 1
        return RerunDataIterator(iterator)

    def _new_coordination_dummy(self, teacher_idx: int) -> Dict:
        if self._coordination_dummy_encoded is None:
            encoded = self.template.encode({
                'messages': [{'role': 'user', 'content': '.'}, {'role': 'assistant', 'content': '.'}]
            }, return_length=True)
            batch = self.template.data_collator([encoded], padding_to=get_padding_to(get_args()))
            batch['labels'].fill_(-100)
            batch['num_samples'] = 1
            self._coordination_dummy_encoded = batch
        batch = {k: v.clone() if isinstance(v, torch.Tensor) else v
                 for k, v in self._coordination_dummy_encoded.items()}
        batch.update(teacher_idx=teacher_idx, data_source=DataSource.DATASET,
                     _mopd_coordination_padding=True, mopd_loss_scale=0.)
        return batch

    def _align_teacher_microbatches(self, batches: List[Dict]) -> List[Dict]:
        """Align whole teacher phases without resampling or caching full-vocab targets.

        Real samples remain on their original DP stream. Extra execution slots
        carry no supervised tokens. Native token-mean gradients need no change;
        sequence/domain means compensate the schedule's larger divisor. The
        microbatch calculator, optimizer updates and LR sample count stay real.
        """
        if (not self.multi_teacher_mode or mpu.get_expert_model_parallel_world_size() <= 1
                or not any(self._teacher_is_moe.values())):
            return batches
        # An auxiliary router objective or independent SFT stream would also
        # need padding-aware semantics. Preserve those recipes via teacher-only
        # rounds instead; pure MOPD has neither.
        args = get_args()
        if (self.use_sft_loss or self.sft_alpha > 0 or args.moe_aux_loss_coeff != 0
                or getattr(args, 'moe_z_loss_coeff', None) not in (None, 0)
                or getattr(args, 'moe_router_enable_expert_bias', False)):
            return batches
        if getattr(get_args(), 'moe_expert_capacity_factor', None) is not None:
            raise ValueError('Coordinated MOPD teacher phases require dropless MoE.')
        buckets = [[] for _ in self.teacher_groups]
        for batch in batches:
            buckets[int(batch['teacher_idx'])].append(batch)
        maxima = torch.tensor([len(bucket) for bucket in buckets], dtype=torch.int64, device=self.device)
        torch.distributed.all_reduce(maxima, op=torch.distributed.ReduceOp.MAX)
        lengths = maxima.tolist()
        real_count, scheduled_count = len(batches), sum(lengths)
        scale = scheduled_count / real_count
        result = []
        for teacher_idx, count in enumerate(lengths):
            for slot in range(count):
                batch = (buckets[teacher_idx][slot] if slot < len(buckets[teacher_idx])
                         else self._new_coordination_dummy(teacher_idx))
                batch['_mopd_teacher_scheduled'] = True
                batch['_mopd_coordination_scale'] = scale
                result.append(batch)
        self._scheduled_num_microbatches = scheduled_count
        return result

    def train_step(self, forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config, *args,
                   **kwargs):
        # Native Megatron owns the resumed iteration and sets it before every
        # train_step. Keep the OPD RNG/profile clock aligned without changing
        # _last_loaded_step: a fresh server still needs its first weight sync.
        self._step = int(get_args().curr_iteration)
        stream_actor = (getattr(self, '_remote_teacher_actor', None)
                        or getattr(self, '_local_completion_actor', None))
        if stream_actor is None:
            result = super().train_step(forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config,
                                       *args, **kwargs)
        else:
            # Keep ONE native full-window schedule. It already zeroes once,
            # reduces once and updates once. Sequence mode divides by the full
            # microbatch count; global_token accumulates raw sums, then native
            # finalize_model_grads divides by the full DP window's token count.
            # Streaming only supplies inputs; chunk boundaries never rescale.
            if args:
                schedule, remaining = args[0], args[1:]
            else:
                schedule, remaining = kwargs.pop('forward_backward_func'), ()

            def finish_window(*call_args, **call_kwargs):
                output = schedule(*call_args, **call_kwargs)
                stream_actor.finish_before_update()
                return output

            try:
                result = super().train_step(
                    forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config,
                    finish_window, *remaining, **kwargs)
            except BaseException as failure:
                try:
                    stream_actor.abort()
                except Exception as abort_error:
                    failure.add_note(f'MOPD streaming cleanup also failed: {abort_error}')
                raise
        if stream_actor is not None and getattr(self, 'calculate_per_token_loss', False) and 'jsd loss' in result[0]:
            # Native train_step has already reduced this sum/count across the
            # entire DP window. Keep legacy rank-local sequence-mean `loss`
            # intact and expose the optimized objective without another reduce.
            result[0]['loss/token_mean'] = result[0]['jsd loss']
        # Ratio diagnostics must be formed AFTER the microbatch aggregation. The native
        # path averages each of these per-microbatch ratios with a plain mean, which is
        # not the ratio of the means logged beside them (see _DIVERGENCE_RATIOS / _SUM_RATIOS
        # / _POOLED_NORMS). Re-derive them from the already-reduced components: pure scalar
        # arithmetic, no collective, no memory. `result[0]` is {} on a non-last pipeline
        # stage, hence the guard.
        #
        # Resolved through ``globals()`` rather than called by bare name: this method is
        # also executed by the source-extraction tests
        # (tests/mopd/source_test_utils.load_symbols / gkd_symbols), whose namespace holds
        # only the names handed to it. A bare reference would be a NameError there. During
        # training it is a module global, so the recompute always runs; the extracted form
        # skips a logging-only step it does not exercise.
        _recompute = globals().get('recompute_logged_diagnostics')
        if _recompute is not None:
            try:
                _recompute(result[0])
            except Exception as e:  # pragma: no cover - defensive, never break training
                _log = globals().get('logger')
                if _log is not None:
                    _log.warning(f'divergence ratio recompute skipped at step {self._step}: {e}')
        return result



    def _compute_teacher_logits_for_one_batch(
        self, encoded_batch: Dict, vp_stage: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """Compute teacher logits for a single encoded batch (one microbatch). Used for lazy teacher forward in forward_step to save memory.
        When use_opsd is True, uses teacher_opsd_batch (prompt + reference solution + student output) and maps teacher logits back to student sequence length."""
        # MOPD domain routing: select the teacher this microbatch was routed to. Non-MOPD
        # modes use teacher 0 (self.teacher_models, unchanged).
        teacher_idx = int(encoded_batch.get('teacher_idx', 0)) if self.multi_teacher_mode else 0
        # Deep copy to avoid modifying original batch
        teacher_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in encoded_batch.items()}
        teacher_batch.pop('data_source', None)
        teacher_batch.pop('teacher_idx', None)
        for key in ('_mopd_teacher_scheduled', '_mopd_coordination_scale', '_mopd_coordination_padding',
                    '_mopd_router_replay'):
            teacher_batch.pop(key, None)

        use_opsd = self.use_opsd and 'teacher_opsd_batch' in encoded_batch
        if use_opsd:
            teacher_opsd_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in encoded_batch['teacher_opsd_batch'].items()}
            opsd_completion_start = encoded_batch['opsd_completion_start']  # list of int
            teacher_batch_for_forward = teacher_opsd_batch
            labels_student = encoded_batch['labels']  # [B, L_student]
        else:
            teacher_batch_for_forward = teacher_batch

        teacher_data = self._prepare_batch(teacher_batch_for_forward)
        teacher_data.pop('loss_scale', None)
        teacher_labels = teacher_data.pop('labels', None)
        if self.mopd_response_only_logits:
            if use_opsd:
                raise ValueError('mopd_response_only_logits does not support OPSD teacher prompt remapping.')
            teacher_data['valid_token_indices'] = self._response_head_indices(teacher_labels)
            if getattr(self, 'mopd_head_chunk_size', 0):
                teacher_data['mopd_return_hidden_states'] = True
                teacher_data['fp32_output'] = False
        # On the MOPD routing path with offload enabled, keep
        # the teacher body resident across consecutive same-teacher microbatches instead
        # of the per-microbatch load/offload dance in `load_teacher_model_context`. The
        # body is offloaded at the START of the next step's `_replace_data_iterator`, so
        # rollout / student memory is not held during the swap window. When the flag is
        # off (or the requirements aren't met), fall through to the legacy context — the
        # existing behavior is byte-identical.
        _use_body_reuse = (self.teacher_body_reuse
                           and self.offload_teacher_model
                           and self.multi_teacher_mode)
        teacher_logits = None
        rounds = [teacher_idx] if encoded_batch.get('_mopd_teacher_scheduled') else self._teacher_forward_rounds(teacher_idx)
        for forward_idx in rounds:
            teacher_group = self.teacher_groups[forward_idx] if self.multi_teacher_mode else self.teacher_models
            teacher_model = teacher_group[vp_stage or 0]
            selected = forward_idx == teacher_idx
            forward_data = teacher_data if selected else self._get_teacher_dummy_batch(vp_stage)
            if _use_body_reuse:
                self._ensure_teacher_resident(forward_idx)
                load_context = nullcontext()
            else:
                load_context = self.load_teacher_model_context(forward_idx)
            with load_context, self._teacher_args_context(forward_idx), torch.no_grad():
                with nullcontext():
                    output = forward_step_helper(teacher_model, forward_data)
                if selected and output is not None:
                    if getattr(self, 'mopd_head_chunk_size', 0):
                        from .streamed_distillation import snapshot_teacher_head
                        native_teacher = unwrap_model(teacher_model)
                        native_teacher = getattr(native_teacher, 'language_model', native_teacher)
                        teacher_logits = snapshot_teacher_head(native_teacher, output)
                    else:
                        teacher_logits = output.detach()
                # A non-selected rank participates in MoE collectives but never
                # supplies a target or changes the student's microbatch/count.
                del output

        if use_opsd and teacher_logits is not None:
            # Map teacher logits (on OPSD sequence) to student sequence length: only completion positions
            B, L_opsd, V = teacher_logits.shape
            L_student = labels_student.shape[1]
            device, dtype = teacher_logits.device, teacher_logits.dtype
            teacher_logits_full = torch.zeros(B, L_student, V, device=device, dtype=dtype)
            for b in range(B):
                comp_mask = labels_student[b] != -100
                comp_indices = comp_mask.nonzero(as_tuple=True)[0]
                start_opsd = opsd_completion_start[b] if isinstance(opsd_completion_start[b], int) else int(opsd_completion_start[b])
                num_comp = comp_indices.numel()
                if num_comp > 0:
                    if start_opsd < 1 or int(comp_indices[0]) < 1 or start_opsd - 1 + num_comp > L_opsd:
                        raise RuntimeError(f'OPSD causal alignment is invalid on row {b}: '
                                           f'teacher_start={start_opsd}, teacher_length={L_opsd}, '
                                           f'completion_tokens={num_comp}.')
                    teacher_logits_full[b, comp_indices - 1, :] = \
                        teacher_logits[b, start_opsd - 1:start_opsd - 1 + num_comp, :]
            teacher_logits = teacher_logits_full

        return teacher_logits

    def _teacher_forward_rounds(self, teacher_idx: int) -> List[int]:
        """Coordinate teacher-only rounds; leave student samples and loss weights intact.

        EP collectives must use one teacher identity. Synchronizing the union
        across training ranks also covers TP groups crossing an EP boundary.
        Local sorting keeps the common case to one round. Mixed slots add tiny
        no-gradient forwards, not padded student microbatches or cached full
        step teacher logits.
        """
        if not self.multi_teacher_mode or mpu.get_expert_model_parallel_world_size() <= 1:
            return [teacher_idx]
        local = torch.tensor(
            [teacher_idx, int(self._teacher_is_moe.get(teacher_idx, True))],
            dtype=torch.int64, device=self.device)
        world_size = torch.distributed.get_world_size()
        gathered = torch.empty(world_size * 2, dtype=local.dtype, device=local.device)
        torch.distributed.all_gather_into_tensor(gathered, local)
        identities = gathered.view(world_size, 2)
        if not bool(identities[:, 1].any()):
            return [teacher_idx]  # all dense: no expert collectives
        rounds = sorted(set(identities[:, 0].tolist()))
        if len(rounds) > 1 and getattr(get_args(), 'moe_expert_capacity_factor', None) is not None:
            raise ValueError('Coordinated MOPD teacher rounds require dropless MoE '
                             '(moe_expert_capacity_factor=None); dummy tokens must not consume '
                             'a shared expert capacity budget.')
        return rounds

    def _get_teacher_dummy_batch(self, vp_stage=None) -> Dict:
        """A minimal valid teacher input, never consumed by the student loss."""
        if self._teacher_dummy_batch is None:
            encoded = self.template.encode({
                'messages': [{'role': 'user', 'content': '.'}, {'role': 'assistant', 'content': '.'}]
            }, return_length=True)
            batch = self.template.data_collator([encoded], padding_to=get_padding_to(get_args()))
            batch['num_samples'] = 1
            batch = self._prepare_batch(batch, vp_stage=vp_stage)
            batch.pop('labels', None)
            batch.pop('loss_scale', None)
            self._teacher_dummy_batch = batch
        return dict(self._teacher_dummy_batch)

    def _compute_teacher_logits(self, encoded_batches: List[Dict], vp_stage: Optional[int] = None) -> None:
        """Compute teacher logits for all encoded batches (legacy / batch path). Prefer per-microbatch in forward_step to save memory."""
        for encoded_batch in encoded_batches:
            encoded_batch['teacher_logits'] = self._compute_teacher_logits_for_one_batch(encoded_batch, vp_stage)

    def _loss_aligned_labels(self, encoded_batch: Dict) -> torch.Tensor:
        """Return the labels in the SAME alignment the student loss uses.

        For `task_type == 'causal_lm'`, `get_batch_on_this_tp_rank` (inside `_prepare_batch`)
        rolls labels by -1 so each position predicts the next token; the student loss masks
        with those rolled labels. The fused-teacher path must select teacher rows with the
        identical mask, so we replicate that roll here (no-op for non-causal_lm task types).
        """
        labels = encoded_batch['labels']
        if getattr(get_args(), 'task_type', 'causal_lm') == 'causal_lm':
            return torch.roll(labels, -1, dims=-1)
        return labels

    def _forward_one_teacher_masked_rows(
        self, teacher_idx: int, encoded_batch: Dict, vp_stage: Optional[int], mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Run a single teacher's forward and return ONLY the masked completion rows
        `[N, V]` (N = number of valid tokens), in the same flattened mask order as
        `teacher_logits[mask]` for `mask = labels != -100`. Returns None on a non-last
        PP rank.

        Memory: this never materializes a dense `[B, L_student, V]` tensor. For OPSD it
        gathers each sample's completion slice and concatenates; for the non-OPSD case it
        indexes the forward output with the mask directly. Only the forward output
        `[B, L, V]` is transiently held, then reduced to `[N, V]` and the dense tensor is
        released by the caller.
        """
        teacher_model = self.teacher_groups[teacher_idx][vp_stage or 0]
        teacher_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in encoded_batch.items()}
        teacher_batch.pop('data_source', None)

        use_opsd = self.use_opsd and 'teacher_opsd_batch' in encoded_batch
        if use_opsd:
            teacher_opsd_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v
                                  for k, v in encoded_batch['teacher_opsd_batch'].items()}
            opsd_completion_start = encoded_batch['opsd_completion_start']  # list of int
            teacher_batch_for_forward = teacher_opsd_batch
            # OPSD completion positions (opsd_completion_start) index the UNROLLED labels,
            # so this path uses the unrolled labels to count/select completion rows. NOTE:
            # OPSD is rejected for two-teacher fusion in argument validation (the rolled
            # student-loss alignment vs unrolled OPSD indexing is not safely reconcilable in
            # this version), so this branch is single-teacher-only in practice.
            labels_student = encoded_batch['labels']  # [B, L_student] (unrolled)
        else:
            teacher_batch_for_forward = teacher_batch

        teacher_data = self._prepare_batch(teacher_batch_for_forward)
        teacher_data.pop('loss_scale', None)
        teacher_data.pop('labels', None)
        with self._teacher_args_context(teacher_idx), torch.no_grad():
            with nullcontext():
                teacher_logits = forward_step_helper(teacher_model, teacher_data)
            if teacher_logits is not None:
                teacher_logits = teacher_logits.detach()

        if teacher_logits is None:
            return None

        if use_opsd:
            # Gather ONLY completion rows per sample and concatenate to [N, V]. The
            # per-sample completion slice teacher_logits[b, start:start+num_comp] lines up
            # row-for-row with that sample's `labels[b] != -100` positions (ascending), so
            # the concatenation equals teacher_logits_full[mask] WITHOUT ever allocating
            # the dense [B, L_student, V] tensor.
            rows = []
            B = teacher_logits.shape[0]
            for b in range(B):
                comp_indices = (labels_student[b] != -100).nonzero(as_tuple=True)[0]
                num_comp = comp_indices.numel()
                if num_comp > 0:
                    start_opsd = opsd_completion_start[b] if isinstance(opsd_completion_start[b], int) \
                        else int(opsd_completion_start[b])
                    rows.append(teacher_logits[b, start_opsd:start_opsd + num_comp, :])
            if rows:
                masked_rows = torch.cat(rows, dim=0)  # [N, V]
            else:
                masked_rows = teacher_logits.new_zeros((0, teacher_logits.shape[-1]))
            del teacher_logits, rows
            return masked_rows

        # Non-OPSD: forward output is already aligned with the student sequence; index
        # directly with the mask to get [N, V] and let the caller release the dense tensor.
        masked_rows = teacher_logits[mask]
        del teacher_logits
        return masked_rows

    def _compute_fused_teacher_log_probs_for_one_batch(
        self, encoded_batch: Dict, vp_stage: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """Two-teacher probability-space fusion. Returns a MASKED fused teacher
        log-prob target of shape `[N, partition_vocab_size]` (N = number of valid
        completion tokens, mask order matching `labels != -100`), or None on a
        non-last PP rank.

        Memory contract: teachers are forwarded SEQUENTIALLY. Each teacher's dense
        `[B, L, V]` logits are masked to `[N, V]`, converted to vocab-parallel
        log-probs, accumulated into a single running fused log-prob tensor, then the
        dense tensor is discarded before the next teacher is loaded. At most one dense
        teacher tensor exists at any time.

        Fusion modes:
          - 'prob_avg': fused = logsumexp_i(log w_i + log_softmax(z_i / T))  (probability mixture)
          - 'logit_avg': fused = log_softmax(sum_i w_i * (z_i / T))          (log-linear pool)

        Temperature is applied here (once, per teacher); the loss-side branch must NOT
        re-divide. Teacher label smoothing is applied once AFTER fusion.
        """
        # Build the mask from the SAME loss-aligned (rolled for causal_lm) labels the
        # student loss masks with, so fused-teacher rows line up with student rows.
        labels = self._loss_aligned_labels(encoded_batch)  # [B, L_student]
        # Defense-in-depth: two-teacher fusion is not valid under context parallelism (the
        # mask here is full-sequence while teacher logits are CP-sliced in _prepare_batch).
        # Argument validation already rejects CP>1 for two teachers; assert here too in case
        # the arg-time check was bypassed.
        if int(getattr(get_args(), 'context_parallel_size', 1) or 1) > 1:
            raise ValueError(
                'Two-teacher fused distillation does not support context_parallel_size > 1 '
                '(full-sequence mask vs CP-sliced teacher logits). Use context_parallel_size=1.')
        # Defense-in-depth: PP>1 is rejected at argument validation (the sequential fused
        # forward returns None on non-last PP ranks, skipping later teachers and deadlocking
        # the pipeline); assert here too.
        if int(getattr(get_args(), 'pipeline_model_parallel_size', 1) or 1) > 1:
            raise ValueError(
                'Two-teacher fused distillation does not support pipeline_model_parallel_size > 1 '
                '(non-last PP ranks would skip later teacher forwards and deadlock). Use '
                'pipeline_model_parallel_size=1.')
        # Defense-in-depth: OPSD is rejected for two-teacher fusion at argument validation
        # (unrolled OPSD completion indexing vs rolled student-loss labels cannot be safely
        # aligned in this version); assert here too.
        if self.use_opsd and 'teacher_opsd_batch' in encoded_batch:
            raise ValueError(
                'Two-teacher fused distillation does not support use_opsd (OPSD completion '
                'mapping vs rolled student-loss labels misalign). Disable use_opsd.')
        mask = labels != -100
        loss_temperature = self.loss_temperature
        weights = self.teacher_fusion_weights
        offload = self.offload_teacher_model

        prob_avg_acc = None    # running fused log-prob accumulator [N, V] for prob_avg
        logit_avg_acc = None   # running weighted logit accumulator [N, V] for logit_avg

        for teacher_idx in range(self.num_teachers):
            if offload:
                with nullcontext():
                    self._load_one_teacher_to_gpu(teacher_idx)
            try:
                t_rows = self._forward_one_teacher_masked_rows(teacher_idx, encoded_batch, vp_stage, mask)
            finally:
                if offload:
                    with nullcontext():
                        self._offload_one_teacher(teacher_idx)

            if t_rows is None:
                # Non-last PP rank: nothing to fuse on this rank.
                return None

            # Already masked to [N, V] by the helper (no dense [B, L_student, V] for OPSD).
            # Upcast to fp32 to match the single-teacher loss numerics.
            t_masked = t_rows.float()
            del t_rows
            if loss_temperature != 1.0:
                t_masked = t_masked / loss_temperature

            if self.teacher_fusion == 'logit_avg':
                term = weights[teacher_idx] * t_masked
                logit_avg_acc = term if logit_avg_acc is None else (logit_avg_acc + term)
                del t_masked, term
            else:  # prob_avg
                lp = vocab_parallel_log_softmax(t_masked)
                del t_masked
                log_w = math.log(weights[teacher_idx])
                term = lp + log_w
                del lp
                if prob_avg_acc is None:
                    prob_avg_acc = term
                else:
                    # Elementwise two-term logsumexp. Equivalent to
                    # logsumexp(stack([acc, term]), dim=0) but WITHOUT materializing the
                    # transient [2, N, V] stack, which would otherwise double the
                    # fused-target memory on large vocab/seq and defeat the mask-and-discard
                    # contract. Matches the in-tree logaddexp used for teacher smoothing.
                    prob_avg_acc = torch.logaddexp(prob_avg_acc, term)
                del term

        if self.teacher_fusion == 'logit_avg':
            # Log-linear pool: one log_softmax over the weighted-logit sum.
            fused_log_probs = vocab_parallel_log_softmax(logit_avg_acc)
            del logit_avg_acc
        else:
            fused_log_probs = prob_avg_acc

        # Teacher label smoothing applied ONCE after fusion. For normalized weights this
        # equals per-teacher smoothing: sum_i w_i[(1-a)P_i + a/V] = (1-a) sum_i w_i P_i + a/V.
        if self.teacher_label_smoothing > 0:
            alpha = self.teacher_label_smoothing
            V = int(get_args().padded_vocab_size)
            log_1_minus_alpha = math.log(1.0 - alpha)
            log_alpha_over_V = math.log(alpha / V)
            fused_log_probs = torch.logaddexp(
                fused_log_probs + log_1_minus_alpha,
                torch.full_like(fused_log_probs, log_alpha_over_V),
            )

        return fused_log_probs

    def _align_vocab_size(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> tuple:
        """Align vocab size between student and teacher logits.

        When student and teacher have different vocab sizes, pad the smaller one
        and copy logits from the larger one for the extra tokens.

        Args:
            student_logits: Student logits [..., student_vocab_size]
            teacher_logits: Teacher logits [..., teacher_vocab_size]

        Returns:
            Tuple of aligned (student_logits, teacher_logits)
        """
        stu_vocab = student_logits.shape[-1]
        tea_vocab = teacher_logits.shape[-1]

        if stu_vocab == tea_vocab:
            return student_logits, teacher_logits

        if stu_vocab < tea_vocab:
            # Pad student logits and copy teacher's extra vocab logits
            student_logits = F.pad(student_logits, (0, tea_vocab - stu_vocab), 'constant', 0)
            student_logits[..., stu_vocab:] = teacher_logits[..., stu_vocab:]
        else:
            # Pad teacher logits and copy student's extra vocab logits
            teacher_logits = F.pad(teacher_logits, (0, stu_vocab - tea_vocab), 'constant', 0)
            teacher_logits[..., tea_vocab:] = student_logits[..., tea_vocab:]

        return student_logits, teacher_logits

    def generalized_jsd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        beta: float = 0.5,
        chunk_size: int = 512,
        diagnostics: Optional[Dict[str, torch.Tensor]] = None,
        student_logits_pre_masked: bool = False,
        teacher_target_log_probs_masked: Optional[torch.Tensor] = None,
        return_sum_count: bool = False,
        teacher_logits_pre_masked: bool = False,
        raw_diagnostics: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # When `teacher_target_log_probs_masked` is provided (two-teacher ensemble), it is
        # the already-masked `[N, partition_vocab]` fused teacher LOG-PROB target. In that
        # mode the teacher distribution is precomputed, so this function skips teacher-side
        # vocab alignment, masking, temperature division, and log_softmax: `teacher_logits`
        # is ignored. The student path is byte-identical to the single-teacher case.
        teacher_is_log_probs = teacher_target_log_probs_masked is not None
        # Defaults also preserve callers that extract this method for CPU checks.
        loss_type = getattr(self, 'gkd_loss_type', 'jsd')
        if loss_type not in ('jsd', 'todi'):
            raise ValueError(f'Unsupported gkd_loss_type={loss_type!r}.')
        todi_beta = getattr(self, 'gkd_todi_beta', 1.0)
        stable_reverse_kl = loss_type == 'jsd' and beta == 1
        args = get_args()
        if raw_diagnostics is not None and args.context_parallel_size != 1:
            raise ValueError('Streamed diagnostic sufficient statistics currently require CP1.')
        mask = labels != -100
        local_num_valid = mask.sum()
        num_valid = local_num_valid.float()
        # CP-LOCAL valid count for the per-token grad path. finalize_model_grads does
        # its own dp_cp all-reduce over the returned count, so returning the CP-global
        # num_valid would double-reduce and inflate the denominator. Snapshot before the
        # CP all-reduce below; used only when return_sum_count=True.
        local_num_valid_denom = num_valid.clone()

        # All-reduce num_valid across CP group for correct averaging
        if args.context_parallel_size > 1:
            torch.distributed.all_reduce(
                num_valid, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group())

        if num_valid == 0:
            zero_loss = (student_logits.sum() * 0).reshape(())
            if return_sum_count:
                # Keep the (loss_sum, count) contract: CP-LOCAL zero count (0.0 here).
                return zero_loss, local_num_valid_denom.detach()
            return zero_loss

        # Align vocab size between student and teacher. Skipped in fused mode: the fused
        # target is already a masked log-prob over the (identical, fail-fast-enforced) vocab grid.
        if not teacher_is_log_probs:
            student_logits, teacher_logits = self._align_vocab_size(student_logits, teacher_logits)

        # Apply mask FIRST (shrinks [T, V] -> [valid, V]), then temperature scaling.
        # Masking before division avoids materializing a full-vocab temporary
        # [T, vocab] tensor (~8 GB at T~=29K, vocab=151936, bf16), which is the
        # peak-memory term on the largest DP-rank batches. x[mask]/t is identical to
        # (x/t)[mask] for a scalar t. Sampling, including greedy rollout, never
        # changes this explicitly configured loss temperature.
        loss_temperature = self.loss_temperature
        local_num_valid_int = local_num_valid.item()
        if student_logits_pre_masked:
            # Student logits already came pre-selected from `_postprocess`
            # via the `valid_token_indices` kwarg. Shape is `[valid, 1, V]`
            # (a single-batch S*B=valid layout coming back through the
            # `logits.transpose(0, 1)` post-step in gpt_model). Flatten the
            # leading dims to match the `[valid, V]` shape every downstream
            # path expects, but DO NOT mask again: the row order already
            # matches the row order of `labels[mask]` because both used the
            # same flattened `labels != -100` selection.
            student_logits_masked = student_logits.reshape(-1, student_logits.shape[-1])
            if student_logits_masked.shape[0] != local_num_valid_int:
                raise ValueError(
                    f'selected student rows={student_logits_masked.shape[0]} do not match '
                    f'supervised rows={local_num_valid_int}. Check the LM-head selection handoff.')
        else:
            student_logits_masked = student_logits[mask]  # [local_num_valid_tokens, partition_vocab_size]
        if teacher_is_log_probs:
            # Fused target is already masked [N, V] in the same mask order as labels[mask].
            teacher_logits_masked = teacher_target_log_probs_masked
        elif teacher_logits_pre_masked:
            teacher_logits_masked = teacher_logits.reshape(-1, teacher_logits.shape[-1])
            if teacher_logits_masked.shape[0] != local_num_valid_int:
                raise ValueError(
                    f'selected teacher rows={teacher_logits_masked.shape[0]} do not match '
                    f'supervised rows={local_num_valid_int}.')
        else:
            teacher_logits_masked = teacher_logits[mask]
        del student_logits, teacher_logits

        # forward_step now calls the student with `fp32_output=False`, so the LM-head
        # output is bf16 here instead of full-vocab fp32, which would materialize a
        # `[tokens, vocab]` fp32 temporary inside Float16Module.forward — several GiB
        # at this vocabulary size. Promote the much smaller `[valid, vocab]` slice to
        # fp32 right after masking so log_softmax / KL / entropy below see fp32 inputs
        # — keeping the full-precision objective numerics. Teacher logits arrive in
        # fp32 (its Float16Module still upcasts) and need no extra cast.
        if student_logits_masked.dtype != torch.float32:
            student_logits_masked = student_logits_masked.float()
        if teacher_logits_masked.dtype != torch.float32:
            teacher_logits_masked = teacher_logits_masked.float()
        if loss_temperature != 1.0:
            student_logits_masked = student_logits_masked / loss_temperature
            # Fused teacher target already had temperature applied per-teacher before
            # log_softmax; do NOT divide it again.
            if not teacher_is_log_probs:
                teacher_logits_masked = teacher_logits_masked / loss_temperature

        # Use local count for iteration, global count for averaging
        total_loss = student_logits_masked.new_zeros(())
        collect = diagnostics is not None or raw_diagnostics is not None

        if loss_type == 'jsd' and beta != 0 and beta != 1:
            beta_t = torch.tensor(beta, dtype=student_logits_masked.dtype, device=student_logits_masked.device)
            log_beta = torch.log(beta_t)
            log_1_minus_beta = torch.log1p(-beta_t)
        else:
            beta_t = log_beta = log_1_minus_beta = None

        # Diagnostic accumulators
        if collect:
            _zero = student_logits_masked.new_zeros(())
            d_kl_teacher_sum = _zero.clone()
            d_kl_student_sum = _zero.clone()
            d_kl_sq_sum = _zero.clone()
            # Bidirectional KL (always compute both directions)
            d_kl_forward_sum = _zero.clone()
            d_kl_reverse_sum = _zero.clone()
            # Stratified accumulators: per-token KL split by top-1 agreement
            d_kl_agree_sum = _zero.clone()
            d_kl_disagree_sum = _zero.clone()
            d_n_agree = _zero.clone()
            d_n_disagree = _zero.clone()
            d_n_total = _zero.clone()
            d_teacher_maxprob_disagree_sum = _zero.clone()
            d_student_maxprob_disagree_sum = _zero.clone()
            d_agreement_sum = _zero.clone()
            # Distribution shape
            d_s_entropy_sum = _zero.clone()
            d_t_entropy_sum = _zero.clone()
            # Logit scale (pre-temperature)
            d_s_logit_sq_sum = _zero.clone()
            d_t_logit_sq_sum = _zero.clone()
            # Per-token KL values for percentile computation
            all_per_token_kl = []

        for start_idx in range(0, local_num_valid_int, chunk_size):
            end_idx = min(start_idx + chunk_size, local_num_valid_int)
            s_chunk = student_logits_masked[start_idx:end_idx]
            t_chunk = teacher_logits_masked[start_idx:end_idx]

            # Reverse KL owns student normalization and its centered logit VJP.
            # Other objectives retain the ordinary differentiable log_softmax.
            if not stable_reverse_kl:
                s_log_probs = vocab_parallel_log_softmax(s_chunk)
            if teacher_is_log_probs:
                # Fused two-teacher target is ALREADY a log-prob; do not re-normalize.
                t_log_probs = t_chunk
            else:
                t_log_probs = vocab_parallel_log_softmax(t_chunk)

            # Teacher label smoothing: T' = (1-α)T + α/V, applied in log-space.
            # Bounds log(1/T) and tames reverse-KL gradnorm when teacher is peaky.
            # Operates on partition slices: the affine combination preserves the
            # global normalization, so no extra all-reduce is needed.
            # Skipped in fused mode: smoothing was already applied once after fusion.
            if self.teacher_label_smoothing > 0 and not teacher_is_log_probs:
                alpha = self.teacher_label_smoothing
                V = int(get_args().padded_vocab_size)
                log_1_minus_alpha = math.log(1.0 - alpha)
                log_alpha_over_V = math.log(alpha / V)
                t_log_probs = torch.logaddexp(
                    t_log_probs + log_1_minus_alpha,
                    torch.full_like(t_log_probs, log_alpha_over_V),
                )

            if stable_reverse_kl:
                jsd_chunk, s_log_probs = vocab_parallel_reverse_kl_div(s_chunk, t_log_probs)
            elif loss_type == 'todi':
                jsd_chunk = vocab_parallel_todi_div(s_log_probs, t_log_probs, beta=todi_beta)
            elif beta == 0:
                # JSD = KL(teacher || student)
                jsd_chunk = vocab_parallel_kl_div(s_log_probs, t_log_probs)
            else:
                mixture_log_probs = torch.logsumexp(
                    torch.stack([s_log_probs + log_1_minus_beta, t_log_probs + log_beta]),
                    dim=0,
                )
                kl_teacher = vocab_parallel_kl_div(mixture_log_probs, t_log_probs)
                kl_student = vocab_parallel_kl_div(mixture_log_probs, s_log_probs)
                del mixture_log_probs

                jsd_chunk = beta_t * kl_teacher + (1 - beta_t) * kl_student

            # --- Diagnostic collection (no_grad, detached) ---
            if collect:
                with torch.no_grad():
                    jsd_det = jsd_chunk.detach()
                    d_kl_sq_sum += (jsd_det ** 2).sum()
                    all_per_token_kl.append(jsd_det.clone())

                    # Bidirectional KL (compute BOTH directions regardless of beta)
                    kl_fwd_chunk = vocab_parallel_kl_div(s_log_probs.detach(), t_log_probs.detach())
                    kl_rev_chunk = vocab_parallel_kl_div(t_log_probs.detach(), s_log_probs.detach())
                    d_kl_forward_sum += kl_fwd_chunk.sum()
                    d_kl_reverse_sum += kl_rev_chunk.sum()

                    if loss_type == 'todi':
                        # Pure directional diagnostics, not the weighted ToDi
                        # terms and never a ToDi value mislabeled as reverse KL.
                        d_kl_teacher_sum += kl_fwd_chunk.sum()
                        d_kl_student_sum += kl_rev_chunk.sum()
                    elif beta == 0:
                        d_kl_teacher_sum += jsd_det.sum()
                    elif beta == 1:
                        d_kl_student_sum += jsd_det.sum()
                    else:
                        d_kl_teacher_sum += kl_teacher.detach().sum()
                        d_kl_student_sum += kl_student.detach().sum()

                    # Distribution shape
                    d_s_entropy_sum += vocab_parallel_entropy(s_log_probs.detach()).sum()
                    d_t_entropy_sum += vocab_parallel_entropy(t_log_probs.detach()).sum()

                    # Logit scale (pre-temperature, use raw chunks before log_softmax).
                    # In fused two-teacher mode t_chunk is a LOG-PROB, not a raw logit, so
                    # its squared "norm" is meaningless; skip it (emitted as NaN below).
                    d_s_logit_sq_sum += (s_chunk.detach() ** 2).mean(dim=-1).sum()
                    if not teacher_is_log_probs:
                        d_t_logit_sq_sum += (t_chunk.detach() ** 2).mean(dim=-1).sum()

                    # Stratified agreement diagnostics. Must use the GLOBAL argmax: a
                    # per-rank argmax over the vocab shard compares per-shard winners,
                    # which is measurably biased upward rather than merely noisy.
                    # log-probs are fine here -- argmax is invariant under log_softmax,
                    # and this also sidesteps the teacher_is_log_probs asymmetry of the
                    # raw chunks.
                    s_top1, t_top1 = vocab_parallel_top1_indices(
                        s_log_probs.detach(), t_log_probs.detach())
                    agree_mask = (s_top1 == t_top1)
                    disagree_mask = ~agree_mask
                    n_agree = agree_mask.sum().to(jsd_det.dtype)
                    n_disagree = disagree_mask.sum().to(jsd_det.dtype)

                    d_n_agree += n_agree
                    d_n_disagree += n_disagree
                    d_n_total += jsd_det.shape[0]
                    d_agreement_sum += n_agree
                    d_kl_agree_sum += jsd_det[agree_mask].sum()
                    d_kl_disagree_sum += jsd_det[disagree_mask].sum()

                    t_probs = t_log_probs.detach().exp()
                    t_maxprob = t_probs.max(dim=-1).values
                    if mpu.get_tensor_model_parallel_world_size() > 1:
                        torch.distributed.all_reduce(
                            t_maxprob, op=torch.distributed.ReduceOp.MAX,
                            group=mpu.get_tensor_model_parallel_group())
                    d_teacher_maxprob_disagree_sum += t_maxprob[disagree_mask].sum()

                    s_probs = s_log_probs.detach().exp()
                    s_maxprob = s_probs.max(dim=-1).values
                    if mpu.get_tensor_model_parallel_world_size() > 1:
                        torch.distributed.all_reduce(
                            s_maxprob, op=torch.distributed.ReduceOp.MAX,
                            group=mpu.get_tensor_model_parallel_group())
                    d_student_maxprob_disagree_sum += s_maxprob[disagree_mask].sum()

                    del t_probs, t_maxprob, s_probs, s_maxprob
                    del s_top1, t_top1, agree_mask, disagree_mask
                    del kl_fwd_chunk, kl_rev_chunk

            if loss_type == 'jsd' and beta != 0 and beta != 1:
                del kl_teacher, kl_student

            total_loss = total_loss + jsd_chunk.sum()
            del jsd_chunk, s_log_probs, t_log_probs, s_chunk, t_chunk

        # Clean up masked logits
        del student_logits_masked, teacher_logits_masked

        # Snapshot the CP-LOCAL loss sum for the per-token grad path only. Megatron's
        # finalize_model_grads dp_cp all-reduces the returned loss_sum, so CP-reducing
        # here too would double-count. None when the flag is off -> graph unchanged.
        local_total_loss = total_loss.clone() if return_sum_count else None

        # All-reduce total_loss across CP group for correct sum
        if args.context_parallel_size > 1:
            torch.distributed.all_reduce(
                total_loss, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group())
        # A raw-statistics caller may return early inside the no_grad diagnostic
        # block below. Construct its MEAN while the loss graph is still enabled.
        raw_mean_loss = (total_loss / num_valid
                         if raw_diagnostics is not None and not return_sum_count else None)

        # Populate diagnostics
        if collect:
            with torch.no_grad():
                if mpu.get_tensor_model_parallel_world_size() > 1:
                    logit_stats = torch.stack([d_s_logit_sq_sum, d_t_logit_sq_sum])
                    torch.distributed.all_reduce(logit_stats, group=mpu.get_tensor_model_parallel_group())
                    logit_stats /= mpu.get_tensor_model_parallel_world_size()
                    d_s_logit_sq_sum, d_t_logit_sq_sum = logit_stats.unbind()
                # At CP>1 this diagnostic stack is reduced on collection steps.
                if args.context_parallel_size > 1:
                    cp_group = mpu.get_context_parallel_group()
                    d_stats = torch.stack([
                        d_kl_teacher_sum, d_kl_student_sum, d_kl_sq_sum,
                        d_kl_forward_sum, d_kl_reverse_sum,
                        d_kl_agree_sum, d_kl_disagree_sum,
                        d_n_agree, d_n_disagree, d_n_total,
                        d_teacher_maxprob_disagree_sum, d_student_maxprob_disagree_sum,
                        d_agreement_sum,
                        d_s_entropy_sum, d_t_entropy_sum,
                        d_s_logit_sq_sum, d_t_logit_sq_sum,
                    ])
                    torch.distributed.all_reduce(d_stats, op=torch.distributed.ReduceOp.SUM, group=cp_group)
                    (d_kl_teacher_sum, d_kl_student_sum, d_kl_sq_sum,
                     d_kl_forward_sum, d_kl_reverse_sum,
                     d_kl_agree_sum, d_kl_disagree_sum,
                     d_n_agree, d_n_disagree, d_n_total,
                     d_teacher_maxprob_disagree_sum, d_student_maxprob_disagree_sum,
                     d_agreement_sum,
                     d_s_entropy_sum, d_t_entropy_sum,
                     d_s_logit_sq_sum, d_t_logit_sq_sum) = d_stats.unbind()

                nv = num_valid

                if raw_diagnostics is not None:
                    # Keep only scalar/token-sized statistics. A streamed head
                    # calls this SAME criterion per chunk, then finalizes once
                    # for the complete microbatch (not a mean of percentiles).
                    raw_diagnostics.update({
                        'kl_teacher_sum': d_kl_teacher_sum,
                        'kl_student_sum': d_kl_student_sum,
                        'jsd_sq_sum': d_kl_sq_sum,
                        'kl_forward_sum': d_kl_forward_sum,
                        'kl_reverse_sum': d_kl_reverse_sum,
                        'kl_agree_sum': d_kl_agree_sum,
                        'kl_disagree_sum': d_kl_disagree_sum,
                        'n_agree': d_n_agree, 'n_disagree': d_n_disagree, 'n_total': d_n_total,
                        'teacher_maxprob_disagree_sum': d_teacher_maxprob_disagree_sum,
                        'student_maxprob_disagree_sum': d_student_maxprob_disagree_sum,
                        'agreement_sum': d_agreement_sum,
                        's_entropy_sum': d_s_entropy_sum, 't_entropy_sum': d_t_entropy_sum,
                        's_logit_sq_sum': d_s_logit_sq_sum, 't_logit_sq_sum': d_t_logit_sq_sum,
                        'kl_values': torch.cat(all_per_token_kl),
                    })
                    if diagnostics is None:
                        if return_sum_count:
                            return local_total_loss, local_num_valid_denom.detach()
                        return raw_mean_loss

                # Basic KL
                diagnostics['kl_teacher'] = d_kl_teacher_sum / nv
                diagnostics['kl_student'] = d_kl_student_sum / nv
                kl_mean = (total_loss / nv).detach()
                kl_sq_mean = d_kl_sq_sum / nv
                diagnostics['jsd_std'] = (kl_sq_mean - kl_mean ** 2).clamp(min=0).sqrt()

                # Bidirectional KL
                diagnostics['kl_forward'] = d_kl_forward_sum / nv
                diagnostics['kl_reverse'] = d_kl_reverse_sum / nv
                kl_fwd = diagnostics['kl_forward']
                diagnostics['kl_asymmetry'] = torch.where(
                    kl_fwd > 1e-8, diagnostics['kl_reverse'] / kl_fwd, kl_fwd.new_ones(()))

                # Stratified KL
                n_agree_safe = d_n_agree.clamp(min=1)
                n_disagree_safe = d_n_disagree.clamp(min=1)
                n_total_safe = d_n_total.clamp(min=1)
                diagnostics['kl_at_agree'] = d_kl_agree_sum / n_agree_safe
                diagnostics['kl_at_disagree'] = d_kl_disagree_sum / n_disagree_safe
                diagnostics['kl_disagree_ratio'] = (
                    diagnostics['kl_at_disagree'] / diagnostics['kl_at_agree'].clamp(min=1e-6))
                diagnostics['top1_agreement'] = d_agreement_sum / n_total_safe
                diagnostics['teacher_maxprob_at_disagree'] = d_teacher_maxprob_disagree_sum / n_disagree_safe
                diagnostics['student_maxprob_at_disagree'] = d_student_maxprob_disagree_sum / n_disagree_safe
                # Raw sufficient statistics for the post-aggregation rebuild of the five
                # stratified ``sum/count`` diagnostics above (see _SUM_RATIOS). The ratios
                # here are per-microbatch and get mean-averaged by the native path, so
                # train_step re-forms them from these sums/counts and pops the raw keys --
                # they are never logged. Nothing extra is communicated: these scalars are
                # already computed and CP/TP-reduced by this point.
                #
                # The prefix is inlined rather than read from the module constant: this
                # function is executed by the source-extraction test harness
                # (tests/mopd/source_test_utils.py), whose namespace holds
                # only the names it is handed, so a module global here is a NameError.
                diagnostics['_mopd_raw/kl_agree_sum'] = d_kl_agree_sum
                diagnostics['_mopd_raw/kl_disagree_sum'] = d_kl_disagree_sum
                diagnostics['_mopd_raw/n_agree'] = d_n_agree
                diagnostics['_mopd_raw/n_disagree'] = d_n_disagree
                diagnostics['_mopd_raw/n_total'] = d_n_total
                diagnostics['_mopd_raw/agreement_sum'] = d_agreement_sum
                diagnostics['_mopd_raw/teacher_maxprob_disagree_sum'] = d_teacher_maxprob_disagree_sum
                diagnostics['_mopd_raw/student_maxprob_disagree_sum'] = d_student_maxprob_disagree_sum
                # Sufficient statistics for the pooled RMS/std recombination (see
                # _POOLED_NORMS / _POOLED_STDS). ``total_loss`` is graph-connected, so it
                # must be detached -- this runs inside ``torch.no_grad()`` but the tensor
                # itself still carries grad_fn from the loss computation above.
                diagnostics['_mopd_raw/jsd_sq_sum'] = d_kl_sq_sum
                diagnostics['_mopd_raw/jsd_sum'] = total_loss.detach()
                diagnostics['_mopd_raw/s_logit_sq_sum'] = d_s_logit_sq_sum
                diagnostics['_mopd_raw/t_logit_sq_sum'] = d_t_logit_sq_sum
                diagnostics['_mopd_raw/nv'] = nv

                # Distribution shape
                diagnostics['student_entropy'] = d_s_entropy_sum / nv
                diagnostics['teacher_entropy'] = d_t_entropy_sum / nv
                t_ent = diagnostics['teacher_entropy']
                diagnostics['entropy_ratio'] = torch.where(
                    t_ent > 1e-8, diagnostics['student_entropy'] / t_ent, t_ent.new_ones(()))

                # Logit scale
                diagnostics['student_logit_norm'] = (d_s_logit_sq_sum / nv).sqrt()
                # teacher_logit_norm is a raw-logit-only statistic. In fused two-teacher
                # mode the teacher target is a log-prob (no raw logits exist), so the metric
                # is undefined: do NOT populate it at all. loss_func gates the corresponding
                # efficiency/* export on the same config-level condition, so the key is
                # consistently absent on every rank (rank-symmetric).
                if not teacher_is_log_probs:
                    diagnostics['teacher_logit_norm'] = (d_t_logit_sq_sum / nv).sqrt()

                # KL percentiles (local computation, approximate for CP>1)
                if all_per_token_kl:
                    all_kl = torch.cat(all_per_token_kl)
                    if all_kl.numel() > 0:
                        diagnostics['kl_p50'] = all_kl.quantile(0.5)
                        diagnostics['kl_p90'] = all_kl.quantile(0.9)
                        diagnostics['kl_p99'] = all_kl.quantile(0.99)
                        diagnostics['kl_max'] = all_kl.max()
                    del all_kl
                del all_per_token_kl

        denom = num_valid
        local_denom = local_num_valid_denom
        if return_sum_count:
            # Per-token grad path: return CP-LOCAL (sum, count). Megatron's per-token
            # schedule sums these across microbatches and dp_cp all-reduces the count in
            # finalize_model_grads. Diagnostics above still use CP-global values.
            # Count is raw (unclamped) to match the SFT
            # reference; finalize_model_grads guards the division on num_tokens > 0.
            return local_total_loss, local_denom.detach()
        return total_loss / denom

    def loss_func(self,
                  output_tensor: torch.Tensor,
                  *,
                  labels: torch.Tensor,
                  teacher_logits: torch.Tensor,
                  data_source: DataSource = DataSource.DATASET,
                  student_logits_pre_masked: bool = False,
                  teacher_logits_pre_masked: bool = False,
                  teacher_target_log_probs_masked: Optional[torch.Tensor] = None,
                  teacher_idx: int = 0,
                  mopd_loss_scale: Optional[float] = None,
                  coordination_scale: float = 1.0,
                  coordination_padding: bool = False,
                  streamed_loss=None):
        """Compute GKD loss (JSD + optional SFT loss).

        When `student_logits_pre_masked` is True, `output_tensor` is already
        the `[valid_tokens, vocab]` (or `[valid_tokens, 1, vocab]`) slice
        produced by the model when given `valid_token_indices`. The JSD
        loss must skip its own `student_logits[mask]` step in that case;
        teacher logits follow the equivalent `teacher_logits_pre_masked` contract.
        Labels remain full-length for token/sample normalization and diagnostics.

        When `teacher_target_log_probs_masked` is provided (two-teacher ensemble),
        it is the precomputed masked fused teacher log-prob target and `teacher_logits`
        is ignored.
        """
        student_logits = output_tensor

        interval = self._diagnostics_interval
        should_collect = (interval <= 0
                          or self._step <= 1
                          or self._step % interval == 0)
        diagnostics: Optional[Dict[str, torch.Tensor]] = {} if should_collect else None
        if streamed_loss is not None:
            diagnostics = streamed_loss[2] if should_collect else None

        def calculate_jsd(return_sum_count=False):
            if streamed_loss is not None:
                loss_sum, count, _ = streamed_loss
                return (loss_sum, count) if return_sum_count else loss_sum / count.clamp(min=1)
            return self.generalized_jsd_loss(
                student_logits=student_logits, teacher_logits=teacher_logits, labels=labels,
                beta=self.beta, diagnostics=diagnostics,
                student_logits_pre_masked=student_logits_pre_masked,
                teacher_logits_pre_masked=teacher_logits_pre_masked,
                teacher_target_log_probs_masked=teacher_target_log_probs_masked,
                return_sum_count=return_sum_count)
        # Two loss protocols. Flag OFF (default): generalized_jsd_loss returns the CP-global
        # scalar mean and loss_func returns a 2-tuple -> legacy per-microbatch averaging,
        # byte-identical to the sequence-mean path. Flag ON: ask for the CP-local (sum, count)
        # and return a 3-tuple so Megatron's calculate_per_token_loss schedule scales
        # gradients by 1/global_total_tokens (proportional to sequence length).
        if self.calculate_per_token_loss:
            loss_sum, num_tokens = calculate_jsd(return_sum_count=True)
            # Scalar mean for the metric-dict consumers below (jsd_loss / loss log,
            # _accumulate_domain_metrics which token-weights this mean). NOT gradient-bearing.
            jsd_mean = (loss_sum / num_tokens.clamp(min=1)).detach()
        elif self.mopd_domain_loss_mode:
            # MOPD domain normalization (2-tuple path, calculate_per_token_loss False). The
            # per-microbatch gradient-bearing `loss` is the domain scale (computed in
            # _replace_data_iterator via a DP all-reduce) times the microbatch base:
            #   domain_token   -> scale * loss_sum   (CP-LOCAL sum; CP==1 so == global sum)
            #   domain_sequence-> scale * mean        (CP-global per-token mean)
            # See _compute_domain_loss_scales for why this realizes the target objective
            # after Megatron's num_microbatches / DP averaging. jsd_mean stays the per-token
            # MEAN so loss traces line up across normalization modes.
            assert mpu.get_context_parallel_world_size() == 1, (
                'MOPD domain loss normalization requires context_parallel_size == 1 '
                '(the per-domain sums are CP-local by construction).')
            scale = float(mopd_loss_scale) if mopd_loss_scale is not None else 0.0
            if self.mopd_loss_normalization == 'domain_token':
                loss_sum, num_tokens = calculate_jsd(return_sum_count=True)
                jsd_mean = (loss_sum / num_tokens.clamp(min=1)).detach()
                loss = scale * loss_sum
            else:  # domain_sequence
                jsd_loss = calculate_jsd()
                jsd_mean = jsd_loss.detach().clone()
                loss = scale * jsd_loss
        else:
            jsd_loss = calculate_jsd()
            loss = jsd_loss
            jsd_mean = jsd_loss.detach().clone()

        metric = {
            'jsd_loss': jsd_mean,
        }
        if getattr(self, 'gkd_loss_type', 'jsd') == 'todi':
            # Config-constant key, including zero-token ranks. Keep jsd_loss as
            # the generic KD-loss alias for existing log consumers.
            metric['loss/todi'] = jsd_mean

        # Populate grouped diagnostics metrics.
        #
        # RANK-SYMMETRIC COLLECTIVE INVARIANT (do not break): _all_reduce_metric stacks
        # metric.values() into ONE tensor and all-reduces it over the DATA_PARALLEL_GROUP.
        # That collective REQUIRES an identical tensor length (== number of metric keys)
        # on every DP rank, or the all-reduce shape mismatches and NCCL hangs until the
        # watchdog timeout.
        #
        # Two sources of asymmetry, both rooted in a rank with 0 valid completion tokens
        # this micro-batch (possible under on-policy GKD where DP ranks get different
        # rollouts):
        #   (a) generalized_jsd_loss early-returns at `if num_valid == 0:` BEFORE populating
        #       anything, so `diagnostics` stays an empty dict {} on that rank. Guarding on
        #       truthiness would therefore skip this block there and emit a smaller key set
        #       than its peers; guard on `is not None` instead (should_collect is driven by
        #       diagnostics_interval, in lockstep on all ranks, so `diagnostics is None` is
        #       itself rank-symmetric).
        #   (b) Even when populated, the kl_p50 percentile group is gated on a non-empty
        #       per-token KL list inside the loss, i.e. it is missing exactly when
        #       num_valid==0 -- the same case as (a).
        # Invariant: when collecting, ALWAYS emit the SAME key set, defaulting every value to
        # a shape/dtype/device-matched 0-dim zero placeholder when the loss didn't compute it.
        #
        # The all-reduce is AVG, so on steps where some rank had 0 valid tokens its zero
        # placeholders pull the averaged percentile/diagnostic downward. These are coarse
        # logging diagnostics and a masked/weighted average would need a second collective,
        # so shape symmetry takes priority over the exact value.
        if diagnostics is not None:
            _zero = jsd_mean.detach().new_zeros(())
            metric['loss/kl_teacher'] = diagnostics.get('kl_teacher', _zero)
            metric['loss/kl_student'] = diagnostics.get('kl_student', _zero)
            metric['loss/jsd_per_token_std'] = diagnostics.get('jsd_std', _zero)
            metric['divergence/kl_forward'] = diagnostics.get('kl_forward', _zero)
            metric['divergence/kl_reverse'] = diagnostics.get('kl_reverse', _zero)
            metric['divergence/kl_asymmetry'] = diagnostics.get('kl_asymmetry', _zero)
            metric['divergence/kl_p50'] = diagnostics.get('kl_p50', _zero)
            metric['divergence/kl_p90'] = diagnostics.get('kl_p90', _zero)
            metric['divergence/kl_p99'] = diagnostics.get('kl_p99', _zero)
            metric['divergence/kl_max'] = diagnostics.get('kl_max', _zero)
            metric['divergence/kl_at_agree'] = diagnostics.get('kl_at_agree', _zero)
            metric['divergence/kl_at_disagree'] = diagnostics.get('kl_at_disagree', _zero)
            metric['divergence/kl_disagree_ratio'] = diagnostics.get('kl_disagree_ratio', _zero)
            metric['divergence/top1_agreement'] = diagnostics.get('top1_agreement', _zero)
            metric['divergence/teacher_maxprob_at_disagree'] = diagnostics.get('teacher_maxprob_at_disagree', _zero)
            metric['divergence/student_maxprob_at_disagree'] = diagnostics.get('student_maxprob_at_disagree', _zero)
            metric['divergence/student_entropy'] = diagnostics.get('student_entropy', _zero)
            metric['divergence/teacher_entropy'] = diagnostics.get('teacher_entropy', _zero)
            metric['divergence/entropy_ratio'] = diagnostics.get('entropy_ratio', _zero)
            metric['efficiency/student_logit_norm'] = diagnostics.get('student_logit_norm', _zero)
            # teacher_logit_norm is a raw-logit-only metric. In fused two-teacher mode the
            # teacher target is a log-prob, so omit this key entirely. The gate is the
            # config-level `teacher_target_log_probs_masked is None` (the two-teacher path is
            # selected identically on every rank), so the key is present single-teacher and
            # absent fused-teacher on ALL ranks -- rank-symmetric.
            if teacher_target_log_probs_masked is None:
                metric['efficiency/teacher_logit_norm'] = diagnostics.get('teacher_logit_norm', _zero)
            # Sufficient statistics for the stratified sum/count diagnostics. They are
            # averaged by the native path like any other scalar and then divided once in
            # ``recompute_sum_ratios`` (train_step), which also pops them so they never
            # reach the log. ``_zero`` keeps the key set identical on every rank.
            for _raw in ('kl_agree_sum', 'kl_disagree_sum', 'n_agree', 'n_disagree', 'n_total',
                         'agreement_sum', 'teacher_maxprob_disagree_sum',
                         'student_maxprob_disagree_sum',
                         'jsd_sq_sum', 'jsd_sum', 's_logit_sq_sum', 't_logit_sq_sum', 'nv'):
                metric['_mopd_raw/' + _raw] = diagnostics.get('_mopd_raw/' + _raw, _zero)

        # Add SFT loss if enabled (skip for student-generated responses)
        if self.sft_alpha > 0 and data_source != DataSource.STUDENT:
            args = get_args()
            logits_sbv = student_logits.transpose(0, 1).contiguous()
            per_token_loss = self.unwrapped_models[0].compute_language_model_loss(labels, logits_sbv)

            loss_mask = labels != -100
            sft_loss_sum = (per_token_loss * loss_mask).sum()
            sft_loss_count = loss_mask.sum().float()

            if self.calculate_per_token_loss:
                # Per-token grad path: add the CP-LOCAL SFT sum into loss_sum but DO NOT
                # touch num_tokens. JSD and SFT share the same completion mask
                # (labels != -100), so both are per-token objectives over the SAME N
                # tokens. The effective per-token loss = loss_sum / num_tokens =
                # (JSD_sum + alpha*SFT_sum)/N == jsd_mean + alpha*sft_mean. Adding
                # sft_loss_count would double the denominator (2N) and halve the scale.
                # The metric report uses the CP-global mean.
                if args.context_parallel_size > 1:
                    sft_stats_g = torch.stack([sft_loss_sum.detach(), sft_loss_count.detach()])
                    torch.distributed.all_reduce(
                        sft_stats_g, op=torch.distributed.ReduceOp.SUM,
                        group=mpu.get_context_parallel_group())
                    sft_loss_mean = (sft_stats_g[0] / sft_stats_g[1].clamp(min=1)).detach()
                else:
                    sft_loss_mean = (sft_loss_sum.detach() / sft_loss_count.clamp(min=1)).detach()
                loss_sum = loss_sum + self.sft_alpha * sft_loss_sum
                metric['sft_loss'] = sft_loss_mean
            else:
                if args.context_parallel_size > 1:
                    sft_stats = torch.stack([sft_loss_sum, sft_loss_count])
                    torch.distributed.all_reduce(
                        sft_stats, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group())
                    sft_loss_sum, sft_loss_count = sft_stats[0], sft_stats[1]

                sft_loss = sft_loss_sum / sft_loss_count

                loss = loss + self.sft_alpha * sft_loss
                metric['sft_loss'] = sft_loss.detach()

        if self.calculate_per_token_loss:
            # Scalar loss report: same meaning as the flag-off path (per-token mean of
            # jsd + alpha*sft) so traces line up across matched runs.
            sft_mean = metric.get('sft_loss')
            metric['loss'] = (jsd_mean + self.sft_alpha * sft_mean).detach() if sft_mean is not None else jsd_mean
        elif self.mopd_domain_loss_mode:
            # `loss` here is the gradient-scaled value (scale * base); report the per-token
            # MEAN so the `loss` trace is comparable across normalization modes.
            # sft_alpha==0 is enforced for domain modes, so loss == jsd (no SFT term).
            metric['loss'] = jsd_mean
        else:
            metric['loss'] = loss.detach()
        if coordination_scale != 1.0 or coordination_padding:
            if not self.calculate_per_token_loss:
                loss = loss * coordination_scale
            # These scalar metrics are always averaged over execution slots by
            # native train_step, even for token-normalized gradients. Compensate
            # that divisor too; otherwise zero-padding dilutes the logged means.
            metric = {key: torch.zeros_like(value) if coordination_padding else value * coordination_scale
                      for key, value in metric.items()}
        if not self.skip_metric_dp_allreduce:
            metric = self._all_reduce_metric(metric)

        if self.calculate_per_token_loss:
            # Reporting loss follows the SFT trainer's `'lm loss'` convention: a 2-element
            # [sum, count] CP-LOCAL tensor. Megatron's training.py aggregates numel()==2
            # metrics by summing across microbatches, DP-all-reducing, then sum/count -- so
            # it MUST be added AFTER _all_reduce_metric (which stacks 0-dim scalars + AVG,
            # incompatible with this shape/semantics). The key is added unconditionally in
            # this branch on every rank, so the downstream aggregation stays rank-symmetric.
            metric['jsd loss'] = torch.stack([
                loss_sum.detach().float().view(1),
                num_tokens.detach().float().view(1),
            ]).view(-1)
            # No `loss / cp_world_size`: the per-token schedule neither divides by
            # num_tokens nor num_microbatches, and finalize_model_grads applies the CP
            # scaling in its dp_cp grad-reduce. Dividing here would double-count.
            return loss_sum, num_tokens.detach().to(torch.int), metric

        loss = loss / mpu.get_context_parallel_world_size()

        return loss, metric

    def forward_step(self, data_iterator, model):
        timers = get_timers()

        unwrapped_model = model.module.module
        input_tensor = unwrapped_model.get_input_tensor()
        vp_stage = unwrapped_model.vp_stage
        # Only unpack when data_iterator is (main_iter, sft_iter); avoid len() on RerunDataIterator (no __len__)
        if isinstance(data_iterator, (tuple, list)) and len(data_iterator) == 2:
            sft_data_iterator = data_iterator[1]
            data_iterator = data_iterator[0]

        timers('batch-generator', log_level=2).start()
        with self.stimer(bdata=True):
            data = dict(next(data_iterator))
            # Student route metadata must never enter a teacher forward.
            router_payload = data.pop('_mopd_router_replay', None)
            if getattr(self, '_local_completion_actor', None) is not None:
                # Upload the consumed MBS1 once and share those exact tensors
                # between teacher and student; queued inputs remain on CPU.
                data = to_device(data, self.device)
            rollout_temperature = data.pop('_mopd_rollout_temperature', None)
            data_source = data.pop('data_source', DataSource.DATASET)
            rollout_weight_step = data.pop('_rollout_weight_step', -1)
            # Record rollout/train freshness once per consumed micro-batch.
            # `_replace_data_iterator` increments `self._step` AFTER the
            # iterator is built, so by the time forward_step runs, `self._step`
            # already points at the *next* training step. The step that is
            # consuming this batch is therefore `self._step - 1` -- using that
            # keeps lag == 0 for the synchronous on-policy path and lag == 1
            # for overlap (where the producer wrote the rollout one step before
            # the consumer reads it), keeping lag <= 1.
            train_step = max(int(self._step) - 1, 0)
            lag = _weight_lag(rollout_weight_step, train_step)
            _validate_weight_lag(lag, overlap_enabled=bool(self.overlap_rollout_training))
            teacher_target_log_probs_masked = None
            remote_target = data.pop('_mopd_remote_teacher_target', None)
            if getattr(self, '_remote_teacher_actor', None) is not None:
                if remote_target is None:
                    raise ValueError('Streaming student batch has no matching remote teacher target.')
                teacher_logits = remote_target
            elif self.multi_teacher_mode:
                # MOPD domain routing: this microbatch is single-teacher (teacher_idx tagged in
                # _replace_data_iterator). Forward only that teacher; standard raw-logits path
                # and standard beta KL in loss_func (no fusion, no per-route beta).
                with nullcontext():
                    teacher_logits = self._compute_teacher_logits_for_one_batch(data, vp_stage)
            elif self.is_multi_teacher:
                # Two-teacher ensemble: build the fused masked log-prob target from a
                # sequential mask-and-discard forward over all teachers (lazy is enforced
                # for multi-teacher in argument validation, so `data` still carries
                # labels / teacher_opsd_batch here).
                with nullcontext():
                    teacher_target_log_probs_masked = \
                        self._compute_fused_teacher_log_probs_for_one_batch(data, vp_stage)
                teacher_logits = None
            elif self.lazy_teacher_logits:
                with nullcontext():
                    teacher_logits = self._compute_teacher_logits_for_one_batch(data, vp_stage)
            else:
                teacher_logits = data.pop('teacher_logits', None)
            data.pop('teacher_opsd_batch', None)
            data.pop('opsd_completion_start', None)
            # Capture teacher_idx BEFORE popping so we can pass it into loss_func for the
            # MOPD per-domain diagnostics channel. Non-MOPD runs default to 0.
            teacher_idx = int(data.pop('teacher_idx', 0) or 0)
            data.pop('_mopd_teacher_scheduled', None)
            coordination_scale = float(data.pop('_mopd_coordination_scale', 1.0))
            coordination_padding = bool(data.pop('_mopd_coordination_padding', False))
            # MOPD loss-normalization domain modes: per-microbatch gradient scale stamped in
            # _replace_data_iterator. Pop before _prepare_batch (side-channel scalar, not a
            # model kwarg). None on non-domain-mode runs -> loss_func ignores it.
            mopd_loss_scale = data.pop('mopd_loss_scale', None)
            data.pop('rollout_logprobs', None)
            data.pop('finish_reason', None)
            data.pop('is_truncated', None)
            data = self._prepare_batch(data, vp_stage)
        timers('batch-generator').stop()

        data.pop('loss_scale', None)
        labels = data.pop('labels', None)
        if input_tensor is not None:
            unwrapped_model.set_input_tensor(input_tensor)
        # Both native heads use the same supervised positions. Keep labels
        # full-length: only the expensive vocabulary projections are compact.
        # Argument validation excludes SFT/teacher-prompt remapping paths.
        valid_token_indices = None
        if self.mopd_response_only_logits:
            valid_token_indices = self._response_head_indices(labels)
        replay_session = getattr(unwrapped_model, '_mopd_router_replay_session', None)
        if replay_session is not None and replay_session.mode == 'r3':
            data['_mopd_routing_context'] = {
                'routes': router_payload, 'labels': labels, 'padding': coordination_padding}
        head_chunk_size = getattr(self, 'mopd_head_chunk_size', 0)
        if head_chunk_size:
            data['mopd_return_hidden_states'] = True
        with nullcontext():
            with self.stimer:
                # fp32_output=False keeps the LM-head logits in the model
                # precision (bf16). Combined with valid_token_indices, the
                # student forward only materialises `[valid, vocab]` bf16
                # logits, which is the smallest size the JSD loss can
                # consume while preserving strict numerical equivalence.
                student_output = model(
                    **data,
                    fp32_output=False,
                    valid_token_indices=valid_token_indices,
                )

        streamed_loss = None
        if head_chunk_size:
            from .streamed_distillation import TeacherHeadTarget, head_weight, streamed_head_loss
            if not isinstance(teacher_logits, TeacherHeadTarget):
                raise ValueError('Streamed student heads require the matching teacher hidden/head target.')
            native_student = getattr(unwrapped_model, 'language_model', unwrapped_model)
            interval = self._diagnostics_interval
            collect = interval <= 0 or self._step <= 1 or self._step % interval == 0
            with nullcontext():
                streamed_loss = streamed_head_loss(
                    student_output, teacher_logits, labels, native_student.output_layer,
                    head_weight(native_student), self.generalized_jsd_loss,
                    beta=self.beta, chunk_size=head_chunk_size, collect_diagnostics=collect)

        return student_output, partial(
            self.loss_func,
            labels=labels,
            teacher_logits=teacher_logits,
            data_source=data_source,
            student_logits_pre_masked=valid_token_indices is not None,
            teacher_logits_pre_masked=self.mopd_response_only_logits,
            coordination_scale=coordination_scale,
            coordination_padding=coordination_padding,
            teacher_target_log_probs_masked=teacher_target_log_probs_masked,
            teacher_idx=teacher_idx,
            mopd_loss_scale=mopd_loss_scale,
            streamed_loss=streamed_loss,
        )

    def patched_validate_args(self, args, *_args, **kwargs):
        """Override patched_validate_args to adjust EP parameters for Dense student.

        This is called before Megatron's validate_args, allowing us to reset EP to 1
        when student is Dense but EP > 1 was configured (for MoE teacher).
        """
        student_is_moe = getattr(args, 'num_experts', None) is not None
        if not student_is_moe:
            # Reset EP to 1 in Megatron args for Dense student
            self._original_ep_size = args.expert_model_parallel_size
            self._original_etp_size = args.expert_tensor_parallel_size
            args._original_ep_size = args.expert_model_parallel_size
            args._original_etp_size = args.expert_tensor_parallel_size
            args.expert_model_parallel_size = 1
            args.expert_tensor_parallel_size = 1
        self.student_is_moe = student_is_moe
        return self._origin_validate_args(args, *_args, **kwargs)
