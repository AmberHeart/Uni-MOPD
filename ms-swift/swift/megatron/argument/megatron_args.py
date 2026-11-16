# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import json
import megatron.core
import torch
from packaging import version
from transformers.utils.versions import require_version

from swift.llm import get_model_info_meta
from swift.utils import get_dist_setting, get_logger, json_parse_to_dict

mcore_015 = version.parse(megatron.core.__version__) >= version.parse('0.15.0rc0')
logger = get_logger()


@dataclass
class RLHFMegatronArgumentsMixin:
    rlhf_type: Literal['dpo', 'kto', 'grpo', 'gkd', 'rm'] = None
    ref_load: Optional[str] = None
    ref_adapter_load: Optional[str] = None

    beta: Optional[float] = None
    rpo_alpha: Optional[float] = None
    reference_free: bool = False
    label_smoothing: float = 0.
    f_divergence_type: str = 'reverse_kl'
    loss_type: Optional[str] = None

    # kto
    desirable_weight: float = 1.
    undesirable_weight: float = 1.
    calculate_KL: Optional[bool] = None

    # rm
    center_rewards_coefficient: Optional[float] = None

    # gkd
    teacher_model: Optional[str] = field(default=None)
    teacher_model_type: Optional[str] = field(default=None)
    teacher_model_revision: Optional[str] = field(default=None)
    # Optional second teacher for probability-space ensemble distillation. When set, both
    # teachers are fused into a single target distribution before the reverse-KL loss
    # (one KL, not two). When None, the pipeline is byte-identical to single-teacher GKD.
    teacher_model_2: Optional[str] = field(default=None)
    teacher_model_2_type: Optional[str] = field(default=None)
    teacher_model_2_revision: Optional[str] = field(default=None)
    # Fusion mode for the two-teacher ensemble:
    #   'prob_avg'  — true probability mixture, log-space: logsumexp_i(log w_i + log_softmax(z_i)).
    #   'logit_avg' — log-linear pool: log_softmax(sum_i w_i * z_i). A DIFFERENT distribution
    #                 from prob_avg (weighted geometric mean of probs), exposed as experimental.
    teacher_fusion: Literal['prob_avg', 'logit_avg'] = 'prob_avg'
    # Per-teacher mixture weights, one per configured teacher. None defaults to equal weights.
    # Normalized once at validation; every weight must be > 0.
    teacher_fusion_weights: Optional[List[float]] = None
    # MOPD multi-teacher distillation by data-domain routing. When `teacher_models`
    # is set it takes precedence over the
    # teacher_model_2 fusion path: each training sample carries a `domain` field, and
    # `domain_teacher_mapping` routes it to one teacher (one teacher per microbatch). KL
    # direction stays the standard `beta` (no per-route beta). teacher_models is an ordered
    # list of teacher paths; domain_teacher_mapping is a JSON dict {domain: teacher_index}.
    teacher_models: Optional[List[str]] = field(default=None)
    teacher_models_type: Optional[List[str]] = field(default=None)
    teacher_models_revision: Optional[List[str]] = field(default=None)
    domain_teacher_mapping: Optional[str] = None  # JSON: {"gui": 0, "spatial": 1}
    lmbda: float = 0.5  # On-policy probability: with prob lmbda, use student-generated responses
    seq_kd: bool = False  # Sequential KD: use teacher-generated responses when not on-policy
    offload_teacher_model: bool = False  # Offload teacher model to CPU to save GPU memory
    lazy_teacher_logits: bool = True  # GKD: True = compute teacher logits per microbatch in forward_step (save memory); False = pre-compute all in _replace_data_iterator
    overlap_rollout_training: bool = False  # GKD: True = rollout one microbatch then train it (interleave); False = rollout full batch then train all microbatches
    # GKD overlap: GLOBAL samples-per-mini-batch chunk used to interleave rollout with
    # training. Counted across the full global batch (all DP ranks), NOT per-rank. Each DP
    # rank rolls out its share = mini_batch_size // dp_size samples per vLLM request
    # (domains may mix within a chunk; vLLM is domain-agnostic). With MICRO_BATCH_SIZE=1
    # each rolled-out sample becomes its own single-domain microbatch (MBS=1 makes the
    # per-microbatch teacher grouping trivial); before chunking, the trainer stably sorts
    # each rank's samples by teacher_idx to minimize teacher body offload/load switches.
    # Must divide global_batch_size, be a positive multiple of micro_batch_size, and be a
    # multiple of dp_size so each rank owns whole per-rank chunks.
    overlap_rollout_mini_batch_size: int = 128
    sft_alpha: float = 0.0  # Weight for SFT loss in GKD (0 = pure JSD, >0 = JSD + sft_alpha * SFT)
    use_opsd: bool = False  # GKD: True = prepend reference solution to teacher prompt (OPSD) for better guidance; use with lazy_teacher_logits
    use_sft_loss: bool = False
    # GKD: floor the teacher distribution with uniform mass: T' = (1-α)T + α/V.
    # Bounds log(1/T) and tames reverse-KL gradnorm when teacher is sharply peaked.
    # 0 disables smoothing.
    teacher_label_smoothing: float = 0.0

    # Keep a teacher body resident across consecutive
    # same-teacher microbatches in MOPD domain-routing mode. Only meaningful when combined
    # with `offload_teacher_model=True` and `teacher_models` (routing), because
    # `_group_batch_by_teacher` already sorts microbatches by teacher — so consecutive
    # microbatches usually share the same teacher, and per-microbatch load+offload is redundant.
    # When on: offload the previous resident teacher and load the next ONLY when `teacher_idx`
    # actually changes. Cuts body loads from `num_microbatches` down to
    # `num_distinct_teachers_this_step`. When off: byte-identical to the legacy path
    # (`load_teacher_model_context` per microbatch).
    #
    # Default is True: for eligible runs (routed MOPD + offload_teacher_model) this is a
    # pure scheduling win with byte-identical teacher forward math. For ineligible runs
    # (single-teacher, weighted_multi_kl, or offload off) `_validate_teacher_body_reuse`
    # AUTO-DOWNGRADES this to False with an info log, and the trainer's runtime guard
    # `_use_body_reuse = teacher_body_reuse and offload_teacher_model and multi_teacher_mode`
    # is itself a no-op there — so flipping the default is safe for every existing config.
    teacher_body_reuse: bool = True

    # Skip the per-microbatch DP metric all-reduce inside GKD loss_func. When True,
    # metrics are rank-local. Optimization is unchanged because Megatron reduces
    # gradients separately. This avoids interleaving variable-shape metric
    # collectives with MoE expert/tensor-parallel collectives.
    skip_metric_dp_allreduce: bool = True

    # Diagnostic-metrics collection interval (0 = every step, N = every N steps + step 1).
    # Read by the GKD trainer; declared here so it is accepted as a CLI arg.
    # Default 1 collects the no-grad full-vocabulary diagnostics every step.
    diagnostics_interval: int = 1

    # MOPD encode-resample transient-retry knobs (truncation_strategy='delete' path only).
    # When a sample fails to encode, the trainer first RETRIES THE ORIGINAL sample a few
    # times before discarding it for a route-local replacement. This preserves
    # the intended domain mixture across transient filesystem failures.
    #   mopd_resample_retries: retries of the ORIGINAL sample before replacement (0 = legacy
    #     behaviour: never retry, replace immediately).
    #   mopd_resample_backoff_base_s: first backoff delay; each retry doubles it (exp backoff),
    #     so 3 retries -> 0.5 / 1.0 / 2.0s, worst case 3.5s per failed sample.
    #   mopd_resample_retry_budget_s: hard per-call limit on total sleep across ALL failed
    #     samples in one resample call, so a pathological burst (hundreds of failures) cannot
    #     add hours to a long run — once exceeded, remaining failures skip straight to replacement.
    # Only transient I/O (OSError/IOError, e.g. FileNotFoundError) is retried; decode/parse
    # failures (UnidentifiedImageError, ValueError, KeyError) are never retried — see the
    # classification in gkd_trainer.resample_encode_failed_inputs. Env overrides:
    # MOPD_RESAMPLE_RETRIES / MOPD_RESAMPLE_BACKOFF_BASE_S / MOPD_RESAMPLE_RETRY_BUDGET_S.
    mopd_resample_retries: int = 0
    mopd_resample_backoff_base_s: float = 0.5
    mopd_resample_retry_budget_s: float = 60.0

    # MOPD loss normalization mode. Default None = fully legacy: behavior is derived from
    # ``calculate_per_token_loss`` and NOTHING changes anywhere. The four modes:
    #   'sequence'     -- pure alias: SETS calculate_per_token_loss=False (the legacy
    #                     per-microbatch / per-sequence mean path, byte-identical).
    #   'global_token' -- pure alias: SETS calculate_per_token_loss=True (the existing
    #                     token-weighted global path, byte-identical). A long sequence is
    #                     weighted proportionally to its token count.
    #   'domain_sequence' / 'domain_token' -- PER-DOMAIN normalization for MOPD multi-teacher
    #                     routing. The global-step loss is a weighted sum over domain groups
    #                     (group key = the sample's `domain` name; domains sharing a teacher
    #                     stay separate groups), so each domain's contribution is decoupled
    #                     from its token / sample share of the batch:
    #                       domain_sequence: L = sum_d q_d * (1/B_d) * sum_{i in d}(loss_sum_i / T_i)
    #                                          -- every SAMPLE in a domain weighs equally.
    #                       domain_token:    L = sum_d q_d * (sum_{i in d} loss_sum_i) / N_d
    #                                          -- every TOKEN in a domain weighs equally.
    #                     where B_d = #samples, N_d = #valid completion tokens, T_i = sample i's
    #                     valid tokens, q_d = normalized domain weight (uniform by default),
    #                     all measured over the FULL global step (across DP). Require MOPD
    #                     routing (teacher_models + domain_teacher_mapping), calculate_per_token_loss
    #                     False (they use the 2-tuple loss protocol), sft_alpha==0,
    #                     context_parallel_size==1, micro_batch_size==1, and no
    #                     overlap_rollout_training. See gkd_trainer._compute_domain_loss_scales
    #                     for the verified Megatron scaling chain. Validated in
    #                     _init_mopd_loss_normalization.
    mopd_loss_normalization: Optional[Literal['sequence', 'global_token', 'domain_sequence',
                                              'domain_token']] = None
    # Optional per-domain loss weights for the two domain modes only: JSON dict
    # {domain_name: weight}. Keys must be a SUBSET of domain_teacher_mapping; every weight
    # must be finite and > 0; unspecified domains default to weight 1.0. Each domain is its
    # own group (keyed by domain name — domains sharing a teacher are NOT merged). Normalized
    # internally each step over the domains actually present. Setting this without a domain
    # mode is an error. Default None = uniform.
    mopd_domain_loss_weights: Optional[str] = None

    # grpo/gkd
    temperature: float = 0.9  # Temperature for sampling and loss computation

    # grpo
    generation_batch_size: Optional[int] = None
    steps_per_generation: Optional[int] = None
    num_generations: int = 8
    num_generations_eval: Optional[int] = None
    max_completion_length: int = 512
    # GSPO https://arxiv.org/abs/2507.18071
    importance_sampling_level: Literal['token', 'sequence', 'sequence_token'] = 'token'

    # SAPO https://arxiv.org/abs/2511.20347
    # Temperature parameters for soft adaptive gate
    tau_pos: float = 1.0
    tau_neg: float = 1.05

    epsilon: float = 0.2
    epsilon_high: Optional[float] = None
    delta: Optional[float] = None
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.

    use_vllm: bool = True
    vllm_mode: Optional[Literal['server', 'colocate']] = None

    vllm_enable_prefix_caching: bool = True
    vllm_gpu_memory_utilization: float = 0.9
    vllm_tensor_parallel_size: int = 1
    vllm_max_model_len: Optional[int] = None
    vllm_enforce_eager: bool = False
    vllm_limit_mm_per_prompt: Optional[Union[dict, str]] = None  # '{"image": 5, "video": 2}'
    vllm_disable_cascade_attn: bool = False
    vllm_max_num_seqs: Optional[int] = None
    vllm_mm_processor_cache_gb: Optional[float] = None
    vllm_engine_kwargs: Optional[Dict[str, Any]] = None

    sleep_level: Literal[0, 1, 2] = 0
    offload_optimizer: bool = False
    offload_model: bool = False
    offload_bridge: bool = False

    vllm_server_base_url: Optional[List[str]] = None
    vllm_server_host: Optional[List[str]] = None
    vllm_server_port: List[int] = field(default_factory=lambda: [8000])
    vllm_server_timeout: float = 240.0
    vllm_server_group_port: Optional[List[int]] = None

    reward_funcs: List[str] = field(default_factory=list)
    reward_weights: List[float] = None
    # see details in swift/plugin/orm.py
    # cosine reward, https://arxiv.org/abs/2502.03373
    cosine_min_len_value_wrong: float = -0.5  # r^w_0 in paper, Reward for wrong answers with zero completion length.
    cosine_max_len_value_wrong: float = 0.0  # r^w_L in paper, Reward for wrong answers with max completion length.
    cosine_min_len_value_correct: float = 1.0  # r^c_0 in paper, Reward for correct answers with zero completion length.
    cosine_max_len_value_correct: float = 0.5  # r^c_L in paper, Reward for correct answers with max completion length.
    cosine_max_len: Optional[int] = None  # Lmax in paper, default equal to max_completion_length
    # repetition penalty, https://arxiv.org/abs/2502.03373
    repetition_n_grams: int = 3
    repetition_max_penalty: float = -1.0
    # soft_overlong, https://arxiv.org/abs/2503.14476
    soft_max_length: Optional[int] = None
    soft_cache_length: Optional[int] = None
    # DAPO, https://arxiv.org/abs/2503.14476
    dynamic_sample: bool = False
    max_resample_times: int = 3
    overlong_filter: bool = False

    # Dr. GRPO, https://arxiv.org/abs/2503.20783
    scale_rewards: Literal['none', 'group', 'batch'] = 'group'

    # RLOO / REINFORCE++
    advantage_estimator: Literal['grpo', 'rloo', 'reinforce_plus_plus'] = 'grpo'
    kl_in_reward: bool = False

    wandb_log_unique_prompts: Optional[bool] = None
    log_completions: bool = False

    rollout_importance_sampling_mode: Optional[Literal['token_truncate', 'token_mask', 'sequence_truncate',
                                                       'sequence_mask']] = None
    rollout_importance_sampling_threshold: float = 2.0
    log_rollout_offpolicy_metrics: bool = False
    # Off-Policy Sequence Masking: mask out sequences that deviate too much from rollout policy
    # If set, compute mean(rollout_per_token_logps - per_token_logps) per sequence,
    # and mask sequences where this delta > threshold AND advantage < 0
    # Falls back to old_per_token_logps if rollout_per_token_logps is not available
    off_policy_sequence_mask_delta: Optional[float] = None

    # entropy
    log_entropy: bool = False
    # Beyond the 80/20 Rule, https://arxiv.org/abs/2506.01939
    top_entropy_quantile: float = 1.0

    # ───────────────────────────  Not Supported Yet  ───────────────────────────

    # reward model
    reward_model: Optional[List[str]] = None
    reward_model_plugin: Optional[List[str]] = None
    # sync ref model
    sync_ref_model: bool = False
    ref_model_sync_steps: int = 512
    ref_model_mixup_alpha: float = 0.6

    async_generate: bool = False

    move_model_batches: Optional[int] = None

    # multi turn
    multi_turn_scheduler: Optional[str] = None
    max_turns: Optional[int] = None
    completion_length_limit_scope: Literal['total', 'per_round'] = 'per_round'
    vllm_server_pass_dataset: bool = False

    num_iterations: int = 1

    # dataset
    dataset_shuffle: Optional[bool] = True

    def _init_kto(self):
        if self.calculate_KL is None:
            # Not all losses require a KL calculation
            self.calculate_KL = True
            if self.loss_type in ['apo_zero_unpaired']:
                self.calculate_KL = False

    def __post_init__(self):
        if self.rlhf_type is None:
            return
        default_loss_type = {'kto': 'kto', 'dpo': 'sigmoid', 'grpo': 'grpo'}
        default_beta = {'gkd': 0.5, 'grpo': 0.04}
        if self.beta is None:
            self.beta = default_beta.get(self.rlhf_type, 0.1)
        if self.loss_type is None:
            self.loss_type = default_loss_type.get(self.rlhf_type)
        if self.rlhf_type == 'kto':
            self._init_kto()
        if self.rlhf_type == 'gkd':
            self._init_gkd_ensemble()
            self._init_mopd_loss_normalization()
            self._validate_teacher_body_reuse()
        if self.rlhf_type == 'grpo':
            assert self.vllm_mode is not None, 'vllm_mode is required for Megatron GRPO'
            self._init_grpo()
            if self.vllm_limit_mm_per_prompt is not None:
                self.vllm_limit_mm_per_prompt = json_parse_to_dict(self.vllm_limit_mm_per_prompt)
            self.vllm_engine_kwargs = json_parse_to_dict(self.vllm_engine_kwargs)

    def _init_gkd_ensemble(self):
        """Validate and normalize the multi-teacher configuration.

        Two independent multi-teacher modes:
          - MOPD domain routing (`teacher_models` set): per-sample `domain` routes to one
            teacher via `domain_teacher_mapping`; standard `beta` KL. Takes precedence.
          - Two-teacher probability fusion (`teacher_model_2` set): fuse into one target.
        Single-teacher GKD (neither set) is left untouched / byte-identical.
        """
        # MOPD domain-routing mode.
        if self.teacher_models is not None:
            if len(self.teacher_models) == 0:
                raise ValueError('teacher_models is set but empty.')
            num_teachers = len(self.teacher_models)
            if self.domain_teacher_mapping is None:
                raise ValueError('domain_teacher_mapping is required when teacher_models is set.')
            mapping = json_parse_to_dict(self.domain_teacher_mapping) \
                if isinstance(self.domain_teacher_mapping, str) else self.domain_teacher_mapping
            if not isinstance(mapping, dict) or len(mapping) == 0:
                raise ValueError(
                    f'domain_teacher_mapping must be a non-empty JSON dict {{domain: teacher_index}}, '
                    f'got {self.domain_teacher_mapping!r}.')
            for domain, idx in mapping.items():
                if not isinstance(idx, int) or idx < 0 or idx >= num_teachers:
                    raise ValueError(
                        f'domain_teacher_mapping["{domain}"]={idx} out of range [0, {num_teachers}).')
            # Store the parsed mapping back so the trainer reads a dict (not a JSON string).
            self.domain_teacher_mapping = mapping
            if self.teacher_model_2 is not None:
                raise ValueError('teacher_models (domain routing) and teacher_model_2 (fusion) are '
                                 'mutually exclusive; set only one.')
            if not self.lazy_teacher_logits:
                raise ValueError('lazy_teacher_logits=False is not supported with teacher_models; '
                                 'set lazy_teacher_logits=True.')
            # MOPD domain routing requires MBS=1 UNCONDITIONALLY (not only under overlap,
            # where a second copy of this check lives below): microbatches must be
            # single-sample so each is single-domain — the route stamps, the per-domain
            # metric accumulation and domain loss scales all read the microbatch's single raw
            # sample (raw_batch[0]). With MBS>1 a mixed-domain microbatch would silently
            # mis-attribute metrics/scales to the first sample's domain. Fail fast.
            _mbs = int(getattr(self, 'micro_batch_size', 1) or 1)
            if _mbs != 1:
                raise ValueError(
                    f'micro_batch_size must be 1 when teacher_models is set (MOPD domain '
                    f'routing tiles one single-domain microbatch per sample), got {_mbs}.')
            # MOPD + overlap: overlap_rollout_mini_batch_size is a GLOBAL sample count.
            # The producer walks each rank's share (mini_bs // dp_size samples per chunk)
            # after stably sorting by teacher_idx; with MBS=1 each rolled-out sample is its
            # own single-domain microbatch (teacher_idx tagged per microbatch). Requires
            # mini_bs to divide global_batch_size cleanly and be a positive multiple of
            # micro_batch_size. Note: dp_size may not be known here yet (constructed later),
            # so the dp_size-divisibility check is done defensively in the trainer producer.
            if getattr(self, 'overlap_rollout_training', False):
                mini_bs = int(getattr(self, 'overlap_rollout_mini_batch_size', 0) or 0)
                mbs = int(getattr(self, 'micro_batch_size', 1) or 1)
                gbs = int(getattr(self, 'global_batch_size', 0) or 0)
                if mini_bs <= 0:
                    raise ValueError(
                        f'overlap_rollout_mini_batch_size must be > 0 when overlap_rollout_training '
                        f'is enabled with teacher_models (MOPD), got {mini_bs}.')
                # The MOPD overlap producer emits one single-domain microbatch per rolled-out
                # sample (teacher_idx tagged per microbatch), which is only correct when each
                # microbatch is exactly one sample. With MBS>1 the producer would queue
                # num_microbatches*MBS single-sample microbatches into a schedule expecting
                # MBS-sized batches -> late abort or corrupted training step. Fail fast.
                if mbs != 1:
                    raise ValueError(
                        f'micro_batch_size must be 1 when overlap_rollout_training is enabled with '
                        f'teacher_models (MOPD domain routing); the overlap producer tiles one '
                        f'single-domain microbatch per sample. Got micro_batch_size={mbs}.')
                if mini_bs % mbs != 0:
                    raise ValueError(
                        f'overlap_rollout_mini_batch_size ({mini_bs}) must be a positive multiple of '
                        f'micro_batch_size ({mbs}) so each mini-batch splits into whole microbatches.')
                if gbs > 0 and gbs % mini_bs != 0:
                    raise ValueError(
                        f'overlap_rollout_mini_batch_size ({mini_bs}) must divide global_batch_size '
                        f'({gbs}) so the global batch tiles cleanly into mini-batches.')
            # Reject pipeline parallelism: teacher forward uses PP send/recv collectives, and
            # per-microbatch teacher routing can select different teacher stages across a PP
            # group -> mismatched activations / hang. This recipe uses PP=1; fail fast.
            pp_size = int(getattr(self, 'pipeline_model_parallel_size', 1) or 1)
            if pp_size > 1:
                raise ValueError(
                    f'pipeline_model_parallel_size={pp_size} is not supported with teacher_models '
                    '(MOPD domain routing): per-microbatch teacher routing can mismatch teacher '
                    'stages across a pipeline group. Use pipeline_model_parallel_size=1.')
            # Reject OPSD: MOPD groups/reorders samples by domain, so the positional OPSD
            # ref-answer mapping is invalid. It is silently skipped in the trainer for MOPD;
            # fail fast here instead of ignoring a requested option.
            if getattr(self, 'use_opsd', False):
                raise ValueError(
                    'use_opsd is not supported with teacher_models (MOPD domain routing): OPSD '
                    'ref-answer mapping is positional and MOPD reorders samples by domain. '
                    'Disable use_opsd.')
            # If teacher type/revision lists are given, they must match teacher_models length.
            for name in ('teacher_models_type', 'teacher_models_revision'):
                val = getattr(self, name, None)
                if val is not None and len(val) != num_teachers:
                    raise ValueError(
                        f'{name} has length {len(val)} but must match teacher_models '
                        f'({num_teachers}) when provided.')
            return

        # Ordered list of configured teacher paths. Index 0 is the primary teacher.
        teachers = [self.teacher_model]
        if self.teacher_model_2 is not None:
            teachers.append(self.teacher_model_2)
        num_teachers = len([t for t in teachers if t is not None])

        if self.teacher_fusion not in ('prob_avg', 'logit_avg'):
            raise ValueError(
                f"teacher_fusion must be one of 'prob_avg' or 'logit_avg', got '{self.teacher_fusion}'.")

        single_teacher = self.teacher_model_2 is None
        if single_teacher:
            # Nothing else to validate; do NOT mutate any state that would change the
            # byte-identical single-teacher path. Leave teacher_fusion_weights as-is.
            return

        # --- Two-teacher constraints ---
        # Force lazy teacher logits: non-lazy would precompute/store two full teacher
        # tensors per encoded batch, which is unacceptable memory.
        if not self.lazy_teacher_logits:
            raise ValueError(
                'lazy_teacher_logits=False is not supported with two teachers '
                '(teacher_model_2 set). Non-lazy mode would precompute and store two full '
                'teacher logit tensors per batch. Set lazy_teacher_logits=True.')

        # Reject context parallelism: the fused-teacher helper builds the completion mask
        # from full-sequence labels, while the per-teacher forward CP-slices its logits to
        # the CP-local sequence. Mixing a full-sequence mask with CP-local teacher logits
        # would misalign (or shape-mismatch) the fused target. CP-sliced fusion is out of
        # scope for this version; fail fast rather than fuse over mismatched sequences.
        cp_size = int(getattr(self, 'context_parallel_size', 1) or 1)
        if cp_size > 1:
            raise ValueError(
                f'context_parallel_size={cp_size} is not supported with two teachers '
                '(teacher_model_2 set): the fused-teacher mask is built from full-sequence '
                'labels while teacher logits are CP-sliced, which would misalign the fused '
                'target. Use context_parallel_size=1 for two-teacher ensemble distillation.')

        # Reject pipeline parallelism: the fused helper forwards teachers sequentially and
        # returns None on a non-last PP rank. Under PP>1 that early-return makes non-last
        # ranks skip the second teacher's forward while the last rank expects all teachers'
        # activations -> pipeline hang / mismatched activations. A PP-correct multi-teacher
        # forward is out of scope this version (this recipe uses PP=1); fail fast.
        pp_size = int(getattr(self, 'pipeline_model_parallel_size', 1) or 1)
        if pp_size > 1:
            raise ValueError(
                f'pipeline_model_parallel_size={pp_size} is not supported with two teachers '
                '(teacher_model_2 set): the sequential fused-teacher forward would skip the '
                'second teacher on non-last PP ranks and deadlock the pipeline. Use '
                'pipeline_model_parallel_size=1 for two-teacher ensemble distillation.')

        # Reject OPSD: the OPSD completion-row mapping (opsd_completion_start) indexes the
        # UNROLLED labels, but the two-teacher fused mask uses the loss-aligned (rolled for
        # causal_lm) labels. Reconciling those two alignments for OPSD is not safe in this
        # version, so OPSD ensemble distillation is out of scope; fail fast.
        if getattr(self, 'use_opsd', False):
            raise ValueError(
                'use_opsd is not supported with two teachers (teacher_model_2 set): the OPSD '
                'completion mapping indexes unrolled labels while the fused mask uses the '
                'rolled student-loss labels, which would misalign the fused target. Disable '
                'use_opsd for two-teacher ensemble distillation.')

        # Normalize fusion weights ONCE here so every rank reads the identical tuple.
        if self.teacher_fusion_weights is None:
            weights = [1.0 / num_teachers] * num_teachers
        else:
            weights = list(self.teacher_fusion_weights)
            if len(weights) != num_teachers:
                raise ValueError(
                    f'teacher_fusion_weights must have one weight per teacher '
                    f'({num_teachers}), got {len(weights)}.')
            # Reject non-finite weights BEFORE the > 0 check: NaN slips past `w <= 0`
            # (nan comparisons are False) and inf poisons normalization, and either turns
            # math.log(w) / the fused target into NaNs downstream.
            if any(not math.isfinite(w) for w in weights):
                raise ValueError(
                    f'teacher_fusion_weights must all be finite, got {weights}.')
            if any(w <= 0 for w in weights):
                raise ValueError(
                    f'teacher_fusion_weights must all be > 0, got {weights}. '
                    'A zero/negative weight is rejected (skipping a teacher is out of scope).')
            total = sum(weights)
            weights = [w / total for w in weights]
        self.teacher_fusion_weights = weights


    def _init_mopd_loss_normalization(self):
        """Validate the MOPD loss-normalization mode and optional per-domain weights.

        Runs AFTER ``_init_gkd_ensemble`` so ``domain_teacher_mapping`` is already parsed
        into a dict on the MOPD routing path. Contract:

          - None -> fully legacy; behavior derived from ``calculate_per_token_loss``.
            Nothing is mutated (and ``mopd_domain_loss_weights`` must be unset).
          - 'sequence' / 'global_token' -> pure aliases that SET
            ``calculate_per_token_loss`` False / True. The RLHF default is False, so an
            explicitly-set True is the only cleanly-detectable value: we fail iff the user
            asked for 'sequence' (target False) while calculate_per_token_loss is True.
            For 'global_token' (target True) a currently-False value is indistinguishable
            from the default, so we simply set it True (documented limitation).
          - 'domain_sequence' / 'domain_token' -> per-domain normalization; require MOPD
            multi-teacher routing, calculate_per_token_loss False, sft_alpha==0,
            context_parallel_size==1, micro_batch_size==1, and no overlap_rollout_training.
          - ``mopd_domain_loss_weights`` is only meaningful for a domain mode; keys must be
            a subset of ``domain_teacher_mapping`` and every weight finite and > 0.
        """
        mode = self.mopd_loss_normalization
        weights_raw = self.mopd_domain_loss_weights

        if mode is None:
            if weights_raw is not None:
                raise ValueError(
                    'mopd_domain_loss_weights requires mopd_loss_normalization to be a '
                    "domain mode ('domain_sequence' or 'domain_token').")
            return

        # RLHF default for calculate_per_token_loss is False (rlhf_args.py); a True value
        # can therefore only have been passed explicitly. That is the sole conflict this
        # dataclass flow can detect without a provenance sentinel.
        cptl = bool(self.calculate_per_token_loss)

        if mode in ('sequence', 'global_token'):
            if weights_raw is not None:
                raise ValueError(
                    'mopd_domain_loss_weights is only meaningful for the domain '
                    f"normalization modes, not '{mode}'.")
            target = (mode == 'global_token')
            if cptl and not target:
                raise ValueError(
                    "mopd_loss_normalization='sequence' conflicts with "
                    'calculate_per_token_loss=True. Drop the redundant '
                    'calculate_per_token_loss flag; the mode alias sets it.')
            self.calculate_per_token_loss = target
            return

        # --- domain modes ---
        if self.teacher_models is None or not self.domain_teacher_mapping:
            raise ValueError(
                f"mopd_loss_normalization='{mode}' requires MOPD multi-teacher routing "
                '(set teacher_models and domain_teacher_mapping).')
        if cptl:
            raise ValueError(
                f"mopd_loss_normalization='{mode}' uses the per-sequence (2-tuple) loss "
                'protocol and is incompatible with calculate_per_token_loss=True. Drop the '
                'calculate_per_token_loss flag.')
        # Pin False so the trainer's 2-tuple loss branch is taken.
        self.calculate_per_token_loss = False
        # Reject ANY non-zero sft_alpha, including negative and NaN. `not (x == 0)` is True
        # for NaN (NaN == 0 is False) and for negatives, which a plain `> 0` check would let
        # through to the trainer's `sft_alpha > 0` gate and silently treat as disabled.
        if not (self.sft_alpha == 0):
            raise ValueError(
                f"mopd_loss_normalization='{mode}' does not define the SFT auxiliary term "
                f'(sft_alpha={self.sft_alpha}); set sft_alpha=0. Pure MOPD is lambda=1 '
                'pure on-policy, so this costs nothing.')
        cp_size = int(getattr(self, 'context_parallel_size', 1) or 1)
        if cp_size > 1:
            raise ValueError(
                f"mopd_loss_normalization='{mode}' requires context_parallel_size==1 "
                f'(got {cp_size}); the per-domain loss/token sums are CP-local by construction.')
        mbs = int(getattr(self, 'micro_batch_size', 1) or 1)
        if mbs != 1:
            raise ValueError(
                f"mopd_loss_normalization='{mode}' requires micro_batch_size==1 (got {mbs}); "
                'each microbatch must be one single-domain sample so the per-domain sample '
                'and token counts are well defined.')
        if bool(getattr(self, 'overlap_rollout_training', False)):
            raise ValueError(
                f"mopd_loss_normalization='{mode}' is not supported with "
                'overlap_rollout_training: the per-domain counts need one fixed-shape DP '
                'all-reduce over the whole step before any forward, which the interleaved '
                'overlap producer cannot provide.')

        if weights_raw is not None:
            weights = json_parse_to_dict(weights_raw) if isinstance(weights_raw, str) else weights_raw
            if not isinstance(weights, dict) or len(weights) == 0:
                raise ValueError(
                    'mopd_domain_loss_weights must be a non-empty JSON dict {domain: weight}, '
                    f'got {weights_raw!r}.')
            mapping_keys = set(self.domain_teacher_mapping.keys())
            for domain, w in weights.items():
                if domain not in mapping_keys:
                    raise ValueError(
                        f'mopd_domain_loss_weights domain "{domain}" is not in '
                        f'domain_teacher_mapping (available: {sorted(mapping_keys)}).')
                if isinstance(w, bool) or not isinstance(w, (int, float)) \
                        or not math.isfinite(w) or w <= 0:
                    raise ValueError(
                        f'mopd_domain_loss_weights["{domain}"]={w!r} must be a finite '
                        'weight > 0.')
            # Store back a plain float dict so the trainer reads it directly.
            self.mopd_domain_loss_weights = {k: float(v) for k, v in weights.items()}

    def _validate_teacher_body_reuse(self):
        """Validate the teacher body-reuse optimization.

        The optimization keeps a teacher's body resident across consecutive same-teacher
        microbatches, avoiding redundant load/offload cycles. It is only meaningful when
        (a) `offload_teacher_model=True` (otherwise bodies are already resident) and
        (b) MOPD domain routing (`teacher_models` set) is active — that is the routed
        path where `_group_batch_by_teacher` sorts microbatches by teacher, so residency
        is what actually shortens the schedule.

        Ineligible configurations are downgraded to ``False`` with one
        explanatory log entry so the effective configuration remains accurate.
        """
        if not self.teacher_body_reuse:
            return
        eligible = self.offload_teacher_model and (self.teacher_models is not None)
        if eligible:
            return
        # Explain WHY it was downgraded so a reader inspecting the run log can tell
        # whether to flip offload on or switch to routed MOPD if they wanted the speedup.
        if not self.offload_teacher_model and self.teacher_models is None:
            reason = ('offload_teacher_model=False and no teacher_models (single-teacher or '
                      'weighted_multi_kl run): teachers are already resident and there is no '
                      'routing sort to exploit')
        elif not self.offload_teacher_model:
            reason = 'offload_teacher_model=False: teachers already resident, no load/offload to elide'
        else:
            reason = ('teacher_models is None (single-teacher or weighted_multi_kl run): the '
                      'residency schedule only pays off on the routed path where '
                      '_group_batch_by_teacher sorts microbatches by teacher')
        logger.info(
            f'teacher_body_reuse auto-disabled: {reason}. This run uses '
            'per-microbatch load_teacher_model_context.')
        self.teacher_body_reuse = False

    def _init_grpo(self):

        def _check_not_supported():
            if self.async_generate:
                raise ValueError('async_generate is not supported for Megatron GRPO right now')
            if self.sync_ref_model:
                raise ValueError('sync_ref_model is not supported for Megatron GRPO right now')
            if not self.dataset_shuffle:
                raise ValueError('dataset_shuffle false is not supported for Megatron GRPO')
            if self.multi_turn_scheduler:
                raise ValueError('multi_turn_scheduler is not supported for Megatron GRPO right now')
            if self.num_iterations > 1:
                raise ValueError('num_iterations > 1 is not supported for Megatron GRPO right now')

        def _check_batch_params():
            # Set default values if both are None
            if self.generation_batch_size is None and self.steps_per_generation is None:
                self.steps_per_generation = 1
                self.generation_batch_size = self.global_batch_size * self.steps_per_generation
            # Both configured - error
            elif self.generation_batch_size is not None and self.steps_per_generation is not None:
                raise ValueError("'generation_batch_size' and 'steps_per_generation' cannot be both configured")
            # Only generation_batch_size configured
            elif self.generation_batch_size is not None:
                if self.generation_batch_size % self.global_batch_size != 0:
                    raise ValueError(f'generation_batch_size ({self.generation_batch_size}) '
                                     f'must be divisible by global_batch_size ({self.global_batch_size})')
                self.steps_per_generation = self.generation_batch_size // self.global_batch_size
            # Only steps_per_generation configured
            else:
                self.generation_batch_size = self.global_batch_size * self.steps_per_generation

            world_size = torch.distributed.get_world_size()
            dp_size = world_size // (
                self.pipeline_model_parallel_size * self.tensor_model_parallel_size * self.context_parallel_size)
            num_rollout_prompt = self.generation_batch_size // self.num_generations
            if num_rollout_prompt % dp_size != 0:
                raise ValueError(f'num_rollout_prompt ({num_rollout_prompt}) = generation_batch_size '
                                 f'({self.generation_batch_size}) // num_generations ({self.num_generations}) '
                                 f'must be divisible by dp_size ({dp_size}). '
                                 f'Please adjust generation_batch_size/steps_per_generation/num_generations.')

            per_device_num_rollout_prompt = num_rollout_prompt // dp_size
            assert per_device_num_rollout_prompt >= 1, \
                (f'per_device_num_rollout_prompt ({per_device_num_rollout_prompt}) must be greater than 1, '
                 f'please adjust generation_batch_size/steps_per_generation/num_generations to make it greater than 1')

            if per_device_num_rollout_prompt % self.micro_batch_size != 0:
                raise ValueError(f'Per-device rollout prompt count ({per_device_num_rollout_prompt}) = '
                                 f'(generation_batch_size ({self.generation_batch_size}) // '
                                 f'num_generations ({self.num_generations})) // dp_size ({dp_size}) '
                                 f'must be divisible by micro_batch_size ({self.micro_batch_size}). '
                                 f'Please adjust arguments to satisfy: '
                                 f'(generation_batch_size // num_generations) // dp_size % '
                                 f'micro_batch_size == 0')

            self.per_device_generation_batch_size = self.generation_batch_size // world_size
            assert self.per_device_generation_batch_size >= 1, \
                (f'per_device_generation_batch_size ({self.per_device_generation_batch_size}) must be greater than 1, '
                 f'please adjust generation_batch_size/steps_per_generation/num_generations to make it greater than 1')

        _check_not_supported()
        _check_batch_params()
        self.remove_unused_columns = False
        logger.info(f'Setting args.remove_unused_columns: {self.remove_unused_columns}')
        if self.truncation_strategy is None:
            self.truncation_strategy = 'left'
        assert self.truncation_strategy in ['left', 'delete'
                                            ], ("GRPO requires `truncation_strategy 'left' or 'delete'`, "
                                                f"Current value: `truncation_strategy='{self.truncation_strategy}'`."
                                                )  # noqa
        if self.beta is None:
            self.beta = 0.04  # https://arxiv.org/abs/2402.03300
        if self.async_generate:
            logger.info('Using async mode. This is a approximate version which '
                        'will use the old weights to generate responses to accelerate. '
                        'This will ignore the `CLIP` of advantages, if you found the training '
                        'is unstable, you may consider using --async_generate false.')
        if 'soft_overlong' in self.reward_funcs:
            assert self.soft_cache_length is not None, \
                'The soft_cache_length must be set when using soft overlong rewards.'
            if self.soft_max_length is None:
                self.soft_max_length = self.max_completion_length
                logger.info(f'Auto-configured soft_max_length = max_completion_length {self.max_completion_length}')
        assert self.use_vllm, 'use_vllm must be True for Megatron GRPO'


@dataclass
class MegatronTunerMixin:
    train_type: Literal['lora', 'full'] = 'full'
    freeze_llm: bool = False
    freeze_vit: bool = True
    freeze_aligner: bool = True
    # full
    freeze_parameters: List[str] = field(default_factory=list)
    freeze_parameters_regex: Optional[str] = None
    freeze_parameters_ratio: float = 0.  # 0 ~ 1
    trainable_parameters: List[str] = field(default_factory=list)
    trainable_parameters_regex: Optional[str] = None
    # lora
    adapter_load: Optional[str] = None
    target_modules: List[str] = field(default_factory=lambda: ['all-linear'])
    target_regex: Optional[str] = None
    modules_to_save: List[str] = field(default_factory=list)

    # lora
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: Literal['none', 'all'] = 'none'
    lora_dtype: Literal['float16', 'bfloat16', 'float32', None] = None
    use_rslora: bool = False

    def __post_init__(self):
        if 0 < self.freeze_parameters_ratio < 1 and self.pipeline_model_parallel_size > 1:
            raise ValueError('`freeze_parameters_ratio` is not supported when `pipeline_model_parallel_size` > 1')
        if self.target_regex:
            self.target_modules = self.target_regex


@dataclass
class ExtraMegatronArguments(RLHFMegatronArgumentsMixin, MegatronTunerMixin):
    check_model: bool = True
    padded_vocab_size: Optional[int] = None
    initialize_embedding: bool = False
    rope_scaling: Optional[Union[dict, str]] = None
    torch_dtype: Optional[Union[torch.dtype, str]] = None
    padding_free: bool = True
    mlp_padding_free: bool = False
    # mcore-bridge
    model: Optional[str] = None
    model_type: Optional[str] = None
    load_safetensors: Optional[bool] = None
    save_safetensors: bool = True
    adapters: List[str] = field(default_factory=list)
    ref_model: Optional[str] = None
    ref_adapters: List[str] = field(default_factory=list)
    use_hf: bool = False
    # None: use env var `MODELSCOPE_API_TOKEN`
    hub_token: Optional[str] = field(
        default=None, metadata={'help': 'SDK token can be found in https://modelscope.cn/my/myaccesstoken'})
    merge_lora: Optional[bool] = None
    max_shard_size: str = '5GB'

    # dataloader
    train_dataloader_shuffle: bool = True
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 2
    group_by_length: bool = False
    mopd_domain_balanced_sampler: bool = False

    architectures: Optional[str] = None
    llm_architectures: Optional[str] = None
    max_epochs: Optional[int] = None
    enable_dft_loss: bool = False
    enable_channel_loss: bool = False
    task_type: Literal['causal_lm', 'seq_cls'] = None
    num_labels: Optional[int] = None
    problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] = None
    save_strategy: Literal['steps', 'epoch'] = 'steps'
    mopd_prune_old_optimizer_states: bool = False

    original_max_position_embeddings: Optional[int] = None
    partial_rotary_factor: Optional[float] = None
    use_shared_expert_gate: Optional[bool] = None

    report_to: Optional[Literal['wandb', 'swanlab']] = None

    # visual
    vit_gradient_checkpointing: bool = True
    vit_lr: Optional[float] = None
    aligner_lr: Optional[float] = None
    gradient_checkpointing_kwargs: Optional[Union[dict, str]] = None
    # qwen3_next
    linear_num_value_heads: Optional[int] = None
    linear_num_key_heads: Optional[int] = None
    linear_key_head_dim: Optional[int] = None
    linear_value_head_dim: Optional[int] = None
    linear_conv_kernel_dim: Optional[int] = None
    layer_types: Optional[List[str]] = None
    # qwen3_vl, qwen3_omni
    mrope_interleaved: Optional[bool] = None

    @staticmethod
    def load_args_config(ckpt_dir: Optional[str]) -> Dict[str, Any]:
        res = {}
        if ckpt_dir is None:
            return res
        args_path = os.path.join(ckpt_dir, 'args.json')
        if os.path.exists(args_path):
            with open(args_path, 'r', encoding='utf-8') as f:
                old_args = json.load(f)
            keys = list(f.name for f in fields(MegatronTunerMixin))
            keys += ['load', 'padded_vocab_size', 'task_type', 'num_labels']
            for key in keys:
                old_value = old_args.get(key)
                if old_value is not None:
                    res[key] = old_value
            res.pop('adapter_load', None)
            if res['train_type'] != 'lora':
                res.pop('load', None)
        return res


@dataclass
class MegatronArguments(ExtraMegatronArguments):
    # training
    micro_batch_size: int = 1
    global_batch_size: int = 16
    recompute_granularity: Literal['selective', 'full'] = 'selective'
    recompute_method: Literal['uniform', 'block'] = None
    recompute_num_layers: Optional[int] = None
    recompute_modules: List[str] = field(default_factory=lambda: ['core_attn'])
    use_cpu_initialization: bool = False
    deterministic_mode: bool = False
    train_iters: Optional[int] = None
    log_interval: int = 5
    tensorboard_dir: Optional[str] = None
    no_masked_softmax_fusion: bool = False
    no_bias_dropout_fusion: Optional[bool] = None
    no_bias_swiglu_fusion: bool = False
    no_rope_fusion: Optional[bool] = None
    no_gradient_accumulation_fusion: bool = False
    cross_entropy_loss_fusion: bool = False
    cross_entropy_fusion_impl: Literal['native', 'te'] = 'native'
    calculate_per_token_loss: Optional[bool] = None
    use_flash_attn: bool = False
    attention_backend: str = 'flash'  # flash, fused, unfused, local, auto
    optimizer: Literal['adam', 'sgd'] = 'adam'
    optimizer_cpu_offload: bool = False
    optimizer_offload_fraction: float = 1.
    use_precision_aware_optimizer: bool = False
    main_grads_dtype: Literal['fp32', 'bf16'] = 'fp32'
    main_params_dtype: Literal['fp32', 'fp16'] = 'fp32'
    exp_avg_dtype: Literal['fp32', 'fp16', 'bf16', 'fp8'] = 'fp32'
    exp_avg_sq_dtype: Literal['fp32', 'fp16', 'bf16', 'fp8'] = 'fp32'
    dataloader_type: Literal['single', 'cyclic', 'external'] = 'cyclic'
    manual_gc: bool = False
    manual_gc_interval: int = 0

    # learning rate
    lr: Optional[float] = None
    lr_decay_style: Literal['cosine', 'linear', 'constant'] = 'cosine'
    # The default is None, which will be set to `train_iters`.
    lr_decay_iters: Optional[int] = None
    lr_warmup_iters: int = 0
    lr_warmup_fraction: Optional[float] = None
    min_lr: float = 0

    # regularization
    weight_decay: float = 0.1
    clip_grad: float = 1.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    sgd_momentum: float = 0.9

    # checkpoint
    save: Optional[str] = None
    save_interval: int = 500
    save_retain_interval: Optional[int] = None
    no_save_optim: bool = False
    no_save_rng: bool = False
    load: Optional[str] = None
    no_load_optim: bool = False
    no_load_rng: bool = False
    finetune: bool = False
    ckpt_format: Literal['torch', 'torch_dist', 'zarr'] = 'torch_dist'
    no_initialization: bool = True
    auto_detect_ckpt_format: bool = True
    exit_on_missing_checkpoint: bool = True
    async_save: bool = False
    use_persistent_ckpt_worker: bool = False
    ckpt_fully_parallel_load: bool = False
    ckpt_assume_constant_structure: bool = False

    # dist
    distributed_backend: Literal['nccl', 'gloo'] = 'nccl'
    local_rank: Optional[int] = None
    use_distributed_optimizer: bool = True
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    decoder_first_pipeline_num_layers: Optional[int] = None
    decoder_last_pipeline_num_layers: Optional[int] = None
    account_for_embedding_in_pipeline_split: bool = False
    account_for_loss_in_pipeline_split: bool = False

    sequence_parallel: bool = False
    context_parallel_size: int = 1
    tp_comm_overlap: bool = False
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False
    distributed_timeout_minutes: int = 300000
    num_layers_per_virtual_pipeline_stage: Optional[int] = None
    num_virtual_stages_per_pipeline_rank: Optional[int] = None
    microbatch_group_size_per_virtual_pipeline_stage: Optional[int] = None
    pipeline_model_parallel_layout: Optional[str] = None

    # model
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    ffn_hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    group_query_attention: Optional[bool] = None
    num_query_groups: Optional[int] = None
    softmax_type: Optional[Literal['vanilla', 'off-by-one', 'learnable']] = None
    window_size: Optional[str] = None
    window_attn_skip_freq: Optional[str] = None
    max_position_embeddings: Optional[int] = None
    position_embedding_type: Optional[Literal['learned_absolute', 'rope', 'mrope', 'relative', 'none']] = None
    mrope_section: Optional[List[int]] = None
    rotary_base: Optional[int] = None
    rotary_percent: float = 1.
    rotary_interleaved: Optional[bool] = None
    normalization: Literal['LayerNorm', 'RMSNorm'] = 'RMSNorm'
    norm_epsilon: Optional[float] = None
    swiglu: Optional[bool] = None
    quick_geglu: Optional[bool] = None
    activation_func_clamp_value: Optional[float] = None
    glu_linear_offset: Optional[float] = None
    untie_embeddings_and_output_weights: Optional[bool] = None
    disable_bias_linear: Optional[bool] = None
    add_qkv_bias: Optional[bool] = None
    attention_dropout: Optional[float] = None
    hidden_dropout: float = 0.
    kv_channels: Optional[int] = None
    qk_layernorm: Optional[bool] = None
    qk_l2_norm: Optional[bool] = None
    no_rope_freq: Optional[int] = None
    moe_apply_probs_on_input: Optional[bool] = None
    transformer_impl: Literal['local', 'transformer_engine'] = 'transformer_engine'

    # moe
    num_experts: Optional[int] = None
    moe_layer_freq: Optional[str] = None
    moe_ffn_hidden_size: Optional[int] = None
    moe_shared_expert_intermediate_size: Optional[int] = None

    moe_router_topk: Optional[int] = None
    moe_router_num_groups: Optional[int] = None
    moe_router_group_topk: Optional[int] = None
    moe_router_pre_softmax: Optional[bool] = None
    moe_router_dtype: Literal['none', 'fp32', 'fp64'] = 'fp32'
    moe_router_score_function: Literal['sigmoid', 'softmax'] = None
    moe_router_bias_update_rate: Optional[float] = None
    moe_router_enable_expert_bias: Optional[bool] = None
    moe_router_topk_scaling_factor: Optional[float] = None
    moe_router_load_balancing_type: Literal['aux_loss', 'seq_aux_loss', 'global_aux_loss', 'sinkhorn', 'none'] = None

    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    moe_token_dispatcher_type: Literal['allgather', 'alltoall', 'flex', 'alltoall_seq'] = 'alltoall'
    moe_enable_deepep: bool = False
    moe_grouped_gemm: bool = True
    moe_permute_fusion: bool = False
    moe_aux_loss_coeff: float = 0.
    moe_z_loss_coeff: Optional[float] = None
    moe_shared_expert_overlap: bool = False
    moe_layer_recompute: bool = False
    moe_expert_capacity_factor: Optional[float] = None
    moe_pad_expert_input_to_capacity: bool = False
    moe_token_drop_policy: Literal['probs', 'position'] = 'probs'

    # mla
    multi_latent_attention: Optional[bool] = None
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_head_dim: Optional[int] = None
    qk_pos_emb_head_dim: Optional[int] = None

    # mtp
    mtp_num_layers: Optional[int] = None
    mtp_loss_scaling_factor: float = 0.1

    # fp8
    fp8_format: Literal['e4m3', 'hybrid'] = None
    fp8_recipe: Literal['tensorwise', 'delayed', 'mxfp8', 'blockwise'] = 'delayed'
    fp8_amax_history_len: int = 1024
    fp8_amax_compute_algo: Literal['most_recent', 'max'] = 'max'
    fp8_param_gather: bool = False

    # mixed precision
    fp16: Optional[bool] = None
    bf16: Optional[bool] = None
    apply_query_key_layer_scaling: Optional[bool] = None
    attention_softmax_in_fp32: bool = True

    # logging
    log_params_norm: bool = False
    log_throughput: bool = False
    tensorboard_log_interval: int = 1
    tensorboard_queue_size: int = 50
    log_timers_to_tensorboard: bool = True
    no_log_learning_rate_to_tensorboard: bool = False
    log_validation_ppl_to_tensorboard: bool = True
    log_memory_to_tensorboard: bool = True
    logging_level: Optional[str] = None
    wandb_project: str = 'megatron-swift'
    wandb_exp_name: Optional[str] = None
    wandb_save_dir: Optional[str] = None

    # evaluate
    eval_iters: int = -1
    eval_interval: Optional[int] = None

    # other
    seed: int = 42
    seq_length: Optional[int] = None
    num_workers: int = 4
    no_data_sharding: bool = False

    # extra_args for megatron
    megatron_extra_kwargs: Optional[Union[dict, str]] = None

    def _set_default(self):
        if self.mlp_padding_free and (self.sequence_parallel or self.context_parallel_size > 1):
            raise ValueError('mlp_padding_free is not compatible with sequence parallel or context parallel.')
        if self.local_rank is None:
            self.local_rank = get_dist_setting()[1]
        if self.lr is None:
            if self.train_type == 'full':
                self.lr = 1e-5
            else:
                self.lr = 1e-4
        if self.num_query_groups is None:
            self.num_query_groups = 1
        if self.softmax_type is None and mcore_015:
            self.softmax_type = 'vanilla'
        if self.norm_epsilon is None:
            self.norm_epsilon = 1e-5
        if self.rotary_base is None:
            self.rotary_base = 10000
        if self.rotary_interleaved is None:
            self.rotary_interleaved = False
        if self.attention_dropout is None:
            self.attention_dropout = 0.
        if self.untie_embeddings_and_output_weights is None:
            self.untie_embeddings_and_output_weights = True
        if self.swiglu is None:
            self.swiglu = True
        if self.quick_geglu is None:
            self.quick_geglu = False
        if self.glu_linear_offset is None and mcore_015:
            self.glu_linear_offset = 0.
        if self.add_qkv_bias is None:
            self.add_qkv_bias = True
        if self.disable_bias_linear is None:
            self.disable_bias_linear = True
        if self.qk_layernorm is None:
            self.qk_layernorm = False
        if self.multi_latent_attention is None:
            self.multi_latent_attention = False
        if self.kv_lora_rank is None:
            self.kv_lora_rank = 32
        if self.qk_head_dim is None:
            self.qk_head_dim = 128
        if self.qk_pos_emb_head_dim is None:
            self.qk_pos_emb_head_dim = 64
        if self.task_type is None:
            self.task_type = 'causal_lm'
        if self.calculate_per_token_loss is None:
            self.calculate_per_token_loss = self.task_type == 'causal_lm'
        if self.no_bias_dropout_fusion is None:
            self.no_bias_dropout_fusion = False
        # moe
        MegatronArguments._set_moe_default(self)
        # log
        if self.wandb_exp_name is None:
            self.wandb_exp_name = self.save

    @staticmethod
    def _set_moe_default(self):
        if self.use_shared_expert_gate is None:
            self.use_shared_expert_gate = False
        if self.moe_router_score_function is None:
            self.moe_router_score_function = 'softmax'
        if self.moe_router_topk is None:
            self.moe_router_topk = 2
        if self.moe_router_pre_softmax is None:
            self.moe_router_pre_softmax = False
        if self.moe_router_load_balancing_type is None:
            self.moe_router_load_balancing_type = 'aux_loss'
        if self.moe_router_enable_expert_bias is None:
            self.moe_router_enable_expert_bias = False
        if self.moe_layer_freq is None:
            self.moe_layer_freq = 1
        if self.mrope_interleaved is None:
            self.mrope_interleaved = False

    def _init_mixed_precision(self):
        from swift.llm.argument.base_args.model_args import ModelArguments
        ModelArguments._init_mixed_precision(self)
        if self.apply_query_key_layer_scaling is None:
            self.apply_query_key_layer_scaling = self.fp16
        if self.apply_query_key_layer_scaling:
            os.environ['NVTE_APPLY_QK_LAYER_SCALING'] = '1'

    def _init_moe(self):
        if self.moe_router_dtype.lower() == 'none':
            self.moe_router_dtype = None
        if self.moe_shared_expert_intermediate_size == 0:
            self.moe_shared_expert_intermediate_size = None
        if self.num_experts is not None:
            if self.moe_ffn_hidden_size is None:
                self.moe_ffn_hidden_size = self.ffn_hidden_size

    def __post_init__(self):
        require_version('numpy<2.0', 'Please install numpy<2.0 by running: `pip install "numpy<2.0"`.')
        if self.train_type == 'lora':
            if self.num_experts is not None:
                require_version('peft>=0.15')
            else:
                require_version('peft>=0.12')
        RLHFMegatronArgumentsMixin.__post_init__(self)
        MegatronTunerMixin.__post_init__(self)
        os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
        self._set_default()
        self.model_info, self.model_meta = get_model_info_meta(
            self.model, model_type=self.model_type, use_hf=self.use_hf, hub_token=self.hub_token)
        self.model_type = self.model_info.model_type
        if self.pipeline_model_parallel_size == 1 and (self.decoder_first_pipeline_num_layers is not None
                                                       or self.decoder_last_pipeline_num_layers is not None):
            raise ValueError('pipeline_model_parallel_size must be greater than 1 if you want to set '
                             'decoder_first_pipeline_num_layers or decoder_last_pipeline_num_layers.')
        if hasattr(self, 'ddp_timeout'):
            self.distributed_timeout_minutes = self.ddp_timeout // 60
        self.group_query_attention = self.num_query_groups > 1
        if self.rope_scaling is not None:
            self.rope_scaling = json_parse_to_dict(self.rope_scaling)
            if 'type' in self.rope_scaling and 'rope_type' not in self.rope_scaling:
                self.rope_scaling['rope_type'] = self.rope_scaling['type']
        if self.task_type != 'causal_lm':
            self.untie_embeddings_and_output_weights = True
        if self.gradient_checkpointing_kwargs is not None:
            self.gradient_checkpointing_kwargs = json_parse_to_dict(self.gradient_checkpointing_kwargs)
        if self.save_strategy == 'epoch':
            self.save_interval = 1
            self.eval_interval = 1
        if not self.no_gradient_accumulation_fusion:
            try:
                import apex
            except ImportError:
                logger.warning('apex is not installed, so gradient accumulation fusion is disabled.')
                self.no_gradient_accumulation_fusion = True
        if isinstance(self.ref_adapters, str):
            self.ref_adapters = [self.ref_adapters]
        if self.eval_interval is None:
            self.eval_interval = self.save_interval
        if self.seq_length is None:
            self.seq_length = self.max_position_embeddings
        if self.position_embedding_type is None:
            self.position_embedding_type = 'rope'
        if self.merge_lora is None:
            self.merge_lora = self.save_safetensors
        if self.adapters or self.adapter_load or self.ref_adapter_load:
            if self.train_type == 'full':
                self.train_type = 'lora'
                logger.info('Setting args.train_type: lora')
        if self.adapters:
            self._load_adapter_config()
        self._init_moe()
        self._init_mixed_precision()

        self.megatron_extra_kwargs = json_parse_to_dict(self.megatron_extra_kwargs)
        self._init_no_rope_fusion()

    def _init_no_rope_fusion(self):
        if self.no_rope_fusion is not None:
            return
        if self.multi_latent_attention or self.rotary_interleaved:
            # Upgrading transformer_engine requires checking here.
            self.no_rope_fusion = True
        else:
            self.no_rope_fusion = False
        logger.info(f'Setting args.no_rope_fusion: {self.no_rope_fusion}.')

    def _args_to_argv(self) -> Tuple[List[Any], Dict[str, Any]]:
        new_args = []
        args_dict = asdict(self)
        extra_args = {}
        extra_args['model_dir'] = self.model_info.model_dir
        extra_args['is_multimodal'] = self.model_meta.is_multimodal
        # model_type may be overridden by megatron
        extra_args['hf_model_type'] = self.model_type
        megatron_extra_kwargs = args_dict.pop('megatron_extra_kwargs')
        args_dict.update(megatron_extra_kwargs)
        for k, value in args_dict.items():
            if k not in MegatronArguments.__annotations__ and k not in megatron_extra_kwargs:
                extra_args[k] = value
                continue
            if value is None or value is False:
                continue
            new_args.append(f"--{k.replace('_', '-')}")
            if isinstance(value, list):
                new_args += [str(v) for v in value]
            elif value is not True:
                new_args.append(str(value))

        return new_args, extra_args

    def parse_to_megatron(self):
        new_args, extra_args = self._args_to_argv()
        sys._old_argv = sys.argv
        sys.argv = sys.argv[:1] + new_args
        # parameter conflict
        extra_args.pop('loss_scale', None)
        return extra_args

    def _load_adapter_config(self):
        assert len(self.adapters) == 1, 'Currently only support one adapter'
        adapter_path = self.adapters[0]
        adapter_config_path = os.path.join(adapter_path, 'adapter_config.json')
        adapter_config = {}
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
        mapping = {'r': 'lora_rank', 'bias': 'lora_bias'}
        for k in ['lora_alpha', 'lora_dropout', 'use_rslora']:
            mapping[k] = k
        for k, v in adapter_config.items():
            if k not in mapping:
                continue
            k = mapping[k]
            if v != getattr(self, k):
                setattr(self, k, v)
                logger.info(f'Setting {k}: {v}')
