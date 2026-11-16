# Copyright (c) Alibaba, Inc. and its affiliates.
import math
from dataclasses import dataclass
from typing import Literal, Optional, Union

from .train_args import MegatronTrainArguments


@dataclass
class MegatronRLHFArguments(MegatronTrainArguments):
    rlhf_type: Literal['dpo', 'kto', 'grpo', 'gkd', 'rm'] = 'dpo'
    loss_scale: str = 'last_round'
    truncation_strategy: Optional[Literal['delete', 'left', 'right', 'split', None]] = None

    calculate_per_token_loss: bool = False
    # GKD coordinates recovered rows/rollouts and has its own route-local retry
    # policy; preserve the existing RLHF opt-in strict-mode contract.
    strict: bool = False
    early_stopping_patience: int = 0
    # Independent from rollout sampling; non-unit values change the KD objective.
    gkd_loss_temperature: float = 1.0
    # "jsd" preserves beta=0 FKL / beta=1 RKL / intermediate generalized JSD.
    # This is independent of the DPO-only f_divergence_type.
    gkd_loss_type: Literal['jsd', 'todi'] = 'jsd'
    # ToDi sigmoid sharpness, NOT the legacy GKD beta interpolation coefficient.
    gkd_todi_beta: float = 1.0
    # Restrict the distillation loss to response tokens; the released recipe enables it.
    mopd_response_only_logits: bool = False
    # 0 keeps the existing path; positive values checkpoint teacher-head+KL in
    # token chunks while retaining the full vocabulary.
    mopd_head_chunk_size: int = 0
    mopd_domain_generation_config: Optional[Union[dict, str]] = None
    # Actual P+R bound, independent of the separate prompt/response envelopes.
    mopd_max_sequence_length: Optional[int] = None
    # Independent CUDA send streams, with per-bucket completion before reuse.
    mopd_parallel_weight_sync: bool = False
    # Independent Megatron teacher node; local teacher remains the default.
    mopd_teacher_endpoint: Optional[str] = None
    mopd_teacher_token_file: Optional[str] = None
    mopd_teacher_chunk_size: int = 64
    mopd_teacher_max_inflight_bytes: int = 8 * 1024**3
    mopd_teacher_rpc_timeout: float = 600.0
    # Keep frozen teacher weights in CPU memory between microbatches; the released
    # recipe enables it alongside offload_teacher_model.
    mopd_teacher_cpu_cache: bool = False
    # Zero preserves batch-at-once rollout. Positive values are whole-response
    # consumption chunks inside the same native optimizer window.
    mopd_rollout_completion_chunk_size: int = 0
    mopd_rollout_queue_max_bytes: int = 8 * 1024**3
    # Number of rollout records written per step for inspection; 0 (the default)
    # disables the writer entirely. Records contain prompts, responses and image
    # references, so enable it only when you intend to retain that content.
    # This does not change the number of training/rollout occurrences.
    mopd_rollout_samples_per_step: int = 0
    mopd_vllm_session_id: Optional[str] = None
    mopd_vllm_runtime_identity: Optional[str] = None

    def __post_init__(self):
        if type(self.mopd_rollout_samples_per_step) is not int or self.mopd_rollout_samples_per_step < 0:
            raise ValueError('mopd_rollout_samples_per_step must be a nonnegative integer.')
        # MegatronTrainArguments.__post_init__ initializes the process group.
        # Reject objective mistakes before entering that distributed setup.
        self._validate_gkd_objective()
        self._validate_gkd_optimizations()
        if self.rlhf_type == 'rm':
            self.task_type = 'seq_cls'
            self.num_labels = 1
        self._init_truncation_strategy()
        super().__post_init__()
        self._validate_domain_generation()
        self._validate_parallel_weight_sync()
        self._validate_remote_teacher()
        self._validate_completion_stream()
        self._validate_teacher_cpu_cache()
        if self.mopd_vllm_session_id or self.mopd_vllm_runtime_identity:
            if (not self.mopd_vllm_session_id or not self.mopd_vllm_runtime_identity
                    or self.rlhf_type != 'gkd' or self.train_type != 'full'
                    or not self.use_vllm or self.vllm_mode != 'server'
                    or self.lmbda != 1 or self.seq_kd or self.overlap_rollout_training):
                raise ValueError('Persistent rollout sessions require synchronous full-model on-policy GKD')

    def _validate_gkd_objective(self):
        if self.gkd_loss_type not in ('jsd', 'todi'):
            raise ValueError(f'Unsupported gkd_loss_type={self.gkd_loss_type!r}; choose jsd or todi.')
        if (isinstance(self.gkd_todi_beta, bool)
                or not isinstance(self.gkd_todi_beta, (int, float))
                or not math.isfinite(self.gkd_todi_beta) or self.gkd_todi_beta <= 0):
            raise ValueError('gkd_todi_beta must be finite and strictly positive.')
        if self.rlhf_type != 'gkd':
            if self.gkd_loss_type != 'jsd' or self.gkd_todi_beta != 1.0:
                raise ValueError('gkd_loss_type and gkd_todi_beta require rlhf_type=gkd.')
            return
        # The inherited mixin resolves None to 0.5 later; keep that default.
        if self.beta is not None and (
                isinstance(self.beta, bool) or not isinstance(self.beta, (int, float))
                or not math.isfinite(self.beta) or not 0 <= self.beta <= 1):
            raise ValueError('GKD beta must be finite and in [0, 1] (0=FKL, 1=RKL).')

    def _validate_teacher_cpu_cache(self):
        if not self.mopd_teacher_cpu_cache:
            return
        if (self.rlhf_type != 'gkd' or not self.teacher_models or not self.offload_teacher_model
                or not self.teacher_body_reuse or not self.lazy_teacher_logits
                or self.mopd_teacher_endpoint is not None):
            raise ValueError(
                'Frozen teacher CPU caching requires local routed GKD with lazy logits, '
                'teacher offload and teacher body reuse; it is not a student/remote cache.')

    def _validate_completion_stream(self):
        size = self.mopd_rollout_completion_chunk_size
        if type(size) is not int or size < 0:
            raise ValueError('Completion chunk size must be a nonnegative integer.')
        if not size:
            return
        # The local iterator feeds ONE native schedule, so token sums/counts
        # can accumulate until its final DP reduction. Remote-teacher recipes
        # keep their separate sequence-only admission.
        valid_normalization = (
            (self.mopd_loss_normalization == 'sequence' and not self.calculate_per_token_loss)
            or (self.mopd_teacher_endpoint is None
                and self.mopd_loss_normalization == 'global_token' and self.calculate_per_token_loss))
        if (self.rlhf_type != 'gkd' or self.train_type != 'full' or not self.use_vllm
                or self.vllm_mode != 'server' or self.lmbda != 1 or self.seq_kd
                or self.pipeline_model_parallel_size != 1 or self.context_parallel_size != 1
                or self.micro_batch_size != 1 or self.use_sft_loss or self.sft_alpha != 0
                or self.use_opsd or self.teacher_model_2 is not None
                or self.overlap_rollout_training or not self.lazy_teacher_logits
                or not self.teacher_models
                or not self.mopd_response_only_logits
                # Local completion streaming only supplies inputs. Its native
                # forward supports dense heads; remote hidden targets still
                # require the streamed-head consumer.
                or (self.mopd_teacher_endpoint is not None and self.mopd_head_chunk_size <= 0)
                or self.mopd_router_replay != 'r3'
                or self.offload_model or self.offload_optimizer
                or not valid_normalization or not self.strict
                or size > self.global_batch_size or self.global_batch_size % size
                or (self.mopd_teacher_endpoint is not None and size % self.mopd_teacher_chunk_size)):
            raise ValueError(
                'Complete-response streaming requires strict on-policy R3 MOPD, PP1/CP1/MBS1, '
                'sequence normalization (or local global_token with calculate_per_token_loss=True), '
                'response-only heads (streamed heads for remote teachers) and a synchronous vLLM server; '
                'no student offload, SFT/OPSD/ensemble/overlap, and chunks must divide GBS.')
        if type(self.mopd_rollout_queue_max_bytes) is not int or self.mopd_rollout_queue_max_bytes <= 0:
            raise ValueError('Complete-response streaming requires a positive CPU queue byte limit.')
        if not math.isfinite(self.mopd_teacher_rpc_timeout) or self.mopd_teacher_rpc_timeout <= 0:
            raise ValueError('Complete-response streaming requires a positive transport timeout.')
        if self.mopd_teacher_endpoint is None:
            # Like remote completion windows, local delivery is single-pass.
            # Keep R3 and fail-fast numeric checks; disable only native retries.
            from swift.utils import json_parse_to_dict
            extra = dict(json_parse_to_dict(getattr(self, 'megatron_extra_kwargs', None)))
            if extra.get('rerun_mode', 'disabled') != 'disabled':
                raise ValueError('Local completion streaming requires megatron_extra_kwargs.rerun_mode=disabled.')
            extra['rerun_mode'] = 'disabled'
            self.megatron_extra_kwargs = extra

    def _validate_remote_teacher(self):
        if self.mopd_teacher_endpoint is None:
            return
        if (self.rlhf_type != 'gkd' or self.train_type != 'full' or not self.use_vllm
                or self.vllm_mode != 'server' or self.lmbda != 1 or self.seq_kd
                or self.pipeline_model_parallel_size != 1 or self.context_parallel_size != 1
                or self.micro_batch_size != 1 or self.use_sft_loss or self.sft_alpha != 0
                or self.use_opsd or self.teacher_model_2 is not None
                or self.overlap_rollout_training or not self.lazy_teacher_logits
                or not self.teacher_models
                or self.mopd_loss_normalization != 'sequence' or self.calculate_per_token_loss
                or not self.mopd_response_only_logits or self.mopd_head_chunk_size <= 0):
            raise ValueError(
                'Remote teacher currently requires on-policy routed full-vocab GKD, PP1/CP1/MBS1, '
                'sequence normalization, response-only/streamed heads, a vLLM server and no '
                'SFT/OPSD/ensemble/overlap pipeline.')
        if (self.mopd_teacher_token_file is None or type(self.mopd_teacher_chunk_size) is not int
                or self.mopd_teacher_chunk_size <= 0
                or type(self.mopd_teacher_max_inflight_bytes) is not int
                or self.mopd_teacher_max_inflight_bytes <= 0
                or not math.isfinite(self.mopd_teacher_rpc_timeout) or self.mopd_teacher_rpc_timeout <= 0):
            raise ValueError('Remote teacher needs a private token file and positive chunk, memory and timeout limits.')
        # A remote teacher window is consumed once and cannot be rewound by
        # Megatron's numerical-error attribution reruns. This does not disable
        # R3 or validate_result(): disabled mode still raises on rejected values.
        # init_model_args normalizes this field later; at this point an actual
        # CLI invocation still supplies the JSON string (or a JSON file path).
        from swift.utils import json_parse_to_dict
        extra = dict(json_parse_to_dict(getattr(self, 'megatron_extra_kwargs', None)))
        if extra.get('rerun_mode', 'disabled') != 'disabled':
            raise ValueError('Remote teacher streaming requires megatron_extra_kwargs.rerun_mode=disabled.')
        extra['rerun_mode'] = 'disabled'
        self.megatron_extra_kwargs = extra

    def _validate_parallel_weight_sync(self):
        if not self.mopd_parallel_weight_sync:
            return
        if (self.rlhf_type != 'gkd' or self.train_type != 'full' or not self.use_vllm
                or self.vllm_mode != 'server' or self.offload_bridge or self.overlap_rollout_training
                or self.lmbda != 1 or self.seq_kd
                or self.pipeline_model_parallel_size != 1 or self.context_parallel_size != 1):
            raise ValueError(
                'mopd_parallel_weight_sync requires synchronous full-training on-policy GKD, '
                'lambda=1, PP1/CP1, a vLLM server, and GPU bridge export (offload_bridge=False).')

    def _validate_domain_generation(self):
        total = getattr(self, 'mopd_max_sequence_length', None)
        if total is not None:
            if (type(total) is not int or not self.max_length < total
                    <= self.max_length + self.max_completion_length
                    or self.rlhf_type != 'gkd' or not self.use_vllm
                    or self.vllm_mode != 'server' or self.lmbda != 1
                    or self.pipeline_model_parallel_size != 1 or not self.strict
                    or self.truncation_strategy != 'delete'):
                raise ValueError(
                    'mopd_max_sequence_length requires strict PP1 on-policy GKD and '
                    'prompt envelope < actual sequence bound <= prompt+response envelopes.')
        if self.mopd_domain_generation_config is None:
            return
        if (self.rlhf_type != 'gkd' or not self.use_vllm or self.vllm_mode != 'server'
                or self.lmbda != 1 or self.seq_kd or self.micro_batch_size != 1
                or self.overlap_rollout_training):
            raise ValueError(
                'mopd_domain_generation_config requires synchronous on-policy GKD, '
                'lambda=1, micro_batch_size=1 and a student vLLM server.')
        from swift.llm.infer.mopd_generation import parse_domain_generation_config
        from swift.utils import json_parse_to_dict
        mapping = json_parse_to_dict(self.domain_teacher_mapping) if self.domain_teacher_mapping else None
        self.mopd_domain_generation_config = parse_domain_generation_config(
            self.mopd_domain_generation_config, max_tokens=self.max_completion_length,
            temperature=self.temperature, domains=None if mapping is None else set(mapping),
            max_prompt_tokens=self.max_length)
        if any('max_prompt_tokens' in value for value in self.mopd_domain_generation_config.values()):
            if not self.strict or self.truncation_strategy != 'delete':
                raise ValueError(
                    'Domain prompt admission requires strict mode and truncation_strategy=delete; '
                    'budget violations must raise, not truncate or resample.')

    def _validate_gkd_optimizations(self):
        head_chunk_size = getattr(self, 'mopd_head_chunk_size', 0)
        if type(head_chunk_size) is not int or head_chunk_size < 0:
            raise ValueError('mopd_head_chunk_size must be a nonnegative integer.')
        if head_chunk_size and (not self.mopd_response_only_logits or self.train_type != 'full'):
            raise ValueError('mopd_head_chunk_size requires response-only logits and full native training.')
        if self.rlhf_type != 'gkd':
            if self.mopd_response_only_logits or head_chunk_size:
                raise ValueError('mopd_response_only_logits requires rlhf_type=gkd.')
            return
        if not math.isfinite(self.gkd_loss_temperature) or self.gkd_loss_temperature <= 0:
            raise ValueError('gkd_loss_temperature must be finite and strictly positive.')
        if not self.mopd_response_only_logits:
            return
        if (self.pipeline_model_parallel_size != 1 or self.context_parallel_size != 1
                or self.micro_batch_size != 1 or self.sft_alpha != 0 or self.use_sft_loss
                or self.use_opsd or self.teacher_model_2 is not None
                or self.overlap_rollout_training or not self.lazy_teacher_logits):
            raise ValueError(
                'mopd_response_only_logits currently supports synchronous GKD/domain-routed MOPD '
                'with PP1/CP1, micro_batch_size=1, lazy_teacher_logits=True, and no '
                'SFT auxiliary loss, OPSD or teacher ensemble.')

    def _init_truncation_strategy(self):
        if self.truncation_strategy is not None:
            return
        if self.rlhf_type == 'grpo':
            self.truncation_strategy = 'left'
        else:
            self.truncation_strategy = 'delete'
