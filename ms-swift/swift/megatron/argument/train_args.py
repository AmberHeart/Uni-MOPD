# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass
from typing import Literal, Optional

import json

from swift.llm.argument.base_args import to_abspath
from swift.utils import add_version_to_work_dir, get_logger, init_process_group, is_last_rank
from .megatron_base_args import MegatronBaseArguments

logger = get_logger()


@dataclass
class MegatronTrainArguments(MegatronBaseArguments):
    add_version: bool = True
    load_args: bool = False
    # TP/PP/CP replicas must not independently substitute different SFT rows.
    # Lazy encoding retries transient I/O in place, then stops on bad input.
    strict: bool = True
    # 'r2' also applies to SFT; 'r3' requires synchronous on-policy GKD rollouts.
    mopd_router_replay: Literal['disabled', 'r2', 'r3'] = 'disabled'
    # Auto preserves the existing TE selector. Version 2 is opt-in for the
    # validated BF16 SFT/GKD kernel recipe; serving remains unchanged.
    mopd_train_flash_attention_version: Literal['auto', '2'] = 'auto'
    # Model-only rank-local conversion cache. Never a resume/optimizer cache.
    mopd_model_cache_dir: Optional[str] = None
    mopd_model_cache_reserve_gb: int = 128

    def _validate_train_flash_attention(self):
        choice = self.mopd_train_flash_attention_version
        if choice == 'auto':
            return
        if (choice != '2' or self.rlhf_type not in (None, 'gkd') or self.attention_backend != 'flash'
                or self.pipeline_model_parallel_size != 1 or self.context_parallel_size != 1):
            raise ValueError(
                'mopd_train_flash_attention_version=2 currently requires SFT/GKD, '
                'attention_backend=flash and PP1/CP1.')

    def _validate_router_replay(self):
        mode = self.mopd_router_replay
        if mode == 'disabled':
            return
        if mode not in ('r2', 'r3'):
            raise ValueError(f'Unsupported mopd_router_replay={mode!r}.')
        if self.rlhf_type not in (None, 'gkd'):
            raise ValueError('mopd_router_replay currently supports SFT and GKD only.')
        if (self.pipeline_model_parallel_size != 1 or self.context_parallel_size != 1
                or self.micro_batch_size != 1 or self.overlap_rollout_training
                or self.use_sft_loss or self.mtp_num_layers):
            raise ValueError(
                'mopd_router_replay requires synchronous PP1/CP1, micro_batch_size=1, '
                'no independent SFT stream and no MTP.')
        if mode == 'r3' and (self.rlhf_type != 'gkd' or not self.use_vllm
                             or self.vllm_mode != 'server' or self.lmbda != 1 or self.seq_kd):
            raise ValueError('R3 requires on-policy GKD with lambda=1 and a student vLLM server.')
    def _init_save(self):
        init_process_group(backend=self.ddp_backend, timeout=self.ddp_timeout)
        if self.save is None:
            self.save = f'megatron_output/{self.model_suffix}'
        self.save = to_abspath(self.save)
        if self.add_version:
            self.save = add_version_to_work_dir(self.save)
            logger.info(f'args.save: {self.save}')
        if is_last_rank():
            os.makedirs(self.save, exist_ok=True)

    def _init_ckpt_dir(self, adapters=None):
        super()._init_ckpt_dir(adapters)
        if self.ckpt_dir and self.model is None:
            args_path = os.path.join(self.ckpt_dir, 'args.json')
            if not os.path.exists(args_path):
                return
            with open(args_path, 'r', encoding='utf-8') as f:
                old_args = json.load(f)
            self.model = old_args.get('model')

    def __post_init__(self):
        if self.mopd_model_cache_dir is not None:
            if (not os.path.isabs(self.mopd_model_cache_dir)
                    or self.train_type != 'full' or self.rlhf_type not in (None, 'gkd')
                    or type(self.mopd_model_cache_reserve_gb) is not int
                    or self.mopd_model_cache_reserve_gb < 0):
                raise ValueError('Converted weights require an absolute cache directory, full SFT/GKD and nonnegative headroom')
        self.load = to_abspath(self.load, check_path_exist=True)
        super().__post_init__()
        self._validate_router_replay()
        self._validate_train_flash_attention()
        if len(self.dataset) == 0 and len(self.cached_dataset) == 0:
            raise ValueError(f'self.dataset: {self.dataset}, self.cached_dataset: {self.cached_dataset}. '
                             'Please input the training dataset.')
        self._init_save()
        if self.tensorboard_dir is None and self.save is not None:
            self.tensorboard_dir = f'{self.save}/runs'
        self.tensorboard_dir = to_abspath(self.tensorboard_dir)
        if self.load is None and self.model is None and self.no_initialization:
            raise ValueError('You did not pass `--load/--model` to read weights, so you need to set '
                             '`--no_initialization false` to allow the model to initialize weights properly.')
