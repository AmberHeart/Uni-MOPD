# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List, Optional, Union

from swift.llm.train.kto import prepare_kto_dataset
from swift.trainers.rlhf_trainer.utils import identity_data_collator
from swift.utils import get_current_device, get_logger, is_last_rank
from ..argument import MegatronRLHFArguments
from ..trainers import (MegatronDPOTrainer, MegatronGKDTrainer, MegatronGRPOTrainer, MegatronKTOTrainer,
                        MegatronRewardTrainer)
from .sft import MegatronSft

logger = get_logger()


class MegatronRLHF(MegatronSft):
    args_class = MegatronRLHFArguments
    args: args_class

    def run(self):
        result = super().run()
        # Normal completion must observe release failures before exit status
        # is decided. atexit remains an idempotent fallback, not the sole path.
        if self.args.mopd_vllm_session_id and self.trainer.vllm_client is not None:
            self.trainer.vllm_client.close_communicator()
        return result

    def prepare_trainer(self):
        args = self.args
        trainer_mapping = {
            'dpo': MegatronDPOTrainer,
            'gkd': MegatronGKDTrainer,
            'grpo': MegatronGRPOTrainer,
            'kto': MegatronKTOTrainer,
            'rm': MegatronRewardTrainer
        }
        trainer_cls = trainer_mapping.get(args.rlhf_type)
        if trainer_cls is None:
            raise ValueError(f'The current Megatron-SWIFT does not support rlhf_type: {args.rlhf_type}.')
        kwargs = {}
        if args.rlhf_type in ('grpo', 'gkd'):
            kwargs['vllm_client'] = self._prepare_vllm_client()
        return trainer_cls(args, self.template, **kwargs)

    def _prepare_template(self) -> None:
        super()._prepare_template()
        model_mapping = {'grpo': 'train', 'gkd': 'train', 'kto': 'kto'}
        self.template.set_mode(model_mapping.get(self.args.rlhf_type, 'rlhf'))

    def _get_data_collator(self):
        if self.args.rlhf_type in ('grpo', 'gkd'):
            return identity_data_collator
        return super()._get_data_collator()

    def _get_dataset(self):
        args = self.args
        train_dataset, val_dataset = super()._get_dataset()
        if args.rlhf_type == 'kto':
            train_dataset, val_dataset = prepare_kto_dataset(args, train_dataset, val_dataset)
        return train_dataset, val_dataset

    def _prepare_vllm_client(self):
        # Only prepare vLLM client for server mode
        if self.args.rlhf_type not in ('grpo', 'gkd') or self.args.vllm_mode != 'server':
            return None
        # GKD may not use vLLM (off-policy mode)
        if not getattr(self.args, 'use_vllm', False):
            return None

        from swift.trainers.rlhf_trainer.vllm_client import VLLMClient
        import torch.distributed as dist
        vllm_client = None
        # Use rank == (NPROC_PER_NODE - 1) to always pick the last GPU of node 0,
        # regardless of how many training nodes are used. is_last_rank() returns
        # rank == world_size-1 which shifts to node 1 in multi-node setups, causing
        # NCCL communicator init to fail when node 1 cannot reach the vLLM server.
        nproc_per_node = int(os.environ.get('NPROC_PER_NODE', dist.get_world_size()))
        is_node0_last_rank = (dist.get_rank() == nproc_per_node - 1)
        if is_node0_last_rank:
            logger.info('Start connecting to vLLM server')
            vllm_client = VLLMClient(
                base_urls=self.args.vllm_server_base_url,
                hosts=self.args.vllm_server_host,
                server_ports=self.args.vllm_server_port,
                group_ports=self.args.vllm_server_group_port,
                connection_timeout=self.args.vllm_server_timeout,
                parallel_weight_sync=self.args.mopd_parallel_weight_sync,
                persistent_session_id=self.args.mopd_vllm_session_id,
                runtime_identity=self.args.mopd_vllm_runtime_identity)
            if not self.args.mopd_vllm_session_id:
                vllm_client.close_communicator()
            vllm_client.init_communicator(device=get_current_device())
            logger.info('Connected to vLLM server')
        return vllm_client


def megatron_rlhf_main(args: Optional[Union[List[str], MegatronRLHFArguments]] = None):
    return MegatronRLHF(args).main()
