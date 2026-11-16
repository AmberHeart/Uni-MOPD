# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, MOPD contributors.
"""Independent native Megatron teacher server; no student/optimizer allocation."""
from dataclasses import asdict, dataclass
from datetime import timedelta
import json
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.training import training

from ..argument import MegatronRLHFArguments
from ..trainers.remote_teacher_engine import NativeTeacherEngine
from ..trainers.remote_teacher_protocol import require
from ..trainers.remote_teacher_transport import TeacherRPCService
from .rlhf import MegatronRLHF


@dataclass
class TeacherServerArguments(MegatronRLHFArguments):
    rlhf_type: str = 'gkd'
    use_vllm: bool = False
    lmbda: float = 0.0
    teacher_body_reuse: bool = True
    offload_teacher_model: bool = True
    mopd_router_replay: str = 'disabled'
    mopd_response_only_logits: bool = True
    mopd_head_chunk_size: int = 0
    train_iters: int = 1  # parser/runtime initialization only; there is no train loop
    max_epochs: int = None
    check_model: bool = False
    teacher_bind_host: str = '0.0.0.0'
    teacher_port: int = 8013
    teacher_token_file: str = None
    teacher_rpc_timeout: float = 600.0
    teacher_max_response_bytes: int = 4 * 1024**3
    teacher_forward_tokens: int = 65536
    teacher_cpu_prefetch: bool = False

    def __post_init__(self):
        super().__post_init__()
        require(self.rlhf_type == 'gkd' and not self.use_vllm and self.lmbda == 0
                and self.mopd_router_replay == 'disabled', 'Teacher server must be forward-only with natural routing')
        require(self.pipeline_model_parallel_size == self.context_parallel_size == 1
                and self.micro_batch_size == 1 and self.train_type == 'full',
                'Native teacher server currently requires full-model PP1/CP1/MBS1')
        require(self.teacher_models and not self.load and not self.use_sft_loss and not self.use_opsd,
                'Teacher server needs an explicit teacher bank, not a student checkpoint or SFT/OPSD stream')
        require(self.teacher_token_file is not None, 'Teacher server needs a private token file')
        require(self.max_epochs is None and self.save_strategy == 'steps', 'Teacher server must not initialize epochs')
        require(type(self.teacher_forward_tokens) is int and self.teacher_forward_tokens > 0,
                'Teacher forward token budget must be positive')
        require(type(self.teacher_cpu_prefetch) is bool, 'Teacher CPU prefetch must be an explicit boolean')


class MegatronTeacherServer(MegatronRLHF):
    args_class = TeacherServerArguments

    def run(self):
        args = self.args
        # Reuse the pinned runtime's validation/tokenizer/MPU initialization.
        # Do NOT call trainer.train or setup_model_and_optimizer: those allocate
        # the student and Adam, neither belongs on this independent teacher node.
        with self.trainer._get_iters(None, None):
            training.initialize_megatron(
                extra_args_provider=args.megatron_model_meta.extra_args_provider,
                args_defaults=args.extra_args)
        # Idle waits are kept alive by owner-thread heartbeats; the margin
        # allows a bounded RPC failure to reach peers before Gloo times out.
        control = dist.new_group(backend='gloo', timeout=timedelta(seconds=args.teacher_rpc_timeout + 60))
        engine = NativeTeacherEngine(self.trainer, control)
        identities = engine.load_bank()
        token_file = Path(args.teacher_token_file)
        require(token_file.is_file() and token_file.stat().st_mode & 0o077 == 0,
                'Teacher token file must be private')
        if dist.get_rank() == 0:
            service = TeacherRPCService(
                engine.score, identities, auth_token=token_file.read_text().strip(),
                max_response_bytes=args.teacher_max_response_bytes, timeout_seconds=args.teacher_rpc_timeout,
                on_shutdown=engine.stop, on_idle=engine.heartbeat)
            port = service.start_http(args.teacher_bind_host, args.teacher_port)
            ready = Path(args.save) / 'teacher-ready.json'
            require(not ready.exists(), 'Do not reuse a teacher server output directory')
            temporary = ready.with_suffix('.tmp')
            with temporary.open('x') as stream:
                json.dump({
                    'port': port, 'session_id': service.session_id, 'world_size': dist.get_world_size(),
                    'teachers': {key: asdict(value) for key, value in identities.items()},
                    'student_optimizer_allocated': False,
                    'teacher_cpu_prefetch': args.teacher_cpu_prefetch,
                }, stream, indent=2)
            temporary.rename(ready)
            service.serve()
        else:
            engine.follow()
        self.trainer._offload_teacher_models()
        torch.cuda.synchronize()
        dist.destroy_process_group()


def megatron_teacher_server_main(args=None):
    return MegatronTeacherServer(args).main()
