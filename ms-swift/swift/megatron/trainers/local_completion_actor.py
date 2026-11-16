# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, MOPD contributors.
"""Complete-response input chunks for the existing singleton local-teacher path.

This adapter owns no model, optimizer, hidden cache or teacher RPC. Each yielded
real microbatch still runs the ordinary teacher forward, student forward and
backward. The native full-window schedule owns all gradient/update boundaries.
"""
from contextlib import contextmanager
from datetime import timedelta
import uuid

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.rerun_state_machine import RerunMode, get_rerun_state_machine
from megatron.training import get_args

from .remote_teacher_protocol import require


class LocalCompletionActor:
    completion_metric_prefix = 'local_completion/'

    def __init__(self, trainer):
        require(get_rerun_state_machine().get_mode() == RerunMode.DISABLED,
                'Local completion streaming requires native rerun_mode=disabled before the first rollout')
        self.trainer = trainer
        self.rank = dist.get_rank()
        self.dp_size = mpu.get_data_parallel_world_size()
        self.completion_multiple = self.dp_size
        self.completion_teacher_keys = [str(index) for index in range(len(trainer.teacher_model_paths))]
        self.control = dist.new_group(
            backend='gloo', timeout=timedelta(seconds=trainer.args.mopd_teacher_rpc_timeout + 60))
        info = [None] * dist.get_world_size()
        local = (mpu.get_data_parallel_rank(), mpu.get_tensor_model_parallel_rank())
        dist.all_gather_object(info, local, group=self.control)
        tp_size = mpu.get_tensor_model_parallel_world_size()
        require(sorted(info) == [(dp, tp) for dp in range(self.dp_size) for tp in range(tp_size)],
                'Local completion streaming needs a complete PP1/CP1 student DP/TP mesh')
        self.rank_to_dp = {rank: value[0] for rank, value in enumerate(info)}
        require(trainer.args.mopd_rollout_completion_chunk_size % self.dp_size == 0
                and get_args().global_batch_size % self.dp_size == 0,
                'Local completion chunks and GBS must divide the student DP mesh')
        run = [uuid.uuid4().hex if self.rank == 0 else None]
        dist.broadcast_object_list(run, src=0, group=self.control)
        self.run_id = run[0]
        self._completion_window = None
        self.active = False
        self.batches, self.bindings = [], []
        self.cursor = 0

    def _agree(self, error):
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error, group=self.control)
        require(not any(errors), 'Local completion preparation failed: ' + '; '.join(
            f'rank{rank}: {value}' for rank, value in enumerate(errors) if value))

    @contextmanager
    def encoding_guard(self):
        error = None
        try:
            yield
        except Exception as failure:
            error = f'{type(failure).__name__}: {failure}'
        self._agree(error)

    def prepare_completion_window(self, raw_batch, data_source):
        from .completion_window import IncrementalRolloutWindow
        require(self._completion_window is None and not self.active,
                'Previous local completion window is still active')
        self._completion_window = IncrementalRolloutWindow(self.trainer, self, data_source)
        return self._completion_window.begin(raw_batch)

    def prepare_window(self, raw_groups, batches, policy_step, *, window_samples):
        """Keep real inputs only; the existing forward coordinates teacher-only rounds."""
        error = None
        bindings = []
        try:
            require(not self.active and 0 < window_samples <= get_args().global_batch_size
                    and len(raw_groups) == len(batches)
                    and len(batches) * self.dp_size == window_samples,
                    'Local completion chunk does not cover its real student samples')
            for (raw, teacher), batch in zip(raw_groups, batches):
                require(all(not isinstance(value, torch.Tensor) or value.device.type == 'cpu'
                            for value in batch.values()),
                        'Queued local-completion inputs must stay on CPU until their MBS1 forward')
                require(len(raw) == 1 and batch['teacher_idx'] == teacher
                        and batch['_rollout_weight_step'] == policy_step,
                        'Local completion sample lost its singleton teacher/policy binding')
                require(not any(key in batch for key in (
                    '_mopd_teacher_scheduled', '_mopd_coordination_padding', '_mopd_coordination_scale',
                    '_mopd_remote_teacher_target')),
                    'Local streaming must not insert student dummy slots, chunk divisors or cached teacher targets')
                bindings.append(raw[0]['_mopd_remote_occurrence'])
            require(len(set(bindings)) == len(bindings), 'Duplicate local completion occurrence')
        except Exception as failure:
            error = f'{type(failure).__name__}: {failure}'
        self._agree(error)
        self.batches, self.bindings = batches, bindings
        self.cursor, self.active = 0, True

    def __next__(self):
        require(self.active and self.cursor < len(self.batches), 'Local completion cursor is outside its chunk')
        batch = self.batches[self.cursor]
        self.batches[self.cursor] = None
        self.cursor += 1
        return batch

    def finish_completion_chunk(self):
        # Called only after the last real microbatch's backward. No student
        # offload/zero/reduction occurs here, so earlier chunk gradients remain.
        self._agree(None if self.active and self.cursor == len(self.batches)
                    else 'Native schedule did not consume the complete local input chunk')
        self.active = False
        return {'chunks': 1, 'samples': len(self.batches) * self.dp_size}

    def finish_before_update(self):
        require(self._completion_window is not None, 'No local completion window to finish')
        result = self._completion_window.finish()
        self._completion_window = None
        return result

    def abort(self):
        try:
            if self._completion_window is not None:
                self._completion_window.abort()
        finally:
            self._completion_window = None
            self.batches, self.bindings = [], []
            self.active = False
