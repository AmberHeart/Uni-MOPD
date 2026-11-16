# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, MOPD contributors.
"""Distributed student-side remote-teacher bridge and streaming input iterator."""
import copy
from contextlib import contextmanager
from dataclasses import asdict
from datetime import timedelta
import json
from pathlib import Path
import time
import uuid

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.rerun_state_machine import RerunMode, get_rerun_state_machine
from megatron.training import get_args
from megatron.training.utils import unwrap_model

from .remote_teacher_heads import build_native_teacher_head, load_teacher_head_shard, tokenizer_hash
from .remote_teacher_collectives import scatter_hidden_buffers
from .remote_teacher_protocol import (
    MAX_METADATA_BYTES, HiddenResult, ScoreRequest, TeacherIdentity, WindowIdentity, bind_encoded_sample, require,
)
from .remote_teacher_stream import PlannedSample, StreamingTeacherWindow, TeacherChunk, balance_teacher_samples
from .remote_teacher_transport import TeacherHTTPClient
from .streamed_distillation import TeacherHeadTarget


class RemoteTeacherActor:
    """Main-thread Gloo dispatch plus one CPU-only HTTP prefetch worker.

    All ranks keep one native Megatron forward/backward invocation per update.
    No chunk-local optimizer step, loss divisor, or gradient finalization hook
    is introduced. Unlike cross-invocation accumulation, the native full-window
    schedule already supplies the correct final reduction and denominator.
    """

    completion_metric_prefix = 'remote_teacher/'

    def __init__(self, trainer):
        require(get_rerun_state_machine().get_mode() == RerunMode.DISABLED,
                'Remote streaming requires native rerun_mode=disabled before the first rollout')
        self.trainer = trainer
        args = trainer.args
        self.rank = dist.get_rank()
        self.tp_rank = mpu.get_tensor_model_parallel_rank()
        self.dp_rank = mpu.get_data_parallel_rank()
        self.dp_size = mpu.get_data_parallel_world_size()
        self.root = self.rank == 0
        control_timeout = timedelta(seconds=args.mopd_teacher_rpc_timeout + 60)
        self.control = dist.new_group(backend='gloo', timeout=control_timeout)
        local = {'rank': self.rank, 'tp': self.tp_rank, 'dp': self.dp_rank,
                 'tp_ranks': dist.get_process_group_ranks(mpu.get_tensor_model_parallel_group())}
        info = [None] * dist.get_world_size()
        dist.all_gather_object(info, local, group=self.control)
        leaders = [item['rank'] for item in sorted(info, key=lambda item: item['dp']) if item['tp'] == 0]
        require(leaders[0] == 0 and len(leaders) == self.dp_size, 'Unsupported student DP coordinator layout')
        self.leader_ranks = leaders
        self.rank_to_dp = {item['rank']: item['dp'] for item in info}
        self.leaders = dist.new_group(ranks=leaders, backend='gloo', timeout=control_timeout)
        for ranks in sorted({tuple(item['tp_ranks']) for item in info}):
            group = dist.new_group(ranks=list(ranks), backend='gloo', timeout=control_timeout)
            if self.rank in ranks:
                self.tp_group = group
                self.tp_root = ranks[0]
        self.client = None
        packet = None
        if self.root:
            try:
                token = Path(args.mopd_teacher_token_file)
                require(token.is_file() and token.stat().st_mode & 0o077 == 0, 'Teacher token file must be private')
                self.client = TeacherHTTPClient(
                    args.mopd_teacher_endpoint, None, auth_token=token.read_text().strip(),
                    timeout_seconds=args.mopd_teacher_rpc_timeout)
                packet = {'teachers': {key: asdict(value) for key, value in self.client.teachers.items()},
                          'run_id': uuid.uuid4().hex, 'error': None,
                          'clock_offset_bounds': self.client.clock_offset_bounds}
            except Exception as error:
                packet = {'error': f'{type(error).__name__}: {error}'}
        packet = self._broadcast(packet)
        require(packet['error'] is None, f'Teacher handshake failed: {packet["error"]}')
        self.run_id = packet['run_id']
        self.teacher_clock_offset_bounds = packet['clock_offset_bounds']
        self.identities = {key: TeacherIdentity.from_dict(value) for key, value in packet['teachers'].items()}
        self.completion_teacher_keys = sorted(self.identities, key=int)
        self.completion_multiple = args.mopd_teacher_chunk_size
        self.heads = {}
        error = None
        try:
            require(args.mopd_teacher_chunk_size > 0 and args.mopd_teacher_chunk_size % self.dp_size == 0
                    and get_args().global_batch_size % args.mopd_teacher_chunk_size == 0,
                    'Remote teacher chunks must divide GBS and the student DP mesh')
            require(set(self.identities) == {str(i) for i in range(len(trainer.teacher_model_paths))},
                    'Teacher server roster differs from the student routes')
            native = unwrap_model(trainer.unwrapped_models[0])
            language = getattr(native, 'language_model', native)
            token_grid = tokenizer_hash(trainer.template.tokenizer)
            for index, path in enumerate(trainer.teacher_model_paths):
                identity = self.identities[str(index)]
                require(identity.tokenizer_hash == token_grid
                        and identity.padded_vocab_size == get_args().padded_vocab_size
                        and len(identity.head_shard_hashes) == mpu.get_tensor_model_parallel_world_size(),
                        'Teacher tokenizer/vocabulary/TP partition differs')
                weight = load_teacher_head_shard(path, identity, self.tp_rank).to(torch.cuda.current_device())
                self.heads[str(index)] = build_native_teacher_head(
                    identity, weight, language.config, mpu.get_tensor_model_parallel_group())
        except Exception as failure:
            error = f'{type(failure).__name__}: {failure}'
        self._agree(error)
        if self.root:
            self.client.bind_teachers(self.identities)
        self.active = None
        self.local_chunk = args.mopd_teacher_chunk_size // self.dp_size
        self.window = None
        self.cursor = 0
        self.current = None
        self.current_index = None
        self.teacher_metrics = {}
        self._completion_window = None

    def _broadcast(self, packet):
        data = [packet]
        dist.broadcast_object_list(data, src=0, group=self.control)
        return data[0]

    def _agree(self, error):
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error, group=self.control)
        require(not any(errors), 'Remote teacher student preparation failed: ' + '; '.join(
            f'rank{i}: {value}' for i, value in enumerate(errors) if value))

    @contextmanager
    def encoding_guard(self):
        """Every rank finishes CPU encoding before any rank starts backward."""
        error = None
        try:
            yield
        except Exception as failure:
            error = f'{type(failure).__name__}: {failure}'
        self._agree(error)

    def rebalance(self, batch):
        """Move complete raw occurrences, including student R3 traces, together."""
        local = None
        error = None
        if self.tp_rank == 0:
            try:
                local = []
                for index, raw in enumerate(batch):
                    sample = copy.deepcopy(raw)
                    sample_id = f'{self.trainer._step}:{self.dp_rank}:{index}'
                    sample['_mopd_remote_occurrence'] = sample_id
                    prompt = sample.get('prompt_length')
                    require(type(prompt) is int and prompt > 0, 'Remote teacher balancing requires rollout prompt usage')
                    response = sample['response_token_ids']
                    if response and isinstance(response[0], (tuple, list)):
                        response = response[0]
                    local.append(PlannedSample(
                        sample_id, str(self.trainer._get_teacher_idx_for_sample(sample)),
                        prompt + len(response), sample))
            except Exception as failure:
                error = f'{type(failure).__name__}: {failure}'
        self._agree(error)
        gathered = [None] * self.dp_size if self.root else None
        if self.tp_rank == 0:
            dist.gather_object(local, gathered, dst=0, group=self.leaders)
        partitions = None
        error = None
        if self.root:
            try:
                samples = [sample for rank in gathered for sample in rank]
                require(len(samples) == get_args().global_batch_size, 'Global remote-teacher sample count changed')
                partitions = balance_teacher_samples(samples, sorted(self.identities, key=int), self.dp_size)
                partitions = [[sample.payload for sample in rank] for rank in partitions]
            except Exception as failure:
                error = f'{type(failure).__name__}: {failure}'
        error = self._broadcast(error)
        require(error is None, f'Remote teacher global batch balancing failed: {error}')
        result = [None]
        if self.tp_rank == 0:
            dist.scatter_object_list(result, partitions, src=0, group=self.leaders)
        dist.broadcast_object_list(result, src=self.tp_root, group=self.tp_group)
        return result[0]

    @staticmethod
    def teacher_record(raw):
        # The teacher gets canonical messages and sampled IDs, never student
        # expert routes. The server independently checks the resulting encoding.
        keys = ('messages', 'images', 'videos', 'audios', 'tools', 'objects', 'uuid',
                'response_token_ids', 'response_loss_mask', 'add_eos', 'domain', 'id')
        record = {key: copy.deepcopy(raw[key]) for key in keys if key in raw}
        json.dumps(record, allow_nan=False)  # reject non-wire inputs before any rank starts backward
        return record

    def prepare_completion_window(self, raw_batch, data_source):
        from .completion_window import IncrementalRolloutWindow
        require(self._completion_window is None and self.active is None, 'Previous optimizer window is still active')
        self._completion_window = IncrementalRolloutWindow(self.trainer, self, data_source)
        return self._completion_window.begin(raw_batch)

    def prepare_window(self, raw_groups, batches, policy_step, *, window_samples=None):
        require(get_rerun_state_machine().get_mode() == RerunMode.DISABLED,
                'Remote streaming requires native rerun_mode=disabled')
        window_samples = get_args().global_batch_size if window_samples is None else window_samples
        require(type(window_samples) is int and 0 < window_samples <= get_args().global_batch_size
                and window_samples % self.trainer.args.mopd_teacher_chunk_size == 0,
                'Teacher subwindow must contain complete RPC chunks inside the native optimizer window')
        self.cursor = 0
        self.current = None
        self.current_index = None
        self.teacher_metrics = {}
        self.batches = batches
        self.bindings = []
        local = []
        error = None
        try:
            require(len(raw_groups) == len(batches)
                    and len(batches) * self.dp_size == window_samples
                    and len(batches) % self.local_chunk == 0,
                    'Local student chunks must cover the complete window')
            for (raw, teacher_index), batch in zip(raw_groups, batches):
                require(len(raw) == 1, 'Remote streaming currently uses native MBS1')
                binding = bind_encoded_sample(raw[0]['_mopd_remote_occurrence'], str(teacher_index), batch)
                self.bindings.append(binding)
                local.append((binding, self.teacher_record(raw[0])))
        except Exception as failure:
            error = f'{type(failure).__name__}: {failure}'
        self._agree(error)
        gathered = [None] * self.dp_size if self.root else None
        if self.tp_rank == 0:
            dist.gather_object(local, gathered, dst=0, group=self.leaders)
        packet = None
        if self.root:
            try:
                window = WindowIdentity(
                    self.run_id, uuid.uuid4().hex, int(policy_step), window_samples,
                    len(batches) // self.local_chunk)
                requests = []
                for index, offset in enumerate(range(0, len(batches), self.local_chunk)):
                    entries = [entry for rank in gathered for entry in rank[offset:offset + self.local_chunk]]
                    bindings, records = map(list, zip(*entries))
                    request = ScoreRequest(window, index, bindings, records)
                    size = sum(len(binding.loss_positions) * self.identities[binding.teacher_key].hidden_size * 2
                               for binding in bindings)
                    require(size + MAX_METADATA_BYTES + 256 * len(bindings) + 8 <= self.client.max_response_bytes,
                            'Teacher chunk exceeds the RPC byte limit including metadata')
                    requests.append(TeacherChunk(index, tuple(item.sample_id for item in bindings), request, max(size, 1)))
                active = StreamingTeacherWindow(
                    requests, lambda chunk: self.client.score(chunk.request),
                    expected_samples=window.samples,
                    max_inflight_bytes=self.trainer.args.mopd_teacher_max_inflight_bytes,
                    timeout_seconds=self.trainer.args.mopd_teacher_rpc_timeout,
                    response_bytes=lambda result: sum(row.hidden.numel() * row.hidden.element_size() for row in result[0]),
                )
                # Admission must finish before acquiring the server lease.
                self.client.open(window)
                self.window = active.begin()
                packet = {'window': asdict(window), 'error': None}
            except Exception as failure:
                packet = {'error': f'{type(failure).__name__}: {failure}'}
        packet = self._broadcast(packet)
        require(packet['error'] is None, f'Remote teacher window failed: {packet["error"]}')
        self.active = WindowIdentity(**packet['window'])
        return iter(self)

    def __iter__(self):
        return self

    def _acquire_chunk(self):
        buffers = None
        error = None
        if self.root:
            try:
                if self.current_index is not None:
                    self.window.release(self.current_index)
                chunk, (results, metrics) = self.window.acquire()
                self.current_index = chunk.index
                buffers = [torch.cat([row.hidden.reshape(-1) for row in results[offset:offset + self.local_chunk]])
                           for offset in range(0, len(results), self.local_chunk)]
                for key, value in metrics.items():
                    if 'peak_' in key:
                        self.teacher_metrics[key] = max(self.teacher_metrics.get(key, 0), value)
                    else:
                        self.teacher_metrics[key] = self.teacher_metrics.get(key, 0) + value
            except Exception as failure:
                error = f'{type(failure).__name__}: {failure}'
        error = self._broadcast(error)
        require(error is None, f'Remote teacher chunk failed before student forward: {error}')
        bindings = self.bindings[self.cursor:self.cursor + self.local_chunk]
        sizes = [len(row.loss_positions) * self.identities[row.teacher_key].hidden_size for row in bindings]
        stamp = time.perf_counter()
        flat = scatter_hidden_buffers(
            buffers, expected_elements=sum(sizes), dp_rank=self.dp_rank, leader_ranks=self.leader_ranks,
            tp_root=self.tp_root, tp_group=self.tp_group, control_group=self.control)
        self.teacher_metrics['student_hidden_distribute_s'] = (
            self.teacher_metrics.get('student_hidden_distribute_s', 0.0) + time.perf_counter() - stamp)
        self.current = [
            HiddenResult(binding, self.identities[binding.teacher_key].fingerprint,
                         part.view(len(binding.loss_positions), self.identities[binding.teacher_key].hidden_size))
            for binding, part in zip(bindings, flat.split(sizes))
        ]

    def __next__(self):
        if self.cursor == len(self.batches):
            raise StopIteration
        if self.cursor % self.local_chunk == 0:
            self._acquire_chunk()
        result = self.current[self.cursor % self.local_chunk]
        head = self.heads[result.binding.teacher_key]
        hidden = result.hidden.unsqueeze(0).to(torch.cuda.current_device(), non_blocking=True)
        if hidden.shape[1] == 0:
            hidden = hidden.new_zeros((1, 1, self.identities[result.binding.teacher_key].hidden_size))
        batch = dict(self.batches[self.cursor])
        batch['_mopd_remote_teacher_target'] = TeacherHeadTarget(hidden, head, head.weight)
        self.cursor += 1
        return batch

    def finish_before_update(self):
        if getattr(self, '_completion_window', None) is not None:
            result = self._completion_window.finish()
            self._completion_window = None
            return result
        return self._finish_teacher_window()

    def finish_completion_chunk(self):
        return self._finish_teacher_window(record_metrics=False)

    def _finish_teacher_window(self, *, record_metrics=True):
        self._agree(None if self.cursor == len(self.batches) else
                    'Native schedule did not consume the complete teacher window')
        packet = None
        if self.root:
            try:
                if self.current_index is not None:
                    self.window.release(self.current_index)
                    self.current_index = None
                metrics = {**self.teacher_metrics, **self.window.finish()}
                self.client.close()
                packet = {'metrics': metrics, 'error': None}
            except Exception as failure:
                packet = {'error': f'{type(failure).__name__}: {failure}'}
        packet = self._broadcast(packet)
        require(packet['error'] is None, f'Remote teacher window incomplete before optimizer: {packet["error"]}')
        self.current = None
        self.active = None
        return packet['metrics']

    def abort(self):
        # On an error all training ranks leave before an optimizer update.
        # The coordinator settles its HTTP request before attempting remote abort.
        if getattr(self, '_completion_window', None) is not None:
            self._completion_window.abort()
            self._completion_window = None
        if self.root:
            if self.window is not None:
                self.window.abort()
            if self.client is not None:
                self.client.abort()
        self.current = None
        self.active = None
