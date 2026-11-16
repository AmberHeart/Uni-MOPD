# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, MOPD contributors.
"""Complete-response delivery feeding one native full-window student schedule.

Only the existing vLLM coordinator owns HTTP readers. All data distribution,
encoding, local/remote teacher preparation and model work remain on native rank
threads. Closing an input chunk never zeros, reduces or updates gradients.
"""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import threading
import uuid

import torch.distributed as dist
from megatron.training import get_args

from swift.llm.infer.completion_stream import CompletionWindow
from swift.llm.infer.protocol import RolloutOutput
from .remote_teacher_protocol import require
from .remote_teacher_stream import PlannedSample, balance_teacher_samples
from .rollout_samples import sample_indices, sample_record


class IncrementalRolloutWindow:
    def __init__(self, trainer, backend, data_source):
        self.trainer, self.backend, self.data_source = trainer, backend, data_source
        self.owner_thread = threading.get_ident()
        self.coordinator = trainer._main_process_rank
        self.is_coordinator = backend.rank == self.coordinator
        self.samples = get_args().global_batch_size
        self.policy_step = int(trainer._step)
        self.chunk_size = trainer.args.mopd_rollout_completion_chunk_size
        self.producer = self.lease = None
        self.current_samples = self.consumed_samples = 0
        self.seen_local = set()
        self.teacher_metrics = {}
        self.raw_by_id = {}
        self.sample_routes = {}
        self.log_ids, self.log_samples = [], {}
        self.closed = False

    def _owner(self):
        require(threading.get_ident() == self.owner_thread, 'Native streaming work must stay on the rank owner thread')
        require(not self.closed, 'Native completion window is closed')

    def _broadcast(self, packet):
        value = [packet]
        dist.broadcast_object_list(value, src=self.coordinator, group=self.backend.control)
        require(value[0]['error'] is None, f'Complete-response window failed before update: {value[0]["error"]}')
        return value[0]

    def begin(self, raw_batch):
        self._owner()
        error = None
        local = None
        try:
            local = self.trainer._get_local_rollout_batch(raw_batch)
            local = self.trainer._preprocess_rollout_inputs(copy.deepcopy(local))
            local = self.trainer._set_inputs_system(local)
        except Exception as failure:
            error = f'{type(failure).__name__}: {failure}'
        self.backend._agree(error)
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local, group=self.backend.control)
        raw = [row for rank in gathered for row in rank]
        require(len(raw) == self.samples, 'Selected rollout occurrences do not cover the native global batch')
        if self.trainer._last_loaded_step != self.policy_step:
            self.trainer._move_model_to_vllm()
            dist.barrier()
            self.trainer._last_loaded_step = self.policy_step
        packet = None
        if self.is_coordinator:
            try:
                ids = tuple(f'{self.backend.run_id}:{self.policy_step}:{index}' for index in range(self.samples))
                self.identity = CompletionWindow(self.backend.run_id, uuid.uuid4().hex, self.policy_step, ids)
                self.raw_by_id = dict(zip(ids, raw))
                self.sample_routes = {
                    key: {'domain': row.get('domain'), 'teacher': str(self.trainer._get_teacher_idx_for_sample(row))}
                    for key, row in self.raw_by_id.items()
                }
                self.log_ids = [ids[index] for index in sample_indices(
                    self.samples, self.trainer.args.mopd_rollout_samples_per_step,
                    self.trainer.args.seed, self.policy_step)]
                client = self.trainer.vllm_client
                client.require_routing_replay()
                if self.trainer._mopd_domain_generation:
                    client.require_domain_generation()
                requests = self.trainer._inputs_to_requests(raw)
                self.producer = client.begin_completion_window(
                    requests, self.trainer._get_request_config(), self.identity,
                    max_samples=2 * self.chunk_size,
                    max_tokens=2 * self.chunk_size * (
                        getattr(self.trainer.args, 'mopd_max_sequence_length', None)
                        or int(self.trainer.args.max_length) + self.trainer.max_completion_length),
                    max_bytes=self.trainer.args.mopd_rollout_queue_max_bytes,
                    timeout_seconds=self.trainer.args.mopd_teacher_rpc_timeout)
                packet = {'error': None, 'window': asdict(self.identity)}
            except Exception as failure:
                packet = {'error': f'{type(failure).__name__}: {failure}'}
        packet = self._broadcast(packet)
        self.window = packet['window']
        return self

    def __iter__(self):
        return self

    def _finish_current(self):
        if not self.current_samples:
            return
        # Native PP1 calls the next iterator item only after the previous
        # microbatch's backward. The last chunk is closed by finish(), after
        # the entire native forward/backward invocation returns.
        metrics = self.backend.finish_completion_chunk()
        for key, value in metrics.items():
            if 'peak_' in key or key.endswith('_max'):
                self.teacher_metrics[key] = max(self.teacher_metrics.get(key, 0), value)
            else:
                self.teacher_metrics[key] = self.teacher_metrics.get(key, 0) + value
        packet = None
        if self.is_coordinator:
            try:
                self.producer.queue.release(self.lease)
                self.lease = None
                packet = {'error': None}
            except Exception as failure:
                packet = {'error': f'{type(failure).__name__}: {failure}'}
        self._broadcast(packet)
        self.consumed_samples += self.current_samples
        self.current_samples = 0
        self.backend.batches, self.backend.bindings = [], []

    def _encode(self, raw, teacher):
        batch = self.trainer._encode_batch(raw)
        batch.update(data_source=self.data_source, _rollout_weight_step=self.policy_step, teacher_idx=teacher)
        return batch

    def _load_chunk(self):
        packet = partitions = None
        if self.is_coordinator:
            try:
                self.lease = self.producer.queue.acquire(
                    self.chunk_size, multiple=self.backend.completion_multiple)
                rows = []
                for sample in self.lease.samples:
                    raw = self.raw_by_id.pop(sample.occurrence_id)
                    output = RolloutOutput.model_validate(sample.output)
                    result = self.trainer._postprocess_rollout_outputs([raw], [output])[0]
                    result['_mopd_remote_occurrence'] = sample.occurrence_id
                    teacher = str(self.trainer._get_teacher_idx_for_sample(result))
                    require(result.get('domain') == self.sample_routes[sample.occurrence_id]['domain']
                            and teacher == self.sample_routes[sample.occurrence_id]['teacher'],
                            'Rollout changed the selected domain or serving teacher')
                    rows.append(PlannedSample(sample.occurrence_id, teacher, sample.tokens, result))
                    if sample.occurrence_id in self.log_ids:
                        self.log_samples[sample.occurrence_id] = sample_record(
                            result, step=self.policy_step,
                            row_index=int(sample.occurrence_id.rsplit(':', 1)[1]), batch_size=self.samples)
                by_dp = balance_teacher_samples(rows, self.backend.completion_teacher_keys, self.backend.dp_size)
                partitions = [[row.payload for row in by_dp[self.backend.rank_to_dp[rank]]]
                              for rank in range(dist.get_world_size())]
                packet = {'error': None, 'samples': len(rows)}
            except Exception as failure:
                packet = {'error': f'{type(failure).__name__}: {failure}'}
        packet = self._broadcast(packet)
        received = [None]
        dist.scatter_object_list(received, partitions, src=self.coordinator, group=self.backend.control)
        local = received[0]
        encoded, grouped = [], []
        with self.backend.encoding_guard():
            require(len(local) * self.backend.dp_size == packet['samples'], 'Incomplete per-rank completion chunk')
            ids = {row['_mopd_remote_occurrence'] for row in local}
            require(len(ids) == len(local) and not ids & self.seen_local, 'Duplicated native student occurrence')
            self.seen_local.update(ids)
            grouped = self.trainer._group_batch_by_teacher(local, len(local))
            encoded = [self._encode(raw, teacher) for raw, teacher in grouped]
        self.backend.prepare_window(grouped, encoded, self.policy_step, window_samples=packet['samples'])
        self.current_samples = packet['samples']

    def __next__(self):
        self._owner()
        if self.current_samples and self.backend.cursor == len(self.backend.batches):
            self._finish_current()
        if self.consumed_samples == self.samples:
            raise StopIteration
        if not self.current_samples:
            self._load_chunk()
        return next(self.backend)

    def finish(self):
        self._owner()
        self._finish_current()
        self.backend._agree(None if (
            self.consumed_samples == self.samples and len(self.seen_local) * self.backend.dp_size == self.samples
        ) else 'Native streaming did not consume exactly the full selected optimizer window')
        packet = None
        if self.is_coordinator:
            try:
                metrics = self.producer.finish()
                require(not self.raw_by_id, 'Selected source records remain outside the consumed window')
                metrics.pop('events')
                packet = {'error': None, 'metrics': metrics,
                          'log_samples': [self.log_samples[key] for key in self.log_ids]}
            except Exception as failure:
                packet = {'error': f'{type(failure).__name__}: {failure}'}
        packet = self._broadcast(packet)
        self.trainer._log_student_rollout_samples(
            packet['log_samples'], step=self.policy_step, preselected=True, batch_size=self.samples)
        self.closed = True
        return packet['metrics']

    def abort(self):
        if self.producer is not None:
            self.producer.abort(RuntimeError('Native student/teacher window aborted before its optimizer update'))
        self.closed = True
