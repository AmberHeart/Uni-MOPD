# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, MOPD contributors.
"""Native forward-only Megatron teacher bank for the independent teacher node."""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import threading
import time

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args
from megatron.training.utils import unwrap_model

from ..utils import forward_step_helper
from .remote_teacher_heads import checkpoint_hash, tokenizer_hash
from .remote_teacher_protocol import (
    MODEL_TENSOR_KEYS, HiddenResult, ScoreRequest, TeacherIdentity, bind_encoded_sample, require, tensor_hash,
)
from .remote_teacher_stream import TeacherInputPrefetch
from .streamed_distillation import head_weight


class NativeTeacherEngine:
    """Reuse the existing teacher loader/forward; never build a student or Adam."""

    def __init__(self, trainer, control_group):
        self.trainer = trainer
        self.control_group = control_group
        self.owner_thread = threading.get_ident()
        self.identities = {}
        self.tp_rank = mpu.get_tensor_model_parallel_rank()
        self.dp_rank = mpu.get_data_parallel_rank()
        self.dp_size = mpu.get_data_parallel_world_size()
        # Canonical encoding and its identity checks are CPU work. Only
        # _prepare_batch moves a selected input onto this teacher's GPU.
        self.trainer.device = torch.device('cpu')
        self.teacher_forward_calls = 0
        self.cpu_prefetch = getattr(trainer.args, 'teacher_cpu_prefetch', False)
        self.cpu_encoding_args = copy.copy(get_args()) if self.cpu_prefetch else None
        self._cpu_preparation_failed = False
        self._cpu_encode_seconds = 0.0

    def _check_owner(self):
        require(threading.get_ident() == self.owner_thread,
                'Megatron teacher forwards/collectives must run on the native owner thread')
        require(not getattr(self, '_cpu_preparation_failed', False),
                'CPU teacher preparation failed; do not reuse this teacher engine')

    def _agree_errors(self, error):
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error, group=self.control_group)
        require(not any(errors), 'Teacher input preparation failed: ' + '; '.join(
            f'rank{rank}: {value}' for rank, value in enumerate(errors) if value))

    def load_bank(self):
        self._check_owner()
        native = get_args()
        require(native.pipeline_model_parallel_size == native.context_parallel_size == 1,
                'Remote teacher bank currently supports PP1/CP1')
        require(native.bf16 and self.trainer.args.mopd_router_replay == 'disabled',
                'Teacher bank must retain BF16 and independent natural routing')
        for index, path in enumerate(self.trainer.teacher_model_paths):
            self.trainer._load_teacher_model(path, ModelType.encoder_or_decoder, teacher_idx=index)
            if not self.trainer.is_multi_teacher:
                self.trainer._check_teacher_tokenizer_matches_student(index, path, path)
            models = self.trainer.teacher_groups[index]
            require(len(models) == 1, 'Teacher bank does not support virtual/pipeline model chunks')
            if self.trainer.args.mopd_train_flash_attention_version == '2':
                from .attention_backend import force_native_flash_attention_v2
                force_native_flash_attention_v2(models[0])
            model = unwrap_model(models[0])
            language = getattr(model, 'language_model', model)
            head = head_weight(language)
            norm = language.decoder.final_layernorm.weight
            require(head.dtype == norm.dtype == torch.bfloat16 and not head.requires_grad,
                    'Teacher head/norm did not retain native frozen BF16')
            config = json.loads((Path(path) / 'config.json').read_text())
            config = config.get('text_config', config)
            local = {'tp_rank': self.tp_rank, 'head': tensor_hash(head), 'norm': tensor_hash(norm),
                     'hidden': head.shape[1], 'vocab_shard': head.shape[0]}
            replicas = [None] * dist.get_world_size()
            dist.all_gather_object(replicas, local, group=self.control_group)
            tp_size = mpu.get_tensor_model_parallel_world_size()
            shards = []
            for tp in range(tp_size):
                values = [record for record in replicas if record['tp_rank'] == tp]
                require(values and all(record == values[0] for record in values),
                        'Teacher head replica identity differs across DP ranks')
                shards.append(values[0]['head'])
            require(all(record['norm'] == local['norm'] for record in replicas),
                    'Teacher final-norm identity differs across workers')
            identity = TeacherIdentity(
                str(index), checkpoint_hash(Path(path)), tokenizer_hash(self.trainer.template.tokenizer),
                local['norm'], tuple(shards), local['hidden'], int(config['vocab_size']),
                int(local['vocab_shard'] * tp_size))
            self.identities[str(index)] = identity
        return self.identities

    def _prepare(self, request, index):
        expected = request.bindings[index]
        record = copy.deepcopy(request.records[index])
        encoding_args = getattr(self, 'cpu_encoding_args', None)
        encoded = (self.trainer._encode_batch([record]) if encoding_args is None else
                   self.trainer._encode_batch([record], megatron_args=encoding_args))
        require(all(value.device.type == 'cpu' for key, value in encoded.items()
                    if key in MODEL_TENSOR_KEYS and isinstance(value, torch.Tensor)),
                'Teacher input encoding must remain CPU-only')
        actual = bind_encoded_sample(expected.sample_id, expected.teacher_key, encoded)
        require(actual == expected, f'Teacher re-encoding changed tokens/media/labels: {expected.sample_id}')
        return actual, {key: value for key, value in encoded.items()
                        if key in MODEL_TENSOR_KEYS or key == 'num_samples'}

    def _pack_indices(self, request, indices):
        limit = self.trainer.args.teacher_forward_tokens
        # Native BF16 fused kernels are not batch invariant, so a packed forward
        # and a singleton forward of the same example do not produce identical
        # hidden states. Preserve the original per-example teacher computation;
        # RPC chunks and teacher/student overlap remain batched.
        batches = []
        for index in indices:
            length = request.bindings[index].input_length
            require(length <= limit, 'One teacher input exceeds the admitted forward token budget')
            batches.append([index])
        return batches

    @staticmethod
    def _validate_packed_inputs(singles, encoded):
        """Packing may remove SP tail padding, not alter any real input token."""
        boundaries = [0]
        for _, single in singles:
            positions = single['text_position_ids'].reshape(-1)
            resets = positions[1:].eq(0).nonzero(as_tuple=True)[0]
            length = int(resets[0]) + 1 if resets.numel() else positions.numel()
            require(torch.equal(positions[:length], torch.arange(length, dtype=positions.dtype)),
                    'Singleton teacher text positions are not a contiguous causal sequence')
            start, end = boundaries[-1], boundaries[-1] + length
            for key in ('input_ids', 'labels', 'position_ids', 'text_position_ids', 'mm_token_type_ids'):
                value = single.get(key)
                if value is not None:
                    require(key in encoded and torch.equal(value[..., :length], encoded[key][..., start:end]),
                            f'Teacher packing changed real input tokens/positions/provenance: {key}')
            boundaries.append(end)
        require(encoded['labels'][..., boundaries[-1]:].eq(-100).all().item(),
                'Teacher packing added supervised padding')
        if encoded.get('mm_token_type_ids') is not None:
            require(encoded['mm_token_type_ids'][..., boundaries[-1]:].eq(0).all().item(),
                    'Teacher packing added media padding')
        return boundaries

    def _encode_pack(self, request, indices):
        # Verify every example against the actor before packing. This first
        # implementation retains the original single-example encoder oracle;
        # its CPU cost is measured separately from native forward time.
        stamp = time.perf_counter()
        singles = [self._prepare(request, index) for index in indices]
        bindings = [item[0] for item in singles]
        boundaries = None
        if len(indices) == 1:
            encoded = singles[0][1]
        else:
            require(get_args().padding_free, 'Packed teacher forward requires native padding-free attention')
            encoded = self.trainer._encode_batch([copy.deepcopy(request.records[index]) for index in indices])
            encoded = {key: value for key, value in encoded.items()
                       if key in MODEL_TENSOR_KEYS or key == 'num_samples'}
            boundaries = self._validate_packed_inputs(singles, encoded)
            for key in ('pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw'):
                parts = [value[key] for _, value in singles if value.get(key) is not None]
                if parts:
                    require(key in encoded and torch.equal(torch.cat(parts), encoded[key]),
                            f'Teacher packing changed media: {key}')
                else:
                    require(encoded.get(key) is None, f'Teacher packing added media: {key}')
        require(encoded['input_ids'].numel() <= self.trainer.args.teacher_forward_tokens,
                'Packed teacher input exceeds the hard forward token budget')
        self._cpu_encode_seconds += time.perf_counter() - stamp
        return bindings, encoded, boundaries

    def _prepare_pack(self, request, indices, *, encoded_pack=None):
        # GPU transfer, label shift and packed-sequence construction stay on
        # the native owner thread; only the canonical CPU encoding is prefetched.
        bindings, encoded, boundaries = (
            self._encode_pack(request, indices) if encoded_pack is None else encoded_pack)
        data = self.trainer._prepare_batch(encoded)
        labels = data.pop('labels')
        expected_ids = [token for binding in bindings for token in binding.loss_token_ids]
        require(labels[labels != -100].tolist() == expected_ids, 'Teacher packing changed response IDs/order')
        if len(indices) > 1:
            packed = data['packed_seq_params']
            require(packed.num_samples == len(indices)
                    and packed.cu_seqlens_q[:len(indices) + 1].tolist() == boundaries,
                    'Teacher packing lost attention boundaries')
        data.pop('loss_scale', None)
        data['valid_token_indices'] = self.trainer._response_head_indices(labels)
        return bindings, data

    def score_local(self, request):
        """All native teacher ranks execute this same ordered request."""
        self._check_owner()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        local_results = []
        prepare_seconds = forward_seconds = swap_seconds = 0.0
        self._cpu_encode_seconds = 0.0
        input_wait_seconds = 0.0
        prefetch = getattr(self, 'cpu_prefetch', False)
        forward_count = dummy_count = 0
        for key in sorted({binding.teacher_key for binding in request.bindings}, key=int):
            require(key in self.identities, 'Unknown native teacher identity')
            indices = [index for index, binding in enumerate(request.bindings) if binding.teacher_key == key]
            # Round-robin yields equal slot counts after at most one zero-target
            # teacher-only dummy per DP rank; no student data is padded or dropped.
            local_indices = indices[self.dp_rank::self.dp_size]
            planning_error = None
            packs = []
            try:
                packs = self._pack_indices(request, local_indices)
            except Exception as failure:
                planning_error = f'{type(failure).__name__}: {failure}'
            self._agree_errors(planning_error)
            counts = [None] * dist.get_world_size()
            dist.all_gather_object(counts, len(packs), group=self.control_group)
            slots = max(counts)
            if prefetch:
                error = None
                try:
                    # Dummy creation also uses the template. Finish it before
                    # its single CPU encoder can run beside a GPU forward.
                    if slots > len(packs):
                        self.trainer._get_teacher_dummy_batch()
                except Exception as failure:
                    error = f'{type(failure).__name__}: {failure}'
                self._agree_errors(error)
            teacher_index = int(key)
            swap_started = time.perf_counter()
            if self.trainer.offload_teacher_model:
                self.trainer._ensure_teacher_resident(teacher_index)
            else:
                # Dedicated teacher GPUs may keep the complete immutable bank
                # resident when the measured activation budget permits it.
                self.trainer._assert_teacher_uniform_across_ep(teacher_index)
            torch.cuda.synchronize()
            swap_seconds += time.perf_counter() - swap_started
            model = self.trainer.teacher_groups[teacher_index][0]
            preparation = (TeacherInputPrefetch(
                packs, lambda indices: self._encode_pack(request, indices),
                timeout_seconds=self.trainer.args.teacher_rpc_timeout) if prefetch else nullcontext())
            try:
                with preparation as prepared:
                    for slot in range(slots):
                        error = None
                        bindings, data = [], None
                        stamp = time.perf_counter()
                        try:
                            if slot < len(packs):
                                if prefetch:
                                    bindings, data = self._prepare_pack(
                                        request, packs[slot], encoded_pack=next(prepared))
                                else:
                                    bindings, data = self._prepare_pack(request, packs[slot])
                            else:
                                data = self.trainer._get_teacher_dummy_batch()
                                data['valid_token_indices'] = torch.zeros(
                                    (1,), dtype=torch.long, device=torch.cuda.current_device())
                                dummy_count += 1
                            data['mopd_return_hidden_states'] = True
                            data['fp32_output'] = False
                        except Exception as failure:
                            error = f'{type(failure).__name__}: {failure}'
                        # No rank may enter expert collectives while a peer rejects an input.
                        self._agree_errors(error)
                        prepare_seconds += time.perf_counter() - stamp
                        stamp = time.perf_counter()
                        with self.trainer._teacher_args_context(teacher_index), torch.no_grad():
                            output = forward_step_helper(model, data)
                        torch.cuda.synchronize()
                        forward_seconds += time.perf_counter() - stamp
                        forward_count += 1
                        error = None
                        try:
                            if bindings:
                                count = sum(len(binding.loss_positions) for binding in bindings)
                                require(output.ndim == 3 and output.shape == (
                                    1, max(count, 1), self.identities[key].hidden_size)
                                    and output.dtype == torch.bfloat16, 'Invalid native teacher hidden output')
                                require(torch.isfinite(output).all().item(), 'Non-finite native teacher hidden output')
                                if self.tp_rank == 0:
                                    hidden = output[0, :count].detach().cpu().contiguous()
                                    offset = 0
                                    for binding in bindings:
                                        end = offset + len(binding.loss_positions)
                                        # Own each slice's storage before object serialization.
                                        local_results.append(HiddenResult(
                                            binding, self.identities[key].fingerprint, hidden[offset:end].clone()))
                                        offset = end
                        except Exception as failure:
                            error = f'{type(failure).__name__}: {failure}'
                        self._agree_errors(error)
                        del output, data
            except BaseException:
                if prefetch:
                    self._cpu_preparation_failed = True
                raise
            if prefetch:
                input_wait_seconds += preparation.wait_seconds
        self.teacher_forward_calls += forward_count
        # Only TP0 replicas return full hidden, avoiding duplicate response rows.
        gathered = [None] * self.dp_size if dist.get_rank() == 0 else None
        if self.tp_rank == 0:
            dist.gather_object(local_results, gathered, dst=0, group=mpu.get_data_parallel_group_gloo())
        local_metrics = {
            'prepare_s': prepare_seconds, 'forward_s': forward_seconds, 'swap_s': swap_seconds,
            'cpu_encode_s': self._cpu_encode_seconds, 'input_wait_s': input_wait_seconds,
            'wall_s': time.perf_counter() - started,
            'forwards': forward_count, 'dummy_forwards': dummy_count if self.tp_rank == 0 else 0,
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
        }
        metrics = [None] * dist.get_world_size()
        dist.all_gather_object(metrics, local_metrics, group=self.control_group)
        if dist.get_rank() != 0:
            return None
        by_id = {item.binding.sample_id: item for part in gathered for item in part}
        require(len(by_id) == len(request.bindings), 'Native teacher did not return every sample occurrence')
        return [by_id[binding.sample_id] for binding in request.bindings], {
            # Prepare includes re-encoding, transfer and rank agreement; it is
            # not a pure tokenizer timer.
            'teacher_prepare_s': max(item['prepare_s'] for item in metrics),
            'teacher_cpu_encode_s': max(item['cpu_encode_s'] for item in metrics),
            'teacher_input_wait_s': max(item['input_wait_s'] for item in metrics),
            'teacher_forward_s': max(item['forward_s'] for item in metrics),
            'teacher_swap_s': max(item['swap_s'] for item in metrics),
            'teacher_wall_s': max(item['wall_s'] for item in metrics),
            'teacher_forwards_max': max(item['forwards'] for item in metrics),
            'teacher_dummy_forwards_total': sum(item['dummy_forwards'] for item in metrics),
            'teacher_peak_allocated_gib': max(item['peak_allocated_gib'] for item in metrics),
            'teacher_peak_reserved_gib': max(item['peak_reserved_gib'] for item in metrics),
        }

    def score(self, request):
        self._check_owner()
        require(dist.get_rank() == 0, 'Only the teacher coordinator dispatches RPC work')
        command = [{'op': 'score', 'request': request.to_dict()}]
        dist.broadcast_object_list(command, src=0, group=self.control_group)
        return self.score_local(request)

    def heartbeat(self):
        """Keep followers alive during startup, rollouts, student work and saves."""
        self._check_owner()
        require(dist.get_rank() == 0, 'Only the teacher coordinator sends idle heartbeats')
        dist.broadcast_object_list([{'op': 'heartbeat'}], src=0, group=self.control_group)

    def stop(self):
        self._check_owner()
        dist.broadcast_object_list([{'op': 'stop'}], src=0, group=self.control_group)

    def follow(self):
        self._check_owner()
        require(dist.get_rank() != 0, 'The coordinator owns the HTTP work queue')
        while True:
            command = [None]
            dist.broadcast_object_list(command, src=0, group=self.control_group)
            require(isinstance(command[0], dict), 'Invalid native teacher control message')
            operation = command[0]['op']
            if operation == 'stop':
                return
            if operation == 'heartbeat':
                continue
            require(operation == 'score', 'Unknown native teacher control operation')
            self.score_local(ScoreRequest.from_dict(command[0]['request']))
