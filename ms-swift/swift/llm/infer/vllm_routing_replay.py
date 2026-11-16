# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, MOPD contributors.
"""Graph-compatible R3 capture adapter for the pinned public vLLM 0.11.0.

The physical-KV-slot design follows public vLLM v0.14.0's
``routed_experts_capturer``. This adapter uses existing worker RPCs instead
of changing EngineCore's wire schema or depending on an internal vLLM fork.
"""
import os
import re
from contextvars import ContextVar
from functools import wraps

import numpy as np
import torch

from .routing_protocol import ROUTING_PROTOCOL_VERSION, RoutingSlotStore

_routing_call = ContextVar('mopd_vllm_routing_call', default=None)
BOOTSTRAP_ENV = 'SWIFT_MOPD_ROUTING_EXPORT'
CONFIG_KEY = 'mopd_routing_export'


class RoutingCall:
    def __init__(self, capturer, layer):
        self.capturer = capturer
        self.layer = layer
        self.offset = 0

    def capture(self, indices, dp_metadata):
        self.capturer.select_calls += 1
        # 0.11's naive DP dispatch selects experts AFTER gathering DP tokens.
        # Modular/chunked kernels select local rows. Preserve only this DP
        # replica's rows in either case; never confuse DP0 with local request 0.
        if dp_metadata is not None and self.layer.dp_size > 1:
            sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
            local = sizes[self.capturer.dp_rank]
            if indices.shape[0] == sum(sizes):
                start = sum(sizes[:self.capturer.dp_rank])
                indices = indices[start:start + local]
            elif indices.shape[0] != local:
                raise RuntimeError('Unknown vLLM DP routing layout; refusing misaligned capture.')
        count = indices.shape[0]
        end = self.offset + count
        if end > self.capturer.buffer.shape[0]:
            raise RuntimeError('MoE routing capture exceeded its persistent graph buffer.')
        self.capturer.buffer[self.offset:end, self.layer._mopd_layer_index].copy_(indices)
        self.offset = end


class VllmRoutingCapturer:
    def __init__(self, runner, layers, tp_rank, dp_rank):
        self.runner = runner
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        config = runner.model_config.hf_text_config
        self.num_layers = config.num_hidden_layers
        self.topk = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.dtype = torch.uint8 if self.num_experts < 256 else torch.int32
        self.missing = 255 if self.dtype == torch.uint8 else -1
        self.store = None
        self.step = None
        self.buffer = None
        self.select_calls = 0
        self.calls_before_step = 0
        self.model_steps = 0
        self.graph_replay_steps = 0
        self.prefix_cache_hit_tokens = 0
        self.model_epoch = 0
        self.new_request_ids = set()
        if tp_rank != 0:
            return
        self.buffer = torch.full(
            (runner.max_num_tokens, self.num_layers, self.topk), self.missing,
            dtype=self.dtype, device=runner.device)
        for index, layer in layers:
            layer._mopd_capturer = self
            layer._mopd_layer_index = index

    def initialize_slots(self):
        if self.tp_rank:
            return
        cache = self.runner.kv_cache_config
        if len(cache.kv_cache_groups) != 1:
            raise ValueError('R3 capture currently requires one full-attention KV cache group.')
        table = self.runner.input_batch.block_table[0]
        if table.dcp_world_size != 1:
            raise ValueError('R3 capture currently requires decode_context_parallel_size=1.')
        self.store = RoutingSlotStore(
            cache.num_blocks * table.block_size, self.num_layers, self.topk,
            self.num_experts, table.block_size)

    def before_execute(self, schedule):
        self.step = None
        if self.tp_rank:
            return
        if self.store is None:
            raise RuntimeError('R3 KV routing slots were not initialized before generation.')
        self.store.finish_requests(schedule.finished_req_ids)
        self.store.suspend_requests(schedule.num_scheduled_tokens)
        self.new_request_ids = {request.req_id for request in schedule.scheduled_new_reqs}
        self.store.start_requests(self.new_request_ids)
        self.calls_before_step = self.select_calls
        # Outside graph replay, on the runner's current stream. Graphs contain
        # the subsequent per-layer copy_ operations into this fixed allocation.
        self.buffer.fill_(self.missing)

    def after_prepare(self, schedule):
        if self.tp_rank:
            return
        batch = self.runner.input_batch
        table = batch.block_table[0]
        count = schedule.total_num_scheduled_tokens
        views = []
        for index, request_id in enumerate(batch.req_ids):
            start = int(batch.num_computed_tokens_cpu[index])
            length = start + schedule.num_scheduled_tokens[request_id]
            if request_id in self.new_request_ids:
                self.prefix_cache_hit_tokens += start
            blocks = table.block_table.np[index, :table.num_blocks_per_row[index]].copy()
            views.append((request_id, blocks, length))
        self.step = (count, table.slot_mapping.np[:count].copy(), views)

    def after_execute(self):
        if self.step is None:
            return
        count, slots, views = self.step
        # Copy the real GPU input-ID buffer, not async scheduling's possibly
        # stale CPU token list. Both copies happen before any next model step.
        tokens = self.runner.input_ids.gpu[:count].cpu().numpy()
        routes = self.buffer[:count].cpu().numpy()
        self.store.write_step(slots, tokens, routes, views)
        self.model_steps += 1
        # A filled buffer without a Python top-k call proves that the capture
        # copies executed inside a replayed CUDA graph, not only in eager mode.
        if count and self.select_calls == self.calls_before_step:
            self.graph_replay_steps += 1
        self.step = None

    def capabilities(self):
        return {
            'version': ROUTING_PROTOCOL_VERSION, 'source': self.tp_rank == 0,
            'ready': self.tp_rank != 0 or self.store is not None,
            'layers': self.num_layers, 'topk': self.topk, 'experts': self.num_experts,
            'cuda_graph_capture': True, 'kv_slot_prefix_capture': True,
        }

    def observations(self):
        return {
            'model_epoch': self.model_epoch, 'model_steps': self.model_steps,
            'cuda_graph_replay_steps': self.graph_replay_steps,
            'prefix_cache_hit_tokens': self.prefix_cache_hit_tokens,
        }

    def clear(self):
        if self.tp_rank == 0:
            self.store.reset()
        self.model_epoch += 1
        self.model_steps = self.graph_replay_steps = self.prefix_cache_hit_tokens = 0


def install_vllm_routing_hooks():
    """Install once per worker process; disabled runner instances are untouched."""
    import vllm
    from vllm.distributed import get_dp_group, get_tensor_model_parallel_rank
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, '_mopd_routing_hooks', False):
        return
    if vllm.__version__.split('+')[0] != '0.11.0':
        raise ValueError('This R3 adapter targets public vLLM 0.11.0; other versions need their native adapter.')

    original_select = FusedMoE.select_experts

    @wraps(original_select)
    def select_experts(*args, **kwargs):
        result = original_select(*args, **kwargs)
        call = _routing_call.get()
        if call is not None:
            call.capture(result[1], get_forward_context().dp_metadata)
        return result

    def wrap_layer_forward(original):
        @wraps(original)
        def forward(layer, *args, **kwargs):
            capturer = getattr(layer, '_mopd_capturer', None)
            if capturer is None:
                return original(layer, *args, **kwargs)
            current = _routing_call.get()
            if current is not None and current.layer is layer:
                return original(layer, *args, **kwargs)  # nested chunked forward
            token = _routing_call.set(RoutingCall(capturer, layer))
            try:
                return original(layer, *args, **kwargs)
            finally:
                _routing_call.reset(token)
        return forward

    original_load = GPUModelRunner.load_model

    @wraps(original_load)
    def load_model(runner, *args, **kwargs):
        result = original_load(runner, *args, **kwargs)
        if not runner.vllm_config.additional_config.get(CONFIG_KEY, False):
            return result
        parallel = runner.parallel_config
        if (parallel.pipeline_parallel_size != 1 or parallel.enable_eplb
                or parallel.enable_dbo or runner.vllm_config.speculative_config is not None
                or runner.vllm_config.kv_transfer_config is not None):
            raise ValueError('R3 capture requires PP1, no EPLB/DBO/speculation or external KV transfer.')
        if runner.model_config.hf_config.model_type not in ('qwen3_moe', 'qwen3_vl_moe'):
            raise ValueError('The current R3 capture contract is Qwen3 MoE / Qwen3-VL MoE.')
        layers = []
        for name, layer in runner.compilation_config.static_forward_context.items():
            if not isinstance(layer, FusedMoE):
                continue
            match = re.search(r'(?:^|\.)layers\.(\d+)\.', name)
            if match is None or layer.is_sequence_parallel:
                raise ValueError('R3 capture requires named decoder layers without rollout MoE sequence parallelism.')
            layers.append((int(match.group(1)), layer))
        expected = runner.model_config.hf_text_config.num_hidden_layers
        if sorted(index for index, _ in layers) != list(range(expected)):
            raise ValueError('R3 capture must account for every decoder MoE layer exactly once.')
        runner._mopd_routing_capturer = VllmRoutingCapturer(
            runner, layers, get_tensor_model_parallel_rank(), get_dp_group().rank_in_group)
        buffer = runner._mopd_routing_capturer.buffer
        if buffer is not None:
            # This persistent allocation is created after vLLM's model loader
            # profiler; include it before sizing the KV cache.
            runner.model_memory_usage += buffer.numel() * buffer.element_size()
        return result

    original_kv = GPUModelRunner.initialize_kv_cache

    @wraps(original_kv)
    def initialize_kv_cache(runner, *args, **kwargs):
        result = original_kv(runner, *args, **kwargs)
        capturer = getattr(runner, '_mopd_routing_capturer', None)
        if capturer is not None:
            capturer.initialize_slots()
        return result

    original_prepare = GPUModelRunner._prepare_inputs

    @wraps(original_prepare)
    def prepare_inputs(runner, schedule, *args, **kwargs):
        result = original_prepare(runner, schedule, *args, **kwargs)
        capturer = getattr(runner, '_mopd_routing_capturer', None)
        if capturer is not None:
            capturer.after_prepare(schedule)
        return result

    original_execute = GPUModelRunner.execute_model

    @wraps(original_execute)
    def execute_model(runner, schedule, *args, **kwargs):
        capturer = getattr(runner, '_mopd_routing_capturer', None)
        if capturer is not None:
            capturer.before_execute(schedule)
        result = original_execute(runner, schedule, *args, **kwargs)
        if capturer is not None:
            capturer.after_execute()
        return result

    FusedMoE.select_experts = staticmethod(select_experts)
    FusedMoE.forward_impl = wrap_layer_forward(FusedMoE.forward_impl)
    FusedMoE.forward_impl_chunked = wrap_layer_forward(FusedMoE.forward_impl_chunked)
    GPUModelRunner.load_model = load_model
    GPUModelRunner.initialize_kv_cache = initialize_kv_cache
    GPUModelRunner._prepare_inputs = prepare_inputs
    GPUModelRunner.execute_model = execute_model
    GPUModelRunner._mopd_routing_hooks = True


def bootstrap_worker_routing():
    if os.environ.get(BOOTSTRAP_ENV) == '1':
        install_vllm_routing_hooks()
