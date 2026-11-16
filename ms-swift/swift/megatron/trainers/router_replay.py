# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, MOPD contributors.
"""Student-owned routing replay, with explicit full-sequence to SP alignment."""
from typing import Optional

import torch
from megatron.core import mpu
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.routing_replay import RoutingReplayFrame, validate_expert_indices


class StudentRouterReplay:
    """One live frame per local router; intended for synchronous PP1/CP1."""

    def __init__(self, model, mode: str):
        if mode not in ('r2', 'r3'):
            raise ValueError(f'Unsupported router replay mode: {mode}.')
        self.mode = mode
        self.enabled = True
        self.check_recompute = False
        self.routers = sorted(
            [(module.layer_number - 1, module) for module in model.modules()
             if isinstance(module, TopKRouter)],
            key=lambda item: item[0])
        if not self.routers:
            raise ValueError('Router replay requires a student with native MoE TopKRouter layers.')
        if len({index for index, _ in self.routers}) != len(self.routers):
            raise ValueError('Student MoE layer numbers must be unique.')
        for _, router in self.routers:
            config = router.config
            if (router.routing_type == 'sinkhorn' or config.moe_router_fusion
                    or config.moe_expert_capacity_factor is not None
                    or config.moe_router_force_load_balancing):
                raise ValueError(
                    'Router replay requires native non-fused top-k, no token capacity/drop, '
                    'and no forced/random load balancing.')
            if router.is_aux_loss_enabled() and (router.enable_expert_bias or config.moe_router_group_topk):
                raise ValueError(
                    'Replay with router auxiliary loss currently requires ungrouped, unbiased top-k.')
        self.num_layers = self.routers[0][1].config.num_layers
        self.num_experts = self.routers[0][1].config.num_moe_experts
        self.topk = self.routers[0][1].topk
        self.sequence_parallel = self.routers[0][1].config.sequence_parallel
        self.frames = []
        self.completed_batches = 0
        self._counts = [0, 0, 0]  # initial calls, recompute calls, token-layer rows
        self._extras = {name: [] for name in ('forced_tokens', 'changed_tokens', 'recompute_changed_tokens')}
        self._hook = model.register_forward_pre_hook(self._before_forward, with_kwargs=True)

    def _collect_previous(self):
        if self.frames:
            for frame in self.frames:
                self._counts[0] += frame.forward_calls
                self._counts[1] += frame.replay_calls
                if frame.recorded_indices is not None:
                    self._counts[2] += frame.recorded_indices.shape[0]
                for name, values in self._extras.items():
                    value = getattr(frame, name)
                    if torch.is_tensor(value):
                        values.append(value)
            self.completed_batches += 1
            self.frames = []

    def clear(self):
        self._collect_previous()
        for _, router in self.routers:
            router.routing_replay = None

    def stats(self, *, reset: bool = False):
        counts = list(self._counts)
        for frame in self.frames:
            counts[0] += frame.forward_calls
            counts[1] += frame.replay_calls
            if frame.recorded_indices is not None:
                counts[2] += frame.recorded_indices.shape[0]
        result = {
            'routing/initial_calls': counts[0],
            'routing/recompute_calls': counts[1],
            'routing/token_layer_rows': counts[2],
            'routing/microbatches': self.completed_batches + bool(self.frames),
        }
        for name, key in (
            ('forced_tokens', 'routing/rollout_token_layers'),
            ('changed_tokens', 'routing/rollout_changed_token_layers'),
            ('recompute_changed_tokens', 'routing/recompute_changed_token_layers'),
        ):
            values = list(self._extras[name])
            values.extend(getattr(frame, name) for frame in self.frames if torch.is_tensor(getattr(frame, name)))
            result[key] = int(torch.stack(values).sum().item()) if values else 0
        if reset:
            self.clear()
            self._counts = [0, 0, 0]
            self.completed_batches = 0
            self._extras = {name: [] for name in self._extras}
        return result

    def _prepare_forced(self, payload, input_ids: torch.Tensor, labels: torch.Tensor):
        """Validate the actual consumed tokens, then take this rank's SP rows.

        Router IDs describe input-token positions, NOT shifted loss positions.
        Only the coverage requirement uses the already causally shifted labels.
        """
        token_ids = payload['token_ids']
        expert_ids = payload['expert_ids']
        if not torch.is_tensor(token_ids) or not torch.is_tensor(expert_ids):
            raise ValueError('Decode routing payload tensors on CPU before installing frames.')
        if token_ids.device.type != 'cpu' or token_ids.ndim != 1:
            raise ValueError('Routing token IDs must be a one-dimensional CPU tensor.')
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError('Router replay currently requires micro_batch_size=1.')
        if payload['num_experts'] != self.num_experts:
            raise ValueError('Rollout and student expert banks have different sizes.')
        if expert_ids.ndim != 3 or expert_ids.shape[:2] != (len(token_ids), self.num_layers):
            raise ValueError('Rollout routing token/layer layout differs from the student.')
        covered = len(token_ids)
        if covered > input_ids.shape[1]:
            raise ValueError('Rollout routing is longer than the encoded student sequence.')
        if not torch.equal(token_ids.long(), input_ids[0, :covered].detach().cpu().long()):
            raise ValueError('Rollout routing token IDs differ from the actual student input.')
        if labels is None or labels.shape != input_ids.shape:
            raise ValueError('R3 needs full causally shifted labels to validate route coverage.')
        if labels[:, covered:].ne(-100).any():
            raise ValueError('Rollout routing does not cover every supervised prediction input.')
        # The terminal sampled token and right padding can lack rollout routes.
        # They use natural initial-forward routes and replay those during backward.
        full = torch.full((input_ids.shape[1], self.num_layers, self.topk), -1, dtype=torch.long)
        validated = validate_expert_indices(expert_ids, self.num_experts, self.topk)
        if (validated[:, [index for index, _ in self.routers]] < 0).any():
            raise ValueError('Captured rollout prefix has missing MoE routes.')
        full[:covered] = validated
        if self.sequence_parallel:
            size = mpu.get_tensor_model_parallel_world_size()
            rank = mpu.get_tensor_model_parallel_rank()
            if full.shape[0] % size:
                raise ValueError('Student sequence must be padded to its tensor-parallel size.')
            full = full.chunk(size, dim=0)[rank]
        return full.to(input_ids.device)

    def _before_forward(self, model, positional, kwargs):
        context = kwargs.pop('_mopd_routing_context', None)
        self.clear()
        if not self.enabled or not model.training:
            return positional, kwargs  # evaluation always uses natural routing
        forced: Optional[torch.Tensor] = None
        if self.mode == 'r3':
            if context is None:
                raise ValueError('R3 student forward is missing its rollout routing context.')
            if not context['padding']:
                if context['routes'] is None:
                    raise ValueError('R3 student microbatch has no captured rollout expert IDs.')
                forced = self._prepare_forced(context['routes'], kwargs['input_ids'], context['labels'])
        for layer_index, router in self.routers:
            frame = RoutingReplayFrame(
                None if forced is None else forced[:, layer_index].contiguous(),
                check_recompute=self.check_recompute)
            router.routing_replay = frame
            self.frames.append(frame)
        return positional, kwargs


def install_student_router_replay(model, mode: str):
    if hasattr(model, '_mopd_router_replay_session'):
        raise RuntimeError('Student routing replay was already installed on this model.')
    session = StudentRouterReplay(model, mode)
    model._mopd_router_replay_session = session
    return session
