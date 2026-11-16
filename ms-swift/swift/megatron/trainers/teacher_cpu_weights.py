# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, MOPD contributors.
"""CPU backing for immutable teachers; never use for student/optimizer state."""
import json
from pathlib import Path

import torch


def converted_cpu_weights(model, cache_result):
    """Borrow parameter storage from an already validated native model cache.

    The tensor payload stays mmap-backed: copying it into another pinned bank
    would duplicate all five teachers beside the converted-weight cache.
    Recheck the immutable payload before exposing its storage to the teacher.
    Do not decode a second, independent version of the native cache format.
    """
    from swift.megatron.model.weight_cache import SCHEMA, _digest, _unpack

    if not isinstance(cache_result, dict) or not cache_result.get('path'):
        raise ValueError('Converted teacher backing requires a completed native-cache load.')
    root = Path(cache_result['path'])
    payload_path = root / 'model.pt'
    manifest = json.loads((root / 'manifest.json').read_text())
    if (manifest['identity']['schema'] != SCHEMA or payload_path.is_symlink()
            or _digest(payload_path) != manifest['payload_sha256']):
        raise ValueError(f'Converted teacher backing failed integrity/schema check: {root}')
    payload = torch.load(payload_path, map_location='cpu', weights_only=True, mmap=True)
    state = _unpack(payload['state'])
    weights = []
    for name, parameter in model.named_parameters():
        value = state.get(name)
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                or value.requires_grad or value.shape != parameter.shape or value.dtype != parameter.dtype):
            raise ValueError(f'Converted teacher parameter does not match its native model: {name}')
        weights.append(value)
    return tuple(weights)


class FrozenTeacherCPUWeights:
    """Keep a teacher's original CPU values while its Parameter objects move.

    Parameters must remain frozen and must not be loaded with a different
    checkpoint during this object's lifetime. A teacher reload constructs a new
    cache. Buffers are left untouched, matching the existing teacher offloader.
    Each load creates fresh device storage: detached head tensors retained by
    checkpointed KL backward cannot be overwritten by a later teacher load.
    """

    @torch.no_grad()
    def __init__(self, models, *, pin_memory=True, cpu_weights=None):
        self.models = tuple(models)
        self.parameters = tuple(param for model in self.models for param in model.parameters())
        if not self.parameters:
            raise ValueError('Frozen teacher cache requires materialized parameters.')
        self._identities = tuple(id(param) for param in self.parameters)
        self._specs = tuple((param.shape, param.dtype) for param in self.parameters)
        self._check_frozen()
        if cpu_weights is None:
            backing = []
            for param in self.parameters:
                cpu = torch.empty_like(param, device='cpu', pin_memory=pin_memory, requires_grad=False)
                # Initial GPU -> pinned CPU copy must finish before the source
                # can be released. Later offloads only rebind Parameter.data.
                cpu.copy_(param, non_blocking=False)
                backing.append(cpu)
            self.backing_kind = 'pinned_copy' if pin_memory else 'cpu_copy'
        else:
            if pin_memory:
                raise ValueError('Borrowed CPU backing requires pin_memory=False; pinning would copy it.')
            backing = tuple(cpu_weights)
            if len(backing) != len(self.parameters) or any(
                    not isinstance(cpu, torch.Tensor) or cpu.device.type != 'cpu' or cpu.requires_grad
                    or cpu.shape != param.shape or cpu.dtype != param.dtype
                    for param, cpu in zip(self.parameters, backing)):
                raise ValueError('Borrowed CPU tensors must match every frozen teacher parameter.')
            self.backing_kind = 'borrowed_cpu'
        self.cpu_weights = tuple(backing)
        self.nbytes = sum(tensor.numel() * tensor.element_size() for tensor in self.cpu_weights)
        self.cpu_copy_bytes = self.nbytes if cpu_weights is None else 0
        self.resident = True
        self.device = self.parameters[0].device

    def _check_frozen(self):
        current = tuple(param for model in self.models for param in model.parameters())
        if tuple(id(param) for param in current) != self._identities:
            raise ValueError('Teacher parameters were replaced; rebuild their CPU cache.')
        if any(param.requires_grad or param.grad is not None for param in current):
            raise ValueError('CPU backing is only valid for frozen teacher parameters without gradients.')
        if tuple((param.shape, param.dtype) for param in current) != self._specs:
            raise ValueError('Teacher parameter shape/dtype changed; rebuild their CPU cache.')

    @torch.no_grad()
    def offload(self):
        self._check_frozen()
        if not self.resident:
            return False
        for param, cpu in zip(self.parameters, self.cpu_weights):
            param.data = cpu
        self.resident = False
        return True

    @torch.no_grad()
    def load(self, device):
        self._check_frozen()
        device = torch.device('cuda', device) if isinstance(device, int) else torch.device(device)
        if self.resident:
            if device != self.device:
                raise ValueError('Offload the teacher before selecting another device.')
            return False
        # Allocate all tensors before rebinding any Parameter. An allocation
        # failure must not leave a mixture of old/new teacher device storage.
        values = tuple(cpu.to(device=device, non_blocking=True, copy=True) for cpu in self.cpu_weights)
        for param, value in zip(self.parameters, values):
            param.data = value
        self.device = device
        self.resident = True
        return True
