# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0

"""Immutable, rank-local cache of HF -> Megatron *initial* model state.

This is not a training checkpoint: no optimizer, iteration, RNG or data position
is saved/restored. The ordinary model constructor still runs. In particular a
cache hit never replaces Parameter objects owned by Megatron/DDP.
"""
from collections import OrderedDict
from functools import lru_cache
import fcntl
import hashlib
import importlib.metadata
import io
import json
import logging
import os
from pathlib import Path
import tempfile
import time

import torch


logger = logging.getLogger(__name__)
SCHEMA = 1


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def source_identity(source):
    """Match the immutable local HF source, not a dataset/run directory."""
    root = Path(source).resolve(strict=True)
    weights = sorted(root.glob('*.safetensors'))
    if not weights or not (root / 'config.json').is_file():
        raise ValueError(f'Megatron weight cache requires a complete local HF model: {root}')
    files = []
    for path in sorted(set(weights) | {root / 'config.json'} |
                       set(root.glob('*.safetensors.index.json'))):
        stat = path.stat()
        with path.open('rb') as stream:
            first = stream.read(64 * 1024)
            stream.seek(max(0, stat.st_size - 64 * 1024))
            fingerprint = hashlib.sha256(first + stream.read(64 * 1024)).hexdigest()
        files.append([path.name, stat.st_size, stat.st_mtime_ns, fingerprint])
    return {'source': str(root), 'files': files}


@lru_cache(maxsize=1)
def code_identity():
    """Content identity survives a new frozen-source directory, not code edits."""
    swift = Path(__file__).resolve().parents[1]
    import megatron.core
    core = Path(megatron.core.__file__).resolve().parent
    digests = {}
    for label, root in (
        ('swift_models', swift / 'model'), ('swift_model_utils', swift / 'utils'),
        ('swift_tuners', swift / 'tuners'), ('swift_hf_models', swift.parent / 'llm/model'),
        ('megatron_core', core),
    ):
        digests[label] = [
            [str(path.relative_to(root)), _digest(path)]
            for path in sorted(root.rglob('*.py')) if '__pycache__' not in path.parts
        ]
    digests['megatron_model_args'] = _digest(swift / 'argument/megatron_args.py')
    versions = {}
    for name in ('torch', 'transformers', 'transformer-engine', 'safetensors'):
        versions[name] = importlib.metadata.version(name)
    return {'code': digests, 'packages': versions}


def _pack(value):
    # Transformer Engine extra state may be BytesIO; encode it as ordinary
    # bytes so reading the cache never needs unrestricted pickle globals.
    if isinstance(value, torch.Tensor):
        return ('tensor', value.detach().to(device='cpu', copy=True))
    if isinstance(value, io.BytesIO):
        return ('bytesio', value.getvalue())
    if isinstance(value, dict):
        return ('ordered' if isinstance(value, OrderedDict) else 'dict',
                [(key, _pack(item)) for key, item in value.items()])
    if isinstance(value, tuple):
        return ('tuple', [_pack(item) for item in value])
    if isinstance(value, list):
        return ('list', [_pack(item) for item in value])
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return ('scalar', value)
    raise TypeError(f'Unsupported native model extra state in weight cache: {type(value).__name__}')


def _unpack(value):
    kind, payload = value
    if kind in ('scalar', 'tensor'):
        return payload
    if kind == 'bytesio':
        return io.BytesIO(payload)
    if kind in ('dict', 'ordered'):
        constructor = OrderedDict if kind == 'ordered' else dict
        return constructor((key, _unpack(item)) for key, item in payload)
    if kind in ('tuple', 'list'):
        result = [_unpack(item) for item in payload]
        return tuple(result) if kind == 'tuple' else result
    raise ValueError(f'Unknown cached model-state kind: {kind}')


def _tensor_layout(state):
    result = []
    for name, value in state.items():
        if isinstance(value, torch.Tensor):
            result.append([name, list(value.shape), str(value.dtype)])
        else:
            result.append([name, type(value).__name__])
    return result


def cache_identity(bridge, model, source):
    args = bridge.args
    return {
        'schema': SCHEMA,
        'source': source_identity(source),
        'runtime': code_identity(),
        'layout': {key: getattr(bridge, key) for key in
                   ('tp_size', 'pp_size', 'ep_size', 'etp_size',
                    'tp_rank', 'pp_rank', 'ep_rank', 'etp_rank')},
        'world_size': torch.distributed.get_world_size(),
        'model': {key: str(getattr(args, key, None)) for key in
                  ('hf_model_type', 'torch_dtype', 'params_dtype', 'padded_vocab_size',
                   'sequence_parallel', 'moe_grouped_gemm', 'moe_use_legacy_grouped_gemm',
                   'untie_embeddings_and_output_weights', 'fp8', 'fp8_param_gather')},
        'tensors': _tensor_layout(model.state_dict()),
    }


def _available_memory():
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key == 'MemAvailable':
            memory[key] = int(value.split()[0]) * 1024
    available = memory['MemAvailable']
    for limit, usage in (
        ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
        ('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes'),
    ):
        if Path(limit).is_file() and Path(usage).is_file():
            value = Path(limit).read_text().strip()
            if value != 'max':
                available = min(available, int(value) - int(Path(usage).read_text()))
            break
    return available


def _check_capacity(root, state, reserve_bytes):
    needed = sum(value.numel() * value.element_size() for value in state.values()
                 if isinstance(value, torch.Tensor))
    disk = os.statvfs(root)
    # Account for the CPU snapshot as well as the tmpfs/file payload. Serialize
    # writers per node so concurrent ranks cannot all spend the same headroom.
    if disk.f_bavail * disk.f_frsize < needed * 1.1 + reserve_bytes:
        raise RuntimeError(f'Insufficient weight-cache filesystem headroom at {root}')
    if _available_memory() < needed * 2.1 + reserve_bytes:
        raise RuntimeError(f'Insufficient host/cgroup headroom for converted weights at {root}')


def load_or_create(model, source, identity, root, load_hf, *, reserve_bytes=128 * 1024**3):
    """Load all state strictly, or populate an atomic cache after native import.

    Hit/miss decisions are rank-local: neither branch adds collectives. A
    partially populated node cache therefore cannot change collective ordering.
    Corrupt entries fail loudly instead of training from a partial model.
    """
    started = time.monotonic()
    root = Path(root)
    if not root.is_absolute() or root.is_symlink():
        raise ValueError('Use an absolute, node-local physical weight-cache directory')
    root.mkdir(parents=True, exist_ok=True)
    if root.resolve() != root:
        raise ValueError('Weight-cache path must not traverse symlinks')
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    target = root / key
    manifest_path = target / 'manifest.json'

    def restore():
        manifest = json.loads(manifest_path.read_text())
        payload_path = target / 'model.pt'
        if (manifest['identity'] != identity or payload_path.is_symlink()
                or _digest(payload_path) != manifest['payload_sha256']):
            raise ValueError(f'Converted model cache failed integrity/identity check: {target}')
        payload = torch.load(payload_path, map_location='cpu', weights_only=True, mmap=True)
        state = _unpack(payload['state'])
        state._metadata = _unpack(payload['metadata'])
        if _tensor_layout(state) != _tensor_layout(model.state_dict()):
            raise ValueError(f'Converted model cache tensor layout mismatch: {target}')
        model.load_state_dict(state, strict=True)
        logger.info('MOPD converted-weight cache hit: source=%s cache=%s seconds=%.3f',
                    source, target, time.monotonic() - started)
        return {'hit': True, 'path': str(target), 'seconds': time.monotonic() - started}

    if target.exists():
        return restore()
    load_hf()
    if source_identity(source) != identity['source']:
        raise ValueError('HF source changed during Megatron conversion; refusing to cache')
    state = model.state_dict()
    with (root / '.write.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            return restore()
        _check_capacity(root, state, reserve_bytes)
        stage = Path(tempfile.mkdtemp(prefix=f'.partial-{key[:12]}-', dir=root))
        payload_path = stage / 'model.pt'
        payload = {'state': _pack(state), 'metadata': _pack(getattr(state, '_metadata', {}))}
        torch.save(payload, payload_path)
        del payload
        manifest = {'identity': identity, 'payload_sha256': _digest(payload_path)}
        (stage / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True))
        # Failed writes remain isolated .partial directories, never cache hits.
        stage.rename(target)
    logger.info('MOPD converted-weight cache created: source=%s cache=%s seconds=%.3f',
                source, target, time.monotonic() - started)
    return {'hit': False, 'path': str(target), 'seconds': time.monotonic() - started}


def load_cached_weights(bridge, model, source, load_hf):
    args = bridge.args
    root = getattr(args, 'mopd_model_cache_dir', None)
    if not root:
        return load_hf()
    if (getattr(args, 'train_type', None) != 'full'
            or getattr(args, 'rlhf_type', None) not in (None, 'gkd')
            or getattr(args, 'fp8', None) or getattr(args, 'fp8_param_gather', False)
            or getattr(args, 'mtp_num_layers', None)
            or bridge._is_peft_format):
        raise ValueError('Converted-weight cache currently supports full BF16/FP32 SFT/GKD without FP8/MTP/adapters')
    identity = cache_identity(bridge, model, source)
    return load_or_create(
        model, source, identity, root, load_hf,
        reserve_bytes=int(getattr(args, 'mopd_model_cache_reserve_gb', 128)) * 1024**3)
