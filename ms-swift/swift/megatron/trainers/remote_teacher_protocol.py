# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, MOPD contributors.
"""Input-bound BF16 teacher hidden transport for synchronous full-vocab MOPD.

Only post-final-norm response prediction rows cross the service boundary.
Each row remains bound to the exact teacher input, supervised token positions,
teacher checkpoint and native vocabulary-head shards. No student routing data,
pickled Python objects, top-k approximation or hidden-state regression.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import struct
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors


PROTOCOL_VERSION = 1
_HEADER = struct.Struct('<Q')
MAX_METADATA_BYTES = 16 * 1024**2
MODEL_TENSOR_KEYS = (
    'input_ids', 'labels', 'position_ids', 'text_position_ids', 'attention_mask',
    'attention_mask_2d', 'pixel_values', 'image_grid_thw',
    'pixel_values_videos', 'video_grid_thw', 'mm_token_type_ids',
)
MEDIA_KEYS = ('pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw', 'mm_token_type_ids')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps([str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _digest(value: str) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in '0123456789abcdef' for char in value))


@dataclass(frozen=True)
class TeacherIdentity:
    key: str
    checkpoint_hash: str
    tokenizer_hash: str
    norm_hash: str
    head_shard_hashes: Tuple[str, ...]
    hidden_size: int
    vocab_size: int
    padded_vocab_size: int
    dtype: str = 'bfloat16'
    attribution: str = 'post_final_norm'

    def __post_init__(self):
        require(isinstance(self.key, str) and bool(self.key), 'Teacher key is required')
        require(all(_digest(value) for value in
                    (self.checkpoint_hash, self.tokenizer_hash, self.norm_hash, *self.head_shard_hashes)),
                'Teacher identity needs checkpoint/tokenizer/norm/head fingerprints')
        require(bool(self.head_shard_hashes), 'Teacher head shards are required')
        require(all(type(value) is int and value > 0 for value in
                    (self.hidden_size, self.vocab_size, self.padded_vocab_size)), 'Invalid teacher dimensions')
        require(self.padded_vocab_size >= self.vocab_size
                and self.padded_vocab_size % len(self.head_shard_hashes) == 0, 'Invalid vocabulary partition')
        require(self.dtype == 'bfloat16' and self.attribution == 'post_final_norm',
                'Remote teacher hidden must retain native BF16 post-final-norm semantics')

    @property
    def fingerprint(self) -> str:
        return canonical_hash(asdict(self))

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, 'head_shard_hashes': tuple(value['head_shard_hashes'])})


@dataclass(frozen=True)
class SampleBinding:
    # Per-occurrence identity, not only a dataset ID which may recur in another window.
    sample_id: str
    teacher_key: str
    input_hash: str
    media_hash: str
    loss_positions: Tuple[int, ...]
    loss_token_ids: Tuple[int, ...]
    input_length: int

    def __post_init__(self):
        require(isinstance(self.sample_id, str) and bool(self.sample_id), 'Sample identity is required')
        require(isinstance(self.teacher_key, str) and bool(self.teacher_key), 'Sample teacher key is required')
        require(_digest(self.input_hash) and _digest(self.media_hash), 'Invalid encoded-input fingerprints')
        require(type(self.input_length) is int and self.input_length > 0, 'Invalid input length')
        require(len(self.loss_positions) == len(self.loss_token_ids), 'Response IDs/positions differ in length')
        require(all(type(pos) is int and 0 < pos < self.input_length for pos in self.loss_positions)
                and list(self.loss_positions) == sorted(set(self.loss_positions)),
                'Supervised positions must be ordered causal targets inside the teacher input')
        require(all(type(token) is int and token >= 0 for token in self.loss_token_ids),
                'Invalid supervised token ID')

    @property
    def capture_positions(self) -> Tuple[int, ...]:
        return tuple(position - 1 for position in self.loss_positions)

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, 'loss_positions': tuple(value['loss_positions']),
                      'loss_token_ids': tuple(value['loss_token_ids'])})


def bind_encoded_sample(sample_id: str, teacher_key: str, encoded: Mapping[str, Any]) -> SampleBinding:
    """Bind the unshifted single-example encoding before Megatron rolls labels."""
    ids, labels = encoded['input_ids'], encoded['labels']
    require(isinstance(ids, torch.Tensor) and isinstance(labels, torch.Tensor),
            'Teacher binding needs encoded input and label tensors')
    require(ids.dtype == labels.dtype == torch.int64 and ids.ndim == 2
            and ids.shape[0] == 1 and ids.shape == labels.shape,
            'Teacher binding currently requires one int64-encoded example')
    ids, labels = ids.detach().cpu()[0], labels.detach().cpu()[0]
    positions = labels.ne(-100).nonzero(as_tuple=True)[0]
    require(torch.equal(ids[positions], labels[positions]), 'Supervised IDs differ from the encoded input')
    fingerprints = {}
    for key in MODEL_TENSOR_KEYS:
        value = encoded.get(key)
        if value is not None:
            require(isinstance(value, torch.Tensor), f'Model input {key} must be a tensor')
            fingerprints[key] = tensor_hash(value)
    # Packed-sequence cardinality affects attention boundaries, not just shape.
    samples = encoded.get('num_samples', 1)
    require(type(samples) is int and samples == 1, 'Bind each example before teacher-side packing')
    return SampleBinding(
        sample_id, teacher_key, canonical_hash({'tensors': fingerprints, 'num_samples': samples}),
        canonical_hash({key: fingerprints[key] for key in MEDIA_KEYS if key in fingerprints}),
        tuple(positions.tolist()), tuple(labels[positions].tolist()), int(ids.numel()))


@dataclass(frozen=True)
class WindowIdentity:
    run_id: str
    window_id: str
    policy_step: int
    samples: int
    chunks: int

    def __post_init__(self):
        require(isinstance(self.run_id, str) and bool(self.run_id)
                and isinstance(self.window_id, str) and bool(self.window_id), 'Window identity is required')
        require(type(self.policy_step) is int and self.policy_step >= 0, 'Invalid rollout policy step')
        require(type(self.samples) is int and self.samples > 0
                and type(self.chunks) is int and 0 < self.chunks <= self.samples, 'Invalid window cardinality')


@dataclass
class HiddenResult:
    binding: SampleBinding
    teacher_fingerprint: str
    hidden: torch.Tensor  # CPU [response rows, teacher hidden size], native BF16


@dataclass
class ScoreRequest:
    window: WindowIdentity
    chunk_index: int
    bindings: List[SampleBinding]
    records: List[Dict[str, Any]]

    def __post_init__(self):
        require(type(self.chunk_index) is int and 0 <= self.chunk_index < self.window.chunks,
                'Invalid score chunk index')
        require(0 < len(self.bindings) == len(self.records) <= self.window.samples,
                'Teacher request records/bindings differ')
        require(len({binding.sample_id for binding in self.bindings}) == len(self.bindings),
                'Duplicate teacher request occurrence')
        for record in self.records:
            require(isinstance(record, dict) and isinstance(record.get('messages'), list),
                    'Teacher requests need canonical template messages')
            require(not {'routed_experts', '_mopd_router_replay', '_mopd_router_replay_wire',
                         '_mopd_routing', 'rollout_infos'} & set(record),
                    'Student routing/replay state must not enter the teacher request')

    def to_dict(self):
        return {'protocol_version': PROTOCOL_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, value):
        require(type(value['protocol_version']) is int and value['protocol_version'] == PROTOCOL_VERSION,
                'Teacher request protocol differs')
        require(set(value) == {'protocol_version', 'window', 'chunk_index', 'bindings', 'records'},
                'Unexpected teacher request fields')
        return cls(WindowIdentity(**value['window']), value['chunk_index'],
                   [SampleBinding.from_dict(binding) for binding in value['bindings']], value['records'])


def encode_hidden_response(
    window: WindowIdentity,
    chunk_index: int,
    results: Sequence[HiddenResult],
    *,
    session_id: str,
    max_bytes: int,
    metrics: Mapping[str, float] = None,
) -> bytes:
    require(type(chunk_index) is int and 0 <= chunk_index < window.chunks and bool(results),
            'Invalid teacher response chunk')
    require(len({result.binding.sample_id for result in results}) == len(results),
            'Duplicate response sample occurrence')
    require(isinstance(session_id, str) and bool(session_id), 'Teacher server session identity is required')
    tensors, samples = {}, []
    for index, result in enumerate(results):
        hidden = result.hidden
        require(hidden.device.type == 'cpu' and hidden.dtype == torch.bfloat16
                and not hidden.requires_grad and hidden.ndim == 2
                and hidden.shape[0] == len(result.binding.loss_positions),
                'Teacher response must contain detached CPU BF16 prediction rows only')
        require(torch.isfinite(hidden).all().item(), 'Teacher response contains non-finite hidden values')
        require(_digest(result.teacher_fingerprint), 'Missing actual teacher fingerprint')
        tensors[str(index)] = hidden.contiguous()
        samples.append({'binding': asdict(result.binding), 'teacher_fingerprint': result.teacher_fingerprint})
    metadata = json.dumps({
        'protocol_version': PROTOCOL_VERSION, 'window': asdict(window), 'chunk_index': chunk_index,
        'session_id': session_id, 'samples': samples, 'metrics': dict(metrics or {}),
    }, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
    require(len(metadata) <= MAX_METADATA_BYTES, 'Teacher metadata exceeds its limit')
    expected_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    require(_HEADER.size + len(metadata) + expected_bytes <= max_bytes, 'Teacher hidden exceeds payload budget')
    tensor_bytes = save_tensors(tensors)
    require(_HEADER.size + len(metadata) + len(tensor_bytes) <= max_bytes, 'Teacher payload exceeds byte budget')
    return _HEADER.pack(len(metadata)) + metadata + tensor_bytes


def decode_hidden_response(
    payload: bytes,
    window: WindowIdentity,
    chunk_index: int,
    expected: Sequence[SampleBinding],
    teachers: Mapping[str, TeacherIdentity],
    *,
    session_id: str,
    max_bytes: int,
) -> Tuple[List[HiddenResult], Dict[str, float]]:
    require(_HEADER.size < len(payload) <= max_bytes, 'Invalid teacher payload size')
    metadata_length, = _HEADER.unpack_from(payload)
    require(0 < metadata_length <= MAX_METADATA_BYTES
            and _HEADER.size + metadata_length < len(payload), 'Invalid teacher metadata length')
    offset = _HEADER.size + metadata_length
    metadata = json.loads(payload[_HEADER.size:offset])
    require(type(metadata['protocol_version']) is int and metadata['protocol_version'] == PROTOCOL_VERSION,
            'Teacher protocol version differs')
    require(WindowIdentity(**metadata['window']) == window
            and type(metadata['chunk_index']) is int and metadata['chunk_index'] == chunk_index,
            'Teacher response belongs to another window/policy/chunk')
    require(metadata['session_id'] == session_id, 'Teacher server changed during the window')
    require(len(metadata['samples']) == len(expected), 'Teacher response sample count changed')
    require(len({binding.sample_id for binding in expected}) == len(expected),
            'Expected sample occurrences must be unique')
    tensors = load_tensors(payload[offset:])
    require(set(tensors) == {str(index) for index in range(len(expected))}, 'Unexpected teacher tensor fields')
    results = []
    for index, (record, binding) in enumerate(zip(metadata['samples'], expected)):
        actual = SampleBinding.from_dict(record['binding'])
        require(actual == binding, f'Teacher input/positions/media/order changed: {binding.sample_id}')
        identity = teachers[binding.teacher_key]
        require(record['teacher_fingerprint'] == identity.fingerprint,
                f'Teacher checkpoint/head/tokenizer changed: {binding.sample_id}')
        hidden = tensors[str(index)]
        require(hidden.dtype == torch.bfloat16 and tuple(hidden.shape) ==
                (len(binding.loss_positions), identity.hidden_size), 'Teacher hidden dtype or dimensions changed')
        require(torch.isfinite(hidden).all().item(), 'Non-finite teacher hidden response')
        results.append(HiddenResult(binding, identity.fingerprint, hidden))
    metrics = metadata['metrics']
    require(isinstance(metrics, dict) and all(
        isinstance(key, str) and type(value) in (int, float) and math.isfinite(value)
        for key, value in metrics.items()), 'Invalid teacher timing/counter packet')
    return results, metrics
