# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, MOPD contributors.
"""Load only frozen native BF16 teacher heads, not their transformer backbones."""
import hashlib
import json
from pathlib import Path
from typing import Dict

from safetensors import safe_open
import torch

from .remote_teacher_protocol import TeacherIdentity, require, tensor_hash


def checkpoint_hash(path: Path) -> str:
    value = hashlib.sha256()
    for name in ('model.safetensors.index.json', 'config.json'):
        value.update((path / name).read_bytes())
    return value.hexdigest()


def tokenizer_hash(tokenizer) -> str:
    # Match the existing GKD token-to-ID contract, independent of JSON whitespace.
    from .remote_teacher_protocol import canonical_hash
    vocab = tokenizer.get_vocab()
    specials = {name: getattr(tokenizer, name, None)
                for name in ('pad_token_id', 'eos_token_id', 'bos_token_id', 'unk_token_id')}
    return canonical_hash({'vocab': sorted(vocab.items(), key=lambda item: (item[1], item[0])),
                           'specials': specials})


def _weight_keys(mapping: Dict[str, str]):
    heads = [name for name in mapping if name == 'lm_head.weight' or name.endswith('.lm_head.weight')]
    norms = [name for name in mapping if name.endswith('.norm.weight')
             and '.layers.' not in name and 'visual.' not in name]
    require(len(heads) == len(norms) == 1, 'Remote Qwen teacher needs one explicit untied LM head/final norm')
    return heads[0], norms[0]


def load_teacher_head_shard(path, identity: TeacherIdentity, tp_rank: int) -> torch.Tensor:
    """Apply the same zero vocabulary padding/TP row slicing as GPTBridge.

    Hashes are compared against the ACTUAL native teacher worker's loaded
    tensors, after native BF16 conversion, not merely against file metadata.
    """
    path = Path(path)
    require(type(tp_rank) is int and 0 <= tp_rank < len(identity.head_shard_hashes), 'Invalid head TP rank')
    require(checkpoint_hash(path) == identity.checkpoint_hash, 'Teacher checkpoint metadata changed')
    mapping = json.loads((path / 'model.safetensors.index.json').read_text())['weight_map']
    head_key, norm_key = _weight_keys(mapping)
    for key in (head_key, norm_key):
        require(Path(mapping[key]).name == mapping[key], 'Teacher weight shard escapes checkpoint directory')
    per_rank = identity.padded_vocab_size // len(identity.head_shard_hashes)
    start = tp_rank * per_rank
    end = min(start + per_rank, identity.vocab_size)
    result = torch.zeros(per_rank, identity.hidden_size, dtype=torch.bfloat16)
    with safe_open(str(path / mapping[head_key]), framework='pt', device='cpu') as stream:
        source = stream.get_slice(head_key)
        require(source.get_shape() == [identity.vocab_size, identity.hidden_size], 'Teacher head shape differs')
        if start < end:
            result[:end - start].copy_(source[start:end].to(torch.bfloat16))
    with safe_open(str(path / mapping[norm_key]), framework='pt', device='cpu') as stream:
        norm = stream.get_tensor(norm_key).to(torch.bfloat16)
    require(tuple(norm.shape) == (identity.hidden_size,) and tensor_hash(norm) == identity.norm_hash,
            'Teacher final-norm identity differs from the hidden producer')
    require(tensor_hash(result) == identity.head_shard_hashes[tp_rank],
            'Local teacher head differs from the native hidden producer')
    return result


def build_native_teacher_head(identity, weight, native_config, tp_group):
    """A dedicated native operator with frozen TEACHER weights and no backbone."""
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    require(not weight.requires_grad and weight.dtype == torch.bfloat16, 'Teacher replay head must stay frozen BF16')
    head = ColumnParallelLinear(
        identity.hidden_size, identity.padded_vocab_size, config=native_config,
        init_method=native_config.init_method, bias=False, skip_bias_add=False,
        gather_output=False, skip_weight_param_allocation=True, tp_group=tp_group)
    head.weight = torch.nn.Parameter(weight, requires_grad=False)
    head.eval()
    return head
