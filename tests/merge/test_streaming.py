# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from unimopd.merge.streaming import (
    SAFETENSORS_INDEX,
    load_recipe,
    merge,
    parameter_seed,
)


def checkpoint(path, shards):
    path.mkdir()
    weight_map = {}
    total_size = 0
    for filename, tensors in shards.items():
        save_file(tensors, path / filename)
        for key, tensor in tensors.items():
            weight_map[key] = filename
            total_size += tensor.numel() * tensor.element_size()
    (path / SAFETENSORS_INDEX).write_text(json.dumps({
        'metadata': {'total_size': total_size},
        'weight_map': weight_map,
    }))
    (path / 'config.json').write_text(json.dumps({
        'torch_dtype': 'float32',
        'text_config': {'dtype': 'float32'},
    }))
    (path / 'tokenizer.json').write_text('{}')


def read_tensor(path, key):
    index = json.loads((path / SAFETENSORS_INDEX).read_text())
    with safe_open(path / index['weight_map'][key], framework='pt', device='cpu') as handle:
        return handle.get_tensor(key)


def test_average_and_task_vector(tmp_path):
    base, first, second = (tmp_path / name for name in ('base', 'first', 'second'))
    shards = {
        'model-00001-of-00002.safetensors': {
            'float': torch.tensor([1.0, 2.0]),
        },
        'model-00002-of-00002.safetensors': {
            'buffer': torch.tensor([7], dtype=torch.int64),
        },
    }
    checkpoint(base, shards)
    checkpoint(first, {
        **shards,
        'model-00001-of-00002.safetensors': {'float': torch.tensor([3.0, 4.0])},
    })
    checkpoint(second, {
        **shards,
        'model-00001-of-00002.safetensors': {'float': torch.tensor([5.0, 8.0])},
    })

    average = merge('average', [first, second], tmp_path / 'average', workers=1)
    assert torch.equal(read_tensor(average, 'float'), torch.tensor([4.0, 6.0]))
    assert torch.equal(read_tensor(average, 'buffer'), torch.tensor([7]))

    task = merge(
        'task_vector',
        [first, second],
        tmp_path / 'task',
        base=base,
        weights=[0.25, 0.75],
        workers=1,
    )
    expected = torch.tensor([1.0, 2.0]) + 0.25 * torch.tensor([2.0, 2.0]) + 0.75 * torch.tensor([4.0, 6.0])
    assert torch.equal(read_tensor(task, 'float'), expected)
    assert json.loads((task / 'config.json').read_text())['torch_dtype'] == 'float32'


def test_dare_is_stable_across_worker_counts(tmp_path):
    base = tmp_path / 'base'
    teachers = [tmp_path / f'teacher-{index}' for index in range(2)]
    checkpoint(base, {
        'model-00001-of-00002.safetensors': {'a': torch.arange(64, dtype=torch.float32)},
        'model-00002-of-00002.safetensors': {'b': torch.arange(64, dtype=torch.float32)},
    })
    for index, teacher in enumerate(teachers, start=1):
        checkpoint(teacher, {
            'model-00001-of-00002.safetensors': {
                'a': torch.arange(64, dtype=torch.float32) + index,
            },
            'model-00002-of-00002.safetensors': {
                'b': torch.arange(64, dtype=torch.float32) - index,
            },
        })

    one = merge(
        'dare', teachers, tmp_path / 'one', base=base, weights=[0.5, 0.5],
        drop_rate=0.7, seed=42, workers=1)
    two = merge(
        'dare', teachers, tmp_path / 'two', base=base, weights=[0.5, 0.5],
        drop_rate=0.7, seed=42, workers=2)
    assert torch.equal(read_tensor(one, 'a'), read_tensor(two, 'a'))
    assert torch.equal(read_tensor(one, 'b'), read_tensor(two, 'b'))
    assert parameter_seed(42, 'a', 0) != parameter_seed(42, 'a', 1)


def test_recipe_expands_environment(tmp_path, monkeypatch):
    recipe = tmp_path / 'recipe.yaml'
    recipe.write_text(
        'method: average\ninputs: ["${FIRST}", "${SECOND}"]\noutput_dir: "${OUT}"\n')
    monkeypatch.setenv('FIRST', 'first')
    monkeypatch.setenv('SECOND', 'second')
    monkeypatch.setenv('OUT', 'merged')
    assert load_recipe(recipe) == {
        'method': 'average',
        'inputs': ['first', 'second'],
        'output_dir': 'merged',
    }


def test_rejects_mismatched_parameter_sets(tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    checkpoint(first, {'model-00001-of-00001.safetensors': {'a': torch.ones(1)}})
    checkpoint(second, {'model-00001-of-00001.safetensors': {'b': torch.ones(1)}})
    with pytest.raises(ValueError, match='parameter set differs'):
        merge('average', [first, second], tmp_path / 'out', workers=1)
