# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0

"""Shard-streaming merges for Hugging Face safetensors checkpoints.

The merge follows the output layout of the first input for ``average`` and of
the base checkpoint for ``task_vector`` and ``dare``. Floating-point tensors
are accumulated in FP32 and converted back to the layout tensor's dtype.
Integer buffers are copied from the layout checkpoint.

DARE masks use one deterministic seed per ``(parameter key, input index)``.
This keeps a merge identical across worker counts and shard scheduling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SAFETENSORS_INDEX = 'model.safetensors.index.json'
SINGLE_SAFETENSORS = 'model.safetensors'
MERGE_FORMAT_VERSION = 1

_SIDECAR_NAMES = {
    'config.json',
    'generation_config.json',
    'preprocessor_config.json',
    'processor_config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'special_tokens_map.json',
    'added_tokens.json',
    'chat_template.json',
    'chat_template.jinja',
    'vocab.json',
    'merges.txt',
    SAFETENSORS_INDEX,
}
_ENV_PATTERN = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}')
def parameter_seed(seed: int, key: str, input_index: int) -> int:
    """Return the historical deterministic DARE seed for one task vector."""
    text = f'{seed}|{key}|{input_index}'.encode()
    return int(hashlib.md5(text).hexdigest()[:8], 16)


def _weight_map(model_dir: Path) -> dict[str, str]:
    index = model_dir / SAFETENSORS_INDEX
    if index.is_file():
        payload = json.loads(index.read_text(encoding='utf-8'))
        weight_map = payload.get('weight_map')
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f'{index} has no nonempty weight_map.')
        return {str(key): str(value) for key, value in weight_map.items()}
    single = model_dir / SINGLE_SAFETENSORS
    if single.is_file():
        with safe_open(single, framework='pt', device='cpu') as handle:
            return {key: SINGLE_SAFETENSORS for key in handle.keys()}
    raise FileNotFoundError(
        f'{model_dir} contains neither {SAFETENSORS_INDEX} nor {SINGLE_SAFETENSORS}.')


def _normalized_weights(weights: Sequence[float] | None, count: int) -> list[float]:
    if count <= 0:
        raise ValueError('At least one input checkpoint is required.')
    values = [1.0 / count] * count if weights is None else [float(value) for value in weights]
    if len(values) != count:
        raise ValueError(f'Expected {count} merge weights, got {len(values)}.')
    if any(not math.isfinite(value) for value in values):
        raise ValueError('Merge weights must be finite.')
    return values


def _normalized_drop_rates(
    drop_rates: Sequence[float] | None,
    drop_rate: float,
    count: int,
) -> list[float]:
    values = [float(drop_rate)] * count if drop_rates is None else [
        float(value) for value in drop_rates
    ]
    if len(values) != count:
        raise ValueError(f'Expected {count} DARE drop rates, got {len(values)}.')
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError('DARE drop rates must be finite values in [0, 1].')
    return values


def _validate_key_sets(
    layout: Mapping[str, str],
    inputs: Sequence[Mapping[str, str]],
    base: Mapping[str, str] | None,
) -> None:
    expected = set(layout)
    for index, weight_map in enumerate(inputs):
        missing = expected.difference(weight_map)
        extra = set(weight_map).difference(expected)
        if missing or extra:
            raise ValueError(
                f'Input {index} parameter set differs from the layout checkpoint: '
                f'missing={len(missing)}, extra={len(extra)}.')
    if base is not None and set(base) != expected:
        raise ValueError('Base parameter set differs from the selected layout checkpoint.')


def _merge_shard(
    method: str,
    shard: str,
    keys: Sequence[str],
    inputs: Sequence[str],
    weights: Sequence[float],
    base: str | None,
    drop_rates: Sequence[float],
    seed: int,
    input_weight_maps: Sequence[Mapping[str, str]],
    base_weight_map: Mapping[str, str] | None,
    output_dir: str,
) -> tuple[str, int]:
    merged: dict[str, torch.Tensor] = {}
    with ExitStack() as stack:
        input_handles = []
        for model_dir, weight_map in zip(inputs, input_weight_maps):
            handles = {
                filename: stack.enter_context(
                    safe_open(
                        str(Path(model_dir) / filename),
                        framework='pt',
                        device='cpu',
                    ))
                for filename in {weight_map[key] for key in keys}
            }
            input_handles.append(handles)

        def get_input(index: int, key: str) -> torch.Tensor:
            filename = input_weight_maps[index][key]
            return input_handles[index][filename].get_tensor(key)

        base_handles = {}
        if base is not None and base_weight_map is not None:
            base_handles = {
                filename: stack.enter_context(
                    safe_open(
                        str(Path(base) / filename),
                        framework='pt',
                        device='cpu',
                    ))
                for filename in {base_weight_map[key] for key in keys}
            }

        def get_base(key: str) -> torch.Tensor:
            assert base_weight_map is not None
            return base_handles[base_weight_map[key]].get_tensor(key)

        for key in keys:
            if method == 'average':
                reference = get_input(0, key)
                if not torch.is_floating_point(reference):
                    merged[key] = reference
                    continue
                accumulator = None
                for index, weight in enumerate(weights):
                    tensor = get_input(index, key)
                    if tensor.shape != reference.shape:
                        raise ValueError(f'Shape mismatch for parameter {key!r}.')
                    term = weight * tensor.float()
                    accumulator = term if accumulator is None else accumulator + term
                assert accumulator is not None
                merged[key] = accumulator.to(reference.dtype)
                continue

            base_tensor = get_base(key)
            if not torch.is_floating_point(base_tensor):
                merged[key] = base_tensor
                continue
            base_fp32 = base_tensor.float()
            accumulator = base_fp32.clone()
            for input_index, weight in enumerate(weights):
                tensor = get_input(input_index, key)
                if tensor.shape != base_tensor.shape:
                    raise ValueError(f'Shape mismatch for parameter {key!r}.')
                delta = tensor.float() - base_fp32
                if float(delta.abs().sum()) == 0:
                    continue
                probability = drop_rates[input_index]
                if method == 'dare' and probability:
                    if probability == 1:
                        continue
                    generator = torch.Generator().manual_seed(
                        parameter_seed(seed, key, input_index))
                    mask = (torch.rand(delta.shape, generator=generator) >= probability).float()
                    delta = delta * mask / (1 - probability)
                accumulator += weight * delta
            merged[key] = accumulator.to(base_tensor.dtype)

    save_file(merged, str(Path(output_dir) / shard))
    return shard, len(keys)


def _copy_runtime_sidecars(source: Path, output: Path) -> None:
    for name in sorted(_SIDECAR_NAMES):
        path = source / name
        if path.is_file():
            shutil.copy2(path, output / name)


def merge(
    method: str,
    inputs: Sequence[str | os.PathLike],
    output_dir: str | os.PathLike,
    *,
    weights: Sequence[float] | None = None,
    base: str | os.PathLike | None = None,
    drop_rate: float = 0,
    drop_rates: Sequence[float] | None = None,
    seed: int = 42,
    workers: int = 4,
) -> Path:
    """Merge checkpoints and return the output directory."""
    if method not in {'average', 'task_vector', 'dare'}:
        raise ValueError(f'Unsupported merge method: {method!r}.')
    input_paths = [Path(path).expanduser().resolve() for path in inputs]
    if method in {'task_vector', 'dare'} and base is None:
        raise ValueError(f'{method} requires a base checkpoint.')
    base_path = None if base is None else Path(base).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)

    normalized_weights = _normalized_weights(weights, len(input_paths))
    normalized_drops = _normalized_drop_rates(drop_rates, drop_rate, len(input_paths))
    if type(workers) is not int or workers <= 0:
        raise ValueError('workers must be a positive integer.')

    input_weight_maps = [_weight_map(path) for path in input_paths]
    base_weight_map = None if base_path is None else _weight_map(base_path)
    layout_source = base_path if method in {'task_vector', 'dare'} else input_paths[0]
    layout_weight_map = (
        base_weight_map if method in {'task_vector', 'dare'} else input_weight_maps[0])
    assert layout_source is not None and layout_weight_map is not None
    _validate_key_sets(layout_weight_map, input_weight_maps, base_weight_map)
    _copy_runtime_sidecars(layout_source, output)

    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in layout_weight_map.items():
        shard_to_keys.setdefault(shard, []).append(key)

    tasks = sorted(shard_to_keys.items())
    arguments = (
        method,
        [str(path) for path in input_paths],
        normalized_weights,
        None if base_path is None else str(base_path),
        normalized_drops,
        int(seed),
        input_weight_maps,
        base_weight_map,
        str(output),
    )

    def collect(result):
        shard, count = result
        print(f'[merge] wrote {shard} ({count} tensors)', flush=True)

    if workers == 1:
        for shard, keys in tasks:
            collect(_merge_shard(method, shard, keys, *arguments[1:]))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            futures = [
                executor.submit(_merge_shard, method, shard, keys, *arguments[1:])
                for shard, keys in tasks
            ]
            for future in as_completed(futures):
                collect(future.result())

    output_index = output / SAFETENSORS_INDEX
    if output_index.is_file():
        payload = json.loads(output_index.read_text(encoding='utf-8'))
        payload['weight_map'] = dict(layout_weight_map)
        output_index.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )

    manifest = {
        'format_version': MERGE_FORMAT_VERSION,
        'method': method,
        'source_count': len(input_paths),
        'weights': normalized_weights,
        'drop_rates': normalized_drops if method == 'dare' else None,
        'seed': int(seed) if method == 'dare' else None,
        'accumulation_dtype': 'float32',
    }
    (output / 'merge_info.json').write_text(
        json.dumps(manifest, indent=2) + '\n',
        encoding='utf-8',
    )
    return output


def _expand_environment(value):
    if isinstance(value, str):
        def replace(match):
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(f'Configuration requires environment variable {name}.')
            return os.environ[name]
        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def load_recipe(path: str | os.PathLike) -> dict:
    """Load a YAML recipe and expand ``${ENV_NAME}`` placeholders."""
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError('PyYAML is required to load merge recipes.') from error
    recipe = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(recipe, dict):
        raise ValueError('A merge recipe must be a YAML mapping.')
    return _expand_environment(recipe)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='YAML merge recipe')
    args = parser.parse_args(argv)
    recipe = load_recipe(args.config)
    allowed = {
        'method', 'inputs', 'output_dir', 'weights', 'base', 'drop_rate',
        'drop_rates', 'seed', 'workers',
    }
    unknown = set(recipe).difference(allowed)
    if unknown:
        raise ValueError(f'Unknown merge recipe fields: {sorted(unknown)}.')
    merge(**recipe)


if __name__ == '__main__':
    main()
