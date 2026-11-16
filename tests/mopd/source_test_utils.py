# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
"""Load selected production definitions without importing CUDA or Megatron."""
import ast
import copy
import importlib.util
import logging
import math
from pathlib import Path
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]


def load_module(relative_path, name):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_symbols(relative_path, names, *, parent=None, namespace=None):
    """Execute only the named functions/classes from a production source file."""
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(), str(path))
    if parent:
        tree = next(node for node in ast.walk(tree) if getattr(node, 'name', None) == parent)
    nodes = [copy.deepcopy(node) for node in tree.body if getattr(node, 'name', None) in names]
    assert {node.name for node in nodes} == set(names), (path, names)
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = [
                decorator for decorator in node.decorator_list
                if not isinstance(decorator, ast.Name) or decorator.id not in {'staticmethod', 'classmethod'}
            ]
    env = {
        'torch': torch,
        'F': F,
        'np': np,
        'math': math,
        'logger': logging.getLogger('unimopd-source-test'),
    }
    env.update(namespace or {})
    module = ast.Module(
        body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), env)
    return env


def gkd_symbols(names, namespace=None):
    return load_symbols(
        'ms-swift/swift/megatron/trainers/gkd_trainer.py',
        names,
        parent='MegatronGKDTrainer',
        namespace=namespace,
    )


def load_vocab_parallel(mpu):
    """Load the vocabulary-parallel leaf module against a CPU-only MPU stub."""
    megatron = types.ModuleType('megatron')
    core = types.ModuleType('megatron.core')
    core.mpu = mpu
    megatron.core = core
    absent = object()
    saved = {
        'megatron': sys.modules.get('megatron', absent),
        'megatron.core': sys.modules.get('megatron.core', absent),
    }
    sys.modules.update({'megatron': megatron, 'megatron.core': core})
    try:
        return load_module(
            'ms-swift/swift/megatron/trainers/vocab_parallel_utils.py',
            f'unimopd_test_vocab_parallel_{id(mpu)}',
        )
    finally:
        for name, previous in saved.items():
            if previous is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
