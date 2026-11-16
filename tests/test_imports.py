# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
"""The published package imports and declares the dependencies it actually uses."""
import importlib
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

MODULES = [
    'unimopd',
    'unimopd.data',
    'unimopd.data.allocate',
    'unimopd.merge',
    'unimopd.merge.calibrate',
    'unimopd.merge.objectives',
    'unimopd.merge.solver',
    'unimopd.merge.streaming',
]


@pytest.mark.parametrize('name', MODULES)
def test_public_modules_import(name):
    assert importlib.import_module(name) is not None


def test_declared_dependencies_are_installed():
    metadata = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    distributions = {
        'numpy': 'numpy',
        'omegaconf': 'omegaconf',
        'PyYAML': 'yaml',
        'safetensors': 'safetensors',
        'torch': 'torch',
        'transformers': 'transformers',
    }
    declared = {
        requirement.split('>=')[0].split('<')[0].split('==')[0].strip()
        for requirement in metadata['project']['dependencies']
    }
    assert declared == set(distributions), declared
    for name in sorted(declared):
        assert importlib.import_module(distributions[name]) is not None


def test_requirements_pin_every_reference_version():
    lines = [
        line.strip() for line in (ROOT / 'requirements.txt').read_text().splitlines()
        if line.strip() and not line.startswith('#')
    ]
    assert lines, 'requirements.txt declares no packages'
    unpinned = [line for line in lines if '==' not in line]
    assert not unpinned, f'reference matrix must pin exact versions: {unpinned}'
