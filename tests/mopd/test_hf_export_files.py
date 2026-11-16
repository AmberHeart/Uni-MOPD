# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract for the HF files written next to exported Megatron weights."""
import json
import os
import shutil

from tests.mopd.source_test_utils import load_symbols


class _Processor:

    def save_pretrained(self, output_dir):
        pass


def _save_checkpoint():
    return load_symbols('ms-swift/swift/llm/utils.py', ['save_checkpoint'],
                        namespace={'os': os, 'shutil': shutil})['save_checkpoint']


def test_megatron_export_copies_source_generation_config(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'output'
    source.mkdir()
    output.mkdir()
    config = {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20}
    (source / 'generation_config.json').write_text(json.dumps(config))

    _save_checkpoint()(None, _Processor(), str(output), model_dirs=[str(source)])

    assert json.loads((output / 'generation_config.json').read_text()) == config


def test_existing_generation_config_is_not_replaced(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'output'
    source.mkdir()
    output.mkdir()
    (source / 'generation_config.json').write_text('{"temperature": 0.7}')
    (output / 'generation_config.json').write_text('{"temperature": 0.2}')

    _save_checkpoint()(None, _Processor(), str(output), model_dirs=[str(source)])

    assert json.loads((output / 'generation_config.json').read_text()) == {'temperature': 0.2}
