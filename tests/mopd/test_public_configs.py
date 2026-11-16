# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
"""Published recipes and launchers resolve without importing the training stack."""
import ast
import os
import subprocess
import sys
from unittest.mock import patch

from omegaconf import OmegaConf
import pytest

from tests.mopd.source_test_utils import ROOT

CONFIG_DIR = ROOT / 'configs/train'
CONFIGS = {
    'rkl': CONFIG_DIR / 'unimopd_uniform_rkl.yaml',
    'fkl': CONFIG_DIR / 'unimopd_uniform_fkl.yaml',
    'todi': CONFIG_DIR / 'unimopd_uniform_todi.yaml',
}
PUBLIC_ENV = {
    'STUDENT_MODEL': 'models/merge-equal',
    'TEACHER_STEM': 'models/teacher-stem',
    'TEACHER_VQA': 'models/teacher-vqa',
    'TEACHER_DOCUMENT': 'models/teacher-document',
    'TEACHER_GUI': 'models/teacher-gui',
    'TEACHER_EMBODIED': 'models/teacher-embodied',
    'DATA_PATH': 'data/train.jsonl',
    'OUTPUT_DIR': 'outputs/unimopd',
    'ROLLOUT_HOST': 'rollout.example',
    'ROLLOUT_PORT': '8123',
    'ROLLOUT_GROUP_PORT': '20123',
}


def resolve_config(name, tmp_path, *, cwd=None):
    output = tmp_path / f'{name}.yaml'
    subprocess.run(
        [
            sys.executable,
            str(ROOT / 'scripts/resolve_train_config.py'),
            str(CONFIGS[name]),
            str(output),
        ],
        cwd=cwd,
        env={**os.environ, **PUBLIC_ENV},
        check=True,
    )
    return resolved_container(output)


def resolved_container(path):
    with patch.dict(os.environ, PUBLIC_ENV):
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def declared_argument_fields():
    """Collect fields from the Megatron RLHF dataclass inheritance surface."""
    paths = [
        *(ROOT / 'ms-swift/swift/megatron/argument').glob('*.py'),
        *(ROOT / 'ms-swift/swift/llm/argument/base_args').glob('*.py'),
    ]
    fields = set()
    for path in paths:
        tree = ast.parse(path.read_text(), str(path))
        fields.update(
            node.target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        )
    return fields


def option_map(argv):
    assert argv
    result = {}
    index = 1
    while index < len(argv):
        assert argv[index].startswith('--'), argv[index:]
        result[argv[index]] = argv[index + 1]
        index += 2
    return result


def fake_executable(path, capture, *, copy_config=False):
    lines = ['#!/usr/bin/env bash', 'set -euo pipefail', f'printf \"%s\\\\n\" \"$@\" > \"{capture}\"']
    if copy_config:
        lines.extend([
            'while (($#)); do',
            '  if [[ \"$1\" == --config ]]; then',
            '    cp \"$2\" \"${CAPTURE_CONFIG}\"',
            '    break',
            '  fi',
            '  shift',
            'done',
        ])
    path.write_text('\n'.join(lines) + '\n')
    path.chmod(0o755)


def test_variants_resolve_from_outside_the_repository(tmp_path):
    foreign_cwd = tmp_path / 'caller'
    foreign_cwd.mkdir()
    resolved = {
        name: resolve_config(name, tmp_path, cwd=foreign_cwd)
        for name in CONFIGS
    }
    assert all('defaults' not in recipe for recipe in resolved.values())
    assert resolved['rkl']['model'] == PUBLIC_ENV['STUDENT_MODEL']
    assert resolved['rkl']['teacher_models'] == [
        PUBLIC_ENV['TEACHER_STEM'],
        PUBLIC_ENV['TEACHER_VQA'],
        PUBLIC_ENV['TEACHER_DOCUMENT'],
        PUBLIC_ENV['TEACHER_GUI'],
        PUBLIC_ENV['TEACHER_EMBODIED'],
    ]
    assert resolved['rkl']['vllm_server_port'] == [PUBLIC_ENV['ROLLOUT_PORT']]
    assert resolved['rkl']['vllm_server_group_port'] == [PUBLIC_ENV['ROLLOUT_GROUP_PORT']]


def test_all_published_keys_are_declared_cli_dataclass_fields(tmp_path):
    fields = declared_argument_fields()
    for name in CONFIGS:
        recipe = resolve_config(name, tmp_path)
        assert set(recipe) <= fields, f'{name}: undeclared keys {sorted(set(recipe) - fields)}'


def test_matched_objectives_change_only_the_published_controls(tmp_path):
    rkl = resolve_config('rkl', tmp_path)
    fkl = resolve_config('fkl', tmp_path)
    todi = resolve_config('todi', tmp_path)

    assert {key for key in set(rkl) | set(fkl) if rkl.get(key) != fkl.get(key)} == {'beta'}
    assert (rkl['gkd_loss_type'], rkl['beta']) == ('jsd', 1)
    assert (fkl['gkd_loss_type'], fkl['beta']) == ('jsd', 0)

    assert {key for key in set(rkl) | set(todi) if rkl.get(key) != todi.get(key)} == {
        'gkd_loss_type',
        'gkd_todi_beta',
    }
    assert (todi['gkd_loss_type'], todi['gkd_todi_beta'], todi['beta']) == ('todi', 1, 1)


def test_main_recipe_freezes_batch_parallelism_and_domain_budgets(tmp_path):
    recipe = resolve_config('rkl', tmp_path)
    assert (recipe['micro_batch_size'], recipe['global_batch_size']) == (1, 512)
    assert (
        recipe['tensor_model_parallel_size'],
        recipe['expert_model_parallel_size'],
        recipe['expert_tensor_parallel_size'],
        recipe['pipeline_model_parallel_size'],
        recipe['context_parallel_size'],
    ) == (2, 8, 1, 1, 1)
    assert recipe['mopd_domain_generation_config'] == {
        'StemPuzzle': {'max_tokens': 12288, 'temperature': 1, 'max_prompt_tokens': 20480},
        'GeneralVQA': {'max_tokens': 8192, 'temperature': 1, 'max_prompt_tokens': 20480},
        'TextChartUnder': {'max_tokens': 12288, 'temperature': 1, 'max_prompt_tokens': 20480},
        'guiagent': {'max_tokens': 1024, 'temperature': 1, 'max_prompt_tokens': 20480},
        'EmbodUnde': {'max_tokens': 4096, 'temperature': 1, 'max_prompt_tokens': 20480},
    }
    assert recipe['mopd_router_replay'] == 'r3'
    assert recipe['mopd_loss_normalization'] == 'global_token'
    assert recipe['mopd_response_only_logits'] is True


def test_teacher_server_is_registered_as_a_megatron_cli_route():
    route_file = ROOT / 'ms-swift/swift/cli/_megatron/main.py'
    tree = ast.parse(route_file.read_text(), str(route_file))
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == 'ROUTE_MAPPING'
    )
    routes = ast.literal_eval(assignment.value)
    assert routes['teacher-server'] == 'swift.cli._megatron.teacher_server'
    entry = (ROOT / 'ms-swift/swift/cli/_megatron/teacher_server.py').read_text()
    assert 'megatron_teacher_server_main()' in entry


@pytest.mark.parametrize('script', ['train_unimopd.sh', 'launch_rollout.sh'])
def test_shell_entry_points_parse(script):
    subprocess.run(['bash', '-n', str(ROOT / 'scripts' / script)], check=True)


def test_train_launcher_resolves_config_and_forwards_optional_cache(tmp_path):
    binary_dir = tmp_path / 'bin'
    binary_dir.mkdir()
    capture = tmp_path / 'megatron-argv.txt'
    captured_config = tmp_path / 'resolved.yaml'
    fake_executable(binary_dir / 'megatron', capture, copy_config=True)
    env = {
        **os.environ,
        **PUBLIC_ENV,
        'PATH': f'{binary_dir}:{os.environ["PATH"]}',
        'CAPTURE_CONFIG': str(captured_config),
        'MOPD_MODEL_CACHE_DIR': 'cache/converted-mcore',
    }
    subprocess.run(
        [str(ROOT / 'scripts/train_unimopd.sh'), str(CONFIGS['fkl'])],
        cwd=tmp_path,
        env=env,
        check=True,
    )
    argv = capture.read_text().splitlines()
    assert argv[0] == 'rlhf'
    options = option_map(argv)
    assert '--config' in options
    assert options['--mopd_model_cache_dir'] == 'cache/converted-mcore'
    recipe = resolved_container(captured_config)
    assert (recipe['gkd_loss_type'], recipe['beta']) == ('jsd', 0)


def test_rollout_launcher_forwards_public_topology(tmp_path):
    binary_dir = tmp_path / 'bin'
    binary_dir.mkdir()
    capture = tmp_path / 'swift-argv.txt'
    fake_executable(binary_dir / 'swift', capture)
    output = tmp_path / 'rollout-output'
    env = {
        **os.environ,
        'PATH': f'{binary_dir}:{os.environ["PATH"]}',
        'STUDENT_MODEL': 'models/merge-equal',
        'OUTPUT_DIR': str(output),
        'ROLLOUT_PORT': '8123',
        'ROLLOUT_TP': '2',
        'ROLLOUT_DP': '4',
    }
    subprocess.run(
        [str(ROOT / 'scripts/launch_rollout.sh')],
        cwd=tmp_path,
        env=env,
        check=True,
    )
    argv = capture.read_text().splitlines()
    assert argv[0] == 'rollout'
    options = option_map(argv)
    assert options['--model'] == 'models/merge-equal'
    assert options['--port'] == '8123'
    assert options['--vllm_tensor_parallel_size'] == '2'
    assert options['--vllm_data_parallel_size'] == '4'
    assert options['--vllm_return_routed_experts'] == 'true'
    assert options['--vllm_max_model_len'] == '32768'
