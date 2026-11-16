# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
"""The published merge recipes parse into the arguments the merger accepts."""
import inspect
from pathlib import Path

import pytest

from unimopd.merge import streaming
from unimopd.merge.streaming import load_recipe, merge

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / 'configs' / 'merge'
TEACHERS = ('TEACHER_STEM', 'TEACHER_VQA', 'TEACHER_DOCUMENT', 'TEACHER_GUI', 'TEACHER_EMBODIED')
PLACEHOLDERS = ('BASE_MODEL', 'OUTPUT_DIR') + TEACHERS


@pytest.fixture
def environment(monkeypatch, tmp_path):
    for name in PLACEHOLDERS:
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    return tmp_path


def recipes():
    return sorted(CONFIGS.glob('*.yaml'))


def test_the_published_recipes_are_present():
    assert [path.name for path in recipes()] == ['tuned.yaml', 'uniform.yaml']


@pytest.mark.parametrize('config', recipes(), ids=lambda path: path.stem)
def test_every_recipe_field_is_accepted_by_the_merger(config, environment):
    recipe = load_recipe(config)
    accepted = set(inspect.signature(merge).parameters)

    assert set(recipe).issubset(accepted), set(recipe) - accepted
    # main() rejects anything outside its allowlist before calling merge().
    source = inspect.getsource(streaming.main)
    for field in recipe:
        assert f"'{field}'" in source, field


@pytest.mark.parametrize('config', recipes(), ids=lambda path: path.stem)
def test_every_recipe_is_a_five_teacher_dare_merge(config, environment):
    recipe = load_recipe(config)

    assert recipe['method'] == 'dare'
    assert len(recipe['inputs']) == 5
    assert recipe['drop_rates'] == [0.7] * 5
    assert recipe['seed'] == 42
    assert sum(recipe['weights']) == pytest.approx(1.0)


@pytest.mark.parametrize('config', recipes(), ids=lambda path: path.stem)
def test_recipes_carry_no_absolute_paths(config):
    text = config.read_text()

    assert '/mnt/' not in text and '/root/' not in text
    for field in ('base', 'output_dir'):
        assert f'{field}: "${{' in text, field


def test_the_uniform_recipe_matches_the_paper_initialization(environment):
    recipe = load_recipe(CONFIGS / 'uniform.yaml')

    assert recipe['weights'] == [0.2] * 5


def test_the_tuned_recipe_matches_the_reported_coefficients(environment):
    recipe = load_recipe(CONFIGS / 'tuned.yaml')

    assert recipe['weights'] == [0.1, 0.15, 0.35, 0.3, 0.1]


@pytest.mark.parametrize('config', recipes(), ids=lambda path: path.stem)
def test_teacher_paths_come_from_the_documented_variables(config, environment):
    recipe = load_recipe(config)
    resolved = [str(environment / name.lower()) for name in TEACHERS]

    assert recipe['inputs'] == resolved
    assert recipe['base'] == str(environment / 'base_model')
    assert recipe['output_dir'] == str(environment / 'output_dir')


@pytest.mark.parametrize('config', recipes(), ids=lambda path: path.stem)
@pytest.mark.parametrize('missing', PLACEHOLDERS)
def test_an_unset_path_variable_fails_before_any_merge_work(config, environment, monkeypatch, missing):
    monkeypatch.delenv(missing)

    with pytest.raises(ValueError, match=f'requires environment variable {missing}'):
        load_recipe(config)
