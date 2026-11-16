# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
"""Per-image pixel limits remain explicit and independently validated."""
import pytest

from tests.mopd.source_test_utils import load_module


@pytest.fixture(scope='module')
def image_budget():
    return load_module(
        'ms-swift/swift/llm/template/image_budget.py',
        'unimopd_test_image_budget',
    )


@pytest.mark.parametrize(
    'value',
    [
        {'min_pixels': 0},
        {'max_pixels': -1},
        {'max_pixels': 12.5},
        {'min_pixels': True},
        {'max_pixels': '65536'},
        {'min_pixels': 65536, 'max_pixels': 4096},
    ],
)
def test_invalid_limits_are_not_defaulted(image_budget, value):
    with pytest.raises(ValueError):
        image_budget.image_pixel_budget(value)


def test_nullable_and_partial_limits(image_budget):
    assert image_budget.image_pixel_budget({'min_pixels': None, 'max_pixels': None}) == {}
    assert image_budget.image_pixel_budget({'min_pixels': None, 'max_pixels': 65536}) == {
        'max_pixels': 65536,
    }
    assert image_budget.image_pixel_budget('image.png') == {}


@pytest.mark.parametrize(
    'entry,source,budget',
    [
        (
            {'path': 'first.png', 'min_pixels': 4096, 'max_pixels': 16384},
            'first.png',
            {'min_pixels': 4096, 'max_pixels': 16384},
        ),
        (
            {'image': 'second.png', 'min_pixels': 65536},
            'second.png',
            {'min_pixels': 65536},
        ),
        (
            {'image_url': {'url': 'https://example.org/image.png'}, 'max_pixels': 262144},
            'https://example.org/image.png',
            {'max_pixels': 262144},
        ),
        (
            {'bytes': b'png', 'path': 'fallback.png', 'min_pixels': 4096},
            b'png',
            {'min_pixels': 4096},
        ),
    ],
)
def test_split_entry_keeps_each_images_source_and_budget(image_budget, entry, source, budget):
    assert image_budget.split_image_entry(entry) == (source, budget)


def test_missing_image_source_is_rejected(image_budget):
    with pytest.raises(ValueError, match='needs bytes, path, image or image_url'):
        image_budget.split_image_entry({'min_pixels': 4096, 'max_pixels': 16384})
