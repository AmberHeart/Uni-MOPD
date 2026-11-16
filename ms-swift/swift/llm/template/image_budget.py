# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
"""Preserve explicit per-image pixel limits through dataset and request adapters."""
from typing import Any, Dict, Tuple


def image_pixel_budget(image: Any) -> Dict[str, int]:
    if not isinstance(image, dict):
        return {}
    budget = {key: image[key] for key in ('min_pixels', 'max_pixels') if image.get(key) is not None}
    for key, value in budget.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f'Image {key} must be a positive integer, got {value!r}')
    if len(budget) == 2 and budget['min_pixels'] > budget['max_pixels']:
        raise ValueError('Image min_pixels exceeds max_pixels')
    return budget


def split_image_entry(image: Any) -> Tuple[Any, Dict[str, int]]:
    """Accept HF bytes/path entries and public Qwen-style image entries."""
    budget = image_pixel_budget(image)
    if not isinstance(image, dict):
        return image, budget
    if image.get('bytes'):
        return image['bytes'], budget
    for key in ('path', 'image', 'image_url'):
        source = image.get(key)
        if source is not None:
            if isinstance(source, dict):
                source = source['url']
            return source, budget
    raise ValueError('Image entry needs bytes, path, image or image_url')
