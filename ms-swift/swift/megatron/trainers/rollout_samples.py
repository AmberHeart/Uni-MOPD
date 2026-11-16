# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, MOPD contributors.
"""Small, local-only rollout records; no model, image decoding or cloud SDK.

Records use the ``generations_media/stepN/index.jsonl`` layout. Keep original
image references instead of copying or resizing training images.
Only selected human-readable fields are retained: never routing tensors,
log-probability arrays, encoded pixels or teacher targets.
"""
import copy
import json
import os
from pathlib import Path
import random
import re
import tempfile


def sample_indices(total: int, count: int, seed: int, step: int) -> list:
    if type(count) is not int or count < 0:
        raise ValueError('mopd_rollout_samples_per_step must be a nonnegative integer')
    return random.Random(int(seed) + int(step)).sample(range(total), min(count, total))


def message_text(content) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise TypeError(f'Unsupported message content: {type(content).__name__}')
    parts = []
    for part in content:
        if part.get('type') == 'text':
            parts.append(part.get('text', ''))
        elif part.get('type') in ('image', 'image_url'):
            parts.append('<image>')
        elif part.get('type') in ('video', 'audio'):
            parts.append(f'<{part["type"]}>')
        else:
            raise ValueError(f'Unsupported message part: {part.get("type")!r}')
    return ''.join(parts)


def normalized_messages(sample: dict) -> list:
    roles = {'human': 'user', 'gpt': 'assistant'}
    result = []
    for message in sample.get('messages', sample.get('conversations', [])):
        role = message.get('role', message.get('from'))
        result.append({'role': roles.get(role, role),
                       'content': message_text(message.get('content', message.get('value')))})
    return result


def image_entries(sample: dict) -> list:
    images = sample.get('images', sample.get('image')) or []
    if images:
        return images if isinstance(images, (list, tuple)) else [images]
    result = []
    for message in sample.get('messages', []):
        content = message.get('content')
        if isinstance(content, list):
            for part in content:
                if part.get('type') in ('image', 'image_url'):
                    result.append(part.get('image', part.get('image_url')))
    return result


def image_reference(entry):
    """Return a disk/URL reference and pixel controls without embedding bytes."""
    budget = {}
    if isinstance(entry, dict):
        budget = {key: entry[key] for key in ('min_pixels', 'max_pixels', 'resized_height', 'resized_width')
                  if entry.get(key) is not None}
        entry = entry.get('path') or entry.get('image') or entry.get('url')
        if isinstance(entry, dict):
            entry = entry.get('url')
    if not isinstance(entry, str) or not entry or entry.startswith('data:') or len(entry) > 8192:
        return None, budget
    return entry, budget


def response_flags(response: str, *, finish_reason=None, is_truncated=False) -> list:
    """Browsing heuristics, not correctness labels or whole-batch metrics."""
    flags = []
    if not response.strip():
        flags.append('empty')
    if is_truncated or finish_reason == 'length':
        flags.append('truncated')
    if re.search(r'</?think(?:ing)?>', response, re.I):
        flags.append('think_present')
    if len(re.findall(r'<think(?:ing)?>', response, re.I)) > len(
            re.findall(r'</think(?:ing)?>', response, re.I)):
        flags.append('think_unclosed')
    tail = response.rstrip()[-1024:]
    # Four identical trailing blocks (at least 32 characters altogether).
    # Bounded scan; the result is a heuristic flag, not a correctness signal.
    for width in range(1, min(128, len(tail) // 4) + 1):
        repeats = max(4, (32 + width - 1) // width)
        block = tail[-width:]
        if block.strip() and tail.endswith(block * repeats):
            flags.append('repetition')
            break
    return flags


def sample_record(sample: dict, *, step: int, row_index, batch_size: int) -> dict:
    messages = normalized_messages(sample)
    response = ''
    if messages and messages[-1]['role'] == 'assistant':
        response = messages.pop()['content']
    prompt = '\n\n'.join(message['content'] for message in messages)
    references = [image_reference(entry) for entry in image_entries(sample)]
    images = [value for value, _ in references]
    finish_reason = sample.get('finish_reason')
    flags = response_flags(response, finish_reason=finish_reason, is_truncated=sample.get('is_truncated', False))
    tags = prompt.count('<image>')
    if tags != len(images):
        flags.append('image_tag_mismatch')
    if any(image is None for image in images):
        flags.append('image_unavailable')
    tokens = sample.get('response_token_ids')
    token_count = None
    if tokens is not None:
        token_count = (sum(len(turn) for turn in tokens)
                       if tokens and isinstance(tokens[0], (list, tuple)) else len(tokens))
    return {
        'schema_version': 1, 'step': int(step), 'row_index': row_index,
        'batch_size': int(batch_size), 'source': 'native_rollout',
        'sampling_scope': 'optimizer_window', 'sampling_population_size': int(batch_size),
        'source_match': 'captured',
        'sample_id': sample.get('id', sample.get('sid')),
        'occurrence_id': sample.get('_mopd_remote_occurrence', sample.get('request_id')),
        'domain': sample.get('domain') or '',
        'messages': messages, 'prompt': prompt, 'response': response, 'response_len': token_count,
        'response_chars': len(response), 'prompt_length': sample.get('prompt_length'),
        'finish_reason': finish_reason, 'is_truncated': sample.get('is_truncated', finish_reason == 'length'),
        'generation': copy.deepcopy(sample.get('_mopd_generation')),
        'images': images, 'image_budgets': [budget for _, budget in references],
        'image_count': len(images), 'image_tag_count': tags, 'flags': flags,
        'format_checks': ';'.join(flags),
    }


def write_sample_records(media_root, step: int, records: list) -> Path:
    """Publish a complete index atomically, preserving any different old attempt."""
    directory = Path(media_root) / f'step{int(step)}'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'index.jsonl'
    text = ''.join(json.dumps({**record, 'position': position}, ensure_ascii=False, allow_nan=False) + '\n'
                   for position, record in enumerate(records))
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise FileExistsError(f'Existing rollout samples differ; preserve them and use a new output: {path}')
        return path
    # link() publishes without overwriting a concurrently written attempt.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory, prefix='.index-', delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(text)
            file.flush()
            os.link(temporary, path)
        finally:
            temporary.unlink()
    return path
