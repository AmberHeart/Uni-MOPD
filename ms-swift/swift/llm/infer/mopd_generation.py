# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, MOPD contributors.
"""Explicit per-domain sampling budgets, independent of the KD objective."""
import json
import math
from copy import deepcopy
from pathlib import Path

MOPD_GENERATION_VERSION = 1
MOPD_GENERATION_INFO_KEY = 'mopd_generation'


def _validate_sampling(max_tokens, temperature, *, max_allowed=None):
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError('MOPD max_tokens must be a positive integer.')
    if max_allowed is not None and max_tokens > max_allowed:
        raise ValueError('A domain max_tokens budget cannot exceed the global max_completion_length.')
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature < 0):
        raise ValueError('MOPD rollout temperature must be finite and nonnegative.')


def parse_domain_generation_config(value, *, max_tokens, temperature, domains=None, max_prompt_tokens=None):
    """Resolve a JSON object/path without changing any omitted-domain default."""
    if value is None:
        return {}
    if isinstance(value, str):
        value = json.loads(value) if value.lstrip().startswith('{') else json.loads(Path(value).read_text())
    if not isinstance(value, dict):
        raise ValueError('mopd_domain_generation_config must map domain names to sampling overrides.')
    result = {}
    for domain, overrides in value.items():
        if not isinstance(domain, str) or not domain or not isinstance(overrides, dict) or not overrides:
            raise ValueError('Each domain needs a nonempty sampling override object.')
        if domains is not None and domain not in domains:
            raise ValueError(f'Unknown sampling domain {domain!r}; use the dataset/domain-teacher owner keys.')
        unknown = set(overrides) - {'max_tokens', 'temperature', 'max_prompt_tokens'}
        if unknown:
            raise ValueError(f'Unsupported MOPD sampling fields: {sorted(unknown)}.')
        entry = {'max_tokens': overrides.get('max_tokens', max_tokens),
                 'temperature': overrides.get('temperature', temperature)}
        _validate_sampling(**entry, max_allowed=max_tokens)
        if 'max_prompt_tokens' in overrides:
            prompt = overrides['max_prompt_tokens']
            if (type(prompt) is not int or prompt <= 0
                    or (max_prompt_tokens is not None and prompt > max_prompt_tokens)):
                raise ValueError('A domain prompt budget must be positive and within the global prompt envelope.')
            entry['max_prompt_tokens'] = prompt
        result[domain] = entry
    return result


def generation_for_domain(domain, config, *, max_tokens, temperature):
    if not isinstance(domain, str) or not domain:
        raise ValueError('Per-domain MOPD generation requires each sample to retain its domain.')
    entry = config.get(domain, {'max_tokens': max_tokens, 'temperature': temperature})
    sampling = {key: entry[key] for key in ('max_tokens', 'temperature')}
    _validate_sampling(**sampling, max_allowed=max_tokens)
    # Prompt admission is enforced by the trainer before dispatch. Keep the
    # existing acknowledged wire policy unchanged.
    return {'version': MOPD_GENERATION_VERSION, 'domain': domain, **sampling}


def request_generation_metadata(inputs):
    template_inputs = inputs.get('template_inputs')
    if template_inputs is None:
        return None
    return template_inputs.extra_kwargs.get('data_dict', {}).get(MOPD_GENERATION_INFO_KEY)


def apply_request_generation(config, metadata):
    """Copy the common request config; override only the two documented knobs."""
    result = deepcopy(config)
    if metadata is None:
        return result
    if metadata.get('version') != MOPD_GENERATION_VERSION:
        raise ValueError('Unsupported MOPD per-request generation protocol.')
    if set(metadata) != {'version', 'domain', 'max_tokens', 'temperature'}:
        raise ValueError('Invalid MOPD generation metadata fields.')
    if not isinstance(metadata['domain'], str) or not metadata['domain']:
        raise ValueError('MOPD generation metadata is missing its owner domain.')
    _validate_sampling(metadata['max_tokens'], metadata['temperature'], max_allowed=config.max_tokens)
    result.max_tokens = metadata['max_tokens']
    result.temperature = metadata['temperature']
    return result


def validate_generation_ack(expected, actual):
    """Fail before training if the server ignored or misassociated a budget."""
    if (not isinstance(actual, dict) or actual.get('version') != MOPD_GENERATION_VERSION
            or actual.get('domain') != expected['domain']
            or actual.get('requested_max_tokens') != expected['max_tokens']
            or actual.get('temperature') != expected['temperature']):
        raise RuntimeError('The server did not acknowledge this sample/domain sampling policy.')
    _validate_sampling(actual.get('max_tokens'), actual.get('temperature'), max_allowed=expected['max_tokens'])
    return dict(actual)
