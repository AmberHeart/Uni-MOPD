# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, MOPD contributors.
"""Opt-in native FlashAttention version selection without changing precision.

Validated against Transformer Engine 2.9's explicit flash_attention_backend
forward argument. No package availability flag or process-global selector is
modified. Installation alone does not enable global deterministic algorithms.
"""
import inspect


def flash_version_pre_hook(signature, version, *, record_calls=False):
    def before_forward(module, positional, kwargs):
        bound = signature.bind(*positional, **kwargs)
        if bound.arguments.get('fp8', False):
            raise ValueError('The MOPD FA2 override is for native BF16/FP16 attention, not FP8.')
        bound.arguments['flash_attention_backend'] = version
        if record_calls:
            module._mopd_flash_v2_calls += 1
        return bound.args, bound.kwargs
    return before_forward


def force_native_flash_attention_v2(model, *, record_calls=False):
    from transformer_engine.pytorch.attention.dot_product_attention.backends import FlashAttention
    from transformer_engine.pytorch.attention.dot_product_attention.utils import FlashAttentionUtils
    if not FlashAttentionUtils.is_installed:
        raise ValueError('A Transformer Engine-supported flash-attn v2 installation is required.')
    modules = [module for module in model.modules() if isinstance(module, FlashAttention)]
    if not modules:
        raise ValueError('No native Transformer Engine FlashAttention modules were found.')
    for module in modules:
        if getattr(module, '_mopd_flash_attention_override', None) == 2:
            continue
        signature = inspect.signature(module.forward)
        if 'flash_attention_backend' not in signature.parameters:
            raise ValueError('This TE version does not expose the tested flash_attention_backend argument.')
        module._mopd_flash_v2_calls = 0
        module.register_forward_pre_hook(
            flash_version_pre_hook(signature, FlashAttentionUtils.version, record_calls=record_calls),
            with_kwargs=True)
        module._mopd_flash_attention_override = 2
    return len(modules)
