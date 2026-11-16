# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL MoE DP dummy-input alignment for the pinned vLLM 0.11 runtime.

In 0.11, GPUModelRunner._dummy_run computes num_tokens_after_padding for its
forward context but slices model inputs with the original num_tokens. An idle
DP rank can therefore send one row to MoE with metadata describing many rows.

Align only those unused dummy input views. Keep native DP synchronization,
logical request lengths, attention metadata, CUDA graphs and real forwards.
Nothing is written into the installed vLLM package or model buffers.
"""
import functools
from importlib.metadata import version


def _forward_context():
    from vllm.forward_context import get_forward_context
    return get_forward_context()


class _PaddedDummyModel:

    def __init__(self, runner, model):
        self.runner = runner
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        runner = self.runner
        metadata = _forward_context().dp_metadata
        rank = runner.vllm_config.parallel_config.data_parallel_rank
        expected = int(metadata.num_tokens_across_dp_cpu[rank])
        values = inputs_embeds if inputs_embeds is not None else input_ids
        actual = values.shape[0]
        if expected != actual:
            if expected < actual:
                raise ValueError('vLLM dummy padding must not truncate model inputs.')
            if intermediate_tensors is not None or kwargs:
                raise ValueError('Qwen3-VL dummy padding expects a PP1 causal-LM input.')
            if positions.shape[-1] != actual:
                raise ValueError('vLLM dummy positions already differ from the input rows.')
            if inputs_embeds is not None:
                inputs_embeds = runner.inputs_embeds.gpu[:expected]
                values = inputs_embeds
            else:
                input_ids = runner.input_ids.gpu[:expected]
                values = input_ids
            positions = (runner.mrope_positions.gpu[:, :expected] if runner.uses_mrope
                         else runner.positions.gpu[:expected])
            if values.shape[0] != expected or positions.shape[-1] != expected:
                raise ValueError('vLLM dummy padding exceeds the allocated input buffers.')
            runner._mopd_dummy_padding_adjustments = getattr(runner, '_mopd_dummy_padding_adjustments', 0) + 1
        return self.model(
            input_ids=input_ids, positions=positions, intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds, **kwargs)


def patch_runner_class(runner_class):
    """Worker-local, idempotent patch; the native real-input path is untouched."""
    if getattr(runner_class, '_mopd_dummy_padding_fix', False):
        return
    original = runner_class._dummy_run

    @functools.wraps(original)
    def dummy_run(self, *args, **kwargs):
        parallel = self.vllm_config.parallel_config
        hf = self.vllm_config.model_config.hf_config
        if (getattr(hf, 'model_type', None) != 'qwen3_vl_moe'
                or parallel.data_parallel_size <= 1 or parallel.pipeline_parallel_size != 1
                or getattr(parallel, 'enable_dbo', False)):
            return original(self, *args, **kwargs)
        # vLLM worker execution is serialized. The proxy exists only during a
        # dummy call and delegates to the exact original model/CUDA wrapper.
        model = self.model
        self.model = _PaddedDummyModel(self, model)
        try:
            return original(self, *args, **kwargs)
        finally:
            self.model = model

    runner_class._dummy_run = dummy_run
    runner_class._mopd_dummy_padding_fix = True


def bootstrap_worker_dummy_padding():
    from packaging.version import Version
    if Version(version('vllm')).base_version != '0.11.0':
        return False
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    patch_runner_class(GPUModelRunner)
    return True
