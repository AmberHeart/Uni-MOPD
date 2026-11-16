# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from tqdm.asyncio import tqdm_asyncio

from swift.llm import InferRequest, Template, VllmEngine
from swift.plugin import Metric
from ..protocol import ChatCompletionResponse, ChatCompletionResponseChoice, ChatMessage, RequestConfig, RolloutOutput
from ..routing_protocol import ROUTING_INFO_KEY, ROUTING_PROTOCOL_VERSION
from ..mopd_generation import (MOPD_GENERATION_INFO_KEY, apply_request_generation, request_generation_metadata)
from .utils import AdapterRequest

try:
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = '86400'
    from vllm.lora.request import LoRARequest
except Exception:
    raise


class GRPOVllmEngine(VllmEngine):

    def __init__(
        self,
        model_id_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        *,
        use_async_engine: bool = False,
        model_type: Optional[str] = None,
        use_hf: Optional[bool] = None,
        hub_token: Optional[str] = None,
        revision: Optional[str] = None,
        # engine_kwargs
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        enable_expert_parallel: bool = False,
        max_model_len: Optional[int] = None,
        max_num_seqs: int = 256,
        disable_custom_all_reduce: bool = True,
        enforce_eager: bool = False,
        limit_mm_per_prompt: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        task_type: Optional[str] = None,
        disable_cascade_attn: bool = False,
        load_format: str = 'auto',
        mm_processor_cache_gb: Optional[float] = None,
        logprobs_mode: Optional[str] = None,
        speculative_config: Optional[Union[str, dict]] = None,
        # lora
        enable_lora: bool = False,
        max_loras: int = 1,
        max_lora_rank: int = 16,
        enable_prefix_caching: Optional[bool] = None,
        enable_sleep_mode: bool = False,
        distributed_executor_backend: Optional[str] = None,
        quantization: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        template: Optional[Template] = None,
        return_routed_experts: bool = False,
        **kwargs,
    ) -> None:
        self.return_routed_experts = return_routed_experts
        self._routing_capabilities_checked = False
        self._mopd_generation_by_request = {}
        if return_routed_experts:
            from ..vllm_routing_replay import BOOTSTRAP_ENV, CONFIG_KEY, install_vllm_routing_hooks
            os.environ[BOOTSTRAP_ENV] = '1'
            install_vllm_routing_hooks()
            engine_kwargs = dict(engine_kwargs or {})
            engine_kwargs['additional_config'] = {
                **engine_kwargs.get('additional_config', {}), CONFIG_KEY: True}
        super().__init__(
            model_id_or_path=model_id_or_path,
            torch_dtype=torch_dtype,
            use_async_engine=use_async_engine,
            model_type=model_type,
            use_hf=use_hf,
            hub_token=hub_token,
            revision=revision,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            enable_expert_parallel=enable_expert_parallel,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            disable_custom_all_reduce=disable_custom_all_reduce,
            enforce_eager=enforce_eager,
            limit_mm_per_prompt=limit_mm_per_prompt,
            seed=seed,
            task_type=task_type,
            disable_cascade_attn=disable_cascade_attn,
            load_format=load_format,
            mm_processor_cache_gb=mm_processor_cache_gb,
            logprobs_mode=logprobs_mode,
            speculative_config=speculative_config,
            enable_lora=enable_lora,
            max_loras=max_loras,
            max_lora_rank=max_lora_rank,
            enable_prefix_caching=enable_prefix_caching,
            enable_sleep_mode=enable_sleep_mode,
            distributed_executor_backend=distributed_executor_backend,
            quantization=quantization,
            engine_kwargs=engine_kwargs,
            template=template,
        )

    def _request_config_for_inputs(self, request_config, inputs):
        return apply_request_generation(request_config, request_generation_metadata(inputs))

    def _add_request(self, inputs, generation_config, request_id, *args, **kwargs):
        result = super()._add_request(inputs, generation_config, request_id, *args, **kwargs)
        metadata = request_generation_metadata(inputs)
        if metadata is not None:
            self._mopd_generation_by_request[request_id] = {
                **metadata, 'requested_max_tokens': metadata['max_tokens'],
                'max_tokens': generation_config.max_tokens, 'temperature': generation_config.temperature,
            }
        return result

    def _attach_generation_metadata(self, outputs):
        for output in outputs:
            metadata = self._mopd_generation_by_request.pop(output.response.id, None)
            if metadata is not None:
                output.rollout_infos = {**(output.rollout_infos or {}), MOPD_GENERATION_INFO_KEY: metadata}

    def infer(
        self,
        infer_requests: List[Union[InferRequest, Dict[str, Any]]],
        request_config: Optional[RequestConfig] = None,
        metrics: Optional[List[Metric]] = None,
        *,
        template: Optional[Template] = None,
        use_tqdm: Optional[bool] = None,
        adapter_request: Optional[AdapterRequest] = None,
    ) -> List[RolloutOutput]:
        if self.return_routed_experts:
            request_config = request_config or RequestConfig()
            if request_config.n != 1 or request_config.stream:
                raise ValueError('R3 export requires non-streaming requests with n=1.')
            if not self._routing_capabilities_checked:
                self._check_routing_capabilities(
                    self.engine.collective_rpc(method='mopd_routing_capabilities', timeout=120))
        if not adapter_request and self.enable_lora:
            lora_int_ids = list(self.engine.list_loras())
            if lora_int_ids:
                # since max_lora = 1, pick the first lora
                adapter_request = LoRARequest(
                    lora_name=f'{lora_int_ids[0]}',
                    lora_int_id=lora_int_ids[0],
                    lora_path='dummy_lora_path',
                )

        res = super().infer(
            infer_requests,
            request_config,
            metrics,
            template=template,
            use_tqdm=use_tqdm,
            adapter_request=adapter_request,
        )
        if not isinstance(res, list):
            res = [res]
        for i, result in enumerate(res):
            if not isinstance(result, RolloutOutput):
                if not isinstance(result, ChatCompletionResponse):
                    raise TypeError('Result must be a ChatCompletionResponse or RolloutOutput instance.')
                res[i] = RolloutOutput(response=result)

        if self.return_routed_experts:
            for chunk in self._routing_export_chunks(res):
                request_ids = [output.response.id for output in chunk]
                replies = self.engine.collective_rpc(
                    method='mopd_pop_routing_traces', args=(request_ids,), timeout=120)
                self._attach_routing(chunk, replies)
        self._attach_generation_metadata(res)
        return res

    def infer_completed(self, infer_requests, request_config, *, completion_callback):
        """Deliver whole responses with R3 before the batch's last answer finishes.

        This retains the synchronous native engine and unchanged sampling
        settings. It is deliberately separate from token-streaming/async/LoRA
        paths; callbacks never run model or routing collectives in an I/O thread.
        """
        if self.use_async_engine or self.enable_lora:
            raise ValueError('Complete-response MOPD delivery supports synchronous full-model inference only.')
        if request_config.n != 1 or request_config.stream or not request_config.return_details:
            raise ValueError('Complete-response MOPD delivery requires n=1, full token details and no token stream.')
        if self.return_routed_experts and not self._routing_capabilities_checked:
            self._check_routing_capabilities(
                self.engine.collective_rpc(method='mopd_routing_capabilities', timeout=120))

        def completed(rows):
            outputs = [RolloutOutput(response=response) for _, response, _ in rows]
            if self.return_routed_experts:
                for chunk in self._routing_export_chunks(outputs):
                    replies = self.engine.collective_rpc(
                        method='mopd_pop_routing_traces',
                        args=([output.response.id for output in chunk],), timeout=120)
                    self._attach_routing(chunk, replies)
            self._attach_generation_metadata(outputs)
            completion_callback([(index, output, timing) for (index, _, timing), output in zip(rows, outputs)])

        return super().infer(
            infer_requests, request_config, use_tqdm=False, completion_callback=completed)

    async def async_infer(self,
                          infer_requests: List[InferRequest],
                          request_config: Optional[RequestConfig] = None,
                          metrics: Optional[List[Metric]] = None,
                          *,
                          use_tqdm: Optional[bool] = None,
                          **kwargs) -> List[RolloutOutput]:
        if request_config is None:
            request_config = RequestConfig()
        assert request_config.n == 1
        if self.return_routed_experts:
            if request_config.stream:
                raise ValueError('R3 export requires non-streaming requests.')
            if not self._routing_capabilities_checked:
                self._check_routing_capabilities(
                    await self.engine.collective_rpc(method='mopd_routing_capabilities', timeout=120))

        tasks = [self.infer_async(infer_request, request_config, **kwargs) for infer_request in infer_requests]
        if use_tqdm is None:
            use_tqdm = len(infer_requests) > 1
        res = await self._batch_infer_stream(tasks, request_config.stream, use_tqdm, metrics)

        for i, result in enumerate(res):
            if not isinstance(result, RolloutOutput):
                if not isinstance(result, ChatCompletionResponse):
                    raise TypeError('Result must be a ChatCompletionResponse or RolloutOutput instance.')
                res[i] = RolloutOutput(response=result)

        if self.return_routed_experts:
            for chunk in self._routing_export_chunks(res):
                request_ids = [output.response.id for output in chunk]
                replies = await self.engine.collective_rpc(
                    method='mopd_pop_routing_traces', args=(request_ids,), timeout=120)
                self._attach_routing(chunk, replies)
        self._attach_generation_metadata(res)
        return res

    @staticmethod
    def _routing_export_chunks(outputs, token_budget=131072):
        # ~64 MiB JSON payload for Qwen3-VL-30B's uint8 [tokens, 48, 8]
        # routing bank. Do not build a multi-GiB worker RPC at full global batch size.
        chunk, tokens = [], 0
        for output in outputs:
            usage = output.response.usage
            length = max(1, usage.prompt_tokens + usage.completion_tokens - 1)
            if chunk and tokens + length > token_budget:
                yield chunk
                chunk, tokens = [], 0
            chunk.append(output)
            tokens += length
        if chunk:
            yield chunk

    @staticmethod
    def _routing_replies(replies):
        if isinstance(replies, list):
            for item in replies:
                yield from GRPOVllmEngine._routing_replies(item)
        elif isinstance(replies, dict):
            yield replies
        else:
            raise RuntimeError(f'Invalid routing RPC result: {type(replies).__name__}.')

    def _check_routing_capabilities(self, replies):
        workers = list(self._routing_replies(replies))
        if (not workers or not any(worker.get('source') for worker in workers)
                or any(not worker.get('ready') or worker.get('version') != ROUTING_PROTOCOL_VERSION
                       for worker in workers)):
            raise RuntimeError('All rollout workers must use the matching, initialized R3 exporter.')
        self._routing_capabilities_checked = True

    def _attach_routing(self, outputs, replies):
        from swift.llm.infer.routing_protocol import bind_routing_trace

        captures = {}
        for worker in self._routing_replies(replies):
            for request_id, trace in worker.items():
                if request_id in captures:
                    raise RuntimeError(f'R3 request {request_id} was captured by multiple DP sources.')
                captures[request_id] = trace
        for output in outputs:
            request_id = output.response.id
            if request_id not in captures:
                raise RuntimeError(f'No completed routing trace for rollout request {request_id}.')
            response = output.response
            if len(response.choices) != 1:
                raise RuntimeError('R3 binding requires exactly one returned completion.')
            try:
                trace = bind_routing_trace(
                    captures.pop(request_id),
                    response.prompt_token_ids,
                    response.choices[0].token_ids,
                    max_tokens=self.engine.model_config.max_model_len,
                )
            except ValueError as error:
                raise ValueError(f'R3 request {request_id}: {error}') from error
            output.rollout_infos[ROUTING_INFO_KEY] = trace
        if captures:
            raise RuntimeError('Routing RPC returned traces belonging to another request batch.')

    async def _batch_infer_stream(self,
                                  tasks,
                                  stream: bool = True,
                                  use_tqdm: bool = True,
                                  metrics: Optional[List[Metric]] = None):
        assert not stream
        prog_bar = tqdm_asyncio(total=len(tasks), dynamic_ncols=True, disable=not use_tqdm)

        async def _new_run(task):
            try:
                res = await task
            except Exception as e:
                if getattr(self, 'strict', True):
                    raise
                res = e
            prog_bar.update()
            self._update_metrics(res, metrics)
            return res

        new_tasks = [_new_run(task) for task in tasks]
        return await self.batch_run(new_tasks)

    def _create_chat_completion_response(self, result, inputs, template: Template, request_config,
                                         request_id) -> ChatCompletionResponse:
        assert result is not None
        num_generated_tokens = sum(len(output.token_ids) for output in result.outputs)
        usage_info = self._get_usage_info(len(result.prompt_token_ids), num_generated_tokens)
        choices = []
        for output in result.outputs:
            output.token_ids = list(output.token_ids)
            response = template.decode(output.token_ids)
            logprobs = self._get_logprobs(output.logprobs, output.token_ids, request_config.top_logprobs)
            toolcall = self._get_toolcall(response, template)

            token_ids = output.token_ids if request_config.return_details else None
            choice = ChatCompletionResponseChoice(
                index=output.index,
                message=ChatMessage(role='assistant', content=response, tool_calls=toolcall),
                finish_reason=output.finish_reason,
                logprobs=logprobs,
                token_ids=token_ids,
            )
            choices.append(choice)
        prompt_token_ids = None
        images_size = None
        if request_config.return_details:
            prompt_token_ids = result.prompt_token_ids
            images = inputs['template_inputs'].images
            if all(isinstance(image, Image.Image) for image in images):
                images_size = [image.size for image in images]
        return ChatCompletionResponse(
            model=self.model_name,
            choices=choices,
            usage=usage_info,
            id=request_id,
            prompt_token_ids=prompt_token_ids,
            images_size=images_size)

    def _add_adapter(self, adapter_request: Optional[Union[AdapterRequest, LoRARequest]] = None):
        assert self.enable_lora, f'adapter_request: {adapter_request}, self.enable_lora: {self.enable_lora}'
        from vllm.lora.request import LoRARequest
        if isinstance(adapter_request, AdapterRequest):
            return super()._add_adapter(adapter_request)
        elif isinstance(adapter_request, LoRARequest):
            return adapter_request
        else:
            raise ValueError(f'Invalid adapter request: {adapter_request}')
