# Copyright (c) Alibaba, Inc. and its affiliates.
# Code partially sourced from Hugging Face TRL

# fmt: off
# apply patch before importing trl, which may internally reference GuidedDecodingParams
try:
    import vllm
    try:
        from vllm.sampling_params import GuidedDecodingParams
    except ImportError:
        import vllm.sampling_params
        # removed in https://github.com/vllm-project/vllm/pull/22772
        vllm.sampling_params.GuidedDecodingParams = vllm.sampling_params.StructuredOutputsParams
except ImportError:
    pass
# fmt: on

import asyncio
import inspect
import multiprocessing
import os
import time
import traceback
from collections.abc import Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict
from functools import wraps
from itertools import chain
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Dict, List, Optional, Union

import torch
import uvicorn
from aiohttp import ClientConnectorError
from fastapi import FastAPI, HTTPException
from starlette.responses import StreamingResponse
from trl.scripts.vllm_serve import WeightSyncWorkerExtension as HFWeightSyncWorkerExtension

from swift.llm import RolloutArguments, SwiftPipeline
from swift.llm.template.template_inputs import RolloutInferRequest
from swift.plugin.multi_turn import RolloutScheduler, multi_turns
from swift.trainers.rlhf_trainer.utils import (FlattenedTensorBucket, FlattenedTensorMetadata, TensorLoRARequest,
                                               UpdateAdapterRequest, UpdateFlattenedAdapterRequest,
                                               UpdateFlattenedParamsRequest, check_vllm_version_ge,
                                               patch_vllm_load_adapter)
from swift.utils import get_logger
from .infer_engine import GRPOVllmEngine, InferClient
from .completion_stream import (COMPLETION_STREAM_VERSION, CompletionPipeRelay, CompletionWindow,
                                encode_completion_frame, window_header)
from .protocol import WEIGHT_UPDATE_PROTOCOL_VERSION, InitCommunicatorRequest, RequestConfig, UpdateWeightsRequest
from .persistent_session import PersistentRolloutSession, install_session_middleware
from .vllm_dummy_padding import bootstrap_worker_dummy_padding
from .vllm_routing_replay import bootstrap_worker_routing

# Worker extensions are imported before GPUModelRunner.load_model. This installs
# graph capture hooks early in spawned workers; individual engines still opt in.
bootstrap_worker_dummy_padding()
bootstrap_worker_routing()

try:
    from vllm.utils import get_open_port
    from trl.scripts.vllm_serve import chunk_list

except ImportError:
    pass
"""
This module defines the execution logic for `swift rollout`.
It adds weight synchronization logic based on `vLLMEngine`.

Usage:
    swift rollout \
        --model xxx \
        --vllm_tensor_parallel_size xxx \
        --vllm_data_parallel_size xxx \
        --vllm_use_async_engine true/false \
        --other_vllm_arguments

Note:
- Rollout is intended solely for GRPO training sampling.
- For inference or deployment, please use the `swift infer` or `swift deploy` commands.
"""

patch_vllm_load_adapter()


class WeightSyncWorkerExtension(HFWeightSyncWorkerExtension):

    def mopd_clear_encoder_cache(self, session_id=None):
        # Qwen's vision tower is trained too. Old image hidden states must not
        # survive a weight update even when their input image hash is unchanged.
        self.model_runner.encoder_cache.clear()
        return True

    def mopd_close_communicator(self):
        self.close_communicator()
        return True

    def mopd_reset_experiment(self):
        """Called only after the engine confirms all old requests have drained."""
        from vllm.model_executor import set_random_seed
        runner = self.model_runner
        for request_id in list(runner.input_batch.req_id_to_index):
            runner.input_batch.remove_request(request_id)
        runner.input_batch.condense()
        runner.input_batch.refresh_metadata()
        runner.requests.clear()
        runner.encoder_cache.clear()
        runner.input_batch.prev_sampled_token_ids = None
        runner.input_batch.prev_req_id_to_index = None
        runner.num_discarded_requests = 0
        capturer = getattr(runner, '_mopd_routing_capturer', None)
        if capturer is not None:
            capturer.clear()
        set_random_seed(self.model_config.seed)
        return True

    def mopd_routing_capabilities(self):
        capturer = getattr(self.model_runner, '_mopd_routing_capturer', None)
        if capturer is None:
            raise RuntimeError('This rollout worker did not initialize R3 route capture.')
        return capturer.capabilities()

    def mopd_pop_routing_traces(self, request_ids):
        capturer = getattr(self.model_runner, '_mopd_routing_capturer', None)
        if capturer is None:
            raise RuntimeError('This rollout worker has no R3 capture state.')
        traces = {} if capturer.tp_rank else capturer.store.pop(request_ids)
        for trace in traces.values():
            trace['capture'] = capturer.observations()
        return traces

    def mopd_clear_routing_traces(self):
        capturer = getattr(self.model_runner, '_mopd_routing_capturer', None)
        if capturer is None:
            raise RuntimeError('This rollout worker has no R3 capture state.')
        capturer.clear()
        return True

    def _prepare_weight_receive(self, prepare):
        """Agree on allocation success before any peer enters NCCL broadcast."""
        payload, error = None, None
        try:
            payload = prepare()
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        errors = self._comm.group.all_gather_obj(error)
        failures = [(rank, value) for rank, value in enumerate(errors) if value is not None]
        if failures:
            raise RuntimeError(f'Weight receive preparation failed: {failures}')
        return payload

    def _prepare_flattened_weight_receive(self, metadatas):
        def prepare():
            metadata = [FlattenedTensorMetadata(**item) for item in metadatas]
            dtype = getattr(torch, metadata[-1].dtype.split('.')[-1])
            tensor = torch.empty(metadata[-1].end_idx, dtype=dtype, device=self._comm.device)
            return metadata, tensor
        return self._prepare_weight_receive(prepare)

    def update_named_param(self, name: str, dtype: str, shape: Sequence[int]) -> None:
        """
        Receives updated weights from the client process and updates the named parameter in the model.

        Args:
            name (`str`):
                Name of the weight tensor being updated.
            dtype (`str`):
                Data type of the weight tensor as a string (e.g., `"torch.float32"`).
            shape (`Sequence[int]`):
                Shape of the weight tensor.
        """
        if self._comm is None:
            raise RuntimeError('Communicator not initialized. Call `init_communicator` first.')

        weight = self._prepare_weight_receive(
            lambda: torch.empty(shape, dtype=getattr(torch, dtype.split('.')[-1]), device=self._comm.device))

        # Use NCCL to broadcast the updated weights from the client (src) to all workers.
        self._comm.broadcast(weight, src=self.client_rank)
        self._comm.group.barrier()

        # Load the received weights into the model.
        self.model_runner.model.load_weights(weights=[(name, weight)])

    def update_adapter_flattened_param(self, lora_int_id: int, peft_config: Dict, metadatas: list[Dict]) -> None:
        """
        Receives and applies a flattened LoRA adapter to the model.
        """
        if self._comm is None:
            raise RuntimeError('Communicator not initialized. Call `init_communicator` first.')
        metadatas, flatten_tensor = self._prepare_flattened_weight_receive(metadatas)
        self._comm.broadcast(flatten_tensor, src=self.client_rank)
        self._comm.group.barrier()
        flattened_tensor_bucket = FlattenedTensorBucket(metadata=metadatas, flattened_tensor=flatten_tensor)
        named_params = flattened_tensor_bucket.reconstruct_tensors()
        lora_request = TensorLoRARequest(
            lora_name=f'{lora_int_id}',
            lora_int_id=lora_int_id,
            lora_path='dummy_lora_path',
            peft_config=peft_config,
            lora_tensors=named_params)
        self.add_lora(lora_request)

    def update_adapter_param(self, lora_int_id: int, peft_config: Dict, lora_tensors_metadata: list[Dict]) -> None:
        """
        Receives and applies a LoRA adapter to the model without flattening.
        Each tensor is broadcast individually.

        Args:
            lora_int_id: Integer ID for the LoRA adapter.
            peft_config: PEFT configuration dictionary.
            lora_tensors_metadata: List of metadata dictionaries for each tensor.
        """
        if self._comm is None:
            raise RuntimeError('Communicator not initialized. Call `init_communicator` first.')

        def prepare():
            return {metadata['name']: torch.empty(
                tuple(metadata['shape']), dtype=getattr(torch, metadata['dtype'].split('.')[-1]),
                device=self._comm.device) for metadata in lora_tensors_metadata}
        named_params = self._prepare_weight_receive(prepare)
        for tensor in named_params.values():
            self._comm.broadcast(tensor, src=self.client_rank)

        self._comm.group.barrier()

        lora_request = TensorLoRARequest(
            lora_name=f'{lora_int_id}',
            lora_int_id=lora_int_id,
            lora_path='dummy_lora_path',
            peft_config=peft_config,
            lora_tensors=named_params)
        self.add_lora(lora_request)

    def update_flattened_params(self, metadatas: list[Dict]) -> None:
        """
        Receives updated flattened weights from the client process and updates the model parameters.

        Args:
            metadatas (list[Dict]): List of metadata dictionaries for the flattened tensors.
        """
        if self._comm is None:
            raise RuntimeError('Communicator not initialized. Call `init_communicator` first.')
        metadatas, flatten_tensor = self._prepare_flattened_weight_receive(metadatas)

        self._comm.broadcast(flatten_tensor, src=self.client_rank)
        self._comm.group.barrier()

        flattened_tensor_bucket = FlattenedTensorBucket(metadata=metadatas, flattened_tensor=flatten_tensor)
        named_params = flattened_tensor_bucket.reconstruct_tensors()

        # Load the reconstructed parameters into the model
        self.model_runner.model.load_weights(weights=list(named_params.items()))

    @property
    def _comm(self):
        """
        Compatibility wrapper for communicator access across TRL versions.

        Returns the appropriate communicator attribute based on the TRL version:
        - trl < 0.24.0: self.pynccl_comm
        - trl >= 0.24.0: self.communicator
        """
        # Try new version first
        if hasattr(self, 'communicator'):
            return self.communicator
        # Fall back to old version
        elif hasattr(self, 'pynccl_comm'):
            return self.pynccl_comm
        else:
            return None


logger = get_logger()


def safe_set_start_method():
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method('spawn')


def _reset_prefix_cache_confirmed(rollout_engine, result, error) -> bool:
    """vLLM V1 (including 0.11) not only returns None from reset_prefix_cache —
    EngineCore also discards the scheduler's refusal bool, so a refused reset
    (KV blocks still held by live requests) is indistinguishable from success
    at the engine API. The refusal condition IS live requests, so confirm the
    engine is idle instead of trusting the void return. Fails closed.

    Reads the LOCAL output processor on purpose: this worker's requests are
    what hold its block pool, and engine.has_unfinished_requests() may
    aggregate over a vLLM DP group (a collective an errored peer never joins).
    Valid for configs without a KV-transfer connector (connector-retained
    blocks could refuse a reset while the output processor is already empty).
    """
    if error is not None or result is False:
        return False
    try:
        # Unwrap to the vLLM engine: a multi-turn RolloutScheduler wraps the
        # GRPOVllmEngine, which wraps the LLMEngine/AsyncLLM that actually
        # carries the idle state.
        engine = rollout_engine
        for _ in range(4):
            inner = getattr(engine, 'engine', None)
            if inner is None:
                break
            engine = inner
        output_processor = getattr(engine, 'output_processor', None)
        checker = (output_processor.has_unfinished_requests if output_processor is not None
                   else engine.has_unfinished_requests)
        return not checker()
    except Exception:
        logger.error(f'Could not confirm prefix-cache reset\n{traceback.format_exc()}')
        return False


def get_rollout_engine_type(args: RolloutArguments, engine: GRPOVllmEngine):
    if args.multi_turn_scheduler:
        if args.multi_turn_scheduler not in multi_turns:
            raise ValueError(f"Multi-turn scheduler '{args.multi_turn_scheduler}' not found in multi_turns.")
        scheduler_cls = multi_turns[args.multi_turn_scheduler]

        kwargs = {}
        if 'tokenizer' in list(inspect.signature(scheduler_cls.__init__).parameters):
            kwargs['tokenizer'] = engine.default_template.tokenizer
        # gym kwargs
        if args.use_gym_env:
            kwargs.update({
                'use_gym_env': args.use_gym_env,
                'gym_env': args.gym_env,
                'context_manager': args.context_manager,
            })

        rollout_engine: RolloutScheduler = scheduler_cls(infer_engine=engine, max_turns=args.max_turns, **kwargs)
        if not rollout_engine:
            raise ValueError(f"Failed to initialize multi-turn scheduler '{args.multi_turn_scheduler}'.")
    else:
        rollout_engine = engine
    return rollout_engine


def _all_true_worker_acks(value):
    if value is True:
        return True
    return isinstance(value, (list, tuple)) and bool(value) and all(_all_true_worker_acks(x) for x in value)


def llm_worker(args: RolloutArguments, data_parallel_rank: int, master_port: int, connection: Connection) -> None:
    # Set required environment variables for DP to work with vLLM
    args._import_external_plugins()
    args._init_custom_register()
    os.environ['VLLM_DP_RANK'] = str(data_parallel_rank)
    os.environ['VLLM_DP_RANK_LOCAL'] = str(data_parallel_rank)
    os.environ['VLLM_DP_SIZE'] = str(args.vllm_data_parallel_size)
    os.environ['VLLM_DP_MASTER_PORT'] = str(master_port)
    engine = SwiftRolloutDeploy.get_infer_engine(args, template=args.get_template(None))
    rollout_engine = get_rollout_engine_type(args, engine)
    # Send ready signal to parent process
    connection.send({'status': 'ready'})

    while True:
        # Wait for commands from the parent process
        try:
            command = connection.recv()
        except KeyboardInterrupt:
            engine.engine.collective_rpc(method='close_communicator')
            break

        # Handle commands
        if command['type'] == 'persistent_reset':
            try:
                result = engine.engine.reset_prefix_cache()
                if not _reset_prefix_cache_confirmed(rollout_engine, result, None):
                    raise RuntimeError('Cannot reset a persistent engine with unfinished requests')
                result = engine.engine.collective_rpc(method='mopd_reset_experiment', timeout=120)
                if not _all_true_worker_acks(result):
                    raise RuntimeError('A worker did not acknowledge its experiment reset')
                engine._mopd_generation_by_request.clear()
                connection.send(True)
            except Exception:
                connection.send({'error': traceback.format_exc()})
        elif command['type'] == 'completion_stream':
            sent = 0

            def completed(rows):
                nonlocal sent
                for index, output, timing in rows:
                    connection.send({'event': 'completion', 'index': index,
                                     'output': output.model_dump(mode='json'), 'timing': timing})
                    sent += 1

            try:
                rollout_engine.infer_completed(**command['kwargs'], completion_callback=completed)
                connection.send({'event': 'end', 'samples': sent})
            except Exception as error:
                connection.send({'event': 'error', 'error': f'{type(error).__name__}: {error}'})
                # A partially streamed/failed engine is not reusable. The
                # owned runtime must be restarted, not served as a short batch.
                raise
        elif command['type'] in ['call', 'fire_and_forget', 'weight_update']:
            method_name = command['method']
            args, kwargs = command.get('args', ()), command.get('kwargs', {})
            method = getattr(rollout_engine, method_name, None) or getattr(rollout_engine.engine, method_name, None)
            error = None
            try:
                result = method(*args, **kwargs)
            except Exception as exc:
                logger.error(f'Method execution failed: {method_name}\n{traceback.format_exc()}')
                error = f'{type(exc).__name__}: {exc}'
                result = None
            if command['type'] == 'call':
                if method_name == 'reset_prefix_cache':
                    result = _reset_prefix_cache_confirmed(rollout_engine, result, error)
                connection.send(result)
            elif command['type'] == 'weight_update':
                connection.send({'update_id': command['update_id'], 'error': error,
                                 'status': 'error' if error else 'loaded'})
        elif command['type'] == 'shutdown':
            break


async def async_llm_worker(args: RolloutArguments, data_parallel_rank: int, master_port: int,
                           connection: Connection) -> None:
    # Set required environment variables for DP to work with vLLM
    args._import_external_plugins()
    args._init_custom_register()
    engine = SwiftRolloutDeploy.get_infer_engine(args, template=args.get_template(None))

    rollout_engine = get_rollout_engine_type(args, engine)

    # Send ready signal to parent process
    connection.send({'status': 'ready'})

    loop = asyncio.get_running_loop()
    while True:
        try:
            command = await loop.run_in_executor(None, connection.recv)
        except KeyboardInterrupt:
            await engine.engine.collective_rpc(method='close_communicator')
            break

        # Handle commands
        if command['type'] in ['call', 'fire_and_forget', 'weight_update']:
            method_name = command['method']
            args, kwargs = command.get('args', ()), command.get('kwargs', {})
            method = getattr(rollout_engine, method_name, None) or getattr(rollout_engine.engine, method_name, None)
            error = None
            try:
                result = await method(*args, **kwargs)
            except Exception as exc:
                logger.error(f'Method execution failed: {method_name}\n{traceback.format_exc()}')
                error = f'{type(exc).__name__}: {exc}'
                result = None

            if command['type'] == 'call':
                if method_name == 'reset_prefix_cache':
                    result = _reset_prefix_cache_confirmed(rollout_engine, result, error)
                connection.send(result)
            elif command['type'] == 'weight_update':
                connection.send({'update_id': command['update_id'], 'error': error,
                                 'status': 'error' if error else 'loaded'})
        elif command['type'] == 'shutdown':
            break


def llm_worker_entry(*args, **kwargs):
    asyncio.run(async_llm_worker(*args, **kwargs))


class SwiftRolloutDeploy(SwiftPipeline):
    args_class = RolloutArguments
    args: args_class

    def _register_rl_rollout_app(self):
        self.app.get('/health/')(self.health)
        self.app.get('/get_world_size/')(self.get_world_size)
        self.app.post('/init_communicator/')(self.init_communicator)
        self.app.post('/update_named_param/')(self.update_named_param)
        self.app.post('/update_adapter_flattened_param/')(self.update_adapter_flattened_param)
        self.app.post('/update_adapter_param/')(self.update_adapter_param)
        self.app.post('/update_flattened_params/')(self.update_flattened_params)
        self.app.post('/finish_weight_update/')(self.finish_weight_update)
        self.app.post('/reset_prefix_cache/')(self.reset_prefix_cache)
        self.app.post('/close_communicator/')(self.close_communicator)
        self.app.post('/infer/', response_model=None)(self.infer)
        self.app.post('/infer_completed/', response_model=None)(self.infer_completed)
        self.app.post('/get_engine_type/')(self.get_engine_type)
        self.app.get('/session_status/')(self.session_status)
        self.app.post('/begin_session/')(self.begin_session)
        self.app.post('/end_session/')(self.end_session)

        install_session_middleware(self.app, self)

    def __init__(self, args: Optional[Union[List[str], RolloutArguments]] = None):
        super().__init__(args)
        self.use_gym_env = self.args.use_gym_env
        self.use_async_engine = self.args.vllm_use_async_engine
        self.num_connections = 1 if self.use_async_engine else self.args.vllm_data_parallel_size
        identity = os.environ.get('MOPD_VLLM_RUNTIME_IDENTITY')
        self._persistent_session = PersistentRolloutSession(identity) if identity else None
        self._persistent_http_lock = asyncio.Lock()
        if self._persistent_session is not None and (
                self.use_async_engine or self.use_gym_env or self.args.multi_turn_scheduler
                or self.args.vllm_enable_lora or self.args.vllm_mm_processor_cache_gb != 0):
            raise ValueError('Persistent rollout requires synchronous single-turn full-model serving and mm cache=0')
        safe_set_start_method()
        self.app = FastAPI(lifespan=self.lifespan)
        self._register_rl_rollout_app()
        self.master_port = get_open_port()
        self.connections = []
        self.processes = []
        self._weight_update_sequence = 0
        self._pending_weight_update = None
        self._weight_update_error = None
        # True from the first accepted bucket of a weight sync until the
        # trainer's cache reset commits the WHOLE new model version.
        self._weight_transaction_dirty = False
        self._completion_stream_active = False
        self._completion_stream_error = None
        self._start_data_parallel_workers()

    def _start_data_parallel_workers(self):
        for data_parallel_rank in range(self.num_connections):
            parent_conn, child_conn = Pipe()
            worker_func = llm_worker_entry if self.use_async_engine else llm_worker
            process = Process(target=worker_func, args=(self.args, data_parallel_rank, self.master_port, child_conn))
            process.start()
            self.connections.append(parent_conn)
            self.processes.append(process)

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        # Wait for all workers to send "ready"
        ready_connections = set()

        while len(ready_connections) < self.num_connections:
            for connection in self.connections:
                msg = connection.recv()
                if isinstance(msg, dict) and msg.get('status') == 'ready':
                    ready_connections.add(connection)

        yield

        # Wait for processes to terminate
        for process in self.processes:
            process.join(timeout=10)  # Wait for 10 seconds for the process to terminate
            if process.is_alive():
                logger.warning(f'Process {process} is still alive after 10 seconds, attempting to terminate...')
                process.terminate()
                process.join()  # ensure process termination after calling terminate()

    @staticmethod
    def get_infer_engine(args: RolloutArguments, template=None, **kwargs):
        kwargs.update({
            'model_id_or_path': args.model,
            'model_type': args.model_type,
            'revision': args.model_revision,
            'torch_dtype': args.torch_dtype,
            'template': template,
            'use_async_engine': args.vllm_use_async_engine,
            'max_lora_rank': args.vllm_max_lora_rank,
            'return_routed_experts': args.vllm_return_routed_experts,
        })
        infer_backend = kwargs.pop('infer_backend', None) or args.infer_backend
        if infer_backend != 'vllm':
            infer_backend = 'vllm'
            logger.info('Currently, rollout only supports the vLLM backend. Set vLLM backend')
        kwargs.update(args.get_vllm_engine_kwargs())
        kwargs.update({'enable_lora': args.vllm_enable_lora})  # override
        # Important: Use processed_logprobs so temperature scaling affects the logprobs
        # This is required for correct importance sampling in rollout correction
        kwargs['logprobs_mode'] = 'processed_logprobs' if check_vllm_version_ge('0.10.2') else None
        # used for RL external rollout backend
        engine_kwargs = kwargs.get('engine_kwargs', {})
        # for RL rollout model weight sync
        engine_kwargs.update({'worker_extension_cls': 'swift.llm.infer.rollout.WeightSyncWorkerExtension'})
        if (os.environ.get('MOPD_VLLM_RUNTIME_IDENTITY')
                or os.environ.get('MOPD_RESET_ENCODER_CACHE') == '1'):
            scheduler = 'swift.llm.infer.persistent_scheduler.PersistentMOPDScheduler'
            if engine_kwargs.get('scheduler_cls', scheduler) != scheduler:
                raise ValueError('Persistent rollout requires its paired KV/encoder-cache scheduler reset')
            engine_kwargs['scheduler_cls'] = scheduler
        # Use load_format from engine_kwargs if provided, otherwise default to 'dummy'
        if 'load_format' not in engine_kwargs:
            engine_kwargs['load_format'] = 'dummy'
        if args.vllm_use_async_engine and args.vllm_data_parallel_size > 1:
            engine_kwargs['data_parallel_size'] = args.vllm_data_parallel_size
        kwargs['engine_kwargs'] = engine_kwargs

        return GRPOVllmEngine(**kwargs)

    async def health(self):
        """
        Health check endpoint to verify that the server is running.
        """
        return {'status': 'ok'}

    async def session_status(self):
        session = getattr(self, '_persistent_session', None)
        if session is None:
            raise HTTPException(status_code=404, detail='Persistent rollout is disabled')
        return {**session.status(), 'pid': os.getpid(),
                'failed': self._weight_update_error is not None or self._completion_stream_error is not None,
                'busy': self._completion_stream_active or self._pending_weight_update is not None,
                'dirty': self._weight_transaction_dirty}

    def _persistent_call(self, command):
        for connection in self.connections:
            connection.send(command)
        deadline = time.monotonic() + float(os.environ.get('SWIFT_WEIGHT_UPDATE_ACK_TIMEOUT', '120'))
        for rank, connection in enumerate(self.connections):
            reply = connection.recv() if connection.poll(max(0, deadline - time.monotonic())) else None
            if not _all_true_worker_acks(reply):
                self._weight_update_error = f'Persistent rollout reset/close failed on worker {rank}: {reply!r}'
                raise HTTPException(status_code=503, detail=self._weight_update_error)

    async def begin_session(self, request: Dict):
        session = getattr(self, '_persistent_session', None)
        if session is None:
            raise HTTPException(status_code=404, detail='Persistent rollout is disabled')
        self._require_loaded_weights()
        if self._weight_transaction_dirty:
            raise HTTPException(status_code=409, detail='Previous weight transaction is incomplete')
        try:
            session.begin(request.get('session_id'), request.get('identity'))
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        # Keep serving closed until the new trainer sends its COMPLETE initial
        # policy and successfully resets KV/R3 caches. Reset alone cannot unlock.
        self._persistent_call({'type': 'persistent_reset'})
        return session.status()

    async def end_session(self):
        session = getattr(self, '_persistent_session', None)
        if session is None or session.session_id is None:
            raise HTTPException(status_code=409, detail='No active persistent experiment')
        self._require_loaded_weights()
        if self._weight_transaction_dirty:
            raise HTTPException(status_code=409, detail='Cannot release an incomplete weight transaction')
        session.ready = False
        self._persistent_call({'type': 'persistent_reset'})
        self._persistent_call({'type': 'call', 'method': 'collective_rpc',
                               'kwargs': {'method': 'mopd_close_communicator', 'timeout': 120}})
        session.end()
        return session.status()

    async def get_world_size(self):
        """
        Retrieves the world size from the LLM engine.

        Returns:
            `dict`:
                A dictionary containing the world size.

        Example response:
        ```json
        {"world_size": 8}
        ```
        """
        return {
            'world_size': self.args.vllm_tensor_parallel_size * self.args.vllm_data_parallel_size,
            'weight_update_protocol': WEIGHT_UPDATE_PROTOCOL_VERSION,
        }

    async def init_communicator(self, request: InitCommunicatorRequest):
        """
        Initializes the communicator for synchronizing model weights between a client and multiple server
        workers.

        Args:
            request (`InitCommunicatorRequest`):
                - `host` (`str`): Hostname or IP address of the master node.
                - `port` (`int`): Port number to be used for communication.
                - `world_size` (`int`): Total number of participating processes in the group.
        """
        self._require_completion_idle()
        if request.weight_update_protocol != WEIGHT_UPDATE_PROTOCOL_VERSION:
            raise HTTPException(status_code=409, detail='Trainer/server weight-update protocol mismatch; '
                                'restart both from the same ms-swift revision.')
        world_size = self.args.vllm_tensor_parallel_size * self.args.vllm_data_parallel_size + 1

        # The function init_communicator is called this way: init_communicator(host, port, world_size)
        # So with collective_rpc we need to call it this way:
        # llm.collective_rpc(method="init_communicator", args=(host, port, world_size))
        kwargs = {
            'method':
            'init_communicator',
            'args': (request.host, request.port, world_size, *(() if request.client_device_uuid is None else
                                                               (request.client_device_uuid, )))
        }
        for connection in self.connections:
            connection.send({'type': 'fire_and_forget', 'method': 'collective_rpc', 'kwargs': kwargs})

        return {'message': 'Request received, initializing communicator'}

    async def update_named_param(self, request: UpdateWeightsRequest):
        """
        Updates the model weights with the provided tensor.

        Once this endpoint is called, the client process should broadcast the updated weights to all server workers.

        Args:
            request (`UpdateWeightsRequest`):
                - `name` (`str`): Name of the weight tensor being updated.
                - `dtype` (`str`): Data type of the weight tensor (e.g., `"torch.float32"`).
                - `shape` (list of `int`): Shape of the weight

        """
        # The function update_named_param is called this way: update_named_param("name", torch.float32, (10, 10))
        # So with collective_rpc we need to call it this way:
        # llm.collective_rpc("update_named_param", args=("name", torch.float32, (10, 10)))
        kwargs = {'method': 'update_named_param', 'args': (request.name, request.dtype, tuple(request.shape))}
        return self._begin_weight_update(kwargs)

    async def update_adapter_flattened_param(self, request: UpdateFlattenedAdapterRequest):
        peft_config = asdict(request.peft_config)
        metadatas = [
            metadata.model_dump() if hasattr(metadata, 'model_dump') else metadata.dict()
            for metadata in request.metadatas
        ]
        kwargs = {'method': 'update_adapter_flattened_param', 'args': (request.lora_int_id, peft_config, metadatas)}
        return self._begin_weight_update(kwargs)

    async def update_adapter_param(self, request: UpdateAdapterRequest):
        """
        Updates the LoRA adapter weights without flattening.
        Each tensor is broadcast individually.

        Args:
            request (UpdateAdapterRequest):
                - lora_int_id (int): Integer ID for the LoRA adapter.
                - peft_config (LoraConfig): PEFT configuration for the adapter.
                - lora_tensors_metadata (List[FlattenedTensorMetadata]): Metadata for each tensor.
        """
        peft_config = asdict(request.peft_config)
        lora_tensors_metadata = [
            metadata.model_dump() if hasattr(metadata, 'model_dump') else metadata.dict()
            for metadata in request.lora_tensors_metadata
        ]
        kwargs = {'method': 'update_adapter_param', 'args': (request.lora_int_id, peft_config, lora_tensors_metadata)}
        return self._begin_weight_update(kwargs)

    async def update_flattened_params(self, request: UpdateFlattenedParamsRequest):
        """
        Updates the model weights with flattened tensor data.

        Args:
            request (UpdateFlattenedParamsRequest):
                - metadatas (List[FlattenedTensorMetadata]): Metadata for the flattened tensors.

        """
        metadatas = [
            metadata.model_dump() if hasattr(metadata, 'model_dump') else metadata.dict()
            for metadata in request.metadatas
        ]
        kwargs = {'method': 'update_flattened_params', 'args': (metadatas, )}
        return self._begin_weight_update(kwargs)

    def _require_loaded_weights(self):
        self._require_completion_idle()
        if self._weight_update_error is not None:
            raise HTTPException(status_code=503, detail=self._weight_update_error)
        if self._pending_weight_update is not None:
            raise HTTPException(status_code=409, detail='A weight update has not been acknowledged yet.')

    def _require_completion_idle(self):
        if getattr(self, '_completion_stream_error', None) is not None:
            raise HTTPException(status_code=503, detail=self._completion_stream_error)
        if getattr(self, '_completion_stream_active', False):
            raise HTTPException(status_code=409, detail='A complete-response window is still using the worker pipes.')

    def _begin_weight_update(self, kwargs):
        self._require_loaded_weights()
        self._weight_update_sequence += 1
        update_id = self._weight_update_sequence
        self._pending_weight_update = update_id
        self._weight_transaction_dirty = True
        session = getattr(self, '_persistent_session', None)
        if session is not None:
            session.received_weights = True
        for connection in self.connections:
            connection.send({'type': 'weight_update', 'method': 'collective_rpc',
                             'update_id': update_id, 'kwargs': kwargs})
        # The client must receive acceptance BEFORE broadcasting. Actual loading
        # finishes later; a second endpoint consumes every worker's load ACK.
        return {'status': 'accepted', 'update_id': update_id}

    async def finish_weight_update(self, request: Dict):
        update_id = request.get('update_id')
        if self._pending_weight_update is None or update_id != self._pending_weight_update:
            raise HTTPException(status_code=409, detail='Weight-update acknowledgment ID mismatch.')
        deadline = time.monotonic() + float(os.environ.get('SWIFT_WEIGHT_UPDATE_ACK_TIMEOUT', '120'))
        failures = []
        for rank, connection in enumerate(self.connections):
            if not connection.poll(max(0., deadline - time.monotonic())):
                failures.append(f'worker {rank}: timed out waiting for model load')
                continue
            result = connection.recv()
            if (not isinstance(result, dict) or result.get('update_id') != update_id
                    or result.get('status') != 'loaded'):
                failures.append(f'worker {rank}: {result!r}')
        self._pending_weight_update = None
        if failures:
            # A partially loaded model must not serve further generations.
            self._weight_update_error = f'Weight update {update_id} failed: ' + '; '.join(failures)
            raise HTTPException(status_code=500, detail=self._weight_update_error)
        return {'status': 'loaded', 'update_id': update_id, 'workers': len(self.connections)}

    async def reset_prefix_cache(self):
        """
        Resets the prefix cache for the model.

        Committing point of a weight-sync transaction: only a confirmed reset
        on every worker reopens generation after a bucketed weight update.
        """
        self._require_loaded_weights()
        for connection in self.connections:
            connection.send({'type': 'call', 'method': 'reset_prefix_cache'})
        # Wait for and collect all results, bounded per worker
        deadline = time.monotonic() + float(os.environ.get('SWIFT_WEIGHT_UPDATE_ACK_TIMEOUT', '120'))
        all_outputs = []
        for rank, connection in enumerate(self.connections):
            if not connection.poll(max(0., deadline - time.monotonic())):
                all_outputs.append(f'worker {rank}: timed out waiting for prefix-cache reset')
                continue
            all_outputs.append(connection.recv())
        success = all(output is True for output in all_outputs)
        if not success:
            # An unconfirmed reset can serve stale-KV completions for updated
            # weights; latch the failure so generation stays blocked.
            self._weight_update_error = f'Prefix-cache reset failed: {all_outputs!r}'
            raise HTTPException(status_code=500, detail=self._weight_update_error)
        if self.args.vllm_return_routed_experts:
            for connection in self.connections:
                connection.send({
                    'type': 'call', 'method': 'collective_rpc',
                    'kwargs': {'method': 'mopd_clear_routing_traces', 'timeout': 120}})
            deadline = time.monotonic() + float(os.environ.get('SWIFT_WEIGHT_UPDATE_ACK_TIMEOUT', '120'))
            for rank, connection in enumerate(self.connections):
                reply = connection.recv() if connection.poll(max(0., deadline - time.monotonic())) else None
                if not _all_true_worker_acks(reply):
                    self._weight_update_error = f'R3 routing-cache reset failed on worker {rank}.'
                    raise HTTPException(status_code=500, detail=self._weight_update_error)
        session = getattr(self, '_persistent_session', None)
        if session is not None or os.environ.get('MOPD_RESET_ENCODER_CACHE') == '1':
            # Repeated images also occur inside a nonpersistent experiment.
            # Pair the scheduler reset above with worker hidden-state clearing
            # before committing the new policy; mm_processor_cache_gb=0 alone
            # does not invalidate these weight-dependent vision outputs.
            self._persistent_call({'type': 'call', 'method': 'collective_rpc',
                                   'kwargs': {'method': 'mopd_clear_encoder_cache',
                                              'args': (None if session is None else session.session_id,),
                                              'timeout': 120}})
        if session is not None:
            session.ready = session.received_weights
        self._weight_transaction_dirty = False
        return {'success': True}

    async def get_engine_type(self):
        """
        Return a dictionary describing the runtime engine configuration.

        The returned object contains three keys:
        - engine_type (str): Either 'AsyncLLMEngine' or 'LLMEngine', indicating
        whether the asynchronous or synchronous engine is in use.
        - use_gym_env (bool, optional): Present and True **only when**
        ``use_async_engine`` and ``use_gym_env`` are both True.
        - enable_multi_turn (bool): True if multi-turn scheduling is enabled
        via ``args.multi_turn_scheduler``, otherwise False.

        Returns
        -------
        dict
            A concise specification of the current engine setup.
        """
        enable_multi_turn = False
        if self.args.multi_turn_scheduler:
            enable_multi_turn = True
        use_gym_env = False
        if self.use_async_engine and self.use_gym_env:
            use_gym_env = True
        engine_type = 'AsyncLLMEngine' if self.use_async_engine else 'LLMEngine'
        enable_lora = self.args.vllm_enable_lora
        result = {
            'engine_type': engine_type,
            'enable_multi_turn': enable_multi_turn,
            'use_gym_env': use_gym_env,
            'enable_lora': enable_lora,
            'mopd_generation_version': 1,
        }
        if self.args.vllm_return_routed_experts:
            result['routing_replay_version'] = 1
        if not (self.use_async_engine or enable_multi_turn or self.use_gym_env or enable_lora):
            result['completion_stream_version'] = COMPLETION_STREAM_VERSION
        return result

    async def close_communicator(self):
        """
        Closes the weight update group and cleans up associated resources.
        """
        self._require_completion_idle()
        if getattr(self, '_persistent_session', None) is not None:
            raise HTTPException(status_code=409, detail='Persistent clients must use end_session')
        kwargs = {'method': 'close_communicator'}
        for connection in self.connections:
            connection.send({'type': 'fire_and_forget', 'method': 'collective_rpc', 'kwargs': kwargs})
        return {'message': 'Request received, closing communicator'}

    async def infer_completed(self, request: Dict):
        self._require_loaded_weights()
        session = getattr(self, '_persistent_session', None)
        if session is not None and not session.ready:
            raise HTTPException(status_code=409, detail='New experiment initial weights are not committed')
        if (self._weight_transaction_dirty or self.use_async_engine or self.use_gym_env
                or self.args.multi_turn_scheduler or self.args.vllm_enable_lora):
            raise HTTPException(status_code=409, detail='Complete-response delivery needs committed weights '
                                'and the synchronous, single-turn, full-model engine.')
        window = CompletionWindow(**request['window'], occurrence_ids=tuple(request['occurrence_ids']))
        requests = request['infer_requests']
        if len(requests) != len(window.occurrence_ids):
            raise HTTPException(status_code=400, detail='Completion request/occurrence cardinality differs.')
        config = RequestConfig(**request['request_config'])
        if config.n != 1 or config.stream or not config.return_details:
            raise HTTPException(status_code=400, detail='Complete responses require n=1 and full token details.')
        request_groups = chunk_list(requests, self.num_connections)
        occurrence_groups = chunk_list(list(window.occurrence_ids), self.num_connections)
        relay = CompletionPipeRelay(
            self.connections, occurrence_groups, window_header(window),
            timeout_seconds=request['timeout_seconds'])
        self._completion_stream_active = True
        try:
            for rank, (connection, group) in enumerate(zip(self.connections, request_groups)):
                effective = group or [RolloutInferRequest(messages=[{'role': 'user', 'content': '<placeholder>'}])]
                # Match the existing synchronous server's seed partitioning
                # without mutating the caller's configuration object.
                if config.seed:
                    config.seed += rank * len(effective)
                connection.send({'type': 'completion_stream', 'kwargs': {
                    'infer_requests': effective, 'request_config': config,
                }})
        except BaseException:
            self._completion_stream_error = 'Complete-response dispatch failed; restart the owned runtime.'
            self._completion_stream_active = False
            raise

        def body():
            try:
                yield from relay
            except Exception as error:
                self._completion_stream_error = (
                    f'Complete-response window failed; restart the owned runtime: {type(error).__name__}: {error}')
                yield encode_completion_frame({'event': 'error', 'window': relay.header,
                                               'error': self._completion_stream_error})
            finally:
                if not relay.complete:
                    self._completion_stream_error = 'Complete-response stream was interrupted; restart the owned runtime.'
                self._completion_stream_active = False

        return StreamingResponse(body(), media_type='application/x-ndjson')

    async def infer(
        self,
        infer_requests: List[Union[Dict, RolloutInferRequest]],
        request_config: Optional[RequestConfig] = None,
        *,
        use_tqdm: Optional[bool] = None,
    ):
        self._require_loaded_weights()
        session = getattr(self, '_persistent_session', None)
        if session is not None and not session.ready:
            raise HTTPException(status_code=409, detail='New experiment initial weights are not committed')
        if self._weight_transaction_dirty:
            # Buckets have been applied but the cache reset has not committed
            # the new model version; a partially updated model must not serve.
            raise HTTPException(status_code=409, detail='A weight-sync transaction is still open '
                                '(bucket updates accepted, cache reset not confirmed).')
        chunked_infer_requests = chunk_list(infer_requests, self.num_connections)

        # Send the prompts to each worker
        for i, (connection, requests) in enumerate(zip(self.connections, chunked_infer_requests)):
            # When the number of prompts is less than data_parallel_size, some workers will receive empty prompts.
            # However, vLLM requires that we always send at least one prompt. So we send a placeholder prompt to comply
            # with vLLM's requirement, and we later ignore the result.
            if not requests:
                requests = [RolloutInferRequest(messages=[{'role': 'user', 'content': '<placeholder>'}])]
            # different seed bewteen vLLM Engine
            if request_config.seed:
                request_config.seed += i * len(requests)
            kwargs = {'infer_requests': requests, 'request_config': request_config, 'use_tqdm': use_tqdm}
            method = 'infer' if not self.use_async_engine else 'async_infer'
            connection.send({'type': 'call', 'method': method, 'kwargs': kwargs})

        all_outputs = [connection.recv() for connection in self.connections]
        # Handle empty prompts (see above)
        all_outputs = [output for output, requests in zip(all_outputs, chunked_infer_requests) if requests]
        all_outputs = list(chain.from_iterable(all_outputs))  # from list of list to single list

        return all_outputs

    def run(self):
        args = self.args
        uvicorn.run(self.app, host=args.host, port=args.port, log_level=args.log_level)


def rollout_main(args: Optional[Union[List[str], RolloutArguments]] = None) -> None:
    SwiftRolloutDeploy(args).main()


def is_accessible(port: int):
    infer_client = InferClient(port=port)
    try:
        infer_client.get_model_list()
    except ClientConnectorError:
        return False
    return True


@contextmanager
def run_rollout(args: RolloutArguments, return_url: bool = False):
    if isinstance(args, RolloutArguments) and args.__class__.__name__ == 'RolloutArguments':
        deploy_args = args
    else:
        args_dict = asdict(args)
        parameters = inspect.signature(RolloutArguments).parameters
        for k in list(args_dict.keys()):
            if k not in parameters or args_dict[k] is None:
                args_dict.pop(k)
        deploy_args = RolloutArguments(**args_dict)

    mp = multiprocessing.get_context('spawn')
    process = mp.Process(target=rollout_main, args=(deploy_args, ))
    process.start()
    try:
        while not is_accessible(deploy_args.port):
            time.sleep(1)
        yield f'http://127.0.0.1:{deploy_args.port}/v1' if return_url else deploy_args.port
    finally:
        process.terminate()
        logger.info('The deployment process has been terminated.')
