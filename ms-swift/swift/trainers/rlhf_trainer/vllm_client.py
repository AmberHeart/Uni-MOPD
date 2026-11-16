import atexit
import logging
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import List, Optional, Union
from urllib.parse import urlparse

import json
import requests
import torch
from packaging import version
from pydantic import BaseModel, ValidationError
from requests import ConnectionError
from torch import nn
from transformers.utils import is_torch_cuda_available

from swift.llm import AdapterRequest, RolloutInferRequest, Template
from swift.llm.infer.protocol import WEIGHT_UPDATE_PROTOCOL_VERSION, ChatCompletionResponse, RequestConfig, RolloutOutput
from swift.llm.infer.completion_stream import COMPLETION_STREAM_VERSION, CompletionProducer, CompletionQueue
from swift.llm.infer.routing_protocol import ROUTING_PROTOCOL_VERSION
from swift.llm.infer.persistent_session import SESSION_HEADER, SESSION_PROTOCOL_VERSION
from swift.plugin import Metric
from swift.utils import is_trl_available, is_vllm_ascend_available, is_vllm_available
from .utils import format_host_for_url, is_valid_ipv6_address, peft_config_to_dict, resolve_hostname

if is_vllm_available():
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    if is_vllm_ascend_available():
        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator as PyNcclCommunicator  # noqa

if is_trl_available():
    import trl
    trl_verison = version.parse(trl.__version__)

logger = logging.getLogger(__name__)

# Weight-sync control-plane requests either answer quickly (accept/dispatch)
# or are themselves deadline-bound on the server. A stalled endpoint must
# surface as an error here instead of hanging the coordinator forever while
# other train ranks wait in collectives. Bulk /infer/ traffic is NOT covered.
CONTROL_PLANE_TIMEOUT = float(os.environ.get('SWIFT_WEIGHT_CONTROL_TIMEOUT', '150'))


class VLLMClient:

    def __init__(self,
                 base_urls: Optional[List[str]] = None,
                 hosts: List[str] = ['0.0.0.0'],
                 server_ports: List[int] = [8000],
                 group_ports: Optional[Union[int, List[int]]] = None,
                 connection_timeout: float = 240.0,
                 parallel_weight_sync: bool = False,
                 persistent_session_id: Optional[str] = None,
                 runtime_identity: Optional[str] = None):
        if not is_vllm_available():
            raise ImportError('vLLM is not installed. Please install it with `pip install vllm`.')

        if base_urls is not None:
            self.base_urls = []
            self.hosts = []
            for url in base_urls:
                parsed_url = urlparse(url)
                # Use resolve_hostname instead of gethostbyname for IPv6 support
                host = resolve_hostname(parsed_url.hostname)
                scheme = parsed_url.scheme or 'http'
                base_url_i = f'{scheme}://{parsed_url.netloc}{parsed_url.path}'
                self.base_urls.append(base_url_i)
                self.hosts.append(host)
        else:
            if len(hosts) != len(server_ports):
                raise ValueError('host and server_port must have same length when lists are provided')
            # Format IPv6 addresses correctly in URLs (wrap with brackets)
            self.base_urls = [f'http://{format_host_for_url(h)}:{p}' for h, p in zip(hosts, server_ports)]
            self.hosts = hosts

        self.num_servers = len(self.base_urls)

        if group_ports is None:
            group_ports = [51216 + i for i in range(self.num_servers)]

        self.sessions = [requests.Session() for _ in range(self.num_servers)]

        if isinstance(group_ports, int):
            self.group_ports = [group_ports + i for i in range(self.num_servers)]
        elif isinstance(group_ports, list) and len(group_ports) == self.num_servers:
            self.group_ports = group_ports
        else:
            raise ValueError('group_port must be int or list of length num_servers')

        self.pynccl_comms = []
        self.parallel_weight_sync = parallel_weight_sync
        self._weight_sync_streams = None
        self._routing_replay_checked = False
        self._domain_generation_checked = False
        self.persistent_session_id = persistent_session_id
        self.runtime_identity = runtime_identity
        self._persistent_closed = False
        if bool(persistent_session_id) != bool(runtime_identity):
            raise ValueError('Persistent rollout requires both session and runtime identity')
        self.check_server(connection_timeout)
        if persistent_session_id:
            self.begin_persistent_session()

    def begin_persistent_session(self):
        for session in self.sessions:
            session.headers[SESSION_HEADER] = self.persistent_session_id
        # Verify the entire fleet before claiming any server. A partially
        # failed claim is not rolled back into a false "clean idle" state.
        for session, base_url in zip(self.sessions, self.base_urls):
            response = session.get(f'{base_url}/session_status/', timeout=CONTROL_PLANE_TIMEOUT)
            info = response.json() if response.status_code == 200 else {}
            if (info.get('session_protocol') != SESSION_PROTOCOL_VERSION
                    or info.get('identity') != self.runtime_identity or info.get('state') != 'idle'
                    or info.get('failed') or info.get('busy') or info.get('dirty')):
                raise RuntimeError(f'Persistent rollout is incompatible, busy or failed at {base_url}: {info}')
        for session, base_url in zip(self.sessions, self.base_urls):
            response = session.post(
                f'{base_url}/begin_session/',
                json={'session_id': self.persistent_session_id, 'identity': self.runtime_identity},
                timeout=CONTROL_PLANE_TIMEOUT)
            info = response.json()
            if (response.status_code != 200 or info.get('session_id') != self.persistent_session_id
                    or info.get('ready') is not False):
                raise RuntimeError(f'Failed to reset persistent rollout at {base_url}: {info}')

    def require_domain_generation(self):
        """Reject old servers that would silently ignore per-domain budgets."""
        if self._domain_generation_checked:
            return
        for session, base_url in zip(self.sessions, self.base_urls):
            response = session.post(f'{base_url}/get_engine_type/', timeout=CONTROL_PLANE_TIMEOUT)
            if response.status_code != 200:
                raise RuntimeError(f'Cannot verify per-domain generation support at {base_url}.')
            info = response.json()
            if (info.get('mopd_generation_version') != 1 or info.get('enable_multi_turn')
                    or info.get('use_gym_env')):
                raise RuntimeError(
                    f'Per-domain MOPD generation needs matching single-turn server code at {base_url}.')
        self._domain_generation_checked = True

    def require_routing_replay(self):
        """Require the opt-in protocol on EVERY server before an R3 rollout."""
        if self._routing_replay_checked:
            return
        for session, base_url in zip(self.sessions, self.base_urls):
            response = session.post(f'{base_url}/get_engine_type/', timeout=CONTROL_PLANE_TIMEOUT)
            if response.status_code != 200 or response.json().get('routing_replay_version') != ROUTING_PROTOCOL_VERSION:
                raise RuntimeError(
                    f'R3 requires a matching route-enabled rollout server at {base_url}; '
                    'start it with --vllm_return_routed_experts true.')
        self._routing_replay_checked = True

    def check_server(self, total_timeout: float = 0.0, retry_interval: float = 2.0):
        server_status = [False] * self.num_servers

        def check_single_server(i):
            start_time = time.time()
            url = f'{self.base_urls[i]}/health/'
            while True:
                try:
                    response = requests.get(url, timeout=retry_interval)
                    if response.status_code == 200:
                        server_status[i] = True
                        return
                except Exception:
                    pass

                elapsed = time.time() - start_time
                if elapsed >= total_timeout:
                    return

                time.sleep(retry_interval)

        threads = []
        for i in range(self.num_servers):
            t = threading.Thread(target=check_single_server, args=(i, ))
            t.daemon = True
            t.start()
            threads.append(t)

        for t in threads:
            t.join(total_timeout)

        if not all(server_status):
            failed_servers = [self.base_urls[i] for i, status in enumerate(server_status) if not status]
            raise ConnectionError(f'Servers not reachable after {total_timeout}s: {failed_servers}')

    def infer(
        self,
        infer_requests: List[RolloutInferRequest],
        request_config: Optional[RequestConfig] = None,
        metrics: Optional[List[Metric]] = None,
        *,
        template: Optional[Template] = None,
        use_tqdm: Optional[bool] = None,
        adapter_request: Optional[AdapterRequest] = None,
    ):
        if not hasattr(self, 'use_async_engine') or not hasattr(self, 'use_gym_env'):
            self.get_engine_type()

        n = len(infer_requests)
        chunk_size = (n + self.num_servers - 1) // self.num_servers
        chunks = [infer_requests[i:i + chunk_size] for i in range(0, n, chunk_size)]
        chunks += [[]] * (self.num_servers - len(chunks))

        results = [None] * self.num_servers
        errors = [None] * self.num_servers
        if isinstance(request_config, RequestConfig):
            request_config = asdict(request_config)

        def process_chunk(i, chunk):
            try:
                if len(chunk) > 0 and isinstance(chunk[0], RolloutInferRequest):
                    chunk = [asdict(req) for req in chunk]

                response = self.sessions[i].post(
                    f'{self.base_urls[i]}/infer/',
                    json={
                        'infer_requests': chunk,
                        'request_config': request_config,
                        'metrics': metrics,
                        'template': template,
                        'use_tqdm': use_tqdm,
                        'adapter_request': adapter_request,
                    },
                )

                if response.status_code != 200:
                    errors[i] = Exception(f'Server {i} failed: {response.status_code}, {response.text}')
                    return

                resp_data = response.json()
                parsed: List[Union[RolloutOutput, ChatCompletionResponse]] = []

                for item in resp_data:
                    try:
                        parsed.append(RolloutOutput.model_validate(item))
                    except ValidationError:
                        parsed.append(ChatCompletionResponse(**item))
                results[i] = parsed
            except Exception as e:
                errors[i] = e

        with ThreadPoolExecutor(max_workers=self.num_servers) as executor:
            futures = [executor.submit(process_chunk, i, chunk) for i, chunk in enumerate(chunks)]
            for future in futures:
                future.result()

        all_errors = [e for e in errors if e is not None]
        if all_errors:
            raise RuntimeError(f'Multiple errors: {all_errors}')

        return [res for server_results in results for res in server_results]

    def begin_completion_window(self, infer_requests, request_config, window, *,
                                max_samples, max_tokens, max_bytes, timeout_seconds):
        """Start CPU-only readers after the trainer has synchronized this policy."""
        for session, base_url in zip(self.sessions, self.base_urls):
            response = session.post(f'{base_url}/get_engine_type/', timeout=CONTROL_PLANE_TIMEOUT)
            info = response.json() if response.status_code == 200 else {}
            if (info.get('completion_stream_version') != COMPLETION_STREAM_VERSION
                    or info.get('engine_type') != 'LLMEngine' or info.get('enable_multi_turn')
                    or info.get('use_gym_env') or info.get('enable_lora')):
                raise RuntimeError('Complete-response MOPD needs matching synchronous single-turn server code.')
        requests = [asdict(value) if isinstance(value, RolloutInferRequest) else value for value in infer_requests]
        config = asdict(request_config) if isinstance(request_config, RequestConfig) else request_config
        queue = CompletionQueue(window, max_samples=max_samples, max_tokens=max_tokens,
                                max_bytes=max_bytes, timeout_seconds=timeout_seconds)
        return CompletionProducer(self.sessions, self.base_urls, requests, config, queue)

    def init_communicator(self, device: Union[int, str] = 0):
        self.pynccl_comms = []
        old_nccl_net = os.environ.get('NCCL_NET')
        requested_nccl_net = os.environ.get('MOPD_WEIGHT_SYNC_NCCL_NET')
        if requested_nccl_net:
            os.environ['NCCL_NET'] = requested_nccl_net
        try:
            self._init_communicator_inner(device)
        finally:
            if requested_nccl_net:
                if old_nccl_net is None:
                    os.environ.pop('NCCL_NET', None)
                else:
                    os.environ['NCCL_NET'] = old_nccl_net

    def _init_communicator_inner(self, device: Union[int, str] = 0):
        self.pynccl_comms = []
        self._weight_sync_streams = None
        for i in range(self.num_servers):
            response = self.sessions[i].get(f'{self.base_urls[i]}/get_world_size/', timeout=CONTROL_PLANE_TIMEOUT)
            if response.status_code != 200:
                raise Exception(f'Server {i} failed: {response.text}')
            payload = response.json()
            if payload.get('weight_update_protocol') != WEIGHT_UPDATE_PROTOCOL_VERSION:
                raise RuntimeError(f'Server {i} has an incompatible weight-update protocol; '
                                   'restart server and trainer from the same ms-swift revision.')
            vllm_world_size = payload['world_size']

            world_size = vllm_world_size + 1
            rank = vllm_world_size
            kwargs = {}
            if trl_verison >= version.parse('0.20.0'):
                try:
                    client_device_uuid = str(torch.cuda.get_device_properties(device).uuid)
                except Exception:
                    client_device_uuid = '42'
                kwargs['client_device_uuid'] = client_device_uuid

            # Use '::' for IPv6 hosts, '0.0.0.0' for IPv4 hosts
            bind_host = '::' if is_valid_ipv6_address(self.hosts[i]) else '0.0.0.0'
            response = self.sessions[i].post(
                f'{self.base_urls[i]}/init_communicator/',
                json={
                    'host': bind_host,
                    'port': self.group_ports[i],
                    'world_size': world_size,
                    'weight_update_protocol': WEIGHT_UPDATE_PROTOCOL_VERSION,
                    **kwargs
                },
                timeout=CONTROL_PLANE_TIMEOUT)
            if response.status_code != 200:
                raise Exception(f'Server {i} init failed: {response.text}')

            time.sleep(0.1)

            pg = StatelessProcessGroup.create(
                host=self.hosts[i], port=self.group_ports[i], rank=rank, world_size=world_size)
            comm = PyNcclCommunicator(pg, device=device)
            self.pynccl_comms.append(comm)

        atexit.register(self.close_communicator)

    def update_named_param(self, name: str, weights: torch.Tensor):
        dtype = str(weights.dtype)
        shape = tuple(weights.shape)

        errors = [None] * self.num_servers

        def _update_single_server(i):
            try:
                response = self.sessions[i].post(
                    f'{self.base_urls[i]}/update_named_param/',
                    json={
                        'name': name,
                        'dtype': dtype,
                        'shape': shape
                    },
                    timeout=CONTROL_PLANE_TIMEOUT,
                )
                if response.status_code != 200:
                    raise Exception(f'Server {i} update failed: {response.text}')
                update_id = self._accepted_weight_update(response)
                self._check_weight_receivers_ready(i)

                self.pynccl_comms[i].broadcast(weights, src=self.pynccl_comms[i].rank)
                self.pynccl_comms[i].group.barrier()
                self._finish_weight_update(i, update_id)
            except Exception as e:
                errors[i] = e

        with ThreadPoolExecutor(max_workers=self.num_servers) as executor:
            futures = [executor.submit(_update_single_server, i) for i in range(self.num_servers)]
            for future in futures:
                future.result()

        all_errors = [e for e in errors if e is not None]
        if all_errors:
            raise RuntimeError(f'Multiple errors: {all_errors}')

    def update_adapter_flattened_param(self, peft_config, metadatas, flattened_tensor):
        """
        Adds a LoRA adapter to the model on all servers using flattened tensor.

        Args:
            peft_config: PEFT configuration for LoRA adapter.
            metadatas: List of FlattenedTensorMetadata objects.
            flattened_tensor: The flattened tensor containing all adapter parameters.
        """
        errors = [None] * self.num_servers
        peft_config = peft_config_to_dict(peft_config)
        metadatas = [m.model_dump() if hasattr(m, 'model_dump') else m.dict() for m in metadatas]
        lora_int_id = int(time.time_ns() % 0x7FFFFFFF)

        def _update_single_server(i):
            try:
                data = {
                    'lora_int_id': lora_int_id,
                    'peft_config': {
                        **peft_config
                    },
                    'metadatas': metadatas,
                }

                response = self.sessions[i].post(
                    f'{self.base_urls[i]}/update_adapter_flattened_param/',
                    json=data,
                    timeout=CONTROL_PLANE_TIMEOUT,
                )
                if response.status_code != 200:
                    raise Exception(f'Server {i} update adapter failed: {response.text}')
                update_id = self._accepted_weight_update(response)
                self._check_weight_receivers_ready(i)

                self.pynccl_comms[i].broadcast(flattened_tensor, src=self.pynccl_comms[i].rank)
                self.pynccl_comms[i].group.barrier()
                self._finish_weight_update(i, update_id)
            except Exception as e:
                errors[i] = e

        with ThreadPoolExecutor(max_workers=self.num_servers) as executor:
            futures = [executor.submit(_update_single_server, i) for i in range(self.num_servers)]
            for future in futures:
                future.result()

        all_errors = [e for e in errors if e is not None]
        if all_errors:
            raise RuntimeError(f'Multiple errors: {all_errors}')

    def update_adapter_param(self, peft_config, lora_params):
        """
        Adds a LoRA adapter to the model on all servers without flattening.
        Sends each tensor individually.

        Args:
            peft_config: PEFT configuration for LoRA adapter.
            lora_params: OrderedDict of (name, tensor) pairs for LoRA parameters.
        """
        errors = [None] * self.num_servers
        peft_config = peft_config_to_dict(peft_config)
        lora_int_id = int(time.time_ns() % 0x7FFFFFFF)

        # Build metadata for each tensor
        lora_tensors_metadata = []
        for name, param in lora_params.items():
            metadata = {
                'name': name,
                'dtype': str(param.dtype),
                'shape': tuple(param.shape),
                'start_idx': 0,  # Not used in non-flattened mode
                'end_idx': param.numel(),  # Not used in non-flattened mode
                'numel': param.numel(),
            }
            lora_tensors_metadata.append(metadata)

        def _update_single_server(i):
            try:
                data = {
                    'lora_int_id': lora_int_id,
                    'peft_config': {
                        **peft_config
                    },
                    'lora_tensors_metadata': lora_tensors_metadata,
                }

                response = self.sessions[i].post(
                    f'{self.base_urls[i]}/update_adapter_param/',
                    json=data,
                    timeout=CONTROL_PLANE_TIMEOUT,
                )
                if response.status_code != 200:
                    raise Exception(f'Server {i} update adapter failed: {response.text}')
                update_id = self._accepted_weight_update(response)
                self._check_weight_receivers_ready(i)

                # Broadcast each tensor individually
                for name, param in lora_params.items():
                    self.pynccl_comms[i].broadcast(param, src=self.pynccl_comms[i].rank)
                self.pynccl_comms[i].group.barrier()
                self._finish_weight_update(i, update_id)
            except Exception as e:
                errors[i] = e

        with ThreadPoolExecutor(max_workers=self.num_servers) as executor:
            futures = [executor.submit(_update_single_server, i) for i in range(self.num_servers)]
            for future in futures:
                future.result()

        all_errors = [e for e in errors if e is not None]
        if all_errors:
            raise RuntimeError(f'Multiple errors: {all_errors}')

    def update_flattened_params(self, metadatas, flattened_tensor):
        """
        Updates model parameters using flattened tensor data.

        Args:
            metadatas: List of FlattenedTensorMetadata objects
            flattened_tensor: The flattened tensor containing all parameters
        """
        errors = [None] * self.num_servers
        metadatas = [m.model_dump() if hasattr(m, 'model_dump') else m.dict() for m in metadatas]
        streams = None
        completed = None
        if self.parallel_weight_sync:
            if not flattened_tensor.is_cuda:
                raise ValueError('parallel_weight_sync requires CUDA weight buffers.')
            with torch.cuda.device(flattened_tensor.device):
                if self._weight_sync_streams is None:
                    self._weight_sync_streams = [
                        torch.cuda.Stream(device=flattened_tensor.device) for _ in range(self.num_servers)
                    ]
                streams = self._weight_sync_streams
                if any(stream.device != flattened_tensor.device for stream in streams):
                    raise ValueError('Weight-buffer device changed; reinitialize the weight-sync communicator.')
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(flattened_tensor.device))
                completed = [torch.cuda.Event() for _ in streams]
                for stream in streams:
                    stream.wait_event(ready)
                    # Also protect storage if a control-plane request fails
                    # after another server has already enqueued its broadcast.
                    flattened_tensor.record_stream(stream)

        def _update_single_server(i):
            try:
                data = {
                    'metadatas': metadatas,
                }

                response = self.sessions[i].post(
                    f'{self.base_urls[i]}/update_flattened_params/',
                    json=data,
                    timeout=CONTROL_PLANE_TIMEOUT,
                )
                if response.status_code != 200:
                    raise Exception(f'Server {i} update flattened params failed: {response.text}')
                update_id = self._accepted_weight_update(response)
                self._check_weight_receivers_ready(i)

                if streams is None:
                    self.pynccl_comms[i].broadcast(flattened_tensor, src=self.pynccl_comms[i].rank)
                else:
                    # CUDA's current device is thread-local. Both the explicit
                    # send stream and its completion event belong to this GPU.
                    with torch.cuda.device(flattened_tensor.device):
                        self.pynccl_comms[i].broadcast(
                            flattened_tensor, src=self.pynccl_comms[i].rank, stream=streams[i])
                        completed[i].record(streams[i])
                self.pynccl_comms[i].group.barrier()
                self._finish_weight_update(i, update_id)
            except Exception as e:
                errors[i] = e

        with ThreadPoolExecutor(max_workers=self.num_servers) as executor:
            futures = [executor.submit(_update_single_server, i) for i in range(self.num_servers)]
            for future in futures:
                future.result()

        all_errors = [e for e in errors if e is not None]
        if all_errors:
            raise RuntimeError(f'Multiple errors: {all_errors}')
        if completed is not None:
            # Finish this bucket before the caller releases it or packs another.
            # Do not replace this with a profiler-only/global CUDA synchronize.
            self._wait_for_weight_sends(completed)

    @staticmethod
    def _wait_for_weight_sends(events):
        deadline = time.monotonic() + CONTROL_PLANE_TIMEOUT
        while True:
            pending = [index for index, event in enumerate(events) if not event.query()]
            if not pending:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f'Parallel weight sends did not complete within {CONTROL_PLANE_TIMEOUT}s; '
                    f'pending servers: {pending}.')
            time.sleep(0.001)

    @staticmethod
    def _accepted_weight_update(response):
        payload = response.json()
        if payload.get('status') != 'accepted' or not isinstance(payload.get('update_id'), int):
            raise RuntimeError('Rollout server lacks the acknowledged weight-update protocol; '
                               'restart it from the same ms-swift revision as the trainer.')
        return payload['update_id']

    def _finish_weight_update(self, server_idx: int, update_id: int):
        response = self.sessions[server_idx].post(
            f'{self.base_urls[server_idx]}/finish_weight_update/', json={'update_id': update_id},
            timeout=CONTROL_PLANE_TIMEOUT)
        if response.status_code != 200:
            raise RuntimeError(f'Server {server_idx} did not load update {update_id}: {response.text}')
        payload = response.json()
        if payload.get('status') != 'loaded' or payload.get('update_id') != update_id:
            raise RuntimeError(f'Server {server_idx} returned an invalid weight-load acknowledgment.')

    def _check_weight_receivers_ready(self, server_idx: int):
        errors = self.pynccl_comms[server_idx].group.all_gather_obj(None)
        failures = [(rank, value) for rank, value in enumerate(errors) if value is not None]
        if failures:
            raise RuntimeError(f'Server {server_idx} cannot receive weights: {failures}')

    def update_model_params(self, model: nn.Module):
        for name, param in model.named_parameters():
            self.update_named_param(name, param.data)

    def reset_prefix_cache(self):
        errors = [None] * self.num_servers

        def _reset_single_server(i):
            try:
                response = self.sessions[i].post(f'{self.base_urls[i]}/reset_prefix_cache/',
                                                 timeout=CONTROL_PLANE_TIMEOUT)
                if response.status_code != 200:
                    raise Exception(f'Server {i} reset failed: {response.text}')
                if response.json().get('success') is not True:
                    raise RuntimeError(f'Server {i} did not confirm prefix-cache reset.')
            except Exception as e:
                errors[i] = e

        with ThreadPoolExecutor(max_workers=self.num_servers) as executor:
            futures = [executor.submit(_reset_single_server, i) for i in range(self.num_servers)]
            for future in futures:
                future.result()
        all_errors = [e for e in errors if e is not None]
        if all_errors:
            raise RuntimeError(f'Multiple errors on reset_prefix_cache: {all_errors}')

    def get_engine_type(self):
        # assume that all server has same engine type
        response = self.sessions[0].post(f'{self.base_urls[0]}/get_engine_type/')
        if response.status_code != 200:
            raise Exception(f'Engine type request failed: {response.text}')

        result = response.json()
        self.use_async_engine = result['engine_type'] == 'AsyncLLMEngine'
        self.enable_multi_turn = result.get('enable_multi_turn', False)
        self.use_gym_env = result.get('use_gym_env', False)
        self.enable_lora = result.get('enable_lora', False)
        return result

    def close_communicator(self):
        if getattr(self, 'persistent_session_id', None):
            if self._persistent_closed:
                return
            errors = []
            for session, base_url in zip(self.sessions, self.base_urls):
                try:
                    response = session.post(f'{base_url}/end_session/', timeout=CONTROL_PLANE_TIMEOUT)
                    info = response.json()
                    if (response.status_code != 200 or info.get('state') != 'idle'
                            or info.get('session_id') is not None):
                        raise RuntimeError(f'Persistent rollout release failed: {info}')
                except Exception as error:
                    errors.append(f'{base_url}: {error}')
            self.pynccl_comms.clear()
            self._weight_sync_streams = None
            if errors:
                raise RuntimeError('; '.join(errors))
            self._persistent_closed = True
            return
        for i in range(self.num_servers):
            try:
                response = self.sessions[i].post(f'{self.base_urls[i]}/close_communicator/',
                                                 timeout=CONTROL_PLANE_TIMEOUT)
                if response.status_code != 200:
                    logger.warning(f'Server {i} close failed: {response.text}')
            except Exception as e:
                logger.warning(f'Error closing server {i} communicator: {str(e)}')
