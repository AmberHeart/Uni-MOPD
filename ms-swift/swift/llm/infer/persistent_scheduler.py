# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0

"""Pinned vLLM V1 scheduler extension for weight-dependent encoder invalidation.

vLLM 0.11 reset_prefix_cache resets KV metadata only. EncoderCacheManager keeps
unreferenced image hashes across requests, independently of mm_processor_cache.
Clearing only GPUModelRunner.encoder_cache would leave dangling scheduler hits.
"""
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.sched.scheduler import Scheduler


class PersistentMOPDScheduler(Scheduler):
    def reset_prefix_cache(self):
        # EngineCore discards a False return in v0.11. Raise on refusal so the
        # caller cannot accidentally commit a weight update on a busy engine.
        if self.has_unfinished_requests():
            raise RuntimeError('Cannot reset persistent rollout caches with unfinished requests')
        manager = self.encoder_cache_manager
        if any(manager.cached.values()):
            raise RuntimeError('Cannot reset persistent rollout caches with live encoder references')
        if not super().reset_prefix_cache():
            raise RuntimeError('Persistent rollout KV-cache reset was refused')
        self.encoder_cache_manager = EncoderCacheManager(manager.cache_size)
        return True
