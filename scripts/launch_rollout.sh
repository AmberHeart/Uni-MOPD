#!/usr/bin/env bash
# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${STUDENT_MODEL:?Set STUDENT_MODEL to the equal-weight merge or a training checkpoint.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for rollout logs.}"

export PYTHONPATH="${repo_root}/ms-swift:${PYTHONPATH:-}"
export MIN_PIXELS="${MIN_PIXELS:-3136}"
export MAX_PIXELS="${MAX_PIXELS:-16777216}"
export MOPD_REQUIRE_IMAGE_BUDGETS="${MOPD_REQUIRE_IMAGE_BUDGETS:-1}"

# This server and the trainer's weight-synchronization client form one NCCL
# communicator, so both ends must agree on how that communicator is built.
# NCCL_ALGO selects the collective algorithm and is pinned for the same reason
# as in the training launcher: an inherited value would leave the two ends on
# different algorithms.
export NCCL_ALGO=Ring
# The transport is separate. When the trainer selects the socket backend for
# the weight-sync communicator, this server must select it too, or the two ends
# negotiate different transports and the communicator fails to initialize.
# Only the network plugin is affected, so the server's own intra-node
# tensor-parallel transports are untouched.
if [[ "${MOPD_WEIGHT_SYNC_NCCL_NET:-}" == 'Socket' ]]; then
  export NCCL_NET=Socket
  export NCCL_IB_DISABLE=1
fi

mkdir -p "${OUTPUT_DIR}"
# Replace the shell so the server receives termination signals directly: the
# server is stopped once training finishes, and that is a normal shutdown.
exec swift rollout \
  --model "${STUDENT_MODEL}" \
  --result_path "${OUTPUT_DIR}/rollout_results.jsonl" \
  --port "${ROLLOUT_PORT:-8000}" \
  --vllm_tensor_parallel_size "${ROLLOUT_TP:-2}" \
  --vllm_data_parallel_size "${ROLLOUT_DP:-4}" \
  --vllm_enable_expert_parallel false \
  --vllm_enforce_eager false \
  --vllm_gpu_memory_utilization "${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.95}" \
  --vllm_max_num_seqs "${ROLLOUT_MAX_NUM_SEQS:-512}" \
  --vllm_enable_prefix_caching true \
  --vllm_return_routed_experts true \
  --vllm_mm_processor_cache_gb 0 \
  --vllm_max_model_len 32768
