#!/usr/bin/env bash
# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="${1:-${repo_root}/configs/train/unimopd_uniform_rkl.yaml}"

required=(
  STUDENT_MODEL TEACHER_STEM TEACHER_VQA TEACHER_DOCUMENT
  TEACHER_GUI TEACHER_EMBODIED DATA_PATH OUTPUT_DIR ROLLOUT_HOST
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

export ROLLOUT_PORT="${ROLLOUT_PORT:-8000}"
export ROLLOUT_GROUP_PORT="${ROLLOUT_GROUP_PORT:-20000}"
export MIN_PIXELS="${MIN_PIXELS:-3136}"
export MAX_PIXELS="${MAX_PIXELS:-16777216}"
export MOPD_REQUIRE_IMAGE_BUDGETS="${MOPD_REQUIRE_IMAGE_BUDGETS:-1}"
export PYTHONPATH="${repo_root}/ms-swift:${PYTHONPATH:-}"
export MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-${repo_root}/Megatron-LM}"

# The recipe trains with `deterministic_mode: true`, which has exactly one
# correct setting for each of the following, so they are pinned rather than
# defaulted: an inherited value would otherwise change reduction order or
# re-enable nondeterministic kernels without failing. Megatron rejects a
# non-deterministic `NCCL_ALGO`, and Torch's deterministic algorithms reject
# cuBLAS matmuls without a fixed workspace. `Ring` also applies to the rollout
# server: the two roles share one weight-synchronization communicator.
export NCCL_ALGO=Ring
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export FLASH_ATTENTION_DETERMINISTIC=1
# Fabric- and memory-dependent, so these stay overridable. A single long
# response can exceed the default collective watchdog while the remaining ranks
# wait on the slowest rollout, which aborts an otherwise healthy step.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

resolved="$(mktemp "${TMPDIR:-/tmp}/unimopd-train.XXXXXX.yaml")"
trap 'unlink "${resolved}" 2>/dev/null || true' EXIT
python3 "${repo_root}/scripts/resolve_train_config.py" "${config}" "${resolved}"

extra_args=()
if [[ -n "${MOPD_MODEL_CACHE_DIR:-}" ]]; then
  extra_args+=(--mopd_model_cache_dir "${MOPD_MODEL_CACHE_DIR}")
fi

megatron rlhf --config "${resolved}" "${extra_args[@]}"
