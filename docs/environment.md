# Environment

Uni-MOPD uses a vendored ms-swift tree and a pinned Megatron-LM checkout. The
paper run used Python 3.11, ms-swift 3.13.0.dev0, Megatron Core 0.15.0, vLLM
0.11.0, NCCL 2.27.3, and eight-GPU H200 nodes.

`requirements.txt` pins the reference build: PyTorch 2.8.0 (CUDA 12.8),
Transformers 4.57.1, vLLM 0.11.0, Transformer Engine 2.9.0, FlashAttention
2.7.4.post1, TRL 0.23.1, and the libraries that affect data loading and
configuration. NCCL 2.27.3 and cuDNN 9.10.2 come with the PyTorch wheels.

## System prerequisites

Transformer Engine, FlashAttention, and apex compile CUDA extensions during
installation, so the host needs more than the Python wheels:

- Python 3.11 and a CUDA 12.8 toolkit providing `nvcc`.
- cuDNN 9 development headers, matching the 9.10.2 runtime that the PyTorch
  wheel installs. On Ubuntu 22.04 with NVIDIA's CUDA repository these are
  `libcudnn9-dev-cuda-12`, `libcudnn9-headers-cuda-12`, and `libcudnn9-cuda-12`
  at version `9.10.2.21-1`; Transformer Engine fails to build without the
  headers.
- A C++ toolchain and `ninja`.

`nvidia/cuda:12.8.2-devel-ubuntu22.04` provides the toolkit and toolchain and
is a convenient starting point for a container image.

Export the target GPU architectures before installing, otherwise each extension
picks its own default set and the build takes considerably longer:

```bash
export TORCH_CUDA_ARCH_LIST=9.0     # apex; Hopper (H100/H200)
export NVTE_CUDA_ARCHS=90           # Transformer Engine
export FLASH_ATTN_CUDA_ARCHS=90     # FlashAttention
export MAX_JOBS=16                  # bounds build memory use
```

Use `8.0`/`80` for A100, or list several (`"8.0;9.0"` and `"80;90"`) to build
one environment for both.

## Installation

With the prerequisites above in place:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-build-isolation -r requirements.txt
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 python -m pip install --no-build-isolation \
  "git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4"
python -m pip install -e . -e ms-swift
bash setup.sh
export MEGATRON_LM_PATH="$PWD/Megatron-LM"
```

FlashAttention 2.7.4.post1 has no prebuilt wheel for PyTorch 2.8 and compiles
from source; set `MAX_JOBS` to bound memory use during the build. apex provides
the fused gradient accumulation used by the paper recipe; without it training
falls back to unfused accumulation, which is numerically different.

`setup.sh` checks out Megatron-LM at
`0d7e02bd26d3eb347ad8d1bef625002babfa6df7` and applies
`patches/megatron-lm.patch`. It is safe to run again on an already patched
checkout.

For another GPU generation, keep BF16 and the mathematical recipe fixed first.
Tune only the rollout server's placement and memory headroom —
`ROLLOUT_GPU_MEMORY_UTILIZATION` (default 0.95), `ROLLOUT_MAX_NUM_SEQS`
(default 512), `ROLLOUT_TP`, `ROLLOUT_DP` — and the optional local model-cache
directory. TP, EP, batch size, sequence limits, image budgets, and response
caps are part of the reproduced recipe.

`MOPD_REQUIRE_IMAGE_BUDGETS` defaults to 1 and makes both launchers fail when a
sample reaches the model without explicit per-image pixel limits. Leave it on:
silently falling back to processor defaults changes the visual token count and
therefore the recipe.

## Collectives and determinism

The recipe sets `deterministic_mode: true`, so `scripts/train_unimopd.sh` pins
`NCCL_ALGO=Ring` (Megatron accepts only deterministic all-reduce algorithms),
`CUBLAS_WORKSPACE_CONFIG=:4096:8` (Torch's deterministic algorithms reject
cuBLAS matmuls without a fixed workspace), `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`
and `FLASH_ATTENTION_DETERMINISTIC=1`. These four are set unconditionally and
ignore the surrounding environment: each has one correct value here, and an
inherited one would change reduction order or re-enable nondeterministic
kernels without failing. Change them by editing the launcher, as a deliberate
departure from the published recipe.

The collective watchdog timeout, its monitor, and the allocator setting are
fabric- and memory-dependent and stay overridable through the environment. The
published values raise the watchdog because one long response can otherwise
trip it while the remaining ranks wait on the slowest rollout.

The trainer's weight-synchronization client and the rollout server join a
single NCCL communicator, so both ends must agree on its algorithm and
transport. Both launchers therefore pin the same `NCCL_ALGO`, and
`MOPD_WEIGHT_SYNC_NCCL_NET=Socket` — needed only on networks without working
IB/RoCE for this communicator — makes `scripts/launch_rollout.sh` select the
socket backend on the server as well. Setting it on one side only leaves the
two ends on different transports and weight synchronization fails to
initialize. Only the network plugin is affected, so the rollout server's
intra-node tensor-parallel transports are unchanged; without that variable,
IB/RoCE stays available to both roles.

Useful checks:

```bash
python -m pytest -q tests
git -C Megatron-LM apply --reverse --check ../patches/megatron-lm.patch
```
