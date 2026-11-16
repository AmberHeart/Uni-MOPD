# Reproducibility

What this repository lets you verify, and how. The *method* is fully contained
here and checkable on a laptop; reproducing the paper's *scores* additionally
takes the training queries and an evaluation harness of your own.

## Verified: CPU test suite

```bash
python -m pytest tests -q
```

195 tests, no GPU, no model weights, no data, under ten seconds. Run on Python
3.11 with the pinned dependency set. These are contracts against the production
sources — the tests load the real trainer and sampler definitions rather than
reimplementing them, and the domain keys, teacher indices, and loss
normalization come from `configs/train/unimopd_uniform_rkl.yaml` rather than
being restated — so they fail if the published code or recipe drifts.

The contracts that matter most were checked by deliberately breaking them and
confirming the suite notices: swapping two domains onto one teacher, routing
each query to the next teacher instead of its own, and replacing global-token
normalization with per-sequence averaging or a wrong constant each produce
failures rather than silent passes.

| Area | Tests | What is checked |
|---|---:|---|
| Package and dependencies | 10 | Every public module imports; declared dependencies are present; the reference matrix pins exact versions |
| Merge recipes | 25 | Both published recipes parse, every field is accepted by the merger, the coefficients match the paper's uniform and tuned merges, no absolute paths, and an unset path variable fails before any merge work begins |
| Merge arithmetic | 11 | Average and task-vector merges against closed forms; DARE is bit-identical across worker counts via per-key deterministic seeding; calibration objectives and simplex projection; mismatched parameter sets are rejected |
| Distillation objectives | 38 | Full-vocabulary RKL, FKL, and ToDi against independent reference implementations, including masking, temperature, and gradients; identical teacher and student are stationary; response-only rows match the dense result; invalid configurations fail before training |
| Streamed teacher head | 20 | Chunked and checkpointed teacher-head evaluation matches the dense computation in both value and gradient, across chunk sizes; frozen teacher weights are never updated |
| Domain routing | 19 | The five domain keys and teacher indices are read from the published recipe, not restated in the test; each routes to its own teacher; every microbatch carries exactly one teacher; no sample is dropped or duplicated; a histogram that cannot tile the microbatch grid is rejected; domain loss scales renormalize over present domains |
| Query sampler | 20 | Uniform allocation is without replacement and deterministic; hash ranking matches the frozen recipe; selection and ordering seeds are independent; data-parallel ranks receive disjoint shares; resume skips exactly what was consumed; unusable domain columns fail loudly |
| R3 router replay | 21 | Rollout expert IDs survive transport in compact integer form; returned-prefix binding trims only the unreturned tail; snapshots survive KV slot reuse; preemption discards stale slot ownership; malformed payloads are rejected |
| Published configs | 9 | The training recipes resolve from outside the repository, every published key is a real CLI field, the matched objective variants change only the objective, and the shell entry points parse |
| Microbatch encoding | 1 | Complete-response encoding tags only the fields the forward step consumes |
| End-to-end step | 7 | One full MOPD update on a tiny student with five mock teachers: sampler to routing to routed-teacher RKL to loss normalization to optimizer step. The step's gradient and loss match a closed-form routed reverse KL over all response tokens, pinning the published `global_token` objective and its scale; gradients reach the student and never the frozen teachers; the routed reverse KL decreases under training; a student initialized as its teacher is already stationary |
| Export files | 2 | Exported Megatron checkpoints carry the source generation config without overwriting an existing one |
| Image budgets | 12 | Per-image pixel limits stay explicit and independently validated; a missing image source is rejected rather than defaulted |

## Verified: inference

The README Quick Start snippet was run verbatim against the released
checkpoint's weights on a single 140 GB GPU: 13 BF16 shards load through
`AutoModelForImageTextToText`, and greedy decoding on a multimodal prompt
returns a correct description of the input image.

Reproduce it after downloading the model:

```bash
python -m pytest tests -q          # no weights needed
# then the Quick Start snippet in README.md with a local image
```

## Verified: installation

`setup.sh` clones Megatron-LM at the pinned commit and applies
`patches/megatron-lm.patch`. Verified against a fresh clone of upstream
`0d7e02b`: the patch applies cleanly, and re-running `setup.sh` detects the
applied state instead of applying it twice. It modifies six upstream files
(MoE routing hooks, router state, MoE RNG resume, mixed student/teacher
parallel layouts, execution padding, logger-shutdown isolation) and adds
`megatron/core/transformer/moe/routing_replay.py`, the R3 replay frame. The
replay frame was exercised directly: a recorded route is reproduced exactly on
the second call, and external expert IDs are validated on CPU before use.

## Full-scale runs

| Item | What it needs | Status |
|---|---|---|
| Merge reproduction | The six public teacher checkpoints (released on Hugging Face and ModelScope) and roughly 62 GB per output | Runnable now. Deterministic per-key seeding means re-merging the same inputs with `configs/merge/uniform.yaml` reproduces the initialization |
| MOPD training run | Three eight-GPU nodes (two training, one rollout), the five first-bank teachers, and a multi-domain query set in the documented format | Runnable with your own queries. The published config is the effective paper recipe |
| Paper's benchmark scores | The public benchmarks and an evaluation harness of your own | The README reports the benchmarks and the scores to match |
| Bitwise-identical training | — | Rollouts are served asynchronously and collected in completion order by design, so the reproduction target is the loss curve and the evaluation average, not tensor equality |

## Determinism

The training launcher pins the settings that `deterministic_mode: true`
requires: a deterministic NCCL algorithm, a fixed cuBLAS workspace, and
non-deterministic Transformer Engine and FlashAttention kernels disabled. These
are pinned rather than defaulted because an inherited value would silently
change reduction order. Fabric-dependent timeouts stay overridable.

The merge is deterministic by construction: DARE's per-parameter mask is seeded
from the merge seed, the parameter name, and the input index, so the result is
independent of worker count and shard order.

## Environment

The paper run used Python 3.11, CUDA 12.8, torch 2.8.0, ms-swift 3.13.0.dev0,
Megatron Core 0.15.0, vLLM 0.11.0, NCCL 2.27.3, and H200 GPUs. The CPU test
suite needs only torch, numpy, safetensors, transformers, omegaconf, and PyYAML.
See [docs/environment.md](docs/environment.md) for pinned versions and build
flags.
