# Repository structure

| Path | Purpose |
|---|---|
| `ms-swift/` | Pinned ms-swift source plus the Uni-MOPD trainer, routed teachers, response-only heads, and vLLM integration |
| `patches/megatron-lm.patch` | Minimal patch for R3 route replay, execution padding, MoE RNG resume, and mixed student/teacher parallel layouts |
| `unimopd/merge/` | Streaming DARE/task-vector merge and coefficient calibration |
| `unimopd/data/` | Deterministic query allocation |
| `configs/train/` | Effective RKL recipe and matched FKL/ToDi controls |
| `configs/merge/` | Uniform and tuned merge recipes |
| `scripts/` | Rollout, training, merge, and calibration entry points |
| `tests/` | CPU contract tests; GPU/distributed acceptance remains a separate reproduction step |
| `docs/` | Environment, data schema, allocation, merge, and training contracts |

The ms-swift baseline is commit
`6c2ee40ddb43831da1aac7a9e54ea5d36f74f5f4`. Repository-specific changes are
kept as ordinary source files so readers can inspect the complete training
path. Megatron-LM remains an external pinned checkout because its license and
release cadence are separate; the exact patch is readable in `patches/`.
