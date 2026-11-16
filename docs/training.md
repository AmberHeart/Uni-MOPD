# Multi-teacher on-policy distillation

## Effective recipe

The released checkpoint is update 50 of one run initialized from the
uniform merge. The learning rate is constant, so a 50-update horizon and a
longer one produce the same optimizer schedule over the first 50 updates: the
published configuration reproduces that prefix exactly.

| Component | Effective value |
|---|---|
| Objective | Full-vocabulary reverse KL on response tokens |
| KD / rollout temperature | 1 / 1 |
| Loss normalization | Global response-token mean |
| Optimizer | AdamW, LR `3e-6`, constant, no warmup |
| Adam / regularization | betas `(0.9, 0.99)`, eps `1e-5`, WD `0.01`, clip `1.0` |
| Prompt / total sequence cap | 20,480 / 32,768 tokens |
| Router contract | R3: replay rollout expert IDs during student training |
| Student precision | BF16 full-parameter update, ViT and aligner trainable |
| Sampling | `top_p=1`, `top_k=-1`, repetition penalty 1, thinking disabled |

FKL is the matched `gkd_loss_type=jsd, beta=0` control. ToDi uses
`gkd_loss_type=todi, gkd_todi_beta=1`. The RKL recipe uses `beta=1`.

## Parallel layout and batch semantics

The paper layout uses three eight-GPU nodes:

| Role | GPUs | Layout |
|---|---:|---|
| Student training | 16 | TP2, EP8, ETP1, PP1, CP1, Megatron DP8 |
| Student rollout | 8 | vLLM TP2 × DP4 |
| Teachers | training nodes | five frozen teachers, CPU cached and loaded per routed block |

`micro_batch_size=1` is per data-parallel rank. `global_batch_size=512` spans
all eight Megatron data-parallel replicas, so each update accumulates
`512 / 8 / 1 = 64` microbatches per data-parallel rank. This is not a
per-GPU batch of 512.

The source data contains 204,800 unique queries: 40,960 from each domain. It is
selected without replacement within domains and globally hash-shuffled. It is
not forced into a five-domain-balanced pattern inside every batch. The first
25,600 queries consumed by update 50 contain 5,111 STEM, 5,044 VQA, 5,163
document, 5,104 GUI, and 5,178 embodied queries.

## Launch

Start the rollout server on the rollout node:

```bash
export STUDENT_MODEL=/models/Uni-MOPD-Merge-Uniform-30B-A3B
export OUTPUT_DIR=outputs/rollout
bash scripts/launch_rollout.sh
```

On both training nodes, expose the same distributed variables used by
`torchrun`:

```bash
export NNODES=2 NPROC_PER_NODE=8
export NODE_RANK=0                    # 1 on the second node
export MASTER_ADDR=trainer-node-0 MASTER_PORT=29500

export STUDENT_MODEL=/models/Uni-MOPD-Merge-Uniform-30B-A3B
export TEACHER_STEM=/models/teacher-stem
export TEACHER_VQA=/models/teacher-vqa
export TEACHER_DOCUMENT=/models/teacher-document
export TEACHER_GUI=/models/teacher-gui
export TEACHER_EMBODIED=/models/teacher-embodied
export DATA_PATH=/data/train.jsonl
export OUTPUT_DIR=outputs/unimopd-rkl
export ROLLOUT_HOST=rollout-node
export ROLLOUT_PORT=8000
export ROLLOUT_GROUP_PORT=20000

bash scripts/train_unimopd.sh configs/train/unimopd_uniform_rkl.yaml
```

Use `unimopd_uniform_fkl.yaml` or `unimopd_uniform_todi.yaml` for the matched
objective controls. The launcher resolves their small override files against
the full RKL config before invoking ms-swift.

`MOPD_MODEL_CACHE_DIR` optionally enables a node-local converted-model cache.
It changes load time, not model math. Each cache directory should be
private to one compatible model/runtime layout.

The paper config keeps safetensors for every 10-update checkpoint but retains
optimizer/RNG state only for the newest checkpoint
(`mopd_prune_old_optimizer_states=true`). Set this option to `false` when older
checkpoints must remain resumable.
