# Merge and coefficient calibration

## Uniform DARE merge

The paper initialization applies DARE independently to each specialist task
vector relative to Qwen3-VL-30B-A3B-Instruct:

\[
\theta = \theta_0 + \sum_{k=1}^{5} 0.2\,
\operatorname{DARE}_{\lambda=0.7}(\theta_k-\theta_0).
\]

Each `(parameter name, teacher index)` receives its own deterministic seed.
The result therefore does not depend on shard scheduling or worker count.
Floating-point task vectors accumulate in FP32 and are converted back to the
base checkpoint dtype.

```bash
export BASE_MODEL=/models/Qwen3-VL-30B-A3B-Instruct
export TEACHER_STEM=/models/teacher-stem
export TEACHER_VQA=/models/teacher-vqa
export TEACHER_DOCUMENT=/models/teacher-document
export TEACHER_GUI=/models/teacher-gui
export TEACHER_EMBODIED=/models/teacher-embodied
export OUTPUT_DIR=outputs/merge-uniform

bash scripts/merge.sh configs/merge/uniform.yaml
```

The output directory contains HF safetensors shards, runtime tokenizer/config
sidecars, and `merge_info.json`. The command refuses to overwrite a nonempty
directory. `configs/merge/tuned.yaml` contains the paper's hand-tuned
comparison coefficients `(0.10, 0.15, 0.35, 0.30, 0.10)`.

## Calibration objectives and solvers

`unimopd.merge.objectives` implements full-vocabulary reverse KL, forward KL,
ToDi, and response cross-entropy. `unimopd.merge.solver` provides simplex PGD
with Armijo backtracking and projected Adam.

The public CPU calibration CLI consumes a tensor-only artifact:

```bash
bash scripts/calibrate_merge.sh \
  --artifact calibration.pt \
  --objective rkl \
  --solver pgd \
  --steps 8 \
  --learning-rate 0.05 \
  --output outputs/pgd8.json
```

The exact schema is in
[`unimopd/merge/CALIBRATION_FORMAT.md`](../unimopd/merge/CALIBRATION_FORMAT.md).
The compact affine-logit oracle is exact only when the artifact producer
establishes that contract. Reproducing the paper's model-space calibration
requires an external GPU producer that rematerializes the merged model at each
coefficient point and emits the same value/gradient oracle.
