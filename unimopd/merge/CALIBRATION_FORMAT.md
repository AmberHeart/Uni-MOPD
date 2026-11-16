# Response-logit calibration artifact

`unimopd_response_logit_calibration_v1` is a CPU reference format for
coefficient optimization after response data and teacher targets have been
prepared. It deliberately does not collect teacher rollouts or load a model.

The artifact is a tensor-only file written with `torch.save` and loaded with
`weights_only=True`. It contains:

| Field | Shape or type | Meaning |
|---|---|---|
| `format` | string | `unimopd_response_logit_calibration_v1` |
| `coefficient_names` | list of strings | Ordered merge branches |
| `student_base_logits` | `[N, V]` float tensor | Student response logits at zero coefficients |
| `student_logit_deltas` | `[N, K, V]` float tensor | Logit direction for each coefficient |
| `teacher_logits` | `[N, V]` float tensor | Frozen routed-teacher logits |
| `labels` | `[N]` long tensor | Teacher-response token IDs |
| `sequence_offsets` | `[S+1]` long tensor | Strict partition of the `N` response tokens |
| `sequence_weights` | `[S]` float tensor | Positive per-sequence weights summing to one |

The oracle evaluates
`student_base_logits + einsum("k,nkv->nv", alpha, student_logit_deltas)`.
`sequence_weights` can implement equal-domain, per-sequence aggregation by
assigning each sequence its domain weight divided by the number of sequences
from that domain.

This compact artifact is exact only when its producer establishes the affine
response-logit contract. Full-paper reproduction composes DARE weights and
runs the model at every coefficient point; that model/GPU oracle is outside
this CPU artifact and must report the same objective and gradient interface.
