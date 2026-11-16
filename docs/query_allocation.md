# Query allocation

Uni-MOPD routes each query to one specialist teacher through its `domain` field.
The paper's default policy is **uniform domain sampling**: allocate the same number of unique
queries to each of the five domains, select independently within each domain,
then shuffle all selected queries together with a second deterministic hash.
The global order is intentionally not balanced batch by batch.

The allocator expects JSONL objects with globally unique `id` (or `query_id`)
and a nonempty `domain`. It never repeats a query and fails if a domain cannot
supply its quota.

```bash
python -m unimopd.data.allocate \
  --input data/query_pool.jsonl \
  --output data/train.jsonl \
  --total 204800 \
  --domains StemPuzzle GeneralVQA TextChartUnder guiagent EmbodUnde \
  --policy uniform \
  --selection-seed 20261015 \
  --order-seed 42
```

This writes `data/train.jsonl` and `data/train.jsonl.manifest.json`. Selection
rank is `SHA256(selection_seed:selection:query_id)` within each domain. Final order is
`SHA256(order_seed:global-order:query_id)` across domains. The
purpose strings are part of the hash definition and must be kept verbatim to
reproduce a selection. Keeping the seeds separate lets an experiment reuse the
same selected set with another training order.

The paper run selected 204,800 unique queries, 40,960 per domain. Its update-50
checkpoint consumed the first 25,600 rows of that globally shuffled file; it
did not select a separate 25,600-row, exactly balanced dataset. Consequently,
the consumed prefix is only approximately balanced across domains. A smaller
`--total` value is useful for a new experiment but does not reconstruct the
paper checkpoint's input prefix.

## Optional signal-weighted policies

`gap` allocates rows in proportion to a supplied nonnegative domain gap.
`gap_sqrt_tokens` uses

\[
w_d = \frac{\mathrm{gap}_d}{\sqrt{\mathrm{mean\_tokens}_d}}.
\]

These are optional allocation controls, not the paper's default recipe. Supply
the measurements explicitly so the derived dataset records its assumptions:

```json
{
  "StemPuzzle": {"gap": 4.0, "mean_tokens": 1600},
  "GeneralVQA": {"gap": 3.0, "mean_tokens": 700},
  "TextChartUnder": {"gap": 2.0, "mean_tokens": 2200},
  "guiagent": {"gap": 1.0, "mean_tokens": 300},
  "EmbodUnde": {"gap": 2.5, "mean_tokens": 1100}
}
```

Save that object to a file of your own and pass its path to `--signals`:

```bash
python -m unimopd.data.allocate \
  --input data/query_pool.jsonl \
  --output data/train-gap.jsonl \
  --total 25600 \
  --domains StemPuzzle GeneralVQA TextChartUnder guiagent EmbodUnde \
  --policy gap_sqrt_tokens \
  --signals allocation_signals.json
```

Integer quotas use the largest-remainder rule, with domain name as the stable
tie-breaker. A capacity shortfall is reported instead of silently reallocating
queries to another domain.
