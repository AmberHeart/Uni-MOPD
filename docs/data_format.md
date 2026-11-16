# Training data format

Training data is JSONL. Each record contains a conversation, one or more image
references, and the domain key used to select a teacher.

```json
{
  "id": "public-dataset:split:000001",
  "domain": "GeneralVQA",
  "messages": [
    {"role": "user", "content": "<image> What is shown?"},
    {"role": "assistant", "content": "A placeholder reference answer."}
  ],
  "images": [
    {
      "path": "images/000001.jpg",
      "min_pixels": 262144,
      "max_pixels": 1800000
    }
  ]
}
```

The released training path accepts these domain keys:

| Public role | Data key | Teacher index | Response cap |
|---|---|---:|---:|
| STEM | `StemPuzzle` | 0 | 12,288 |
| General visual QA | `GeneralVQA` | 1 | 8,192 |
| Document/OCR/chart | `TextChartUnder` | 2 | 12,288 |
| GUI grounding | `guiagent` | 3 | 1,024 |
| Embodied understanding | `EmbodUnde` | 4 | 4,096 |

Every image should carry its own `min_pixels` and `max_pixels`. The paper data
used the following ranges:

| Domain | `min_pixels` | `max_pixels` |
|---|---:|---:|
| STEM | 65,536 | 16,777,216 |
| General visual QA | 262,144 | 1,800,000 |
| Document/OCR/chart | 262,144 | 4,194,304 |
| GUI grounding | 262,144 | 16,777,216 |
| Embodied understanding | 262,144 | 4,194,304 |
| Embodied high-resolution subset | 2,097,152 | 6,291,456 |

`MIN_PIXELS=3136` and `MAX_PIXELS=16777216` are compatibility fallbacks, not
the effective per-domain recipe. Missing per-image budgets are rejected by the
Qwen3-VL template on the Uni-MOPD path.

The assistant answer is retained for format compatibility and optional
off-policy controls. The main run uses `lmbda=1`: the student produces a fresh
answer, and only response-token positions contribute to distillation.
