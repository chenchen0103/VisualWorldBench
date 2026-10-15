# Evaluation protocol

Run commands from the repository root.

## Inference settings

The local runner uses native offline inference with temperature 0, top_p 1,
batch size 258 per worker, 2048 output tokens, automatic BF16/model context,
and TP=1 per worker.
MCQ uses eight independent workers; Grid-Sel uses one for 2B and eight for
4B/8B; Grid-Loc uses four. It preserves dataset order, complete target questions,
native image preprocessing, and chat-template behavior. This is data parallel
inference, not a single TP=8 server.

## Scoring

```bash
python score.py --bench-dir ./data \
  --pred-dir /path/to/run/predictions --output ./scores.tsv \
  --tasks mcq,sel_2x2,sel_3x3,loc_2x2,loc_3x3
```

| Task | Metric |
|---|---|
| MCQ | Parsed option-index accuracy |
| Grid-Sel | Parsed cell-index accuracy, separately for 2×2 and 3×3 |
| Grid-Loc | Fraction with bounding-box IoU ≥ 0.5 |
| Open-QA | Alias exact match plus judge-approved answers |

Scores are fractions, not percentages. Parse failures count as incorrect.
Before scoring each task, prediction qids must cover its benchmark exactly once.
Missing, duplicate, unknown or malformed records and infrastructure failures
cause scoring to fail. Retry failed requests before scoring.
The default `--tasks auto` scores existing task files only, validating each one;
use an explicit task list to require those files, or `--tasks all` to require
all six task files. Missing tasks never produce an overall score.
The option and bbox parsers match the native evaluation rules, including
coordinate clipping to [0,1000] and corner ordering. MCQ/Grid-Sel scoring uses
stored correctness flags when present.

The overall score is the mean of MCQ, mean(Grid-Sel), mean(Grid-Loc), and
judge-corrected Open-QA. It remains unavailable without completed Open-QA judgments.
