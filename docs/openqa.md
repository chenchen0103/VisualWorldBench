# API inference and Open-QA

Run commands from the repository root.

For local Qwen3-VL-8B Open-QA generation, use the same installed environment and
eight available GPUs as the closed-task runs:

```bash
MODEL_ROOT=/path/to/models CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/run_openqa_qwen3_vl_8b.sh
```

This uses the native vLLM backend, `open_reason` prompt, temperature 0, top_p 1,
batch size 258 per worker, TP=1, and 2048 maximum output tokens. It runs the
complete local Open-QA split and reports the number of pending judgments.
Predictions are saved to
`results/openqa-qwen3-vl-8b/predictions/open_qa/Qwen3-VL-8B/open_qa.jsonl`.
Set `OUTPUT_ROOT` to a new directory for another run. Use `--check-only` for
input validation.
To judge and score those predictions, set `JUDGE_API_KEY` and run:

```bash
python judge.py --bench-jsonl ./data/open_qa.jsonl \
  --pred-file ./results/openqa-qwen3-vl-8b/predictions/open_qa/Qwen3-VL-8B/open_qa.jsonl \
  --base-url YOUR_JUDGE_API_BASE_URL --judge-model YOUR_MODEL_ID
python score.py --bench-dir ./data \
  --pred-dir ./results/openqa-qwen3-vl-8b/predictions \
  --tasks open_qa --output ./results/openqa-qwen3-vl-8b/scores.tsv
```

Before judging, the score is exact match only.

`infer_api.py` supports OpenAI-compatible endpoints for other models or Open-QA.
Use `evaluate.py` for the local Qwen3-VL setup above.
API runs exit with an error if requests remain unsuccessful. Rerun the same
command to retry failed or missing samples. Successful responses, including
wrong answers and answer-parse failures, are retained when resuming a run.
A truncated final JSONL record is recovered on resume; malformed records
elsewhere require correction.

```bash
# Set OPENAI_API_KEY in your environment.
python infer_api.py --bench-dir ./data --pred-dir ./results/api/predictions \
  --model YOUR_MODEL_ID --model-name MyModel --base-url YOUR_API_BASE_URL \
  --tasks open_qa --temperature 0 --max-tokens 2048

# Set JUDGE_API_KEY and JUDGE_BASE_URL in your environment.
# Choose a judge model supported by your provider.
python judge.py --bench-jsonl ./data/open_qa.jsonl \
  --pred-file ./results/api/predictions/open_qa/MyModel/open_qa.jsonl \
  --base-url YOUR_JUDGE_API_BASE_URL --judge-model YOUR_MODEL_ID
```

Set the judge model with `--judge-model` or `JUDGE_MODEL`. Use `--dry-run`
to count pending judgments.
The default `--api-format gemini-responses` sends **text only** using Gemini
`contents`/`parts` to `/responses`. Configure a provider that supports this
schema, including its `/v1` base URL and API key. For a conventional
OpenAI-compatible judge, specify `--api-format chat-completions` and
`--judge-model YOUR_MODEL_ID`.
Changing the model or protocol requires a separate output file or an explicit
overwrite.
Alias matches are scored directly; remaining answers are sent to the judge
for exact category matching. Judge calls may incur API charges.
The judge must return JSON with `judgment` set to `correct` or `wrong`.
HTTP failures and invalid judge responses are retried up to `--max-retries`.
Unresolved failures are saved as `judgment: error`; rerunning retries them.
Complete these judgments before computing corrected Open-QA and overall scores.
