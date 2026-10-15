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
complete local Open-QA split and only counts pending judge requests afterward;
it does not call a judge API. Predictions are saved to
`results/openqa-qwen3-vl-8b/predictions/open_qa/Qwen3-VL-8B/open_qa.jsonl`.
Set `OUTPUT_ROOT` to a new directory for another run; existing runs are not
overwritten. Add `--check-only` to validate inputs without loading a model.
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
It is an alternative backend; use `evaluate.py` for the Qwen3-VL setup above.
API runs exit with an error if requests remain unsuccessful. Rerun the same
command to retry failed or missing samples. Successful responses, including
wrong answers and answer-parse failures, are retained; retries do not create
duplicate prediction records. A truncated final JSONL record can be recovered
on resume, while malformed records elsewhere are rejected.

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

There is no default judge model. Set `--judge-model` or `JUDGE_MODEL` before
calling the API.
A `--dry-run` only counts pending work and does not require a model ID.
The default `--api-format gemini-responses` sends **text only** using Gemini
`contents`/`parts` to `/responses`; this is not the OpenAI Responses schema.
Supply your own provider URL including `/v1` and key. For a conventional
OpenAI-compatible judge, specify `--api-format chat-completions` and
`--judge-model YOUR_MODEL_ID`. The model can also be set with `JUDGE_MODEL`.
Changing the model or protocol requires a separate output file or an explicit
overwrite; results from different judge configurations cannot be resumed together.
Judge calls may incur API charges. Judgments require exact category identity,
not only a coarser category. Alias matches
do not require a judge call. Use `--dry-run` on the judge command to inspect
pending work. Never commit API keys, datasets, model weights, or run outputs.
The judge must return JSON with `judgment` set to `correct` or `wrong`.
HTTP failures and invalid judge responses are retried up to `--max-retries`.
Unresolved failures are saved as `judgment: error`; rerunning retries them.
They do not count as wrong answers, and corrected Open-QA scores and the
overall score remain unavailable while judgments are pending.
