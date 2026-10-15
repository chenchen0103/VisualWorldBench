#!/usr/bin/env bash
# Generate answers locally. This script does not call a paid judge API.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
: "${MODEL_ROOT:?Set MODEL_ROOT to the directory containing Qwen3-VL-8B-Instruct}"
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./results/openqa-qwen3-vl-8b}"

"$PYTHON" evaluate.py \
  --data-dir "$DATA_DIR" \
  --model "$MODEL_ROOT/Qwen3-VL-8B-Instruct" \
  --model-size 8B --model-name Qwen3-VL-8B \
  --tasks open_qa --batch-size 258 --max-tokens 2048 \
  --output "$OUTPUT_ROOT" "$@"

# --check-only must not proceed to judging or require existing predictions.
for arg in "$@"; do
  if [[ "$arg" == "--check-only" ]]; then exit 0; fi
done

PRED_FILE="$OUTPUT_ROOT/predictions/open_qa/Qwen3-VL-8B/open_qa.jsonl"
"$PYTHON" judge.py \
  --bench-jsonl "$DATA_DIR/open_qa.jsonl" --pred-file "$PRED_FILE" \
  --dry-run
printf '\nPredictions: %s\nNo judge API requests were made.\n' "$PRED_FILE"
