#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_ROOT:?Set MODEL_ROOT to the directory containing Qwen3-VL-2B/4B/8B-Instruct}"
PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-./data}"
MODEL_SIZES="${MODEL_SIZES:-2B 4B 8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./results/qwen3-vl-$(date +%Y%m%d-%H%M%S)-$$}"
for size in $MODEL_SIZES; do
  "$PYTHON" evaluate.py \
    --data-dir "$DATA_DIR" \
    --model "$MODEL_ROOT/Qwen3-VL-${size}-Instruct" \
    --model-size "$size" \
    --output "$OUTPUT_ROOT/Qwen3-VL-$size" "$@"
done
