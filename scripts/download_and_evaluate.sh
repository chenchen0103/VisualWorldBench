#!/usr/bin/env bash
# Download the pinned dataset, validate inputs, then evaluate 2B -> 4B -> 8B.
# Required: MODEL_ROOT. Optional: PYTHON, DATA_DIR, MODEL_SIZES, OUTPUT_ROOT,
# CUDA_VISIBLE_DEVICES, HF_CLI, CHECK_ONLY=1, SKIP_DOWNLOAD=1.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${MODEL_ROOT:?Set MODEL_ROOT to the directory containing Qwen3-VL-2B/4B/8B-Instruct}"
export PYTHON="${PYTHON:-python}"
PYTHON_BIN="$(command -v "$PYTHON")" || {
  printf 'Python not found: %s\n' "$PYTHON" >&2
  exit 1
}
HF_CLI="${HF_CLI:-$(dirname "$PYTHON_BIN")/hf}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
if [[ "$SKIP_DOWNLOAD" != 0 && "$SKIP_DOWNLOAD" != 1 ]]; then
  printf 'SKIP_DOWNLOAD must be 0 or 1.\n' >&2
  exit 1
fi
if [[ "$SKIP_DOWNLOAD" == 0 && ! -x "$HF_CLI" ]]; then
  printf 'Hugging Face CLI not found: %s\nInstall requirements.txt in the selected Python environment.\n' "$HF_CLI" >&2
  exit 1
fi

export DATA_DIR="${DATA_DIR:-$PWD/data}"
export MODEL_SIZES="${MODEL_SIZES:-2B 4B 8B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/results/qwen3-vl-$(date +%Y%m%d-%H%M%S)-$$}"
DATASET_REVISION=31cd7e3d037898df2a1a94d1bb8e15ba65e9630c

if [[ -z "${MODEL_SIZES// /}" ]]; then
  printf 'MODEL_SIZES must not be empty.\n' >&2
  exit 1
fi
seen_sizes=" "
for size in $MODEL_SIZES; do
  case "$size" in
    2B|4B|8B) ;;
    *) printf 'Unsupported model size: %s\n' "$size" >&2; exit 1 ;;
  esac
  if [[ "$seen_sizes" == *" $size "* ]]; then
    printf 'Duplicate model size: %s\n' "$size" >&2
    exit 1
  fi
  seen_sizes="$seen_sizes$size "
  if [[ ! -f "$MODEL_ROOT/Qwen3-VL-$size-Instruct/config.json" ]]; then
    printf 'Model not found: %s/Qwen3-VL-%s-Instruct/config.json\n' "$MODEL_ROOT" "$size" >&2
    exit 1
  fi
done

# Refuse an existing run directory, including an incomplete earlier run.
mkdir -p "$(dirname "$OUTPUT_ROOT")"
mkdir "$OUTPUT_ROOT" || {
  printf 'Choose a new OUTPUT_ROOT; refusing to overwrite: %s\n' "$OUTPUT_ROOT" >&2
  exit 1
}

{
  printf 'Output directory: %s\n' "$OUTPUT_ROOT"
  if [[ "$SKIP_DOWNLOAD" == 1 ]]; then
    printf '\n[1/3] Using local dataset: %s\n' "$DATA_DIR"
    [[ -d "$DATA_DIR" ]] || { printf 'Local data directory does not exist.\n' >&2; exit 1; }
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
  else
    printf '\n[1/3] Downloading VisualWorldBench (%s)\n' "$DATASET_REVISION"
    if ! "$HF_CLI" download orange0318/VisualWorldBench \
      --repo-type dataset --revision "$DATASET_REVISION" --local-dir "$DATA_DIR"; then
      printf '\nDataset download failed. Check the network connection and dataset revision.\n' >&2
      exit 1
    fi
  fi

  printf '\n[2/3] Checking all selected models and task inputs (no inference)\n'
  bash scripts/run_qwen3_vl.sh --check-only --batch-size 258 --max-tokens 2048

  if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
    printf '\nChecks passed. CHECK_ONLY=1: no GPU inference started.\n'
  else
    printf '\n[3/3] Running %s sequentially; Open-QA is excluded\n' "$MODEL_SIZES"
    bash scripts/run_qwen3_vl.sh --batch-size 258 --max-tokens 2048
    printf '\nFinished. Score files:\n'
    for size in $MODEL_SIZES; do
      printf '  %s/Qwen3-VL-%s/scores.tsv\n' "$OUTPUT_ROOT" "$size"
    done
  fi
} 2>&1 | tee "$OUTPUT_ROOT/run.log"
