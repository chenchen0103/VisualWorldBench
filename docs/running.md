# Running Qwen3-VL

Run commands from the repository root. Relative data, model, and output paths
in the shell launchers are resolved from the repository root.


Download the desired Instruct checkpoint(s):

```bash
hf download Qwen/Qwen3-VL-2B-Instruct --local-dir ./models/Qwen3-VL-2B-Instruct
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir ./models/Qwen3-VL-4B-Instruct
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir ./models/Qwen3-VL-8B-Instruct
```

Run **2B → 4B → 8B sequentially** on eight available GPUs:

```bash
MODEL_ROOT=./models DATA_DIR=./data bash scripts/run_qwen3_vl.sh
```

To combine dataset download, input checks, and the same sequential evaluation:

```bash
MODEL_ROOT=./models bash scripts/download_and_evaluate.sh
```

This uses the dataset revision pinned in the launcher and writes a new run directory
with `run.log` and per-model scores. Set `PYTHON=/path/to/environment/bin/python`
to select an existing environment. Add `CHECK_ONLY=1` to download and validate
without inference.

On a compute node without internet access, download the data on a connected
machine into a shared directory first, then run with `SKIP_DOWNLOAD=1` and
`DATA_DIR=/path/to/downloaded/data`. This enables offline mode and checks all
selected inputs before starting inference.

Or run one model:

```bash
python evaluate.py --data-dir ./data \
  --model ./models/Qwen3-VL-2B-Instruct --model-size 2B
```

Add `--check-only` to validate data and paths without loading a model, or
`--tasks sel_2x2,sel_3x3` to select tasks. `--gpu-ids 0,1,2,3,4,5,6,7`
selects available physical devices. The runner never stops other GPU jobs.

The default run covers 30,332 questions: MCQ 14,717; Grid-Sel 4,085/4,125;
Grid-Loc 3,959/3,446. **No Open-QA or judge calls are made by this launcher.**
Predictions and `scores.tsv` are saved under `results/`. Each run requires a
new output directory. Missing outputs and infrastructure failures stop the run.
