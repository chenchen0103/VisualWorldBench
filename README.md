# VisualWorldBench

Evaluation code for **VisualWorldBench: A Fine-Grained Multi-Task Benchmark for
Evaluating Visual World Knowledge in MLLMs**.

[Dataset](https://huggingface.co/datasets/orange0318/VisualWorldBench) ·
[Environment](docs/environment.md) · [Evaluation protocol](docs/evaluation.md) ·
[API inference and Open-QA](docs/openqa.md)

VisualWorldBench evaluates fine-grained visual recognition through four tasks:

| Task | What the model predicts | Metric |
|---|---|---|
| MCQ | Category option | Accuracy |
| Grid-Sel | Target grid cell | Accuracy |
| Grid-Loc | Target bounding box | IoU@0.5 accuracy |
| Open-QA | Category name | Alias exact match + LLM judge |

## Quick start

### 1. Set up the environment

The local Qwen3-VL runner requires Linux, eight NVIDIA GPUs, Python 3.10.19,
vLLM 0.13.0, PyTorch 2.9.0, and Transformers 4.57.3. See the
[environment instructions](docs/environment.md).

### 2. Download the dataset and a model

```bash
hf download orange0318/VisualWorldBench --repo-type dataset \
  --revision 31cd7e3d037898df2a1a94d1bb8e15ba65e9630c --local-dir ./data
hf download Qwen/Qwen3-VL-2B-Instruct --local-dir ./models/Qwen3-VL-2B-Instruct
```

The data directory must contain the task JSONL files and `images/`.

### 3. Run evaluation

```bash
python evaluate.py --data-dir ./data \
  --model ./models/Qwen3-VL-2B-Instruct --model-size 2B
```

The default run evaluates **MCQ, Grid-Sel, and Grid-Loc** (30,332 samples).
It does not call a judge or run Open-QA. Predictions, run settings, and scores
are saved under a new directory in `results/`.

To run 2B → 4B → 8B sequentially after downloading all three models:

```bash
MODEL_ROOT=./models DATA_DIR=./data bash scripts/run_qwen3_vl.sh
```

See [running Qwen3-VL](docs/running.md) for model downloads, offline use, and
input checks. Open-QA is a separate, optional step described in
[the Open-QA guide](docs/openqa.md); there is no default judge model.

### 4. Score saved predictions

```bash
python score.py --bench-dir ./data \
  --pred-dir /path/to/run/predictions --output ./scores.tsv \
  --tasks mcq,sel_2x2,sel_3x3,loc_2x2,loc_3x3
```

Scores are fractions. Missing predictions and infrastructure failures must be
resolved before scoring. The four-task overall score requires completed
Open-QA judgments. See [evaluation and scoring rules](docs/evaluation.md).

## Repository layout

```text
VisualWorldBench/
├── evaluate.py       # Local Qwen3-VL evaluation
├── infer_api.py      # OpenAI-compatible API inference
├── judge.py          # Open-QA judging, with a user-selected model
├── score.py          # Unified scoring
├── vwb/              # Inference, prompts, parsing, and scoring implementation
├── scripts/          # Ready-to-run Qwen3-VL examples
├── docs/             # Environment, usage, and evaluation details
├── tests/            # CPU regression tests using local mock APIs
├── requirements.txt  # Recorded evaluation environment
└── README.md
```

Run `python <entrypoint>.py --help` for the arguments of each command.

## Tests

```bash
python -m unittest discover -s tests -v
```

The regression tests do not load a model or call a paid API.
