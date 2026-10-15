# Environment

Run all commands from the repository root. The Qwen3-VL runner requires Linux,
eight NVIDIA GPUs, Python 3.10.19, vLLM 0.13.0, PyTorch 2.9.0 with CUDA 12.8,
and Transformers 4.57.3. The recorded GPU setup is 8×H200 with driver 580.95.05.

[`requirements.txt`](../requirements.txt) pins the installed package versions.
Some packages declare conflicting dependencies (including Pydantic and NumPy);
use `--no-deps` in an isolated environment to install the pinned versions:

```bash
conda create -n visualworldbench python=3.10.19 -y
conda activate visualworldbench
python -m pip install --no-deps -r requirements.txt
hf download orange0318/VisualWorldBench --repo-type dataset \
  --revision 31cd7e3d037898df2a1a94d1bb8e15ba65e9630c --local-dir ./data
```

The data directory must contain the task JSONL files and `images/`.
Set `PYTHON=/path/to/environment/bin/python` when running the shell scripts with
a different Python environment.
