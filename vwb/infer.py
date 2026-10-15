#!/usr/bin/env python3
"""VisualWorldBench inference script.

Runs inference on all four benchmark tasks using an OpenAI-compatible API.
Produces structured prediction files that can be scored by score_bench.py.

Usage:
    # OpenAI-compatible API
    OPENAI_API_KEY=your_key python3 infer_api.py \
        --bench-dir ./data --pred-dir ./predictions \
        --model gpt-4o --model-name GPT-4o

    # Custom endpoint (e.g. vLLM serve, ollama, etc.)
    OPENAI_API_KEY=your_key python3 infer_api.py \
        --bench-dir ./data --pred-dir ./predictions \
        --model my-model --model-name MyModel \
        --base-url http://localhost:8000/v1

Dependencies: aiohttp  (pip install aiohttp)
"""
import argparse
import asyncio
import base64
import json
import mimetypes
import os
import re
import time
import hashlib
import tempfile
from .prediction_validation import infrastructure_failure, read_records
from .eval_protocol import (normalize_question, build_prompt_for_task, parse_index, PROTOCOL_VERSION)
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Task type helpers ────────────────────────────────────────────

TASK_MAP = {
    "mcq":    {"task_type": "mcq",      "subdir": "mcq",      "file": "mcq.jsonl"},
    "sel_2x2": {"task_type": "grid_loc", "subdir": "sel_2x2",  "file": "sel_2x2.jsonl"},
    "sel_3x3": {"task_type": "grid_loc", "subdir": "sel_3x3",  "file": "sel_3x3.jsonl"},
    "loc_2x2": {"task_type": "grid_box", "subdir": "loc_2x2",  "file": "loc_2x2.jsonl"},
    "loc_3x3": {"task_type": "grid_box", "subdir": "loc_3x3",  "file": "loc_3x3.jsonl"},
    "open_qa": {"task_type": "open",     "subdir": "open_qa",  "file": "open_qa.jsonl"},
}


def _is_index_task(task_type: str) -> bool:
    return task_type in {"mcq", "grid_loc"}


# ── Question loading & normalization ─────────────────────────────

def load_questions(path: str, task_type: str) -> List[Dict[str, Any]]:
    questions_dir = os.path.dirname(os.path.abspath(path))
    is_grid = "grid" in os.path.basename(path).lower() or "sel_" in os.path.basename(path).lower() or "loc_" in os.path.basename(path).lower()
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            for key in ("image_path", "image", "image_cropped"):
                img = raw.get(key)
                if isinstance(img, str) and img and not os.path.isabs(img):
                    cand = os.path.join(questions_dir, img)
                    if os.path.exists(cand):
                        raw[key] = cand
            try:
                data.append(_normalize(raw, task_type, is_grid))
            except Exception as e:
                raise ValueError(f"{path}: invalid record {raw.get('qid')}: {e}") from e
    qids = [q.get("qid") for q in data]
    if any(qid is None for qid in qids) or len(set(qids)) != len(qids):
        raise ValueError("Benchmark qids must be present and unique within each task.")
    return data


def _normalize(q: Dict, task_type: str, is_grid: bool) -> Dict:
    is_grid = is_grid or task_type == "grid_loc"
    if task_type == "grid_loc" and not str(q.get("prompt") or "").strip():
        raise ValueError("Grid-Sel requires its full prompt, including the target category")
    nq = normalize_question(q, task_type=task_type, is_grid_bench=is_grid)
    _copy_fields(q, nq, ["grid_size", "target_cell", "target_class_id", "target_class_name"])
    if task_type == "open" and not nq.get("gt_name"):
        aliases = nq.get("gt_aliases_base") or []
        if not aliases:
            raise ValueError("Open-QA requires ground-truth aliases for judging")
        nq["gt_name"] = aliases[0]
    return nq


def _copy_fields(src, dst, keys):
    for k in keys:
        if k in src:
            dst[k] = src[k]


# ── Prompt construction ──────────────────────────────────────────

def build_prompt(q: Dict, task_type: str, open_prompt_style: str = "open_reason") -> str:
    style = open_prompt_style if task_type == "open" else "plain"
    return build_prompt_for_task(q, task_type, style)


# ── Response parsing ─────────────────────────────────────────────




def finalize_result(task_type: str, base: Dict, output_text: str) -> Dict:
    if _is_index_task(task_type):
        options = base.get("options") or []
        answer_index = int(base["answer_index"])
        pred_index = parse_index(output_text, len(options))
        correct = pred_index == answer_index if pred_index is not None else False
        reason = "ok" if correct else ("parse_fail" if pred_index is None else "wrong")
        return {**base, "pred_index": pred_index, "model_output": output_text,
                "reason": reason, "correct": correct, "counted": True}
    return {**base, "model_output": output_text, "reason": "ok", "correct": False, "counted": False}


# ── Image encoding ───────────────────────────────────────────────

def image_to_data_url(image_path: str) -> Optional[str]:
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


# ── API call with retries ────────────────────────────────────────

async def api_call(session, url, headers, payload, *, timeout_sec=120, max_retries=3):
    import aiohttp
    for attempt in range(max_retries + 1):
        try:
            async with session.post(url, headers=headers, json=payload,
                                    timeout=aiohttp.ClientTimeout(total=timeout_sec)) as resp:
                if resp.status == 429 or resp.status >= 500:
                    if attempt < max_retries:
                        retry_after = float(resp.headers.get("Retry-After", 0))
                        wait = max(retry_after, min(2 ** attempt, 8.0))
                        await asyncio.sleep(wait)
                        continue
                resp.raise_for_status()
                return await resp.json()
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            if attempt < max_retries:
                await asyncio.sleep(min(2 ** attempt, 8.0))
                continue
            raise
    raise RuntimeError(f"Failed after {max_retries + 1} attempts")


def extract_text(resp_json: Dict) -> str:
    choices = resp_json.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            return str(content.get("text") or "")
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            return "\n".join(parts)
    return ""


# ── Main inference loop ──────────────────────────────────────────

async def run_task(
    task_key: str,
    bench_path: str,
    pred_dir: str,
    model_name: str,
    *,
    model: str,
    base_url: str,
    api_key: str,
    num_workers: int,
    max_tokens: int,
    temperature: float,
    timeout_sec: int,
    max_retries: int,
    resume: bool,
    open_prompt_style: str = "open_reason",
):
    import aiohttp

    info = TASK_MAP[task_key]
    task_type = info["task_type"]
    subdir = info["subdir"]
    pred_file = info["file"]

    questions = load_questions(bench_path, task_type)
    if not questions:
        print(f"  [{task_key}] no questions loaded, skip")
        return

    out_dir = Path(pred_dir) / subdir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / pred_file

    config = {
        "protocol_version": PROTOCOL_VERSION, "model": model,
        "base_url": base_url, "temperature": temperature, "max_tokens": max_tokens,
        "task": task_key, "open_prompt_style": open_prompt_style,
        "bench_sha256": hashlib.sha256(Path(bench_path).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(Path(__file__).with_name("eval_protocol.py").read_bytes()).hexdigest(),
    }
    config_path = out_path.with_suffix(".run.json")
    if resume and out_path.exists() and out_path.stat().st_size:
        if not config_path.exists() or json.loads(config_path.read_text()) != config:
            raise ValueError("Existing predictions use an unknown or different evaluation configuration. "
                             "Use a new --pred-dir, or --no-resume to explicitly replace them.")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    successful = {}
    if resume and out_path.exists():
        expected_qids = {q["qid"] for q in questions}
        for obj in read_records(out_path, recover_truncated_tail=True):
            qid = obj["qid"]
            if qid not in expected_qids:
                raise ValueError(f"Existing prediction qid is not in benchmark: {qid}")
            if not infrastructure_failure(obj) and not (
                _is_index_task(task_type) and obj.get("counted") is False
            ):
                successful[qid] = obj
        # Compact retries to one successful record per qid. Atomic replacement
        # preserves the original file if writing the compacted file fails.
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out_dir,
                                             prefix=".resume-", delete=False) as tmp:
                temp_path = Path(tmp.name)
                for obj in successful.values():
                    tmp.write(json.dumps(obj, ensure_ascii=False) + "\n")
            os.replace(temp_path, out_path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
    resume_qids = set(successful)
    if resume_qids:
        print(f"  [{task_key}] resuming: {len(resume_qids)} successful responses retained")

    pending = [(i, q) for i, q in enumerate(questions) if q.get("qid") not in resume_qids]
    if not pending:
        print(f"  [{task_key}] all {len(questions)} done, skip")
        return

    print(f"  [{task_key}] {len(pending)} samples to run ({len(resume_qids)} resumed)")

    endpoint = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    correct = 0
    total = 0
    done_count = 0
    failure_count = 0
    start_t = time.time()

    in_q: asyncio.Queue = asyncio.Queue()
    out_q: asyncio.Queue = asyncio.Queue()
    for item in pending:
        in_q.put_nowait(item)
    for _ in range(num_workers):
        in_q.put_nowait(None)

    connector = aiohttp.TCPConnector(limit=num_workers * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        async def worker():
            while True:
                item = await in_q.get()
                if item is None:
                    in_q.task_done()
                    return
                idx, q = item
                try:
                    result = await eval_one(session, endpoint, headers, idx, q,
                                            task_type=task_type, model=model,
                                            max_tokens=max_tokens, temperature=temperature,
                                            timeout_sec=timeout_sec, max_retries=max_retries,
                                            open_prompt_style=open_prompt_style)
                except Exception as e:
                    base = {"idx": idx, "qid": q.get("qid"), "image_path": q.get("image_path")}
                    if "options" in q:
                        base["options"] = q["options"]
                    if "answer_index" in q:
                        base["answer_index"] = q["answer_index"]
                    if "bbox_norm" in q:
                        base["bbox_norm"] = q["bbox_norm"]
                    _copy_fields(q, base, ["gt_id", "gt_name", "meta", "gt_aliases_base",
                                            "gt_aliases_expanded", "gt_ancestors"])
                    result = {**base, "model_output": "", "reason": "request_error",
                              "exception": repr(e), "correct": False, "counted": False}
                await out_q.put(result)
                in_q.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(num_workers)]

        mode = "a" if resume and out_path.exists() else "w"
        with open(out_path, mode, encoding="utf-8") as f:
            remaining = len(pending)
            while remaining > 0:
                result = await out_q.get()
                if infrastructure_failure(result):
                    failure_count += 1
                remaining -= 1
                done_count += 1
                if result.get("counted"):
                    total += 1
                    if result.get("correct"):
                        correct += 1
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                if done_count % 100 == 0 or remaining == 0:
                    elapsed = time.time() - start_t
                    rate = done_count / elapsed if elapsed > 0 else 0
                    acc_str = f" acc={correct/total:.4f}" if total > 0 else ""
                    print(f"  [{task_key}] {done_count}/{len(pending)} ({rate:.1f}/s){acc_str}")

        await asyncio.gather(*workers)

    acc_str = f" acc={correct/total:.4f}" if total > 0 else ""
    print(f"  [{task_key}] done: {len(pending)} samples{acc_str} -> {out_path}")
    if failure_count:
        raise RuntimeError(f"{task_key}: {failure_count} failed requests; rerun the same command "
                           "to retry them while keeping successful responses")


async def eval_one(session, endpoint, headers, idx, q, *,
                   task_type, model, max_tokens, temperature, timeout_sec, max_retries,
                   open_prompt_style="open_reason"):
    image_path = q["image_path"]
    prompt_text = build_prompt(q, task_type, open_prompt_style)
    data_url = await asyncio.to_thread(image_to_data_url, image_path)
    if not data_url:
        base = {"idx": idx, "qid": q.get("qid"), "image_path": image_path}
        _copy_fields(q, base, ["options", "answer_index", "bbox_norm", "gt_id", "gt_name", "meta",
                                "gt_aliases_base", "gt_aliases_expanded", "gt_ancestors"])
        return {**base, "model_output": "", "reason": "request_error",
                "correct": False, "counted": False}

    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]}]
    payload = {"model": model, "stream": False, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}

    resp_json = await api_call(session, endpoint, headers, payload,
                               timeout_sec=timeout_sec, max_retries=max_retries)
    output_text = extract_text(resp_json)

    base = {"idx": idx, "qid": q.get("qid"), "image_path": q.get("image_path")}
    _copy_fields(q, base, ["options", "answer_index", "bbox_norm", "gt_id", "gt_name", "meta",
                            "grid_size", "target_cell", "target_class_id", "target_class_name",
                            "gt_aliases_base", "gt_aliases_expanded", "gt_ancestors",
                            "source_qid", "source_file"])
    base["prompt"] = prompt_text
    return finalize_result(task_type, base, output_text)


# ── Entry point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run VisualWorldBench inference via OpenAI-compatible API.")
    parser.add_argument("--bench-dir", required=True, help="Directory containing benchmark JSONL files.")
    parser.add_argument("--pred-dir", default="./predictions", help="Output predictions directory.")
    parser.add_argument("--model", required=True, help="Model identifier for the API (e.g. gpt-4o).")
    parser.add_argument("--model-name", default=None, help="Display name for output subdirectory (defaults to --model).")
    parser.add_argument("--base-url", default="https://api.openai.com/v1", help="API base URL.")
    parser.add_argument("--api-key", default=None, help="API key (or set OPENAI_API_KEY env var).")
    parser.add_argument("--tasks", default="all", help="Comma-separated task keys to run, or 'all'.")
    parser.add_argument("--num-workers", type=int, default=32, help="Concurrent API workers.")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max output tokens; match the model-specific experimental setting.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--timeout", type=int, default=120, help="Per-request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per request.")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing predictions.")
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--open-prompt-style", choices=["open_reason", "plain"], default="open_reason")
    args = parser.parse_args()
    if args.num_workers < 1 or args.max_tokens < 1 or args.max_retries < 0:
        parser.error("--num-workers and --max-tokens must be positive; --max-retries must be nonnegative")

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        parser.error("API key required: set OPENAI_API_KEY or pass --api-key")

    model_name = args.model_name or args.model.strip("/").replace("/", "--")
    if model_name in {".", ".."} or "/" in model_name or "\\" in model_name:
        parser.error("--model-name must be a single directory name")
    bench_dir = Path(args.bench_dir)

    if args.tasks == "all":
        task_keys = list(TASK_MAP.keys())
    else:
        task_keys = [t.strip() for t in args.tasks.split(",")]

    print(f"Model: {model_name} ({args.model})")
    print(f"Base URL: {args.base_url}")
    print(f"Tasks: {', '.join(task_keys)}")
    print(f"Workers: {args.num_workers}, Max tokens: {args.max_tokens}")
    print()

    for task_key in task_keys:
        if task_key not in TASK_MAP:
            print(f"[WARN] unknown task: {task_key}, skip")
            continue
        bench_file = bench_dir / TASK_MAP[task_key]["file"]
        if not bench_file.exists():
            print(f"[WARN] bench file not found: {bench_file}, skip")
            continue
        asyncio.run(run_task(
            task_key, str(bench_file), args.pred_dir, model_name,
            model=args.model, base_url=args.base_url, api_key=api_key,
            num_workers=args.num_workers, max_tokens=args.max_tokens,
            temperature=args.temperature, timeout_sec=args.timeout,
            max_retries=args.max_retries, resume=args.resume,
            open_prompt_style=args.open_prompt_style,
        ))

    print("\nAll tasks complete.")


if __name__ == "__main__":
    main()
