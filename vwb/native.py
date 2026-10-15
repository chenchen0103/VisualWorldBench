"""Native offline inference core used by the Qwen3-VL evaluation runner."""
import json
import multiprocessing as mp
import os
import queue
import re
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple
from .scoreboard import update_score_tsv

_INFRA_FAILURE_REASONS = {"request_error", "image_open_fail", "image_to_data_url_fail", "worker_exception"}


def _check_writer_processes(*writers):
    for proc in writers:
        if proc is not None and proc.exitcode is not None and proc.exitcode != 0:
            raise RuntimeError(
                f"Result writer exited unexpectedly (pid={proc.pid}, exitcode={proc.exitcode}); "
                "stopping inference because results cannot be saved."
            )


def _is_special_path(p: str) -> bool:
    p = (p or "").strip().lower()
    return p.startswith("http://") or p.startswith("https://") or p.startswith("data:")


def resolve_image_path(image_path, *, meta, questions_dir):
    p = (image_path or "").strip()
    if not p or os.path.isabs(p) or _is_special_path(p):
        return p
    candidate = os.path.join(questions_dir, p)
    if os.path.exists(candidate):
        return candidate
    if meta and meta.get("image_root"):
        candidate = os.path.join(meta["image_root"], p)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(questions_dir, p)



def _is_index_task(task_type: str) -> bool:
    return task_type in {"mcq", "grid_loc"}


def _copy_optional_fields(src: Dict[str, Any], dst: Dict[str, Any], fields: List[str]) -> None:
    for key in fields:
        if key in src:
            dst[key] = src[key]


def normalize_question(
    q: Dict[str, Any],
    *,
    task_type: str = "mcq",
    is_grid_bench: bool = False,
) -> Dict[str, Any]:
    """
    支持三种输入：
    A) 旧格式：image_path + options(list[str]) + answer_index
    B) 新bench格式：image + options(list[dict{id,name,type}]) + gt_index
    C) grid定位格式：image + prompt + choices(list[str]) + answer("1".."N")
    输出统一成：
      image_path: str
      options: list[str]
      answer_index: int
      (可选保留原始信息: qid, gt_id, meta...)
    """
    if task_type == "grid_box":
        image_path = q.get("image_path") or q.get("image")
        if not image_path:
            raise ValueError("Missing image_path/image")
        prompt = q.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            task_box = (q.get("tasks") or {}).get("box") or {}
            prompt = task_box.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Missing prompt for grid_box")
        bbox_norm = q.get("bbox_norm")
        if bbox_norm is None:
            bbox_norm = q.get("answer")
        if bbox_norm is None and isinstance(q.get("meta"), dict):
            bbox_norm = q["meta"].get("bbox_norm")
        if bbox_norm is None:
            task_box = (q.get("tasks") or {}).get("box") or {}
            bbox_norm = task_box.get("answer")
        if not isinstance(bbox_norm, list) or len(bbox_norm) != 4:
            raise ValueError("Missing bbox_norm for grid_box")
        nq = {
            "image_path": image_path,
            "prompt": prompt.strip(),
            "bbox_norm": [float(v) for v in bbox_norm],
        }
        _copy_optional_fields(
            q,
            nq,
            [
                "qid",
                "gt_id",
                "gt_name",
                "meta",
                "class_id",
                "class_name",
                "grid_id",
                "grid_size",
                "bbox_abs",
                "target_cell",
                "target_class_id",
                "target_class_name",
                "answer",
            ],
        )
        return nq

    if task_type == "open":
        image_path = q.get("image_cropped") or q.get("image_path") or q.get("image")
        if not image_path:
            raise ValueError("Missing image_cropped/image_path/image")
        prompt = q.get("question") or q.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Missing question/prompt for open task")
        nq = {
            "image_path": image_path,
            "prompt": prompt.strip(),
        }
        _copy_optional_fields(
            q,
            nq,
            [
                "qid",
                "gt_id",
                "gt_name",
                "meta",
                "gt_aliases_base",
                "gt_aliases_expanded",
                "gt_ancestors",
                "source_qid",
                "source_file",
            ],
        )
        meta = q.get("meta")
        if isinstance(meta, dict):
            if "source_qid" not in nq and meta.get("source_qid") is not None:
                nq["source_qid"] = meta.get("source_qid")
            if "source_file" not in nq and meta.get("source_file") is not None:
                nq["source_file"] = meta.get("source_file")
        return nq

    # image path
    image_path = q.get("image_path") or q.get("image")
    if not image_path:
        raise ValueError("Missing image_path/image")

    # options/choices -> list[str]
    opts = q.get("options")
    if (not isinstance(opts, list) or len(opts) == 0) and is_grid_bench:
        opts = q.get("choices")
    if not isinstance(opts, list) or len(opts) == 0:
        raise ValueError("Missing options/choices")

    raw_options = None
    if isinstance(opts[0], dict):
        # Structured options: [{"id":..., "name":..., ...}, ...]
        raw_options = opts
        options = [str(x.get("name", x.get("id", ""))) for x in opts]
    else:
        # String options: ["cat", "dog", ...]
        options = [str(x) for x in opts]

    # answer index
    answer_index = q.get("answer_index")
    if answer_index is None:
        answer_index = q.get("gt_index")
    if answer_index is None and is_grid_bench:
        answer = q.get("answer")
        if answer is not None:
            if isinstance(answer, (int, float)):
                ai = int(answer)
                if 0 <= ai < len(options):
                    answer_index = ai
                elif 1 <= ai <= len(options):
                    answer_index = ai - 1
                elif str(ai) in options:
                    answer_index = options.index(str(ai))
            else:
                s = str(answer).strip()
                if s in options:
                    answer_index = options.index(s)
                elif re.fullmatch(r"-?\d+", s):
                    ai = int(s)
                    if 1 <= ai <= len(options):
                        answer_index = ai - 1
                    elif 0 <= ai < len(options):
                        answer_index = ai
    if answer_index is None:
        raise ValueError("Missing answer_index/gt_index/answer")
    answer_index = int(answer_index)
    if not (0 <= answer_index < len(options)):
        raise ValueError(f"answer_index out of range: {answer_index}, n_options={len(options)}")

    nq = {
        "image_path": image_path,
        "options": options,
        "answer_index": answer_index,
    }
    if raw_options is not None:
        nq["raw_options"] = raw_options
    if is_grid_bench and "prompt" in q and isinstance(q.get("prompt"), str):
        nq["prompt"] = q.get("prompt")

    # 可选：保留一些信息方便 error log
    for k in ("qid", "gt_id", "gt_name", "meta", "class_id", "class_name", "grid_id", "bbox_abs"):
        if k in q:
            nq[k] = q[k]
    return nq


def load_questions(path: str, *, task_type: str = "mcq") -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    questions_dir = os.path.dirname(os.path.abspath(path))
    is_grid_bench_file = "grid" in os.path.basename(path).lower()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            # Grid benches are primarily identified by filename, but we also
            # fallback to schema-based detection for files like bench_hard_2x2_loc.jsonl.
            is_grid_bench = is_grid_bench_file
            if isinstance(raw, dict) and not is_grid_bench:
                if isinstance(raw.get("choices"), list) and raw.get("answer") is not None:
                    is_grid_bench = True
            # Resolve relative image path early so both vLLM/OpenAI paths work.
            if isinstance(raw, dict):
                meta = raw.get("meta")
                if not isinstance(meta, dict):
                    meta = None
                for key in ("image_path", "image", "image_cropped"):
                    img = raw.get(key)
                    if isinstance(img, str) and img:
                        raw[key] = resolve_image_path(img, meta=meta, questions_dir=questions_dir)
            try:
                data.append(normalize_question(raw, task_type=task_type, is_grid_bench=is_grid_bench))
            except Exception as e:
                # 你也可以选择 print 或跳过
                print(f"[WARN] skip bad question: {repr(e)}")
    return data


def build_task_base_record(backend: str, idx: int, q: Dict[str, Any]) -> Dict[str, Any]:
    rec = {
        "backend": backend,
        "idx": idx,
        "qid": q.get("qid"),
        "gt_id": q.get("gt_id"),
        "gt_name": q.get("gt_name"),
        "meta": q.get("meta"),
        "image_path": q.get("image_path"),
    }
    if "options" in q:
        rec["options"] = q.get("options")
    if "raw_options" in q:
        rec["raw_options"] = q.get("raw_options")
    if "answer_index" in q:
        rec["answer_index"] = q.get("answer_index")
    if "bbox_norm" in q:
        rec["bbox_norm"] = q.get("bbox_norm")
    if "bbox_abs" in q:
        rec["bbox_abs"] = q.get("bbox_abs")
    if "grid_size" in q:
        rec["grid_size"] = q.get("grid_size")
    if "target_cell" in q:
        rec["target_cell"] = q.get("target_cell")
    if "target_class_id" in q:
        rec["target_class_id"] = q.get("target_class_id")
    if "target_class_name" in q:
        rec["target_class_name"] = q.get("target_class_name")
    if "gt_aliases_base" in q:
        rec["gt_aliases_base"] = q.get("gt_aliases_base")
    if "gt_aliases_expanded" in q:
        rec["gt_aliases_expanded"] = q.get("gt_aliases_expanded")
    if "gt_ancestors" in q:
        rec["gt_ancestors"] = q.get("gt_ancestors")
    if "source_qid" in q:
        rec["source_qid"] = q.get("source_qid")
    if "source_file" in q:
        rec["source_file"] = q.get("source_file")
    return rec


def finalize_task_result(task_type: str, base: Dict[str, Any], output_text: str, prompt_style: str = "plain") -> Dict[str, Any]:
    if _is_index_task(task_type):
        options = base.get("options") or []
        answer_index = int(base["answer_index"])
        pred_index = parse_index(output_text, len(options), prompt_style)
        if pred_index is None:
            rec = {
                **base,
                "pred_index": None,
                "model_output": output_text,
                "reason": "parse_fail",
                "correct": False,
                "counted": True,
            }
            return {"ok": False, "counted": True, "error": rec, "pred": rec}
        if pred_index == answer_index:
            pred = {
                **base,
                "pred_index": pred_index,
                "correct": True,
                "model_output": output_text,
                "reason": "ok",
                "counted": True,
            }
            return {"ok": True, "counted": True, "error": None, "pred": pred}
        rec = {
            **base,
            "pred_index": pred_index,
            "model_output": output_text,
            "reason": "wrong",
            "correct": False,
            "counted": True,
        }
        return {"ok": False, "counted": True, "error": rec, "pred": rec}

    pred = {
        **base,
        "pred_index": None,
        "model_output": output_text,
        "reason": "ok",
        "correct": False,
        "counted": False,
    }
    return {"ok": True, "counted": False, "error": None, "pred": pred}


def build_prompt(options: List[str], prompt_style: str = "plain", prompt_override: Optional[str] = None, meta: Optional[Dict] = None) -> str:
    if prompt_override and prompt_override.strip():
        base = prompt_override.strip()
        if prompt_style == "think":
            return (
                f"{base}\n"
                "First provide concise visual reasoning, then provide the final option number.\n"
                "Use exact output format:\n"
                "<think>...</think>\n"
                "<answer>number</answer>"
            )
        return base

    # iNat21 prompt style (uses letter options A, B, C, D)
    if prompt_style == "inat21":
        prompt_order = meta.get("prompt_order", 1) if meta else 1

        # Determine organism type from dataset metadata
        dataset_name = meta.get("dataset", "") if meta else ""
        if "plant" in dataset_name:
            organism = "plant"
        else:
            organism = "animal"

        # Prompt templates (organism type determined by dataset)
        prompt_templates = {
            0: f"What is the taxonomic category of the {organism} in this image?",
            1: f"How can the {organism} in this image be categorized in taxonomy?",
            2: f"Based on classification, what type of {organism} is this?",
            3: f"What is the hierarchical class of the {organism} shown here?",
            4: f"Where does this {organism} belong in the taxonomic hierarchy?"
        }

        question = prompt_templates.get(prompt_order, prompt_templates[0])

        # Use letter options (A, B, C, D)
        letters = ["A", "B", "C", "D"]
        option_text = "\n".join([f"{letters[i]}. {name}" for i, name in enumerate(options)])

        return (
            f"{question}\n"
            f"{option_text}\n"
            "Answer with the option's letter from the given choices directly."
        )

    n = len(options)
    if prompt_style == "think":
        option_text = "\n".join([f"{i}. {name}" for i, name in enumerate(options, 1)])
        return (
            "<image>\n"
            f"Choose the correct category label for the image 1-{n}.\n"
            "First provide concise visual reasoning, then provide the final option number.\n"
            "Use exact output format:\n"
            "<think>...</think>\n"
            "<answer>number</answer>\n"
            f"{option_text}"
        )
    option_text = "\n".join([f"{i}: {name}" for i, name in enumerate(options, 1)])
    return (
        f"Look at the object(s) in the image and choose the single best category from 1-{n}. "
        "Only output the number.\n"
        f"{option_text}"
    )


def _enforce_grid_box_output_contract(prompt: str) -> str:
    prompt = (prompt or "").strip()
    contract = (
        "Return exactly one bounding box for the target object.\n"
        "Your entire reply must be exactly one single-line JSON array: [x1, y1, x2, y2]\n"
        "Rules:\n"
        "- Output exactly one JSON array with four numeric coordinates.\n"
        "- Coordinates must be normalized to [0,1000].\n"
        "- Do not output markdown, code fences, labels, keys, comments, explanations, prefixes, or suffixes.\n"
        "- Do not output multiple boxes, alternatives, candidate answers, or reasoning.\n"
        "- If uncertain, still output your single best box in the required format."
    )
    return f"{prompt}\n{contract}" if prompt else contract


def build_prompt_for_task(q: Dict[str, Any], task_type: str, prompt_style: str) -> str:
    if _is_index_task(task_type):
        return build_prompt(q["options"], prompt_style, q.get("prompt"), q.get("meta"))
    prompt = str(q["prompt"]).strip()
    if task_type == "grid_box":
        return _enforce_grid_box_output_contract(prompt)
    if task_type == "open" and prompt_style in {"prompt_reason", "open_reason"}:
        return (
            "Identify the object in the image as specifically as possible.\n"
            f"{prompt}\n"
            "First give one brief sentence describing the key visual evidence.\n"
            "Then give the final answer on a new line in the format:\n"
            "Answer: <category name>\n"
            "Keep the evidence brief.\n"
            "Do not add anything after the answer line."
        )
    return prompt


def _apply_chat_template(processor, messages, *, thinking_mode: str = "auto", **kwargs):
    if thinking_mode not in {"auto", "on", "off"}:
        raise ValueError(f"Unexpected thinking_mode: {thinking_mode}")

    if thinking_mode == "auto":
        return processor.apply_chat_template(messages, **kwargs)

    try:
        return processor.apply_chat_template(
            messages,
            enable_thinking=(thinking_mode == "on"),
            **kwargs,
        )
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        return processor.apply_chat_template(messages, **kwargs)


def parse_index(text: str, n_options: int, prompt_style: str = "plain") -> Optional[int]:
    stripped = (text or "").strip()

    # iNat21 uses letter options (A, B, C, D)
    if prompt_style == "inat21":
        # Try to extract letter answer
        letter_patterns = [
            r"(?is)<answer>\s*([A-D])\s*</answer>",
            r"(?is)<final_answer>\s*([A-D])\s*</final_answer>",
            r"(?im)^\s*(?:answer|final\s+answer|答案|最终答案)\s*[:：]?\s*([A-D])\s*$",
            r"(?is)(?:answer|final\s+answer|答案|最终答案)\s*(?:is|=|:|：)\s*([A-D])",
        ]
        for pat in letter_patterns:
            m = re.search(pat, stripped)
            if m:
                letter = m.group(1).upper()
                idx = ord(letter) - ord('A')
                if 0 <= idx < n_options:
                    return idx

        # Try standalone letter on its own line
        for line in reversed(stripped.splitlines()):
            line = line.strip()
            if re.fullmatch(r"[A-D]", line, re.IGNORECASE):
                letter = line.upper()
                idx = ord(letter) - ord('A')
                if 0 <= idx < n_options:
                    return idx

        # Fallback: find any letter A-D in the text
        for m in re.finditer(r"\b([A-D])\b", stripped, re.IGNORECASE):
            letter = m.group(1).upper()
            idx = ord(letter) - ord('A')
            if 0 <= idx < n_options:
                return idx

        return None

    # Number-based parsing
    if re.fullmatch(r"-?\d+", stripped):
        i = int(stripped)
        return i - 1 if 1 <= i <= n_options else None

    def _to_idx(val: int) -> Optional[int]:
        return val - 1 if 1 <= val <= n_options else None

    # Highest-priority: explicit answer tags/fields.
    tagged_patterns = [
        r"(?is)<answer>\s*(-?\d+)\s*</answer>",
        r"(?is)<final_answer>\s*(-?\d+)\s*</final_answer>",
        r"(?im)^\s*(?:answer|final\s+answer|答案|最终答案|选项)\s*[:：]?\s*(?:option\s*)?(-?\d+)\s*$",
        r"(?is)(?:answer|final\s+answer|答案|最终答案)\s*(?:is|=|:|：)\s*(?:option\s*)?(-?\d+)",
    ]
    for pat in tagged_patterns:
        m = re.search(pat, stripped)
        if not m:
            continue
        idx = _to_idx(int(m.group(1)))
        if idx is not None:
            return idx

    # Some models answer in a single line like "3: class name".
    # Accept this compact form, but only when the whole output is a single
    # non-empty line to avoid mis-parsing copied option lists.
    nonempty_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(nonempty_lines) == 1:
        m = re.fullmatch(r"(?:option\s*)?(-?\d+)\s*[:：]\s*.+", nonempty_lines[0], re.IGNORECASE)
        if m:
            idx = _to_idx(int(m.group(1)))
            if idx is not None:
                return idx

    # Prefer a standalone integer on its own line (robust against extra text).
    for line in reversed(stripped.splitlines()):
        line = line.strip()
        if re.fullmatch(r"-?\d+", line):
            i = int(line)
            idx = _to_idx(i)
            if idx is not None:
                return idx

    # Fallback: extract integers but ignore ones that look like option list indices (e.g. "1: cat").
    candidates: list[int] = []
    for m in re.finditer(r"-?\d+", stripped):
        val = int(m.group(0))
        # Skip if immediately followed by ":" (option list "k: xxx").
        j = m.end()
        while j < len(stripped) and stripped[j].isspace():
            j += 1
        if j < len(stripped) and stripped[j] == ":":
            continue
        if 1 <= val <= n_options:
            candidates.append(val)
    if not candidates:
        return None
    # Use the first candidate as a safer fallback for CoT outputs that contain
    # many trailing numbers (e.g., coordinates or counts in the reasoning).
    return candidates[0] - 1


def _dataset_names(questions_path: str) -> Tuple[str, str]:
    dataset_filename = os.path.basename(questions_path)
    dataset_stem = os.path.splitext(dataset_filename)[0]
    return dataset_filename, dataset_stem


def _resolved_dataset_stem(questions_path: str, dataset_stem_override: str = "") -> str:
    _, dataset_stem = _dataset_names(questions_path)
    override = (dataset_stem_override or "").strip()
    return override or dataset_stem


def _safe_component(name: str) -> str:
    name = (name or "").strip() or "unknown"
    name = name.replace(os.sep, "_")
    if os.altsep:
        name = name.replace(os.altsep, "_")
    return name


def make_error_path(error_dir: str, model_name: str, dataset_stem: str) -> str:
    model_component = _safe_component(model_name)
    out_dir = os.path.join(error_dir, model_component)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{dataset_stem}.jsonl")


def make_pred_path(
    pred_dir: str,
    model_name: str,
    dataset_stem: str,
    *,
    shard_rank: int = 0,
    num_shards: int = 1,
) -> str:
    model_component = _safe_component(model_name)
    out_dir = os.path.join(pred_dir, model_component)
    os.makedirs(out_dir, exist_ok=True)
    if num_shards and int(num_shards) > 1:
        return os.path.join(out_dir, f"shard{int(shard_rank)}_of_{int(num_shards)}.jsonl")
    return os.path.join(out_dir, f"{dataset_stem}.jsonl")


def pred_writer(pred_path: str, pred_q, num_workers: int, mode: str = "w"):
    with open(pred_path, mode, encoding="utf-8") as f:
        finished = 0
        while finished < num_workers:
            rec = pred_q.get()
            if rec is None:
                finished += 1
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def error_router_writer(
    error_path: str,
    infra_error_path: str,
    err_q,
    num_workers: int,
    err_mode: str = "w",
    infra_mode: str = "w",
):
    with open(error_path, err_mode, encoding="utf-8") as err_f, open(
        infra_error_path, infra_mode, encoding="utf-8"
    ) as infra_f:
        finished = 0
        while finished < num_workers:
            rec = err_q.get()
            if rec is None:
                finished += 1
                continue
            reason = str(rec.get("reason", ""))
            out_f = infra_f if _is_infra_failure_reason(reason) else err_f
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _safe_cache_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip())
    value = value.strip("._-")
    return value or "unnamed"


def _make_vllm_cache_root(model_name: str, dataset_stem: str) -> str:
    cache_parent = os.environ.get("VLLM_EVAL_CACHE_PARENT") or tempfile.gettempdir()
    os.makedirs(cache_parent, exist_ok=True)
    prefix = (
        f"vllm_eval_{_safe_cache_name(model_name)}_"
        f"{_safe_cache_name(dataset_stem)}_"
    )
    return tempfile.mkdtemp(prefix=prefix, dir=cache_parent)


def _configure_vllm_cache_env(cache_mode: str, worker_cache_root: str | None) -> None:
    os.environ["VLLM_COMPILE_CACHE_SAVE_FORMAT"] = "binary"
    if cache_mode == "disabled":
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
        return

    os.environ.pop("VLLM_DISABLE_COMPILE_CACHE", None)
    if cache_mode == "isolated" and worker_cache_root:
        os.makedirs(worker_cache_root, exist_ok=True)
        os.environ["VLLM_CACHE_ROOT"] = worker_cache_root


def _parse_gpu_ids(raw: str) -> List[int]:
    gpu_ids = [int(x) for x in raw.split(",") if x.strip() != ""]
    if not gpu_ids:
        raise RuntimeError("Empty --gpu-ids")
    return gpu_ids


def _build_vllm_gpu_groups(gpu_ids: List[int], gpus_per_worker: int) -> List[List[int]]:
    if gpus_per_worker <= 0:
        raise RuntimeError(f"--vllm-gpus-per-worker must be >= 1, got {gpus_per_worker}")
    if len(gpu_ids) % gpus_per_worker != 0:
        raise RuntimeError(
            f"GPU count ({len(gpu_ids)}) must be divisible by "
            f"--vllm-gpus-per-worker ({gpus_per_worker})"
        )
    return [
        gpu_ids[i : i + gpus_per_worker]
        for i in range(0, len(gpu_ids), gpus_per_worker)
    ]


def vllm_worker(
    rank,
    gpu_ids,
    shard,
    batch_size,
    task_type,
    prompt_style,
    err_q,
    pred_q,
    out_q,
    model_path,
    trust_remote_code,
    dtype,
    gpu_memory_utilization,
    max_tokens_vllm,
    vllm_cache_mode,
    vllm_cache_root,
    thinking_mode,
):
    import os
    gpu_spec = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_spec
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print(
        f"[vllm-worker] rank={rank} gpu_group={gpu_spec} "
        f"tensor_parallel_size={len(gpu_ids)} shard_size={len(shard)} "
        f"batch_size={batch_size}",
        flush=True,
    )
    worker_cache_root = None
    if vllm_cache_root:
        worker_cache_root = os.path.join(
            vllm_cache_root, f"worker_{rank}_gpu_{gpu_spec.replace(',', '_')}"
        )
    # Avoid cross-process cache corruption when multiple workers compile the same model.
    _configure_vllm_cache_env(vllm_cache_mode, worker_cache_root)

    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    llm = LLM(
        model=model_path,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        tensor_parallel_size=len(gpu_ids),
        max_num_seqs=batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=max_tokens_vllm
    )

    correct = 0
    total = 0
    reqs = []
    metas = []

    def flush():
        nonlocal correct, total, reqs, metas
        if not reqs:
            return
        outs = llm.generate(reqs, sampling_params=sampling_params)
        for out, meta in zip(outs, metas):
            output_text = out.outputs[0].text
            res = finalize_task_result(task_type, meta, output_text, prompt_style)
            pred = {
                **res["pred"],
                "gpu_id": gpu_ids[0],
                "gpu_ids": gpu_ids,
                "gpu_group": gpu_spec,
            }
            if res["counted"]:
                total += 1
                if res["ok"]:
                    correct += 1
            if res["error"] is not None:
                err_q.put(
                    {
                        **res["error"],
                        "gpu_id": gpu_ids[0],
                        "gpu_ids": gpu_ids,
                        "gpu_group": gpu_spec,
                    }
                )
            if pred_q is not None:
                pred_q.put(pred)
        reqs, metas = [], []

    for item in shard:
        q = item["q"]
        image_path = q["image_path"]
        prompt_text = build_prompt_for_task(q, task_type, prompt_style)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        prompt_str = _apply_chat_template(
            processor,
            messages, tokenize=False, add_generation_prompt=True
            , thinking_mode=thinking_mode
        )

        try:
            with Image.open(image_path) as im:
                image = im.convert("RGB")
        except Exception as e:
            base = build_task_base_record("vllm", item["idx"], q)
            rec = {
                **base,
                "pred_index": None,
                "model_output": "",
                "reason": f"image_open_fail:{type(e).__name__}",
                "correct": False,
                "counted": False,
                "gpu_id": gpu_ids[0],
                "gpu_ids": gpu_ids,
                "gpu_group": gpu_spec,
            }
            err_q.put(rec)
            if pred_q is not None:
                pred_q.put(rec)
            continue

        reqs.append({"prompt": prompt_str, "multi_modal_data": {"image": image}})
        metas.append(build_task_base_record("vllm", item["idx"], q))

        if len(reqs) >= batch_size:
            flush()

    flush()
    # Best-effort cleanup to avoid NCCL process group warnings.
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    out_q.put((correct, total))


def run_vllm(args) -> int:
    questions_raw = load_questions(args.questions, task_type=args.task_type)
    questions = [{"idx": i, "q": q} for i, q in enumerate(questions_raw)]

    model_name = args.model_name or os.path.basename(args.model.rstrip("/"))
    dataset_stem = _resolved_dataset_stem(args.questions, getattr(args, "dataset_stem_override", ""))
    error_path = make_error_path(args.error_dir, model_name, dataset_stem)
    pred_path = None
    if getattr(args, "pred_dir", None):
        pred_path = make_pred_path(
            args.pred_dir,
            model_name=model_name,
            dataset_stem=dataset_stem,
            shard_rank=int(getattr(args, "shard_rank", 0) or 0),
            num_shards=int(getattr(args, "num_shards", 1) or 1),
        )

    infra_error_dir = (getattr(args, "infra_error_dir", "") or "").strip()
    if not infra_error_dir:
        infra_error_dir = f"{args.error_dir}_infra"
    infra_error_path = make_error_path(infra_error_dir, model_name, dataset_stem)

    resume_done_idxs: set[int] = set()
    resume_done_n = 0
    resume_total_n = 0
    resume_correct_n = 0
    if args.resume:
        if not pred_path:
            print("[WARN] --resume enabled but --pred-dir not set; resume disabled.")
        elif os.path.isfile(pred_path):
            (
                resume_done_idxs,
                resume_done_n,
                resume_total_n,
                resume_correct_n,
            ) = _load_resume_done_from_pred(pred_path, len(questions))
            if resume_done_n > 0:
                pending_n = len(questions) - resume_done_n
                print(
                    f"[resume] skipped={resume_done_n} counted={resume_total_n} "
                    f"correct={resume_correct_n} pending={pending_n} from {pred_path}"
                )

    pending_questions = [item for item in questions if item["idx"] not in resume_done_idxs]

    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    gpu_groups = _build_vllm_gpu_groups(gpu_ids, args.vllm_gpus_per_worker)
    num_workers = len(gpu_groups)
    vllm_cache_root = None
    if args.vllm_cache_mode == "isolated":
        vllm_cache_root = _make_vllm_cache_root(model_name, dataset_stem)
        print(f"vLLM compile cache mode: {args.vllm_cache_mode}")
        print(f"vLLM compile cache root: {vllm_cache_root}")
    else:
        print(f"vLLM compile cache mode: {args.vllm_cache_mode}")
    print(
        "vLLM GPU groups: "
        f"{gpu_groups} | workers={num_workers} | tensor_parallel_size={args.vllm_gpus_per_worker}"
    )
    print(
        f"vLLM question summary: total={len(questions)} "
        f"resume_skipped={resume_done_n} pending={len(pending_questions)}"
    )
    shards = [pending_questions[i::num_workers] for i in range(num_workers)]
    for rank, (gpu_group, shard) in enumerate(zip(gpu_groups, shards)):
        print(
            f"vLLM shard assignment: rank={rank} gpu_group={gpu_group} shard_size={len(shard)}"
        )

    if not pending_questions:
        correct = resume_correct_n
        total = resume_total_n
        if total:
            acc = correct / total
            print(f"总样本数(计入分母): {total}")
            print(f"Top-1 准确率: {acc:.4f}")
            if _is_index_task(args.task_type):
                for score_path in _iter_score_paths(args.score_path):
                    update_score_tsv(
                        score_path,
                        model_name=model_name,
                        dataset_name=dataset_stem,
                        score=acc,
                    )
                    print(f"得分表已更新: {score_path}")
        else:
            print("没有待处理样本，且没有可计分的已完成样本。")

        _write_metrics(
            args.metrics_path,
            model_name=model_name,
            dataset_name=dataset_stem,
            correct=correct,
            total=total,
        )
        print(f"错误样本已保存到: {error_path}")
        print(f"基础设施失败样本已保存到: {infra_error_path}")
        if pred_path:
            print(f"逐条预测已保存到: {pred_path}")
        return 0

    ctx = mp.get_context("spawn")
    err_q = ctx.SimpleQueue()
    pred_q = ctx.SimpleQueue() if pred_path else None
    out_q = ctx.Queue()

    writer = None
    pred_writer_proc = None
    procs = []
    writers_drained = False
    err_mode = "a" if (args.resume and os.path.isfile(error_path)) else "w"
    infra_mode = "a" if (args.resume and os.path.isfile(infra_error_path)) else "w"
    pred_mode = "a" if (args.resume and pred_path and os.path.isfile(pred_path)) else "w"
    try:
        writer = ctx.Process(
            target=error_router_writer,
            args=(error_path, infra_error_path, err_q, num_workers, err_mode, infra_mode),
        )
        writer.start()
        if pred_path:
            pred_writer_proc = ctx.Process(
                target=pred_writer,
                args=(pred_path, pred_q, num_workers, pred_mode),
            )
            pred_writer_proc.start()

        for rank, gpu_group in enumerate(gpu_groups):
            proc = ctx.Process(
                target=vllm_worker,
                args=(
                    rank,
                    gpu_group,
                    shards[rank],
                    args.batch_size_per_gpu,
                    args.task_type,
                    args.prompt_style,
                    err_q,
                    pred_q,
                    out_q,
                    args.model,
                    args.trust_remote_code,
                    args.dtype,
                    args.gpu_memory_utilization,
                    args.max_tokens_vllm,
                    args.vllm_cache_mode,
                    vllm_cache_root,
                    args.thinking_mode,
                ),
            )

            proc.start()
            procs.append(proc)

        correct = resume_correct_n
        total = resume_total_n
        expected = len(procs)
        received = 0
        reported_failed = set()
        while received < expected:
            _check_writer_processes(writer, pred_writer_proc)
            try:
                c, t = out_q.get(timeout=5)
                correct += c
                total += t
                received += 1
                continue
            except queue.Empty:
                pass

            _check_writer_processes(writer, pred_writer_proc)
            failed = []
            all_exited = True
            for idx, proc in enumerate(procs):
                if proc.exitcode is None:
                    all_exited = False
                    continue
                if proc.exitcode != 0 and idx not in reported_failed:
                    failed.append((idx, proc.pid, proc.exitcode))
                    reported_failed.add(idx)

            if failed:
                detail = ", ".join(
                    [f"worker#{i}(pid={pid}, exitcode={code})" for i, pid, code in failed]
                )
                raise RuntimeError(
                    f"vLLM worker exited before reporting results: {detail}; "
                    f"received {received}/{expected}"
                )

            if all_exited and received < expected:
                exitcodes = [proc.exitcode for proc in procs]
                raise RuntimeError(
                    f"All vLLM workers exited but only got {received}/{expected} results. "
                    f"exitcodes={exitcodes}"
                )

        for proc in procs:
            while proc.is_alive():
                _check_writer_processes(writer, pred_writer_proc)
                proc.join(timeout=1)
        _check_writer_processes(writer, pred_writer_proc)
        for _ in range(num_workers):
            err_q.put(None)
        writer.join()
        if pred_writer_proc is not None:
            for _ in range(num_workers):
                pred_q.put(None)
            pred_writer_proc.join()
        _check_writer_processes(writer, pred_writer_proc)
        writers_drained = True

        if total:
            acc = correct / total
            print(f"总样本数(计入分母): {total}")
            print(f"Top-1 准确率: {acc:.4f}")
            if _is_index_task(args.task_type):
                for score_path in _iter_score_paths(args.score_path):
                    update_score_tsv(
                        score_path,
                        model_name=model_name,
                        dataset_name=dataset_stem,
                        score=acc,
                    )
                    print(f"得分表已更新: {score_path}")
        elif not _is_index_task(args.task_type):
            print("回答生成结束；Open-QA / Grid-Loc 的分数由独立计分程序计算。")
        else:
            print("没有有效样本被评测（请求失败或读取失败）。")

        _write_metrics(
            args.metrics_path,
            model_name=model_name,
            dataset_name=dataset_stem,
            correct=correct,
            total=total,
        )
        print(f"错误样本已保存到: {error_path}")
        if pred_path:
            print(f"逐条预测已保存到: {pred_path}")
        return 0
    finally:
        if not writers_drained:
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()
            for proc in procs:
                proc.join(timeout=5)
            # On failure, a SimpleQueue may be full with no surviving reader.
            # Do not enqueue shutdown markers on that pipe: put() could block forever.
            for proc in (writer, pred_writer_proc):
                if proc is not None:
                    if proc.is_alive():
                        proc.terminate()
                    proc.join(timeout=5)
        if vllm_cache_root:
            shutil.rmtree(vllm_cache_root, ignore_errors=True)


def _iter_score_paths(score_path: str) -> list[str]:
    if not score_path:
        return []
    return [p.strip() for p in score_path.split(",") if p.strip()]


def _write_metrics(metrics_path: str | None, *, model_name: str, dataset_name: str, correct: int, total: int) -> None:
    if not metrics_path:
        return
    payload = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "correct": correct,
        "total": total,
        "acc": (correct / total) if total > 0 else None,
    }
    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _is_infra_failure_reason(reason: str) -> bool:
    return str(reason or "") in _INFRA_FAILURE_REASONS


def _load_resume_done_from_pred(pred_path: str, n_questions: int) -> tuple[set[int], int, int, int]:
    done_idxs: set[int] = set()
    done = 0
    total = 0
    correct = 0
    latest_by_idx: dict[int, dict] = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            idx = rec.get("idx")
            if not isinstance(idx, int):
                continue
            if idx < 0 or idx >= n_questions:
                continue
            latest_by_idx[idx] = rec

    for idx, rec in latest_by_idx.items():
        reason = str(rec.get("reason", ""))
        if _is_infra_failure_reason(reason):
            continue
        done_idxs.add(idx)
        done += 1
        counted = rec.get("counted")
        if counted is None:
            counted = True
        if bool(counted):
            total += 1
            if bool(rec.get("correct", False)):
                correct += 1
    return done_idxs, done, total, correct
