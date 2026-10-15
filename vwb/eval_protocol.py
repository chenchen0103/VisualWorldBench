"""Prompt construction, record normalization, and answer parsing.

The internal task key grid_loc denotes cell selection; grid_box denotes
bounding-box localization.
"""
import re
from typing import Any, Dict, List, Optional

PROTOCOL_VERSION = "2026-10-13"

def _copy_optional_fields(src, dst, fields):
    for field in fields:
        if field in src:
            dst[field] = src[field]

def _is_index_task(task_type: str) -> bool:
    return task_type in {"mcq", "grid_loc"}

def normalize_question(
    q: Dict[str, Any],
    *,
    task_type: str = "mcq",
    is_grid_bench: bool = False,
) -> Dict[str, Any]:
    """Normalize supported task records for inference and scoring.

    Accepts string or structured options, grid-cell choices, and box targets.
    Preserves optional identifiers and metadata when available.
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

    # Preserve identifiers and metadata for prediction and error records.
    for k in ("qid", "gt_id", "gt_name", "meta", "class_id", "class_name", "grid_id", "bbox_abs"):
        if k in q:
            nq[k] = q[k]
    return nq

def build_prompt(options: List[str], prompt_style: str = "plain", prompt_override: Optional[str] = None, meta: Optional[Dict] = None) -> str:
    if prompt_override and prompt_override.strip():
        return prompt_override.strip()

    n = len(options)
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

def parse_index(text: str, n_options: int, prompt_style: str = "plain") -> Optional[int]:
    stripped = (text or "").strip()

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
