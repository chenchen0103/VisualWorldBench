"""Score Open-QA predictions via exact match with alias expansion.

Evaluates model free-form text predictions against canonical labels and alias sets.
Supports hierarchical ancestor scoring and domain/depth-stratified analysis.

Usage:
    python score_open_qa.py --pred-dir <pred_jsonl> --output <output_json> [--bench-jsonl <bench_file>]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score open-ended predictions.")
    parser.add_argument("--pred-dir", required=True, help="Prediction jsonl file or directory.")
    parser.add_argument("--output", required=True, help="Output JSON path for results.")
    parser.add_argument("--alias-mode", choices=["base", "expanded"], default="expanded")
    parser.add_argument("--bench-jsonl", action="append", default=[], help="Optional benchmark jsonl for GT metadata.")
    return parser.parse_args()


def iter_jsonl_paths(path_str: str) -> List[Path]:
    path = Path(path_str)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(p for p in path.glob("*.jsonl") if p.is_file())


def iter_jsonl_rows(paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def latest_rows(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    latest: Dict[tuple, Dict[str, Any]] = {}
    for row in iter_jsonl_rows(paths):
        qid = row.get("qid")
        idx = row.get("idx")
        if qid is not None:
            latest[("qid", str(qid))] = row
        elif isinstance(idx, int):
            latest[("idx", str(idx))] = row
    return list(latest.values())


def load_bench_rows(paths: List[str]) -> Dict[str, Dict[str, Any]]:
    bench_rows: Dict[str, Dict[str, Any]] = {}
    for path_str in paths:
        path = Path(path_str)
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = row.get("qid")
                if qid is not None:
                    bench_rows[str(qid)] = row
    return bench_rows


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip().replace("_", " ")
    # Remove punctuation (keep only alphanumeric and spaces)
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_final_name(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    def _strip_answer_prefix(candidate: str) -> str:
        candidate = (candidate or "").strip()
        if not candidate:
            return ""
        m = re.match(r"(?is)^\s*(?:answer|final\s+answer|答案|最终答案)\s*[:：]\s*(.+?)\s*$", candidate)
        if m:
            return m.group(1).strip()
        # Fallback: search for an embedded "Answer:" line within multi-line content
        # (e.g. <answer>description...\nAnswer: species_name</answer>)
        m2 = re.search(r"(?im)^(?:answer|final\s+answer|答案|最终答案)\s*[:：]\s*(.+?)\s*$", candidate)
        if m2:
            return m2.group(1).strip()
        return candidate

    tagged_patterns = [
        r"(?is)<answer>\s*(.*?)\s*</answer>",
        r"(?is)<final_answer>\s*(.*?)\s*</final_answer>",
        r"(?im)^\s*(?:answer|final\s+answer|答案|最终答案)\s*[:：]\s*(.+?)\s*$",
    ]
    for pat in tagged_patterns:
        matches = re.findall(pat, text)
        if matches:
            candidate = _strip_answer_prefix(str(matches[-1]))
            if candidate:
                return candidate
    return _strip_answer_prefix(text)


def is_english(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return False
    return any("a" <= ch.lower() <= "z" for ch in text)


def choose_aliases(row: Dict[str, Any], alias_mode: str) -> List[str]:
    aliases = row.get("gt_aliases_base") or []
    if alias_mode == "expanded" and row.get("gt_aliases_expanded"):
        aliases = row["gt_aliases_expanded"]
    return [str(a) for a in aliases if is_english(str(a))]


def score_sample(row: Dict[str, Any], alias_mode: str) -> float:
    pred = normalize_text(extract_final_name(str(row.get("model_output") or "")))
    if not pred:
        return 0.0
    aliases = choose_aliases(row, alias_mode)
    if any(normalize_text(alias) == pred for alias in aliases):
        return 1.0
    return 0.0


class ScoreBucket:
    def __init__(self) -> None:
        self.total = 0
        self.request_error = 0
        self.valid_total = 0
        self.exact = 0

    def add(self, score: float, *, request_error: bool = False) -> None:
        self.total += 1
        if request_error:
            self.request_error += 1
            return
        self.valid_total += 1
        if score >= 1.0:
            self.exact += 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "request_error": self.request_error,
            "valid_total": self.valid_total,
            "exact_match_rate": self.exact / self.valid_total if self.valid_total > 0 else 0.0,
        }


def main() -> int:
    args = parse_args()
    pred_paths = iter_jsonl_paths(args.pred_dir)
    rows = latest_rows(pred_paths)
    bench_rows = load_bench_rows(args.bench_jsonl)

    if bench_rows:
        merged = []
        for row in rows:
            qid = str(row.get("qid"))
            if qid in bench_rows:
                r = dict(row)
                for key in ("gt_aliases_base", "gt_aliases_expanded", "gt_ancestors"):
                    if key in bench_rows[qid]:
                        r[key] = bench_rows[qid][key]
                merged.append(r)
        rows = merged if merged else rows

    overall = ScoreBucket()
    by_domain: Dict[str, ScoreBucket] = {}

    for row in rows:
        request_error = str(row.get("reason") or "") == "request_error"
        score = score_sample(row, args.alias_mode)
        overall.add(score, request_error=request_error)
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        domain = str(meta.get("domain") or "unknown")
        by_domain.setdefault(domain, ScoreBucket()).add(score, request_error=request_error)

    result = {
        **overall.as_dict(),
        "by_domain": {k: v.as_dict() for k, v in sorted(by_domain.items())},
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
