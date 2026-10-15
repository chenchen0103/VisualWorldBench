"""Score Grid-Box (Grid-Loc bounding box) predictions.

Evaluates model predictions against ground-truth bounding boxes using IoU metric.
Coordinates are normalized to [0, 1000] where (0,0) is top-left and (1000,1000) is bottom-right.

Usage:
    python score_grid_box.py --pred-dir <pred_jsonl> --output <output_json> [--bench-jsonl <bench_file>]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


STRICT_BBOX_PATTERN = re.compile(
    r"\[\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*,?\s*"
    r"([-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*,?\s*"
    r"([-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*,?\s*"
    r"([-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*\]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Grid-Box predictions with normalized [0,1000] bbox answers.")
    parser.add_argument("--pred-dir", required=True, help="Prediction jsonl file or directory containing jsonl shards.")
    parser.add_argument("--output", required=True, help="Output JSON path for results.")
    parser.add_argument("--bench-jsonl", action="append", default=[], help="Optional bench jsonl for meta fallback.")
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


def row_key(row: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    idx = row.get("idx")
    if isinstance(idx, int):
        return ("idx", str(idx))
    qid = row.get("qid")
    if qid is not None:
        return ("qid", str(qid))
    return None


def latest_rows(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    latest: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in iter_jsonl_rows(paths):
        key = row_key(row)
        if key is None:
            continue
        latest[key] = row
    return list(latest.values())


def load_bench_meta(paths: List[str]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
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
                    rows[str(qid)] = row
    return rows


def _clip_1000(v: float) -> float:
    return max(0.0, min(1000.0, float(v)))


def sanitize_norm_bbox(values: List[float]) -> List[float]:
    x1, y1, x2, y2 = [_clip_1000(v) for v in values]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def parse_bbox_with_stats(text: str) -> Tuple[Optional[List[float]], bool]:
    matches = STRICT_BBOX_PATTERN.findall(text or "")
    if len(matches) > 1:
        return sanitize_norm_bbox([float(v) for v in matches[0]]), True
    if len(matches) == 1:
        return sanitize_norm_bbox([float(v) for v in matches[0]]), False

    nums = re.findall(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", text or "")
    if len(nums) < 4:
        return None, False
    try:
        values = [float(v) for v in nums[:4]]
    except ValueError:
        return None, False
    return sanitize_norm_bbox(values), False


def get_gt_bbox(row: Dict[str, Any], bench_row: Optional[Dict[str, Any]]) -> Optional[List[float]]:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    for candidate in (row.get("bbox_norm"), row.get("answer"), meta.get("bbox_norm")):
        if isinstance(candidate, list) and len(candidate) == 4:
            return sanitize_norm_bbox([float(v) for v in candidate])
    if bench_row is not None:
        bench_meta = bench_row.get("meta") if isinstance(bench_row.get("meta"), dict) else {}
        for candidate in (bench_row.get("bbox_norm"), bench_row.get("answer"), bench_meta.get("bbox_norm")):
            if isinstance(candidate, list) and len(candidate) == 4:
                return sanitize_norm_bbox([float(v) for v in candidate])
    return None


def get_image_size(row: Dict[str, Any], bench_row: Optional[Dict[str, Any]]) -> Optional[List[int]]:
    for container in (row.get("meta"), (bench_row or {}).get("meta")):
        if not isinstance(container, dict):
            continue
        image_size = container.get("image_size")
        if isinstance(image_size, list) and len(image_size) == 2:
            return [int(image_size[0]), int(image_size[1])]
        rows = container.get("rows")
        cols = container.get("cols")
        if rows and cols:
            return [int(cols) * 384, int(rows) * 384]
    return None


def norm_to_abs(bbox_norm: List[float], image_size: List[int]) -> List[float]:
    w, h = image_size
    x1, y1, x2, y2 = bbox_norm
    return [
        x1 / 1000.0 * w,
        y1 / 1000.0 * h,
        x2 / 1000.0 * w,
        y2 / 1000.0 * h,
    ]


def compute_iou(pred_abs: List[float], gt_abs: List[float]) -> float:
    px1, py1, px2, py2 = pred_abs
    gx1, gy1, gx2, gy2 = gt_abs
    inter_w = max(0.0, min(px2, gx2) - max(px1, gx1))
    inter_h = max(0.0, min(py2, gy2) - max(py1, gy1))
    inter = inter_w * inter_h
    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    gt_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = pred_area + gt_area - inter
    if union <= 0:
        return 0.0
    return inter / union


class MetricBucket:
    def __init__(self) -> None:
        self.total = 0
        self.request_error = 0
        self.parse_fail = 0
        self.multi_bbox_count = 0
        self.valid_total = 0
        self.acc_iou_07 = 0
        self.acc_iou_05 = 0
        self.acc_iou_025 = 0
        self.iou_sum = 0.0

    def add(self, iou: float, *, parse_fail: bool, request_error: bool, is_multi_bbox: bool) -> None:
        self.total += 1
        if request_error:
            self.request_error += 1
            return
        self.valid_total += 1
        if parse_fail:
            self.parse_fail += 1
        if is_multi_bbox:
            self.multi_bbox_count += 1
        if iou >= 0.7:
            self.acc_iou_07 += 1
        if iou >= 0.5:
            self.acc_iou_05 += 1
        if iou >= 0.25:
            self.acc_iou_025 += 1
        self.iou_sum += iou

    def as_dict(self) -> Dict[str, Any]:
        total = self.total if self.total > 0 else 1
        valid_total = self.valid_total if self.valid_total > 0 else 1
        return {
            "total": self.total,
            "request_error": self.request_error,
            "request_error_rate": self.request_error / total if self.total > 0 else 0.0,
            "valid_total": self.valid_total,
            "parse_fail": self.parse_fail,
            "parse_fail_rate": self.parse_fail / valid_total if self.valid_total > 0 else 0.0,
            "multi_bbox_count": self.multi_bbox_count,
            "multi_bbox_rate": self.multi_bbox_count / valid_total if self.valid_total > 0 else 0.0,
            "acc_iou_0.7": self.acc_iou_07 / valid_total if self.valid_total > 0 else 0.0,
            "acc_iou_0.5": self.acc_iou_05 / valid_total if self.valid_total > 0 else 0.0,
            "acc_iou_0.25": self.acc_iou_025 / valid_total if self.valid_total > 0 else 0.0,
            "mean_iou": self.iou_sum / valid_total if self.valid_total > 0 else 0.0,
        }


def main() -> int:
    args = parse_args()
    pred_paths = iter_jsonl_paths(args.pred_dir)
    bench_meta = load_bench_meta(args.bench_jsonl)

    rows = latest_rows(pred_paths)

    overall = MetricBucket()
    by_grid_size: Dict[str, MetricBucket] = {}
    by_domain: Dict[str, MetricBucket] = {}

    for row in rows:
        qid = str(row.get("qid") or "")
        bench_row = bench_meta.get(qid)
        gt_bbox_norm = get_gt_bbox(row, bench_row)
        image_size = get_image_size(row, bench_row)
        if gt_bbox_norm is None or image_size is None:
            raise ValueError(f"Missing gt bbox or image_size for qid={qid}")

        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        grid_size = str(row.get("grid_size") or meta.get("grid_size") or "unknown")

        request_error = str(row.get("reason") or "") == "request_error"
        pred_bbox_norm, is_multi_bbox = parse_bbox_with_stats(str(row.get("model_output") or ""))
        parse_fail = (pred_bbox_norm is None) and not request_error
        iou = 0.0
        if pred_bbox_norm is not None:
            iou = compute_iou(norm_to_abs(pred_bbox_norm, image_size), norm_to_abs(gt_bbox_norm, image_size))

        overall.add(iou, parse_fail=parse_fail, request_error=request_error, is_multi_bbox=is_multi_bbox)
        domain = str(meta.get("domain") or "unknown")
        by_grid_size.setdefault(grid_size, MetricBucket()).add(iou, parse_fail=parse_fail, request_error=request_error, is_multi_bbox=is_multi_bbox)
        by_domain.setdefault(domain, MetricBucket()).add(iou, parse_fail=parse_fail, request_error=request_error, is_multi_bbox=is_multi_bbox)

    result = {
        **overall.as_dict(),
        "by_grid_size": {k: v.as_dict() for k, v in sorted(by_grid_size.items())},
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
