#!/usr/bin/env python3
"""VisualWorldBench unified scoring script.

Scores model predictions on all four benchmark tasks:
  - MCQ: accuracy over the supplied MCQ records
  - Grid-Sel: accuracy on 2x2 and 3x3 grid selection
  - Grid-Loc: IoU@0.5 on 2x2 and 3x3 bounding box prediction
  - Open-QA: exact match with alias expansion + optional LLM judge correction

Usage:
    python score.py --bench-dir <bench_data_dir> --pred-dir <predictions_dir> [--output <output.tsv>]

Expected bench-dir structure:
    bench_data/
      mcq.jsonl
      sel_2x2.jsonl
      sel_3x3.jsonl
      loc_2x2.jsonl
      loc_3x3.jsonl
      open_qa.jsonl

Alternative benchmark names such as grid_2x2_loc.jsonl and grid_2x2_box.jsonl are
also accepted for compatibility.

Expected pred-dir structure:
    predictions/
      mcq/<model_name>/mcq.jsonl
      sel_2x2/<model_name>/sel_2x2.jsonl
      sel_3x3/<model_name>/sel_3x3.jsonl
      loc_2x2/<model_name>/loc_2x2.jsonl
      loc_3x3/<model_name>/loc_3x3.jsonl
      open_qa/<model_name>/open_qa.jsonl

Alternative prediction subdirectories such as grid_2x2_loc/ and grid_2x2_box/ are
also accepted for compatibility.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from .prediction_validation import validate_prediction_file

SCRIPT_DIR = Path(__file__).resolve().parent

# IoU threshold for Grid-Loc evaluation
BOX_IOU_THRESHOLD: float = 0.5


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_qids(jsonl_path):
    qids = set()
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                qids.add(json.loads(line)["qid"])
            except (json.JSONDecodeError, KeyError):
                continue
    return qids


def first_existing_path(*candidates):
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


# ─── MCQ Scoring ─────────────────────────────────────────────────
def score_mcq(pred_file, filtered_qids):
    total = correct = 0
    for row in read_jsonl(pred_file):
        qid = row.get("qid")
        if qid not in filtered_qids:
            continue
        if not row.get("counted", True):
            continue
        total += 1
        if row.get("correct", False):
            correct += 1
    return correct / total if total > 0 else None, total


# ─── Grid-Sel Scoring ──────────────────────────────────────────────
def score_sel(pred_file, filtered_sel_qids):
    rows = read_jsonl(pred_file)
    if not rows:
        return None, 0
    rows = [r for r in rows if r.get("qid") in filtered_sel_qids]
    if not rows:
        return None, 0

    has_correct_field = any("correct" in r for r in rows)
    if has_correct_field:
        total = correct = 0
        for row in rows:
            if not row.get("counted", True):
                continue
            total += 1
            if row.get("correct", False):
                correct += 1
        return correct / total if total > 0 else None, total
    else:
        total = correct = 0
        for row in rows:
            total += 1
            answer_idx = row.get("answer_index")
            model_output = str(row.get("model_output", "")).strip()
            m = re.search(r'\b(\d+)\b', model_output)
            if m and answer_idx is not None:
                pred_cell = int(m.group(1))
                if pred_cell == answer_idx + 1:
                    correct += 1
        return correct / total if total > 0 else None, total


# ─── Grid-Loc Scoring ─────────────────────────────────────────────
def score_loc(pred_file, bench_jsonl_path=None):
    """Score bounding box predictions via score_grid_box.py.
    When bench_jsonl_path is provided, filter predictions to only qids in the bench file."""
    pred_path = Path(pred_file)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        output_path = tmp.name

    # If bench file given, create a temp filtered pred file with only bench qids
    filtered_pred_dir = None
    if bench_jsonl_path and Path(bench_jsonl_path).exists():
        bench_qids = load_qids(bench_jsonl_path)
        filtered_pred_dir = tempfile.mkdtemp()
        filtered_pred_path = Path(filtered_pred_dir) / pred_path.name
        with open(pred_path, encoding="utf-8", errors="replace") as fin, \
             open(filtered_pred_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("qid") in bench_qids:
                        fout.write(line + "\n")
                except (json.JSONDecodeError, KeyError):
                    pass
        pred_dir = filtered_pred_dir
    else:
        pred_dir = str(pred_path.parent)

    try:
        cmd = [sys.executable, str(SCRIPT_DIR / "score_grid_box.py"),
               "--pred-dir", pred_dir, "--output", output_path]
        if bench_jsonl_path:
            cmd.extend(["--bench-jsonl", str(bench_jsonl_path)])
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        with open(output_path) as f:
            result = json.load(f)
        acc = result.get(f"acc_iou_{BOX_IOU_THRESHOLD}")
        total = result.get("valid_total", 0)
        return acc, total
    except Exception as e:
        print(f"  [score_loc error] {pred_file}: {e}", file=sys.stderr)
        return None, 0
    finally:
        Path(output_path).unlink(missing_ok=True)
        if filtered_pred_dir:
            import shutil
            shutil.rmtree(filtered_pred_dir, ignore_errors=True)


# ─── Open-QA Scoring ─────────────────────────────────────────────
def score_open_qa(pred_file, bench_jsonl_path=None):
    """Score Open-QA predictions via score_open_qa.py."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        output_path = tmp.name
    try:
        cmd = [sys.executable, str(SCRIPT_DIR / "score_open_qa.py"),
               "--pred-dir", pred_file, "--output", output_path]
        if bench_jsonl_path:
            cmd.extend(["--bench-jsonl", str(bench_jsonl_path)])
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        with open(output_path) as f:
            result = json.load(f)
        acc = result.get("exact_match_rate")
        total = result.get("valid_total", 0)
        return acc, total
    except Exception as e:
        print(f"  [score_open error] {pred_file}: {e}", file=sys.stderr)
        return None, 0
    finally:
        Path(output_path).unlink(missing_ok=True)


def score_open_qa_corrected(pred_file, open_qa_qids=None):
    """Open-QA corrected EM: auto exact_match + LLM judge rescued.
    Reads .judge.jsonl from alongside the pred file."""
    pred_path = Path(pred_file)
    judge_path = pred_path.parent / pred_path.name.replace(".jsonl", ".judge.jsonl")
    if not judge_path.exists():
        return None, 0

    # Load pred rows
    rows = {}
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            qid = obj.get("qid")
            if open_qa_qids is not None and qid not in open_qa_qids:
                continue
            rows[qid] = obj
        except json.JSONDecodeError:
            continue

    # Load judge results
    judge_results = {}
    for line in judge_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            judge_results[obj.get("qid")] = obj
        except json.JSONDecodeError:
            continue

    total = len(rows)
    auto_correct = 0
    judge_rescued = 0
    request_errors = 0
    pending_judgments = 0

    for qid, row in rows.items():
        if str(row.get("reason") or "") == "request_error":
            request_errors += 1
            continue
        pred = _normalize_text(_extract_final_name(str(row.get("model_output") or "")))
        aliases = row.get("gt_aliases_expanded") or row.get("gt_aliases_base") or []
        aliases = [str(a) for a in aliases if _is_english(str(a))]
        if pred and any(_normalize_text(a) == pred for a in aliases):
            auto_correct += 1
        elif qid in judge_results and judge_results[qid].get("judgment") == "correct":
            judge_rescued += 1
        elif judge_results.get(qid, {}).get("judgment") != "wrong":
            pending_judgments += 1

    valid_total = total - request_errors
    if valid_total <= 0:
        return None, 0
    if pending_judgments:
        print(f"  [incomplete judge] {pred_file}: {pending_judgments} non-exact answers lack completed judgments; corrected score withheld", file=sys.stderr)
        return None, valid_total
    corrected_em = (auto_correct + judge_rescued) / valid_total
    return corrected_em, valid_total


def _normalize_text(text):
    from .score_open_qa import normalize_text
    return normalize_text(text)


def _extract_final_name(text):
    from .score_open_qa import extract_final_name
    return extract_final_name(text)


def _is_english(text):
    from .score_open_qa import is_english
    return is_english(text)


# ─── Average Computation ─────────────────────────────────────────
def compute_avg(row):
    """Avg = mean(MCQ, Sel_Avg, Loc_Avg, Open_QA_corrected).
    Returns None if any component is missing.
    Note: only open_qa_corrected is used (no fallback to open_qa)."""
    open_metric = row.get("open_qa_corrected")
    if open_metric is None:
        return None
    mcq = row.get("mcq")
    sel2 = row.get("sel_2x2")
    sel3 = row.get("sel_3x3")
    loc2 = row.get("loc_2x2")
    loc3 = row.get("loc_3x3")
    if any(v is None for v in (mcq, sel2, sel3, loc2, loc3)):
        return None
    sel_avg = (sel2 + sel3) / 2
    loc_avg = (loc2 + loc3) / 2
    return (mcq + sel_avg + loc_avg + open_metric) / 4


# ─── Main ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Score all models on VisualWorldBench.")
    parser.add_argument("--bench-dir", required=True,
                        help="Directory containing benchmark JSONL files.")
    parser.add_argument("--pred-dir", required=True,
                        help="Directory containing model predictions organized by task/model.")
    parser.add_argument("--output", "-o", default="bench_scores.tsv",
                        help="Output TSV path (default: bench_scores.tsv)")
    parser.add_argument("--tasks", default="auto",
                        help="auto: score complete existing task files; all or a comma-separated "
                             "task list: also require those task files for every model")
    args = parser.parse_args()

    bench_dir = Path(args.bench_dir)
    pred_root = Path(args.pred_dir)

    # Load bench qid sets
    bench_files = {
        "mcq": first_existing_path(bench_dir / "mcq.jsonl", bench_dir / "mcq_filtered.jsonl"),
        "sel_2x2": first_existing_path(bench_dir / "sel_2x2.jsonl", bench_dir / "grid_2x2_loc.jsonl"),
        "sel_3x3": first_existing_path(bench_dir / "sel_3x3.jsonl", bench_dir / "grid_3x3_loc.jsonl"),
        "loc_2x2": first_existing_path(bench_dir / "loc_2x2.jsonl", bench_dir / "grid_2x2_box.jsonl"),
        "loc_3x3": first_existing_path(bench_dir / "loc_3x3.jsonl", bench_dir / "grid_3x3_box.jsonl"),
        "open_qa": first_existing_path(bench_dir / "open_qa.jsonl", bench_dir / "open_qa_filtered.jsonl"),
    }

    for name, path in bench_files.items():
        if not path.exists():
            print(f"WARNING: bench file not found: {path}", file=sys.stderr)

    mcq_qids = load_qids(bench_files["mcq"]) if bench_files["mcq"].exists() else set()
    sel_2x2_qids = load_qids(bench_files["sel_2x2"]) if bench_files["sel_2x2"].exists() else set()
    sel_3x3_qids = load_qids(bench_files["sel_3x3"]) if bench_files["sel_3x3"].exists() else set()
    open_qa_qids = load_qids(bench_files["open_qa"]) if bench_files["open_qa"].exists() else set()

    # Discover models
    task_subdirs = {
        "mcq": ["mcq", "mcq_filtered"],
        "sel_2x2": ["sel_2x2", "grid_2x2_loc"],
        "sel_3x3": ["sel_3x3", "grid_3x3_loc"],
        "loc_2x2": ["loc_2x2", "grid_2x2_box"],
        "loc_3x3": ["loc_3x3", "grid_3x3_box"],
        "open_qa": ["open_qa", "open_qa_filtered"],
    }
    selected = set(task_subdirs) if args.tasks in {"auto", "all"} else set(args.tasks.split(","))
    if not selected or not selected <= set(task_subdirs):
        parser.error("--tasks must be auto, all, or valid comma-separated task names")

    def resolve_task_dir(task_key):
        candidates = [pred_root / subdir for subdir in task_subdirs[task_key]]
        return first_existing_path(*candidates)

    def resolve_prediction_file(task_key, model_name):
        task_dir = resolve_task_dir(task_key)
        if not task_dir.exists():
            return task_dir / model_name / f"{task_subdirs[task_key][0]}.jsonl"
        for subdir in task_subdirs[task_key]:
            pred = pred_root / subdir / model_name / f"{subdir}.jsonl"
            if pred.exists():
                return pred
        primary = task_subdirs[task_key][0]
        return pred_root / primary / model_name / f"{primary}.jsonl"

    models = set()
    for task_key in selected:
        task_dir = resolve_task_dir(task_key)
        if task_dir.is_dir():
            for d in task_dir.iterdir():
                if d.is_dir():
                    models.add(d.name)

    print(f"Found {len(models)} models", file=sys.stderr)
    if not models:
        raise ValueError("No model predictions found for selected tasks")

    # Score each model
    results = []
    for mname in sorted(models):
        row = {"model": mname}
        active = {task for task in selected if resolve_prediction_file(task, mname).exists()}
        if args.tasks != "auto" and active != selected:
            raise ValueError(f"{mname}: missing task files: {sorted(selected - active)}")
        if not active:
            raise ValueError(f"{mname}: no prediction files")
        expected_counts = {}
        for task in sorted(active):
            predictions = validate_prediction_file(
                resolve_prediction_file(task, mname), bench_files[task],
                index_task=task == "mcq" or task.startswith("sel_"))
            expected_counts[task] = len(predictions)

        # MCQ
        pred = resolve_prediction_file("mcq", mname)
        if "mcq" in active:
            acc, n = score_mcq(str(pred), mcq_qids)
            row["mcq"] = acc
            row["mcq_n"] = n
            print(f"  {mname}: MCQ={acc:.4f} (n={n})" if acc is not None else f"  {mname}: MCQ=N/A", file=sys.stderr)

        # Grid-Sel 2x2 / 3x3
        for grid, qids in [("2x2", sel_2x2_qids), ("3x3", sel_3x3_qids)]:
            pred = resolve_prediction_file(f"sel_{grid}", mname)
            if f"sel_{grid}" in active:
                acc, n = score_sel(str(pred), qids)
                row[f"sel_{grid}"] = acc
                row[f"sel_{grid}_n"] = n

        # Grid-Loc 2x2 / 3x3
        for grid in ("2x2", "3x3"):
            pred = resolve_prediction_file(f"loc_{grid}", mname)
            if f"loc_{grid}" in active:
                acc, n = score_loc(str(pred), str(bench_files[f"loc_{grid}"]))
                row[f"loc_{grid}"] = acc
                row[f"loc_{grid}_n"] = n

        # Open-QA
        pred = resolve_prediction_file("open_qa", mname)
        if "open_qa" in active:
            acc, n = score_open_qa(str(pred), str(bench_files["open_qa"]))
            row["open_qa"] = acc
            row["open_qa_n"] = n
            # Try corrected scoring
            corrected, cn = score_open_qa_corrected(str(pred), open_qa_qids)
            row["open_qa_corrected_n"] = cn
            if corrected is not None:
                row["open_qa_corrected"] = corrected

        for task, count in expected_counts.items():
            if row.get(task + "_n") != count or row.get(task) is None:
                raise ValueError(f"{mname}/{task}: scorer did not account for all {count} samples")
        row["avg"] = compute_avg(row)
        results.append(row)

    # Sort by avg descending
    results.sort(key=lambda r: r.get("avg") if r.get("avg") is not None else -1, reverse=True)

    # Write TSV
    cols = ["model", "mcq", "sel_2x2", "sel_3x3", "loc_2x2", "loc_3x3", "open_qa", "open_qa_corrected", "avg"]
    cols += [f"{task}_n" for task in ("mcq", "sel_2x2", "sel_3x3", "loc_2x2", "loc_3x3", "open_qa", "open_qa_corrected")]
    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        f.write("\t".join(cols) + "\n")
        for row in results:
            vals = []
            for c in cols:
                v = row.get(c)
                if v is None:
                    vals.append("")
                elif isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            f.write("\t".join(vals) + "\n")

    print(f"\nResults saved to {outpath} ({len(results)} models)", file=sys.stderr)

    # Print summary
    print(f"\n{'Model':<60} {'MCQ':>6} {'Sel2':>6} {'Sel3':>6} {'Loc2':>6} {'Loc3':>6} {'Open':>6} {'OpnC':>6} {'Avg':>6}")
    print("-" * 120)
    for row in results:
        def fmt(k):
            v = row.get(k)
            return f"{v:.4f}" if v is not None else "  --"
        print(f"{row['model']:<60} {fmt('mcq'):>6} {fmt('sel_2x2'):>6} {fmt('sel_3x3'):>6} {fmt('loc_2x2'):>6} {fmt('loc_3x3'):>6} {fmt('open_qa'):>6} {fmt('open_qa_corrected'):>6} {fmt('avg'):>6}")


if __name__ == "__main__":
    main()
