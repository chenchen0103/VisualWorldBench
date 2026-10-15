import csv
import fcntl
import os
from contextlib import contextmanager
from typing import Dict, List, Tuple


@contextmanager
def _score_lock(path: str):
    lock_path = f"{path}.lock"
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def _read_score_tsv(path: str) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    if not os.path.exists(path):
        return [], {}

    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))

    if not rows:
        return [], {}

    header = rows[0]
    # Filter out empty column names and "Average" column (we'll regenerate it)
    datasets = [ds.strip() for ds in header[1:] if ds.strip() and ds.strip() != "Average"]

    table: Dict[str, Dict[str, str]] = {}
    for row in rows[1:]:
        if not row:
            continue
        model = (row[0] if len(row) > 0 else "").strip()
        if not model:
            continue
        # Map non-empty datasets to their values (skip Average column)
        table[model] = {}
        col_idx = 1
        for ds in header[1:]:
            if col_idx < len(row):
                ds_stripped = ds.strip()
                if ds_stripped and ds_stripped != "Average":  # Skip Average column
                    table[model][ds_stripped] = row[col_idx]
            col_idx += 1
    return datasets, table


def _sort_datasets(datasets: List[str]) -> List[str]:
    """Sort datasets by predefined order."""
    # Define benchmark order priority
    order_map = {
        # MCQ
        "mcq_all": 1,
        "mcq_bamboo_all.jsonl": 1,
        "mcq_bamboo_hard.jsonl": 2,
        # Grid-Loc
        "grid_2x2_loc": 3,
        "grid_2x2_loc.jsonl": 3,
        "grid_3x3_loc": 4,
        "grid_3x3_loc.jsonl": 4,
        # Grid-Box
        "grid_2x2_box@iou0.5": 5,
        "grid_3x3_box@iou0.5": 6,
        # Open
        "open_v1_all@exact_match": 7,
        # Grid-Loc-Hard
        "bench_hard_2x2_loc.jsonl": 8,
        "bench_hard_3x3_loc.jsonl": 9,
    }

    def sort_key(ds: str) -> Tuple[int, str]:
        return (order_map.get(ds, 999), ds)

    return sorted(datasets, key=sort_key)


def _compute_average(model_row: Dict[str, str], datasets: List[str]) -> float:
    """Compute average score for a model across all datasets."""
    scores = []
    for ds in datasets:
        value = model_row.get(ds, "").strip()
        if value:
            try:
                scores.append(float(value))
            except ValueError:
                pass
    return sum(scores) / len(scores) if scores else 0.0


def _write_score_tsv(path: str, datasets: List[str], table: Dict[str, Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Sort datasets by predefined order
    sorted_datasets = _sort_datasets(datasets)

    # Compute average for each model
    model_averages = {
        model: _compute_average(table[model], sorted_datasets)
        for model in table
    }

    # Sort models by average score (descending)
    sorted_models = sorted(table.keys(), key=lambda m: (-model_averages[m], m))

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        # Write header with Average column
        w.writerow(["model", *sorted_datasets, "Average"])
        # Write model rows
        for model in sorted_models:
            avg = model_averages[model]
            avg_str = f"{avg:.4f}" if avg > 0 else ""
            w.writerow([model, *[table[model].get(ds, "") for ds in sorted_datasets], avg_str])


def update_score_cells_tsv(
    score_path: str,
    *,
    model_name: str,
    updates: Dict[str, str],
) -> None:
    with _score_lock(score_path):
        datasets, table = _read_score_tsv(score_path)

        for dataset_name in updates:
            if dataset_name not in datasets:
                datasets.append(dataset_name)

        model_row = table.setdefault(model_name, {})
        for dataset_name, value in updates.items():
            model_row[dataset_name] = value

        _write_score_tsv(score_path, datasets, table)


def update_score_tsv(
    score_path: str,
    *,
    model_name: str,
    dataset_name: str,
    score: float,
    score_fmt: str = "{:.4f}",
) -> None:
    update_score_cells_tsv(
        score_path,
        model_name=model_name,
        updates={dataset_name: score_fmt.format(score)},
    )
