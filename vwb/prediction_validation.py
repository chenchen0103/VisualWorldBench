"""Integrity checks shared by API recovery and standalone scoring."""
import json
from pathlib import Path


def infrastructure_failure(row):
    reason = str(row.get("reason") or "").split(":", 1)[0]
    return reason in {"request_error", "image_open_fail", "image_to_data_url_fail",
                      "worker_error", "worker_exception"}


def read_records(path, *, recover_truncated_tail=False):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    last = max((i for i, line in enumerate(lines) if line.strip()), default=-1)
    records = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if recover_truncated_tail and i == last:
                continue
            raise ValueError(f"{path}:{i + 1}: malformed JSON") from exc
        if not isinstance(row, dict) or not isinstance(row.get("qid"), (str, int)):
            raise ValueError(f"{path}:{i + 1}: missing or invalid qid")
        records.append(row)
    return records


def unique_records(path):
    records = read_records(path)
    by_qid = {}
    for row in records:
        qid = row["qid"]
        if qid in by_qid:
            raise ValueError(f"{path}: duplicate qid {qid}")
        by_qid[qid] = row
    if not by_qid:
        raise ValueError(f"{path}: no records")
    return by_qid


def validate_prediction_file(pred_file, bench_file, *, index_task=False):
    expected = unique_records(bench_file)
    predictions = unique_records(pred_file)
    missing = set(expected) - set(predictions)
    unknown = set(predictions) - set(expected)
    if missing or unknown:
        raise ValueError(f"{pred_file}: incomplete/mismatched predictions: "
                         f"missing={len(missing)}, unknown={len(unknown)}")
    failures = sum(infrastructure_failure(row) or
                   (index_task and row.get("counted") is False)
                   for row in predictions.values())
    if failures:
        raise ValueError(f"{pred_file}: {failures} infrastructure/uncounted failures; "
                         "retry them before scoring")
    return predictions
