#!/usr/bin/env python3
"""Generate Open-QA judgments with the configured judge prompt."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

from .infer import api_call, extract_text, load_questions
from .judge_protocol import build_judge_message, parse_judge_response
from .score_open_qa import choose_aliases, extract_final_name, normalize_text
from .prediction_validation import validate_prediction_file


def build_request(args, row):
    messages = build_judge_message(row)
    if args.api_format == "gemini-responses":
        # This endpoint expects Gemini contents/parts, not OpenAI Responses input.
        return args.base_url.rstrip("/") + "/responses", {
            "model": args.judge_model,
            "contents": [{"role": "user", "parts": [{
                "text": "\n\n".join(m["content"] for m in messages)}]}],
            "generation_config": {"temperature": args.temperature,
                                  "maxOutputTokens": args.max_tokens},
            "stream": False,
        }
    return args.base_url.rstrip("/") + "/chat/completions", {
        "model": args.judge_model, "messages": messages,
        "temperature": args.temperature, "max_tokens": args.max_tokens,
    }


def judge_response_text(response, api_format):
    if api_format == "chat-completions":
        return extract_text(response)
    candidates = response.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini response has no candidates")
    candidate = candidates[0]
    if candidate.get("finishReason") != "STOP":
        raise ValueError("Gemini response did not finish normally: " + str(candidate.get("finishReason")))
    return "\n".join(p["text"] for p in candidate.get("content", {}).get("parts", [])
                     if isinstance(p.get("text"), str) and not p.get("thought"))


def exact_match(row):
    prediction = normalize_text(extract_final_name(row.get("model_output", "")))
    return bool(prediction) and any(
        normalize_text(alias) == prediction for alias in choose_aliases(row, "expanded")
    )


def load_latest(path):
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["qid"]] = row
    return rows


async def judge_file(args):
    import aiohttp

    bench = {q["qid"]: q for q in load_questions(args.bench_jsonl, "open")}
    predictions = validate_prediction_file(args.pred_file, args.bench_jsonl)
    if not predictions:
        raise ValueError("Prediction file is empty")
    unknown = set(predictions) - set(bench)
    if unknown:
        raise ValueError(f"{len(unknown)} prediction qids are not in this benchmark")
    output = Path(args.output) if args.output else Path(args.pred_file).with_suffix(".judge.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "model": args.judge_model, "base_url": args.base_url,
        "api_format": args.api_format,
        "temperature": args.temperature, "max_tokens": args.max_tokens,
        "pred_sha256": hashlib.sha256(Path(args.pred_file).read_bytes()).hexdigest(),
        "bench_sha256": hashlib.sha256(Path(args.bench_jsonl).read_bytes()).hexdigest(),
        "judge_protocol_sha256": hashlib.sha256(Path(__file__).with_name("judge_protocol.py").read_bytes()).hexdigest(),
    }
    config_path = output.with_suffix(".run.json")
    if not args.overwrite and output.exists() and output.stat().st_size:
        previous = json.loads(config_path.read_text()) if config_path.exists() else {}
        if previous:
            previous.setdefault("api_format", "chat-completions")
        if previous != config:
            raise ValueError("Judge configuration or predictions changed. Choose a new --output or use --overwrite.")
    existing = {} if args.overwrite else load_latest(output)
    pending = []
    for qid, prediction in predictions.items():
        row = {**prediction, **bench[qid]}
        if row.get("reason") == "request_error" or exact_match(row):
            continue
        if existing.get(qid, {}).get("judgment") in {"correct", "wrong"}:
            continue
        pending.append(row)
    print(f"Predictions: {len(predictions)}/{len(bench)}; pending judge calls: {len(pending)}")
    if args.dry_run:
        return 0
    api_key = os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if pending and not api_key:
        raise ValueError("Set JUDGE_API_KEY or OPENAI_API_KEY")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    semaphore = asyncio.Semaphore(args.num_workers)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        async def evaluate(row):
            base = {"qid": row["qid"], "gt_name": row["gt_name"],
                    "pred_text": extract_final_name(row.get("model_output", ""))}
            try:
                endpoint, payload = build_request(args, row)
                usage_attempts = []
                for attempt in range(args.max_retries + 1):
                    text = ""
                    try:
                        async with semaphore:
                            response = await api_call(session, endpoint, headers, payload,
                                                      timeout_sec=args.timeout, max_retries=0)
                        usage = response.get("usageMetadata", response.get("usage"))
                        if usage is not None:
                            usage_attempts.append(usage)
                        text = judge_response_text(response, args.api_format)
                        parsed = parse_judge_response(text)
                    except Exception as exc:
                        parsed = {"judgment": "error", "reason": str(exc)[:200]}
                    if parsed["judgment"] != "error" or attempt == args.max_retries:
                        return {**base, **parsed, "raw_response": text,
                                "usage_attempts": usage_attempts}
                    await asyncio.sleep(min(2 ** attempt, 8))
            except Exception as exc:
                return {**base, "judgment": "error", "reason": str(exc)[:200]}

        # Append completed judgments immediately so interrupted runs can resume.
        with output.open("w" if args.overwrite else "a") as fout:
            for task in asyncio.as_completed([evaluate(row) for row in pending]):
                result = await task
                existing[result["qid"]] = result
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
    failures = sum(row.get("judgment") == "error" for row in existing.values())
    print(f"Judge file: {output}; API/format failures: {failures} (rerun to retry)")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-file", required=True)
    parser.add_argument("--bench-jsonl", required=True)
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL"),
                        help="Judge model ID; required for API calls, or set JUDGE_MODEL")
    parser.add_argument("--base-url", default=os.getenv("JUDGE_BASE_URL"),
                        help="Your provider's base URL, including /v1; or set JUDGE_BASE_URL")
    parser.add_argument("--api-format", choices=["gemini-responses", "chat-completions"],
                        default=os.getenv("JUDGE_API_FORMAT", "gemini-responses"))
    parser.add_argument("--output")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0, help="Judge sampling temperature (default: 1.0)")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Count pending judgments without API calls")
    args = parser.parse_args()
    if not (args.judge_model or "").strip() and not args.dry_run:
        parser.error("Set JUDGE_MODEL or pass --judge-model")
    if not args.base_url and not args.dry_run:
        parser.error("Set JUDGE_BASE_URL or pass --base-url")
    if args.num_workers < 1 or args.max_retries < 0:
        parser.error("--num-workers must be positive and --max-retries nonnegative")
    return asyncio.run(judge_file(args))


if __name__ == "__main__":
    raise SystemExit(main())
