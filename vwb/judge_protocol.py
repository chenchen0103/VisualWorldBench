"""Open-QA judge prompt construction and response parsing."""
import re
import json
from .score_open_qa import extract_final_name

JUDGE_SYSTEM = 'You are a taxonomy and fine-grained recognition expert.\nYour task: determine whether a model\'s prediction refers to the SAME specific species/object as the ground truth.\nBe strict — only judge "correct" if they refer to the exact same species or object (synonyms, common name variants, and minor spelling differences are acceptable).\nIf the prediction is at a coarser level (e.g., genus instead of species, or "bird" instead of "American Robin"), judge "wrong".'
JUDGE_USER_TEMPLATE = 'Ground truth: {gt_name}\nAccepted aliases: {aliases}\nModel\'s prediction: {pred_text}\n\nDoes the model\'s prediction refer to the same specific species/object as the ground truth?\n\nOutput exactly one JSON object (no other text):\n{{"judgment": "correct" or "wrong", "reason": "one sentence explanation"}}'

def build_judge_message(row: dict) -> list[dict]:
    gt_name = row.get("gt_name", "")
    aliases = row.get("gt_aliases_base") or []
    pred_text = extract_final_name(str(row.get("model_output") or ""))

    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
            gt_name=gt_name,
            aliases=", ".join(str(a) for a in aliases) if aliases else "(none)",
            pred_text=pred_text or "(empty)",
        )},
    ]

def parse_judge_response(text: str) -> dict:
    """Parse judge LLM output into {judgment, reason}."""
    text = (text or "").strip()
    # Accept the requested JSON object, optionally inside a Markdown code fence.
    # Ambiguous prose or malformed output is a judge failure, not a wrong answer.
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            judgment = str(obj.get("judgment", "")).lower().strip()
            if judgment in ("correct", "wrong"):
                return {"judgment": judgment, "reason": str(obj.get("reason", ""))}
    except json.JSONDecodeError:
        pass
    return {"judgment": "error", "reason": "invalid_judge_response"}
