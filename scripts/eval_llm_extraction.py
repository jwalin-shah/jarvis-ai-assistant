#!/usr/bin/env python3
"""Evaluate LFM-1.2B as a structured fact extractor against the gold set.

Uses NuExtract-style extraction prompts to test whether our already-loaded
LFM model can extract personal facts from iMessage text, potentially replacing
GLiNER with zero additional model downloads.

Usage:
    uv run python scripts/eval_llm_extraction.py
    uv run python scripts/eval_llm_extraction.py --limit 50
    uv run python scripts/eval_llm_extraction.py --gold PATH --output-dir results/

Output:
    - Printed P/R/F1 comparison table
    - JSONL predictions file for manual inspection
    - JSON metrics file
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Adjust path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from eval_shared import DEFAULT_LABEL_ALIASES, spans_match

logger = logging.getLogger(__name__)

DEFAULT_GOLD_PATH = Path("training_data/gliner_goldset/candidate_gold_merged_r4.json")
OUTPUT_DIR = Path("results/llm_extraction")

# JSON schema for extraction (used in system prompt per Liquid AI recommendation)
EXTRACTION_SCHEMA = """{
  "family": [{"name": "", "relation": ""}],
  "friends": [{"name": ""}],
  "location": {"current": "", "moving_to": ""},
  "work": {"employer": "", "job_title": ""},
  "school": "",
  "health": {"conditions": [], "allergies": []},
  "food": {"likes": [], "dislikes": []},
  "hobbies": [],
  "pets": [{"name": "", "type": ""}]
}"""

# System prompt for the Extract model (per Liquid AI docs: schema in system prompt)
EXTRACT_SYSTEM_PROMPT = (
    "Extract personal facts about the message sender. "
    "Return data as a JSON object with the following schema:\n"
    + EXTRACTION_SCHEMA
)

# Fallback prompt for Instruct model (all-in-one user message)
INSTRUCT_USER_PROMPT = (
    "Extract personal facts about the sender from this text message. "
    "Fill ONLY fields explicitly mentioned. Leave empty strings and empty lists "
    "for anything not mentioned. Return valid JSON only.\n\n"
    "Schema:\n" + EXTRACTION_SCHEMA + "\n\n"
    "Text: {message_text}\n\nJSON:"
)

# Map structured JSON fields to (span_label, fact_type) for evaluation
FIELD_TO_LABEL: dict[tuple[str, ...], tuple[str, str]] = {
    ("family", "name"): ("family_member", "relationship.family"),
    ("family", "relation"): ("family_member", "relationship.family"),
    ("friends", "name"): ("person_name", "relationship.friend"),
    ("location", "current"): ("place", "location.current"),
    ("location", "moving_to"): ("place", "location.future"),
    ("work", "employer"): ("org", "work.employer"),
    ("work", "job_title"): ("job_role", "work.job_title"),
    ("school",): ("org", "personal.school"),
    ("health", "conditions"): ("health_condition", "health.condition"),
    ("health", "allergies"): ("health_condition", "health.allergy"),
    ("food", "likes"): ("food_item", "preference.food_like"),
    ("food", "dislikes"): ("food_item", "preference.food_dislike"),
    ("hobbies",): ("activity", "preference.activity"),
    ("pets", "name"): ("person_name", "personal.pet"),
    ("pets", "type"): ("activity", "personal.pet"),
}


@dataclass
class Metrics:
    """Metrics container."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


def setup_logging() -> None:
    """Configure logging with file + console output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("llm_extraction_eval.log", mode="w"),
        ],
    )


def load_gold_set(gold_path: Path) -> list[dict]:
    """Load gold labeled dataset."""
    logger.info("Loading gold set from %s", gold_path)
    with open(gold_path) as f:
        data = json.load(f)
    logger.info("Loaded %d gold records", len(data))
    return data


def parse_llm_json(raw_text: str) -> dict | None:
    """Parse JSON from LLM output with fallbacks for common issues.

    Handles:
    - Markdown code fences (```json ... ```)
    - Leading/trailing whitespace
    - Partial JSON (truncated output)
    - Extra text before/after JSON

    Returns:
        Parsed dict or None if unparseable.
    """
    text = raw_text.strip()

    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    brace_start = text.find("{")
    if brace_start == -1:
        return None

    # Find matching closing brace
    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[brace_start : i + 1])
                except json.JSONDecodeError:
                    break

    # Try to fix common LLM JSON errors before giving up
    json_text = text[brace_start:]

    # Fix: mismatched brackets/braces (e.g., [{"name": "x"]] -> [{"name": "x"}])
    # Replace ]] with }] and ]} with }] when they look wrong
    fixed = json_text
    # Fix double closing brackets: ]] -> }]
    fixed = re.sub(r'"\](\]|,)', r'"}\1', fixed)
    # Fix array-close then brace-close mismatch: "] instead of "}
    fixed = re.sub(r'"\]\s*,\s*"', '"},  "', fixed)
    try:
        result = json.loads(fixed)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try to fix truncated JSON by closing brackets
    for closer in ["}", "]}", "]}}", "]}}",  "]}]}}"]:
        try:
            result = json.loads(json_text + closer)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

        # Also try with the fixed version
        try:
            result = json.loads(fixed + closer)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

    return None


def json_to_spans(parsed: dict) -> list[dict]:
    """Convert parsed extraction JSON to span predictions for evaluation.

    Returns list of {"span_text": ..., "span_label": ..., "fact_type": ...}
    """
    spans: list[dict] = []

    def _add_span(text: str, label: str, fact_type: str) -> None:
        text = text.strip()
        if text and len(text) > 1:
            spans.append({
                "span_text": text,
                "span_label": label,
                "fact_type": fact_type,
            })

    # Family members
    for member in parsed.get("family", []):
        if isinstance(member, dict):
            name = member.get("name", "")
            relation = member.get("relation", "")
            if name:
                _add_span(name, "family_member", "relationship.family")
            if relation and relation != name:
                _add_span(relation, "family_member", "relationship.family")
        elif isinstance(member, str) and member.strip():
            _add_span(member, "family_member", "relationship.family")

    # Friends
    for friend in parsed.get("friends", []):
        if isinstance(friend, dict):
            name = friend.get("name", "")
            if name:
                _add_span(name, "person_name", "relationship.friend")
        elif isinstance(friend, str) and friend.strip():
            _add_span(friend, "person_name", "relationship.friend")

    # Location
    loc = parsed.get("location", {})
    if isinstance(loc, dict):
        if loc.get("current"):
            _add_span(loc["current"], "place", "location.current")
        if loc.get("moving_to"):
            _add_span(loc["moving_to"], "place", "location.future")
    elif isinstance(loc, str) and loc.strip():
        _add_span(loc, "place", "location.current")

    # Work
    work = parsed.get("work", {})
    if isinstance(work, dict):
        if work.get("employer"):
            _add_span(work["employer"], "org", "work.employer")
        if work.get("job_title"):
            _add_span(work["job_title"], "job_role", "work.job_title")

    # School
    school = parsed.get("school", "")
    if isinstance(school, str) and school.strip():
        _add_span(school, "org", "personal.school")

    # Health
    health = parsed.get("health", {})
    if isinstance(health, dict):
        for cond in health.get("conditions", []):
            if isinstance(cond, str) and cond.strip():
                _add_span(cond, "health_condition", "health.condition")
        for allergy in health.get("allergies", []):
            if isinstance(allergy, str) and allergy.strip():
                _add_span(allergy, "health_condition", "health.allergy")

    # Food
    food = parsed.get("food", {})
    if isinstance(food, dict):
        for item in food.get("likes", []):
            if isinstance(item, str) and item.strip():
                _add_span(item, "food_item", "preference.food_like")
        for item in food.get("dislikes", []):
            if isinstance(item, str) and item.strip():
                _add_span(item, "food_item", "preference.food_dislike")

    # Hobbies
    for hobby in parsed.get("hobbies", []):
        if isinstance(hobby, str) and hobby.strip():
            _add_span(hobby, "activity", "preference.activity")

    # Pets
    for pet in parsed.get("pets", []):
        if isinstance(pet, dict):
            name = pet.get("name", "")
            ptype = pet.get("type", "")
            if name:
                _add_span(name, "person_name", "personal.pet")
            if ptype:
                _add_span(ptype, "activity", "personal.pet")
        elif isinstance(pet, str) and pet.strip():
            _add_span(pet, "person_name", "personal.pet")

    return spans


def compute_metrics(
    gold_records: list[dict],
    predictions: dict[int, list[dict]],
) -> dict:
    """Compute P/R/F1 metrics matching predictions against gold spans."""
    overall = Metrics()
    per_label: dict[str, Metrics] = defaultdict(Metrics)

    for rec in gold_records:
        msg_id = rec["message_id"]
        gold_cands = rec.get("expected_candidates") or []
        pred_cands = predictions.get(msg_id, [])

        gold_matched = [False] * len(gold_cands)
        pred_matched = [False] * len(pred_cands)

        # Match predictions to gold
        for gi, gc in enumerate(gold_cands):
            for pi, pc in enumerate(pred_cands):
                if pred_matched[pi]:
                    continue
                if spans_match(
                    pc.get("span_text", ""),
                    pc.get("span_label", ""),
                    gc.get("span_text", ""),
                    gc.get("span_label", ""),
                    label_aliases=DEFAULT_LABEL_ALIASES,
                ):
                    gold_matched[gi] = True
                    pred_matched[pi] = True
                    overall.tp += 1
                    per_label[gc["span_label"]].tp += 1
                    break

        # Unmatched gold = FN
        for gi, gc in enumerate(gold_cands):
            if not gold_matched[gi]:
                overall.fn += 1
                per_label[gc["span_label"]].fn += 1

        # Unmatched preds = FP
        for pi, pc in enumerate(pred_cands):
            if not pred_matched[pi]:
                overall.fp += 1
                label = pc.get("span_label", "unknown")
                per_label[label].fp += 1

    return {
        "overall": overall.to_dict(),
        "per_label": {k: v.to_dict() for k, v in sorted(per_label.items())},
    }


MODEL_CONFIGS = {
    "lfm2-350m-extract": {
        "display": "LFM2-350M-Extract",
        "path": "models/lfm2-350m-extract-mlx-4bit",
        "use_system_prompt": True,
    },
    "lfm2-extract": {
        "display": "LFM2-1.2B-Extract",
        "path": "models/lfm2-1.2b-extract-mlx-4bit",
        "use_system_prompt": True,  # Per Liquid AI docs: schema in system prompt
    },
    "lfm25-instruct": {
        "display": "LFM2.5-1.2B-Instruct",
        "path": "LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit",
        "use_system_prompt": False,  # All-in-one user message
    },
}


def _build_prompt(
    msg_text: str,
    tokenizer: Any,
    use_system_prompt: bool,
) -> str:
    """Build a properly formatted chat prompt for extraction.

    For Extract model: system prompt has schema, user message is just the text.
    For Instruct model: user message has everything.
    """
    if use_system_prompt:
        messages = [
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": msg_text},
        ]
    else:
        messages = [
            {"role": "user", "content": INSTRUCT_USER_PROMPT.replace("{message_text}", msg_text)},
        ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run_llm_extraction(
    gold_records: list[dict],
    model_key: str = "lfm2-extract",
    limit: int | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Run LLM extraction on the gold set.

    Loads the specified model, runs extraction on each message, parses JSON
    output, converts to spans, and scores against gold.
    """
    from models.loader import MLXModelLoader, ModelConfig

    cfg = MODEL_CONFIGS[model_key]
    records = gold_records[:limit] if limit else gold_records
    logger.info("Running %s extraction on %d messages...", cfg["display"], len(records))

    # Load model
    logger.info("Loading %s from %s...", cfg["display"], cfg["path"])
    loader = MLXModelLoader(ModelConfig(model_path=cfg["path"]))
    load_start = time.time()
    loader.load()
    logger.info("Model loaded in %.1fs", time.time() - load_start)

    # Run extraction
    predictions: dict[int, list[dict]] = {}
    timing: dict[int, float] = {}
    parse_failures = 0
    empty_outputs = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    incremental_path = output_dir / f"{model_key}_predictions.jsonl"

    start_time = time.time()

    with open(incremental_path, "w") as inc_f:
        for i, rec in enumerate(records):
            if (i + 1) % 50 == 0 or i == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(records) - i - 1) / rate if rate > 0 else 0
                logger.info(
                    "  [%s] %d/%d messages (%.0fs elapsed, ETA %.0fs, "
                    "parse_fail=%d, empty=%d)",
                    cfg["display"],
                    i + 1,
                    len(records),
                    elapsed,
                    eta,
                    parse_failures,
                    empty_outputs,
                )

            msg_id = rec["message_id"]
            msg_text = rec["message_text"]

            # Build prompt using chat template
            prompt = _build_prompt(
                msg_text, loader._tokenizer, cfg["use_system_prompt"]
            )

            # Generate with temperature=0 (greedy) per Liquid AI recommendation
            msg_start = time.perf_counter()
            raw_output = ""
            parsed = None
            try:
                result = loader.generate_sync(
                    prompt=prompt,
                    max_tokens=300,
                    temperature=0.0,
                    top_p=1.0,
                    repetition_penalty=1.0,
                    timeout_seconds=30.0,
                    pre_formatted=True,
                )
                elapsed_ms = (time.perf_counter() - msg_start) * 1000
                raw_output = result.text

                # Parse JSON
                parsed = parse_llm_json(raw_output)
                if parsed is None:
                    parse_failures += 1
                    pred_list = []
                elif not any(
                    v
                    for v in parsed.values()
                    if v and v != "" and v != [] and v != {}
                ):
                    empty_outputs += 1
                    pred_list = []
                else:
                    pred_list = json_to_spans(parsed)

                predictions[msg_id] = pred_list
                timing[msg_id] = elapsed_ms

            except Exception as e:
                logger.error("Generation failed for message %d: %s", msg_id, e)
                predictions[msg_id] = []
                timing[msg_id] = 0
                raw_output = f"ERROR: {e}"

            # Write incrementally
            inc_f.write(
                json.dumps({
                    "message_id": msg_id,
                    "message_text": msg_text,
                    "raw_output": raw_output,
                    "parsed": parsed,
                    "predictions": predictions[msg_id],
                    "elapsed_ms": timing.get(msg_id, 0),
                })
                + "\n"
            )
            inc_f.flush()

    total_time = time.time() - start_time
    total_spans = sum(len(v) for v in predictions.values())
    msgs_with_preds = sum(1 for v in predictions.values() if v)
    parse_rate = (len(records) - parse_failures) / len(records) * 100

    logger.info(
        "%s extraction complete in %.1fs: %d spans from %d/%d messages "
        "(parse rate: %.1f%%, empty: %d)",
        cfg["display"],
        total_time,
        total_spans,
        msgs_with_preds,
        len(records),
        parse_rate,
        empty_outputs,
    )

    # Compute metrics
    metrics = compute_metrics(records, predictions)
    metrics["extractor_name"] = cfg["display"]
    metrics["model_key"] = model_key
    metrics["num_messages"] = len(records)
    metrics["total_time_s"] = round(total_time, 2)
    metrics["ms_per_message"] = round(total_time / len(records) * 1000, 1) if records else 0
    metrics["parse_failures"] = parse_failures
    metrics["parse_rate_pct"] = round(parse_rate, 1)
    metrics["empty_outputs"] = empty_outputs

    # Unload model to free memory
    loader.unload()
    gc.collect()

    return metrics


def print_results(all_metrics: list[dict]) -> None:
    """Print comparison table of all models vs GLiNER baseline."""
    print("\n" + "=" * 78)
    print("LLM EXTRACTION BAKEOFF")
    print("=" * 78)

    # GLiNER baseline from previous bakeoff
    gliner_baseline = {"precision": 0.337, "recall": 0.263, "f1": 0.295}

    header = "{:24} {:>7} {:>7} {:>7} {:>9} {:>7} {:>8}".format(
        "Extractor", "P", "R", "F1", "Time/msg", "Parse%", "Empty"
    )
    print(f"\n{header}")
    print("-" * 78)

    # GLiNER baseline row
    print(
        "{:24} {:>7.3f} {:>7.3f} {:>7.3f} {:>7}ms {:>7} {:>8}".format(
            "GLiNER (baseline)",
            gliner_baseline["precision"],
            gliner_baseline["recall"],
            gliner_baseline["f1"],
            "~5",
            "n/a",
            "n/a",
        )
    )

    best_f1 = 0.0
    best_name = ""
    for m in all_metrics:
        ov = m["overall"]
        print(
            "{:24} {:>7.3f} {:>7.3f} {:>7.3f} {:>7.0f}ms {:>6.1f}% {:>8}".format(
                m["extractor_name"],
                ov["precision"],
                ov["recall"],
                ov["f1"],
                m["ms_per_message"],
                m["parse_rate_pct"],
                m["empty_outputs"],
            )
        )
        if ov["f1"] > best_f1:
            best_f1 = ov["f1"]
            best_name = m["extractor_name"]

    # Per-label breakdown for best model
    print("\n\nPer-Label Breakdown (best: %s):" % best_name)
    best_m = next(m for m in all_metrics if m["extractor_name"] == best_name)
    print(
        "  {:20} {:>6} {:>6} {:>6} {:>6}".format("Label", "P", "R", "F1", "Sup")
    )
    print("  " + "-" * 50)
    for label, lm in sorted(best_m["per_label"].items()):
        support = lm["tp"] + lm["fn"]
        if support > 0:
            print(
                "  {:20} {:>6.3f} {:>6.3f} {:>6.3f} {:>6}".format(
                    label[:20], lm["precision"], lm["recall"], lm["f1"], support
                )
            )

    # Verdict
    print("\n" + "-" * 78)
    if best_f1 > 0.35:
        print("VERDICT: %s PASSES threshold (F1=%.3f > 0.35)" % (best_name, best_f1))
        print("  -> Use this model for LLM extraction pipeline")
    elif best_f1 > gliner_baseline["f1"]:
        print(
            "VERDICT: %s beats GLiNER (F1=%.3f > %.3f) but below 0.35"
            % (best_name, best_f1, gliner_baseline["f1"])
        )
        print("  -> Prompt tuning or try NuExtract-tiny")
    else:
        print("VERDICT: No LLM model beats GLiNER baseline (best F1=%.3f)" % best_f1)
        print("  -> Download NuExtract-tiny for dedicated extraction")

    print("=" * 78)


def main() -> None:
    """Main entry point."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Evaluate LFM models as structured fact extractors"
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help="Path to gold set JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to first N messages",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="lfm2-350m-extract,lfm2-extract",
        help="Comma-separated model keys: " + ", ".join(MODEL_CONFIGS.keys()),
    )
    args = parser.parse_args()

    if not args.gold.exists():
        logger.error("Gold set not found: %s", args.gold)
        sys.exit(1)

    # Load gold set
    gold_records = load_gold_set(args.gold)

    # Run each model sequentially (memory constraint: one model at a time)
    model_keys = [k.strip() for k in args.models.split(",")]
    all_metrics: list[dict] = []

    for model_key in model_keys:
        if model_key not in MODEL_CONFIGS:
            logger.error("Unknown model key: %s (available: %s)",
                         model_key, ", ".join(MODEL_CONFIGS.keys()))
            continue

        metrics = run_llm_extraction(
            gold_records,
            model_key=model_key,
            limit=args.limit,
            output_dir=args.output_dir,
        )
        all_metrics.append(metrics)

        # Save individual metrics
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = args.output_dir / f"{model_key}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved %s metrics to %s", model_key, metrics_path)

    # Print comparison
    if all_metrics:
        print_results(all_metrics)


if __name__ == "__main__":
    main()
