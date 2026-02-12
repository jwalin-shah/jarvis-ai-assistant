"""Clean extraction evaluation script.

Evaluates candidate extraction against a goldset with ZERO hardcoded entities.
Reports per-label P/R/F1 and macro F1. Only evaluates on dev set (never test).

Usage:
    uv run python scripts/eval_extraction.py --goldset training_data/goldset_v6/dev.json
    uv run python scripts/eval_extraction.py --goldset training_data/goldset_v6/dev.json --extractor spacy
    uv run python scripts/eval_extraction.py --goldset training_data/goldset_v6/dev.json --extractor llm
    uv run python scripts/eval_extraction.py --goldset training_data/goldset_v6/dev.json --extractor hybrid
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Any

# Allow importing eval_shared from scripts/
sys.path.insert(0, os.path.dirname(__file__))
from eval_shared import DEFAULT_LABEL_ALIASES, spans_match  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label normalization (no hardcoded entity lists!)
# ---------------------------------------------------------------------------

# Maps various gold/pred label names to a canonical set.
# This is purely label aliasing, NOT entity-specific rules.
LABEL_NORMALIZE: dict[str, str] = {
    "current_location": "place",
    "future_location": "place",
    "past_location": "place",
    "hometown": "place",
    "employer": "org",
    "school": "org",
    "friend_name": "person_name",
    "partner_name": "person_name",
    "allergy": "health_condition",
    "dietary": "health_condition",
    "hobby": "activity",
    "food_like": "food_item",
    "food_dislike": "food_item",
    "job_title": "job_role",
}

CANONICAL_LABELS = {
    "family_member",
    "person_name",
    "place",
    "org",
    "job_role",
    "food_item",
    "activity",
    "health_condition",
}


def normalize_label(label: str) -> str:
    """Normalize a label to its canonical form."""
    return LABEL_NORMALIZE.get(label, label)


# ---------------------------------------------------------------------------
# Span trimming heuristics (generalizable, no entity-specific rules)
# ---------------------------------------------------------------------------

# Common words to strip from span boundaries
_TRIM_PREFIXES = {"my", "the", "a", "an", "this", "that", "his", "her", "their", "our"}
_TRIM_SUFFIXES = {"too", "also", "tho", "though", "lol", "haha", "like"}


def trim_span(text: str) -> str:
    """Clean up extracted span text with generalizable heuristics."""
    text = text.strip().strip(".,!?;:'\"()[]{}").strip()

    # Strip leading determiners/possessives
    words = text.split()
    if len(words) > 1 and words[0].lower() in _TRIM_PREFIXES:
        text = " ".join(words[1:])

    # Strip trailing filler words
    words = text.split()
    if len(words) > 1 and words[-1].lower() in _TRIM_SUFFIXES:
        text = " ".join(words[:-1])

    return text.strip()


# ---------------------------------------------------------------------------
# Transient family filter (generalizable linguistic patterns only)
# ---------------------------------------------------------------------------

# These patterns indicate TRANSIENT mentions of family, not lasting facts.
# e.g., "my mom called" = transient event, "my mom is a nurse" = lasting fact
import re

_TRANSIENT_FAMILY_RE = re.compile(
    r"\b(?:my\s+(?:mom|dad|sister|brother|mother|father|aunt|uncle|cousin|"
    r"grandma|grandpa|grandmother|grandfather))\s+"
    r"(?:just|never|didn't|didn't|won't|won't|can't|can't|"
    r"called|texted|said|told|asked|came|left|went|is coming|"
    r"is leaving|was here|was there|dropped|picked|sent|"
    r"gave|brought|took|made me|wants me|needs me)",
    re.IGNORECASE,
)


def is_transient_family_mention(text: str) -> bool:
    """Check if a family mention is transient (event-based, not a lasting fact)."""
    return bool(_TRANSIENT_FAMILY_RE.search(text))


# ---------------------------------------------------------------------------
# Extraction methods
# ---------------------------------------------------------------------------


def extract_spacy(text: str, message_id: int) -> list[dict[str, str]]:
    """Extract candidates using spaCy NER only."""
    try:
        import spacy
    except ImportError:
        logger.error("spaCy not installed. Run: uv pip install spacy")
        return []

    nlp = _get_spacy_model()
    doc = nlp(text)

    # Map spaCy labels to our taxonomy
    spacy_to_ours = {
        "PERSON": "person_name",
        "ORG": "org",
        "GPE": "place",
        "LOC": "place",
        "FAC": "place",
    }

    candidates = []
    seen: set[tuple[str, str]] = set()
    for ent in doc.ents:
        our_label = spacy_to_ours.get(ent.label_)
        if our_label is None:
            continue
        span_text = trim_span(ent.text)
        if len(span_text) < 2:
            continue
        key = (span_text.lower(), our_label)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"span_text": span_text, "span_label": our_label})

    return candidates


_spacy_nlp = None


def _get_spacy_model():
    """Lazy-load spaCy model (singleton)."""
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp


def extract_llm(
    text: str,
    message_id: int,
    context_prev: str = "",
    context_next: str = "",
    few_shot_examples: list[dict] | None = None,
) -> list[dict[str, str]]:
    """Extract candidates using LLM (cleaned prompt, no hardcoded entities).

    Uses the system prompt from the goldset labeler with few-shot examples
    drawn ONLY from the train split.
    """
    # Build the prompt
    system_prompt = _build_llm_system_prompt(few_shot_examples or [])
    user_prompt = _build_llm_user_prompt(text, context_prev, context_next)

    # Call the LLM
    try:
        from models.generator import get_generator
        gen = get_generator()
        response = gen.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=512,
            temperature=0.1,
        )
        return _parse_llm_response(response)
    except Exception as e:
        logger.warning("LLM extraction failed: %s", e)
        return []


def _build_llm_system_prompt(few_shot_examples: list[dict]) -> str:
    """Build LLM system prompt for extraction."""
    prompt = """You are a personal fact extractor. Given an iMessage, extract lasting personal facts as structured spans.

Labels: family_member, person_name, place, org, job_role, food_item, activity, health_condition

Rules:
- Only extract LASTING personal facts (not transient events)
- Extract minimal spans (just the entity, not surrounding words)
- Skip vague references (it, that, stuff)
- Skip bot/spam messages
- Output JSON array of {span_text, span_label} or empty array []
"""

    if few_shot_examples:
        prompt += "\n## Examples\n\n"
        for ex in few_shot_examples[:5]:  # Max 5 few-shot examples
            msg = ex.get("message_text", "")
            candidates = ex.get("expected_candidates", [])
            prompt += f'Message: "{msg}"\n'
            prompt += f"Output: {json.dumps(candidates)}\n\n"

    return prompt


def _build_llm_user_prompt(
    text: str, context_prev: str = "", context_next: str = "",
) -> str:
    """Build LLM user prompt for a single message."""
    parts = []
    if context_prev:
        parts.append(f"Previous messages: {context_prev}")
    parts.append(f'Message: "{text}"')
    if context_next:
        parts.append(f"Next messages: {context_next}")
    parts.append(
        "\nExtract lasting personal facts as JSON array of "
        '{span_text, span_label} objects. Output [] if none.'
    )
    return "\n".join(parts)


def _parse_llm_response(response: str) -> list[dict[str, str]]:
    """Parse LLM response into span dicts with fallbacks."""
    response = response.strip()

    # Try direct JSON parse
    try:
        result = json.loads(response)
        if isinstance(result, list):
            return _validate_spans(result)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(1))
            if isinstance(result, list):
                return _validate_spans(result)
        except json.JSONDecodeError:
            pass

    # Try finding array in response
    bracket_match = re.search(r"\[.*\]", response, re.DOTALL)
    if bracket_match:
        try:
            result = json.loads(bracket_match.group(0))
            if isinstance(result, list):
                return _validate_spans(result)
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse LLM response: %s", response[:100])
    return []


def _validate_spans(spans: list) -> list[dict[str, str]]:
    """Validate and clean span dicts from LLM output."""
    valid = []
    for s in spans:
        if not isinstance(s, dict):
            continue
        text = s.get("span_text", "").strip()
        label = s.get("span_label", "").strip()
        if text and label:
            valid.append({"span_text": trim_span(text), "span_label": normalize_label(label)})
    return valid


def extract_hybrid(
    text: str,
    message_id: int,
    context_prev: str = "",
    context_next: str = "",
    few_shot_examples: list[dict] | None = None,
) -> list[dict[str, str]]:
    """Hybrid extraction: union of spaCy + LLM, deduplicated."""
    spacy_candidates = extract_spacy(text, message_id)
    llm_candidates = extract_llm(
        text, message_id, context_prev, context_next, few_shot_examples,
    )

    # Merge: LLM takes priority on overlap
    merged = list(llm_candidates)
    llm_keys = {(c["span_text"].lower(), c["span_label"]) for c in llm_candidates}

    for c in spacy_candidates:
        key = (c["span_text"].lower(), c["span_label"])
        if key not in llm_keys:
            # Check for partial overlap with any LLM candidate
            overlaps = False
            for lc in llm_candidates:
                from eval_shared import jaccard_tokens
                if (
                    c["span_label"] == lc["span_label"]
                    and jaccard_tokens(c["span_text"], lc["span_text"]) > 0.5
                ):
                    overlaps = True
                    break
            if not overlaps:
                merged.append(c)

    return merged


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    goldset: list[dict[str, Any]],
    extractor: str = "hybrid",
    few_shot_examples: list[dict] | None = None,
) -> dict[str, Any]:
    """Evaluate extraction on a goldset.

    Args:
        goldset: List of goldset messages with expected_candidates.
        extractor: One of "spacy", "llm", "hybrid".
        few_shot_examples: Few-shot examples for LLM (from train split only).

    Returns:
        Dict with per-label and macro metrics.
    """
    extract_fn = {
        "spacy": lambda msg: extract_spacy(msg["message_text"], msg["message_id"]),
        "llm": lambda msg: extract_llm(
            msg["message_text"],
            msg["message_id"],
            msg.get("context_prev", ""),
            msg.get("context_next", ""),
            few_shot_examples,
        ),
        "hybrid": lambda msg: extract_hybrid(
            msg["message_text"],
            msg["message_id"],
            msg.get("context_prev", ""),
            msg.get("context_next", ""),
            few_shot_examples,
        ),
    }

    if extractor not in extract_fn:
        raise ValueError(f"Unknown extractor: {extractor}. Use: spacy, llm, hybrid")

    fn = extract_fn[extractor]

    # Per-label counters
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn_count: dict[str, int] = defaultdict(int)

    total = len(goldset)
    start_time = time.time()

    for idx, msg in enumerate(goldset):
        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total - idx - 1) / rate if rate > 0 else 0
            print(
                f"  [{idx + 1}/{total}] {rate:.1f} msg/s, ETA: {eta:.0f}s",
                flush=True,
            )

        # Get predictions
        preds = fn(msg)

        # Get gold candidates
        gold = msg.get("expected_candidates", [])

        # Normalize labels
        for p in preds:
            p["span_label"] = normalize_label(p["span_label"])
        for g in gold:
            g["span_label"] = normalize_label(g.get("span_label", ""))

        # Match predictions to gold
        gold_matched = [False] * len(gold)
        pred_matched = [False] * len(preds)

        for pi, pred in enumerate(preds):
            for gi, g in enumerate(gold):
                if gold_matched[gi]:
                    continue
                if spans_match(
                    pred["span_text"],
                    pred["span_label"],
                    g["span_text"],
                    g["span_label"],
                    label_aliases=DEFAULT_LABEL_ALIASES,
                ):
                    gold_matched[gi] = True
                    pred_matched[pi] = True
                    tp[pred["span_label"]] += 1
                    break

        # Unmatched predictions = false positives
        for pi, pred in enumerate(preds):
            if not pred_matched[pi]:
                fp[pred["span_label"]] += 1

        # Unmatched gold = false negatives
        for gi, g in enumerate(gold):
            if not gold_matched[gi]:
                fn_count[g["span_label"]] += 1

    # Compute per-label metrics
    all_labels = sorted(set(tp.keys()) | set(fp.keys()) | set(fn_count.keys()))
    per_label: dict[str, dict[str, float]] = {}
    for label in all_labels:
        t = tp[label]
        f = fp[label]
        n = fn_count[label]
        precision = t / (t + f) if (t + f) > 0 else 0.0
        recall = t / (t + n) if (t + n) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_label[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "tp": t,
            "fp": f,
            "fn": n,
            "support": t + n,
        }

    # Macro F1 (only over labels with support > 0)
    labels_with_support = [l for l in all_labels if per_label[l]["support"] > 0]
    macro_f1 = (
        sum(per_label[l]["f1"] for l in labels_with_support) / len(labels_with_support)
        if labels_with_support
        else 0.0
    )

    # Micro F1
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn_count.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    )

    return {
        "extractor": extractor,
        "num_messages": total,
        "macro_f1": round(macro_f1, 3),
        "micro_f1": round(micro_f1, 3),
        "micro_precision": round(micro_p, 3),
        "micro_recall": round(micro_r, 3),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "per_label": per_label,
    }


def print_results(results: dict[str, Any]) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'=' * 70}", flush=True)
    print(f"Extractor: {results['extractor']}", flush=True)
    print(f"Messages:  {results['num_messages']}", flush=True)
    print(f"{'=' * 70}", flush=True)

    print(f"\n{'Label':<20} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'Sup':>5}", flush=True)
    print("-" * 70, flush=True)

    for label, metrics in sorted(results["per_label"].items()):
        print(
            f"{label:<20} {metrics['precision']:>6.3f} {metrics['recall']:>6.3f} "
            f"{metrics['f1']:>6.3f} {metrics['tp']:>5} {metrics['fp']:>5} "
            f"{metrics['fn']:>5} {metrics['support']:>5}",
            flush=True,
        )

    print("-" * 70, flush=True)
    print(
        f"{'MICRO':>20} {results['micro_precision']:>6.3f} {results['micro_recall']:>6.3f} "
        f"{results['micro_f1']:>6.3f} {results['total_tp']:>5} {results['total_fp']:>5} "
        f"{results['total_fn']:>5}",
        flush=True,
    )
    print(
        f"{'MACRO F1':>20} {'':>6} {'':>6} {results['macro_f1']:>6.3f}",
        flush=True,
    )
    print(f"{'=' * 70}\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fact extraction on goldset")
    parser.add_argument(
        "--goldset",
        required=True,
        help="Path to goldset JSON (should be dev.json, NEVER test.json)",
    )
    parser.add_argument(
        "--extractor",
        choices=["spacy", "llm", "hybrid"],
        default="hybrid",
        help="Extraction method to evaluate",
    )
    parser.add_argument(
        "--train-set",
        default=None,
        help="Path to train.json for few-shot examples (LLM/hybrid only)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write results JSON",
    )
    args = parser.parse_args()

    # Safety check: warn if evaluating on test set
    if "test.json" in args.goldset:
        print(
            "WARNING: You are evaluating on the TEST set. "
            "This should ONLY be done for the final evaluation!",
            flush=True,
        )
        response = input("Continue? [y/N] ").strip().lower()
        if response != "y":
            print("Aborted.", flush=True)
            sys.exit(1)

    # Load goldset
    print(f"Loading goldset from {args.goldset}...", flush=True)
    with open(args.goldset) as f:
        goldset = json.load(f)
    print(f"Loaded {len(goldset)} messages", flush=True)

    # Load few-shot examples from train set
    few_shot = None
    if args.train_set and args.extractor in ("llm", "hybrid"):
        print(f"Loading few-shot examples from {args.train_set}...", flush=True)
        with open(args.train_set) as f:
            train_data = json.load(f)
        # Select examples that have candidates (positive examples)
        few_shot = [
            ex for ex in train_data if ex.get("expected_candidates")
        ][:5]
        print(f"Using {len(few_shot)} few-shot examples", flush=True)

    # Run evaluation
    print(f"\nRunning {args.extractor} extraction...", flush=True)
    start = time.time()
    results = evaluate(goldset, extractor=args.extractor, few_shot_examples=few_shot)
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.1f}s", flush=True)

    # Print results
    print_results(results)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
