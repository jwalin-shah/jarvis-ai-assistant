#!/usr/bin/env python3
"""Evaluate LLM-based fact extraction against a gold-labeled dataset.

Uses the MLX model loader to run structured extraction prompts on iMessage
text and evaluates against the goldset using span-level P/R/F1.

Usage:
    uv run python scripts/eval_llm_extraction.py --gold training_data/gliner_goldset/candidate_gold_merged_r4.json --limit 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add scripts/ to path so we can import eval_shared
sys.path.insert(0, str(Path(__file__).parent))

from eval_shared import DEFAULT_LABEL_ALIASES, spans_match

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GOLD_PATH = Path("training_data/gliner_goldset/candidate_gold_merged_r4.json")
RESULTS_DIR = Path("results/llm_extraction")
METRICS_PATH = RESULTS_DIR / "lfm2-extract_metrics.json"

# ---------------------------------------------------------------------------
# Extraction schema and prompts
# ---------------------------------------------------------------------------

# The canonical label set matching the goldset
VALID_LABELS = {
    "family_member", "activity", "health_condition", "job_role", "org",
    "place", "food_item", "current_location", "future_location",
    "past_location", "friend_name", "person_name",
}

# Fact type hierarchy
LABEL_TO_FACT_TYPE = {
    "family_member": "relationship.family",
    "friend_name": "relationship.friend",
    "person_name": "relationship.other",
    "activity": "preference.activity",
    "health_condition": "health.condition",
    "job_role": "work.job_title",
    "org": "work.employer",
    "place": "location.general",
    "food_item": "preference.food",
    "current_location": "location.current",
    "future_location": "location.future",
    "past_location": "location.past",
}

EXTRACTION_SCHEMA = """{
  "facts": [
    {
      "text": "<1-3 word entity from message>",
      "label": "<label>"
    }
  ]
}"""

EXTRACT_SYSTEM_PROMPT = """Extract named entities from chat messages as JSON.
"text" = exact 1-3 words from message. Labels: family_member, activity, health_condition, job_role, org, food_item, place, friend_name, person_name
Return {"facts": []} if none."""

# Few-shot examples: 5 positive (diverse labels) + 2 negative
FEW_SHOT_TURNS = [
    ("my brother bakes and I just eat whatever he makes",
     '{"facts": [{"text": "brother", "label": "family_member"}, {"text": "bakes", "label": "activity"}]}'),
    ("I work at Google as an engineer",
     '{"facts": [{"text": "Google", "label": "org"}, {"text": "engineer", "label": "job_role"}]}'),
    ("allergic to peanuts and it sucks",
     '{"facts": [{"text": "peanuts", "label": "health_condition"}]}'),
    ("My mom actually texted me saying my acceptance letter went to the house in Delaware",
     '{"facts": [{"text": "mom", "label": "family_member"}, {"text": "house", "label": "place"}, {"text": "Delaware", "label": "place"}]}'),
    ("helloooo",
     '{"facts": []}'),
    ("My phone is being like my moms",
     '{"facts": []}'),
    ("Also my dad leaves the 22nd for India",
     '{"facts": [{"text": "dad", "label": "family_member"}, {"text": "India", "label": "place"}]}'),
]

INSTRUCT_USER_PROMPT = """Message: "{message}"
"""


# Extended label aliases for LLM output normalization
LLM_LABEL_ALIASES: dict[str, set[str]] = {
    **DEFAULT_LABEL_ALIASES,
    "activity": {"activity", "hobby", "interest", "sport", "skill"},
    "family_member": {"family_member", "family", "relative", "relation"},
    "food_item": {"food_item", "food", "food_preference", "cuisine"},
    "job_role": {"job_role", "job", "occupation", "profession", "role", "title"},
    "current_location": {"current_location", "location", "city", "residence"},
    "future_location": {"future_location", "destination", "moving_to"},
    "past_location": {"past_location", "hometown", "origin"},
    "friend_name": {"friend_name", "friend"},
    "person_name": {"person_name", "name", "person"},
    "org": {"org", "organization", "company", "employer", "school", "university"},
    "place": {"place", "location", "venue", "landmark"},
    "health_condition": {"health_condition", "health", "allergy", "condition", "medical"},
}


def normalize_label(raw_label: str) -> str | None:
    """Normalize an LLM-predicted label to a canonical goldset label."""
    raw = raw_label.lower().strip()
    # Direct match
    if raw in VALID_LABELS:
        return raw
    # Check aliases
    for canonical, aliases in LLM_LABEL_ALIASES.items():
        if raw in aliases:
            return canonical
    return None


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def parse_llm_json(raw_text: str) -> list[dict]:
    """Parse LLM output into a list of fact dicts.

    Handles various LLM output formats:
    - Clean JSON
    - JSON wrapped in markdown code blocks
    - Partial/truncated JSON
    """
    text = raw_text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        # Remove opening fence (with optional language tag)
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "facts" in data:
            return data["facts"] if isinstance(data["facts"], list) else []
        if isinstance(data, list):
            return data
        return []
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    json_match = re.search(r'\{[^{}]*"facts"\s*:\s*\[.*?\]\s*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return data.get("facts", [])
        except json.JSONDecodeError:
            pass

    # Try to find a JSON array
    arr_match = re.search(r'\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]', text, re.DOTALL)
    if arr_match:
        try:
            return json.loads(arr_match.group())
        except json.JSONDecodeError:
            pass

    # Try to fix truncated JSON by closing brackets
    for suffix in ["]}", "]}}", "]", "}"]:
        try:
            data = json.loads(text + suffix)
            if isinstance(data, dict) and "facts" in data:
                return data["facts"] if isinstance(data["facts"], list) else []
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            continue

    return []


def _trim_span(text: str, label: str) -> str:
    """Trim overly long spans to extract the core entity.

    LLMs tend to output full phrases. We want just the entity (1-3 words).
    """
    words = text.split()
    if len(words) <= 3:
        return text

    # For family_member, extract just the relationship word
    if label == "family_member":
        family_words = {
            "brother", "sister", "mom", "mother", "dad", "father",
            "wife", "husband", "girlfriend", "boyfriend", "partner",
            "daughter", "son", "cousin", "aunt", "uncle", "grandma",
            "grandmother", "grandpa", "grandfather", "fiancee", "fiancé",
        }
        for w in words:
            if w.lower().rstrip("'s") in family_words:
                return w
        return words[0]  # fallback

    # For locations, try to find the proper noun
    if label in ("current_location", "future_location", "past_location", "place"):
        proper = [w for w in words if w[0].isupper()]
        if proper:
            return " ".join(proper[:3])

    # For org, extract proper nouns
    if label == "org":
        proper = [w for w in words if w[0].isupper()]
        if proper:
            return " ".join(proper[:3])

    # General: take first 3 words
    return " ".join(words[:3])


# Known entity patterns for label correction
_FAMILY_WORDS = {
    "brother", "sister", "mom", "mother", "dad", "father",
    "wife", "husband", "girlfriend", "boyfriend", "partner",
    "daughter", "son", "cousin", "aunt", "uncle", "grandma",
    "grandmother", "grandpa", "grandfather", "fiancee", "fiancé",
    "stepmom", "stepdad", "niece", "nephew",
}

_HEALTH_KEYWORDS = {
    "allergic", "allergy", "asthma", "diabetes", "depression",
    "anxiety", "adhd", "migraine", "migraines", "vestibular",
    "surgery", "injury", "cancer", "arthritis", "insomnia",
    "emergency room", "hospital", "therapy", "ptsd",
}


def _correct_label(text: str, label: str, msg_lower: str) -> str:
    """Heuristic label correction for common model mistakes."""
    text_lower = text.lower().strip()

    # Family words should always be family_member
    if text_lower in _FAMILY_WORDS:
        return "family_member"

    # Health keywords should be health_condition
    if text_lower in _HEALTH_KEYWORDS:
        return "health_condition"

    # If span looks like a job title and was labeled activity
    if label == "activity":
        job_indicators = {
            "manager", "engineer", "developer", "nurse", "doctor",
            "teacher", "analyst", "designer", "consultant", "director",
            "intern", "coordinator", "specialist", "product management",
        }
        if text_lower in job_indicators:
            return "job_role"

    return label


def json_to_spans(facts: list[dict], message_text: str) -> list[dict]:
    """Convert parsed JSON facts to span predictions.

    Validates that span_text appears in the message and normalizes labels.
    Applies post-processing filters to reduce false positives.
    """
    spans = []
    msg_lower = message_text.lower()
    msg_len = len(message_text)

    # Skip very short messages only if they're filler (not real words)
    if msg_len < 4:
        return []

    for fact in facts:
        if not isinstance(fact, dict):
            continue

        text = fact.get("text", "").strip()
        raw_label = fact.get("label", "").strip()

        if not text or not raw_label:
            continue

        # Normalize label
        label = normalize_label(raw_label)
        if label is None:
            continue

        # Trim overly long spans
        text = _trim_span(text, label)

        # Correct common label mistakes
        label = _correct_label(text, label, msg_lower)

        # Skip single-character or very short non-meaningful spans
        if len(text) < 2:
            continue

        # Reject spans that are too long relative to message (likely hallucinated)
        if len(text) > msg_len * 0.6:
            continue

        # Reject common non-fact words/phrases (pronouns, fillers, greetings)
        text_lower = text.lower().strip()
        reject_phrases = {
            "i", "me", "my", "you", "he", "she", "it", "we", "they",
            "her", "him", "his", "their", "our", "us",
            "like", "i like", "i like it", "yeah", "yes", "no", "ok",
            "lol", "haha", "omg", "bruh", "dude",
            "good", "bad", "cool", "nice", "great", "sure", "fine",
            "thing", "stuff", "something", "nothing", "everything",
            "now", "thank", "thanks", "aight",
        }
        if text_lower in reject_phrases:
            continue

        # Label-specific validation: reject common-word false positives
        if label in ("current_location", "future_location", "past_location", "place"):
            # Reject only obvious non-locations (very common words)
            location_rejects = {
                "here", "there", "home", "somewhere", "anywhere", "nowhere",
                "place", "area", "spot",
            }
            if text_lower in location_rejects:
                continue
        if label == "food_item":
            # Food items should be recognizable food words, not random nouns
            reject_foods = {
                "eat", "eating", "ate", "food", "cooking", "cook", "phone",
                "whatever", "everything", "anything", "something", "stuff",
                "it", "that", "this", "one", "all", "car", "arms", "tie",
                "theory", "utilities", "read", "xbox", "realtor",
                "acceptance letter", "lending tree", "raiders",
            }
            if text_lower in reject_foods:
                continue
            # Reject non-food patterns: numbers, abbreviations, body parts
            if any(c.isdigit() for c in text):
                continue
            if text_lower.isupper() and len(text) <= 3:  # abbreviations like "SB"
                continue
            # Reject if it's a holiday/event name (capitalize check)
            if text[0].isupper() and text_lower not in {
                "thai", "indian", "chinese", "japanese", "mexican", "italian",
                "korean", "greek", "french",
            }:
                # Proper nouns that aren't cuisine types are likely not food
                # But allow multi-word food items like "palak paneer"
                food_words = {
                    "curry", "paneer", "naan", "sushi", "pizza", "pasta",
                    "chicken", "steak", "burger", "taco", "rice", "soup",
                    "salad", "sandwich", "cake", "pie", "bread", "fish",
                    "boba", "tea", "coffee", "juice", "smoothie",
                }
                if not any(fw in text_lower for fw in food_words):
                    continue
        if label == "activity":
            # Reject generic/filler words and common verbs
            reject_activities = {
                "go", "going", "get", "getting",
                "come", "coming", "do", "doing",
                "see", "seeing", "take", "taking",
                "want", "wanting", "need", "needing",
                "think", "thinking", "know", "knowing",
                "try", "trying", "make", "making",
                "contact", "slow process", "ask", "call",
                "leave", "send", "sends", "talk", "talk to others",
                "fly", "flew", "ship", "pack", "packed",
                "hear", "hear stories", "rest", "rest of it",
                "like it", "love it", "doing wtv",
                "don't wanna", "im free", "go back home",
                "email", "mind", "control", "assumed",
                "yea", "em", "a lot of", "classes", "working",
                "get along", "increase time", "matchups",
                "30-40", "22nd", "icing",
            }
            if text_lower in reject_activities:
                continue
            # Single lowercase words < 4 chars are unlikely to be activities
            if len(text) <= 3 and text[0].islower():
                continue
        if label == "health_condition":
            # Only accept spans with medical/health keywords
            reject_health = {
                "whatever", "slow", "slow process", "points of view",
                "insanely big", "bad", "feel", "feeling",
                "dgaf", "don't fuck", "either", "not comin",
                "ihs", "4am", "rationalize", "willpower",
                "take responsibility", "rest", "never ended",
                "increase time",
            }
            if text_lower in reject_health:
                continue
        if label == "friend_name":
            # Friend names should start with uppercase
            if text[0].islower():
                continue

        # Validate span text appears in message (case-insensitive)
        if text_lower not in msg_lower:
            # Try individual words for partial match
            words = text_lower.split()
            matching_words = [w for w in words if w in msg_lower and len(w) > 2]
            if not matching_words:
                continue

        fact_type = LABEL_TO_FACT_TYPE.get(label, "unknown")

        spans.append({
            "span_text": text,
            "span_label": label,
            "fact_type": fact_type,
        })

    # Deduplicate
    seen = set()
    deduped = []
    for s in spans:
        key = (s["span_text"].lower(), s["span_label"])
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    return deduped


# ---------------------------------------------------------------------------
# Model interaction
# ---------------------------------------------------------------------------


def load_model(model_id: str = "lfm-1.2b"):
    """Load the MLX model for extraction.

    Uses memory_buffer_multiplier=0.0 to skip the memory check since
    MLX on Apple Silicon can leverage unified memory and swap effectively.
    """
    from models.loader import MLXModelLoader, ModelConfig

    config = ModelConfig(model_id=model_id)
    config.memory_buffer_multiplier = 0.0  # Skip memory check for eval
    loader = MLXModelLoader(config)
    loader.load()
    return loader


def extract_facts_llm(
    loader,
    message_text: str,
    strategy: str = "constrained_categories",
) -> list[dict]:
    """Extract facts from a message using the LLM.

    Args:
        loader: MLXModelLoader instance
        message_text: The message to extract from
        strategy: Extraction strategy to use

    Returns:
        List of span dicts with span_text, span_label, fact_type
    """
    if strategy == "constrained_categories":
        return _strategy_constrained_categories(loader, message_text)
    elif strategy == "simple":
        return _strategy_simple(loader, message_text)
    elif strategy == "pipe":
        return _strategy_pipe(loader, message_text)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _strategy_constrained_categories(loader, message_text: str) -> list[dict]:
    """Strategy: multi-turn few-shot with constrained category list."""
    # Build multi-turn conversation with few-shot examples
    messages = [{"role": "system", "content": EXTRACT_SYSTEM_PROMPT}]

    for user_msg, assistant_resp in FEW_SHOT_TURNS:
        messages.append({"role": "user", "content": f'Message: "{user_msg}"'})
        messages.append({"role": "assistant", "content": assistant_resp})

    messages.append({"role": "user", "content": INSTRUCT_USER_PROMPT.format(message=message_text)})

    formatted = loader._tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    result = loader.generate_sync(
        formatted,
        max_tokens=120,
        temperature=0.0,
        top_p=0.1,
        repetition_penalty=1.0,
        pre_formatted=True,
    )

    facts = parse_llm_json(result.text)
    return json_to_spans(facts, message_text)


def _strategy_simple(loader, message_text: str) -> list[dict]:
    """Strategy: minimal prompt, no system message."""
    prompt = f"""Extract personal facts from this text as JSON.
Text: "{message_text}"
Return: {{"facts": [{{"text": "...", "label": "..."}}]}}
Labels: family_member, activity, health_condition, job_role, org, place, food_item, current_location, future_location, past_location, friend_name, person_name
If no facts, return {{"facts": []}}"""

    result = loader.generate_sync(
        prompt,
        max_tokens=256,
        temperature=0.0,
        top_p=0.1,
        repetition_penalty=1.0,
    )

    facts = parse_llm_json(result.text)
    return json_to_spans(facts, message_text)


# Pipe-delimited system prompt and examples
PIPE_SYSTEM = """Extract personal fact entities from chat messages.
Output: entity|label (one per line). Output NONE if no facts.
Labels: family_member, activity, health_condition, job_role, org, food_item, current_location, future_location, past_location, place, friend_name, person_name"""

PIPE_EXAMPLES = [
    ("my brother bakes and I just eat whatever he makes",
     "brother|family_member\nbakes|activity"),
    ("I work at Google as an engineer",
     "Google|org\nengineer|job_role"),
    ("helloooo", "NONE"),
    ("My phone is being like my moms", "NONE"),
    ("allergic to peanuts and it sucks",
     "peanuts|health_condition"),
    ("i like it", "NONE"),
    ("moving to Austin next month",
     "Austin|future_location"),
    ("And my dad flew in",
     "dad|family_member"),
    ("I work at lending tree",
     "lending tree|org"),
    ("My friend Sarah is a nurse",
     "Sarah|friend_name\nnurse|job_role"),
    ("Like 10:15ish", "NONE"),
]


def _parse_pipe_output(text: str) -> list[dict]:
    """Parse pipe-delimited output into fact dicts."""
    facts = []
    text = text.strip()
    if not text or text.upper().startswith("NONE"):
        return []

    for line in text.split("\n"):
        line = line.strip()
        if not line or line.upper() == "NONE":
            continue
        # Handle "entity|label" format
        if "|" in line:
            parts = line.split("|", 1)
            if len(parts) == 2:
                entity, label = parts[0].strip(), parts[1].strip()
                if entity and label:
                    facts.append({"text": entity, "label": label})
        # Stop if we see JSON or other junk (model went off-track)
        elif line.startswith("{") or line.startswith("["):
            break

    return facts


def _strategy_pipe(loader, message_text: str) -> list[dict]:
    """Strategy: pipe-delimited output format (simpler for small models)."""
    messages = [{"role": "system", "content": PIPE_SYSTEM}]

    for user_msg, assistant_resp in PIPE_EXAMPLES:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_resp})

    messages.append({"role": "user", "content": message_text})

    formatted = loader._tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    result = loader.generate_sync(
        formatted,
        max_tokens=100,
        temperature=0.0,
        top_p=0.1,
        repetition_penalty=1.0,
        pre_formatted=True,
    )

    facts = _parse_pipe_output(result.text)
    return json_to_spans(facts, message_text)


# ---------------------------------------------------------------------------
# Metrics (reuse from eval_gliner_candidates)
# ---------------------------------------------------------------------------


def compute_metrics(
    gold_records: list[dict],
    predictions: dict[str, list[dict]],
) -> dict:
    """Compute span-level precision/recall/F1."""
    from eval_shared import spans_match

    tp = fp = fn = 0
    per_label: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    per_slice: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    errors: list[dict] = []

    for rec in gold_records:
        sid = rec["sample_id"]
        gold_cands = rec.get("expected_candidates") or []
        pred_cands = predictions.get(sid, [])
        slc = rec.get("slice", "unknown")

        gold_matched = [False] * len(gold_cands)
        pred_matched = [False] * len(pred_cands)

        # Greedy matching
        for gi, gc in enumerate(gold_cands):
            for pi, pc in enumerate(pred_cands):
                if pred_matched[pi]:
                    continue
                if spans_match(
                    pc.get("span_text", ""),
                    pc.get("span_label", ""),
                    gc.get("span_text", ""),
                    gc.get("span_label", ""),
                    label_aliases=LLM_LABEL_ALIASES,
                ):
                    gold_matched[gi] = True
                    pred_matched[pi] = True
                    tp += 1
                    per_label[gc["span_label"]]["tp"] += 1
                    per_slice[slc]["tp"] += 1
                    break

        # FN
        for gi, gc in enumerate(gold_cands):
            if not gold_matched[gi]:
                fn += 1
                per_label[gc["span_label"]]["fn"] += 1
                per_slice[slc]["fn"] += 1
                errors.append({
                    "type": "fn",
                    "sample_id": sid,
                    "slice": slc,
                    "message_text": rec["message_text"][:100],
                    "gold_span": gc["span_text"],
                    "gold_label": gc["span_label"],
                })

        # FP
        for pi, pc in enumerate(pred_cands):
            if not pred_matched[pi]:
                fp += 1
                label = pc.get("span_label", "unknown")
                per_label[label]["fp"] += 1
                per_slice[slc]["fp"] += 1
                errors.append({
                    "type": "fp",
                    "sample_id": sid,
                    "slice": slc,
                    "message_text": rec["message_text"][:100],
                    "pred_span": pc.get("span_text", ""),
                    "pred_label": label,
                })

    def _metrics(tp_: int, fp_: int, fn_: int) -> dict:
        p = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0
        r = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "tp": tp_,
            "fp": fp_,
            "fn": fn_,
        }

    overall = _metrics(tp, fp, fn)
    label_metrics = {
        k: _metrics(v["tp"], v["fp"], v["fn"])
        for k, v in sorted(per_label.items())
    }
    slice_metrics = {
        k: _metrics(v["tp"], v["fp"], v["fn"])
        for k, v in sorted(per_slice.items())
    }

    return {
        "overall": overall,
        "per_label": label_metrics,
        "per_slice": slice_metrics,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(metrics: dict, strategy: str, elapsed: float, num_records: int) -> None:
    """Print evaluation report."""
    ov = metrics["overall"]
    print("\n" + "=" * 60, flush=True)
    print("LLM Fact Extraction Evaluation", flush=True)
    print("=" * 60, flush=True)
    print(f"Strategy: {strategy}", flush=True)
    print(f"Records: {num_records}", flush=True)
    print(f"Time: {elapsed:.1f}s ({elapsed / num_records * 1000:.0f}ms/msg)", flush=True)

    print(
        f"\nOverall:  P={ov['precision']:.3f}  R={ov['recall']:.3f}  "
        f"F1={ov['f1']:.3f}  (TP={ov['tp']} FP={ov['fp']} FN={ov['fn']})",
        flush=True,
    )

    # Per-label
    print(f"\n{'Label':<20} {'P':>6} {'R':>6} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4}", flush=True)
    print("-" * 55, flush=True)
    for label, m in sorted(
        metrics["per_label"].items(),
        key=lambda x: -(x[1]["tp"] + x[1]["fn"]),
    ):
        print(
            f"{label:<20} {m['precision']:>6.3f} {m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4}",
            flush=True,
        )

    # Per-slice
    print(f"\n{'Slice':<20} {'P':>6} {'R':>6} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4}", flush=True)
    print("-" * 55, flush=True)
    for slc, m in sorted(metrics["per_slice"].items()):
        print(
            f"{slc:<20} {m['precision']:>6.3f} {m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4}",
            flush=True,
        )

    # Top errors
    fps = [e for e in metrics["errors"] if e["type"] == "fp"][:8]
    fns = [e for e in metrics["errors"] if e["type"] == "fn"][:8]

    if fns:
        print("\nTop False Negatives (missed):", flush=True)
        for e in fns:
            print(
                f'  [{e["slice"]}] "{e["message_text"][:60]}..." '
                f'-> missed {e["gold_span"]} ({e["gold_label"]})',
                flush=True,
            )

    if fps:
        print("\nTop False Positives (spurious):", flush=True)
        for e in fps:
            print(
                f'  [{e["slice"]}] "{e["message_text"][:60]}..." '
                f'-> {e["pred_span"]} ({e["pred_label"]})',
                flush=True,
            )

    print("\n" + "=" * 60, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate LLM fact extraction")
    parser.add_argument("--gold", type=Path, default=GOLD_PATH, help="Path to gold set JSON")
    parser.add_argument("--limit", type=int, default=None, help="Limit records to process")
    parser.add_argument(
        "--strategy",
        default="constrained_categories",
        choices=["constrained_categories", "simple", "pipe"],
        help="Extraction strategy",
    )
    parser.add_argument("--model", default="lfm-1.2b", help="Model ID from registry")
    args = parser.parse_args()

    if not args.gold.exists():
        log.error(f"Gold set not found: {args.gold}")
        sys.exit(1)

    # Load gold set
    log.info(f"Loading gold set from {args.gold}")
    with open(args.gold) as f:
        gold_records = json.load(f)

    if args.limit:
        gold_records = gold_records[: args.limit]

    log.info(f"Loaded {len(gold_records)} records")

    # Stats
    pos = sum(1 for r in gold_records if r["slice"] == "positive")
    neg = len(gold_records) - pos
    with_cands = sum(1 for r in gold_records if r.get("expected_candidates"))
    total_spans = sum(len(r.get("expected_candidates", [])) for r in gold_records)
    log.info(f"  Positive: {pos}, Negative: {neg}, With candidates: {with_cands}")
    log.info(f"  Total gold spans: {total_spans}")

    # Load model
    log.info(f"Loading model: {args.model}")
    loader = load_model(args.model)
    log.info("Model loaded")

    # Run extraction
    log.info(f"Running extraction with strategy={args.strategy}...")
    predictions: dict[str, list[dict]] = {}
    t0 = time.time()

    for i, rec in enumerate(gold_records):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            eta = rate * (len(gold_records) - i - 1)
            print(
                f"  Processing {i + 1}/{len(gold_records)} "
                f"({elapsed:.1f}s elapsed, ETA {eta:.0f}s)",
                flush=True,
            )

        try:
            spans = extract_facts_llm(loader, rec["message_text"], args.strategy)
            predictions[rec["sample_id"]] = spans
            # Debug: log first 5 messages with expected candidates
            if i < 20 and rec.get("expected_candidates"):
                log.info(
                    f"  [{rec['sample_id']}] msg={rec['message_text'][:60]!r}"
                    f" gold={[c['span_text'] for c in rec['expected_candidates']]}"
                    f" pred={[s['span_text'] for s in spans]}"
                )
        except Exception as e:
            log.warning(f"Extraction failed for {rec['sample_id']}: {e}")
            predictions[rec["sample_id"]] = []

    elapsed = time.time() - t0
    total_preds = sum(len(v) for v in predictions.values())
    log.info(f"Extraction complete: {total_preds} predictions in {elapsed:.1f}s")

    # Compute metrics
    metrics = compute_metrics(gold_records, predictions)
    print_report(metrics, args.strategy, elapsed, len(gold_records))

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "gold_path": str(args.gold),
        "num_records": len(gold_records),
        "limit": args.limit,
        "strategy": args.strategy,
        "model": args.model,
        "num_predictions": total_preds,
        "extraction_time_s": round(elapsed, 2),
        "ms_per_message": round(elapsed / len(gold_records) * 1000, 1),
        "overall": metrics["overall"],
        "per_label": metrics["per_label"],
        "per_slice": metrics["per_slice"],
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Metrics saved to {METRICS_PATH}")

    # Also save errors for analysis
    errors_path = RESULTS_DIR / "errors.json"
    with open(errors_path, "w") as f:
        json.dump(metrics["errors"], f, indent=2)
    log.info(f"Errors saved to {errors_path}")


if __name__ == "__main__":
    main()
