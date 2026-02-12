#!/usr/bin/env python3
"""Extraction Bakeoff v2: Model Variants + Constrained Decoding + Automated Scoring.

Tests 4 LFM models x 5 prompt strategies x 2 decoding modes (constrained/unconstrained)
with automated P/R/F1 scoring against the frozen goldset.

Key improvements over v1:
- Automated scoring (no human review needed for metrics)
- Constrained decoding via Outlines (eliminates degenerate outputs)
- More model variants (Instruct, Base, Extract)
- Fewer, better prompt strategies (5 vs 24)

Usage:
    # Quick test: single model, single strategy, 10 messages
    uv run python scripts/extraction_bakeoff_v2.py \
        --models lfm-1.2b-extract --strategies schema_system --no-constrained --limit 10

    # Constrained only
    uv run python scripts/extraction_bakeoff_v2.py --constrained-only --limit 50

    # Full bakeoff (all models, all strategies, both modes, 100 messages)
    uv run python scripts/extraction_bakeoff_v2.py --full

    # With DSPy-optimized prompt
    uv run python scripts/extraction_bakeoff_v2.py --dspy-prompt results/dspy_optimized_prompt.json
"""

import argparse
import gc
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from eval_shared import DEFAULT_LABEL_ALIASES, spans_match

# ─── Constants ──────────────────────────────────────────────────────────────

GOLD_PATH = Path("training_data/gliner_goldset/candidate_gold_merged_r4.json")
OUTPUT_DIR = Path("results/extraction_bakeoff_v2")

# Label aliases: map LLM category names -> goldset span_labels
LLM_LABEL_ALIASES: dict[str, set[str]] = {
    **DEFAULT_LABEL_ALIASES,
    "location": {
        "current_location", "past_location", "future_location", "place", "hometown",
    },
    "person": {"friend_name", "partner_name", "person_name", "family_member"},
    "job": {"employer", "job_role", "job_title"},
    "school": {"school"},
    "health": {"allergy", "health_condition", "dietary"},
    "relationship": {"family_member", "partner_name"},
    "preference": {"food_like", "food_dislike", "food_item", "preference", "hobby"},
    "activity": {"activity", "hobby"},
}

# ─── Model Definitions ──────────────────────────────────────────────────────

MODELS: dict[str, dict] = {
    "lfm-1.2b-extract": {
        "path": "models/lfm2-1.2b-extract-mlx-4bit",
        "chat_template": True,
        "is_base": False,
        "description": "LFM2.5 1.2B Extract (4-bit)",
    },
    "lfm-1.2b-instruct": {
        "path": "LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit",
        "chat_template": True,
        "is_base": False,
        "description": "LFM2.5 1.2B Instruct (4-bit)",
    },
    "lfm-350m-extract": {
        "path": "models/lfm2-350m-extract-mlx-4bit",
        "chat_template": True,
        "is_base": False,
        "description": "LFM2.5 350M Extract (4-bit)",
    },
    "lfm-350m-base": {
        "path": "mlx-community/LFM2-350M-8bit",
        "chat_template": False,
        "is_base": True,
        "description": "LFM2 350M Base (8-bit)",
    },
}

# Strategies incompatible with base models (require chat template / system prompt)
# Strategies that require system prompts / chat templates (skip for base models)
BASE_INCOMPATIBLE_STRATEGIES = {
    "schema_system", "minimal_extract", "direct_extract",
    "few_shot_3", "negative_examples", "extractive_only", "constrained_categories",
}

# ─── Prompt Strategies ──────────────────────────────────────────────────────
# Each returns (system_prompt | None, user_prompt, parse_mode)


def strategy_schema_system(text: str) -> tuple[str | None, str, str]:
    """LiquidAI's recommended pattern for Extract models."""
    system = (
        'Return JSON with schema: {"has_facts": bool, '
        '"facts": [{"category": "...", "value": "..."}]}\n'
        "Categories: location, person, relationship, preference, job, school, health, activity.\n"
        "Only extract facts explicitly stated. Empty array if none."
    )
    user = text
    return system, user, "json"


def strategy_one_category_v2(text: str) -> tuple[str | None, str, str]:
    """v1 winner improved with extractive constraint."""
    user = f"""Message: "{text}"

For this message, answer each. Write ONLY words from the message or "none".
Location:
Person:
Preference:
Relationship:
Job or school:
Health:"""
    return None, user, "kv"


def strategy_completion_2shot(text: str) -> tuple[str | None, str, str]:
    """Completion-style for base models, 2-shot."""
    user = f"""Text: "My sister lives in Austin"
location: Austin
person: sister

Text: "lmaooo"
none

Text: "{text}"
"""
    return None, user, "completion"


def strategy_minimal_extract(text: str) -> tuple[str | None, str, str]:
    """Shortest possible, schema-driven."""
    system = (
        'Extract facts as JSON. null if none. '
        'Schema: {"facts": [{"category": str, "value": str}] | null}'
    )
    user = text
    return system, user, "json"


def strategy_gate_then_extract(text: str) -> tuple[str | None, str, str]:
    """Two-step: gate then extract."""
    user = f"""Message: "{text}"
Contains personal facts? (yes/no):
If yes, list each as category: value"""
    return None, user, "gate"


def strategy_v1_winner(text: str) -> tuple[str | None, str, str]:
    """Exact v1 winner (03_one_category) - fill-in-the-blank per category."""
    user = f"""Message: "{text}"

For this message, answer each question. Write "none" if not mentioned.
Location mentioned:
Person mentioned:
Preference or opinion:
Activity or event:
Relationship:
Job or school:"""
    return None, user, "kv"


def strategy_direct_extract(text: str) -> tuple[str | None, str, str]:
    """Just tell it to extract facts. Simple and direct."""
    system = (
        "You extract personal facts from text messages. "
        "Output each fact on its own line as 'category: value'. "
        "Categories: location, person, relationship, preference, job, school, health, activity. "
        "If no personal facts, output only 'none'. "
        "Only extract what is explicitly stated. Never make up facts."
    )
    user = f"""Message: "{text}"
Facts:"""
    return system, user, "kv"


def strategy_few_shot_3(text: str) -> tuple[str | None, str, str]:
    """3-shot examples (v1 top performer)."""
    system = (
        "You extract personal facts from text messages. "
        "Only extract what is explicitly stated. Never hallucinate."
    )
    user = f"""Extract personal facts from text messages.

Example 1:
Message: "I just moved to Austin last week"
Facts: location: Austin

Example 2:
Message: "haha yeah"
Facts: none

Example 3:
Message: "My sister works at Google"
Facts: relationship: sister; job: Google

Now extract facts from:
Message: "{text}"
Facts:"""
    return system, user, "kv"


def strategy_negative_examples(text: str) -> tuple[str | None, str, str]:
    """Show what NOT to extract (v1 top performer)."""
    system = (
        "You are a precise fact extractor. You ONLY extract personal facts "
        "explicitly stated in text. You NEVER generate or hallucinate."
    )
    user = f"""Extract personal facts from this message. Be precise.

DO extract: names, locations, jobs, schools, relationships, preferences, health info
DO NOT extract: greetings, opinions about weather, generic statements, emotions
DO NOT hallucinate facts not in the message.

Message: "{text}"
Facts (or "none"):"""
    return system, user, "kv"


def strategy_extractive_only(text: str) -> tuple[str | None, str, str]:
    """Emphasize extractive - copy from text, don't generate (v1 top performer)."""
    system = (
        "You are an extractive system. You may ONLY output words that appear "
        "in the input text. Never generate new words. Never hallucinate."
    )
    user = f"""Input text: "{text}"

Copy out any personal facts (names, places, jobs, preferences) using ONLY words from the text above.
Format: category: value (one per line). Write "none" if no facts.
Facts:"""
    return system, user, "kv"


def strategy_constrained_categories(text: str) -> tuple[str | None, str, str]:
    """Strict category list with examples (v1 top performer)."""
    system = (
        "Extract facts from messages. Only output facts in the listed categories. "
        "If no facts found, output 'none'. Never hallucinate."
    )
    user = f"""From the message below, extract ONLY these fact types:
- location: city/state/country they live in or are at
- person: people mentioned by name
- job: company or job title
- school: school they attend/attended
- relationship: family members (brother, sister, mom, dad)
- preference: food, music, hobby preferences
- health: allergies or health conditions

Message: "{text}"

Extracted (or "none"):"""
    return system, user, "kv"


STRATEGIES: dict[str, callable] = {
    "schema_system": strategy_schema_system,
    "one_category_v2": strategy_one_category_v2,
    "v1_winner": strategy_v1_winner,
    "direct_extract": strategy_direct_extract,
    "few_shot_3": strategy_few_shot_3,
    "negative_examples": strategy_negative_examples,
    "extractive_only": strategy_extractive_only,
    "constrained_categories": strategy_constrained_categories,
    "completion_2shot": strategy_completion_2shot,
    "minimal_extract": strategy_minimal_extract,
    "gate_then_extract": strategy_gate_then_extract,
}


# ─── Output Parsers ─────────────────────────────────────────────────────────


JUNK_VALUES = {
    "none", "n/a", "null", "not mentioned", "not stated", "not applicable",
    "no", "na", "unknown", "", "not specified", "unspecified", "not provided",
    "none mentioned", "not explicitly stated", "none stated",
    "not mentioned.", "not specified.", "none.",
}


def _is_junk_value(val: str) -> bool:
    """Check if a parsed value is junk (null, none, prompt echo, etc.)."""
    cleaned = val.lower().strip().strip('"').strip("'")
    if cleaned in JUNK_VALUES:
        return True
    # Filter prompt-echo patterns: values that contain description text from prompts
    prompt_echoes = {
        "city/state/country", "food, music, hobby", "brother, sister, mom, dad",
        "company or job title", "allergies or health",
        "not explicitly", "not provided", "not specified",
    }
    for echo in prompt_echoes:
        if echo in cleaned:
            return True
    return False


def parse_json_output(text: str) -> list[tuple[str, str]]:
    """Parse JSON output -> list of (category, value).

    Handles truncated JSON (common with max_tokens limits) by extracting
    individual fact objects via regex when full JSON parse fails.
    """
    text = text.strip()

    # First try: full JSON parse
    facts = _try_full_json_parse(text)
    if facts:
        return facts

    # Second try: extract individual {"category": "...", "value": "..."} objects
    # This handles truncated JSON where the outer object is incomplete
    facts = []
    for m in re.finditer(
        r'"category"\s*:\s*"([^"]+)"\s*,\s*"value"\s*:\s*"([^"]*)"',
        text,
    ):
        cat, val = m.group(1).lower().strip(), m.group(2).strip()
        if cat and val and not _is_junk_value(val):
            facts.append((cat, val))
    return facts


def _try_full_json_parse(text: str) -> list[tuple[str, str]] | None:
    """Try to parse complete JSON. Returns None if no valid JSON found."""
    # Try to find JSON object
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if not json_match:
        # Try array
        arr_match = re.search(r'\[.*\]', text, re.DOTALL)
        if arr_match:
            try:
                arr = json.loads(arr_match.group())
                return [
                    (item.get("category", "").lower().strip(), item.get("value", "").strip())
                    for item in arr
                    if isinstance(item, dict)
                    and item.get("category")
                    and item.get("value")
                    and not _is_junk_value(str(item.get("value", "")))
                ]
            except (json.JSONDecodeError, TypeError):
                pass
        return None

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    fact_list = data.get("facts", [])
    if fact_list is None:
        return []
    if not isinstance(fact_list, list):
        return None

    facts = []
    for item in fact_list:
        if isinstance(item, dict):
            cat = item.get("category", "")
            val = str(item.get("value", "")).strip()
            if cat and val and not _is_junk_value(val):
                facts.append((cat.lower().strip(), val))
    return facts


# Known category keys for flat-JSON parsing (Extract model outputs these as top-level keys)
KNOWN_CATEGORIES = {
    "location", "person", "preference", "relationship", "job", "school",
    "health", "activity", "job_or_school", "hobby", "hobbies", "family",
    "personal_info", "interests",
}

FLAT_KEY_MAP = {
    "job_or_school": "job",
    "hobby": "activity",
    "hobbies": "activity",
    "family": "relationship",
    "personal_info": "preference",
    "interests": "preference",
}


def _parse_flat_json(text: str) -> list[tuple[str, str]] | None:
    """Parse flat JSON like {"location": "Austin", "person": "brother"}.

    Extract models often output this format instead of nested {"facts": [...]}.
    Returns None if not flat-key JSON.
    """
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON object
        m = re.search(r'\{[^{]*\}', text, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None

    # Check if this looks like flat-key format (has known category keys)
    keys_lower = {k.lower() for k in data.keys()}
    if not keys_lower & KNOWN_CATEGORIES:
        return None

    facts = []
    for key, val in data.items():
        key_lower = key.lower().strip()
        if key_lower in ("message", "text", "has_facts", "facts"):
            continue  # Skip non-category keys
        if val is None or (isinstance(val, str) and _is_junk_value(val)):
            continue
        if isinstance(val, str) and val.strip():
            cat = FLAT_KEY_MAP.get(key_lower, key_lower)
            facts.append((cat, val.strip()))
        elif isinstance(val, list):
            # Handle arrays of strings
            for item in val:
                if isinstance(item, str) and item.strip() and not _is_junk_value(item):
                    cat = FLAT_KEY_MAP.get(key_lower, key_lower)
                    facts.append((cat, item.strip()))
    return facts


def parse_kv_output(text: str) -> list[tuple[str, str]]:
    """Parse key-value output (one_category_v2 / v1_winner style).

    Handles:
    - Standard KV: "Location: Austin"
    - v1 format: "Location mentioned: Austin"
    - Flat JSON: {"location": "Austin", "person": null}
    - Markdown bold: "**Brother** (personal relationship)"
    - Bullet lists: "- location: Austin"
    """
    # If it looks like JSON, try flat-key parse first
    if text.strip().startswith("{"):
        flat = _parse_flat_json(text)
        if flat is not None:
            return flat

    category_map = {
        "location": "location",
        "location mentioned": "location",
        "person": "person",
        "person mentioned": "person",
        "preference": "preference",
        "preference or opinion": "preference",
        "activity": "activity",
        "activity or event": "activity",
        "relationship": "relationship",
        "job or school": "job",
        "job": "job",
        "school": "school",
        "health": "health",
        "food": "preference",
        "hobby": "activity",
        "name": "person",
    }

    facts = []
    for line in text.strip().split("\n"):
        line = line.strip().lstrip("- •*")
        # Strip markdown bold
        line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
        # Skip non-fact lines (e.g. "Facts:" header)
        if not line or ":" not in line:
            continue

        # Handle semicolon-delimited facts: "location: Austin; relationship: brother"
        # First check if this is a header line like "Facts: location: Austin; ..."
        key, _, rest = line.partition(":")
        key_lower = key.strip().lower()
        if key_lower in ("facts", "extracted", "answer", "output", "result"):
            # The value after "Facts:" contains the actual facts
            line = rest.strip()
            if not line:
                continue

        # Split by semicolons for multi-fact lines
        parts = line.split(";") if ";" in line else [line]
        for part in parts:
            part = part.strip()
            if ":" not in part:
                continue
            k, _, v = part.partition(":")
            k = k.strip().lower()
            v = v.strip()
            if not v or _is_junk_value(v):
                continue
            cat = category_map.get(k, k)
            facts.append((cat, v))

    return facts


def parse_completion_output(text: str) -> list[tuple[str, str]]:
    """Parse completion-style output (base model).

    Falls back to flat-JSON parsing if the model outputs JSON.
    """
    text = text.strip()
    # JSON fallback
    if text.startswith("{"):
        flat = _parse_flat_json(text)
        if flat is not None:
            return flat

    if text.lower().startswith("none"):
        return []
    facts = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.lower() == "none":
            continue
        # Stop at next example prompt
        if line.startswith("Text:"):
            break
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()
            if value and not _is_junk_value(value):
                facts.append((key, value))
    return facts


def parse_gate_output(text: str) -> list[tuple[str, str]]:
    """Parse gate-then-extract output.

    Handles:
    - Standard: "yes\\n- location: Austin"
    - JSON fallback
    - Markdown prose: "**Brother** (personal relationship)"
    - Bullet: "- **Bakes** (activity)"
    """
    text = text.strip()
    # JSON fallback
    if text.startswith("{"):
        flat = _parse_flat_json(text)
        if flat is not None:
            return flat

    lines = text.split("\n")
    # Check if gated as "no"
    first_line = lines[0].lower().strip() if lines else ""
    if "no" in first_line and "yes" not in first_line:
        return []

    facts = []
    for line in lines[1:]:
        line = line.strip().lstrip("- •")
        # Strip markdown bold: **Brother** (relationship) -> Brother (relationship)
        line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
        if not line:
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()
            if value and not _is_junk_value(value):
                facts.append((key, value))
        else:
            # Handle "Brother (personal relationship)" format
            paren_match = re.match(r'([^(]+)\s*\(([^)]+)\)', line)
            if paren_match:
                value = paren_match.group(1).strip()
                cat = paren_match.group(2).strip().lower()
                # Map verbose descriptions to categories
                cat_map = {
                    "personal relationship": "relationship",
                    "activity": "activity",
                    "personal choice": "preference",
                    "location": "location",
                    "job": "job",
                    "school": "school",
                    "health": "health",
                }
                mapped = cat_map.get(cat, cat.split()[0] if cat else "")
                if value and mapped and not _is_junk_value(value):
                    facts.append((mapped, value))
    return facts


PARSERS = {
    "json": parse_json_output,
    "kv": parse_kv_output,
    "completion": parse_completion_output,
    "gate": parse_gate_output,
}


# ─── Model Loading & Generation ─────────────────────────────────────────────


def load_model(model_path: str):
    """Load MLX model, returns (model, tokenizer)."""
    import mlx.core as mx
    from mlx_lm import load

    mx.set_memory_limit(1 * 1024 * 1024 * 1024)  # 1GB
    mx.set_cache_limit(512 * 1024 * 1024)  # 512MB

    print(f"  Loading {model_path}...", flush=True)
    t0 = time.time()
    model, tokenizer = load(model_path)
    print(f"  Loaded in {time.time() - t0:.1f}s", flush=True)
    return model, tokenizer


def unload_model(model, tokenizer):
    """Unload model and free memory."""
    import mlx.core as mx

    del model
    del tokenizer
    gc.collect()
    mx.clear_cache()


def generate_unconstrained(
    model, tokenizer, system_prompt: str | None, user_prompt: str,
    is_base: bool = False, max_tokens: int = 200,
) -> tuple[str, float]:
    """Generate with standard mlx_lm.generate(). Returns (text, time_ms)."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_repetition_penalty, make_sampler

    if is_base:
        # Raw completion, no chat template
        prompt = user_prompt
    else:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = user_prompt

    sampler = make_sampler(temp=0.0)
    rep_penalty = make_repetition_penalty(1.1)

    t0 = time.time()
    response = generate(
        model, tokenizer, prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=[rep_penalty],
    )
    elapsed_ms = (time.time() - t0) * 1000

    # Truncate at stop sequences for base model
    if is_base:
        for stop in ["\n\n", "Text:"]:
            idx = response.find(stop)
            if idx >= 0:
                response = response[:idx]

    return response, elapsed_ms


def generate_constrained(
    model, tokenizer, system_prompt: str | None, user_prompt: str,
    is_base: bool = False, max_tokens: int = 200,
) -> tuple[str, float]:
    """Generate with Outlines constrained decoding. Returns (text, time_ms)."""
    try:
        from typing import Literal

        import outlines
        from pydantic import BaseModel as PydanticBaseModel

        class ExtractedFact(PydanticBaseModel):
            category: Literal[
                "location", "person", "relationship", "preference",
                "job", "school", "health", "activity"
            ]
            value: str

        class ExtractionResult(PydanticBaseModel):
            has_facts: bool
            facts: list[ExtractedFact]

    except ImportError as e:
        raise RuntimeError(
            f"Outlines not installed. Install with: uv pip install 'outlines[mlxlm]' -- {e}"
        ) from e

    if is_base:
        prompt = user_prompt
    else:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = user_prompt

    t0 = time.time()
    try:
        outlines_model = outlines.models.mlxlm(model, tokenizer)
        generator = outlines.generate.json(outlines_model, ExtractionResult)
        result = generator(prompt, max_tokens=max_tokens)
        elapsed_ms = (time.time() - t0) * 1000
        # Convert Pydantic model back to JSON string for uniform handling
        response = result.model_dump_json()
    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000
        raise RuntimeError(f"Constrained generation failed: {e}") from e

    return response, elapsed_ms


# ─── Scoring ─────────────────────────────────────────────────────────────────


@dataclass
class ComboMetrics:
    """Metrics for one model+strategy+mode combination."""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    total_ms: float = 0.0
    n_messages: int = 0
    n_errors: int = 0
    n_parse_failures: int = 0

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

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.n_messages if self.n_messages > 0 else 0.0


@dataclass
class ExtractionResult:
    """Result for a single message in the bakeoff."""
    model: str
    strategy: str
    constrained: bool
    message_id: int
    message_text: str
    response: str
    parsed_facts: list[tuple[str, str]]
    time_ms: float
    error: str | None = None
    # Scoring fields (filled after matching)
    tp: int = 0
    fp: int = 0
    fn: int = 0


def score_predictions(
    parsed_facts: list[tuple[str, str]],
    gold_candidates: list[dict],
) -> tuple[int, int, int]:
    """Score parsed facts against gold candidates.

    Returns (tp, fp, fn).
    """
    gold_matched = [False] * len(gold_candidates)
    pred_matched = [False] * len(parsed_facts)

    for gi, gold_cand in enumerate(gold_candidates):
        for pi, (pred_cat, pred_val) in enumerate(parsed_facts):
            if pred_matched[pi]:
                continue
            if spans_match(
                pred_val, pred_cat,
                gold_cand.get("span_text", ""),
                gold_cand.get("span_label", ""),
                label_aliases=LLM_LABEL_ALIASES,
            ):
                gold_matched[gi] = True
                pred_matched[pi] = True
                break

    tp = sum(gold_matched)
    fn = len(gold_candidates) - tp
    fp = len(parsed_facts) - sum(pred_matched)
    return tp, fp, fn


# ─── Main Bakeoff Runner ────────────────────────────────────────────────────


def load_goldset(gold_path: Path, limit: int | None, full: bool) -> list[dict]:
    """Load and subsample the goldset."""
    with open(gold_path) as f:
        records = json.load(f)

    if full:
        print(f"Using FULL goldset: {len(records)} records", flush=True)
        return records

    # Default: balanced 100-message subset (50 positive, 50 negative)
    n = limit or 100
    half = n // 2
    positive = [r for r in records if r.get("expected_candidates")]
    negative = [r for r in records if not r.get("expected_candidates")]
    subset = positive[:half] + negative[:half]
    print(
        f"Using balanced subset: {len(subset)} records "
        f"({min(half, len(positive))} positive, {min(half, len(negative))} negative)",
        flush=True,
    )
    return subset


def run_bakeoff(
    model_names: list[str],
    strategy_names: list[str],
    records: list[dict],
    run_constrained: bool = True,
    run_unconstrained: bool = True,
    max_tokens: int = 200,
    dspy_prompt: dict | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> list[dict]:
    """Run the full bakeoff. Returns list of per-combo result dicts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add DSPy strategy if provided
    all_strategies = dict(STRATEGIES)
    if dspy_prompt:
        def strategy_dspy(text: str) -> tuple[str | None, str, str]:
            system = dspy_prompt.get("system_prompt")
            user_template = dspy_prompt.get("user_prompt_template", "{text}")
            user = user_template.replace("{text}", text)
            parse_mode = dspy_prompt.get("parse_mode", "json")
            return system, user, parse_mode
        all_strategies["dspy_optimized"] = strategy_dspy
        if "dspy_optimized" not in strategy_names:
            strategy_names = list(strategy_names) + ["dspy_optimized"]

    # Calculate combos
    modes = []
    if run_unconstrained:
        modes.append(False)
    if run_constrained:
        modes.append(True)

    combos = []
    for model_name in model_names:
        model_info = MODELS[model_name]
        for strat_name in strategy_names:
            if strat_name not in all_strategies:
                continue
            # Skip incompatible combos
            if model_info["is_base"] and strat_name in BASE_INCOMPATIBLE_STRATEGIES:
                continue
            for constrained in modes:
                combos.append((model_name, strat_name, constrained))

    total_runs = len(combos) * len(records)
    print(f"\n{'=' * 70}", flush=True)
    print("EXTRACTION BAKEOFF V2", flush=True)
    print(f"  Models: {len(model_names)} ({', '.join(model_names)})", flush=True)
    print(f"  Strategies: {len(strategy_names)}", flush=True)
    mode_str = (
        'constrained + unconstrained' if len(modes) == 2
        else ('constrained' if modes[0] else 'unconstrained')
    )
    print(f"  Modes: {mode_str}", flush=True)
    print(f"  Messages: {len(records)}", flush=True)
    print(f"  Total combos: {len(combos)}", flush=True)
    print(f"  Total runs: {total_runs}", flush=True)
    print(f"  Output: {output_dir}/", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    all_results: list[ExtractionResult] = []
    combo_metrics: dict[tuple[str, str, bool], ComboMetrics] = {}
    run_idx = 0

    current_model_name = None
    model = None
    tokenizer = None

    for combo_idx, (model_name, strat_name, constrained) in enumerate(combos):
        model_info = MODELS[model_name]
        mode_str = "constrained" if constrained else "unconstrained"
        combo_key = (model_name, strat_name, constrained)
        metrics = ComboMetrics()
        combo_metrics[combo_key] = metrics

        # Load model if needed
        if model_name != current_model_name:
            if model is not None:
                print(f"  Unloading {current_model_name}...", flush=True)
                unload_model(model, tokenizer)
            try:
                model, tokenizer = load_model(model_info["path"])
                current_model_name = model_name
            except Exception as e:
                print(f"  FAILED to load {model_name}: {e}", flush=True)
                metrics.n_errors = len(records)
                metrics.n_messages = len(records)
                current_model_name = None
                model = None
                tokenizer = None
                run_idx += len(records)
                continue

        print(
            f"\n  [{combo_idx + 1}/{len(combos)}] {model_name} / {strat_name} / {mode_str}",
            flush=True,
        )

        strategy_fn = all_strategies[strat_name]
        is_base = model_info["is_base"]

        for i, rec in enumerate(records):
            run_idx += 1
            msg_text = rec["message_text"]
            msg_id = rec["message_id"]
            gold_cands = rec.get("expected_candidates", [])

            if (i + 1) % 25 == 0 or i == 0:
                print(
                    f"    Message {i + 1}/{len(records)} "
                    f"(run {run_idx}/{total_runs})",
                    flush=True,
                )

            try:
                system_prompt, user_prompt, parse_mode = strategy_fn(msg_text)

                if constrained:
                    response, time_ms = generate_constrained(
                        model, tokenizer, system_prompt, user_prompt,
                        is_base=is_base, max_tokens=max_tokens,
                    )
                    # Constrained output is always valid JSON
                    parsed = parse_json_output(response)
                else:
                    response, time_ms = generate_unconstrained(
                        model, tokenizer, system_prompt, user_prompt,
                        is_base=is_base, max_tokens=max_tokens,
                    )
                    parser = PARSERS.get(parse_mode, parse_json_output)
                    parsed = parser(response)

                # Score
                tp, fp, fn = score_predictions(parsed, gold_cands)

                result = ExtractionResult(
                    model=model_name, strategy=strat_name, constrained=constrained,
                    message_id=msg_id, message_text=msg_text,
                    response=response, parsed_facts=parsed,
                    time_ms=time_ms, tp=tp, fp=fp, fn=fn,
                )
                metrics.tp += tp
                metrics.fp += fp
                metrics.fn += fn
                metrics.total_ms += time_ms
                metrics.n_messages += 1
                if not parsed and gold_cands:
                    metrics.n_parse_failures += 1

            except Exception as e:
                result = ExtractionResult(
                    model=model_name, strategy=strat_name, constrained=constrained,
                    message_id=msg_id, message_text=msg_text,
                    response="", parsed_facts=[], time_ms=0,
                    error=str(e),
                )
                metrics.n_errors += 1
                metrics.n_messages += 1
                # Count gold as FN for errors
                metrics.fn += len(gold_cands)

            all_results.append(result)

        # Print combo summary
        p = metrics.precision
        r = metrics.recall
        f1 = metrics.f1
        print(
            f"    -> P={p:.3f} R={r:.3f} F1={f1:.3f} "
            f"avg={metrics.avg_ms:.0f}ms errors={metrics.n_errors}",
            flush=True,
        )

        # Save incrementally
        _save_incremental(all_results, combo_metrics, output_dir)

    # Unload final model
    if model is not None:
        print(f"  Unloading {current_model_name}...", flush=True)
        unload_model(model, tokenizer)

    # Final save
    _save_incremental(all_results, combo_metrics, output_dir)
    _write_review(all_results, combo_metrics, output_dir)

    return _metrics_to_dicts(combo_metrics)


def _save_incremental(
    results: list[ExtractionResult],
    combo_metrics: dict[tuple, ComboMetrics],
    output_dir: Path,
) -> None:
    """Save results and metrics incrementally. Uses timestamped files to avoid overwriting."""
    # Save raw results (both latest + timestamped copy)
    results_path = output_dir / "all_results.json"
    serializable = []
    for r in results:
        d = {
            "model": r.model, "strategy": r.strategy,
            "constrained": r.constrained, "message_id": r.message_id,
            "message_text": r.message_text, "response": r.response,
            "parsed_facts": r.parsed_facts, "time_ms": r.time_ms,
            "error": r.error, "tp": r.tp, "fp": r.fp, "fn": r.fn,
        }
        serializable.append(d)
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)

    # Save timestamped copy (never overwritten)
    ts = time.strftime("%Y%m%d_%H%M%S")
    ts_path = output_dir / f"results_{ts}.json"
    if not ts_path.exists():
        with open(ts_path, "w") as f:
            json.dump(serializable, f, indent=2)

    # Save metrics summary
    _save_metrics_summary(combo_metrics, output_dir)


def _save_metrics_summary(
    combo_metrics: dict[tuple, ComboMetrics], output_dir: Path
) -> None:
    """Save metrics summary JSON."""
    metrics_path = output_dir / "metrics_summary.json"
    summary = {}
    for (model, strat, constrained), m in combo_metrics.items():
        key = f"{model}/{strat}/{'constrained' if constrained else 'unconstrained'}"
        summary[key] = {
            "precision": round(m.precision, 4),
            "recall": round(m.recall, 4),
            "f1": round(m.f1, 4),
            "avg_ms": round(m.avg_ms, 1),
            "tp": m.tp, "fp": m.fp, "fn": m.fn,
            "n_messages": m.n_messages,
            "n_errors": m.n_errors,
            "n_parse_failures": m.n_parse_failures,
        }
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)


def _metrics_to_dicts(
    combo_metrics: dict[tuple, ComboMetrics]
) -> list[dict]:
    """Convert combo metrics to sorted list of dicts for summary."""
    rows = []
    for (model, strat, constrained), m in sorted(combo_metrics.items()):
        rows.append({
            "model": model,
            "strategy": strat,
            "constrained": constrained,
            "precision": round(m.precision, 4),
            "recall": round(m.recall, 4),
            "f1": round(m.f1, 4),
            "avg_ms": round(m.avg_ms, 1),
            "n_errors": m.n_errors,
        })
    return rows


def _write_review(
    results: list[ExtractionResult],
    combo_metrics: dict[tuple, ComboMetrics],
    output_dir: Path,
) -> None:
    """Write human-readable review markdown."""
    review_path = output_dir / "review.md"
    with open(review_path, "w") as f:
        f.write("# Extraction Bakeoff v2 Results\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Summary table
        f.write("## Summary\n\n")
        f.write("| Model | Strategy | Constrained | P | R | F1 | Avg ms | Errors |\n")
        f.write("|-------|----------|-------------|-------|-------|-------|--------|--------|\n")

        for (model, strat, constrained), m in sorted(combo_metrics.items()):
            mode_str = "yes" if constrained else "no"
            f.write(
                f"| {model} | {strat} | {mode_str} | "
                f"{m.precision:.3f} | {m.recall:.3f} | {m.f1:.3f} | "
                f"{m.avg_ms:.0f} | {m.n_errors} |\n"
            )

        # Best combos
        f.write("\n## Best Combinations\n\n")
        valid = [(k, m) for k, m in combo_metrics.items() if m.n_messages > 0]
        if valid:
            best_f1 = max(valid, key=lambda x: x[1].f1)
            best_p = max(valid, key=lambda x: x[1].precision)
            best_r = max(valid, key=lambda x: x[1].recall)
            k, m = best_f1
            c_str = 'constrained' if k[2] else 'unconstrained'
            f.write(
                f"- **Best F1**: {k[0]}/{k[1]}/{c_str}"
                f" = {m.f1:.3f}\n"
            )
            k, m = best_p
            c_str = 'constrained' if k[2] else 'unconstrained'
            f.write(
                f"- **Best Precision**: {k[0]}/{k[1]}/{c_str}"
                f" = {m.precision:.3f}\n"
            )
            k, m = best_r
            c_str = 'constrained' if k[2] else 'unconstrained'
            f.write(
                f"- **Best Recall**: {k[0]}/{k[1]}/{c_str}"
                f" = {m.recall:.3f}\n"
            )

        # Detailed per-combo results (sample)
        f.write("\n## Sample Outputs (first 5 per combo)\n\n")
        grouped: dict[tuple, list[ExtractionResult]] = defaultdict(list)
        for r in results:
            grouped[(r.model, r.strategy, r.constrained)].append(r)

        for key, group in sorted(grouped.items()):
            model, strat, constrained = key
            mode_str = "constrained" if constrained else "unconstrained"
            f.write(f"\n### {model} / {strat} / {mode_str}\n\n")
            for r in group[:5]:
                "+" if any(
                    rec.get("expected_candidates")
                    for rec in [{}]  # placeholder
                ) else ""
                msg_preview = r.message_text[:60].replace("\n", " ")
                f.write(f"**msg {r.message_id}**: {msg_preview}...\n")
                if r.error:
                    f.write(f"- ERROR: {r.error}\n\n")
                else:
                    f.write(f"- Parsed: {r.parsed_facts}\n")
                    f.write(f"- TP={r.tp} FP={r.fp} FN={r.fn} ({r.time_ms:.0f}ms)\n")
                    f.write(f"- Raw: `{r.response[:200]}`\n\n")

    # Print summary table
    print(f"\n{'=' * 90}", flush=True)
    print("BAKEOFF V2 RESULTS", flush=True)
    print(f"{'=' * 90}", flush=True)
    print(
        f"{'Model':<20} {'Strategy':<20} {'Constr':>7} "
        f"{'P':>7} {'R':>7} {'F1':>7} {'ms':>7} {'Err':>5}",
        flush=True,
    )
    print("-" * 90, flush=True)

    for (model, strat, constrained), m in sorted(combo_metrics.items()):
        c_str = "yes" if constrained else "no"
        print(
            f"{model:<20} {strat:<20} {c_str:>7} "
            f"{m.precision:>7.3f} {m.recall:>7.3f} {m.f1:>7.3f} "
            f"{m.avg_ms:>6.0f}ms {m.n_errors:>4}",
            flush=True,
        )

    print(f"\nResults: {output_dir}/review.md", flush=True)
    print(f"Metrics: {output_dir}/metrics_summary.json", flush=True)
    print(f"Raw:     {output_dir}/all_results.json", flush=True)


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Extraction Bakeoff v2")
    parser.add_argument(
        "--models", type=str, default=None,
        help=f"Comma-separated model names. Available: {','.join(MODELS.keys())}",
    )
    parser.add_argument(
        "--strategies", type=str, default=None,
        help=f"Comma-separated strategy names. Available: {','.join(STRATEGIES.keys())}",
    )
    parser.add_argument(
        "--gold", type=str, default=str(GOLD_PATH),
        help="Path to goldset JSON",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit to N messages (balanced)")
    parser.add_argument("--full", action="store_true", help="Use full goldset (796 records)")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens per generation")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    parser.add_argument(
        "--constrained-only", action="store_true",
        help="Only run constrained mode",
    )
    parser.add_argument(
        "--no-constrained", action="store_true",
        help="Skip constrained mode (unconstrained only)",
    )
    parser.add_argument(
        "--dspy-prompt", type=str, default=None,
        help="Path to DSPy-optimized prompt JSON",
    )
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-strategies", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable models:")
        for name, info in MODELS.items():
            print(f"  {name:<25} {info['description']}")
        return

    if args.list_strategies:
        print("\nAvailable strategies:")
        for name, fn in STRATEGIES.items():
            print(f"  {name:<25} {fn.__doc__}")
        return

    # Load goldset
    gold_path = Path(args.gold)
    if not gold_path.exists():
        print(f"Goldset not found: {gold_path}", flush=True)
        sys.exit(1)
    records = load_goldset(gold_path, args.limit, args.full)

    # Select models
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
        for m in model_names:
            if m not in MODELS:
                print(f"Unknown model: {m}. Use --list-models.", flush=True)
                sys.exit(1)
    else:
        model_names = list(MODELS.keys())

    # Select strategies
    if args.strategies:
        strategy_names = [s.strip() for s in args.strategies.split(",")]
        for s in strategy_names:
            if s not in STRATEGIES:
                print(f"Unknown strategy: {s}. Use --list-strategies.", flush=True)
                sys.exit(1)
    else:
        strategy_names = list(STRATEGIES.keys())

    # Decode modes
    run_constrained = not args.no_constrained
    run_unconstrained = not args.constrained_only

    # DSPy prompt
    dspy_prompt = None
    if args.dspy_prompt:
        dspy_path = Path(args.dspy_prompt)
        if not dspy_path.exists():
            print(f"DSPy prompt not found: {dspy_path}", flush=True)
            sys.exit(1)
        with open(dspy_path) as f:
            dspy_prompt = json.load(f)
        print(f"Loaded DSPy prompt from {dspy_path}", flush=True)

    run_bakeoff(
        model_names=model_names,
        strategy_names=strategy_names,
        records=records,
        run_constrained=run_constrained,
        run_unconstrained=run_unconstrained,
        max_tokens=args.max_tokens,
        dspy_prompt=dspy_prompt,
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
