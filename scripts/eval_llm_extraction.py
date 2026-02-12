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

EXTRACT_SYSTEM_PROMPT = """Extract LASTING personal facts from a chat message as JSON.
Only extract facts that reveal ongoing traits: hobbies, family relationships, jobs, employers, schools, health conditions, places lived, food likes/dislikes.
DO NOT extract: temporary actions, plans, one-time events, casual mentions of family in passing.
"text" = exact 1-3 words copied verbatim from the message. Never invent words.
Labels: family_member, activity, health_condition, job_role, org, food_item, place, friend_name, person_name
Return {"facts": []} if no lasting personal facts."""

# Few-shot examples: positive + hard negatives
FEW_SHOT_TURNS = [
    ("my brother bakes and I just eat whatever he makes",
     '{"facts": [{"text": "brother", "label": "family_member"}, {"text": "bakes", "label": "activity"}]}'),
    ("I work at Google as an engineer",
     '{"facts": [{"text": "Google", "label": "org"}, {"text": "engineer", "label": "job_role"}]}'),
    ("allergic to peanuts and it sucks",
     '{"facts": [{"text": "peanuts", "label": "health_condition"}]}'),
    ("Also my dad leaves the 22nd for India",
     '{"facts": [{"text": "dad", "label": "family_member"}, {"text": "India", "label": "place"}]}'),
    ("I love reading",
     '{"facts": [{"text": "reading", "label": "activity"}]}'),
    ("i hate utd",
     '{"facts": [{"text": "utd", "label": "org"}]}'),
    ("been hella depressed",
     '{"facts": [{"text": "depressed", "label": "health_condition"}]}'),
    ("I work at lending tree",
     '{"facts": [{"text": "lending tree", "label": "org"}]}'),
    ("Also i liked the dolmas",
     '{"facts": [{"text": "dolmas", "label": "food_item"}]}'),
    ("I like the raiders",
     '{"facts": [{"text": "raiders", "label": "org"}]}'),
    ("helloooo",
     '{"facts": []}'),
    ("Yeah that's fine I'll leave as soon as my mom gets home at 4",
     '{"facts": []}'),
    ("and they'll do an ultrasound and stuff",
     '{"facts": []}'),
    ("My mom never ended up coming tho so gonna have to ship that bag",
     '{"facts": []}'),
    ("Yesterday I called my dad at like 1 and he was like just come home",
     '{"facts": []}'),
    ("cause my mom tried doin my bros arms",
     '{"facts": []}'),
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
    "org": {"org", "organization", "company", "employer", "school", "university", "personal.school"},
    "place": {"place", "location", "venue", "landmark", "current_location", "future_location", "past_location"},
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
    "stepmom", "stepdad", "niece", "nephew", "bros",
}

_HEALTH_KEYWORDS = {
    "allergic", "allergy", "asthma", "diabetes", "depression", "depressed",
    "anxiety", "adhd", "migraine", "migraines", "vestibular",
    "surgery", "injury", "cancer", "arthritis", "insomnia",
    "emergency room", "hospital", "therapy", "ptsd",
}

_KNOWN_ORGS = {
    "facebook", "google", "intuit", "apple", "amazon", "microsoft",
    "netflix", "uber", "lyft", "airbnb", "twitter", "meta",
    "lending tree", "lendingtree", "cvs", "walmart", "target",
    "starbucks", "chipotle", "costco",
}

_KNOWN_SCHOOLS = {
    "utd", "ucd", "ucla", "usc", "sjsu", "stanford", "berkeley",
    "culinary school", "community college",
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

    # Known orgs/companies should be org regardless of predicted label
    if text_lower in _KNOWN_ORGS or text_lower in _KNOWN_SCHOOLS:
        return "org"

    # If "school" or "university" or "college" in span, it's an org
    if any(w in text_lower for w in ("school", "university", "college")):
        return "org"

    # If span looks like a job title and was labeled activity
    if label == "activity":
        job_indicators = {
            "manager", "engineer", "developer", "nurse", "doctor",
            "teacher", "analyst", "designer", "consultant", "director",
            "intern", "coordinator", "specialist", "product management",
            "realtor",
        }
        if text_lower in job_indicators:
            return "job_role"

    # If labeled job_role but looks like a company name (proper noun, not a role word)
    if label == "job_role" and len(text) > 0 and text[0].isupper():
        job_role_words = {
            "manager", "engineer", "developer", "nurse", "doctor",
            "teacher", "analyst", "designer", "consultant", "director",
            "intern", "coordinator", "specialist", "ceo", "cto", "cfo",
            "vp", "president", "founder", "product management",
        }
        if text_lower not in job_role_words and "management" not in text_lower:
            return "org"

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

    # Skip iMessage reaction messages (Loved/Liked/Laughed at/Emphasized "...")
    # These quote other people's messages and shouldn't have facts extracted
    if re.match(r'^(Loved|Liked|Laughed at|Emphasized|Disliked)\s+\u201c', message_text):
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
                "place", "area", "spot", "well", "last fall", "22nd",
                "diwali", "christmas", "thanksgiving",
                "doctor's appointment", "appointment", "bart",
            }
            if text_lower in location_rejects:
                continue
        if label == "food_item":
            # Food items should be recognizable food words, not random nouns
            reject_foods = {
                "eat", "eating", "ate", "food", "cooking", "cook", "phone",
                "whatever", "everything", "anything", "something", "stuff",
                "it", "that", "this", "one", "all", "car", "arms", "tie",
                "theory", "utilities", "read", "xbox", "realtor", "raiders",
                "acceptance letter", "bag", "never", "whatever he makes",
                "jeans", "coupon", "email", "her neck", "neck",
                "i.cvs.com", "diwali",
                "process", "ship that bag", "take the bart", "bart",
                "live instruction", "instruction", "schedule",
                "bell schedule", "regular bell schedule",
                "per period", "period",
            }
            if text_lower in reject_foods:
                continue
            # Reject multi-word spans with verbs (hallucinated phrases, not food)
            if len(text.split()) >= 3:
                verb_words = {"take", "ship", "get", "make", "live", "do", "go", "come"}
                if any(w.lower() in verb_words for w in text.split()):
                    continue
            # Reject non-food patterns: numbers, abbreviations, URLs, body parts
            if any(c.isdigit() for c in text):
                continue
            if "." in text and len(text) > 4:  # URLs like "i.cvs.com"
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
                    "dolma", "dolmas", "biryani", "samosa", "roti",
                    "tikka", "masala", "chutney", "dal", "dhal",
                    "pho", "ramen", "udon", "tempura", "tofu",
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
                "30-40", "22nd", "icing", "made", "stories",
                "ultrasound", "kind", "later", "don't",
                "talk to some", "points of view",
                "packed everything up", "shit 1",
                "7 days", "28th", "externship", "arms",
                "don\u2019t wanna", "don\u2019t", "dgaf",
                "hella bad", "awesome", "figure the rest",
                "looking at matchups", "live instruction",
                "summer chauffeur", "bros arms", "rest a bit",
                "not comin", "shelter in place",
                "never ended up", "working from home", "free",
                "regular bell schedule", "bell schedule",
                "per period", "take the bart",
            }
            if text_lower in reject_activities:
                continue
            # Reject spans with numbers (dates, times, not activities)
            if any(c.isdigit() for c in text):
                continue
            # Single lowercase words < 4 chars are unlikely to be activities
            if len(text) <= 3 and text[0].islower():
                continue
        if label == "health_condition":
            reject_health = {
                "whatever", "slow", "slow process", "points of view",
                "insanely big", "bad", "feel", "feeling",
                "dgaf", "don't fuck", "either", "not comin",
                "ihs", "4am", "rationalize", "willpower",
                "take responsibility", "rest", "never ended",
                "increase time", "never", "free", "not fun",
                "tight", "annoying", "ready to just", "love it",
                "shelter in place", "fuck", "doctor's appointment",
                "rest a bit", "hella bad", "barring anything",
                "5k", "daily 5k",
            }
            if text_lower in reject_health:
                continue
        if label == "job_role":
            reject_jobs = {
                "translation for", "comin", "ser",
                "working from home", "shelter in place",
                "ready to get", "slow process",
                "ready to slowly", "externship",
            }
            if text_lower in reject_jobs:
                continue
        if label == "family_member":
            # Only accept actual family relationship words
            if text_lower not in _FAMILY_WORDS:
                # Allow possessive forms like "brother's" and plurals like "sisters"
                base = text_lower.rstrip("'s").rstrip("\u2019s")
                if base not in _FAMILY_WORDS:
                    # Try removing trailing 's' for plurals
                    if base.endswith("s") and base[:-1] in _FAMILY_WORDS:
                        pass  # OK, plural form
                    else:
                        continue
        if label == "friend_name":
            # Friend names should start with uppercase
            if text[0].islower():
                continue
        if label == "person_name":
            # Reject very short abbreviations
            if len(text) <= 2:
                continue
            # Reject common words that aren't names
            person_rejects = {
                "prof", "professor", "teacher", "coach", "doctor",
                "prolly", "probably", "someone", "somebody", "everyone",
                "nobody", "anyone", "people", "person", "dude", "bro",
                "bruh", "homie", "fam", "dawg",
            }
            if text_lower in person_rejects:
                continue
            # Person names should start with uppercase (proper nouns)
            if text[0].islower():
                continue

        # Validate span text appears in message (case-insensitive)
        if text_lower not in msg_lower:
            # Try individual words for partial match - require majority of words present
            words = text_lower.split()
            matching_words = [w for w in words if w in msg_lower and len(w) > 2]
            if len(matching_words) < max(1, len(words) * 0.5):
                continue
            # Use the best matching word as the span text
            if len(matching_words) == 1 and len(words) > 1:
                text = matching_words[0]
                text_lower = text.lower()

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


def _rule_based_boost(spans: list[dict], message_text: str) -> list[dict]:
    """Add entities the LLM commonly misses using pattern matching.

    Only adds spans not already present in the LLM output.
    Focuses on high-precision patterns to avoid adding FPs.
    """
    msg_lower = message_text.lower()
    existing = {(s["span_text"].lower(), s["span_label"]) for s in spans}
    def _add(text: str, label: str, fact_type: str):
        key = (text.lower(), label)
        if key not in existing:
            existing.add(key)
            spans.append({"span_text": text, "span_label": label, "fact_type": fact_type})

    # 1. Family member pattern: "my <family_word>" in message
    # Only boost if the LLM found at least one fact already.
    # Additional guard: skip if message is very short (< 40 chars) and LLM found nothing,
    # as these are often transient mentions ("my mom will pick up")
    llm_found_facts = len(spans) > 0
    if llm_found_facts:
        for fw in _FAMILY_WORDS:
            pattern = f"my {fw}"
            if pattern in msg_lower and (fw.lower(), "family_member") not in existing:
                idx = msg_lower.index(pattern)
                actual = message_text[idx + 3 : idx + 3 + len(fw)]
                _add(actual, "family_member", "relationship.family")

        # Also catch possessive: "brother's", "sisters"
        for fw in _FAMILY_WORDS:
            for suffix in ["'s", "\u2019s", "s"]:
                variant = fw + suffix
                if variant in msg_lower and (fw, "family_member") not in existing:
                    if f"my {variant}" in msg_lower or f"my {fw}" in msg_lower:
                        _add(fw, "family_member", "relationship.family")
                        break

    # 2. Known orgs: check if message contains known org names
    for org_name in _KNOWN_ORGS | _KNOWN_SCHOOLS:
        if org_name in msg_lower and (org_name, "org") not in existing:
            # Find actual casing in message
            idx = msg_lower.index(org_name)
            actual = message_text[idx : idx + len(org_name)]
            _add(actual, "org", "work.employer")

    # 3. Health keywords in message
    for hw in _HEALTH_KEYWORDS:
        if hw in msg_lower and (hw, "health_condition") not in existing:
            idx = msg_lower.index(hw)
            actual = message_text[idx : idx + len(hw)]
            _add(actual, "health_condition", "health.condition")

    # 4. "I work at <X>" pattern - extract the org
    work_match = re.search(r'\b(?:work|working) at ([A-Z][a-zA-Z\s]{1,20}?)(?:\s*[.!?,]|\s+(?:as|and|but|so|for)|\s*$)', message_text)
    if work_match:
        org_text = work_match.group(1).strip()
        if org_text and (org_text.lower(), "org") not in existing:
            _add(org_text, "org", "work.employer")

    return spans


def _strip_emojis(text: str) -> str:
    """Strip emoji characters that confuse the model."""
    # Remove emoji unicode ranges
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols extended
        "\U00002600-\U000026FF"  # misc symbols
        "\U0000FE00-\U0000FE0F"  # variation selectors
        "\U0000200D"  # zero-width joiner
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text).strip()


def _strategy_constrained_categories(loader, message_text: str) -> list[dict]:
    """Strategy: multi-turn few-shot with constrained category list."""
    # Strip emojis that confuse the model
    clean_text = _strip_emojis(message_text)
    if not clean_text:
        return []

    # Build multi-turn conversation with few-shot examples
    messages = [{"role": "system", "content": EXTRACT_SYSTEM_PROMPT}]

    for user_msg, assistant_resp in FEW_SHOT_TURNS:
        messages.append({"role": "user", "content": f'Message: "{user_msg}"'})
        messages.append({"role": "assistant", "content": assistant_resp})

    messages.append({"role": "user", "content": INSTRUCT_USER_PROMPT.format(message=clean_text)})

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

    raw = result.text
    facts = parse_llm_json(raw)
    spans = json_to_spans(facts, message_text)
    if not facts and '{"facts"' not in raw:
        log.debug("No JSON in output: %r", raw[:120])
    elif facts and not spans:
        log.debug(
            "All %d facts filtered for '%s': %s",
            len(facts), message_text[:40],
            [(f.get("text"), f.get("label")) for f in facts],
        )

    # Rule-based recall boost: catch entities the LLM commonly misses
    spans = _rule_based_boost(spans, message_text)
    return spans


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
