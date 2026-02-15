"""Shared regex patterns and word sets for NLP feature extraction.

Centralizes patterns used across multiple modules (category_features,
response_mobilization, etc.) to avoid duplication and drift.
"""

from __future__ import annotations

import re

# === EMOJI DETECTION ===
EMOJI_RE = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
    r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]"
)

# === ABBREVIATIONS / SLANG ===
ABBREVIATION_RE = re.compile(
    r"\b(lol|lmao|omg|wtf|brb|btw|smh|tbh|imo|idk|ngl|fr|rn|ong|nvm|wya|hmu|"
    r"fyi|asap|dm|irl|fomo|goat|sus|bet|cap|no cap)\b",
    re.IGNORECASE,
)

# === PROFESSIONAL LANGUAGE ===
PROFESSIONAL_KEYWORDS_RE = re.compile(
    r"\b(meeting|deadline|project|report|schedule|conference|presentation|"
    r"budget|client|invoice|proposal)\b",
    re.IGNORECASE,
)

# === GREETING DETECTION ===
# Regex for greeting at start of message (used by category_features)
GREETING_PATTERN_RE = re.compile(
    r"^(hey|hi|hello|yo|sup|what's up|wassup|heyy|hiya|heya)\b", re.IGNORECASE
)

# Full-line greeting patterns (used by response_mobilization)
GREETING_PATTERNS = [
    r"^(hey|hi|hello|yo|sup|hiya|howdy|what'?s up|wassup|whaddup)!*$",
    r"^(good morning|good afternoon|good evening|good night|gm|gn)\b",
]

# === IMPERATIVE VERBS ===
# Core set shared by both modules (intersection of both usages)
IMPERATIVE_VERBS_CORE = {
    "send",
    "give",
    "take",
    "get",
    "call",
    "help",
    "come",
    "tell",
    "show",
    "make",
    "let",
}

# Extended set for response_mobilization (includes domain-specific verbs)
IMPERATIVE_VERBS_EXTENDED = IMPERATIVE_VERBS_CORE | {
    "bring",
    "grab",
    "pick",
    "text",
    "email",
    "check",
    "look",
    "go",
    "put",
    "find",
    "buy",
    "read",
    "watch",
    "meet",
    "try",
    "open",
}

# === QUESTION WORDS ===
# WH-words for interrogative detection
WH_WORDS = {"what", "where", "when", "who", "why", "how", "which", "whose"}

# Question starters include WH-words + auxiliary verbs used as question openers
QUESTION_STARTERS = {
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "did",
    "do",
    "does",
    "can",
    "could",
    "would",
    "will",
    "should",
}

# === EMOTIONAL MARKERS ===
EMOTIONAL_MARKERS = ["lmao", "lol", "xd", "haha", "omg", "bruh", "rip", "lmfao", "rofl"]

# === AGREEMENT / BACKCHANNEL ===
AGREEMENT_WORDS = {
    "sure",
    "okay",
    "ok",
    "yes",
    "yeah",
    "yep",
    "yup",
    "sounds good",
    "bet",
    "fs",
}

BRIEF_AGREEMENTS = {
    "ok",
    "okay",
    "k",
    "yeah",
    "yep",
    "yup",
    "sure",
    "cool",
    "bet",
    "fs",
    "aight",
}
