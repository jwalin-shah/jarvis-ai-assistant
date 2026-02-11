"""Shared junk/spam/bot message filters for fact extraction pipelines.

Used by both FactExtractor (regex-based) and CandidateExtractor (GLiNER-based)
to avoid duplicating filtering logic.
"""

from __future__ import annotations

import re


def is_bot_message(text: str, chat_id: str = "") -> bool:
    """Detect high-confidence bot messages (spam, automated replies).

    High-confidence indicators (any 1 match = reject):
    - CVS Pharmacy, Rx Ready (pharmacy bots)
    - "Check out this job at" (LinkedIn spam)
    - Sender is 5-6 digit short code (SMS;-;898287)

    Medium-confidence (any 3 matches = reject):
    - URL + "job" + capitalized company
    - "apply" + "now"
    - >50% all-caps text
    """
    high_confidence_patterns = [
        r"CVS Pharmacy",
        r"Rx Ready",
        r"Check out this job at",
    ]
    for pattern in high_confidence_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    if chat_id and re.match(r"SMS;-;\d{5,6}$", chat_id):
        return True

    medium_factors = 0

    if re.search(r"https?://", text) and re.search(r"\bjob\b", text, re.IGNORECASE):
        if re.search(r"\b[A-Z][a-z]+\s+[A-Z]", text):
            medium_factors += 1

    if re.search(r"\bapply\b", text, re.IGNORECASE) and re.search(r"\bnow\b", text, re.IGNORECASE):
        medium_factors += 1

    letters = [c for c in text if c.isalpha()]
    if letters:
        caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if caps_ratio > 0.5:
            medium_factors += 1

    return medium_factors >= 3


def is_professional_message(text: str) -> bool:
    """Detect professional/business emails that should not be processed.

    Markers:
    - Formal greetings: "Dear", "Hello from"
    - Professional signoffs: "Regards", "Sincerely", "Best regards"
    - Business patterns: "Reminder:", recruiting language
    """
    professional_markers = [
        r"\bdear\s",
        r"hello from",
        r"reminder:\s",
        r"\bregards\b",
        r"\bsincerely\b",
        r"\bbest regards\b",
        r"i appreciate",
        r"opportunity",
        r"from\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|LLC|Ltd|Corp|Hyundai|Recruiting|Marketing|Healthcare|Hospital)",
    ]
    for marker in professional_markers:
        if re.search(marker, text, re.IGNORECASE):
            return True
    return False


def is_junk_message(text: str, chat_id: str = "", min_length: int = 5) -> bool:
    """Combined junk check: too short, bot, or professional message.

    Single entry point for all pre-extraction filtering.
    """
    if not text or len(text) < min_length:
        return True
    if is_bot_message(text, chat_id):
        return True
    if is_professional_message(text):
        return True
    return False
