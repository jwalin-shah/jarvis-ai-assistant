from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from jarvis.prompts.constants import (
    CASUAL_INDICATORS,
    EMOJI_PATTERN,
    PROFESSIONAL_INDICATORS,
    TEXT_ABBREVIATIONS,
    UserStyleAnalysis,
)

if TYPE_CHECKING:
    from jarvis.relationships import RelationshipProfile

# Pre-compiled patterns for tone/style analysis functions
_WORD_RE = re.compile(r"\b\w+\b")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{3,}")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z]")


def detect_tone(messages: list[str]) -> Literal["casual", "professional", "mixed"]:
    """Detect the conversational tone from a list of messages."""
    if not messages:
        return "casual"

    combined = " ".join(messages).lower()
    words = set(_WORD_RE.findall(combined))

    casual_count = len(words & CASUAL_INDICATORS)
    professional_count = len(words & PROFESSIONAL_INDICATORS)

    emoji_count = len(EMOJI_PATTERN.findall(combined))
    casual_count += emoji_count * 2

    exclamation_count = combined.count("!")
    if exclamation_count > 2:
        casual_count += 1

    if _REPEATED_CHAR_RE.search(combined):
        casual_count += 2

    total = casual_count + professional_count
    if total == 0:
        return "casual"

    casual_ratio = casual_count / total
    if casual_ratio >= 0.7:
        return "casual"
    elif casual_ratio <= 0.3:
        return "professional"
    else:
        return "mixed"


def analyze_user_style(messages: list[str]) -> UserStyleAnalysis:
    """Analyze user's texting style from message examples."""
    if not messages:
        return UserStyleAnalysis()

    messages = [m for m in messages if m and m.strip()]
    if not messages:
        return UserStyleAnalysis()

    lengths = [len(m) for m in messages]
    avg_length = sum(lengths) / len(lengths)
    min_length = min(lengths)
    max_length = max(lengths)

    lowercase_count = 0
    for msg in messages:
        letters_only = _NON_ALPHA_RE.sub("", msg)
        if letters_only:
            lowercase_ratio = sum(1 for c in letters_only if c.islower()) / len(letters_only)
            if lowercase_ratio > 0.9:
                lowercase_count += 1
    uses_lowercase = lowercase_count / len(messages) > 0.7

    combined_lower = " ".join(messages).lower()
    words = set(_WORD_RE.findall(combined_lower))
    found_abbreviations = list(words & TEXT_ABBREVIATIONS)
    uses_abbreviations = len(found_abbreviations) >= 2

    total_chars = sum(len(m) for m in messages)
    total_exclamations = sum(m.count("!") for m in messages)
    total_periods = sum(m.count(".") for m in messages)
    exclamation_density = total_exclamations / max(total_chars, 1)
    period_density = total_periods / max(total_chars, 1)
    uses_minimal_punctuation = exclamation_density < 0.02 and period_density < 0.03

    emoji_count = sum(len(EMOJI_PATTERN.findall(m)) for m in messages)
    emoji_frequency = emoji_count / len(messages)
    exclamation_frequency = total_exclamations / len(messages)

    if len(words & CASUAL_INDICATORS) >= 3 or (uses_abbreviations and uses_lowercase):
        formality: Literal["formal", "casual", "very_casual"] = "very_casual"
    elif len(words & CASUAL_INDICATORS) >= 1 or uses_abbreviations or avg_length < 30:
        formality = "casual"
    else:
        formality = "formal"

    return UserStyleAnalysis(
        avg_length=round(avg_length, 1),
        min_length=min_length,
        max_length=max_length,
        formality=formality,
        uses_lowercase=uses_lowercase,
        uses_abbreviations=uses_abbreviations,
        uses_minimal_punctuation=uses_minimal_punctuation,
        common_abbreviations=found_abbreviations[:5],
        emoji_frequency=round(emoji_frequency, 2),
        exclamation_frequency=round(exclamation_frequency, 2),
    )


def build_style_instructions(style: UserStyleAnalysis) -> str:
    """Build style-matching instructions from a UserStyleAnalysis."""
    instructions = []

    if style.avg_length < 15:
        instructions.append(f"Keep response VERY short (1-{max(5, int(style.avg_length * 1.5))} words)")
    elif style.avg_length < 30:
        instructions.append(f"Keep response brief ({int(style.min_length)}-{int(style.avg_length * 1.2)} chars)")
    elif style.avg_length < 60:
        instructions.append("Keep response concise (1 short sentence)")
    else:
        instructions.append("Response can be 1-2 sentences")

    if style.uses_lowercase:
        instructions.append("Use lowercase (no capitalization)")

    if style.formality == "very_casual":
        instructions.append("Be very casual - no formal greetings like 'Hey!' or 'Hi there!'")
    elif style.formality == "casual":
        instructions.append("Keep it casual - skip formal greetings")

    if style.uses_abbreviations and style.common_abbreviations:
        abbrevs = ", ".join(style.common_abbreviations[:3])
        instructions.append(f"Use abbreviations like: {abbrevs}")

    if style.uses_minimal_punctuation:
        instructions.append("Use minimal punctuation")

    if style.emoji_frequency < 0.1:
        instructions.append("Avoid emojis")
    elif style.emoji_frequency > 0.5:
        instructions.append("Feel free to use emojis")

    if style.exclamation_frequency < 0.3:
        instructions.append("Avoid excessive exclamation marks")

    return "\n".join(f"- {inst}" for inst in instructions)


def determine_effective_tone(
    tone: Literal["casual", "professional", "mixed"],
    relationship_profile: RelationshipProfile | None,
) -> Literal["casual", "professional", "mixed"]:
    """Determine effective tone from base tone and optional relationship profile."""
    from jarvis.relationships import MIN_MESSAGES_FOR_PROFILE

    if relationship_profile and relationship_profile.message_count >= MIN_MESSAGES_FOR_PROFILE:
        if relationship_profile.tone_profile:
            formality = relationship_profile.tone_profile.formality_score
            if formality >= 0.7:
                return "professional"
            elif formality < 0.4:
                return "casual"
            else:
                return "mixed"
    return tone
