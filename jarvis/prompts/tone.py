"""Tone detection and user style analysis.

Analyzes messages for casual vs professional indicators and extracts
user texting patterns (length, formality, abbreviations, emoji usage).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

# Casual indicators
CASUAL_INDICATORS: set[str] = {
    # Emoji patterns are detected separately
    "lol",
    "haha",
    "hehe",
    "lmao",
    "omg",
    "btw",
    "brb",
    "ttyl",
    "idk",
    "ikr",
    "nvm",
    "tbh",
    "imo",
    "fyi",
    "np",
    "k",
    "kk",
    "ok",
    "yeah",
    "yep",
    "nope",
    "yup",
    "gonna",
    "wanna",
    "gotta",
    "cuz",
    "bc",
    "u",
    "ur",
    "r",
    "y",
    "thx",
    "ty",
    "pls",
    "plz",
    "omw",
    "wya",
    "wassup",
    "sup",
    "hey",
    "yo",
    "dude",
    "bro",
    "sis",
    "fam",
    "lit",
    "chill",
    "cool",
    "nice",
    "sick",
    "dope",
    "yay",
    "ooh",
    "ahh",
    "hmm",
    "meh",
    "ugh",
    "whoa",
    "wow",
    "aww",
    "oops",
    "whoops",
}

# Professional indicators
PROFESSIONAL_INDICATORS: set[str] = {
    "regarding",
    "pursuant",
    "attached",
    "please",
    "kindly",
    "sincerely",
    "regards",
    "cordially",
    "respectfully",
    "appreciate",
    "opportunity",
    "discussed",
    "confirmed",
    "scheduled",
    "deadline",
    "deliverable",
    "milestone",
    "stakeholder",
    "proposal",
    "presentation",
    "quarterly",
    "annual",
    "fiscal",
    "eod",
    "eow",
    "asap",
    "cc",
    "per",
    "via",
    "ensure",
    "verify",
    "confirm",
    "acknowledge",
    "proceed",
    "follow-up",
    "followup",
    "meeting",
    "conference",
    "agenda",
    "minutes",
    "action item",
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "dear",
    "hello",
    "good morning",
    "good afternoon",
    "good evening",
}

# Emoji regex pattern
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002702-\U000027b0"  # dingbats
    "\U000024c2-\U0001f251"  # enclosed characters
    "]+"
)


def detect_tone(messages: list[str]) -> Literal["casual", "professional", "mixed"]:
    """Detect the conversational tone from a list of messages.

    Analyzes messages for casual indicators (slang, emoji, informal greetings)
    and professional indicators (formal language, business terminology).

    Args:
        messages: List of message strings to analyze

    Returns:
        "casual" if predominantly informal language
        "professional" if predominantly formal language
        "mixed" if both styles are present or unclear
    """
    if not messages:
        return "casual"  # Default to casual for empty input

    # Combine all messages for analysis
    combined = " ".join(messages).lower()
    words = set(re.findall(r"\b\w+\b", combined))

    # Count indicators
    casual_count = len(words & CASUAL_INDICATORS)
    professional_count = len(words & PROFESSIONAL_INDICATORS)

    # Check for emoji (strong casual indicator)
    emoji_count = len(EMOJI_PATTERN.findall(combined))
    casual_count += emoji_count * 2  # Weight emoji heavily

    # Check for exclamation marks (casual indicator)
    exclamation_count = combined.count("!")
    if exclamation_count > 2:
        casual_count += 1

    # Check for multi-character expressions like "hahahaha" or "looool"
    if re.search(r"(.)\1{3,}", combined):  # 4+ repeated chars
        casual_count += 2

    # Determine tone based on counts
    total = casual_count + professional_count

    if total == 0:
        return "casual"  # Default to casual when no strong indicators

    casual_ratio = casual_count / total

    if casual_ratio >= 0.7:
        return "casual"
    elif casual_ratio <= 0.3:
        return "professional"
    else:
        return "mixed"


# =============================================================================
# User Style Analysis
# =============================================================================


@dataclass
class UserStyleAnalysis:
    """Analysis of user's texting style from message examples.

    Attributes:
        avg_length: Average message length in characters
        min_length: Minimum message length seen
        max_length: Maximum message length seen
        formality: Detected formality level
        uses_lowercase: Whether user typically uses lowercase
        uses_abbreviations: Whether user uses text abbreviations (u, ur, gonna)
        uses_minimal_punctuation: Whether user avoids excessive punctuation
        common_abbreviations: List of abbreviations user commonly uses
        emoji_frequency: Emojis per message
        exclamation_frequency: Exclamation marks per message
    """

    avg_length: float = 50.0
    min_length: int = 0
    max_length: int = 200
    formality: Literal["formal", "casual", "very_casual"] = "casual"
    uses_lowercase: bool = False
    uses_abbreviations: bool = False
    uses_minimal_punctuation: bool = False
    common_abbreviations: list[str] = field(default_factory=list)
    emoji_frequency: float = 0.0
    exclamation_frequency: float = 0.0


# Common text abbreviations to detect
TEXT_ABBREVIATIONS: set[str] = {
    "u",
    "ur",
    "r",
    "y",
    "n",
    "k",
    "kk",
    "ok",
    "bc",
    "cuz",
    "gonna",
    "wanna",
    "gotta",
    "thx",
    "ty",
    "pls",
    "plz",
    "idk",
    "nvm",
    "brb",
    "ttyl",
    "omw",
    "lol",
    "lmao",
    "omg",
    "tbh",
    "imo",
    "ikr",
    "rn",
    "atm",
    "btw",
    "fyi",
}


def analyze_user_style(messages: list[str]) -> UserStyleAnalysis:
    """Analyze user's texting style from message examples.

    Examines the user's actual messages to extract style patterns including:
    - Average/min/max message length
    - Formality level
    - Use of lowercase, abbreviations, punctuation
    - Emoji and exclamation frequency

    Args:
        messages: List of message strings from the user

    Returns:
        UserStyleAnalysis with detected patterns
    """
    if not messages:
        return UserStyleAnalysis()

    # Filter to non-empty messages
    messages = [m for m in messages if m and m.strip()]
    if not messages:
        return UserStyleAnalysis()

    # Calculate length statistics
    lengths = [len(m) for m in messages]
    avg_length = sum(lengths) / len(lengths)
    min_length = min(lengths)
    max_length = max(lengths)

    # Analyze case usage - check if messages are predominantly lowercase
    lowercase_count = 0
    for msg in messages:
        # Remove emojis and punctuation for case analysis
        letters_only = re.sub(r"[^a-zA-Z]", "", msg)
        if letters_only:
            lowercase_ratio = sum(1 for c in letters_only if c.islower()) / len(letters_only)
            if lowercase_ratio > 0.9:  # 90%+ lowercase
                lowercase_count += 1
    uses_lowercase = lowercase_count / len(messages) > 0.7  # 70%+ of messages

    # Detect abbreviation usage
    combined_lower = " ".join(messages).lower()
    words = set(re.findall(r"\b\w+\b", combined_lower))
    found_abbreviations = list(words & TEXT_ABBREVIATIONS)
    uses_abbreviations = len(found_abbreviations) >= 2  # At least 2 different abbreviations

    # Analyze punctuation - minimal if low exclamation/period density
    total_chars = sum(len(m) for m in messages)
    total_exclamations = sum(m.count("!") for m in messages)
    total_periods = sum(m.count(".") for m in messages)
    exclamation_density = total_exclamations / max(total_chars, 1)
    period_density = total_periods / max(total_chars, 1)
    uses_minimal_punctuation = exclamation_density < 0.02 and period_density < 0.03

    # Calculate frequencies per message
    emoji_count = sum(len(EMOJI_PATTERN.findall(m)) for m in messages)
    emoji_frequency = emoji_count / len(messages)
    exclamation_frequency = total_exclamations / len(messages)

    # Determine formality level
    casual_count = len(words & CASUAL_INDICATORS)
    if casual_count >= 3 or (uses_abbreviations and uses_lowercase):
        formality: Literal["formal", "casual", "very_casual"] = "very_casual"
    elif casual_count >= 1 or uses_abbreviations or avg_length < 30:
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
        common_abbreviations=found_abbreviations[:5],  # Top 5
        emoji_frequency=round(emoji_frequency, 2),
        exclamation_frequency=round(exclamation_frequency, 2),
    )


def build_style_instructions(style: UserStyleAnalysis) -> str:
    """Build style-matching instructions from a UserStyleAnalysis.

    Generates specific instructions for the model to match the user's
    texting style based on analyzed patterns.

    Args:
        style: UserStyleAnalysis from analyze_user_style()

    Returns:
        String with style-matching instructions for the prompt
    """
    instructions = []

    # Length guidance
    if style.avg_length < 15:
        instructions.append(
            f"Keep response VERY short (1-{max(5, int(style.avg_length * 1.5))} words)"
        )
    elif style.avg_length < 30:
        instructions.append(
            f"Keep response brief ({int(style.min_length)}-{int(style.avg_length * 1.2)} chars)"
        )
    elif style.avg_length < 60:
        instructions.append("Keep response concise (1 short sentence)")
    else:
        instructions.append("Response can be 1-2 sentences")

    # Case/formality guidance
    if style.uses_lowercase:
        instructions.append("Use lowercase (no capitalization)")

    if style.formality == "very_casual":
        instructions.append("Be very casual - no formal greetings like 'Hey!' or 'Hi there!'")
    elif style.formality == "casual":
        instructions.append("Keep it casual - skip formal greetings")

    # Abbreviation guidance
    if style.uses_abbreviations and style.common_abbreviations:
        abbrevs = ", ".join(style.common_abbreviations[:3])
        instructions.append(f"Use abbreviations like: {abbrevs}")

    # Punctuation guidance
    if style.uses_minimal_punctuation:
        instructions.append("Use minimal punctuation")

    # Emoji guidance
    if style.emoji_frequency < 0.1:
        instructions.append("Avoid emojis")
    elif style.emoji_frequency > 0.5:
        instructions.append("Feel free to use emojis")

    # Exclamation guidance
    if style.exclamation_frequency < 0.3:
        instructions.append("Avoid excessive exclamation marks")

    return "\n".join(f"- {inst}" for inst in instructions)
