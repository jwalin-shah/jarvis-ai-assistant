"""Labeling Functions for Weak Supervision - DIY weak supervision pipeline.

Wraps existing heuristics from jarvis/text_normalizer.py, response_mobilization.py,
and category_classifier.py as labeling functions (LFs) for the weak supervision pipeline.

Each LF returns a category string or ABSTAIN.

Categories:
- ack: Acknowledgments, reactions, simple agreements (skip SLM, use template)
- info: Information requests, commitments, direct questions (context=5)
- emotional: Emotional support, celebrations, empathy needs (context=3)
- social: Casual conversation, banter, stories (context=3)
- clarify: Requests for clarification, ambiguous messages (context=5)

Usage:
    from scripts.labeling_functions import get_registry, apply_all_lfs

    registry = get_registry()
    labels = registry.apply_all(text, context, last_message, metadata)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

# Import existing heuristics
from jarvis.classifiers.response_mobilization import classify_response_pressure
from jarvis.text_normalizer import (
    EMOJI_PATTERN,
    extract_temporal_refs,
    is_acknowledgment_only,
    is_emoji_only,
    is_question,
    is_reaction,
    is_spam_message,
    trigger_expects_content,
)
from jarvis.classifiers.category_classifier import (
    DEICTIC_PATTERN,
    EMOTIONAL_PATTERN,
)

# Constants
ABSTAIN = "__ABSTAIN__"

# ---------------------------------------------------------------------------
# Labeling Function Registry
# ---------------------------------------------------------------------------


@dataclass
class LabelingFunction:
    """A labeling function that votes for a category or abstains."""

    name: str
    fn: Callable[[str, list[str], str, dict | None], str]
    weight: float = 1.0


class LabelingFunctionRegistry:
    """Registry of all labeling functions."""

    def __init__(self) -> None:
        self.lfs: list[LabelingFunction] = []

    def register(self, name: str, weight: float = 1.0):
        """Decorator to register a labeling function."""

        def decorator(fn):
            self.lfs.append(LabelingFunction(name=name, fn=fn, weight=weight))
            return fn

        return decorator

    def apply_all(
        self, text: str, context: list[str], last_message: str, metadata: dict | None = None
    ) -> list[str]:
        """Apply all LFs and return list of labels (one per LF).

        Args:
            text: Message text to label.
            context: Recent conversation messages (before this message).
            last_message: The most recent message before this one.
            metadata: Optional metadata (e.g., DailyDialog act/emotion labels).

        Returns:
            List of labels (one per LF), with ABSTAIN for non-votes.
        """
        labels = []
        for lf in self.lfs:
            try:
                label = lf.fn(text, context, last_message, metadata)
                labels.append(label)
            except Exception:
                labels.append(ABSTAIN)
        return labels

    def get_weights(self) -> list[float]:
        """Get the weight for each LF."""
        return [lf.weight for lf in self.lfs]


# ---------------------------------------------------------------------------
# Labeling Functions
# ---------------------------------------------------------------------------

# Create the registry
_registry = LabelingFunctionRegistry()


# === ACK labeling functions ===


@_registry.register("lf_reaction", weight=1.5)
def lf_reaction(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """Reactions (tapbacks) -> ack."""
    return "ack" if is_reaction(text) else ABSTAIN


@_registry.register("lf_acknowledgment", weight=1.5)
def lf_acknowledgment(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """Generic acknowledgments -> ack."""
    return "ack" if is_acknowledgment_only(text) else ABSTAIN


@_registry.register("lf_emoji_only", weight=1.0)
def lf_emoji_only(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """Emoji-only messages -> ack or clarify depending on context."""
    if not is_emoji_only(text):
        return ABSTAIN
    # If context is thin, it's ambiguous (clarify)
    if len(context) < 2:
        return "clarify"
    return "ack"


@_registry.register("lf_spam", weight=1.0)
def lf_spam(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """Spam/bot messages -> ack (ignore them)."""
    return "ack" if is_spam_message(text) else ABSTAIN


@_registry.register("lf_very_short", weight=0.8)
def lf_very_short(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """Very short messages (≤2 words, not questions) -> ack."""
    words = text.split()
    if len(words) <= 2 and not is_question(text):
        return "ack"
    return ABSTAIN


# === INFO labeling functions ===


@_registry.register("lf_question", weight=1.2)
def lf_question(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """Questions -> info."""
    return "info" if is_question(text) else ABSTAIN


@_registry.register("lf_temporal_refs", weight=0.9)
def lf_temporal_refs(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """Messages with temporal references -> info (likely scheduling/logistics)."""
    refs = extract_temporal_refs(text)
    return "info" if refs else ABSTAIN


@_registry.register("lf_expects_content", weight=1.0)
def lf_expects_content(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """Last message expects content-rich response -> info."""
    if not last_message:
        return ABSTAIN
    return "info" if trigger_expects_content(last_message) else ABSTAIN


# === MOBILIZATION-BASED LFs ===


@_registry.register("lf_mob_high_answer", weight=1.3)
def lf_mob_high_answer(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """HIGH pressure + ANSWER type -> info."""
    result = classify_response_pressure(text)
    if result.pressure.value == "high" and result.response_type.value == "answer":
        return "info"
    return ABSTAIN


@_registry.register("lf_mob_high_commitment", weight=1.3)
def lf_mob_high_commitment(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """HIGH pressure + COMMITMENT type -> info."""
    result = classify_response_pressure(text)
    if result.pressure.value == "high" and result.response_type.value == "commitment":
        return "info"
    return ABSTAIN


@_registry.register("lf_mob_medium_emotional", weight=1.2)
def lf_mob_medium_emotional(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """MEDIUM pressure + EMOTIONAL type -> emotional."""
    result = classify_response_pressure(text)
    if result.pressure.value == "medium" and result.response_type.value == "emotional":
        return "emotional"
    return ABSTAIN


@_registry.register("lf_mob_none_closing", weight=1.0)
def lf_mob_none_closing(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """NONE pressure + CLOSING type -> ack."""
    result = classify_response_pressure(text)
    if result.pressure.value == "none" and result.response_type.value == "closing":
        return "ack"
    return ABSTAIN


@_registry.register("lf_mob_backchannel", weight=1.0)
def lf_mob_backchannel(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """Backchannel feature from mobilization -> ack."""
    result = classify_response_pressure(text)
    return "ack" if result.features.get("is_backchannel") else ABSTAIN


@_registry.register("lf_mob_greeting", weight=0.9)
def lf_mob_greeting(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """Greeting feature from mobilization -> social."""
    result = classify_response_pressure(text)
    return "social" if result.features.get("is_greeting") else ABSTAIN


@_registry.register("lf_mob_reactive", weight=1.1)
def lf_mob_reactive(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """Reactive feature from mobilization -> emotional."""
    result = classify_response_pressure(text)
    return "emotional" if result.features.get("is_reactive") else ABSTAIN


@_registry.register("lf_mob_request", weight=1.1)
def lf_mob_request(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """Request feature from mobilization -> info."""
    result = classify_response_pressure(text)
    return "info" if result.features.get("is_request") else ABSTAIN


@_registry.register("lf_mob_imperative", weight=1.0)
def lf_mob_imperative(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """Imperative feature from mobilization -> info."""
    result = classify_response_pressure(text)
    return "info" if result.features.get("is_imperative") else ABSTAIN


@_registry.register("lf_mob_telling", weight=0.8)
def lf_mob_telling(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """Telling feature from mobilization -> social."""
    result = classify_response_pressure(text)
    return "social" if result.features.get("is_telling") else ABSTAIN


@_registry.register("lf_mob_opinion", weight=0.8)
def lf_mob_opinion(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """Opinion feature from mobilization -> social."""
    result = classify_response_pressure(text)
    return "social" if result.features.get("is_opinion") else ABSTAIN


# === CLARIFY labeling functions ===


@_registry.register("lf_bare_punctuation", weight=1.5)
def lf_bare_punctuation(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """Bare punctuation (?, !!, ...) -> clarify."""
    if re.match(r"^[?!.]{1,3}$", text.strip()):
        return "clarify"
    return ABSTAIN


@_registry.register("lf_ellipsis", weight=0.9)
def lf_ellipsis(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """Messages ending with ellipsis -> clarify."""
    return "clarify" if text.rstrip().endswith("...") else ABSTAIN


@_registry.register("lf_deictic_thin_context", weight=1.0)
def lf_deictic_thin_context(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """Deictic pronouns + thin context -> clarify."""
    if DEICTIC_PATTERN.search(text) and len(context) < 2:
        return "clarify"
    return ABSTAIN


@_registry.register("lf_clarify_signals", weight=1.1)
def lf_clarify_signals(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """Multiple clarify signals -> clarify."""
    # Import the detect function from category_classifier
    from jarvis.classifiers.category_classifier import _detect_clarify_signals

    signals = _detect_clarify_signals(text, context, has_attachment=False)
    return "clarify" if signals >= 2 else ABSTAIN


# === EMOTIONAL labeling functions ===


@_registry.register("lf_emotional_distress", weight=1.4)
def lf_emotional_distress(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """Emotional distress patterns -> emotional."""
    return "emotional" if EMOTIONAL_PATTERN.search(text) else ABSTAIN


@_registry.register("lf_multiple_exclamation", weight=0.9)
def lf_multiple_exclamation(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """3+ exclamation marks -> emotional."""
    return "emotional" if text.count("!") >= 3 else ABSTAIN


# === DAILYDIALOG METADATA LFs (use act/emotion labels as weak signals) ===


@_registry.register("lf_dd_directive", weight=0.7)
def lf_dd_directive(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """DailyDialog act=3 (directive) -> info."""
    if metadata and metadata.get("act") == 3:
        return "info"
    return ABSTAIN


@_registry.register("lf_dd_commissive", weight=0.7)
def lf_dd_commissive(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """DailyDialog act=4 (commissive) -> info."""
    if metadata and metadata.get("act") == 4:
        return "info"
    return ABSTAIN


@_registry.register("lf_dd_negative_emotion", weight=0.6)
def lf_dd_negative_emotion(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """DailyDialog negative emotions (anger=1, disgust=2, fear=3, sadness=5) -> emotional."""
    if metadata and metadata.get("emotion") in {1, 2, 3, 5}:
        return "emotional"
    return ABSTAIN


@_registry.register("lf_dd_positive_emotion", weight=0.6)
def lf_dd_positive_emotion(text: str, context: list[str], last_message: str, metadata: dict | None) -> str:
    """DailyDialog positive emotions (happiness=4, surprise=6) -> emotional."""
    if metadata and metadata.get("emotion") in {4, 6}:
        return "emotional"
    return ABSTAIN


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_registry() -> LabelingFunctionRegistry:
    """Get the singleton labeling function registry."""
    return _registry


def apply_all_lfs(
    text: str, context: list[str], last_message: str = "", metadata: dict | None = None
) -> list[str]:
    """Apply all LFs to a single example.

    Args:
        text: Message text.
        context: Recent conversation context.
        last_message: Most recent message before this one.
        metadata: Optional metadata (e.g., DailyDialog act/emotion).

    Returns:
        List of labels (one per LF), with ABSTAIN for non-votes.
    """
    return _registry.apply_all(text, context, last_message, metadata)


__all__ = [
    "ABSTAIN",
    "LabelingFunction",
    "LabelingFunctionRegistry",
    "get_registry",
    "apply_all_lfs",
]
