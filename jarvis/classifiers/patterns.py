"""Pattern Matching Utilities - Structural pattern matching for classifiers.

Provides reusable pattern matching classes that encapsulate the common pattern
of checking text against ordered regex patterns to determine labels.

Usage:
    from jarvis.classifiers.patterns import StructuralPatternMatcher

    # Define patterns as (regex, label, confidence) tuples
    PATTERNS = [
        (r"^(yes|yeah|yep)", "AGREE", 0.95),
        (r"^(no|nope|nah)", "DECLINE", 0.95),
    ]

    matcher = StructuralPatternMatcher(PATTERNS)
    label, confidence = matcher.match("yeah sounds good")
    # Returns ("AGREE", 0.95)
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from re import Pattern

logger = logging.getLogger(__name__)

LabelT = TypeVar("LabelT")


class StructuralPatternMatcher(Generic[LabelT]):
    """Ordered pattern matcher for structural classification.

    Matches text against an ordered list of (pattern, label, confidence) tuples.
    Returns the first matching pattern's label and confidence.

    This replaces the common pattern of iterating through patterns:
        for pattern, label, conf in PATTERNS:
            if pattern.search(text):
                return label, conf
        return None, 0.0

    Type Parameters:
        LabelT: The type of labels returned (e.g., str, Enum).

    Thread Safety:
        Thread-safe. Patterns are compiled once at initialization.
    """

    def __init__(
        self,
        patterns: list[tuple[str | Pattern[str], LabelT, float]],
        flags: int = re.IGNORECASE,
    ) -> None:
        """Initialize the pattern matcher.

        Args:
            patterns: List of (pattern, label, confidence) tuples.
                Patterns can be strings or pre-compiled regex objects.
            flags: Regex flags to use when compiling string patterns.
                Defaults to re.IGNORECASE.
        """
        self._patterns: list[tuple[Pattern[str], LabelT, float]] = []

        for pattern, label, confidence in patterns:
            if isinstance(pattern, str):
                try:
                    compiled = re.compile(pattern, flags)
                    self._patterns.append((compiled, label, confidence))
                except re.error as e:
                    logger.warning("Invalid regex pattern %r: %s", pattern, e)
            else:
                # Already a compiled pattern
                self._patterns.append((pattern, label, confidence))

    def match(self, text: str) -> tuple[LabelT | None, float]:
        """Match text against patterns in order.

        Args:
            text: Text to match against patterns.

        Returns:
            Tuple of (label, confidence) for the first matching pattern,
            or (None, 0.0) if no pattern matches.
        """
        text_stripped = text.strip()

        for pattern, label, confidence in self._patterns:
            if pattern.search(text_stripped):
                return label, confidence

        return None, 0.0

    def match_with_pattern(
        self,
        text: str,
    ) -> tuple[LabelT | None, float, Pattern[str] | None]:
        """Match text and return the matched pattern as well.

        Useful for debugging or logging which pattern matched.

        Args:
            text: Text to match against patterns.

        Returns:
            Tuple of (label, confidence, matched_pattern) or (None, 0.0, None).
        """
        text_stripped = text.strip()

        for pattern, label, confidence in self._patterns:
            if pattern.search(text_stripped):
                return label, confidence, pattern

        return None, 0.0, None

    def __len__(self) -> int:
        """Return the number of patterns."""
        return len(self._patterns)


class PatternMatcherByLabel(Generic[LabelT]):
    """Pattern matcher organized by label.

    Instead of an ordered list, patterns are grouped by their target label.
    Useful when patterns for each label are independent and you want to
    check all patterns for a label at once.

    This is the pattern used in response_classifier.py where patterns
    are organized as STRUCTURAL_PATTERNS[ResponseType] = [patterns...].

    Type Parameters:
        LabelT: The type of labels (e.g., str, Enum).

    Thread Safety:
        Thread-safe. Patterns are compiled once at initialization.
    """

    def __init__(
        self,
        patterns_by_label: dict[LabelT, list[tuple[str, bool]]],
        default_confidence: float = 0.95,
        flags: int = re.IGNORECASE,
    ) -> None:
        """Initialize the pattern matcher.

        Args:
            patterns_by_label: Dict mapping labels to lists of (pattern, is_regex) tuples.
                If is_regex is False, the pattern is used for simple contains checks.
            default_confidence: Confidence to return for matches.
            flags: Regex flags for compiling patterns.
        """
        self._patterns: dict[LabelT, list[Pattern[str]]] = {}
        self._default_confidence = default_confidence

        for label, pattern_list in patterns_by_label.items():
            compiled = []
            for pattern, is_regex in pattern_list:
                if is_regex:
                    try:
                        compiled.append(re.compile(pattern, flags))
                    except re.error as e:
                        logger.warning("Invalid regex %r for %s: %s", pattern, label, e)
                # Non-regex patterns are skipped (handled by simple string matching if needed)
            self._patterns[label] = compiled

    def match(self, text: str) -> tuple[LabelT | None, float]:
        """Match text against all label patterns.

        Checks patterns for each label and returns the first match.

        Args:
            text: Text to match.

        Returns:
            Tuple of (label, confidence) or (None, 0.0).
        """
        text_clean = text.strip().lower()
        text_no_punct = text_clean.rstrip("!.?,")

        for label, patterns in self._patterns.items():
            for pattern in patterns:
                if pattern.search(text_clean) or pattern.search(text_no_punct):
                    return label, self._default_confidence

        return None, 0.0

    def match_label(self, text: str, label: LabelT) -> bool:
        """Check if text matches any pattern for a specific label.

        Args:
            text: Text to match.
            label: Label to check patterns for.

        Returns:
            True if any pattern for the label matches.
        """
        patterns = self._patterns.get(label, [])
        text_clean = text.strip().lower()

        for pattern in patterns:
            if pattern.search(text_clean):
                return True

        return False

    def labels(self) -> list[LabelT]:
        """Get all labels with patterns.

        Returns:
            List of labels that have patterns defined.
        """
        return list(self._patterns.keys())


__all__ = [
    "StructuralPatternMatcher",
    "PatternMatcherByLabel",
]
