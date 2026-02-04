"""Feedback integration for quality improvement.

Provides mechanisms to:
- Learn from user edits to drafts
- Track which suggestions get accepted
- Calibrate quality scores from feedback
- Identify improvement opportunities
"""

from __future__ import annotations

import logging
import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of user feedback."""

    ACCEPTED = "accepted"  # Suggestion accepted as-is
    EDITED = "edited"  # Suggestion edited before sending
    REJECTED = "rejected"  # Suggestion rejected
    RATING = "rating"  # Explicit rating (1-5)


class EditType(str, Enum):
    """Types of edits made to suggestions."""

    MINOR = "minor"  # Small corrections (typos, punctuation)
    MODERATE = "moderate"  # Wording changes
    MAJOR = "major"  # Significant rewrites
    COMPLETE = "complete"  # Completely different response


@dataclass
class FeedbackEntry:
    """A single feedback entry."""

    feedback_type: FeedbackType
    timestamp: datetime
    original_text: str
    final_text: str | None = None  # For edits
    edit_type: EditType | None = None
    edit_distance: int = 0
    rating: int | None = None  # 1-5 for explicit ratings
    quality_scores: dict[str, float] = field(default_factory=dict)  # Predicted scores
    contact_id: str | None = None
    model_name: str | None = None
    context_hash: str | None = None  # For deduplication
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feedback_type": self.feedback_type.value,
            "timestamp": self.timestamp.isoformat(),
            "original_text": self.original_text[:100],  # Truncate
            "final_text": self.final_text[:100] if self.final_text else None,
            "edit_type": self.edit_type.value if self.edit_type else None,
            "edit_distance": self.edit_distance,
            "rating": self.rating,
            "quality_scores": {k: round(v, 4) for k, v in self.quality_scores.items()},
            "contact_id": self.contact_id,
            "model_name": self.model_name,
        }


@dataclass
class EditPattern:
    """A pattern identified in user edits."""

    pattern_type: str  # "length", "tone", "content", "format"
    description: str
    frequency: int
    example_before: str
    example_after: str
    confidence: float = 0.0


@dataclass
class FeedbackStats:
    """Statistics derived from feedback."""

    total_feedback: int = 0
    accepted_count: int = 0
    edited_count: int = 0
    rejected_count: int = 0
    acceptance_rate: float = 0.0
    avg_edit_distance: float = 0.0
    avg_rating: float | None = None
    rating_count: int = 0
    # Per-dimension calibration
    dimension_calibration: dict[str, float] = field(default_factory=dict)
    # Per-model stats
    model_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Common edit patterns
    edit_patterns: list[EditPattern] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_feedback": self.total_feedback,
            "accepted_count": self.accepted_count,
            "edited_count": self.edited_count,
            "rejected_count": self.rejected_count,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "avg_edit_distance": round(self.avg_edit_distance, 2),
            "avg_rating": round(self.avg_rating, 2) if self.avg_rating else None,
            "rating_count": self.rating_count,
            "dimension_calibration": {
                k: round(v, 4) for k, v in self.dimension_calibration.items()
            },
            "model_stats": self.model_stats,
            "edit_patterns": [
                {
                    "type": p.pattern_type,
                    "description": p.description,
                    "frequency": p.frequency,
                }
                for p in self.edit_patterns[:5]  # Top 5
            ],
        }


class EditAnalyzer:
    """Analyzes user edits to identify improvement patterns."""

    # Thresholds for edit type classification
    MINOR_EDIT_THRESHOLD = 10  # characters
    MODERATE_EDIT_THRESHOLD = 50  # characters
    MAJOR_EDIT_THRESHOLD = 0.5  # 50% change

    def analyze_edit(self, original: str, edited: str) -> tuple[EditType, int, list[str]]:
        """Analyze an edit to classify type and identify patterns.

        Args:
            original: Original suggestion
            edited: User's edited version

        Returns:
            Tuple of (edit_type, edit_distance, patterns_found)
        """
        edit_distance = self._compute_edit_distance(original, edited)
        patterns: list[str] = []

        # Classify edit type
        original_len = len(original)
        change_ratio = edit_distance / max(original_len, 1)

        if edit_distance <= self.MINOR_EDIT_THRESHOLD:
            edit_type = EditType.MINOR
        elif edit_distance <= self.MODERATE_EDIT_THRESHOLD:
            edit_type = EditType.MODERATE
        elif change_ratio >= self.MAJOR_EDIT_THRESHOLD:
            edit_type = EditType.MAJOR
        else:
            edit_type = EditType.MODERATE

        # Check for complete rewrite
        if change_ratio >= 0.8:
            edit_type = EditType.COMPLETE
            patterns.append("complete_rewrite")

        # Identify specific patterns
        patterns.extend(self._identify_patterns(original, edited))

        return edit_type, edit_distance, patterns

    def _compute_edit_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance."""
        if len(s1) < len(s2):
            return self._compute_edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _identify_patterns(self, original: str, edited: str) -> list[str]:
        """Identify specific edit patterns."""
        patterns: list[str] = []
        orig_lower = original.lower()
        edit_lower = edited.lower()

        # Length changes
        orig_words = len(original.split())
        edit_words = len(edited.split())
        if edit_words < orig_words * 0.7:
            patterns.append("shortened")
        elif edit_words > orig_words * 1.3:
            patterns.append("lengthened")

        # Tone changes
        formal_markers = {"please", "would", "could", "thank", "appreciate"}
        casual_markers = {"hey", "yeah", "cool", "awesome", "hi"}

        orig_formal = sum(1 for m in formal_markers if m in orig_lower)
        edit_formal = sum(1 for m in formal_markers if m in edit_lower)
        orig_casual = sum(1 for m in casual_markers if m in orig_lower)
        edit_casual = sum(1 for m in casual_markers if m in edit_lower)

        if edit_formal > orig_formal:
            patterns.append("more_formal")
        elif edit_casual > orig_casual:
            patterns.append("more_casual")

        # Punctuation changes
        orig_exclaim = original.count("!")
        edit_exclaim = edited.count("!")
        if edit_exclaim > orig_exclaim:
            patterns.append("added_enthusiasm")
        elif orig_exclaim > edit_exclaim:
            patterns.append("reduced_enthusiasm")

        # Emoji changes
        orig_emoji = len(re.findall(r"[\U0001F300-\U0001F9FF]", original))
        edit_emoji = len(re.findall(r"[\U0001F300-\U0001F9FF]", edited))
        if edit_emoji > orig_emoji:
            patterns.append("added_emoji")
        elif orig_emoji > edit_emoji:
            patterns.append("removed_emoji")

        # Name usage
        if re.search(r"\b[A-Z][a-z]+\b", edited) and not re.search(r"\b[A-Z][a-z]+\b", original):
            patterns.append("added_name")

        return patterns


class FeedbackCollector:
    """Collects and analyzes user feedback for quality improvement.

    Tracks:
    - Acceptance/rejection rates
    - Edit patterns and distances
    - Per-model quality calibration
    - Per-contact preferences
    """

    MAX_ENTRIES = 10000
    CALIBRATION_WINDOW_DAYS = 30

    def __init__(self) -> None:
        """Initialize the feedback collector."""
        self._lock = threading.Lock()
        self._entries: list[FeedbackEntry] = []
        self._edit_analyzer = EditAnalyzer()

        # Running counters
        self._accepted_count = 0
        self._edited_count = 0
        self._rejected_count = 0
        self._total_edit_distance = 0
        self._total_ratings = 0.0
        self._rating_count = 0

        # Pattern tracking
        self._pattern_counts: dict[str, int] = defaultdict(int)

        # Per-model tracking
        self._model_feedback: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "accepted": 0,
                "edited": 0,
                "rejected": 0,
                "total_rating": 0.0,
                "rating_count": 0,
            }
        )

        # Per-contact tracking
        self._contact_feedback: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"accepted": 0, "edited": 0, "rejected": 0, "patterns": []}
        )

        # Dimension calibration data
        self._dimension_feedback: dict[str, list[tuple[float, float]]] = defaultdict(list)

    def record_acceptance(
        self,
        original_text: str,
        quality_scores: dict[str, float] | None = None,
        contact_id: str | None = None,
        model_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record that a suggestion was accepted as-is.

        Args:
            original_text: The accepted suggestion text
            quality_scores: Predicted quality scores
            contact_id: Contact who received the suggestion
            model_name: Model that generated the suggestion
            metadata: Additional metadata
        """
        entry = FeedbackEntry(
            feedback_type=FeedbackType.ACCEPTED,
            timestamp=datetime.now(UTC),
            original_text=original_text,
            quality_scores=quality_scores or {},
            contact_id=contact_id,
            model_name=model_name,
            metadata=metadata or {},
        )

        with self._lock:
            self._entries.append(entry)
            self._accepted_count += 1

            if model_name:
                self._model_feedback[model_name]["accepted"] += 1

            if contact_id:
                self._contact_feedback[contact_id]["accepted"] += 1

            # Update dimension calibration (accepted = high quality)
            for dim, score in (quality_scores or {}).items():
                self._dimension_feedback[dim].append((score, 1.0))

            self._trim_entries()

    def record_edit(
        self,
        original_text: str,
        edited_text: str,
        quality_scores: dict[str, float] | None = None,
        contact_id: str | None = None,
        model_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackEntry:
        """Record that a suggestion was edited before sending.

        Args:
            original_text: Original suggestion
            edited_text: User's edited version
            quality_scores: Predicted quality scores
            contact_id: Contact who received the suggestion
            model_name: Model that generated the suggestion
            metadata: Additional metadata

        Returns:
            FeedbackEntry with edit analysis
        """
        # Analyze the edit
        edit_type, edit_distance, patterns = self._edit_analyzer.analyze_edit(
            original_text, edited_text
        )

        entry = FeedbackEntry(
            feedback_type=FeedbackType.EDITED,
            timestamp=datetime.now(UTC),
            original_text=original_text,
            final_text=edited_text,
            edit_type=edit_type,
            edit_distance=edit_distance,
            quality_scores=quality_scores or {},
            contact_id=contact_id,
            model_name=model_name,
            metadata=metadata or {},
        )

        with self._lock:
            self._entries.append(entry)
            self._edited_count += 1
            self._total_edit_distance += edit_distance

            # Track patterns
            for pattern in patterns:
                self._pattern_counts[pattern] += 1

            if model_name:
                self._model_feedback[model_name]["edited"] += 1

            if contact_id:
                self._contact_feedback[contact_id]["edited"] += 1
                self._contact_feedback[contact_id]["patterns"].extend(patterns)

            # Update dimension calibration (edited = partial quality)
            # Quality proportional to how much was kept
            keep_ratio = 1 - (edit_distance / max(len(original_text), 1))
            keep_ratio = max(0.0, min(1.0, keep_ratio))
            for dim, score in (quality_scores or {}).items():
                self._dimension_feedback[dim].append((score, keep_ratio))

            self._trim_entries()

        return entry

    def record_rejection(
        self,
        original_text: str,
        quality_scores: dict[str, float] | None = None,
        contact_id: str | None = None,
        model_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record that a suggestion was rejected.

        Args:
            original_text: The rejected suggestion
            quality_scores: Predicted quality scores
            contact_id: Contact who received the suggestion
            model_name: Model that generated the suggestion
            metadata: Additional metadata
        """
        entry = FeedbackEntry(
            feedback_type=FeedbackType.REJECTED,
            timestamp=datetime.now(UTC),
            original_text=original_text,
            quality_scores=quality_scores or {},
            contact_id=contact_id,
            model_name=model_name,
            metadata=metadata or {},
        )

        with self._lock:
            self._entries.append(entry)
            self._rejected_count += 1

            if model_name:
                self._model_feedback[model_name]["rejected"] += 1

            if contact_id:
                self._contact_feedback[contact_id]["rejected"] += 1

            # Update dimension calibration (rejected = low quality)
            for dim, score in (quality_scores or {}).items():
                self._dimension_feedback[dim].append((score, 0.0))

            self._trim_entries()

    def record_rating(
        self,
        original_text: str,
        rating: int,
        quality_scores: dict[str, float] | None = None,
        contact_id: str | None = None,
        model_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an explicit user rating.

        Args:
            original_text: The rated suggestion
            rating: Rating from 1-5
            quality_scores: Predicted quality scores
            contact_id: Contact who rated
            model_name: Model that generated the suggestion
            metadata: Additional metadata
        """
        rating = max(1, min(5, rating))  # Clamp to 1-5

        entry = FeedbackEntry(
            feedback_type=FeedbackType.RATING,
            timestamp=datetime.now(UTC),
            original_text=original_text,
            rating=rating,
            quality_scores=quality_scores or {},
            contact_id=contact_id,
            model_name=model_name,
            metadata=metadata or {},
        )

        with self._lock:
            self._entries.append(entry)
            self._total_ratings += rating
            self._rating_count += 1

            if model_name:
                self._model_feedback[model_name]["total_rating"] += rating
                self._model_feedback[model_name]["rating_count"] += 1

            # Update dimension calibration based on rating
            normalized_rating = (rating - 1) / 4  # 0-1 scale
            for dim, score in (quality_scores or {}).items():
                self._dimension_feedback[dim].append((score, normalized_rating))

            self._trim_entries()

    def get_stats(self) -> FeedbackStats:
        """Get feedback statistics.

        Returns:
            FeedbackStats with current statistics
        """
        with self._lock:
            total = self._accepted_count + self._edited_count + self._rejected_count

            if total > 0:
                acceptance_rate = (self._accepted_count + self._edited_count) / total
            else:
                acceptance_rate = 0.0

            if self._edited_count > 0:
                avg_edit_distance = self._total_edit_distance / self._edited_count
            else:
                avg_edit_distance = 0.0

            if self._rating_count > 0:
                avg_rating = self._total_ratings / self._rating_count
            else:
                avg_rating = None

            # Calculate dimension calibration
            dimension_calibration = self._compute_calibration()

            # Get model stats
            model_stats = {}
            for model, stats in self._model_feedback.items():
                model_total = stats["accepted"] + stats["edited"] + stats["rejected"]
                if model_total > 0:
                    model_acceptance = (stats["accepted"] + stats["edited"]) / model_total
                else:
                    model_acceptance = 0.0

                model_stats[model] = {
                    "total": model_total,
                    "acceptance_rate": round(model_acceptance, 4),
                    "avg_rating": (
                        round(stats["total_rating"] / stats["rating_count"], 2)
                        if stats["rating_count"] > 0
                        else None
                    ),
                }

            # Get edit patterns
            edit_patterns = self._get_top_patterns()

            return FeedbackStats(
                total_feedback=total + self._rating_count,
                accepted_count=self._accepted_count,
                edited_count=self._edited_count,
                rejected_count=self._rejected_count,
                acceptance_rate=acceptance_rate,
                avg_edit_distance=avg_edit_distance,
                avg_rating=avg_rating,
                rating_count=self._rating_count,
                dimension_calibration=dimension_calibration,
                model_stats=model_stats,
                edit_patterns=edit_patterns,
            )

    def get_calibration_factor(self, dimension: str) -> float:
        """Get calibration factor for a quality dimension.

        The calibration factor adjusts predicted scores based on
        how they correlate with actual user feedback.

        Args:
            dimension: Quality dimension name

        Returns:
            Calibration factor (1.0 = no adjustment needed)
        """
        with self._lock:
            data = list(self._dimension_feedback.get(dimension, []))

        return self._compute_calibration_factor(data)

    def _compute_calibration_factor(self, data: list[tuple[float, float]]) -> float:
        """Compute calibration factor from data (internal, no lock).

        Args:
            data: List of (predicted_score, actual_quality) tuples

        Returns:
            Calibration factor (1.0 = no adjustment needed)
        """
        if len(data) < 10:
            return 1.0  # Not enough data

        # Simple linear regression to find adjustment
        predicted_scores = [d[0] for d in data]
        actual_quality = [d[1] for d in data]

        # Calculate means
        pred_mean = sum(predicted_scores) / len(predicted_scores)
        actual_mean = sum(actual_quality) / len(actual_quality)

        if pred_mean == 0:
            return 1.0

        # Calculate calibration factor
        return actual_mean / pred_mean

    def get_contact_preferences(self, contact_id: str) -> dict[str, Any]:
        """Get learned preferences for a contact.

        Args:
            contact_id: Contact identifier

        Returns:
            Dictionary of preferences learned from feedback
        """
        with self._lock:
            feedback = self._contact_feedback.get(contact_id, {})

        if not feedback:
            return {}

        total = feedback.get("accepted", 0) + feedback.get("edited", 0)

        # Analyze patterns
        patterns = feedback.get("patterns", [])
        pattern_counts: dict[str, int] = defaultdict(int)
        for p in patterns:
            pattern_counts[p] += 1

        preferences = {
            "acceptance_rate": (
                (feedback.get("accepted", 0) + feedback.get("edited", 0))
                / max(total + feedback.get("rejected", 0), 1)
            ),
            "edit_rate": feedback.get("edited", 0) / max(total, 1),
        }

        # Infer preferences from patterns
        if pattern_counts.get("shortened", 0) > pattern_counts.get("lengthened", 0):
            preferences["length_preference"] = "short"
        elif pattern_counts.get("lengthened", 0) > pattern_counts.get("shortened", 0):
            preferences["length_preference"] = "long"

        if pattern_counts.get("more_formal", 0) > pattern_counts.get("more_casual", 0):
            preferences["tone_preference"] = "formal"
        elif pattern_counts.get("more_casual", 0) > pattern_counts.get("more_formal", 0):
            preferences["tone_preference"] = "casual"

        if pattern_counts.get("added_emoji", 0) > 2:
            preferences["emoji_preference"] = "more"
        elif pattern_counts.get("removed_emoji", 0) > 2:
            preferences["emoji_preference"] = "less"

        return preferences

    def _compute_calibration(self) -> dict[str, float]:
        """Compute calibration factors for all dimensions.

        Note: This is called from within get_stats which already holds the lock,
        so we use the internal _compute_calibration_factor directly.
        """
        calibration = {}
        for dim, data in self._dimension_feedback.items():
            factor = self._compute_calibration_factor(list(data))
            if factor != 1.0:
                calibration[dim] = factor
        return calibration

    def _get_top_patterns(self, n: int = 10) -> list[EditPattern]:
        """Get top N most common edit patterns."""
        patterns = []
        sorted_patterns = sorted(self._pattern_counts.items(), key=lambda x: x[1], reverse=True)[:n]

        pattern_descriptions = {
            "shortened": "User shortened the response",
            "lengthened": "User lengthened the response",
            "more_formal": "User made response more formal",
            "more_casual": "User made response more casual",
            "added_enthusiasm": "User added enthusiasm (!)",
            "reduced_enthusiasm": "User reduced enthusiasm",
            "added_emoji": "User added emoji",
            "removed_emoji": "User removed emoji",
            "added_name": "User added recipient's name",
            "complete_rewrite": "User completely rewrote response",
        }

        for pattern_name, count in sorted_patterns:
            ptype = pattern_name.split("_")[0] if "_" in pattern_name else pattern_name
            patterns.append(
                EditPattern(
                    pattern_type=ptype,
                    description=pattern_descriptions.get(pattern_name, f"Pattern: {pattern_name}"),
                    frequency=count,
                    example_before="",  # Would need to track examples
                    example_after="",
                    confidence=min(1.0, count / 10),
                )
            )

        return patterns

    def _trim_entries(self) -> None:
        """Trim entries if over capacity (called with lock held)."""
        if len(self._entries) > self.MAX_ENTRIES:
            self._entries = self._entries[-self.MAX_ENTRIES :]

    def reset(self) -> None:
        """Reset all feedback data."""
        with self._lock:
            self._entries.clear()
            self._accepted_count = 0
            self._edited_count = 0
            self._rejected_count = 0
            self._total_edit_distance = 0
            self._total_ratings = 0.0
            self._rating_count = 0
            self._pattern_counts.clear()
            self._model_feedback.clear()
            self._contact_feedback.clear()
            self._dimension_feedback.clear()


# Global singleton
_feedback_collector: FeedbackCollector | None = None
_collector_lock = threading.Lock()


def get_feedback_collector() -> FeedbackCollector:
    """Get the global feedback collector instance.

    Returns:
        Shared FeedbackCollector instance
    """
    global _feedback_collector
    if _feedback_collector is None:
        with _collector_lock:
            if _feedback_collector is None:
                _feedback_collector = FeedbackCollector()
    return _feedback_collector


def reset_feedback_collector() -> None:
    """Reset the global feedback collector instance."""
    global _feedback_collector
    with _collector_lock:
        _feedback_collector = None
