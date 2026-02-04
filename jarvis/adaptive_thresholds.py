"""Adaptive Threshold Manager for JARVIS Router.

Learns optimal routing thresholds from user feedback data.

The adaptive threshold system analyzes feedback patterns at different similarity
score ranges to dynamically adjust routing thresholds (quick_reply, context, generate).

Key concepts:
- Feedback is grouped into similarity score buckets (e.g., 0.90-0.95)
- Acceptance rate is computed for each bucket
- Thresholds are adjusted to maximize acceptance while maintaining quality
- Learning rate controls adaptation speed to prevent volatility

Usage:
    from jarvis.adaptive_thresholds import get_adaptive_threshold_manager

    manager = get_adaptive_threshold_manager()
    thresholds = manager.get_adapted_thresholds()
    # Returns: {"quick_reply": 0.92, "context": 0.68, "generate": 0.48}
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from jarvis.config import AdaptiveThresholdConfig, get_config
from jarvis.evaluation import FeedbackAction, FeedbackStore, get_feedback_store

if TYPE_CHECKING:
    from jarvis.evaluation import FeedbackEntry

logger = logging.getLogger(__name__)


@dataclass
class SimilarityBucketStats:
    """Statistics for a similarity score bucket.

    Attributes:
        bucket_start: Lower bound of the bucket (inclusive).
        bucket_end: Upper bound of the bucket (exclusive).
        total_count: Total feedback entries in this bucket.
        sent_count: Entries where user sent suggestion unchanged.
        edited_count: Entries where user edited suggestion.
        dismissed_count: Entries where user dismissed suggestion.
        acceptance_rate: (sent + edited) / total, or 0 if no data.
        avg_similarity: Average similarity score in this bucket.
    """

    bucket_start: float
    bucket_end: float
    total_count: int = 0
    sent_count: int = 0
    edited_count: int = 0
    dismissed_count: int = 0
    acceptance_rate: float = 0.0
    avg_similarity: float = 0.0


@dataclass
class AdaptedThresholds:
    """Result of adaptive threshold computation.

    Attributes:
        quick_reply: Threshold for quick reply routing.
        context: Threshold for context-based generation.
        generate: Threshold for cautious generation.
        computed_at: Timestamp when thresholds were computed.
        sample_count: Number of feedback samples used.
        bucket_stats: Per-bucket statistics used for computation.
        is_fallback: True if using fallback (insufficient data).
    """

    quick_reply: float
    context: float
    generate: float
    computed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    sample_count: int = 0
    bucket_stats: list[SimilarityBucketStats] = field(default_factory=list)
    is_fallback: bool = False


class AdaptiveThresholdManager:
    """Manages adaptive threshold computation from feedback data.

    Thread-safe implementation with caching and configurable parameters.

    The adaptation algorithm:
    1. Groups feedback by similarity score buckets
    2. Computes acceptance rate for each bucket
    3. Finds thresholds where acceptance drops below target
    4. Applies learning rate to smooth transitions
    5. Enforces min/max bounds for safety
    """

    def __init__(
        self,
        config: AdaptiveThresholdConfig | None = None,
        feedback_store: FeedbackStore | None = None,
    ) -> None:
        """Initialize the adaptive threshold manager.

        Args:
            config: Adaptive threshold configuration. Uses default if None.
            feedback_store: Feedback store for data. Uses singleton if None.
        """
        self._config = config
        self._feedback_store = feedback_store
        self._lock = threading.Lock()

        # Cache for computed thresholds
        self._cached_thresholds: AdaptedThresholds | None = None
        self._cache_time: float = 0

        # Store previous thresholds for learning rate smoothing
        self._previous_thresholds: dict[str, float] | None = None

    @property
    def config(self) -> AdaptiveThresholdConfig:
        """Get the adaptive threshold configuration."""
        if self._config is None:
            self._config = get_config().routing.adaptive
        return self._config

    @property
    def feedback_store(self) -> FeedbackStore:
        """Get the feedback store."""
        if self._feedback_store is None:
            self._feedback_store = get_feedback_store()
        return self._feedback_store

    def get_adapted_thresholds(self) -> dict[str, float]:
        """Get current adapted thresholds.

        Returns cached thresholds if still valid, otherwise recomputes.

        Returns:
            Dictionary with threshold values for quick_reply, context, generate.
        """
        if not self.config.enabled:
            # Return base thresholds from config
            routing = get_config().routing
            return {
                "quick_reply": routing.quick_reply_threshold,
                "context": routing.context_threshold,
                "generate": routing.generate_threshold,
            }

        # Check cache validity
        cache_ttl_seconds = self.config.update_interval_minutes * 60
        now = time.time()

        if self._cached_thresholds and (now - self._cache_time) < cache_ttl_seconds:
            return {
                "quick_reply": self._cached_thresholds.quick_reply,
                "context": self._cached_thresholds.context,
                "generate": self._cached_thresholds.generate,
            }

        # Recompute thresholds
        with self._lock:
            # Double-check after acquiring lock
            if self._cached_thresholds and (now - self._cache_time) < cache_ttl_seconds:
                return {
                    "quick_reply": self._cached_thresholds.quick_reply,
                    "context": self._cached_thresholds.context,
                    "generate": self._cached_thresholds.generate,
                }

            adapted = self._compute_adapted_thresholds()
            self._cached_thresholds = adapted
            self._cache_time = now

            logger.info(
                "Adapted thresholds computed: quick_reply=%.3f, context=%.3f, generate=%.3f "
                "(samples=%d, fallback=%s)",
                adapted.quick_reply,
                adapted.context,
                adapted.generate,
                adapted.sample_count,
                adapted.is_fallback,
            )

            return {
                "quick_reply": adapted.quick_reply,
                "context": adapted.context,
                "generate": adapted.generate,
            }

    def _compute_adapted_thresholds(self) -> AdaptedThresholds:
        """Compute adapted thresholds from feedback data.

        Returns:
            AdaptedThresholds with computed values.
        """
        routing = get_config().routing

        # Get base thresholds as fallback
        base_thresholds = {
            "quick_reply": routing.quick_reply_threshold,
            "context": routing.context_threshold,
            "generate": routing.generate_threshold,
        }

        # Get feedback entries within time window
        entries = self._get_feedback_in_window()

        if len(entries) < self.config.min_feedback_samples:
            logger.debug(
                "Insufficient feedback samples (%d < %d), using base thresholds",
                len(entries),
                self.config.min_feedback_samples,
            )
            return AdaptedThresholds(
                quick_reply=base_thresholds["quick_reply"],
                context=base_thresholds["context"],
                generate=base_thresholds["generate"],
                sample_count=len(entries),
                is_fallback=True,
            )

        # Analyze feedback by similarity buckets
        bucket_stats = self._analyze_feedback_by_similarity(entries)

        if not bucket_stats:
            return AdaptedThresholds(
                quick_reply=base_thresholds["quick_reply"],
                context=base_thresholds["context"],
                generate=base_thresholds["generate"],
                sample_count=len(entries),
                is_fallback=True,
            )

        # Compute optimal thresholds based on acceptance rates
        optimal = self._find_optimal_thresholds(bucket_stats, base_thresholds)

        # Apply learning rate smoothing
        smoothed = self._apply_learning_rate(optimal, base_thresholds)

        # Enforce bounds
        bounded = self._enforce_bounds(smoothed)

        # Update previous thresholds for next iteration
        self._previous_thresholds = bounded.copy()

        return AdaptedThresholds(
            quick_reply=bounded["quick_reply"],
            context=bounded["context"],
            generate=bounded["generate"],
            sample_count=len(entries),
            bucket_stats=bucket_stats,
            is_fallback=False,
        )

    def _get_feedback_in_window(self) -> list[FeedbackEntry]:
        """Get feedback entries within the configured time window.

        Returns:
            List of feedback entries with similarity scores in metadata.
        """
        entries = self.feedback_store.get_recent_entries(limit=10000)

        if self.config.adaptation_window_hours == 0:
            # Use all entries
            relevant = entries
        else:
            # Filter by time window
            cutoff = datetime.now(UTC) - timedelta(hours=self.config.adaptation_window_hours)
            relevant = [e for e in entries if e.timestamp >= cutoff]

        # Only include entries with similarity scores in metadata
        with_similarity = [e for e in relevant if e.metadata.get("similarity_score") is not None]

        return with_similarity

    def _analyze_feedback_by_similarity(
        self, entries: list[FeedbackEntry]
    ) -> list[SimilarityBucketStats]:
        """Analyze feedback entries grouped by similarity score buckets.

        Args:
            entries: Feedback entries with similarity_score in metadata.

        Returns:
            List of bucket statistics sorted by bucket_start descending.
        """
        bucket_size = self.config.similarity_bucket_size
        buckets: dict[float, SimilarityBucketStats] = {}

        # Initialize buckets from 0.0 to 1.0
        # Use round() to handle floating point precision issues
        bucket_start = 0.0
        while bucket_start < 1.0:
            bucket_end = min(round(bucket_start + bucket_size, 2), 1.0)
            rounded_start = round(bucket_start, 2)
            buckets[rounded_start] = SimilarityBucketStats(
                bucket_start=rounded_start,
                bucket_end=bucket_end,
            )
            bucket_start = round(bucket_start + bucket_size, 2)

        # Populate buckets with feedback data
        for entry in entries:
            similarity = entry.metadata.get("similarity_score", 0.0)

            # Find the bucket for this similarity
            # Use round() to handle floating point precision issues
            bucket_key = round((similarity // bucket_size) * bucket_size, 2)
            bucket_key = max(0.0, min(bucket_key, round(1.0 - bucket_size, 2)))

            if bucket_key not in buckets:
                continue

            bucket = buckets[bucket_key]
            bucket.total_count += 1
            bucket.avg_similarity = (
                bucket.avg_similarity * (bucket.total_count - 1) + similarity
            ) / bucket.total_count

            if entry.action == FeedbackAction.SENT:
                bucket.sent_count += 1
            elif entry.action == FeedbackAction.EDITED:
                bucket.edited_count += 1
            elif entry.action in (FeedbackAction.DISMISSED, FeedbackAction.WROTE_FROM_SCRATCH):
                bucket.dismissed_count += 1
            # COPIED is neutral, not counted

        # Compute acceptance rates
        for bucket in buckets.values():
            if bucket.total_count > 0:
                accepted = bucket.sent_count + bucket.edited_count
                bucket.acceptance_rate = accepted / bucket.total_count

        # Return sorted by bucket_start descending (highest similarity first)
        result = sorted(buckets.values(), key=lambda b: b.bucket_start, reverse=True)

        # Filter out empty buckets
        return [b for b in result if b.total_count > 0]

    def _find_optimal_thresholds(
        self,
        bucket_stats: list[SimilarityBucketStats],
        base_thresholds: dict[str, float],
    ) -> dict[str, float]:
        """Find optimal thresholds based on bucket acceptance rates.

        Strategy:
        - quick_reply: Find the highest similarity where acceptance is high (>= target)
        - context: Find where acceptance is moderate (>= target * 0.85)
        - generate: Find where acceptance is acceptable (>= target * 0.7)

        Args:
            bucket_stats: List of bucket statistics sorted by similarity descending.
            base_thresholds: Fallback threshold values.

        Returns:
            Dictionary with computed optimal thresholds.
        """
        target = self.config.acceptance_target

        # Compute thresholds for different acceptance targets
        quick_reply_target = target
        context_target = target * 0.85
        generate_target = target * 0.70

        optimal = base_thresholds.copy()

        # Find quick_reply threshold: highest similarity with good acceptance
        # Walk from high to low similarity, find first bucket meeting target
        quick_reply_threshold = self._find_threshold_for_rate(
            bucket_stats, quick_reply_target, min_samples=5
        )
        if quick_reply_threshold is not None:
            optimal["quick_reply"] = quick_reply_threshold

        # Find context threshold: moderate acceptance
        context_threshold = self._find_threshold_for_rate(
            bucket_stats, context_target, min_samples=3
        )
        if context_threshold is not None:
            optimal["context"] = context_threshold

        # Find generate threshold: acceptable acceptance
        generate_threshold = self._find_threshold_for_rate(
            bucket_stats, generate_target, min_samples=3
        )
        if generate_threshold is not None:
            optimal["generate"] = generate_threshold

        # Ensure ordering: quick_reply > context > generate
        if optimal["context"] >= optimal["quick_reply"]:
            optimal["context"] = optimal["quick_reply"] - 0.1
        if optimal["generate"] >= optimal["context"]:
            optimal["generate"] = optimal["context"] - 0.1

        return optimal

    def _find_threshold_for_rate(
        self,
        bucket_stats: list[SimilarityBucketStats],
        target_rate: float,
        min_samples: int = 3,
    ) -> float | None:
        """Find the threshold similarity that achieves target acceptance rate.

        Walks from high to low similarity buckets, finds the lowest similarity
        bucket that still meets the target acceptance rate.

        Args:
            bucket_stats: Bucket statistics sorted by similarity descending.
            target_rate: Target acceptance rate to achieve.
            min_samples: Minimum samples in bucket to consider it valid.

        Returns:
            Threshold value, or None if no valid threshold found.
        """
        last_good_threshold = None

        for bucket in bucket_stats:
            if bucket.total_count < min_samples:
                continue

            if bucket.acceptance_rate >= target_rate:
                # This bucket meets the target, remember its lower bound
                last_good_threshold = bucket.bucket_start
            else:
                # Acceptance dropped below target, stop here
                # Return the previous good threshold (higher similarity)
                break

        return last_good_threshold

    def _apply_learning_rate(
        self,
        optimal: dict[str, float],
        base_thresholds: dict[str, float],
    ) -> dict[str, float]:
        """Apply learning rate to smooth threshold transitions.

        Uses exponential moving average: new = prev + learning_rate * (optimal - prev)

        Args:
            optimal: Newly computed optimal thresholds.
            base_thresholds: Base thresholds from config.

        Returns:
            Smoothed thresholds.
        """
        learning_rate = self.config.learning_rate

        # Use previous thresholds if available, otherwise base
        prev = self._previous_thresholds or base_thresholds

        smoothed = {}
        for key in ("quick_reply", "context", "generate"):
            prev_val = prev.get(key, base_thresholds[key])
            optimal_val = optimal.get(key, base_thresholds[key])
            smoothed[key] = prev_val + learning_rate * (optimal_val - prev_val)

        return smoothed

    def _enforce_bounds(self, thresholds: dict[str, float]) -> dict[str, float]:
        """Enforce minimum and maximum bounds on thresholds.

        Args:
            thresholds: Computed thresholds.

        Returns:
            Bounded thresholds.
        """
        min_bounds = self.config.min_threshold_bounds
        max_bounds = self.config.max_threshold_bounds

        bounded = {}
        for key in ("quick_reply", "context", "generate"):
            val = thresholds.get(key, 0.5)
            min_val = min_bounds.get(key, 0.3)
            max_val = max_bounds.get(key, 0.99)
            bounded[key] = max(min_val, min(max_val, val))

        return bounded

    def get_adaptation_stats(self) -> dict[str, Any]:
        """Get statistics about the adaptive threshold system.

        Returns:
            Dictionary with adaptation statistics.
        """
        entries = self._get_feedback_in_window()
        bucket_stats = self._analyze_feedback_by_similarity(entries) if entries else []

        # Compute overall stats
        total_feedback = len(entries)
        total_sent = sum(1 for e in entries if e.action == FeedbackAction.SENT)
        total_edited = sum(1 for e in entries if e.action == FeedbackAction.EDITED)
        total_dismissed = sum(
            1
            for e in entries
            if e.action in (FeedbackAction.DISMISSED, FeedbackAction.WROTE_FROM_SCRATCH)
        )

        acceptance_rate = (
            (total_sent + total_edited) / total_feedback if total_feedback > 0 else 0.0
        )

        current = self.get_adapted_thresholds()

        return {
            "enabled": self.config.enabled,
            "total_feedback_in_window": total_feedback,
            "feedback_with_similarity": len(entries),
            "min_samples_required": self.config.min_feedback_samples,
            "has_sufficient_data": total_feedback >= self.config.min_feedback_samples,
            "overall_acceptance_rate": round(acceptance_rate, 3),
            "sent_count": total_sent,
            "edited_count": total_edited,
            "dismissed_count": total_dismissed,
            "current_thresholds": current,
            "bucket_count": len(bucket_stats),
            "non_empty_buckets": len([b for b in bucket_stats if b.total_count > 0]),
            "is_using_fallback": (
                self._cached_thresholds.is_fallback if self._cached_thresholds else True
            ),
        }

    def invalidate_cache(self) -> None:
        """Invalidate the cached thresholds.

        Call this after significant feedback changes to force recomputation.
        """
        with self._lock:
            self._cached_thresholds = None
            self._cache_time = 0
            logger.debug("Adaptive threshold cache invalidated")


# =============================================================================
# Singleton Access
# =============================================================================

_manager: AdaptiveThresholdManager | None = None
_manager_lock = threading.Lock()


def get_adaptive_threshold_manager() -> AdaptiveThresholdManager:
    """Get the singleton AdaptiveThresholdManager instance.

    Returns:
        Shared AdaptiveThresholdManager instance.
    """
    global _manager

    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = AdaptiveThresholdManager()

    return _manager


def reset_adaptive_threshold_manager() -> None:
    """Reset the singleton AdaptiveThresholdManager.

    Useful for testing or config changes.
    """
    global _manager

    with _manager_lock:
        _manager = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data classes
    "SimilarityBucketStats",
    "AdaptedThresholds",
    # Main class
    "AdaptiveThresholdManager",
    # Singleton functions
    "get_adaptive_threshold_manager",
    "reset_adaptive_threshold_manager",
]
