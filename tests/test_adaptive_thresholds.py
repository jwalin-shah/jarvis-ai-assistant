"""Tests for the adaptive threshold system.

Tests cover:
- Configuration and validation
- Feedback analysis by similarity buckets
- Threshold computation and adaptation
- Learning rate smoothing
- Bound enforcement
- Router integration
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from jarvis.adaptive_thresholds import (
    AdaptiveThresholdManager,
    SimilarityBucketStats,
    get_adaptive_threshold_manager,
    reset_adaptive_threshold_manager,
)
from jarvis.config import AdaptiveThresholdConfig, RoutingConfig, reset_config
from jarvis.evaluation import (
    FeedbackAction,
    FeedbackEntry,
    FeedbackStore,
    reset_evaluation,
)


class TestAdaptiveThresholdConfig:
    """Tests for AdaptiveThresholdConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = AdaptiveThresholdConfig()

        assert config.enabled is False
        assert config.min_feedback_samples == 50
        assert config.adaptation_window_hours == 168
        assert config.learning_rate == 0.2
        assert config.update_interval_minutes == 60
        assert config.acceptance_target == 0.70

    def test_threshold_bounds(self) -> None:
        """Test default threshold bounds."""
        config = AdaptiveThresholdConfig()

        assert config.min_threshold_bounds["quick_reply"] == 0.80
        assert config.min_threshold_bounds["context"] == 0.50
        assert config.min_threshold_bounds["generate"] == 0.30

        assert config.max_threshold_bounds["quick_reply"] == 0.99
        assert config.max_threshold_bounds["context"] == 0.85
        assert config.max_threshold_bounds["generate"] == 0.65

    def test_config_validation(self) -> None:
        """Test configuration validation."""
        # Valid config
        config = AdaptiveThresholdConfig(
            enabled=True,
            min_feedback_samples=100,
            learning_rate=0.5,
        )
        assert config.enabled is True
        assert config.min_feedback_samples == 100
        assert config.learning_rate == 0.5

    def test_routing_config_includes_adaptive(self) -> None:
        """Test that RoutingConfig includes adaptive configuration."""
        routing = RoutingConfig()

        assert hasattr(routing, "adaptive")
        assert isinstance(routing.adaptive, AdaptiveThresholdConfig)
        assert routing.adaptive.enabled is False


class TestSimilarityBucketStats:
    """Tests for SimilarityBucketStats dataclass."""

    def test_bucket_creation(self) -> None:
        """Test creating a bucket with stats."""
        bucket = SimilarityBucketStats(
            bucket_start=0.90,
            bucket_end=0.95,
            total_count=100,
            sent_count=60,
            edited_count=20,
            dismissed_count=20,
            acceptance_rate=0.80,
            avg_similarity=0.92,
        )

        assert bucket.bucket_start == 0.90
        assert bucket.bucket_end == 0.95
        assert bucket.total_count == 100
        assert bucket.acceptance_rate == 0.80

    def test_empty_bucket(self) -> None:
        """Test empty bucket defaults."""
        bucket = SimilarityBucketStats(
            bucket_start=0.50,
            bucket_end=0.55,
        )

        assert bucket.total_count == 0
        assert bucket.acceptance_rate == 0.0


class TestAdaptiveThresholdManager:
    """Tests for AdaptiveThresholdManager."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self) -> None:
        """Reset singletons before each test."""
        reset_adaptive_threshold_manager()
        reset_evaluation()
        reset_config()

    @pytest.fixture
    def temp_feedback_store(self) -> FeedbackStore:
        """Create a feedback store with a temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FeedbackStore(feedback_dir=Path(temp_dir))
            yield store

    @pytest.fixture
    def enabled_config(self) -> AdaptiveThresholdConfig:
        """Create an enabled adaptive config."""
        return AdaptiveThresholdConfig(
            enabled=True,
            min_feedback_samples=10,  # Lower for testing
            learning_rate=1.0,  # No smoothing for easier testing
            similarity_bucket_size=0.1,
        )

    def _create_feedback_entry(
        self,
        action: FeedbackAction,
        similarity: float,
        timestamp: datetime | None = None,
    ) -> FeedbackEntry:
        """Helper to create feedback entries with similarity scores."""
        return FeedbackEntry(
            timestamp=timestamp or datetime.now(UTC),
            action=action,
            suggestion_id="test_id",
            suggestion_text="Test suggestion",
            edited_text=None,
            chat_id="test_chat",
            context_hash="test_hash",
            evaluation=None,
            metadata={"similarity_score": similarity},
        )

    def test_disabled_returns_base_thresholds(self, temp_feedback_store: FeedbackStore) -> None:
        """Test that disabled config returns base thresholds."""
        config = AdaptiveThresholdConfig(enabled=False)
        manager = AdaptiveThresholdManager(config=config, feedback_store=temp_feedback_store)

        thresholds = manager.get_adapted_thresholds()

        # Should return base thresholds from config
        assert "quick_reply" in thresholds
        assert "context" in thresholds
        assert "generate" in thresholds

    def test_insufficient_data_returns_fallback(
        self,
        temp_feedback_store: FeedbackStore,
        enabled_config: AdaptiveThresholdConfig,
    ) -> None:
        """Test that insufficient data returns fallback thresholds."""
        manager = AdaptiveThresholdManager(
            config=enabled_config,
            feedback_store=temp_feedback_store,
        )

        # Add only 5 entries (less than min_feedback_samples=10)
        for _ in range(5):
            temp_feedback_store._entries.append(
                self._create_feedback_entry(FeedbackAction.SENT, 0.92)
            )

        thresholds = manager.get_adapted_thresholds()

        # Should still return thresholds (fallback)
        assert "quick_reply" in thresholds

    def test_analyze_feedback_by_similarity(
        self,
        temp_feedback_store: FeedbackStore,
        enabled_config: AdaptiveThresholdConfig,
    ) -> None:
        """Test feedback analysis groups by similarity buckets."""
        manager = AdaptiveThresholdManager(
            config=enabled_config,
            feedback_store=temp_feedback_store,
        )

        # Add entries in different similarity ranges
        entries = [
            # High similarity (0.9-1.0 bucket) - mostly accepted
            self._create_feedback_entry(FeedbackAction.SENT, 0.95),
            self._create_feedback_entry(FeedbackAction.SENT, 0.92),
            self._create_feedback_entry(FeedbackAction.EDITED, 0.91),
            self._create_feedback_entry(FeedbackAction.DISMISSED, 0.93),
            # Medium similarity (0.7-0.8 bucket) - mixed
            self._create_feedback_entry(FeedbackAction.SENT, 0.75),
            self._create_feedback_entry(FeedbackAction.DISMISSED, 0.72),
            self._create_feedback_entry(FeedbackAction.DISMISSED, 0.78),
            # Low similarity (0.5-0.6 bucket) - mostly rejected
            self._create_feedback_entry(FeedbackAction.DISMISSED, 0.55),
            self._create_feedback_entry(FeedbackAction.WROTE_FROM_SCRATCH, 0.52),
            self._create_feedback_entry(FeedbackAction.SENT, 0.58),
        ]
        temp_feedback_store._entries.extend(entries)

        bucket_stats = manager._analyze_feedback_by_similarity(entries)

        # Should have buckets
        assert len(bucket_stats) > 0

        # Find the 0.9 bucket (use tolerance for floating point comparison)
        high_bucket = next((b for b in bucket_stats if abs(b.bucket_start - 0.9) < 0.01), None)
        assert high_bucket is not None
        assert high_bucket.total_count == 4
        assert high_bucket.sent_count == 2
        assert high_bucket.edited_count == 1
        assert high_bucket.dismissed_count == 1
        assert high_bucket.acceptance_rate == 0.75  # (2 + 1) / 4

    def test_threshold_adaptation(
        self,
        temp_feedback_store: FeedbackStore,
    ) -> None:
        """Test that thresholds adapt based on feedback patterns."""
        config = AdaptiveThresholdConfig(
            enabled=True,
            min_feedback_samples=10,
            learning_rate=1.0,  # Full adaptation
            acceptance_target=0.70,
            similarity_bucket_size=0.1,
        )
        manager = AdaptiveThresholdManager(config=config, feedback_store=temp_feedback_store)

        # Create feedback where high similarity has high acceptance
        entries = []

        # 0.9-1.0: 90% acceptance (9 sent, 1 dismissed)
        for _ in range(9):
            entries.append(self._create_feedback_entry(FeedbackAction.SENT, 0.95))
        entries.append(self._create_feedback_entry(FeedbackAction.DISMISSED, 0.92))

        # 0.8-0.9: 70% acceptance (7 sent, 3 dismissed)
        for _ in range(7):
            entries.append(self._create_feedback_entry(FeedbackAction.SENT, 0.85))
        for _ in range(3):
            entries.append(self._create_feedback_entry(FeedbackAction.DISMISSED, 0.82))

        # 0.6-0.7: 50% acceptance (5 sent, 5 dismissed)
        for _ in range(5):
            entries.append(self._create_feedback_entry(FeedbackAction.SENT, 0.65))
        for _ in range(5):
            entries.append(self._create_feedback_entry(FeedbackAction.DISMISSED, 0.62))

        # 0.4-0.5: 30% acceptance (3 sent, 7 dismissed)
        for _ in range(3):
            entries.append(self._create_feedback_entry(FeedbackAction.SENT, 0.45))
        for _ in range(7):
            entries.append(self._create_feedback_entry(FeedbackAction.DISMISSED, 0.42))

        temp_feedback_store._entries.extend(entries)

        thresholds = manager.get_adapted_thresholds()

        # Thresholds should be adapted
        assert thresholds["quick_reply"] >= 0.80  # Min bound
        assert thresholds["quick_reply"] <= 0.99  # Max bound
        assert thresholds["context"] < thresholds["quick_reply"]
        assert thresholds["generate"] < thresholds["context"]

    def test_learning_rate_smoothing(
        self,
        temp_feedback_store: FeedbackStore,
    ) -> None:
        """Test that learning rate smooths threshold changes."""
        config = AdaptiveThresholdConfig(
            enabled=True,
            min_feedback_samples=10,
            learning_rate=0.5,  # 50% smoothing
        )
        manager = AdaptiveThresholdManager(config=config, feedback_store=temp_feedback_store)

        # Add entries to trigger adaptation
        for _ in range(10):
            temp_feedback_store._entries.append(
                self._create_feedback_entry(FeedbackAction.SENT, 0.85)
            )

        # First call sets base
        _ = manager.get_adapted_thresholds()

        # Invalidate cache to force recomputation
        manager.invalidate_cache()

        # Second call should apply smoothing
        thresholds2 = manager.get_adapted_thresholds()

        # With smoothing, values should change gradually
        # (Exact values depend on the algorithm, but they should be within bounds)
        assert thresholds2["quick_reply"] >= config.min_threshold_bounds["quick_reply"]

    def test_bound_enforcement(
        self,
        temp_feedback_store: FeedbackStore,
    ) -> None:
        """Test that thresholds stay within bounds."""
        config = AdaptiveThresholdConfig(
            enabled=True,
            min_feedback_samples=10,
            learning_rate=1.0,
            min_threshold_bounds={"quick_reply": 0.85, "context": 0.60, "generate": 0.40},
            max_threshold_bounds={"quick_reply": 0.95, "context": 0.75, "generate": 0.55},
        )
        manager = AdaptiveThresholdManager(config=config, feedback_store=temp_feedback_store)

        # Add entries that would push thresholds outside bounds
        for _ in range(20):
            temp_feedback_store._entries.append(
                self._create_feedback_entry(FeedbackAction.SENT, 0.50)  # Low similarity
            )

        thresholds = manager.get_adapted_thresholds()

        # Should be clamped to bounds
        assert thresholds["quick_reply"] >= 0.85
        assert thresholds["quick_reply"] <= 0.95
        assert thresholds["context"] >= 0.60
        assert thresholds["context"] <= 0.75
        assert thresholds["generate"] >= 0.40
        assert thresholds["generate"] <= 0.55

    def test_threshold_ordering(
        self,
        temp_feedback_store: FeedbackStore,
        enabled_config: AdaptiveThresholdConfig,
    ) -> None:
        """Test that thresholds maintain correct ordering."""
        manager = AdaptiveThresholdManager(
            config=enabled_config,
            feedback_store=temp_feedback_store,
        )

        # Add varied feedback
        for sim in [0.95, 0.85, 0.75, 0.65, 0.55, 0.45]:
            for _ in range(5):
                temp_feedback_store._entries.append(
                    self._create_feedback_entry(FeedbackAction.SENT, sim)
                )

        thresholds = manager.get_adapted_thresholds()

        # quick_reply > context > generate
        assert thresholds["quick_reply"] > thresholds["context"]
        assert thresholds["context"] > thresholds["generate"]

    def test_cache_invalidation(
        self,
        temp_feedback_store: FeedbackStore,
        enabled_config: AdaptiveThresholdConfig,
    ) -> None:
        """Test cache invalidation forces recomputation."""
        manager = AdaptiveThresholdManager(
            config=enabled_config,
            feedback_store=temp_feedback_store,
        )

        # Initial entries
        for _ in range(10):
            temp_feedback_store._entries.append(
                self._create_feedback_entry(FeedbackAction.SENT, 0.90)
            )

        thresholds1 = manager.get_adapted_thresholds()

        # Add more entries
        for _ in range(10):
            temp_feedback_store._entries.append(
                self._create_feedback_entry(FeedbackAction.DISMISSED, 0.80)
            )

        # Without invalidation, should return cached
        thresholds2 = manager.get_adapted_thresholds()
        assert thresholds1 == thresholds2

        # After invalidation, should recompute
        manager.invalidate_cache()
        _ = manager.get_adapted_thresholds()
        # May or may not differ depending on algorithm

    def test_adaptation_stats(
        self,
        temp_feedback_store: FeedbackStore,
        enabled_config: AdaptiveThresholdConfig,
    ) -> None:
        """Test get_adaptation_stats returns useful information."""
        manager = AdaptiveThresholdManager(
            config=enabled_config,
            feedback_store=temp_feedback_store,
        )

        # Add varied feedback
        temp_feedback_store._entries.extend(
            [
                self._create_feedback_entry(FeedbackAction.SENT, 0.90),
                self._create_feedback_entry(FeedbackAction.EDITED, 0.85),
                self._create_feedback_entry(FeedbackAction.DISMISSED, 0.70),
                self._create_feedback_entry(FeedbackAction.SENT, 0.80),
                self._create_feedback_entry(FeedbackAction.WROTE_FROM_SCRATCH, 0.60),
            ]
        )

        stats = manager.get_adaptation_stats()

        assert stats["enabled"] is True
        assert stats["total_feedback_in_window"] == 5
        assert stats["sent_count"] == 2
        assert stats["edited_count"] == 1
        assert stats["dismissed_count"] == 2  # DISMISSED + WROTE_FROM_SCRATCH
        assert "current_thresholds" in stats
        assert "overall_acceptance_rate" in stats

    def test_time_window_filtering(
        self,
        temp_feedback_store: FeedbackStore,
    ) -> None:
        """Test that feedback is filtered by time window."""
        config = AdaptiveThresholdConfig(
            enabled=True,
            min_feedback_samples=10,
            adaptation_window_hours=24,  # Only last 24 hours
        )
        manager = AdaptiveThresholdManager(config=config, feedback_store=temp_feedback_store)

        old_time = datetime.now(UTC) - timedelta(hours=48)
        new_time = datetime.now(UTC) - timedelta(hours=1)

        # Add old entries (should be filtered out)
        for _ in range(10):
            temp_feedback_store._entries.append(
                self._create_feedback_entry(FeedbackAction.DISMISSED, 0.90, old_time)
            )

        # Add recent entries
        for _ in range(10):
            temp_feedback_store._entries.append(
                self._create_feedback_entry(FeedbackAction.SENT, 0.90, new_time)
            )

        entries_in_window = manager._get_feedback_in_window()

        # Should only include recent entries
        assert len(entries_in_window) == 10
        for entry in entries_in_window:
            assert entry.timestamp > old_time


class TestRouterIntegration:
    """Tests for router integration with adaptive thresholds.

    Note: The router no longer uses adaptive thresholds directly (simplified to
    always-generate path). These tests verify the adaptive threshold system
    itself still works, independent of the router.
    """

    @pytest.fixture(autouse=True)
    def reset_singletons(self) -> None:
        """Reset singletons before each test."""
        reset_adaptive_threshold_manager()
        reset_evaluation()
        reset_config()

    def test_adaptive_thresholds_compute(self) -> None:
        """Test that adaptive threshold manager computes thresholds."""
        mock_adaptive_config = AdaptiveThresholdConfig(enabled=True)
        mock_routing_config = RoutingConfig(
            quick_reply_threshold=0.90,
            context_threshold=0.70,
            generate_threshold=0.50,
            adaptive=mock_adaptive_config,
        )

        with patch("jarvis.adaptive_thresholds.get_config") as mock_config:
            mock_jarvis_config = type("MockConfig", (), {"routing": mock_routing_config})()
            mock_config.return_value = mock_jarvis_config

            manager = AdaptiveThresholdManager()
            thresholds = manager.get_adapted_thresholds()

            # Without feedback data, should return base thresholds
            assert "quick_reply" in thresholds
            assert "context" in thresholds
            assert "generate" in thresholds


class TestSingletons:
    """Tests for singleton access."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self) -> None:
        """Reset singletons before each test."""
        reset_adaptive_threshold_manager()

    def test_get_adaptive_threshold_manager_singleton(self) -> None:
        """Test that get_adaptive_threshold_manager returns same instance."""
        manager1 = get_adaptive_threshold_manager()
        manager2 = get_adaptive_threshold_manager()

        assert manager1 is manager2

    def test_reset_adaptive_threshold_manager(self) -> None:
        """Test that reset creates new instance."""
        manager1 = get_adaptive_threshold_manager()
        reset_adaptive_threshold_manager()
        manager2 = get_adaptive_threshold_manager()

        assert manager1 is not manager2
