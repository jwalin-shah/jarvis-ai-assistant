"""Tests for the background prefetch executor."""

import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from jarvis.prefetch.cache import MultiTierCache
from jarvis.prefetch.executor import (
    ExecutorState,
    ExecutorStats,
    PrefetchExecutor,
    PrefetchTask,
    ResourceManager,
)
from jarvis.prefetch.predictor import Prediction, PredictionPriority, PredictionType


class TestResourceManager:
    """Tests for ResourceManager."""

    def test_can_prefetch_default(self) -> None:
        """Test can_prefetch with default values."""
        rm = ResourceManager()
        # With default high memory and low CPU, should allow prefetch
        assert rm.can_prefetch()

    def test_get_concurrency_limit(self) -> None:
        """Test concurrency limit calculation."""
        rm = ResourceManager()
        limit = rm.get_concurrency_limit()
        assert limit >= 1

    def test_battery_level(self) -> None:
        """Test battery level property."""
        rm = ResourceManager()
        assert 0 <= rm.battery_level <= 1

    def test_available_memory(self) -> None:
        """Test available memory property."""
        rm = ResourceManager()
        assert rm.available_memory_mb > 0


class TestPrefetchTask:
    """Tests for PrefetchTask."""

    def test_task_ordering(self) -> None:
        """Test tasks are ordered by priority."""
        pred1 = Prediction(
            type=PredictionType.DRAFT_REPLY,
            priority=PredictionPriority.LOW,
            confidence=0.8,
            key="low",
        )
        pred2 = Prediction(
            type=PredictionType.DRAFT_REPLY,
            priority=PredictionPriority.HIGH,
            confidence=0.8,
            key="high",
        )

        task1 = PrefetchTask(
            priority=-pred1.priority.value,
            created_at=time.time(),
            prediction=pred1,
        )
        task2 = PrefetchTask(
            priority=-pred2.priority.value,
            created_at=time.time(),
            prediction=pred2,
        )

        # Higher priority (more negative value) should sort first
        assert task2 < task1


class TestExecutorStats:
    """Tests for ExecutorStats."""

    def test_record_execution(self) -> None:
        """Test recording execution metrics."""
        stats = ExecutorStats()
        stats.record_execution(100, cached=True)
        stats.record_execution(200, cached=False)

        assert stats.predictions_executed == 2
        assert stats.predictions_cached == 1
        assert stats.total_cost_ms == 300
        assert stats.avg_latency_ms == 150.0


class TestPrefetchExecutor:
    """Tests for PrefetchExecutor."""

    @pytest.fixture
    def cache(self, tmp_path: Path) -> MultiTierCache:
        """Create a cache for testing."""
        return MultiTierCache(
            l1_maxsize=100,
            l2_db_path=tmp_path / "test_cache.db",
            l3_cache_dir=tmp_path / "l3",
        )

    @pytest.fixture
    def executor(self, cache: MultiTierCache) -> PrefetchExecutor:
        """Create an executor for testing."""
        return PrefetchExecutor(
            cache=cache,
            max_workers=2,
            max_queue_size=50,
        )

    def test_start_stop(self, executor: PrefetchExecutor) -> None:
        """Test starting and stopping executor."""
        assert executor._state == ExecutorState.STOPPED

        executor.start()
        assert executor._state == ExecutorState.RUNNING

        executor.stop()
        assert executor._state == ExecutorState.STOPPED

    def test_pause_resume(self, executor: PrefetchExecutor) -> None:
        """Test pausing and resuming executor."""
        executor.start()

        executor.pause()
        assert executor._state == ExecutorState.PAUSED

        executor.resume()
        assert executor._state == ExecutorState.RUNNING

        executor.stop()

    def test_schedule_prediction(self, executor: PrefetchExecutor) -> None:
        """Test scheduling a prediction."""
        executor.start()

        pred = Prediction(
            type=PredictionType.MODEL_WARM,
            priority=PredictionPriority.LOW,
            confidence=0.8,
            key="test_pred",
            params={"model_type": "test"},
        )

        scheduled = executor.schedule(pred)
        assert scheduled

        stats = executor.stats()
        assert stats["scheduled"] == 1

        executor.stop()

    def test_schedule_duplicate_rejected(self, executor: PrefetchExecutor, cache: MultiTierCache) -> None:
        """Test that duplicate predictions are rejected."""
        executor.start()

        # Pre-populate cache
        cache.set("existing_key", {"data": "value"})

        pred = Prediction(
            type=PredictionType.MODEL_WARM,
            priority=PredictionPriority.LOW,
            confidence=0.8,
            key="existing_key",
        )

        scheduled = executor.schedule(pred)
        assert not scheduled  # Should be rejected (already in cache)

        executor.stop()

    def test_schedule_batch(self, executor: PrefetchExecutor) -> None:
        """Test batch scheduling."""
        executor.start()

        predictions = [
            Prediction(
                type=PredictionType.MODEL_WARM,
                priority=PredictionPriority.LOW,
                confidence=0.8,
                key=f"batch_{i}",
            )
            for i in range(5)
        ]

        count = executor.schedule_batch(predictions)
        assert count == 5

        executor.stop()

    def test_register_handler(self, executor: PrefetchExecutor) -> None:
        """Test registering custom handler."""
        handler_called = threading.Event()
        handler_result = {"called": True}

        def custom_handler(pred: Prediction) -> dict[str, Any]:
            handler_called.set()
            return handler_result

        executor.register_handler(PredictionType.SEARCH_RESULTS, custom_handler)
        executor.start()

        pred = Prediction(
            type=PredictionType.SEARCH_RESULTS,
            priority=PredictionPriority.HIGH,
            confidence=0.9,
            key="search_test",
            params={"query": "test"},
        )

        executor.schedule(pred)

        # Wait for handler to be called
        handler_called.wait(timeout=5.0)
        assert handler_called.is_set()

        executor.stop()

    def test_stats(self, executor: PrefetchExecutor) -> None:
        """Test stats reporting."""
        executor.start()

        stats = executor.stats()
        assert "state" in stats
        assert "scheduled" in stats
        assert "executed" in stats
        assert "queue_size" in stats
        assert "resource_manager" in stats

        executor.stop()

    def test_max_queue_size(self, cache: MultiTierCache) -> None:
        """Test queue size limit is respected."""
        executor = PrefetchExecutor(
            cache=cache,
            max_queue_size=5,
        )
        executor.start()
        executor.pause()  # Pause to prevent processing

        scheduled = 0
        for i in range(10):
            pred = Prediction(
                type=PredictionType.MODEL_WARM,
                priority=PredictionPriority.LOW,
                confidence=0.8,
                key=f"overflow_{i}",
            )
            if executor.schedule(pred):
                scheduled += 1

        # Should have scheduled at most max_queue_size
        assert scheduled <= 5

        executor.stop()

    def test_priority_ordering(self, executor: PrefetchExecutor) -> None:
        """Test that high priority tasks are executed first."""
        executor.start()
        executor.pause()

        # Schedule low priority first
        low_pred = Prediction(
            type=PredictionType.MODEL_WARM,
            priority=PredictionPriority.LOW,
            confidence=0.8,
            key="low_priority",
        )
        executor.schedule(low_pred)

        # Then high priority
        high_pred = Prediction(
            type=PredictionType.MODEL_WARM,
            priority=PredictionPriority.HIGH,
            confidence=0.8,
            key="high_priority",
        )
        executor.schedule(high_pred)

        # Check queue - high priority should be at front
        # (Priority queue uses negative values, so higher priority = more negative)
        task = executor._queue.get_nowait()
        assert task.prediction.key == "high_priority"

        executor.stop()

    def test_executor_handles_handler_errors(self, executor: PrefetchExecutor) -> None:
        """Test executor handles handler errors gracefully."""
        def failing_handler(pred: Prediction) -> dict[str, Any]:
            raise ValueError("Handler error")

        executor.register_handler(PredictionType.SEARCH_RESULTS, failing_handler)
        executor.start()

        pred = Prediction(
            type=PredictionType.SEARCH_RESULTS,
            priority=PredictionPriority.HIGH,
            confidence=0.9,
            key="error_test",
        )

        executor.schedule(pred)
        time.sleep(0.5)  # Give time for processing

        # Should not crash, just increment failed count
        stats = executor.stats()
        # Note: May have retried, so check that it didn't crash
        assert stats["state"] == "running"

        executor.stop()


class TestIntegration:
    """Integration tests for executor with real handlers."""

    @pytest.fixture
    def cache(self, tmp_path: Path) -> MultiTierCache:
        """Create a cache for testing."""
        return MultiTierCache(
            l1_maxsize=100,
            l2_db_path=tmp_path / "test_cache.db",
            l3_cache_dir=tmp_path / "l3",
        )

    def test_custom_handler_execution(self, cache: MultiTierCache) -> None:
        """Test custom handler is called and result is cached."""
        executor = PrefetchExecutor(cache=cache)
        handler_called = threading.Event()

        def custom_handler(pred: Prediction) -> dict[str, Any]:
            handler_called.set()
            return {"result": "custom_value", "key": pred.key}

        # Register custom handler for a custom type
        executor.register_handler(PredictionType.SEARCH_RESULTS, custom_handler)
        executor.start()

        pred = Prediction(
            type=PredictionType.SEARCH_RESULTS,
            priority=PredictionPriority.HIGH,
            confidence=0.9,
            key="custom:test",
            params={"query": "test"},
        )

        executor.schedule(pred)

        # Wait for handler to be called
        handler_called.wait(timeout=5.0)
        time.sleep(0.2)  # Extra time for caching

        assert handler_called.is_set()

        # Check result was cached
        cached = cache.get("custom:test")
        assert cached is not None
        assert cached["result"] == "custom_value"

        executor.stop()

    def test_handler_result_none_not_cached(self, cache: MultiTierCache) -> None:
        """Test that None results from handlers are not cached."""
        executor = PrefetchExecutor(cache=cache)
        handler_called = threading.Event()

        def returning_none_handler(pred: Prediction) -> dict[str, Any] | None:
            handler_called.set()
            return None

        executor.register_handler(PredictionType.SEARCH_RESULTS, returning_none_handler)
        executor.start()

        pred = Prediction(
            type=PredictionType.SEARCH_RESULTS,
            priority=PredictionPriority.HIGH,
            confidence=0.9,
            key="none:test",
            params={},
        )

        executor.schedule(pred)
        handler_called.wait(timeout=5.0)
        time.sleep(0.2)

        assert handler_called.is_set()
        # None result should not be cached
        assert cache.get("none:test") is None

        executor.stop()
