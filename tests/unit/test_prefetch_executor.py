"""Comprehensive tests for the prefetch executor system.

Tests cover:
- Priority queue ordering and scheduling
- Resource limits (memory, CPU, battery)
- Cache storage and retrieval after execution
- Cache eviction behavior
- Cache hit detection (skip already-cached predictions)
- Worker pool concurrency
- Error handling and retry logic
- Cancellation via stop()
- Edge cases: empty queue, duplicate predictions, rapid triggers
- Batch scheduling
- Draft deduplication by chat_id
- Cache tier assignment based on priority
- ExecutorStats tracking
- Pause/resume behavior
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from jarvis.prefetch.cache import CacheTier, MultiTierCache
from jarvis.prefetch.executor import (
    ExecutorState,
    ExecutorStats,
    PrefetchExecutor,
    PrefetchTask,
    ResourceManager,
)
from jarvis.prefetch.predictor import Prediction, PredictionPriority, PredictionType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache(tmp_path: Path) -> MultiTierCache:
    """Create an isolated multi-tier cache backed by temp storage."""
    return MultiTierCache(
        l1_maxsize=200,
        l1_max_bytes=10 * 1024 * 1024,
        l2_db_path=tmp_path / "test_cache.db",
        l3_cache_dir=tmp_path / "l3",
    )


@pytest.fixture
def executor(cache: MultiTierCache) -> PrefetchExecutor:
    """Create an executor wired to the temp cache. Stopped after test."""
    ex = PrefetchExecutor(
        cache=cache,
        max_workers=2,
        max_queue_size=50,
        tick_interval=0.05,
    )
    yield ex
    if ex._state != ExecutorState.STOPPED:
        ex.stop(timeout=2.0)


def _make_prediction(
    key: str,
    priority: PredictionPriority = PredictionPriority.MEDIUM,
    pred_type: PredictionType = PredictionType.SEARCH_RESULTS,
    confidence: float = 0.8,
    ttl: float = 300.0,
    params: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Prediction:
    """Helper to build a Prediction with sensible defaults."""
    return Prediction(
        type=pred_type,
        priority=priority,
        confidence=confidence,
        key=key,
        params=params or {},
        ttl_seconds=ttl,
        tags=tags or [],
    )


# ---------------------------------------------------------------------------
# ResourceManager
# ---------------------------------------------------------------------------


class TestResourceManagerDetailed:
    """Detailed tests for ResourceManager beyond the basics."""

    def test_low_memory_blocks_prefetch(self) -> None:
        rm = ResourceManager(memory_threshold_mb=500)
        rm._available_memory = 200 * 1024 * 1024
        rm._cpu_usage = 10.0
        rm._is_plugged_in = True
        rm._last_update = time.time()
        assert rm.can_prefetch() is False

    def test_high_cpu_blocks_prefetch(self) -> None:
        rm = ResourceManager(cpu_threshold_percent=80.0)
        rm._available_memory = 8 * 1024 * 1024 * 1024
        rm._cpu_usage = 95.0
        rm._is_plugged_in = True
        rm._last_update = time.time()
        assert rm.can_prefetch() is False

    def test_low_battery_unplugged_blocks_prefetch(self) -> None:
        rm = ResourceManager(battery_threshold=0.2)
        rm._available_memory = 8 * 1024 * 1024 * 1024
        rm._cpu_usage = 10.0
        rm._is_plugged_in = False
        rm._battery_level = 0.1
        rm._last_update = time.time()
        assert rm.can_prefetch() is False

    def test_low_battery_plugged_in_allows_prefetch(self) -> None:
        rm = ResourceManager(battery_threshold=0.2)
        rm._available_memory = 8 * 1024 * 1024 * 1024
        rm._cpu_usage = 10.0
        rm._is_plugged_in = True
        rm._battery_level = 0.1
        rm._last_update = time.time()
        assert rm.can_prefetch() is True

    def test_concurrency_reduced_under_high_cpu(self) -> None:
        rm = ResourceManager()
        rm._cpu_usage = 70.0
        rm._available_memory = 8 * 1024 * 1024 * 1024
        rm._is_plugged_in = True
        rm._last_update = time.time()
        limit = rm.get_concurrency_limit()
        assert limit >= 1

    def test_concurrency_single_worker_on_low_battery(self) -> None:
        rm = ResourceManager()
        rm._cpu_usage = 10.0
        rm._available_memory = 8 * 1024 * 1024 * 1024
        rm._is_plugged_in = False
        rm._battery_level = 0.3
        rm._last_update = time.time()
        assert rm.get_concurrency_limit() == 1

    def test_update_skipped_within_interval(self) -> None:
        rm = ResourceManager()
        rm._available_memory = 123456789
        rm._last_update = time.time()
        rm.update()
        assert rm._available_memory == 123456789


# ---------------------------------------------------------------------------
# Priority Queue Ordering
# ---------------------------------------------------------------------------


class TestPriorityQueueOrdering:
    """Verify the priority queue processes higher-priority tasks first."""

    def test_higher_priority_dequeued_first(self) -> None:
        low = PrefetchTask(
            priority=-PredictionPriority.LOW.value,
            created_at=time.time(),
            prediction=_make_prediction("low", PredictionPriority.LOW),
        )
        medium = PrefetchTask(
            priority=-PredictionPriority.MEDIUM.value,
            created_at=time.time(),
            prediction=_make_prediction("med", PredictionPriority.MEDIUM),
        )
        high = PrefetchTask(
            priority=-PredictionPriority.HIGH.value,
            created_at=time.time(),
            prediction=_make_prediction("high", PredictionPriority.HIGH),
        )
        critical = PrefetchTask(
            priority=-PredictionPriority.CRITICAL.value,
            created_at=time.time(),
            prediction=_make_prediction("crit", PredictionPriority.CRITICAL),
        )
        assert critical < high < medium < low

    def test_priority_queue_integration(self, executor: PrefetchExecutor) -> None:
        executor.start()
        executor.pause()

        for priority, key in [
            (PredictionPriority.LOW, "low"),
            (PredictionPriority.CRITICAL, "crit"),
            (PredictionPriority.MEDIUM, "med"),
            (PredictionPriority.HIGH, "high"),
            (PredictionPriority.BACKGROUND, "bg"),
        ]:
            executor.schedule(_make_prediction(key, priority))

        keys_in_priority_order = []
        while not executor._queue.empty():
            task = executor._queue.get_nowait()
            keys_in_priority_order.append(task.prediction.key)

        expected = ["crit", "high", "med", "low", "bg"]
        assert keys_in_priority_order == expected
        executor.stop()


# ---------------------------------------------------------------------------
# Cache Storage & Retrieval After Execution
# ---------------------------------------------------------------------------


class TestCacheStorageAfterExecution:
    """Handler results must be stored in cache and retrievable."""

    def test_handler_result_cached(self, executor: PrefetchExecutor, cache: MultiTierCache) -> None:
        cached_data = {"answer": 42, "prefetched": True}
        handler_done = threading.Event()

        def handler(pred: Prediction) -> dict[str, Any]:
            handler_done.set()
            return cached_data

        executor.register_handler(PredictionType.SEARCH_RESULTS, handler)
        executor.start()

        pred = _make_prediction("search:q1", PredictionPriority.HIGH)
        executor.schedule(pred)

        handler_done.wait(timeout=5.0)
        time.sleep(0.5)

        result = cache.get("search:q1")
        assert result is not None
        assert result["answer"] == 42
        executor.stop()

    def test_none_result_not_cached(
        self, executor: PrefetchExecutor, cache: MultiTierCache
    ) -> None:
        handler_done = threading.Event()

        def handler(pred: Prediction) -> None:
            handler_done.set()
            return None

        executor.register_handler(PredictionType.SEARCH_RESULTS, handler)
        executor.start()

        pred = _make_prediction("search:empty", PredictionPriority.HIGH)
        executor.schedule(pred)

        handler_done.wait(timeout=3.0)
        time.sleep(0.15)

        assert cache.get("search:empty") is None
        executor.stop()


# ---------------------------------------------------------------------------
# Cache Hit - Skip Already Cached
# ---------------------------------------------------------------------------


class TestCacheHitSkip:
    """When a prediction key is already cached, scheduling should be rejected."""

    def test_already_cached_prediction_not_scheduled(
        self, executor: PrefetchExecutor, cache: MultiTierCache
    ) -> None:
        cache.set("already:here", {"data": "old"})
        executor.start()
        pred = _make_prediction("already:here")
        result = executor.schedule(pred)
        assert result is False
        stats = executor.stats()
        assert stats["cache_hits"] >= 1
        executor.stop()

    def test_batch_skips_already_cached(
        self, executor: PrefetchExecutor, cache: MultiTierCache
    ) -> None:
        cache.set("batch:1", {"v": 1})
        cache.set("batch:3", {"v": 3})
        executor.start()
        preds = [_make_prediction(f"batch:{i}") for i in range(5)]
        count = executor.schedule_batch(preds)
        assert count == 3
        executor.stop()


# ---------------------------------------------------------------------------
# Cache Eviction
# ---------------------------------------------------------------------------


class TestCacheEviction:
    """Verify the L1 cache evicts old entries when full."""

    def test_l1_eviction_on_capacity(self, tmp_path: Path) -> None:
        small_cache = MultiTierCache(
            l1_maxsize=3,
            l2_db_path=tmp_path / "evict.db",
            l3_cache_dir=tmp_path / "l3",
        )

        for i in range(3):
            small_cache.set(f"key{i}", f"val{i}", tier=CacheTier.L1)

        small_cache.get("key0")
        small_cache.set("key3", "val3", tier=CacheTier.L1)

        assert small_cache.get("key0") is not None
        assert small_cache.get("key1") is None
        assert small_cache.get("key3") is not None


# ---------------------------------------------------------------------------
# Worker Pool Concurrency
# ---------------------------------------------------------------------------


class TestWorkerPoolConcurrency:
    """Verify that tasks run concurrently up to the worker limit."""

    def test_concurrent_execution(self, cache: MultiTierCache) -> None:
        executor = PrefetchExecutor(cache=cache, max_workers=2, tick_interval=0.05)

        in_handler = threading.Barrier(2, timeout=5.0)
        results_lock = threading.Lock()
        concurrent_count = [0]
        max_concurrent = [0]

        def slow_handler(pred: Prediction) -> dict[str, Any]:
            with results_lock:
                concurrent_count[0] += 1
                if concurrent_count[0] > max_concurrent[0]:
                    max_concurrent[0] = concurrent_count[0]
            try:
                in_handler.wait()
            except threading.BrokenBarrierError:
                pass
            with results_lock:
                concurrent_count[0] -= 1
            return {"key": pred.key}

        executor.register_handler(PredictionType.SEARCH_RESULTS, slow_handler)
        executor.start()

        executor.schedule(_make_prediction("conc:a", PredictionPriority.HIGH))
        executor.schedule(_make_prediction("conc:b", PredictionPriority.HIGH))

        time.sleep(2.0)
        assert max_concurrent[0] == 2
        executor.stop()

    def test_single_worker_serializes(self, cache: MultiTierCache) -> None:
        executor = PrefetchExecutor(cache=cache, max_workers=1, tick_interval=0.05)

        results_lock = threading.Lock()
        max_concurrent = [0]
        current_concurrent = [0]

        def handler(pred: Prediction) -> dict[str, Any]:
            with results_lock:
                current_concurrent[0] += 1
                if current_concurrent[0] > max_concurrent[0]:
                    max_concurrent[0] = current_concurrent[0]
            time.sleep(0.1)
            with results_lock:
                current_concurrent[0] -= 1
            return {"key": pred.key}

        executor.register_handler(PredictionType.SEARCH_RESULTS, handler)
        executor.start()

        for i in range(3):
            executor.schedule(_make_prediction(f"serial:{i}", PredictionPriority.HIGH))

        time.sleep(1.5)
        assert max_concurrent[0] == 1
        executor.stop()


# ---------------------------------------------------------------------------
# Error Handling & Retry
# ---------------------------------------------------------------------------


class TestErrorHandlingAndRetry:
    """Handlers that raise exceptions should not crash the executor."""

    def test_failing_handler_does_not_crash_executor(self, executor: PrefetchExecutor) -> None:
        call_count = [0]

        def failing_handler(pred: Prediction) -> dict[str, Any]:
            call_count[0] += 1
            raise RuntimeError("Boom!")

        executor.register_handler(PredictionType.SEARCH_RESULTS, failing_handler)
        executor.start()

        pred = _make_prediction("fail:test", PredictionPriority.HIGH)
        executor.schedule(pred)

        time.sleep(1.0)
        assert executor._state == ExecutorState.RUNNING
        assert call_count[0] >= 1
        executor.stop()

    def test_retry_up_to_max_retries(self, executor: PrefetchExecutor) -> None:
        call_count = [0]

        def counting_failure(pred: Prediction) -> dict[str, Any]:
            call_count[0] += 1
            raise ValueError("Retry me")

        executor.register_handler(PredictionType.SEARCH_RESULTS, counting_failure)
        executor.start()

        pred = _make_prediction("retry:test", PredictionPriority.HIGH)
        executor.schedule(pred)

        time.sleep(2.0)
        assert call_count[0] >= 2
        assert call_count[0] <= 3

        stats = executor.stats()
        assert stats["failed"] >= 1
        executor.stop()

    def test_failed_after_max_retries_increments_failed_stat(
        self, executor: PrefetchExecutor
    ) -> None:
        def always_fails(pred: Prediction) -> dict[str, Any]:
            raise Exception("permanent failure")

        executor.register_handler(PredictionType.SEARCH_RESULTS, always_fails)
        executor.start()

        pred = _make_prediction("perm_fail:test", PredictionPriority.HIGH)
        executor.schedule(pred)

        time.sleep(2.0)
        stats = executor.stats()
        assert stats["failed"] >= 1
        executor.stop()


# ---------------------------------------------------------------------------
# Cancellation / Stop
# ---------------------------------------------------------------------------


class TestCancellation:
    """Stopping the executor should clean up pending tasks."""

    def test_stop_clears_queue(self, executor: PrefetchExecutor) -> None:
        executor.start()
        executor.pause()

        for i in range(10):
            executor.schedule(_make_prediction(f"cancel:{i}"))

        assert not executor._queue.empty()
        executor.stop()
        assert executor._queue.empty()
        assert executor._state == ExecutorState.STOPPED

    def test_stop_idempotent(self, executor: PrefetchExecutor) -> None:
        executor.start()
        executor.stop()
        executor.stop()
        assert executor._state == ExecutorState.STOPPED

    def test_start_idempotent(self, executor: PrefetchExecutor) -> None:
        executor.start()
        executor.start()
        assert executor._state == ExecutorState.RUNNING
        executor.stop()

    def test_schedule_after_stop_rejected(self, executor: PrefetchExecutor) -> None:
        executor.start()
        executor.stop()
        pred = _make_prediction("after_stop")
        assert executor.schedule(pred) is False


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty queue, duplicates, rapid triggers, etc."""

    def test_schedule_when_not_started(self, executor: PrefetchExecutor) -> None:
        pred = _make_prediction("not_started")
        result = executor.schedule(pred)
        assert result is False

    def test_duplicate_key_in_active_tasks_rejected(self, executor: PrefetchExecutor) -> None:
        gate = threading.Event()

        def blocking_handler(pred: Prediction) -> dict[str, Any]:
            gate.wait(timeout=5.0)
            return {"done": True}

        executor.register_handler(PredictionType.SEARCH_RESULTS, blocking_handler)
        executor.start()

        pred1 = _make_prediction("dupe:key", PredictionPriority.HIGH)
        assert executor.schedule(pred1) is True
        time.sleep(0.3)

        pred2 = _make_prediction("dupe:key", PredictionPriority.HIGH)
        assert executor.schedule(pred2) is False

        gate.set()
        executor.stop()

    def test_duplicate_draft_for_same_chat_rejected(self, executor: PrefetchExecutor) -> None:
        gate = threading.Event()

        def blocking_handler(pred: Prediction) -> dict[str, Any]:
            gate.wait(timeout=5.0)
            return {"draft": True}

        executor.register_handler(PredictionType.DRAFT_REPLY, blocking_handler)
        executor.start()

        pred1 = _make_prediction(
            "draft:a:chat42",
            PredictionPriority.HIGH,
            pred_type=PredictionType.DRAFT_REPLY,
            params={"chat_id": "chat42"},
        )
        assert executor.schedule(pred1) is True
        time.sleep(0.3)

        pred2 = _make_prediction(
            "draft:b:chat42",
            PredictionPriority.MEDIUM,
            pred_type=PredictionType.DRAFT_REPLY,
            params={"chat_id": "chat42"},
        )
        assert executor.schedule(pred2) is False

        gate.set()
        executor.stop()

    def test_draft_for_same_chat_allowed_after_completion(
        self, executor: PrefetchExecutor, cache: MultiTierCache
    ) -> None:
        handler_done = threading.Event()

        def fast_handler(pred: Prediction) -> dict[str, Any]:
            handler_done.set()
            return {"text": "hello"}

        executor.register_handler(PredictionType.DRAFT_REPLY, fast_handler)
        executor.start()

        pred1 = _make_prediction(
            "draft:first:chatX",
            PredictionPriority.HIGH,
            pred_type=PredictionType.DRAFT_REPLY,
            params={"chat_id": "chatX"},
        )
        assert executor.schedule(pred1) is True

        handler_done.wait(timeout=3.0)
        time.sleep(0.2)

        pred2 = _make_prediction(
            "draft:second:chatX",
            PredictionPriority.MEDIUM,
            pred_type=PredictionType.DRAFT_REPLY,
            params={"chat_id": "chatX"},
        )
        assert executor.schedule(pred2) is True
        executor.stop()

    def test_rapid_successive_schedules(
        self, executor: PrefetchExecutor, cache: MultiTierCache
    ) -> None:
        completed = threading.Event()
        count = [0]

        def handler(pred: Prediction) -> dict[str, Any]:
            count[0] += 1
            if count[0] >= 20:
                completed.set()
            return {"i": count[0]}

        executor.register_handler(PredictionType.SEARCH_RESULTS, handler)
        executor.start()

        for i in range(20):
            executor.schedule(_make_prediction(f"rapid:{i}", PredictionPriority.HIGH))

        completed.wait(timeout=5.0)
        stats = executor.stats()
        assert stats["scheduled"] == 20
        executor.stop()

    def test_max_queue_size_overflow(self, cache: MultiTierCache) -> None:
        executor = PrefetchExecutor(cache=cache, max_queue_size=3, tick_interval=0.05)
        executor.start()
        executor.pause()

        scheduled = 0
        for i in range(10):
            if executor.schedule(_make_prediction(f"overflow:{i}")):
                scheduled += 1

        assert scheduled <= 3
        executor.stop()

    def test_empty_batch_returns_zero(self, executor: PrefetchExecutor) -> None:
        executor.start()
        assert executor.schedule_batch([]) == 0
        executor.stop()

    def test_stale_prediction_skipped(
        self, executor: PrefetchExecutor, cache: MultiTierCache
    ) -> None:
        handler_called = threading.Event()

        def handler(pred: Prediction) -> dict[str, Any]:
            handler_called.set()
            return {"data": True}

        executor.register_handler(PredictionType.SEARCH_RESULTS, handler)
        executor.start()
        executor.pause()

        pred = _make_prediction("stale:test", ttl=0.1)
        executor.schedule(pred)

        time.sleep(0.2)
        executor.resume()
        time.sleep(0.5)

        stats = executor.stats()
        assert stats["skipped"] >= 1
        executor.stop()


# ---------------------------------------------------------------------------
# Pause / Resume
# ---------------------------------------------------------------------------


class TestPauseResume:
    """Test pause/resume lifecycle."""

    def test_paused_tasks_stay_in_queue(self, executor: PrefetchExecutor) -> None:
        handler_called = threading.Event()

        def handler(pred: Prediction) -> dict[str, Any]:
            handler_called.set()
            return {"data": True}

        executor.register_handler(PredictionType.SEARCH_RESULTS, handler)
        executor.start()
        executor.pause()

        executor.schedule(_make_prediction("paused:task", PredictionPriority.HIGH))
        time.sleep(0.5)

        assert not handler_called.is_set()
        assert not executor._queue.empty()
        executor.stop()

    def test_resume_processes_queued_tasks(
        self, executor: PrefetchExecutor, cache: MultiTierCache
    ) -> None:
        handler_done = threading.Event()

        def handler(pred: Prediction) -> dict[str, Any]:
            handler_done.set()
            return {"resumed": True}

        executor.register_handler(PredictionType.SEARCH_RESULTS, handler)
        executor.start()
        executor.pause()

        executor.schedule(_make_prediction("resume:task", PredictionPriority.HIGH))
        time.sleep(0.3)

        executor.resume()
        handler_done.wait(timeout=3.0)
        time.sleep(0.15)

        assert handler_done.is_set()
        assert cache.get("resume:task") is not None
        executor.stop()

    def test_pause_noop_when_stopped(self, executor: PrefetchExecutor) -> None:
        executor.pause()
        assert executor._state == ExecutorState.STOPPED

    def test_resume_noop_when_running(self, executor: PrefetchExecutor) -> None:
        executor.start()
        executor.resume()
        assert executor._state == ExecutorState.RUNNING
        executor.stop()


# ---------------------------------------------------------------------------
# Cache Tier Assignment Based on Priority
# ---------------------------------------------------------------------------


class TestCacheTierAssignment:
    """Verify _get_cache_tier assigns tiers based on prediction priority."""

    def test_critical_goes_to_l1(self, executor: PrefetchExecutor) -> None:
        pred = _make_prediction("tier:crit", PredictionPriority.CRITICAL)
        tier = executor._get_cache_tier(pred)
        assert tier == CacheTier.L1

    def test_high_goes_to_l1(self, executor: PrefetchExecutor) -> None:
        pred = _make_prediction("tier:high", PredictionPriority.HIGH)
        tier = executor._get_cache_tier(pred)
        assert tier == CacheTier.L1

    def test_medium_goes_to_l2(self, executor: PrefetchExecutor) -> None:
        pred = _make_prediction("tier:med", PredictionPriority.MEDIUM)
        tier = executor._get_cache_tier(pred)
        assert tier == CacheTier.L2

    def test_low_goes_to_l3(self, executor: PrefetchExecutor) -> None:
        pred = _make_prediction("tier:low", PredictionPriority.LOW)
        tier = executor._get_cache_tier(pred)
        assert tier == CacheTier.L3

    def test_background_goes_to_l3(self, executor: PrefetchExecutor) -> None:
        pred = _make_prediction("tier:bg", PredictionPriority.BACKGROUND)
        tier = executor._get_cache_tier(pred)
        assert tier == CacheTier.L3


# ---------------------------------------------------------------------------
# ExecutorStats
# ---------------------------------------------------------------------------


class TestExecutorStatsDetailed:
    """Detailed stats tracking tests."""

    def test_avg_latency_single(self) -> None:
        stats = ExecutorStats()
        stats.record_execution(100, cached=True)
        assert stats.avg_latency_ms == 100.0

    def test_avg_latency_multiple(self) -> None:
        stats = ExecutorStats()
        stats.record_execution(100, cached=True)
        stats.record_execution(300, cached=False)
        assert stats.avg_latency_ms == 200.0

    def test_cached_count(self) -> None:
        stats = ExecutorStats()
        stats.record_execution(50, cached=True)
        stats.record_execution(60, cached=True)
        stats.record_execution(70, cached=False)
        assert stats.predictions_cached == 2
        assert stats.predictions_executed == 3

    def test_total_cost(self) -> None:
        stats = ExecutorStats()
        for ms in [10, 20, 30]:
            stats.record_execution(ms, cached=True)
        assert stats.total_cost_ms == 60


# ---------------------------------------------------------------------------
# Stats Reporting
# ---------------------------------------------------------------------------


class TestStatsReporting:
    """Verify executor.stats() returns complete information."""

    def test_stats_keys(self, executor: PrefetchExecutor) -> None:
        executor.start()
        stats = executor.stats()

        expected_keys = {
            "state",
            "scheduled",
            "executed",
            "cached",
            "skipped",
            "failed",
            "cache_hits",
            "total_cost_ms",
            "avg_latency_ms",
            "queue_size",
            "active_tasks",
            "workers",
            "resource_manager",
        }
        assert expected_keys.issubset(stats.keys())

        rm = stats["resource_manager"]
        assert "can_prefetch" in rm
        assert "battery" in rm
        assert "memory_available_mb" in rm
        executor.stop()

    def test_stats_reflect_scheduling(
        self, executor: PrefetchExecutor, cache: MultiTierCache
    ) -> None:
        handler_done = threading.Event()

        def handler(pred: Prediction) -> dict[str, Any]:
            handler_done.set()
            return {"v": 1}

        executor.register_handler(PredictionType.SEARCH_RESULTS, handler)
        executor.start()

        executor.schedule(_make_prediction("stats:test", PredictionPriority.HIGH))
        handler_done.wait(timeout=3.0)
        time.sleep(0.2)

        stats = executor.stats()
        assert stats["scheduled"] == 1
        assert stats["executed"] >= 1
        assert stats["cached"] >= 1
        executor.stop()


# ---------------------------------------------------------------------------
# Resource-Blocked Execution
# ---------------------------------------------------------------------------


class TestResourceBlockedExecution:
    """When resources are constrained, the executor backs off."""

    def test_executor_backs_off_when_resources_low(self, cache: MultiTierCache) -> None:
        executor = PrefetchExecutor(cache=cache, max_workers=1, tick_interval=0.05)

        handler_called = threading.Event()

        def handler(pred: Prediction) -> dict[str, Any]:
            handler_called.set()
            return {"v": 1}

        executor.register_handler(PredictionType.SEARCH_RESULTS, handler)
        executor._resource_manager._available_memory = 100 * 1024 * 1024
        executor._resource_manager._last_update = time.time() + 9999

        executor.start()
        executor.schedule(_make_prediction("blocked:task", PredictionPriority.HIGH))

        time.sleep(0.5)
        assert not handler_called.is_set()
        executor.stop()


# ---------------------------------------------------------------------------
# Batch Scheduling
# ---------------------------------------------------------------------------


class TestBatchScheduling:
    """Tests for schedule_batch()."""

    def test_batch_returns_count(self, executor: PrefetchExecutor) -> None:
        executor.start()
        preds = [_make_prediction(f"batch:{i}") for i in range(5)]
        count = executor.schedule_batch(preds)
        assert count == 5
        executor.stop()

    def test_batch_skips_active_keys(self, executor: PrefetchExecutor) -> None:
        gate = threading.Event()

        def blocking_handler(pred: Prediction) -> dict[str, Any]:
            gate.wait(timeout=5.0)
            return {"v": 1}

        executor.register_handler(PredictionType.SEARCH_RESULTS, blocking_handler)
        executor.start()

        executor.schedule(_make_prediction("active:key", PredictionPriority.HIGH))
        time.sleep(0.3)

        preds = [
            _make_prediction("active:key"),
            _make_prediction("batch:new1"),
            _make_prediction("batch:new2"),
        ]
        count = executor.schedule_batch(preds)
        assert count == 2

        gate.set()
        executor.stop()

    def test_batch_when_stopped_returns_zero(self, executor: PrefetchExecutor) -> None:
        preds = [_make_prediction(f"stopped:{i}") for i in range(3)]
        count = executor.schedule_batch(preds)
        assert count == 0


# ---------------------------------------------------------------------------
# Handler Registration
# ---------------------------------------------------------------------------


class TestHandlerRegistration:
    """Tests for registering custom handlers."""

    def test_custom_handler_overrides_default(
        self, executor: PrefetchExecutor, cache: MultiTierCache
    ) -> None:
        handler_done = threading.Event()

        def custom(pred: Prediction) -> dict[str, Any]:
            handler_done.set()
            return {"custom": True}

        executor.register_handler(PredictionType.DRAFT_REPLY, custom)
        executor.start()

        pred = _make_prediction(
            "custom:draft",
            PredictionPriority.HIGH,
            pred_type=PredictionType.DRAFT_REPLY,
            params={"chat_id": "test"},
        )
        executor.schedule(pred)
        handler_done.wait(timeout=3.0)
        time.sleep(0.15)

        result = cache.get("custom:draft")
        assert result is not None
        assert result["custom"] is True
        executor.stop()

    def test_no_handler_skips_prediction(
        self, executor: PrefetchExecutor, cache: MultiTierCache
    ) -> None:
        executor._handlers.clear()
        executor.start()

        pred = _make_prediction("no_handler:test", PredictionPriority.HIGH)
        executor.schedule(pred)

        time.sleep(0.5)
        stats = executor.stats()
        assert stats["skipped"] >= 1
        assert cache.get("no_handler:test") is None
        executor.stop()


# ---------------------------------------------------------------------------
# Active Task Tracking
# ---------------------------------------------------------------------------


class TestActiveTaskTracking:
    """Verify _active_tasks and _active_drafts are cleaned up properly."""

    def test_active_tasks_cleared_after_completion(self, executor: PrefetchExecutor) -> None:
        handler_done = threading.Event()

        def handler(pred: Prediction) -> dict[str, Any]:
            handler_done.set()
            return {"v": 1}

        executor.register_handler(PredictionType.SEARCH_RESULTS, handler)
        executor.start()

        pred = _make_prediction("track:test", PredictionPriority.HIGH)
        executor.schedule(pred)

        handler_done.wait(timeout=3.0)
        time.sleep(0.2)

        with executor._lock:
            assert "track:test" not in executor._active_tasks
        executor.stop()

    def test_active_drafts_cleared_after_completion(self, executor: PrefetchExecutor) -> None:
        handler_done = threading.Event()

        def handler(pred: Prediction) -> dict[str, Any]:
            handler_done.set()
            return {"draft": True}

        executor.register_handler(PredictionType.DRAFT_REPLY, handler)
        executor.start()

        pred = _make_prediction(
            "draft:track:chatABC",
            PredictionPriority.HIGH,
            pred_type=PredictionType.DRAFT_REPLY,
            params={"chat_id": "chatABC"},
        )
        executor.schedule(pred)

        handler_done.wait(timeout=3.0)
        time.sleep(0.2)

        with executor._lock:
            assert "chatABC" not in executor._active_drafts
        executor.stop()

    def test_active_tasks_cleared_on_handler_error(self, executor: PrefetchExecutor) -> None:
        call_count = [0]

        def failing(pred: Prediction) -> dict[str, Any]:
            call_count[0] += 1
            raise RuntimeError("fail")

        executor.register_handler(PredictionType.SEARCH_RESULTS, failing)
        executor.start()

        pred = _make_prediction("err:track", PredictionPriority.HIGH)
        executor.schedule(pred)

        time.sleep(2.0)

        with executor._lock:
            assert "err:track" not in executor._active_tasks
        executor.stop()
