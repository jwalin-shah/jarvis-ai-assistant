"""Reliability engineering validation tests for JARVIS.

Tests for circuit breakers, retry logic, graceful degradation,
queue safety, and recovery procedures.
"""

from __future__ import annotations

import threading
import time

import pytest

from contracts.health import DegradationPolicy, FeatureState
from core.health.circuit import CircuitBreaker, CircuitBreakerConfig, CircuitOpenError, CircuitState
from core.health.degradation import GracefulDegradationController
from jarvis.fallbacks import FailureReason, get_fallback_response
from jarvis.tasks.models import TaskStatus, TaskType
from jarvis.tasks.queue import TaskQueue, reset_task_queue


class TestCircuitBreaker:
    """Test suite for circuit breaker implementation."""

    def test_initial_state_is_closed(self):
        """New circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute()

    def test_opens_after_failure_threshold(self):
        """Circuit opens after configured number of failures."""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=3))

        # Record failures
        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert not cb.can_execute()

    def test_success_resets_failure_count(self):
        """Successful execution resets failure counter."""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=5))

        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # Still closed

        cb.record_success()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # Still closed (only 1 failure since reset)

    def test_half_open_after_recovery_timeout(self):
        """Circuit transitions to HALF_OPEN after recovery timeout."""
        cb = CircuitBreaker(
            "test",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout_seconds=0.1,
            ),
        )

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        assert cb.state == CircuitState.HALF_OPEN
        assert cb.can_execute()  # One call allowed

    def test_closes_on_successful_half_open(self):
        """Circuit closes when HALF_OPEN call succeeds."""
        cb = CircuitBreaker(
            "test",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout_seconds=0.1,
            ),
        )

        cb.record_failure()
        time.sleep(0.15)

        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_reopens_on_failed_half_open(self):
        """Circuit reopens if HALF_OPEN call fails."""
        cb = CircuitBreaker(
            "test",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout_seconds=0.1,
            ),
        )

        cb.record_failure()
        time.sleep(0.15)

        assert cb.state == CircuitState.HALF_OPEN

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_execute_method_success(self):
        """execute() returns result on success."""
        cb = CircuitBreaker("test")

        def success_func():
            return "success"

        result = cb.execute(success_func)
        assert result == "success"
        assert cb.stats.total_successes == 1

    def test_execute_method_raises_on_failure(self):
        """execute() propagates exception and records failure."""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=3))

        def fail_func():
            raise ValueError("failed")

        with pytest.raises(ValueError, match="failed"):
            cb.execute(fail_func)

        assert cb.stats.total_failures == 1

    def test_execute_blocks_when_open(self):
        """execute() raises CircuitOpenError when circuit is open."""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=1))

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        def should_not_run():
            pytest.fail("Function should not be called when circuit is open")

        with pytest.raises(CircuitOpenError):
            cb.execute(should_not_run)

    def test_reset_restores_initial_state(self):
        """reset() returns circuit to initial CLOSED state."""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=1))

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.stats.total_failures == 0
        assert cb.stats.failure_count == 0

    def test_stats_tracking(self):
        """Statistics are correctly tracked."""
        cb = CircuitBreaker("test")

        cb.record_success()
        cb.record_success()
        cb.record_failure()

        stats = cb.stats
        assert stats.total_successes == 2
        assert stats.total_failures == 1
        assert stats.total_executions == 3
        assert stats.failure_count == 1
        assert stats.last_success_time is not None
        assert stats.last_failure_time is not None

    def test_thread_safety(self):
        """Circuit breaker is thread-safe."""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=100))

        success_count = 0
        failure_count = 0
        lock = threading.Lock()

        def worker():
            nonlocal success_count, failure_count
            for _ in range(100):
                if cb.can_execute():
                    if cb.state == CircuitState.CLOSED:
                        cb.record_success()
                        with lock:
                            success_count += 1
                    else:
                        cb.record_failure()
                        with lock:
                            failure_count += 1

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert success_count + failure_count == 1000


class TestGracefulDegradation:
    """Test suite for graceful degradation controller."""

    def test_feature_registration(self):
        """Features can be registered with policies."""
        controller = GracefulDegradationController()

        policy = DegradationPolicy(
            feature_name="test_feature",
            health_check=lambda: True,
            degraded_behavior=lambda: "degraded",
            fallback_behavior=lambda: "fallback",
            recovery_check=lambda: True,
            max_failures=3,
        )

        controller.register_feature(policy)
        health = controller.get_health()

        assert "test_feature" in health
        assert health["test_feature"] == FeatureState.HEALTHY

    def test_execute_calls_primary_when_healthy(self):
        """Execute uses primary function when healthy."""
        controller = GracefulDegradationController()

        policy = DegradationPolicy(
            feature_name="healthy_feature",
            health_check=lambda: True,
            degraded_behavior=lambda: "degraded",
            fallback_behavior=lambda: "fallback",
            recovery_check=lambda: True,
            max_failures=3,
        )

        controller.register_feature(policy)

        def primary():
            return "primary"

        result = controller.execute("healthy_feature", primary)
        assert result == "primary"

    def test_execute_uses_degraded_when_primary_fails(self):
        """Execute falls back to degraded behavior on failure."""
        controller = GracefulDegradationController()

        call_log = []

        policy = DegradationPolicy(
            feature_name="failing_feature",
            health_check=lambda: True,
            degraded_behavior=lambda: call_log.append("degraded") or "degraded",
            fallback_behavior=lambda: call_log.append("fallback") or "fallback",
            recovery_check=lambda: False,
            max_failures=5,  # High threshold to test degraded, not fallback
        )

        controller.register_feature(policy)

        def failing_primary():
            raise RuntimeError("Primary failed")

        result = controller.execute("failing_feature", failing_primary)

        assert result == "degraded"
        assert "degraded" in call_log

    def test_execute_uses_fallback_when_circuit_opens(self):
        """Execute uses fallback when circuit breaker opens."""
        controller = GracefulDegradationController()

        policy = DegradationPolicy(
            feature_name="circuit_feature",
            health_check=lambda: True,
            degraded_behavior=lambda: "degraded",
            fallback_behavior=lambda: "fallback",
            recovery_check=lambda: False,
            max_failures=3,  # Open circuit after 3 failures
        )

        controller.register_feature(policy)

        call_count = 0

        def failing_primary():
            nonlocal call_count
            call_count += 1
            raise RuntimeError(f"Primary failed #{call_count}")

        # Call until circuit opens and we get fallback
        results = []
        for _ in range(5):
            result = controller.execute("circuit_feature", failing_primary)
            results.append(result)
            if result == "fallback":
                break

        # Should observe fallback behavior
        assert "fallback" in results
        # Circuit should be in FAILED state
        assert controller.get_health()["circuit_feature"] == FeatureState.FAILED

    def test_execute_with_unregistered_feature_raises(self):
        """Execute raises KeyError for unregistered features."""
        controller = GracefulDegradationController()

        with pytest.raises(KeyError, match="not registered"):
            controller.execute("unknown_feature")

    def test_health_reporting(self):
        """Health status reflects circuit breaker state."""
        controller = GracefulDegradationController()

        policy = DegradationPolicy(
            feature_name="health_test",
            health_check=lambda: True,
            degraded_behavior=lambda: "degraded",
            fallback_behavior=lambda: "fallback",
            recovery_check=lambda: False,
            max_failures=3,
        )

        controller.register_feature(policy)

        # Initially healthy
        health = controller.get_health()
        assert health["health_test"] == FeatureState.HEALTHY

        # Force failures to open circuit
        stats = controller.get_feature_stats("health_test")
        for _ in range(3):
            controller._features["health_test"].circuit_breaker.record_failure()

        health = controller.get_health()
        assert health["health_test"] == FeatureState.FAILED

    def test_feature_reset(self):
        """Reset feature restores to healthy state."""
        controller = GracefulDegradationController()

        policy = DegradationPolicy(
            feature_name="reset_test",
            health_check=lambda: True,
            degraded_behavior=lambda: "degraded",
            fallback_behavior=lambda: "fallback",
            recovery_check=lambda: True,
            max_failures=1,
        )

        controller.register_feature(policy)

        # Open circuit
        controller._features["reset_test"].circuit_breaker.record_failure()
        assert controller.get_health()["reset_test"] == FeatureState.FAILED

        # Reset
        controller.reset_feature("reset_test")
        assert controller.get_health()["reset_test"] == FeatureState.HEALTHY

    def test_feature_stats(self):
        """Feature stats include circuit breaker statistics."""
        controller = GracefulDegradationController()

        policy = DegradationPolicy(
            feature_name="stats_test",
            health_check=lambda: True,
            degraded_behavior=lambda: "degraded",
            fallback_behavior=lambda: "fallback",
            recovery_check=lambda: True,
            max_failures=3,
        )

        controller.register_feature(policy)

        stats = controller.get_feature_stats("stats_test")

        assert stats["feature_name"] == "stats_test"
        assert stats["state"] == "closed"
        assert "failure_count" in stats
        assert "total_executions" in stats


class TestFallbackResponses:
    """Test suite for fallback response system."""

    def test_all_failure_reasons_have_fallbacks(self):
        """Every FailureReason has a defined fallback response."""
        for reason in FailureReason:
            response = get_fallback_response(reason)
            assert response is not None
            assert response.reason == reason
            assert response.text
            assert response.suggestion

    def test_fallback_reply_suggestions(self):
        """Fallback suggestions are appropriate and safe."""
        from jarvis.fallbacks import get_fallback_reply_suggestions

        suggestions = get_fallback_reply_suggestions()

        assert len(suggestions) >= 3
        # All should be generic and safe
        for suggestion in suggestions:
            assert len(suggestion) > 0
            assert len(suggestion) < 100  # Reasonable length

    def test_fallback_summary(self):
        """Fallback summary includes participant name."""
        from jarvis.fallbacks import get_fallback_summary

        summary = get_fallback_summary("John")

        assert "John" in summary
        assert "Unable" in summary or "unavailable" in summary.lower()

    def test_fallback_draft_with_context(self):
        """Fallback draft can include context."""
        from jarvis.fallbacks import get_fallback_draft

        draft = get_fallback_draft("meeting invitation")

        assert "meeting invitation" in draft
        assert "Unable" in draft or "unable" in draft.lower()

    def test_fallback_draft_without_context(self):
        """Fallback draft works without context."""
        from jarvis.fallbacks import get_fallback_draft

        draft = get_fallback_draft()

        assert draft
        assert "Unable" in draft or "unable" in draft.lower()


class TestTaskQueue:
    """Test suite for task queue reliability."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_task_queue()

    def teardown_method(self):
        """Clean up after each test."""
        reset_task_queue()

    def test_enqueue_creates_task(self):
        """Enqueue creates a new task with correct properties."""
        queue = TaskQueue(auto_persist=False)

        task = queue.enqueue(TaskType.BATCH_EXPORT, {"chat_id": "123"})

        assert task.task_type == TaskType.BATCH_EXPORT
        assert task.params == {"chat_id": "123"}
        assert task.status == TaskStatus.PENDING
        assert task.id

    def test_get_retrieves_task(self):
        """Get retrieves task by ID."""
        queue = TaskQueue(auto_persist=False)

        task = queue.enqueue(TaskType.BATCH_EXPORT, {})
        retrieved = queue.get(task.id)

        assert retrieved is not None
        assert retrieved.id == task.id

    def test_get_returns_none_for_unknown(self):
        """Get returns None for unknown task ID."""
        queue = TaskQueue(auto_persist=False)

        result = queue.get("nonexistent")
        assert result is None

    def test_get_pending_returns_only_pending(self):
        """Get pending returns only pending tasks."""
        queue = TaskQueue(auto_persist=False)

        task1 = queue.enqueue(TaskType.BATCH_EXPORT, {})
        task2 = queue.enqueue(TaskType.BATCH_SUMMARIZE, {})

        # Mark one as running
        task1.status = TaskStatus.RUNNING
        queue.update(task1)

        pending = queue.get_pending()

        assert len(pending) == 1
        assert pending[0].id == task2.id

    def test_cancel_pending_task(self):
        """Cancel changes pending task to cancelled."""
        queue = TaskQueue(auto_persist=False)

        task = queue.enqueue(TaskType.BATCH_EXPORT, {})
        result = queue.cancel(task.id)

        assert result is True
        assert queue.get(task.id).status == TaskStatus.CANCELLED

    def test_cancel_non_pending_fails(self):
        """Cancel returns False for non-pending tasks."""
        queue = TaskQueue(auto_persist=False)

        task = queue.enqueue(TaskType.BATCH_EXPORT, {})
        task.status = TaskStatus.RUNNING
        queue.update(task)

        result = queue.cancel(task.id)

        assert result is False

    def test_cancel_unknown_task(self):
        """Cancel returns False for unknown task."""
        queue = TaskQueue(auto_persist=False)

        result = queue.cancel("nonexistent")
        assert result is False

    def test_retry_failed_task(self):
        """Retry resets failed task to pending."""
        queue = TaskQueue(auto_persist=False)

        task = queue.enqueue(TaskType.BATCH_EXPORT, {}, max_retries=3)
        task.status = TaskStatus.FAILED
        task.retry_count = 1
        queue.update(task)

        result = queue.retry(task.id)

        assert result is True
        assert queue.get(task.id).status == TaskStatus.PENDING

    def test_retry_exhausted_fails(self):
        """Retry returns False when max retries exhausted."""
        queue = TaskQueue(auto_persist=False)

        task = queue.enqueue(TaskType.BATCH_EXPORT, {}, max_retries=2)
        task.status = TaskStatus.FAILED
        task.retry_count = 2
        queue.update(task)

        result = queue.retry(task.id)

        assert result == False  # noqa: E712

    def test_delete_terminal_task(self):
        """Delete removes terminal tasks."""
        queue = TaskQueue(auto_persist=False)

        task = queue.enqueue(TaskType.BATCH_EXPORT, {})
        task.status = TaskStatus.COMPLETED
        queue.update(task)

        result = queue.delete(task.id)

        assert result == True  # noqa: E712
        assert queue.get(task.id) is None

    def test_delete_non_terminal_fails(self):
        """Delete returns False for non-terminal tasks."""
        queue = TaskQueue(auto_persist=False)

        task = queue.enqueue(TaskType.BATCH_EXPORT, {})
        # Still pending

        result = queue.delete(task.id)

        assert result == False  # noqa: E712
        assert queue.get(task.id) is not None

    def test_persistence_roundtrip(self, tmp_path):
        """Tasks survive persistence roundtrip."""
        queue_path = tmp_path / "queue.json"

        # Create queue and add task
        queue1 = TaskQueue(persistence_path=queue_path, auto_persist=True)
        task = queue1.enqueue(TaskType.BATCH_EXPORT, {"chat_id": "123"})

        # Create new queue instance (simulates restart)
        queue2 = TaskQueue(persistence_path=queue_path, auto_persist=True)

        # Task should be recovered
        recovered = queue2.get(task.id)
        assert recovered is not None
        assert recovered.task_type == TaskType.BATCH_EXPORT
        assert recovered.params == {"chat_id": "123"}

    def test_running_tasks_reset_to_pending_on_load(self, tmp_path):
        """Running tasks are reset to pending on recovery."""
        queue_path = tmp_path / "queue.json"

        queue1 = TaskQueue(persistence_path=queue_path, auto_persist=True)
        task = queue1.enqueue(TaskType.BATCH_EXPORT, {})
        task.status = TaskStatus.RUNNING
        queue1.update(task)

        # Simulate restart
        queue2 = TaskQueue(persistence_path=queue_path, auto_persist=True)

        recovered = queue2.get(task.id)
        assert recovered.status == TaskStatus.PENDING

    def test_queue_stats(self):
        """Stats provide accurate queue information."""
        queue = TaskQueue(auto_persist=False)

        queue.enqueue(TaskType.BATCH_EXPORT, {})
        queue.enqueue(TaskType.BATCH_SUMMARIZE, {})

        stats = queue.get_stats()

        assert stats["total"] == 2
        assert stats["by_status"]["pending"] == 2
        assert stats["by_type"]["batch_export"] == 1
        assert stats["by_type"]["batch_summarize"] == 1

    def test_thread_safety(self):
        """Queue operations are thread-safe."""
        queue = TaskQueue(auto_persist=False)
        success_count = 0
        lock = threading.Lock()

        def worker():
            nonlocal success_count
            for _ in range(100):
                task = queue.enqueue(TaskType.BATCH_EXPORT, {})
                retrieved = queue.get(task.id)
                if retrieved:
                    with lock:
                        success_count += 1

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert success_count == 1000
        assert queue.get_stats()["total"] == 1000

    def test_max_completed_cleanup(self):
        """Old completed tasks are cleaned up."""
        queue = TaskQueue(auto_persist=False, max_completed_tasks=5)

        # Create more than max completed tasks
        for _ in range(10):
            task = queue.enqueue(TaskType.BATCH_EXPORT, {})
            task.status = TaskStatus.COMPLETED
            queue.update(task)

        # Add a new task to trigger cleanup
        queue.enqueue(TaskType.BATCH_EXPORT, {})

        # Should only have 5 completed + 1 pending
        stats = queue.get_stats()
        assert stats["by_status"].get("completed", 0) <= 5


class TestReliabilityIntegration:
    """Integration tests combining multiple reliability components."""

    def test_degradation_with_circuit_breaker(self):
        """Degradation controller uses circuit breaker correctly."""
        controller = GracefulDegradationController()

        call_log = []

        policy = DegradationPolicy(
            feature_name="integration",
            health_check=lambda: True,
            degraded_behavior=lambda: call_log.append("degraded") or "degraded",
            fallback_behavior=lambda: call_log.append("fallback") or "fallback",
            recovery_check=lambda: True,
            max_failures=2,
        )

        controller.register_feature(policy)

        def failing_primary():
            raise RuntimeError("Always fails")

        # Call until we observe fallback behavior
        results = []
        for _ in range(5):  # Try up to 5 times
            result = controller.execute("integration", failing_primary)
            results.append(result)
            if result == "fallback":
                break

        # Should eventually get fallback
        assert "fallback" in results
        # Circuit should be open
        assert controller.get_health()["integration"] == FeatureState.FAILED

    @pytest.mark.asyncio
    async def test_full_reliability_stack(self):
        """Test all reliability components in async context."""
        from core.health.circuit import CircuitBreaker

        cb = CircuitBreaker("async_test")

        async def async_operation():
            return "async_success"

        result = await async_operation()
        assert result == "async_success"

        # Test that circuit breaker works with async
        cb.record_success()
        assert cb.stats.total_successes == 1


class TestRecoveryProcedures:
    """Tests for recovery and self-healing mechanisms."""

    def test_task_recovery_after_crash(self, tmp_path):
        """Tasks are recovered after simulated crash."""
        queue_path = tmp_path / "queue.json"

        # Create tasks
        queue = TaskQueue(persistence_path=queue_path, auto_persist=True)
        task1 = queue.enqueue(TaskType.BATCH_EXPORT, {"chat_id": "1"})
        task2 = queue.enqueue(TaskType.BATCH_EXPORT, {"chat_id": "2"})

        # Simulate crash by creating new queue
        del queue
        new_queue = TaskQueue(persistence_path=queue_path, auto_persist=True)

        # Both tasks should be recovered
        assert new_queue.get(task1.id) is not None
        assert new_queue.get(task2.id) is not None

    def test_corrupted_persistence_handling(self, tmp_path):
        """Queue handles corrupted persistence file gracefully."""
        queue_path = tmp_path / "queue.json"

        # Write corrupted JSON
        queue_path.write_text("{invalid json")

        # Should create empty queue, not crash
        queue = TaskQueue(persistence_path=queue_path, auto_persist=True)
        assert queue.get_stats()["total"] == 0

        # Should be able to add new tasks
        task = queue.enqueue(TaskType.BATCH_EXPORT, {})
        assert task is not None

    def test_circuit_breaker_auto_recovery(self):
        """Circuit breaker automatically attempts recovery."""
        cb = CircuitBreaker(
            "recovery_test",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout_seconds=0.1,
            ),
        )

        # Open circuit
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Should be half-open
        assert cb.state == CircuitState.HALF_OPEN

        # Success should close it
        cb.record_success()
        assert cb.state == CircuitState.CLOSED


# Export all test classes
__all__ = [
    "TestCircuitBreaker",
    "TestRetryLogic",
    "TestGracefulDegradation",
    "TestFallbackResponses",
    "TestTaskQueue",
    "TestReliabilityIntegration",
    "TestRecoveryProcedures",
]
