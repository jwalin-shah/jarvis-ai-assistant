"""Unit tests for the circuit breaker implementation.

Tests the CircuitBreaker class in core/health/circuit.py for state transitions,
trip conditions, and reset behavior.
"""

import threading
import time

import pytest

from core.health.circuit import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerStats,
    CircuitOpenError,
    CircuitState,
)


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_all_states_have_values(self):
        """All circuit states have string values."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"

    def test_state_count(self):
        """Circuit has exactly three states."""
        assert len(CircuitState) == 3


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig dataclass."""

    def test_default_values(self):
        """Default config has sensible values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 3
        assert config.recovery_timeout_seconds == 60.0
        assert config.half_open_max_calls == 1

    def test_custom_values(self):
        """Can create config with custom values."""
        config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout_seconds=30.0,
            half_open_max_calls=2,
        )
        assert config.failure_threshold == 5
        assert config.recovery_timeout_seconds == 30.0
        assert config.half_open_max_calls == 2


class TestCircuitBreakerStats:
    """Tests for CircuitBreakerStats dataclass."""

    def test_default_values(self):
        """Default stats start at zero."""
        stats = CircuitBreakerStats()
        assert stats.failure_count == 0
        assert stats.success_count == 0
        assert stats.last_failure_time is None
        assert stats.last_success_time is None
        assert stats.total_executions == 0
        assert stats.total_failures == 0
        assert stats.total_successes == 0


class TestCircuitBreakerInitialization:
    """Tests for CircuitBreaker initialization."""

    def test_initializes_in_closed_state(self):
        """Circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED

    def test_uses_provided_config(self):
        """Uses provided configuration."""
        config = CircuitBreakerConfig(failure_threshold=10)
        cb = CircuitBreaker("test", config=config)
        assert cb.config.failure_threshold == 10

    def test_uses_default_config_when_none_provided(self):
        """Uses default config when not provided."""
        cb = CircuitBreaker("test")
        assert cb.config.failure_threshold == 3

    def test_stores_name(self):
        """Stores the provided name."""
        cb = CircuitBreaker("my_circuit")
        assert cb.name == "my_circuit"


class TestCircuitBreakerStateTransitions:
    """Tests for circuit breaker state transitions."""

    def test_stays_closed_on_success(self):
        """Circuit stays closed on successful operations."""
        cb = CircuitBreaker("test")
        cb.record_success()
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_transitions_to_open_after_failures(self):
        """Circuit opens after reaching failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)

        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_transitions_to_half_open_after_timeout(self):
        """Circuit transitions to half-open after recovery timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=0.01)
        cb = CircuitBreaker("test", config)

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_to_closed_on_success(self):
        """Circuit closes on success in half-open state."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=0.01)
        cb = CircuitBreaker("test", config)

        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_to_open_on_failure(self):
        """Circuit reopens on failure in half-open state."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=0.01)
        cb = CircuitBreaker("test", config)

        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_failure()
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerCanExecute:
    """Tests for can_execute method."""

    def test_can_execute_when_closed(self):
        """Returns True when circuit is closed."""
        cb = CircuitBreaker("test")
        assert cb.can_execute() is True

    def test_cannot_execute_when_open(self):
        """Returns False when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=60.0)
        cb = CircuitBreaker("test", config)

        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_can_execute_limited_in_half_open(self):
        """Allows limited executions in half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout_seconds=0.01,
            half_open_max_calls=1,
        )
        cb = CircuitBreaker("test", config)

        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

        # First call should be allowed
        assert cb.can_execute() is True


class TestCircuitBreakerExecute:
    """Tests for execute method."""

    def test_execute_success(self):
        """Executes function and records success."""
        cb = CircuitBreaker("test")
        result = cb.execute(lambda: "result")
        assert result == "result"
        assert cb.stats.total_successes == 1

    def test_execute_failure(self):
        """Records failure when function raises."""
        cb = CircuitBreaker("test")

        with pytest.raises(ValueError):
            cb.execute(lambda: (_ for _ in ()).throw(ValueError("test")))

        assert cb.stats.total_failures == 1

    def test_execute_raises_when_open(self):
        """Raises CircuitOpenError when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        with pytest.raises(CircuitOpenError) as exc_info:
            cb.execute(lambda: "should not run")

        assert "test" in str(exc_info.value)

    def test_execute_passes_args(self):
        """Passes arguments to the function."""
        cb = CircuitBreaker("test")
        result = cb.execute(lambda x, y: x + y, 1, 2)
        assert result == 3

    def test_execute_passes_kwargs(self):
        """Passes keyword arguments to the function."""
        cb = CircuitBreaker("test")
        result = cb.execute(lambda x, y=10: x + y, 5, y=20)
        assert result == 25


class TestCircuitBreakerReset:
    """Tests for reset method."""

    def test_reset_from_open(self):
        """Resets circuit from open to closed."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_reset_clears_stats(self):
        """Reset clears failure and success counts."""
        cb = CircuitBreaker("test")

        cb.record_success()
        cb.record_failure()
        assert cb.stats.success_count > 0 or cb.stats.failure_count > 0

        cb.reset()
        stats = cb.stats
        assert stats.failure_count == 0
        assert stats.success_count == 0


class TestCircuitBreakerStatsTracking:
    """Tests for stats tracking."""

    def test_tracks_total_executions(self):
        """Tracks total number of executions."""
        cb = CircuitBreaker("test")

        cb.record_success()
        cb.record_success()
        cb.record_failure()

        assert cb.stats.total_executions == 3

    def test_tracks_total_successes(self):
        """Tracks total successful executions."""
        cb = CircuitBreaker("test")

        cb.record_success()
        cb.record_success()

        assert cb.stats.total_successes == 2

    def test_tracks_total_failures(self):
        """Tracks total failed executions."""
        config = CircuitBreakerConfig(failure_threshold=10)
        cb = CircuitBreaker("test", config)

        cb.record_failure()
        cb.record_failure()
        cb.record_failure()

        assert cb.stats.total_failures == 3

    def test_tracks_last_failure_time(self):
        """Records timestamp of last failure."""
        config = CircuitBreakerConfig(failure_threshold=10)
        cb = CircuitBreaker("test", config)

        assert cb.stats.last_failure_time is None

        cb.record_failure()

        assert cb.stats.last_failure_time is not None

    def test_tracks_last_success_time(self):
        """Records timestamp of last success."""
        cb = CircuitBreaker("test")

        assert cb.stats.last_success_time is None

        cb.record_success()

        assert cb.stats.last_success_time is not None


class TestCircuitBreakerThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_record_operations(self):
        """Handles concurrent record operations safely."""
        config = CircuitBreakerConfig(failure_threshold=100)
        cb = CircuitBreaker("test", config)

        def record_success():
            for _ in range(50):
                cb.record_success()

        def record_failure():
            for _ in range(50):
                cb.record_failure()

        threads = [
            threading.Thread(target=record_success),
            threading.Thread(target=record_failure),
            threading.Thread(target=record_success),
            threading.Thread(target=record_failure),
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Total should equal sum of all operations
        assert cb.stats.total_executions == 200
        assert cb.stats.total_successes == 100
        assert cb.stats.total_failures == 100


class TestCircuitOpenError:
    """Tests for CircuitOpenError exception."""

    def test_is_exception(self):
        """Is a proper exception."""
        error = CircuitOpenError("test message")
        assert isinstance(error, Exception)

    def test_has_message(self):
        """Stores the error message."""
        error = CircuitOpenError("Circuit is open")
        assert "Circuit is open" in str(error)
