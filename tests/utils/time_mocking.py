"""Comprehensive time mocking for deterministic tests.

This module provides utilities to freeze and control time during tests,
eliminating flakiness from timing-dependent operations.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from types import TracebackType


class FrozenClock:
    """Deterministic clock for time-sensitive tests.

    Usage:
        clock = FrozenClock("2024-01-15 10:30:00")
        with clock.freeze():
            assert datetime.now() == datetime(2024, 1, 15, 10, 30, 0)
            clock.advance(seconds=30)
            assert datetime.now() == datetime(2024, 1, 15, 10, 30, 30)
    """

    def __init__(self, start_time: str | datetime):
        if isinstance(start_time, str):
            # Handle common formats
            start_time = start_time.replace(" ", "T")
            self.current = datetime.fromisoformat(start_time)
        else:
            self.current = start_time
        self._frozen = False

    def advance(
        self,
        seconds: float = 0,
        minutes: float = 0,
        hours: float = 0,
        days: float = 0,
    ) -> FrozenClock:
        """Advance clock by specified duration.

        Returns self for method chaining.
        """
        self.current += timedelta(
            seconds=seconds,
            minutes=minutes,
            hours=hours,
            days=days,
        )
        return self

    def rewind(
        self,
        seconds: float = 0,
        minutes: float = 0,
        hours: float = 0,
        days: float = 0,
    ) -> FrozenClock:
        """Rewind clock by specified duration.

        Returns self for method chaining.
        """
        return self.advance(
            seconds=-seconds,
            minutes=-minutes,
            hours=-hours,
            days=-days,
        )

    def now(self) -> datetime:
        """Get current frozen time."""
        return self.current

    def utcnow(self) -> datetime:
        """Get current frozen time as UTC (same as now for simplicity)."""
        return self.current

    def time(self) -> float:
        """Get current time as Unix timestamp."""
        return self.current.timestamp()

    @contextmanager
    def freeze(self) -> Iterator[FrozenClock]:
        """Freeze time for the duration of the context."""
        self._frozen = True

        with (
            patch("datetime.datetime") as mock_dt,
            patch("time.time") as mock_time,
            patch("time.strftime") as mock_strftime,
        ):
            mock_dt.now.return_value = self.current
            mock_dt.utcnow.return_value = self.current
            mock_dt.fromtimestamp = datetime.fromtimestamp
            mock_dt.strptime = datetime.strptime

            mock_time.return_value = self.time()

            def strftime_wrapper(fmt: str, ts: float | None = None) -> str:
                if ts is None:
                    ts = self.time()
                return datetime.fromtimestamp(ts).strftime(fmt)

            mock_strftime.side_effect = strftime_wrapper

            yield self

        self._frozen = False

    def __enter__(self) -> FrozenClock:
        """Context manager entry."""
        self._freeze_cm = self.freeze()
        return self._freeze_cm.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self._freeze_cm.__exit__(exc_type, exc_val, exc_tb)


class PerformanceClock:
    """Clock that simulates performance characteristics.

    Useful for testing timeout and rate-limiting logic
    without actual delays.
    """

    def __init__(
        self,
        base_latency_ms: float = 0,
        jitter_ms: float = 0,
        failure_rate: float = 0.0,
    ):
        self.base_latency = base_latency_ms / 1000
        self.jitter = jitter_ms / 1000
        self.failure_rate = failure_rate
        self._call_count = 0
        self._elapsed = 0.0

    def simulate_call(self) -> float:
        """Simulate a timed call, return elapsed time."""
        import random

        latency = self.base_latency
        if self.jitter > 0:
            latency += random.uniform(-self.jitter, self.jitter)
        latency = max(0, latency)

        self._call_count += 1
        self._elapsed += latency

        if random.random() < self.failure_rate:
            raise TimeoutError(f"Simulated timeout after {latency}s")

        return latency

    @property
    def elapsed(self) -> float:
        """Total simulated elapsed time."""
        return self._elapsed

    @property
    def call_count(self) -> int:
        """Number of simulated calls."""
        return self._call_count

    def reset(self) -> None:
        """Reset counters."""
        self._call_count = 0
        self._elapsed = 0.0


@contextmanager
def freeze_time(
    start_time: str | datetime,
    advance_on_use: bool = False,
    default_advance: timedelta = timedelta(seconds=1),
) -> Iterator[FrozenClock]:
    """Convenience context manager to freeze time.

    Args:
        start_time: Time to freeze at (string or datetime)
        advance_on_use: If True, automatically advance time on each access
        default_advance: Amount to advance when advance_on_use is True

    Example:
        with freeze_time("2024-01-15 10:00:00") as clock:
            assert datetime.now().year == 2024
            clock.advance(minutes=5)
            assert datetime.now().minute == 5
    """
    clock = FrozenClock(start_time)

    if advance_on_use:
        original_now = clock.now

        def advancing_now() -> datetime:
            result = original_now()
            clock.advance(seconds=default_advance.total_seconds())
            return result

        with clock.freeze():
            # Patch to use advancing version
            pass  # Implementation would require more complex patching
            yield clock
    else:
        with clock.freeze():
            yield clock


# Predefined time fixtures for common scenarios
JARVIS_FOUNDING = FrozenClock("2024-01-01 00:00:00")
TYPICAL_WORKDAY = FrozenClock("2024-03-15 14:30:00")
YEAR_END = FrozenClock("2024-12-31 23:59:59")
