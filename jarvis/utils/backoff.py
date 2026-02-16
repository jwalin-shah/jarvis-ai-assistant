"""Backoff strategies for retry logic and error recovery.

Provides configurable backoff algorithms for handling transient failures
in loops, workers, and connection attempts.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass
class BackoffConfig:
    """Configuration for backoff behavior.

    Attributes:
        base_delay: Initial delay between retries (seconds).
        max_delay: Maximum delay cap (seconds).
        backoff_factor: Multiplier for exponential backoff.
        max_consecutive: Delay starts increasing after this many consecutive errors.
    """

    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    max_consecutive: int = 5


class ConsecutiveErrorTracker:
    """Track consecutive errors and provide backoff delays.

    Used in worker loops to implement exponential backoff when errors
    occur repeatedly. Resets when operations succeed.

    Example:
        tracker = ConsecutiveErrorTracker(
            base_delay=2.0,
            max_delay=30.0,
            backoff_factor=2.0
        )

        while running:
            try:
                do_work()
                tracker.reset()
            except TransientError:
                delay = tracker.on_error()
                time.sleep(delay)
    """

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
        max_consecutive: int = 5,
        name: str = "",
    ) -> None:
        """Initialize error tracker.

        Args:
            base_delay: Initial delay for first backoff (seconds).
            max_delay: Maximum delay cap (seconds).
            backoff_factor: Exponential growth multiplier.
            max_consecutive: Consecutive errors before backoff starts.
            name: Optional name for logging.
        """
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.max_consecutive = max_consecutive
        self.name = name or "tracker"

        self._consecutive = 0
        self._current_delay = base_delay
        self._total_errors = 0

    @property
    def consecutive_errors(self) -> int:
        """Current count of consecutive errors."""
        return self._consecutive

    @property
    def total_errors(self) -> int:
        """Total error count since creation."""
        return self._total_errors

    def reset(self) -> None:
        """Reset consecutive error count and delay. Call on success."""
        if self._consecutive > 0:
            logger.debug(f"{self.name}: Reset after {self._consecutive} consecutive errors")
        self._consecutive = 0
        self._current_delay = self.base_delay

    def on_error(self, log_level: int = logging.WARNING) -> float:
        """Record an error and return the recommended delay.

        Args:
            log_level: Level for logging the error count.

        Returns:
            Seconds to delay before next attempt.
        """
        self._consecutive += 1
        self._total_errors += 1

        # Calculate delay only after threshold
        if self._consecutive >= self.max_consecutive:
            # Exponential backoff: base_delay * factor^(consecutive - threshold)
            exponent = self._consecutive - self.max_consecutive + 1
            delay = min(
                self.base_delay * (self.backoff_factor ** exponent),
                self.max_delay
            )
        else:
            delay = self.base_delay

        self._current_delay = delay

        logger.log(
            log_level,
            f"{self.name}: Consecutive error {self._consecutive}, "
            f"backing off for {delay:.1f}s"
        )

        return delay

    def get_delay(self) -> float:
        """Get current delay without incrementing error count."""
        return self._current_delay


class AsyncConsecutiveErrorTracker:
    """Async version of ConsecutiveErrorTracker for async/await code.

    Example:
        tracker = AsyncConsecutiveErrorTracker(base_delay=2.0)

        while running:
            try:
                await do_work()
                tracker.reset()
            except Exception:
                delay = tracker.on_error()
                await asyncio.sleep(delay)
    """

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
        max_consecutive: int = 5,
        name: str = "",
    ) -> None:
        """Initialize async error tracker with same parameters as sync version."""
        self._tracker = ConsecutiveErrorTracker(
            base_delay=base_delay,
            max_delay=max_delay,
            backoff_factor=backoff_factor,
            max_consecutive=max_consecutive,
            name=name,
        )

    @property
    def consecutive_errors(self) -> int:
        """Current count of consecutive errors."""
        return self._tracker.consecutive_errors

    @property
    def total_errors(self) -> int:
        """Total error count since creation."""
        return self._tracker.total_errors

    def reset(self) -> None:
        """Reset consecutive error count."""
        self._tracker.reset()

    def on_error(self, log_level: int = logging.WARNING) -> float:
        """Record an error and return recommended delay."""
        return self._tracker.on_error(log_level)

    async def sleep(self, delay: float | None = None) -> None:
        """Async sleep for the specified or current delay."""
        import asyncio
        await asyncio.sleep(delay if delay is not None else self._tracker.get_delay())


class CircuitBreaker:
    """Circuit breaker pattern for failing operations.

    Opens after threshold failures, preventing cascading failures.
    Automatically closes after timeout.

    States:
        CLOSED: Normal operation (circuit is closed, current flows)
        OPEN: Failing fast (circuit is open, no current)
        HALF_OPEN: Testing if recovered

    Example:
        breaker = CircuitBreaker(failure_threshold=5, timeout=30.0)

        with breaker:
            # If breaker is OPEN, raises CircuitBreakerOpen immediately
            call_external_service()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 30.0,
        name: str = "",
    ) -> None:
        """Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit.
            timeout: Seconds to wait before trying half-open.
            name: Optional name for logging.
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.name = name or "circuit"

        self._failures = 0
        self._last_failure_time: float | None = None
        self._state = "CLOSED"

    @property
    def state(self) -> str:
        """Current circuit state: CLOSED, OPEN, or HALF_OPEN."""
        if self._state == "OPEN":
            # Check if timeout has elapsed
            if self._last_failure_time and time.time() - self._last_failure_time > self.timeout:
                self._state = "HALF_OPEN"
                logger.info(f"{self.name}: Circuit entering HALF_OPEN state")
        return self._state

    @property
    def is_closed(self) -> bool:
        """True if circuit allows operations."""
        return self.state == "CLOSED"

    def record_success(self) -> None:
        """Record a successful operation."""
        if self._state == "HALF_OPEN":
            logger.info(f"{self.name}: Circuit CLOSED (recovered)")
            self._state = "CLOSED"
        self._failures = 0
        self._last_failure_time = None

    def record_failure(self) -> None:
        """Record a failed operation."""
        self._failures += 1
        self._last_failure_time = time.time()

        if self._failures >= self.failure_threshold:
            if self._state != "OPEN":
                logger.warning(
                    f"{self.name}: Circuit OPEN after {self._failures} failures"
                )
                self._state = "OPEN"

    def __enter__(self) -> "CircuitBreaker":
        if self.state == "OPEN":
            raise CircuitBreakerOpen(f"Circuit {self.name} is OPEN")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        if exc_type is None:
            self.record_success()
        elif exc_type not in (KeyboardInterrupt, SystemExit):
            self.record_failure()
        return False


class CircuitBreakerOpen(Exception):
    """Raised when attempting to use an open circuit breaker."""

    pass


@dataclass
class RetryStats:
    """Statistics from retry operations."""

    attempts: int = 0
    successes: int = 0
    failures: int = 0
    total_delay: float = 0.0

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.attempts == 0:
            return 0.0
        return (self.successes / self.attempts) * 100


def with_retry(
    max_attempts: int = 3,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    config: BackoffConfig | None = None,
    on_retry: Callable[[int, Exception], None] | None = None,
    predicate: Callable[[Exception], bool] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for retrying operations with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts.
        exceptions: Exception types to retry on.
        config: Backoff configuration.
        on_retry: Callback(attempt_number, exception) called on each retry.
        predicate: Optional function that takes an exception and returns True if it should be retried.

    Example:
        @with_retry(max_attempts=3, exceptions=(ConnectionError, TimeoutError))
        def fetch_data():
            ...
    """
    cfg = config or BackoffConfig()

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracker = ConsecutiveErrorTracker(
                base_delay=cfg.base_delay,
                max_delay=cfg.max_delay,
                backoff_factor=cfg.backoff_factor,
                max_consecutive=1,  # Start backing off immediately
            )

            last_error: Exception | None = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except exceptions as e:
                    if predicate and not predicate(e):
                        raise
                        
                    last_error = e
                    if attempt < max_attempts - 1:
                        if on_retry:
                            try:
                                on_retry(attempt + 1, e)
                            except Exception:
                                pass
                        delay = tracker.on_error()
                        time.sleep(delay)
                    else:
                        raise

            # Should not reach here, but type checker needs it
            if last_error:
                raise last_error
            return None

        return wrapper

    return decorator
