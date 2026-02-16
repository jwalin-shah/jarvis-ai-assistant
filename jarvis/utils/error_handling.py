"""Exception handling utilities for consistent error management.

Provides decorators and context managers for common error handling patterns:
- Silencing non-critical exceptions with logging
- Graceful shutdown handling (preserving KeyboardInterrupt/SystemExit)
- Safe execution with default return values
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def silence_exceptions(
    *exceptions: type[Exception],
    log_level: int = logging.DEBUG,
    default_return: Any = None,
    log_msg: str | None = None,
    logger_name: str | None = None,
    preserve_interrupts: bool = True,
) -> Callable[[F], F]:
    """Decorator to silence specified exceptions with optional logging.

    Always re-raises KeyboardInterrupt and SystemExit unless preserve_interrupts=False.

    Args:
        *exceptions: Exception types to catch and silence. Defaults to Exception.
        log_level: Logging level for caught exceptions.
        default_return: Value to return when exception is caught.
        log_msg: Custom log message format. Use {error} and {func_name} placeholders.
        logger_name: Logger name to use. Defaults to module logger.
        preserve_interrupts: If True (default), always re-raise KeyboardInterrupt/SystemExit.

    Returns:
        Decorated function.

    Example:
        @silence_exceptions(sqlite3.OperationalError, log_level=logging.WARNING)
        def fetch_data():
            ...

        @silence_exceptions(default_return=[])
        def get_list():
            ...
    """
    if not exceptions:
        exceptions = (Exception,)

    target_logger = logging.getLogger(logger_name) if logger_name else logger

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except (KeyboardInterrupt, SystemExit):
                if preserve_interrupts:
                    raise
                # If interrupts are not preserved, treat as silenced exception
                error_msg = str(exc) if (exc := __import__('sys').exc_info()[1]) else "Interrupt"
                if log_msg:
                    formatted_msg = log_msg.format(error=error_msg, func_name=func.__qualname__)
                else:
                    formatted_msg = f"{func.__qualname__} interrupted: {error_msg}"
                target_logger.log(log_level, formatted_msg)
                return default_return
            except exceptions as e:
                if log_msg:
                    formatted_msg = log_msg.format(error=str(e), func_name=func.__qualname__)
                else:
                    formatted_msg = f"{func.__qualname__} failed: {e}"
                target_logger.log(log_level, formatted_msg)
                return default_return

        return wrapper  # type: ignore[return-value]

    return decorator


def graceful_shutdown(func: F) -> F:
    """Decorator that always re-raises KeyboardInterrupt and SystemExit.

    Use this for worker loops and long-running tasks that should exit cleanly
    on interrupt signals.

    Example:
        @graceful_shutdown
        def worker_loop():
            while running:
                do_work()
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            # Re-raise other exceptions - this decorator is just for documentation
            # and to ensure interrupts are never accidentally caught
            raise

    return wrapper  # type: ignore[return-value]


@contextmanager
def safe_execution(
    operation_name: str,
    default_return: Any = None,
    log_level: int = logging.DEBUG,
    reraise: tuple[type[Exception], ...] | None = None,
) -> Any:
    """Context manager for safe execution with error handling.

    Args:
        operation_name: Name of the operation for logging.
        default_return: Value to return/yield on exception.
        log_level: Logging level for caught exceptions.
        reraise: Exception types to re-raise instead of silencing.

    Yields:
        None (use the context for side effects) or captures the yield value.

    Example:
        with safe_execution("database query", default_return=[]):
            return fetch_rows()

        with safe_execution("cleanup", log_level=logging.WARNING) as result:
            # do work
            result.value = computed_value
    """
    class Result:
        def __init__(self) -> None:
            self.value = default_return
            self.exc_info: tuple[Any, ...] | None = None

    result = Result()

    try:
        yield result
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        result.exc_info = __import__('sys').exc_info()
        if reraise and isinstance(e, reraise):
            raise
        logger.log(log_level, f"{operation_name} failed: {e}")


class ErrorBoundary:
    """Class-based error boundary for more complex error handling scenarios.

    Tracks consecutive errors and provides hooks for error recovery.

    Example:
        boundary = ErrorBoundary(max_consecutive=5, on_threshold=circuit_breaker.open)

        with boundary:
            do_work()

        if boundary.consecutive_errors > 3:
            logger.warning("Multiple failures detected")
    """

    def __init__(
        self,
        max_consecutive: int = 0,
        on_threshold: Callable[[], None] | None = None,
        reset_on_success: bool = True,
    ) -> None:
        """Initialize error boundary.

        Args:
            max_consecutive: Max errors before calling on_threshold. 0 = disabled.
            on_threshold: Callback when threshold is reached.
            reset_on_success: Whether to reset counter on successful execution.
        """
        self.max_consecutive = max_consecutive
        self.on_threshold = on_threshold
        self.reset_on_success = reset_on_success
        self.consecutive_errors = 0
        self.total_errors = 0

    def __enter__(self) -> ErrorBoundary:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_type is None:
            # Success
            if self.reset_on_success:
                self.consecutive_errors = 0
            return True

        if exc_type in (KeyboardInterrupt, SystemExit):
            # Never suppress interrupts
            return False

        # Error occurred
        self.consecutive_errors += 1
        self.total_errors += 1

        if self.max_consecutive > 0 and self.consecutive_errors >= self.max_consecutive:
            if self.on_threshold:
                try:
                    self.on_threshold()
                except Exception:
                    pass
            # Reset after threshold to avoid spamming
            if self.reset_on_success:
                self.consecutive_errors = 0

        # Suppress the exception
        return True

    @property
    def is_healthy(self) -> bool:
        """Return True if no recent errors."""
        return self.consecutive_errors == 0

    @property
    def error_rate(self) -> float:
        """Return error rate (0.0-1.0) based on consecutive errors vs threshold."""
        if self.max_consecutive <= 0:
            return 0.0
        return min(self.consecutive_errors / self.max_consecutive, 1.0)
