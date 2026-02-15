"""SQLite retry logic for handling database locking.

Provides a decorator to automatically retry SQLite operations when
the 'database is locked' error occurs, which is common when accessing
the iMessage database while the Messages app is active.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def sqlite_retry(
    max_attempts: int = 5,
    base_delay: float = 0.1,
    max_delay: float = 2.0,
    backoff_factor: float = 2.0,
) -> Callable[[F], F]:
    """Decorator to retry functions on SQLite locking errors.

    Args:
        max_attempts: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        backoff_factor: Multiplier for delay after each failure.

    Returns:
        Decorated function.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = base_delay
            last_error: Exception | None = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
                    error_str = str(e).lower()
                    if "database is locked" in error_str or "busy" in error_str:
                        last_error = e
                        if attempt < max_attempts - 1:
                            sleep_time = min(delay, max_delay)
                            logger.debug(
                                f"SQLite locked/busy (attempt {attempt + 1}/{max_attempts}). "
                                f"Retrying in {sleep_time:.2f}s..."
                            )
                            time.sleep(sleep_time)
                            delay *= backoff_factor
                            continue
                    raise  # Re-raise if not a locking error or max attempts reached

            # If we exhausted attempts
            if last_error:
                logger.error(f"SQLite operation failed after {max_attempts} attempts: {last_error}")
                raise last_error
            return None  # Should not be reached

        return cast(F, wrapper)

    return decorator
