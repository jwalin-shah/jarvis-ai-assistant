"""SQLite retry logic for handling database locking.

Provides a decorator to automatically retry SQLite operations when
the 'database is locked' error occurs, which is common when accessing
the iMessage database while the Messages app is active.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from typing import Any, TypeVar, cast

from jarvis.config import get_config

from .backoff import BackoffConfig, with_retry

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def sqlite_retry(
    max_attempts: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
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
    # Set defaults from config if not provided
    config = get_config()
    if max_attempts is None:
        max_attempts = config.retry.sqlite_max_attempts
    if base_delay is None:
        base_delay = config.retry.sqlite_base_delay
    if max_delay is None:
        max_delay = config.retry.sqlite_max_delay

    backoff_config = BackoffConfig(
        base_delay=base_delay, max_delay=max_delay, backoff_factor=backoff_factor
    )

    def is_lock_error(e: Exception) -> bool:
        error_str = str(e).lower()
        return "database is locked" in error_str or "busy" in error_str

    def on_retry(attempt: int, e: Exception) -> None:
        logger.debug(f"SQLite locked/busy (attempt {attempt}/{max_attempts}). Retrying...")

    return cast(
        Callable[[F], F],
        with_retry(
            max_attempts=max_attempts,
            exceptions=(sqlite3.OperationalError, sqlite3.InterfaceError),
            config=backoff_config,
            on_retry=on_retry,
            predicate=is_lock_error,
        ),
    )
