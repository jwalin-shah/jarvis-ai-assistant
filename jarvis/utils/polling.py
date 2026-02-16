"""Polling utilities for status checking and health monitoring.

Provides helpers to wait for conditions to be met with configurable
intervals and timeouts.
"""

from __future__ import annotations

import time
import logging
from collections.abc import Callable
from typing import TypeVar, Any

logger = logging.getLogger(__name__)

T = TypeVar("T")


def poll_until(
    predicate: Callable[[], T],
    timeout: float = 30.0,
    interval: float = 1.0,
    backoff: float = 1.0,
    name: str = "Operation",
) -> T | None:
    """Wait for a predicate to return a truthy value.

    Args:
        predicate: Callable that returns a value.
        timeout: Maximum seconds to wait.
        interval: Initial wait time between attempts.
        backoff: Multiplier for the interval after each failed attempt.
        name: Name of the operation for logging.

    Returns:
        The truthy value returned by the predicate, or None if timeout reached.
    """
    start_time = time.time()
    current_interval = interval
    
    while time.time() - start_time < timeout:
        result = predicate()
        if result:
            return result
        
        time.sleep(current_interval)
        current_interval *= backoff
        
    logger.debug(f"Timed out waiting for {name} after {timeout}s")
    return None


async def async_poll_until(
    predicate: Callable[[], Any],
    timeout: float = 30.0,
    interval: float = 1.0,
    backoff: float = 1.0,
    name: str = "Operation",
) -> Any:
    """Async version of poll_until."""
    import asyncio
    
    start_time = time.time()
    current_interval = interval
    
    while time.time() - start_time < timeout:
        result = await predicate() if asyncio.iscoroutinefunction(predicate) else predicate()
        if result:
            return result
        
        await asyncio.sleep(current_interval)
        current_interval *= backoff
        
    logger.debug(f"Timed out waiting for {name} after {timeout}s")
    return None
