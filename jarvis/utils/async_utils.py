"""Asynchronous programming utilities and helpers.

Provides tools for bridging sync and async code, managing background tasks,
and handling task-specific error logging.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


async def run_in_thread(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Run a synchronous function in a separate thread.

    Consistent wrapper around asyncio.to_thread with logging.

    Args:
        func: The synchronous function to run.
        *args: Positional arguments for func.
        **kwargs: Keyword arguments for func.

    Returns:
        The result of the function.
    """
    return await asyncio.to_thread(func, *args, **kwargs)


def sync_to_async(func: Callable[P, R]) -> Callable[P, Awaitable[R]]:
    """Decorator to expose a synchronous function as an asynchronous one.

    Uses asyncio.to_thread internally.

    Example:
        @sync_to_async
        def blocking_io_op(data):
            ...

        await blocking_io_op(my_data)
    """

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


def log_task_exception(
    task: asyncio.Task[Any],
    msg: str = "Background task failed",
    logger_instance: logging.Logger | None = None,
) -> None:
    """Callback for add_done_callback to log task exceptions.

    Args:
        task: The completed asyncio task.
        msg: Message to log on failure.
        logger_instance: Logger to use. Defaults to module logger.
    """
    log = logger_instance or logger
    try:
        if not task.cancelled():
            task.result()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        log.warning(f"{msg}: {e}", exc_info=True)


def task_callback(
    msg: str = "Background task failed", logger_instance: logging.Logger | None = None
) -> Callable[[asyncio.Task[Any]], None]:
    """Create a callback for add_done_callback with custom message.

    Example:
        task = asyncio.create_task(work())
        task.add_done_callback(task_callback("Sync failed", my_logger))
    """
    return functools.partial(log_task_exception, msg=msg, logger_instance=logger_instance)


async def wait_with_timeout(
    awaitable: Awaitable[R], timeout: float, default: R | None = None
) -> R | None:
    """Wait for an awaitable with a timeout, returning a default on timeout.

    Args:
        awaitable: The task/coroutine to wait for.
        timeout: Max seconds to wait.
        default: Value to return if timeout occurs.

    Returns:
        Result of awaitable or default.
    """
    try:
        return await asyncio.wait_for(awaitable, timeout)
    except TimeoutError:
        return default
