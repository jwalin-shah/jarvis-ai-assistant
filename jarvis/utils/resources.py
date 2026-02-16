"""Resource management and cleanup utilities.

Provides context managers and helpers to ensure resources like connections,
file handles, and background tasks are properly closed and cleaned up.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@contextmanager
def managed_resource(
    resource: T,
    close_func: Callable[[T], Any] | None = None,
    name: str = "Resource",
) -> Generator[T, None, None]:
    """Context manager for a generic resource that needs explicit closing.

    Args:
        resource: The resource to manage.
        close_func: Function to call to close the resource. Defaults to resource.close().
        name: Name for logging purposes.

    Yields:
        The resource itself.
    """
    try:
        yield resource
    finally:
        if close_func:
            try:
                close_func(resource)
            except Exception as e:
                logger.warning(f"Error closing {name}: {e}")
        elif hasattr(resource, "close"):
            try:
                resource.close()
            except Exception as e:
                logger.warning(f"Error closing {name}: {e}")
        elif hasattr(resource, "aclose"):
            # Note: This sync CM can't await aclose.
            # Use managed_async_resource for async resources.
            logger.warning(f"{name} has aclose() but managed_resource is sync")


def safe_close(resource: Any, name: str = "Resource") -> None:
    """Safely call close() on a resource, suppressing exceptions.

    Args:
        resource: Object to close.
        name: Name for logging.
    """
    if resource is None:
        return
    try:
        if hasattr(resource, "close"):
            resource.close()
        elif hasattr(resource, "aclose"):
            logger.warning(f"safe_close called on async resource {name}")
    except Exception as e:
        logger.debug(f"Error during safe_close of {name}: {e}")


@contextmanager
def multi_resource_manager() -> Generator[list[Callable[[], Any]], None, None]:
    """Context manager that collects cleanup functions and runs them all at once.

    Example:
        with multi_resource_manager() as cleanups:
            conn = connect()
            cleanups.append(conn.close)

            file = open('log.txt')
            cleanups.append(file.close)

            # work with resources
    """
    cleanups: list[Callable[[], Any]] = []
    try:
        yield cleanups
    finally:
        for cleanup in reversed(cleanups):
            try:
                cleanup()
            except Exception as e:
                logger.debug(f"Cleanup function failed: {e}")
