"""Thread-safe singleton decorator.

Replaces hand-rolled double-check locking patterns throughout the codebase.

Usage::

    @thread_safe_singleton
    def get_my_service() -> MyService:
        return MyService()

    svc = get_my_service()       # Creates on first call
    svc2 = get_my_service()      # Returns cached instance (svc is svc2)
    get_my_service.reset()       # Clear for testing
    get_my_service.peek()        # Returns cached instance or None (no creation)
"""

from __future__ import annotations

import functools
import threading
from typing import Any, TypeVar

T = TypeVar("T")


def thread_safe_singleton(factory_fn):  # noqa: ANN001, ANN201
    """Decorator that caches the return value of a factory function.

    Thread-safe via double-check locking.  The decorated function gains:
    - ``reset()`` method that clears the cached instance (useful in tests).
    - ``peek()`` method to access the cached value without triggering creation.

    Note: only the *first* call's arguments are used to create the instance.
    Subsequent calls return the cached value regardless of arguments.
    """
    instance: Any = None
    lock = threading.Lock()

    @functools.wraps(factory_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal instance
        if instance is not None:
            return instance
        with lock:
            if instance is None:
                instance = factory_fn(*args, **kwargs)
            return instance

    def reset() -> None:
        nonlocal instance
        with lock:
            instance = None

    def peek() -> Any:
        """Return the cached instance, or None if not yet created."""
        return instance

    wrapper.reset = reset  # type: ignore[attr-defined]
    wrapper.peek = peek  # type: ignore[attr-defined]
    return wrapper
