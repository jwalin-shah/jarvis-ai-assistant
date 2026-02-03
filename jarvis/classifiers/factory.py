"""Singleton Factory - Thread-safe singleton pattern for classifiers.

Provides a generic, reusable singleton factory that replaces the duplicate
pattern found across TriggerClassifier and ResponseClassifier.

Usage:
    from jarvis.classifiers.factory import SingletonFactory

    _factory: SingletonFactory[MyClassifier] = SingletonFactory(MyClassifier)

    def get_classifier() -> MyClassifier:
        return _factory.get()

    def reset_classifier() -> None:
        _factory.reset()
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class SingletonFactory(Generic[T]):
    """Thread-safe singleton factory for classifier instances.

    This replaces the common pattern of:
        _classifier: T | None = None
        _lock = threading.Lock()

        def get_classifier() -> T:
            global _classifier
            if _classifier is None:
                with _lock:
                    if _classifier is None:
                        _classifier = T()
            return _classifier

    With a reusable, type-safe factory.

    Thread Safety:
        All methods are thread-safe using double-checked locking.
    """

    def __init__(
        self,
        factory_fn: Callable[[], T],
    ) -> None:
        """Initialize the singleton factory.

        Args:
            factory_fn: Callable that creates a new instance of T.
                Called lazily on first get().
        """
        self._factory_fn = factory_fn
        self._instance: T | None = None
        self._lock = threading.Lock()

    def get(self) -> T:
        """Get or create the singleton instance.

        Uses double-checked locking for thread safety with minimal overhead.

        Returns:
            The singleton instance of T.
        """
        # Fast path: instance already exists
        if self._instance is not None:
            return self._instance

        # Slow path: need to create instance
        with self._lock:
            # Double-check after acquiring lock
            if self._instance is None:
                self._instance = self._factory_fn()
            return self._instance

    def reset(self) -> None:
        """Reset the singleton, clearing the cached instance.

        The next call to get() will create a new instance.
        Useful for testing or reloading models.
        """
        with self._lock:
            self._instance = None

    def is_initialized(self) -> bool:
        """Check if the singleton has been initialized.

        Returns:
            True if get() has been called at least once since last reset().
        """
        return self._instance is not None


__all__ = ["SingletonFactory"]
