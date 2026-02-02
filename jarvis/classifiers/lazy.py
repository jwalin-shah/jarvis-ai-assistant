"""Lazy Initialization Utility - Thread-safe lazy initialization for expensive computations.

Provides a generic, reusable lazy initialization pattern that replaces the duplicate
pattern found across classifiers for computing centroids, embeddings, and other
expensive data structures.

Usage:
    from jarvis.classifiers.lazy import LazyInitializer

    def compute_centroids() -> dict[str, np.ndarray]:
        # Expensive computation
        return {"class_a": np.array([...]), "class_b": np.array([...])}

    centroids = LazyInitializer(compute_centroids, name="centroids")

    # First call computes the value
    data = centroids.get()

    # Subsequent calls return cached value
    data = centroids.get()

    # Reset to recompute
    centroids.reset()
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LazyInitializer(Generic[T]):
    """Thread-safe lazy initialization utility.

    Encapsulates the double-checked locking pattern used across classifiers
    for computing centroids, embeddings, and other expensive data structures.

    This replaces the common pattern of:
        self._data: T | None = None
        self._lock = threading.Lock()
        self._init_attempted = False

        def _ensure_data_computed(self) -> None:
            if self._data is not None:
                return
            with self._lock:
                if self._data is not None:
                    return
                if self._init_attempted:
                    return
                self._init_attempted = True
                try:
                    self._data = self._compute_data()
                except Exception as e:
                    logger.warning("Failed to compute data: %s", e)
                    self._data = None

    With a reusable, type-safe class.

    Type Parameters:
        T: The type of the lazily-initialized value.

    Thread Safety:
        All methods are thread-safe using double-checked locking.
        Initialization is attempted at most once until reset() is called.

    Error Handling:
        If the compute function raises an exception, the error is logged
        and get() returns None. The initialization is marked as attempted
        so subsequent calls won't retry. Call reset() to allow a retry.
    """

    def __init__(
        self,
        compute_fn: Callable[[], T],
        name: str = "data",
    ) -> None:
        """Initialize the lazy initializer.

        Args:
            compute_fn: Callable that computes the value. Called lazily on first get().
                Should not accept any arguments. For parameterized computation,
                use functools.partial or a lambda.
            name: Human-readable name for the data being computed.
                Used in log messages for debugging.
        """
        self._compute_fn = compute_fn
        self._name = name
        self._value: T | None = None
        self._lock = threading.Lock()
        self._init_attempted = False

    def get(self) -> T | None:
        """Get the lazily-initialized value, computing if needed.

        Uses double-checked locking for thread safety with minimal overhead.
        If computation fails, returns None and logs a warning.

        Returns:
            The computed value, or None if computation failed or hasn't been
            attempted successfully.
        """
        # Fast path: value already computed
        if self._value is not None:
            return self._value

        # Fast path: initialization already attempted and failed
        if self._init_attempted:
            return None

        # Slow path: need to compute value
        with self._lock:
            # Double-check after acquiring lock
            if self._value is not None:
                return self._value

            if self._init_attempted:
                return None

            self._init_attempted = True

            try:
                self._value = self._compute_fn()
                logger.debug("Successfully computed %s", self._name)
                return self._value
            except Exception as e:
                logger.warning("Failed to compute %s: %s", self._name, e)
                self._value = None
                return None

    def get_or_raise(self) -> T:
        """Get the lazily-initialized value, raising if computation fails.

        Unlike get(), this method raises an exception if the computation
        fails or returns None. Useful when the value is required.

        Returns:
            The computed value (never None).

        Raises:
            RuntimeError: If computation fails or returns None.
        """
        # Fast path: value already computed
        if self._value is not None:
            return self._value

        # Slow path: need to compute value
        with self._lock:
            # Double-check after acquiring lock
            if self._value is not None:
                return self._value

            if self._init_attempted:
                raise RuntimeError(f"Failed to compute {self._name} (already attempted)")

            self._init_attempted = True

            try:
                result = self._compute_fn()
                if result is None:
                    raise RuntimeError(f"Compute function for {self._name} returned None")
                self._value = result
                logger.debug("Successfully computed %s", self._name)
                return self._value
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"Failed to compute {self._name}: {e}") from e

    def reset(self) -> None:
        """Reset the initializer to allow recomputation.

        Clears the cached value and the init_attempted flag, allowing
        the next call to get() to recompute the value.

        Thread-safe: uses the internal lock to ensure consistency.
        """
        with self._lock:
            self._value = None
            self._init_attempted = False

    @property
    def is_initialized(self) -> bool:
        """Check if the value has been successfully computed.

        Returns:
            True if get() has been called and computation succeeded.
            False if never called, computation failed, or reset() was called.
        """
        return self._value is not None

    @property
    def was_attempted(self) -> bool:
        """Check if initialization has been attempted.

        Returns:
            True if get() has been called at least once since construction
            or the last reset(), regardless of whether it succeeded.
        """
        return self._init_attempted

    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        status = "initialized" if self._value is not None else "pending"
        if self._init_attempted and self._value is None:
            status = "failed"
        return f"LazyInitializer(name={self._name!r}, status={status})"


__all__ = ["LazyInitializer"]
