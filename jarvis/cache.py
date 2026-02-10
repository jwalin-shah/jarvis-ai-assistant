"""Unified thread-safe TTL cache with LRU eviction and single-flight support.

Replaces the separate implementations in jarvis/db/models.py and jarvis/metrics.py.
"""

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class TTLCache:
    """Thread-safe LRU cache with TTL expiration and single-flight support.

    Features:
    - Time-to-live expiration per entry
    - LRU eviction when at capacity
    - Hit/miss statistics
    - Single-flight: concurrent callers for the same key wait for one computation
    - Thread-safe with RLock
    """

    def __init__(self, ttl_seconds: float = 30.0, maxsize: int = 128) -> None:
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        # Single-flight: maps key -> Event for in-progress computations
        self._inflight: dict[str, threading.Event] = {}
        self._inflight_lock = threading.Lock()

    def get(self, key: str) -> tuple[bool, Any]:
        """Get a value from the cache.

        Returns:
            Tuple of (found, value). found is False if key doesn't exist or expired.
        """
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    self._hits += 1
                    self._cache.move_to_end(key)
                    return True, value
                del self._cache[key]
            self._misses += 1
            return False, None

    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache."""
        with self._lock:
            while len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
            self._cache[key] = (value, time.time())
            self._cache.move_to_end(key)

    def delete(self, key: str) -> None:
        """Remove a specific key from cache."""
        with self._lock:
            self._cache.pop(key, None)

    def invalidate(self, key: str | None = None) -> None:
        """Invalidate cache entries. None clears all."""
        with self._lock:
            if key is None:
                self._cache.clear()
            else:
                self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()

    def get_or_set(self, key: str, factory: Callable[[], T], ttl: float | None = None) -> T:
        """Get from cache or compute via factory with single-flight protection.

        If the key is not cached, calls factory() to compute the value.
        Concurrent callers for the same key will wait for the first computation
        rather than all computing independently (prevents cache stampedes).

        Args:
            key: Cache key
            factory: Callable that produces the value if not cached
            ttl: Optional per-key TTL override (not used, reserves for future)

        Returns:
            The cached or freshly computed value
        """
        found, value = self.get(key)
        if found:
            return value  # type: ignore[return-value]

        # Single-flight: check if another thread is already computing this key
        with self._inflight_lock:
            if key in self._inflight:
                event = self._inflight[key]
            else:
                event = threading.Event()
                self._inflight[key] = event
                event = None  # We are the leader

        if event is not None:
            # Wait for the leader to finish
            event.wait(timeout=60.0)
            found, value = self.get(key)
            if found:
                return value  # type: ignore[return-value]
            # Leader failed or timed out, fall through to compute ourselves

        try:
            result = factory()
            self.set(key, result)
            return result
        finally:
            with self._inflight_lock:
                evt = self._inflight.pop(key, None)
                if evt is not None:
                    evt.set()

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }
