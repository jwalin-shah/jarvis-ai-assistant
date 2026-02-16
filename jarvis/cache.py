"""Unified thread-safe TTL cache with LRU eviction and single-flight support.

Note: This module is now a facade for the infrastructure cache system.
New code should use jarvis.infrastructure.cache directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from jarvis.infrastructure.cache.memory import _SENTINEL, MemoryBackend

T = TypeVar("T")


class TTLCache:
    """Facade for jarvis.infrastructure.cache.memory.MemoryBackend.

    Maintains the legacy TTLCache interface for backward compatibility.
    """

    def __init__(self, ttl_seconds: float = 30.0, maxsize: int = 128) -> None:
        self._backend = MemoryBackend(ttl=ttl_seconds, maxsize=maxsize)

    def get(self, key: str) -> tuple[bool, Any]:
        """Get a value from the cache.

        Returns:
            Tuple of (found, value).
        """
        val = self._backend.get(key)
        if val is not _SENTINEL:
            return True, val
        return False, None

    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache."""
        self._backend.set(key, value)

    def delete(self, key: str) -> None:
        """Remove a specific key from cache."""
        self._backend.delete(key)

    def invalidate(self, key: str | None = None) -> None:
        """Invalidate cache entries. None clears all."""
        if key is None:
            self._backend.clear()
        else:
            self._backend.delete(key)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._backend.clear()

    def get_or_set(self, key: str, factory: Callable[[], T], ttl: float | None = None) -> T:
        """Get from cache or compute via factory with single-flight protection."""
        return self._backend.get_or_set(key, factory, ttl=ttl)

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        s = self._backend.stats()
        # Map new stat keys to legacy ones
        return {
            "size": s["entries"],
            "maxsize": s["maxsize"],
            "ttl_seconds": s["ttl_seconds"],
            "hits": s["hits"],
            "misses": s["misses"],
            "hit_rate": s["hit_rate"],
        }
