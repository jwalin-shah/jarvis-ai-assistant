"""Shared cache utilities for API routers.

Provides common caching patterns to reduce boilerplate across routers.
"""

from __future__ import annotations

import threading
from typing import Any

from jarvis.infrastructure.cache import TTLCache


def get_or_create_ttl_cache(
    cache_ref: list[TTLCache | None],
    lock: threading.Lock,
    ttl_seconds: float = 300.0,
    maxsize: int = 100,
) -> TTLCache | None:
    """Get or create a TTL cache with thread-safe initialization.

    Uses double-checked locking pattern for efficient thread-safe initialization.

    Args:
        cache_ref: List containing the cache reference (list allows mutation in closure).
        lock: Lock object for synchronization.
        ttl_seconds: Time-to-live for cached entries in seconds.
        maxsize: Maximum number of entries in the cache.

    Returns:
        The TTLCache instance.

    Example:
        ```python
        _cache: list[TTLCache | None] = [None]
        _cache_lock = threading.Lock()

        def get_cache() -> TTLCache:
            return get_or_create_ttl_cache(_cache, _cache_lock, ttl_seconds=300.0)
        ```
    """
    if cache_ref[0] is None:
        with lock:
            if cache_ref[0] is None:
                cache_ref[0] = TTLCache(ttl_seconds=ttl_seconds, maxsize=maxsize)
    return cache_ref[0]


class CacheManager:
    """Manager for multiple named caches.

    Provides centralized cache creation and access with consistent configuration.
    """

    def __init__(self) -> None:
        self._caches: dict[str, TTLCache] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def get_cache(
        self,
        name: str,
        ttl_seconds: float = 300.0,
        maxsize: int = 100,
    ) -> TTLCache:
        """Get or create a named cache.

        Args:
            name: Unique name for the cache.
            ttl_seconds: Time-to-live for cached entries.
            maxsize: Maximum number of entries.

        Returns:
            The TTLCache instance.
        """
        if name not in self._caches:
            with self._global_lock:
                if name not in self._caches:
                    self._caches[name] = TTLCache(ttl_seconds=ttl_seconds, maxsize=maxsize)
                    self._locks[name] = threading.Lock()
        return self._caches[name]

    def clear_cache(self, name: str) -> bool:
        """Clear a specific cache by name.

        Args:
            name: Name of the cache to clear.

        Returns:
            True if cache was found and cleared, False otherwise.
        """
        if name in self._caches:
            with self._locks[name]:
                self._caches[name].clear()
            return True
        return False

    def clear_all(self) -> None:
        """Clear all managed caches."""
        with self._global_lock:
            for name, cache in self._caches.items():
                with self._locks[name]:
                    cache.clear()

    def get_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all caches.

        Returns:
            Dictionary mapping cache names to their statistics.
        """
        stats = {}
        for name, cache in self._caches.items():
            cache_stats = cache.stats()
            stats[name] = {
                "size": cache_stats["size"],
                "maxsize": cache_stats["maxsize"],
                "ttl_seconds": cache_stats["ttl_seconds"],
            }
        return stats


# Global cache manager instance
_global_cache_manager: CacheManager | None = None
_global_manager_lock = threading.Lock()


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance.

    Returns:
        The global CacheManager singleton.
    """
    global _global_cache_manager
    if _global_cache_manager is None:
        with _global_manager_lock:
            if _global_cache_manager is None:
                _global_cache_manager = CacheManager()
    return _global_cache_manager
