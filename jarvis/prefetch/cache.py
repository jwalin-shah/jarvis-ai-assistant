"""Unified cache integration for speculative prefetching.

Integrates with the new jarvis.infrastructure.cache system while
maintaining backward compatibility for the prefetch system.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from jarvis.utils.singleton import thread_safe_singleton

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Statistics for the prefetch cache."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class PrefetchCache:
    """Simple in-memory TTL cache for the prefetch system.

    Backward-compatible with tests expecting entries, maxsize, evictions, etc.
    """

    def __init__(
        self,
        maxsize: int = 1000,
        default_ttl: float = 300.0,
        **kwargs: Any,
    ) -> None:
        # Handle backward-compat kwargs
        if "l1_maxsize" in kwargs and maxsize == 1000:
            maxsize = kwargs["l1_maxsize"]

        self._maxsize = maxsize
        self._default_ttl = default_ttl
        self._data: dict[str, Any] = {}
        self._expiry: dict[str, float] = {}
        self._tags: dict[str, list[str]] = {}
        self._lock = threading.RLock()
        self._stats = CacheStats()

    def get(self, key: str) -> Any | None:
        """Get value from cache, checking expiry."""
        with self._lock:
            now = time.time()
            if key in self._data:
                if self._expiry.get(key, float("inf")) > now:
                    self._stats.hits += 1
                    return self._data[key]
                else:
                    # Expired - clean up
                    self._remove(key)
            self._stats.misses += 1
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: float | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Set value in cache with optional TTL and tags."""
        with self._lock:
            # Check if we need to evict
            if key not in self._data and len(self._data) >= self._maxsize:
                self._evict_one()

            self._data[key] = value
            ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
            self._expiry[key] = time.time() + ttl
            if tags:
                self._tags[key] = tags.copy()
            elif key in self._tags:
                del self._tags[key]

    def _remove(self, key: str) -> None:
        """Remove a key from all internal structures."""
        self._data.pop(key, None)
        self._expiry.pop(key, None)
        self._tags.pop(key, None)

    def _evict_one(self) -> None:
        """Evict the entry with soonest expiry."""
        if not self._data:
            return

        # Find key with earliest expiry
        earliest_key = min(self._expiry.keys(), key=lambda k: self._expiry.get(k, float("inf")))
        self._remove(earliest_key)
        self._stats.evictions += 1

    def remove(self, key: str) -> bool:
        """Remove a key from cache. Returns True if key existed."""
        with self._lock:
            existed = key in self._data
            self._remove(key)
            return existed

    def invalidate_by_tag(self, tag: str) -> int:
        """Remove all entries with the given tag."""
        with self._lock:
            to_remove = [k for k, tags in self._tags.items() if tag in tags]
            for key in to_remove:
                self._remove(key)
            return len(to_remove)

    def invalidate_by_pattern(self, pattern: str) -> int:
        """Remove all keys starting with pattern."""
        with self._lock:
            to_remove = [k for k in self._data.keys() if k.startswith(pattern)]
            for key in to_remove:
                self._remove(key)
            return len(to_remove)

    def clear(self) -> None:
        """Clear all entries and reset stats."""
        with self._lock:
            self._data.clear()
            self._expiry.clear()
            self._tags.clear()
            self._stats = CacheStats()

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count cleaned."""
        with self._lock:
            now = time.time()
            to_remove = [k for k, exp in self._expiry.items() if exp <= now]
            for key in to_remove:
                self._remove(key)
            return len(to_remove)

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "entries": len(self._data),
                "maxsize": self._maxsize,
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "evictions": self._stats.evictions,
                "hit_rate": self._stats.hit_rate,
            }


# Backwards-compatible aliases
MultiTierCache = PrefetchCache


class CacheTier(IntEnum):
    L1 = 1
    L2 = 2
    L3 = 3


@thread_safe_singleton
def get_cache() -> PrefetchCache:
    return PrefetchCache()


def reset_cache() -> None:
    cache = get_cache.peek()  # type: ignore[attr-defined]
    if cache is not None:
        cache.clear()
    get_cache.reset()  # type: ignore[attr-defined]


# Stubs for compatibility
class CacheError(Exception):
    pass


class CacheEntry:
    pass


class L1Cache:
    pass


class L2Cache:
    pass


class L3Cache:
    pass
