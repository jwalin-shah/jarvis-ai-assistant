"""Simple in-memory TTL cache for speculative prefetching.

Draft replies live 2-5 minutes and are worthless after process restart.
A single dict with TTL expiry is all we need.

Usage:
    cache = PrefetchCache()
    cache.set("draft:chat123", draft_response, ttl_seconds=300)
    result = cache.get("draft:chat123")  # Returns value or None
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from jarvis.errors import ErrorCode, JarvisError

logger = logging.getLogger(__name__)


class CacheError(JarvisError):
    """Cache operation failed."""

    default_message = "Cache operation failed"
    default_code = ErrorCode.UNKNOWN


@dataclass
class CacheStats:
    """Basic hit/miss counters."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class _Entry:
    """Internal cache entry with TTL."""

    value: Any
    expires_at: float
    tags: list[str] = field(default_factory=list)


class PrefetchCache:
    """Thread-safe in-memory TTL cache.

    Simple dict-based cache with expiry, maxsize eviction, and tag-based
    invalidation. Replaces the former 3-tier (L1/L2/L3) cache -- draft
    replies are ephemeral and don't need SQLite or disk persistence.
    """

    def __init__(
        self,
        maxsize: int = 200,
        default_ttl: float = 300.0,
        # Old MultiTierCache kwargs accepted for backwards compat (ignored)
        l1_maxsize: int | None = None,
        l1_max_bytes: int | None = None,
        l2_db_path: Any = None,
        l3_cache_dir: Any = None,
        l3_max_bytes: int | None = None,
        auto_promote: bool = True,
        promote_threshold: int = 3,
    ) -> None:
        self._data: dict[str, _Entry] = {}
        self._maxsize = l1_maxsize if l1_maxsize is not None else maxsize
        self._default_ttl = default_ttl
        self._lock = threading.Lock()
        self._stats = CacheStats()

    # -- public API used by callers --

    def get(self, key: str) -> Any | None:
        """Return cached value or None if missing/expired."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._stats.misses += 1
                return None
            if time.time() > entry.expires_at:
                del self._data[key]
                self._stats.misses += 1
                return None
            self._stats.hits += 1
            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: float | None = None,
        tags: list[str] | None = None,
        **_kwargs: Any,
    ) -> None:
        """Store a value with TTL. Extra kwargs (tier, etc.) are ignored for compat."""
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        with self._lock:
            self._data[key] = _Entry(
                value=value,
                expires_at=time.time() + ttl,
                tags=tags or [],
            )
            self._evict_if_full()

    def remove(self, key: str) -> bool:
        """Remove a single key. Returns True if it existed."""
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def invalidate_by_tag(self, tag: str) -> int:
        """Remove all entries that have the given tag."""
        with self._lock:
            to_remove = [k for k, e in self._data.items() if tag in e.tags]
            for k in to_remove:
                del self._data[k]
            return len(to_remove)

    def invalidate_by_pattern(self, pattern: str) -> int:
        """Remove all entries whose key starts with pattern."""
        with self._lock:
            to_remove = [k for k in self._data if k.startswith(pattern)]
            for k in to_remove:
                del self._data[k]
            return len(to_remove)

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._data.clear()
            self._stats = CacheStats()

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        now = time.time()
        with self._lock:
            expired = [k for k, e in self._data.items() if now > e.expires_at]
            for k in expired:
                del self._data[k]
            self._stats.evictions += len(expired)
            return len(expired)

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            return {
                "entries": len(self._data),
                "maxsize": self._maxsize,
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "evictions": self._stats.evictions,
                "hit_rate": self._stats.hit_rate,
            }

    # -- internal --

    def _evict_if_full(self) -> None:
        """Evict oldest entries if over maxsize. Must hold self._lock."""
        while len(self._data) > self._maxsize:
            # Remove the entry that expires soonest (cheapest to lose)
            oldest_key = min(self._data, key=lambda k: self._data[k].expires_at)
            del self._data[oldest_key]
            self._stats.evictions += 1


# ---------------------------------------------------------------------------
# Backwards-compatible aliases so existing callers keep working
# ---------------------------------------------------------------------------

# The old code exposed MultiTierCache as the cache type everywhere.
# Alias it so `from jarvis.prefetch.cache import MultiTierCache` still works.
MultiTierCache = PrefetchCache

# CacheTier was used by executor to decide where to store. Now ignored by
# set(), but we keep the enum so callers don't break.
from enum import IntEnum  # noqa: E402


class CacheTier(IntEnum):
    L1 = 1
    L2 = 2
    L3 = 3


# Stub dataclasses that old code imported but aren't needed anymore
@dataclass
class CacheEntry:
    """Kept for import compatibility. Not used internally."""

    key: str = ""
    value: Any = None
    tier: int = 1
    tags: list[str] = field(default_factory=list)


class L1Cache:
    """Stub for import compatibility."""

    pass


class L2Cache:
    """Stub for import compatibility."""

    pass


class L3Cache:
    """Stub for import compatibility."""

    pass


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

from jarvis.utils.singleton import thread_safe_singleton  # noqa: E402


@thread_safe_singleton
def get_cache() -> PrefetchCache:
    """Get or create singleton cache instance."""
    return PrefetchCache()


def reset_cache() -> None:
    """Reset singleton cache (clears all entries)."""
    instance = get_cache.peek()  # type: ignore[attr-defined]
    if instance is not None:
        instance.clear()
    get_cache.reset()  # type: ignore[attr-defined]
