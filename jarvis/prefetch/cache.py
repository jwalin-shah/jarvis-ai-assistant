"""Unified cache integration for speculative prefetching.

Integrates with the new jarvis.infrastructure.cache system while
maintaining backward compatibility for the prefetch system.
"""

from __future__ import annotations

import logging
from typing import Any

from jarvis.infrastructure.cache import get_unified_cache
from jarvis.infrastructure.cache.base import CacheBackend

logger = logging.getLogger(__name__)


class PrefetchCache:
    """Compatibility wrapper for UnifiedCache in the prefetch system."""

    def __init__(self, **kwargs: Any) -> None:
        # Use the singleton unified cache
        self._cache = get_unified_cache()

    def get(self, key: str) -> Any | None:
        return self._cache.get(key)

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: float | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._cache.set(key, value, ttl=ttl_seconds, tags=tags)

    def remove(self, key: str) -> bool:
        return self._cache.delete(key)

    def invalidate_by_tag(self, tag: str) -> int:
        return self._cache.invalidate_by_tag(tag)

    def invalidate_by_pattern(self, pattern: str) -> int:
        return self._cache.invalidate_by_pattern(pattern)

    def clear(self) -> None:
        self._cache.clear()

    def cleanup_expired(self) -> int:
        # Expired entries are cleaned up automatically on get() 
        # or by the backend's internal cleanup mechanism.
        return 0

    def stats(self) -> dict[str, Any]:
        return self._cache.stats()


# Backwards-compatible aliases
MultiTierCache = PrefetchCache

from enum import IntEnum


class CacheTier(IntEnum):
    L1 = 1
    L2 = 2
    L3 = 3


# Singleton
from jarvis.utils.singleton import thread_safe_singleton


@thread_safe_singleton
def get_cache() -> PrefetchCache:
    return PrefetchCache()


def reset_cache() -> None:
    instance = get_cache.peek()
    if instance is not None:
        instance.clear()
    get_cache.reset()


# Stubs for compatibility
class CacheError(Exception): pass
class CacheStats: pass
class CacheEntry: pass
class L1Cache: pass
class L2Cache: pass
class L3Cache: pass
