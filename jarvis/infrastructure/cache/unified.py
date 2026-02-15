from __future__ import annotations

import logging
from typing import Any

from jarvis.infrastructure.cache.base import CacheBackend

logger = logging.getLogger(__name__)


class UnifiedCache(CacheBackend):
    """Multi-tier cache that unifies memory and persistent backends."""

    def __init__(
        self,
        l1: CacheBackend | None = None,
        l2: CacheBackend | None = None,
    ) -> None:
        self.l1 = l1
        self.l2 = l2
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        if self.l1:
            val = self.l1.get(key)
            if val is not None:
                self._hits += 1
                return val

        if self.l2:
            val = self.l2.get(key)
            if val is not None:
                self._hits += 1
                if self.l1:
                    # We don't have the original TTL or tags here,
                    # so we use defaults.
                    self.l1.set(key, val)
                return val

        self._misses += 1
        return None

    def set(
        self, key: str, value: Any, ttl: float | None = None, tags: list[str] | None = None
    ) -> None:
        if self.l1:
            self.l1.set(key, value, ttl, tags)
        if self.l2:
            self.l2.set(key, value, ttl, tags)

    def delete(self, key: str) -> bool:
        deleted = False
        if self.l1:
            deleted |= self.l1.delete(key)
        if self.l2:
            deleted |= self.l2.delete(key)
        return deleted

    def invalidate_by_tag(self, tag: str) -> int:
        count = 0
        if self.l1:
            count += self.l1.invalidate_by_tag(tag)
        if self.l2:
            count += self.l2.invalidate_by_tag(tag)
        return count

    def invalidate_by_pattern(self, pattern: str) -> int:
        count = 0
        if self.l1:
            count += self.l1.invalidate_by_pattern(pattern)
        if self.l2:
            count += self.l2.invalidate_by_pattern(pattern)
        return count

    def clear(self) -> None:
        if self.l1:
            self.l1.clear()
        if self.l2:
            self.l2.clear()

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        s = {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }
        if self.l1:
            s["l1"] = self.l1.stats()
        if self.l2:
            s["l2"] = self.l2.stats()
        return s
