from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from jarvis.infrastructure.cache.base import CacheBackend


@dataclass
class _Entry:
    """Internal cache entry with TTL."""
    value: Any
    expires_at: float
    tags: list[str] = field(default_factory=list)


class MemoryBackend(CacheBackend):
    """In-memory cache backend with TTL and tag support."""

    def __init__(self, ttl: float = 300.0, maxsize: int = 1000) -> None:
        self._data: dict[str, _Entry] = {}
        self._maxsize = maxsize
        self._default_ttl = ttl
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._misses += 1
                return None
            if time.time() > entry.expires_at:
                del self._data[key]
                self._misses += 1
                return None
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: float | None = None, tags: list[str] | None = None) -> None:
        ttl = ttl if ttl is not None else self._default_ttl
        with self._lock:
            self._data[key] = _Entry(
                value=value,
                expires_at=time.time() + ttl,
                tags=tags or [],
            )
            self._evict_if_full()

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def invalidate_by_tag(self, tag: str) -> int:
        with self._lock:
            to_remove = [k for k, e in self._data.items() if tag in e.tags]
            for k in to_remove:
                del self._data[k]
            return len(to_remove)

    def invalidate_by_pattern(self, pattern: str) -> int:
        with self._lock:
            to_remove = [k for k in self._data if k.startswith(pattern)]
            for k in to_remove:
                del self._data[k]
            return len(to_remove)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._data),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }

    def _evict_if_full(self) -> None:
        while len(self._data) > self._maxsize:
            # Remove the entry that expires soonest
            oldest_key = min(self._data, key=lambda k: self._data[k].expires_at)
            del self._data[oldest_key]
            self._evictions += 1
