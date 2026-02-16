from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from jarvis.infrastructure.cache.base import CacheBackend

T = TypeVar("T")

# Sentinel for cache misses to allow caching None values
_SENTINEL = object()


@dataclass
class _Entry:
    """Internal cache entry with TTL."""

    value: Any
    expires_at: float
    tags: list[str] = field(default_factory=list)


class MemoryBackend(CacheBackend):
    """In-memory cache backend with TTL, tag support, and single-flight protection."""

    def __init__(self, ttl: float = 300.0, maxsize: int = 1000) -> None:
        self._data: OrderedDict[str, _Entry] = OrderedDict()
        self._maxsize = maxsize
        self._default_ttl = ttl
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        # Single-flight: maps key -> Event for in-progress computations
        self._inflight: dict[str, threading.Event] = {}
        self._inflight_lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        """Get a value from the cache. Returns _SENTINEL on miss."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._misses += 1
                return _SENTINEL
            if time.time() > entry.expires_at:
                del self._data[key]
                self._misses += 1
                return _SENTINEL
            self._hits += 1
            self._data.move_to_end(key)
            return entry.value

    def set(
        self, key: str, value: Any, ttl: float | None = None, tags: list[str] | None = None
    ) -> None:
        ttl = ttl if ttl is not None else self._default_ttl
        with self._lock:
            if key in self._data:
                del self._data[key]
            elif len(self._data) >= self._maxsize:
                self._data.popitem(last=False)
                self._evictions += 1

            self._data[key] = _Entry(
                value=value,
                expires_at=time.time() + ttl,
                tags=tags or [],
            )

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

    def get_or_set(self, key: str, factory: Callable[[], T], ttl: float | None = None) -> T:
        val = self.get(key)
        if val is not _SENTINEL:
            return val

        with self._inflight_lock:
            if key in self._inflight:
                event = self._inflight[key]
            else:
                event = threading.Event()
                self._inflight[key] = event
                event = None

        if event is not None:
            event.wait(timeout=60.0)
            val = self.get(key)
            if val is not _SENTINEL:
                return val

        try:
            result = factory()
            self.set(key, result, ttl=ttl)
            return result
        finally:
            with self._inflight_lock:
                evt = self._inflight.pop(key, None)
                if evt is not None:
                    evt.set()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._data),
                "maxsize": self._maxsize,
                "ttl_seconds": self._default_ttl,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }
