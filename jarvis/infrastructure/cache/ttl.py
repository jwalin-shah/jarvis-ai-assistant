from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from jarvis.infrastructure.cache.memory import _SENTINEL, MemoryBackend

T = TypeVar("T")

CacheKey = str


class TTLCache:
    """Thread-safe in-memory TTL cache with the legacy tuple-get interface."""

    def __init__(self, ttl_seconds: float = 30.0, maxsize: int = 128) -> None:
        self._backend = MemoryBackend(ttl=ttl_seconds, maxsize=maxsize)

    def get(self, key: str) -> tuple[bool, Any]:
        val = self._backend.get(key)
        if val is not _SENTINEL:
            return True, val
        return False, None

    def set(self, key: str, value: Any) -> None:
        self._backend.set(key, value)

    def delete(self, key: str) -> None:
        self._backend.delete(key)

    def invalidate(self, key: str | None = None) -> None:
        if key is None:
            self._backend.clear()
        else:
            self._backend.delete(key)

    def clear(self) -> None:
        self._backend.clear()

    def get_or_set(self, key: str, factory: Callable[[], T], ttl: float | None = None) -> T:
        return self._backend.get_or_set(key, factory, ttl=ttl)

    def stats(self) -> dict[str, Any]:
        s = self._backend.stats()
        return {
            "size": s["entries"],
            "maxsize": s["maxsize"],
            "ttl_seconds": s["ttl_seconds"],
            "hits": s["hits"],
            "misses": s["misses"],
            "hit_rate": s["hit_rate"],
        }
