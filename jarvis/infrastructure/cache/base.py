from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CacheBackend(ABC):
    """Abstract base class for all cache backends."""

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Get a value from the cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Set a value in the cache with an optional TTL."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a value from the cache. Returns True if deleted."""
        pass

    @abstractmethod
    def invalidate_by_tag(self, tag: str) -> int:
        """Remove all entries that have the given tag. Returns count removed."""
        pass

    @abstractmethod
    def invalidate_by_pattern(self, pattern: str) -> int:
        """Remove all entries whose key starts with pattern. Returns count removed."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all values from the cache."""
        pass

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Get statistics for the cache backend."""
        pass
