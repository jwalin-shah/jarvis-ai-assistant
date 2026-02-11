"""Cache interface contracts.

Defines pluggable cache protocols that all cache implementations should conform to.
This enables swapping cache backends without changing consumer code.
"""

from __future__ import annotations

from typing import Any, Protocol


class Cache(Protocol):
    """Pluggable cache interface.

    All JARVIS cache implementations (TTLCache, MultiTierCache, etc.)
    conform to this protocol, enabling consumers to accept any cache backend.

    Minimum required interface:
        - get(key) -> (found, value)
        - set(key, value) -> None
        - clear() -> None
    """

    def get(self, key: str) -> tuple[bool, Any]:
        """Get a value from the cache.

        Args:
            key: Cache key.

        Returns:
            Tuple of (found, value). found is False if key doesn't exist or expired.
        """
        ...

    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        ...

    def clear(self) -> None:
        """Clear all cache entries."""
        ...


class CacheWithStats(Cache, Protocol):
    """Cache that also reports statistics."""

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats (size, hit_rate, etc.).
        """
        ...


class CacheWithInvalidation(Cache, Protocol):
    """Cache that supports key-level invalidation."""

    def invalidate(self, key: str | None = None) -> None:
        """Invalidate cache entries.

        Args:
            key: Specific key to invalidate, or None to clear all.
        """
        ...

    def delete(self, key: str) -> None:
        """Remove a specific key from cache.

        Args:
            key: Cache key to remove.
        """
        ...
