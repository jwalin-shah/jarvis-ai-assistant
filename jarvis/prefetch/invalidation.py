"""Simple cache invalidation for speculative prefetching.

The core operation is: new message arrives -> remove stale drafts for that chat.
No rules engine, no dependency tracking, no cascading needed.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from jarvis.prefetch.cache import PrefetchCache, get_cache

logger = logging.getLogger(__name__)


@dataclass
class InvalidationStats:
    """Simple invalidation counters."""

    total_invalidations: int = 0
    keys_invalidated: int = 0
    by_reason: dict[str, int] = field(default_factory=lambda: defaultdict(int))


class CacheInvalidator:
    """Invalidates cache entries in response to events."""

    def __init__(self, cache: PrefetchCache | None = None, **_kwargs: Any) -> None:
        self._cache = cache or get_cache()
        self._stats = InvalidationStats()

    def on_new_message(
        self,
        chat_id: str,
        message_text: str | None = None,
        is_from_me: bool = False,
    ) -> int:
        """Invalidate cached drafts when a new message arrives for a chat."""
        count = 0
        # Remove all draft variants for this chat
        for prefix in ("draft:", "draft:cont:", "draft:focus:", "draft:tod:", "draft:hover:"):
            if self._cache.remove(f"{prefix}{chat_id}"):
                count += 1
        # Also remove by pattern and tag for any non-standard keys
        count += self._cache.invalidate_by_pattern(f"draft:{chat_id}")
        count += self._cache.invalidate_by_pattern(f"embed:ctx:{chat_id}")
        count += self._cache.invalidate_by_tag(f"chat:{chat_id}")

        reason = "message_sent" if is_from_me else "new_message"
        self._stats.total_invalidations += 1
        self._stats.by_reason[reason] += 1
        self._stats.keys_invalidated += count
        logger.debug("Invalidated %d entries for %s on chat %s", count, reason, chat_id)
        return count

    def manual_invalidate(
        self,
        keys: list[str] | None = None,
        tags: list[str] | None = None,
        pattern: str | None = None,
    ) -> int:
        """Manually invalidate cache entries."""
        count = 0
        if keys:
            for key in keys:
                if self._cache.remove(key):
                    count += 1
        if tags:
            for tag in tags:
                count += self._cache.invalidate_by_tag(tag)
        if pattern:
            count += self._cache.invalidate_by_pattern(pattern)

        self._stats.total_invalidations += 1
        self._stats.by_reason["manual"] += 1
        self._stats.keys_invalidated += count
        return count

    def cleanup_expired(self) -> int:
        """Clean up expired cache entries."""
        count = self._cache.cleanup_expired()
        if count > 0:
            self._stats.total_invalidations += 1
            self._stats.by_reason["expired"] += 1
            self._stats.keys_invalidated += count
        return count

    def stats(self) -> dict[str, Any]:
        """Return invalidation statistics."""
        return {
            "total_invalidations": self._stats.total_invalidations,
            "keys_invalidated": self._stats.keys_invalidated,
            "by_reason": dict(self._stats.by_reason),
        }


# ---------------------------------------------------------------------------
# Backwards-compatible aliases for old imports
# ---------------------------------------------------------------------------

from enum import Enum  # noqa: E402


class InvalidationReason(str, Enum):
    """Kept for import compatibility."""

    EXPIRED = "expired"
    NEW_MESSAGE = "new_message"
    MESSAGE_SENT = "message_sent"
    CONTACT_UPDATE = "contact_update"
    INDEX_REBUILD = "index_rebuild"
    MODEL_UPDATE = "model_update"
    MANUAL = "manual"
    CASCADE = "cascade"


@dataclass
class InvalidationEvent:
    """Kept for import compatibility."""

    reason: str = ""
    keys: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


class InvalidationRule:
    """Stub for import compatibility."""

    pass


class DependencyTracker:
    """Stub for import compatibility."""

    pass


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

from jarvis.utils.singleton import thread_safe_singleton  # noqa: E402


@thread_safe_singleton
def get_invalidator() -> CacheInvalidator:
    """Get or create singleton invalidator instance."""
    return CacheInvalidator()


def reset_invalidator() -> None:
    """Reset singleton invalidator."""
    get_invalidator.reset()  # type: ignore[attr-defined]
