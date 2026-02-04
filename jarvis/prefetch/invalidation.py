"""Smart cache invalidation for speculative prefetching.

Provides intelligent cache invalidation based on:
- Time-based expiration
- Event-driven invalidation (new messages, contact changes)
- Dependency tracking
- Cascading invalidation

Usage:
    invalidator = CacheInvalidator(cache=cache)
    invalidator.on_new_message(chat_id, message)
    invalidator.on_contact_update(contact_id)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from jarvis.prefetch.cache import MultiTierCache, get_cache

logger = logging.getLogger(__name__)


class InvalidationReason(str, Enum):
    """Reasons for cache invalidation."""

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
    """Represents a cache invalidation event."""

    reason: InvalidationReason
    keys: list[str]
    tags: list[str]
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InvalidationStats:
    """Statistics for cache invalidation."""

    total_invalidations: int = 0
    by_reason: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    keys_invalidated: int = 0
    cascade_invalidations: int = 0
    events_processed: int = 0

    def record(self, event: InvalidationEvent, keys_count: int) -> None:
        """Record an invalidation event."""
        self.total_invalidations += 1
        self.by_reason[event.reason.value] += 1
        self.keys_invalidated += keys_count
        self.events_processed += 1


class DependencyTracker:
    """Tracks dependencies between cache entries.

    When entry A depends on entry B, invalidating B should
    also invalidate A.
    """

    def __init__(self) -> None:
        # key -> set of keys that depend on it
        self._dependents: dict[str, set[str]] = defaultdict(set)
        # key -> set of keys it depends on
        self._dependencies: dict[str, set[str]] = defaultdict(set)
        self._lock = threading.RLock()

    def add_dependency(self, dependent_key: str, dependency_key: str) -> None:
        """Add a dependency relationship.

        Args:
            dependent_key: Key that depends on the dependency.
            dependency_key: Key being depended on.
        """
        with self._lock:
            self._dependents[dependency_key].add(dependent_key)
            self._dependencies[dependent_key].add(dependency_key)

    def remove_key(self, key: str) -> None:
        """Remove a key and its relationships.

        Args:
            key: Key to remove.
        """
        with self._lock:
            # Remove from dependents lists
            for dep in self._dependencies.get(key, set()):
                self._dependents[dep].discard(key)
            self._dependencies.pop(key, None)

            # Remove from dependencies lists
            for dep in self._dependents.get(key, set()):
                self._dependencies[dep].discard(key)
            self._dependents.pop(key, None)

    def get_dependents(self, key: str) -> set[str]:
        """Get all keys that depend on the given key.

        Args:
            key: Key to check.

        Returns:
            Set of dependent keys.
        """
        with self._lock:
            return self._dependents.get(key, set()).copy()

    def get_cascade(self, key: str, visited: set[str] | None = None) -> set[str]:
        """Get all keys that should be invalidated in a cascade.

        Follows dependency chains to find all affected keys.

        Args:
            key: Starting key.
            visited: Already visited keys (for cycle detection).

        Returns:
            Set of all keys to invalidate.
        """
        if visited is None:
            visited = set()

        if key in visited:
            return set()

        visited.add(key)
        result = {key}

        with self._lock:
            for dependent in self._dependents.get(key, set()):
                result |= self.get_cascade(dependent, visited)

        return result

    def clear(self) -> None:
        """Clear all dependencies."""
        with self._lock:
            self._dependents.clear()
            self._dependencies.clear()


class InvalidationRule:
    """A rule that determines when to invalidate cache entries."""

    def __init__(
        self,
        name: str,
        pattern: str | None = None,
        tags: list[str] | None = None,
        condition: Callable[[InvalidationEvent], bool] | None = None,
    ) -> None:
        """Initialize invalidation rule.

        Args:
            name: Rule name.
            pattern: Key prefix pattern to match.
            tags: Tags to match.
            condition: Optional condition function.
        """
        self.name = name
        self.pattern = pattern
        self.tags = tags or []
        self.condition = condition

    def matches(self, event: InvalidationEvent) -> bool:
        """Check if this rule matches the event.

        Args:
            event: Invalidation event.

        Returns:
            True if rule should be applied.
        """
        if self.condition and not self.condition(event):
            return False
        return True

    def get_affected_keys(self, event: InvalidationEvent, cache: MultiTierCache) -> list[str]:
        """Get keys that should be invalidated.

        Args:
            event: Invalidation event.
            cache: Cache to search.

        Returns:
            List of keys to invalidate.
        """
        keys: list[str] = []

        # Match by explicit keys
        keys.extend(event.keys)

        # Match by tags
        for tag in self.tags:
            if tag in event.tags:
                # Find all keys with this tag
                keys.extend(self._find_keys_by_tag(cache, tag))

        # Match by pattern
        if self.pattern:
            keys.extend(self._find_keys_by_pattern(cache, self.pattern))

        return list(set(keys))

    def _find_keys_by_tag(self, cache: MultiTierCache, tag: str) -> list[str]:
        """Find cache keys with a specific tag."""
        # The cache doesn't expose a direct tag lookup, so we use invalidate_by_tag
        # and let the cache handle it
        return []  # Tags are handled directly by the cache

    def _find_keys_by_pattern(self, cache: MultiTierCache, pattern: str) -> list[str]:
        """Find cache keys matching a pattern."""
        # Similar to tags, patterns are handled by the cache
        return []


class CacheInvalidator:
    """Smart cache invalidation manager.

    Handles cache invalidation based on events and rules.
    """

    def __init__(
        self,
        cache: MultiTierCache | None = None,
        enable_cascade: bool = True,
    ) -> None:
        """Initialize invalidator.

        Args:
            cache: Multi-tier cache to invalidate.
            enable_cascade: Whether to follow dependency chains.
        """
        self._cache = cache or get_cache()
        self._enable_cascade = enable_cascade
        self._dependencies = DependencyTracker()
        self._rules: list[InvalidationRule] = []
        self._stats = InvalidationStats()
        self._lock = threading.RLock()
        self._event_handlers: list[Callable[[InvalidationEvent], None]] = []

        # Register default rules
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register default invalidation rules."""
        # New message invalidates drafts for that chat
        self.add_rule(
            InvalidationRule(
                name="new_message_draft",
                tags=["draft"],
                condition=lambda e: e.reason == InvalidationReason.NEW_MESSAGE,
            )
        )

        # Message sent invalidates active conversation predictions
        self.add_rule(
            InvalidationRule(
                name="message_sent_active",
                tags=["active"],
                condition=lambda e: e.reason == InvalidationReason.MESSAGE_SENT,
            )
        )

        # Contact update invalidates contact profiles
        self.add_rule(
            InvalidationRule(
                name="contact_update",
                tags=["contact"],
                condition=lambda e: e.reason == InvalidationReason.CONTACT_UPDATE,
            )
        )

        # Index rebuild invalidates search results
        self.add_rule(
            InvalidationRule(
                name="index_rebuild",
                tags=["search", "embedding"],
                condition=lambda e: e.reason == InvalidationReason.INDEX_REBUILD,
            )
        )

        # Model update invalidates model-dependent entries
        self.add_rule(
            InvalidationRule(
                name="model_update",
                tags=["draft", "embedding", "model"],
                condition=lambda e: e.reason == InvalidationReason.MODEL_UPDATE,
            )
        )

    def add_rule(self, rule: InvalidationRule) -> None:
        """Add an invalidation rule.

        Args:
            rule: Rule to add.
        """
        with self._lock:
            self._rules.append(rule)

    def add_event_handler(self, handler: Callable[[InvalidationEvent], None]) -> None:
        """Add an event handler for invalidation events.

        Args:
            handler: Handler function.
        """
        with self._lock:
            self._event_handlers.append(handler)

    def add_dependency(self, dependent_key: str, dependency_key: str) -> None:
        """Add a cache entry dependency.

        Args:
            dependent_key: Key that depends on the dependency.
            dependency_key: Key being depended on.
        """
        self._dependencies.add_dependency(dependent_key, dependency_key)

    def invalidate(self, event: InvalidationEvent) -> int:
        """Process an invalidation event.

        Args:
            event: Invalidation event.

        Returns:
            Number of entries invalidated.
        """
        with self._lock:
            keys_to_invalidate: set[str] = set()
            tags_to_invalidate: set[str] = set()

            # Collect keys from event
            keys_to_invalidate.update(event.keys)

            # Apply rules
            for rule in self._rules:
                if rule.matches(event):
                    # Collect tags from rule
                    for tag in rule.tags:
                        if tag in event.tags or not event.tags:
                            tags_to_invalidate.add(tag)

            # Handle cascading invalidation
            if self._enable_cascade:
                cascade_keys: set[str] = set()
                for key in keys_to_invalidate:
                    cascade_keys |= self._dependencies.get_cascade(key)

                if len(cascade_keys) > len(keys_to_invalidate):
                    self._stats.cascade_invalidations += len(cascade_keys) - len(keys_to_invalidate)
                    keys_to_invalidate |= cascade_keys

            # Perform invalidation
            count = 0

            # Invalidate by keys
            for key in keys_to_invalidate:
                if self._cache.remove(key):
                    count += 1
                    self._dependencies.remove_key(key)

            # Invalidate by tags
            for tag in tags_to_invalidate:
                count += self._cache.invalidate_by_tag(tag)

            # Invalidate by pattern (for chat-specific invalidation)
            chat_id = event.metadata.get("chat_id")
            if chat_id:
                count += self._cache.invalidate_by_pattern(f"draft:{chat_id}")
                count += self._cache.invalidate_by_pattern(f"embed:ctx:{chat_id}")

            # Update stats
            self._stats.record(event, count)

            # Notify handlers
            for handler in self._event_handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.warning(f"Event handler failed: {e}")

            logger.debug(
                f"Invalidated {count} entries for {event.reason.value} "
                f"(keys={len(keys_to_invalidate)}, tags={len(tags_to_invalidate)})"
            )

            return count

    def on_new_message(
        self,
        chat_id: str,
        message_text: str | None = None,
        is_from_me: bool = False,
    ) -> int:
        """Handle new message event.

        Args:
            chat_id: Chat identifier.
            message_text: Message text (optional).
            is_from_me: Whether message was from user.

        Returns:
            Number of entries invalidated.
        """
        reason = InvalidationReason.MESSAGE_SENT if is_from_me else InvalidationReason.NEW_MESSAGE
        event = InvalidationEvent(
            reason=reason,
            keys=[f"draft:{chat_id}", f"draft:cont:{chat_id}", f"draft:focus:{chat_id}"],
            tags=[f"chat:{chat_id}", "draft", "active"],
            metadata={
                "chat_id": chat_id,
                "message_text": message_text,
                "is_from_me": is_from_me,
            },
        )
        return self.invalidate(event)

    def on_contact_update(self, contact_id: int | None = None, chat_id: str | None = None) -> int:
        """Handle contact update event.

        Args:
            contact_id: Contact ID (optional).
            chat_id: Chat ID (optional).

        Returns:
            Number of entries invalidated.
        """
        keys: list[str] = []
        tags: list[str] = ["contact"]

        if contact_id:
            keys.append(f"contact:{contact_id}")
        if chat_id:
            keys.append(f"contact:{chat_id}")
            tags.append(f"chat:{chat_id}")

        event = InvalidationEvent(
            reason=InvalidationReason.CONTACT_UPDATE,
            keys=keys,
            tags=tags,
            metadata={"contact_id": contact_id, "chat_id": chat_id},
        )
        return self.invalidate(event)

    def on_index_rebuild(self) -> int:
        """Handle FAISS index rebuild event.

        Returns:
            Number of entries invalidated.
        """
        event = InvalidationEvent(
            reason=InvalidationReason.INDEX_REBUILD,
            keys=["warm:faiss"],
            tags=["search", "embedding", "faiss"],
            metadata={},
        )
        return self.invalidate(event)

    def on_model_update(self, model_type: str) -> int:
        """Handle model update event.

        Args:
            model_type: Type of model updated (llm, embeddings, etc.).

        Returns:
            Number of entries invalidated.
        """
        tags = ["model", model_type]
        if model_type == "llm":
            tags.extend(["draft"])
        elif model_type == "embeddings":
            tags.extend(["embedding"])

        event = InvalidationEvent(
            reason=InvalidationReason.MODEL_UPDATE,
            keys=[f"warm:{model_type}"],
            tags=tags,
            metadata={"model_type": model_type},
        )
        return self.invalidate(event)

    def manual_invalidate(
        self,
        keys: list[str] | None = None,
        tags: list[str] | None = None,
        pattern: str | None = None,
    ) -> int:
        """Manually invalidate cache entries.

        Args:
            keys: Specific keys to invalidate.
            tags: Tags to invalidate.
            pattern: Key prefix pattern.

        Returns:
            Number of entries invalidated.
        """
        count = 0

        if keys:
            for key in keys:
                if self._cache.remove(key):
                    count += 1
                    self._dependencies.remove_key(key)

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
        """Clean up expired cache entries.

        Returns:
            Number of entries cleaned up.
        """
        count = self._cache.cleanup_expired()

        event = InvalidationEvent(
            reason=InvalidationReason.EXPIRED,
            keys=[],
            tags=[],
            metadata={"count": count},
        )

        self._stats.record(event, count)
        return count

    def stats(self) -> dict[str, Any]:
        """Get invalidation statistics."""
        with self._lock:
            return {
                "total_invalidations": self._stats.total_invalidations,
                "by_reason": dict(self._stats.by_reason),
                "keys_invalidated": self._stats.keys_invalidated,
                "cascade_invalidations": self._stats.cascade_invalidations,
                "events_processed": self._stats.events_processed,
                "rules_count": len(self._rules),
            }


# Singleton instance
_invalidator: CacheInvalidator | None = None
_invalidator_lock = threading.Lock()


def get_invalidator() -> CacheInvalidator:
    """Get or create singleton invalidator instance."""
    global _invalidator
    with _invalidator_lock:
        if _invalidator is None:
            _invalidator = CacheInvalidator()
        return _invalidator


def reset_invalidator() -> None:
    """Reset singleton invalidator."""
    global _invalidator
    with _invalidator_lock:
        _invalidator = None
