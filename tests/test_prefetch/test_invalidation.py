"""Tests for cache invalidation."""

import tempfile
import time
from pathlib import Path

import pytest

from jarvis.prefetch.cache import CacheTier, MultiTierCache
from jarvis.prefetch.invalidation import (
    CacheInvalidator,
    DependencyTracker,
    InvalidationEvent,
    InvalidationReason,
    InvalidationRule,
)


class TestDependencyTracker:
    """Tests for DependencyTracker."""

    def test_add_dependency(self) -> None:
        """Test adding dependencies."""
        tracker = DependencyTracker()

        tracker.add_dependency("child", "parent")

        dependents = tracker.get_dependents("parent")
        assert "child" in dependents

    def test_remove_key(self) -> None:
        """Test removing a key and its relationships."""
        tracker = DependencyTracker()

        tracker.add_dependency("child1", "parent")
        tracker.add_dependency("child2", "parent")
        tracker.remove_key("child1")

        dependents = tracker.get_dependents("parent")
        assert "child1" not in dependents
        assert "child2" in dependents

    def test_get_cascade(self) -> None:
        """Test cascading dependency resolution."""
        tracker = DependencyTracker()

        # Build dependency chain: A -> B -> C
        tracker.add_dependency("B", "A")
        tracker.add_dependency("C", "B")

        cascade = tracker.get_cascade("A")
        assert "A" in cascade
        assert "B" in cascade
        assert "C" in cascade

    def test_cascade_handles_cycles(self) -> None:
        """Test cascade handles circular dependencies."""
        tracker = DependencyTracker()

        # Create cycle: A -> B -> C -> A
        tracker.add_dependency("B", "A")
        tracker.add_dependency("C", "B")
        tracker.add_dependency("A", "C")

        cascade = tracker.get_cascade("A")
        assert "A" in cascade
        assert "B" in cascade
        assert "C" in cascade
        # Should not infinite loop

    def test_clear(self) -> None:
        """Test clearing all dependencies."""
        tracker = DependencyTracker()

        tracker.add_dependency("B", "A")
        tracker.add_dependency("C", "A")
        tracker.clear()

        assert len(tracker.get_dependents("A")) == 0


class TestInvalidationRule:
    """Tests for InvalidationRule."""

    def test_matches_condition(self) -> None:
        """Test rule matching with condition."""
        rule = InvalidationRule(
            name="test_rule",
            condition=lambda e: e.reason == InvalidationReason.NEW_MESSAGE,
        )

        event_match = InvalidationEvent(
            reason=InvalidationReason.NEW_MESSAGE,
            keys=[],
            tags=[],
        )
        event_no_match = InvalidationEvent(
            reason=InvalidationReason.CONTACT_UPDATE,
            keys=[],
            tags=[],
        )

        assert rule.matches(event_match)
        assert not rule.matches(event_no_match)


class TestCacheInvalidator:
    """Tests for CacheInvalidator."""

    @pytest.fixture
    def cache(self, tmp_path: Path) -> MultiTierCache:
        """Create a cache for testing."""
        return MultiTierCache(
            l1_maxsize=100,
            l2_db_path=tmp_path / "test_cache.db",
            l3_cache_dir=tmp_path / "l3",
        )

    @pytest.fixture
    def invalidator(self, cache: MultiTierCache) -> CacheInvalidator:
        """Create an invalidator for testing."""
        return CacheInvalidator(cache=cache, enable_cascade=True)

    def test_on_new_message(self, invalidator: CacheInvalidator, cache: MultiTierCache) -> None:
        """Test invalidation on new message."""
        # Populate cache
        cache.set("draft:chat123", {"text": "Hello"}, tags=["chat:chat123", "draft"])
        cache.set("draft:cont:chat123", {"text": "World"}, tags=["chat:chat123", "draft"])

        count = invalidator.on_new_message("chat123", "New message", is_from_me=False)

        # Should invalidate draft entries for this chat
        assert count >= 2
        assert cache.get("draft:chat123") is None
        assert cache.get("draft:cont:chat123") is None

    def test_on_contact_update(self, invalidator: CacheInvalidator, cache: MultiTierCache) -> None:
        """Test invalidation on contact update."""
        cache.set("contact:chat123", {"name": "John"}, tags=["contact", "chat:chat123"])

        count = invalidator.on_contact_update(chat_id="chat123")

        assert count >= 1
        assert cache.get("contact:chat123") is None

    def test_on_index_rebuild(self, invalidator: CacheInvalidator, cache: MultiTierCache) -> None:
        """Test invalidation on index rebuild."""
        cache.set("warm:faiss", {"loaded": True}, tags=["faiss"])
        cache.set("search:query1", {"results": []}, tags=["search"])

        count = invalidator.on_index_rebuild()

        assert count >= 1

    def test_on_model_update(self, invalidator: CacheInvalidator, cache: MultiTierCache) -> None:
        """Test invalidation on model update."""
        cache.set("warm:llm", {"loaded": True}, tags=["model", "llm"])
        cache.set("draft:chat1", {"text": "Hi"}, tags=["draft"])

        count = invalidator.on_model_update("llm")

        assert count >= 1

    def test_manual_invalidate_keys(self, invalidator: CacheInvalidator, cache: MultiTierCache) -> None:
        """Test manual invalidation by keys."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        count = invalidator.manual_invalidate(keys=["key1", "key2"])

        assert count == 2
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_manual_invalidate_pattern(self, invalidator: CacheInvalidator, cache: MultiTierCache) -> None:
        """Test manual invalidation by pattern."""
        cache.set("prefix:key1", "value1")
        cache.set("prefix:key2", "value2")
        cache.set("other:key3", "value3")

        count = invalidator.manual_invalidate(pattern="prefix:")

        assert count == 2
        assert cache.get("prefix:key1") is None
        assert cache.get("prefix:key2") is None
        assert cache.get("other:key3") == "value3"

    def test_cascade_invalidation(self, invalidator: CacheInvalidator, cache: MultiTierCache) -> None:
        """Test cascading invalidation."""
        cache.set("parent", "parent_value")
        cache.set("child", "child_value")

        invalidator.add_dependency("child", "parent")

        event = InvalidationEvent(
            reason=InvalidationReason.MANUAL,
            keys=["parent"],
            tags=[],
        )
        count = invalidator.invalidate(event)

        # Should invalidate both parent and child
        assert cache.get("parent") is None
        assert cache.get("child") is None

    def test_cleanup_expired(self, invalidator: CacheInvalidator, cache: MultiTierCache) -> None:
        """Test cleanup of expired entries."""
        cache.set("expire1", "value1", ttl_seconds=0.01)
        cache.set("expire2", "value2", ttl_seconds=0.01)
        cache.set("keep", "value3", ttl_seconds=3600)

        time.sleep(0.02)
        count = invalidator.cleanup_expired()

        assert count >= 2
        assert cache.get("keep") == "value3"

    def test_event_handler(self, invalidator: CacheInvalidator, cache: MultiTierCache) -> None:
        """Test event handler notification."""
        events_received: list[InvalidationEvent] = []

        def handler(event: InvalidationEvent) -> None:
            events_received.append(event)

        invalidator.add_event_handler(handler)

        cache.set("test", "value")
        invalidator.on_new_message("chat123", "test")

        assert len(events_received) == 1
        assert events_received[0].reason == InvalidationReason.NEW_MESSAGE

    def test_stats(self, invalidator: CacheInvalidator, cache: MultiTierCache) -> None:
        """Test stats reporting."""
        cache.set("test1", "value1")
        cache.set("test2", "value2")

        invalidator.on_new_message("chat1", "msg")
        invalidator.on_contact_update(chat_id="chat2")

        stats = invalidator.stats()
        assert stats["total_invalidations"] >= 2
        assert "by_reason" in stats
        assert stats["events_processed"] >= 2

    def test_add_rule(self, invalidator: CacheInvalidator, cache: MultiTierCache) -> None:
        """Test adding custom rule."""
        cache.set("custom:key", "value", tags=["custom_tag"])

        rule = InvalidationRule(
            name="custom_rule",
            tags=["custom_tag"],
            condition=lambda e: "custom" in str(e.metadata),
        )
        invalidator.add_rule(rule)

        event = InvalidationEvent(
            reason=InvalidationReason.MANUAL,
            keys=[],
            tags=["custom_tag"],
            metadata={"type": "custom"},
        )
        count = invalidator.invalidate(event)

        assert count >= 1


class TestInvalidationEvent:
    """Tests for InvalidationEvent."""

    def test_event_creation(self) -> None:
        """Test event creation with defaults."""
        event = InvalidationEvent(
            reason=InvalidationReason.NEW_MESSAGE,
            keys=["key1"],
            tags=["tag1"],
        )

        assert event.reason == InvalidationReason.NEW_MESSAGE
        assert event.keys == ["key1"]
        assert event.tags == ["tag1"]
        assert event.timestamp > 0
        assert event.metadata == {}

    def test_event_with_metadata(self) -> None:
        """Test event with metadata."""
        event = InvalidationEvent(
            reason=InvalidationReason.CONTACT_UPDATE,
            keys=[],
            tags=[],
            metadata={"contact_id": 123},
        )

        assert event.metadata["contact_id"] == 123
