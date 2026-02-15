"""Tests for cache invalidation."""

import time

import pytest

from jarvis.prefetch.cache import PrefetchCache
from jarvis.prefetch.invalidation import CacheInvalidator


class TestCacheInvalidator:
    """Tests for CacheInvalidator."""

    @pytest.fixture
    def cache(self) -> PrefetchCache:
        """Create a cache for testing."""
        return PrefetchCache(maxsize=100)

    @pytest.fixture
    def invalidator(self, cache: PrefetchCache) -> CacheInvalidator:
        """Create an invalidator for testing."""
        return CacheInvalidator(cache=cache)

    def test_on_new_message(self, invalidator: CacheInvalidator, cache: PrefetchCache) -> None:
        """Test invalidation on new message."""
        cache.set("draft:chat123", {"text": "Hello"}, tags=["chat:chat123", "draft"])
        cache.set("draft:cont:chat123", {"text": "World"}, tags=["chat:chat123", "draft"])

        count = invalidator.on_new_message("chat123", "New message", is_from_me=False)

        assert count >= 2
        assert cache.get("draft:chat123") is None
        assert cache.get("draft:cont:chat123") is None

    def test_on_new_message_from_me(
        self, invalidator: CacheInvalidator, cache: PrefetchCache
    ) -> None:
        """Test invalidation tracks reason correctly for sent messages."""
        cache.set("draft:chat456", {"text": "Draft"}, tags=["chat:chat456"])

        invalidator.on_new_message("chat456", "My reply", is_from_me=True)

        stats = invalidator.stats()
        assert stats["by_reason"]["message_sent"] >= 1

    def test_on_new_message_invalidates_embed_keys(
        self, invalidator: CacheInvalidator, cache: PrefetchCache
    ) -> None:
        """Test that embed:ctx: keys are also invalidated."""
        cache.set("embed:ctx:chat789", {"embeddings": [1, 2, 3]})

        _ = invalidator.on_new_message("chat789", "test")
        assert cache.get("embed:ctx:chat789") is None

    def test_manual_invalidate_keys(
        self, invalidator: CacheInvalidator, cache: PrefetchCache
    ) -> None:
        """Test manual invalidation by keys."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        count = invalidator.manual_invalidate(keys=["key1", "key2"])

        assert count == 2
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_manual_invalidate_tags(
        self, invalidator: CacheInvalidator, cache: PrefetchCache
    ) -> None:
        """Test manual invalidation by tags."""
        cache.set("a", "1", tags=["group1"])
        cache.set("b", "2", tags=["group1"])
        cache.set("c", "3", tags=["group2"])

        count = invalidator.manual_invalidate(tags=["group1"])
        assert count == 2
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.get("c") == "3"

    def test_manual_invalidate_pattern(
        self, invalidator: CacheInvalidator, cache: PrefetchCache
    ) -> None:
        """Test manual invalidation by pattern."""
        cache.set("prefix:key1", "value1")
        cache.set("prefix:key2", "value2")
        cache.set("other:key3", "value3")

        count = invalidator.manual_invalidate(pattern="prefix:")

        assert count == 2
        assert cache.get("prefix:key1") is None
        assert cache.get("prefix:key2") is None
        assert cache.get("other:key3") == "value3"

    def test_cleanup_expired(self, invalidator: CacheInvalidator, cache: PrefetchCache) -> None:
        """Test cleanup of expired entries."""
        cache.set("expire1", "value1", ttl_seconds=0.01)
        cache.set("expire2", "value2", ttl_seconds=0.01)
        cache.set("keep", "value3", ttl_seconds=3600)

        time.sleep(0.02)
        count = invalidator.cleanup_expired()

        assert count >= 2
        assert cache.get("keep") == "value3"

    def test_stats(self, invalidator: CacheInvalidator, cache: PrefetchCache) -> None:
        """Test stats reporting."""
        cache.set("test1", "value1")
        cache.set("test2", "value2")

        invalidator.on_new_message("chat1", "msg")
        invalidator.manual_invalidate(keys=["test2"])

        stats = invalidator.stats()
        assert stats["total_invalidations"] >= 2
        assert "by_reason" in stats
        assert "new_message" in stats["by_reason"]
        assert "manual" in stats["by_reason"]

    def test_stats_keys_invalidated(
        self, invalidator: CacheInvalidator, cache: PrefetchCache
    ) -> None:
        """Test that keys_invalidated is tracked."""
        cache.set("draft:chatA", "val", tags=["chat:chatA"])
        cache.set("draft:cont:chatA", "val", tags=["chat:chatA"])

        invalidator.on_new_message("chatA", "hi")

        stats = invalidator.stats()
        assert stats["keys_invalidated"] >= 2

    def test_cleanup_no_expired_does_not_increment_stats(
        self, invalidator: CacheInvalidator, cache: PrefetchCache
    ) -> None:
        """Test that cleanup with no expired entries does not increment stats."""
        cache.set("fresh", "value", ttl_seconds=3600)

        count = invalidator.cleanup_expired()
        assert count == 0

        stats = invalidator.stats()
        assert stats["total_invalidations"] == 0
