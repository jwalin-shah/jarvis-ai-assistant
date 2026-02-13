"""Tests for the PrefetchCache (simple in-memory TTL cache)."""

import threading
import time

import pytest

from jarvis.prefetch.cache import (
    CacheStats,
    MultiTierCache,
    PrefetchCache,
)


class TestPrefetchCache:
    """Tests for PrefetchCache (aliased as MultiTierCache)."""

    def test_alias(self) -> None:
        """MultiTierCache is an alias for PrefetchCache."""
        assert MultiTierCache is PrefetchCache

    @pytest.fixture
    def cache(self) -> PrefetchCache:
        """Create a fresh cache for each test."""
        return PrefetchCache(maxsize=100, default_ttl=300.0)

    def test_set_and_get(self, cache: PrefetchCache) -> None:
        """Test basic set/get operations."""
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_nonexistent(self, cache: PrefetchCache) -> None:
        """Test get returns None for missing keys."""
        assert cache.get("nonexistent") is None

    def test_set_with_tags(self, cache: PrefetchCache) -> None:
        """Test setting values with tags."""
        cache.set("tagged", "value", tags=["tag1", "tag2"])
        assert cache.get("tagged") == "value"

    def test_set_with_ttl(self, cache: PrefetchCache) -> None:
        """Test TTL expiry."""
        cache.set("expire", "value", ttl_seconds=0.01)
        assert cache.get("expire") == "value"
        time.sleep(0.02)
        assert cache.get("expire") is None

    def test_set_ignores_extra_kwargs(self, cache: PrefetchCache) -> None:
        """Test that extra kwargs (tier, etc.) are accepted and ignored."""
        cache.set("key", "value", tier="L1", something_else=42)
        assert cache.get("key") == "value"

    def test_remove(self, cache: PrefetchCache) -> None:
        """Test remove operation."""
        cache.set("key", "value")
        assert cache.remove("key") is True
        assert cache.get("key") is None
        assert cache.remove("key") is False

    def test_invalidate_by_tag(self, cache: PrefetchCache) -> None:
        """Test invalidation by tag."""
        for i in range(5):
            cache.set(f"tagged{i}", f"value{i}", tags=["common_tag"])

        count = cache.invalidate_by_tag("common_tag")
        assert count == 5

        for i in range(5):
            assert cache.get(f"tagged{i}") is None

    def test_invalidate_by_tag_partial(self, cache: PrefetchCache) -> None:
        """Test that invalidate_by_tag only removes entries with that tag."""
        cache.set("a", "1", tags=["x"])
        cache.set("b", "2", tags=["y"])
        cache.set("c", "3", tags=["x", "y"])

        count = cache.invalidate_by_tag("x")
        assert count == 2
        assert cache.get("a") is None
        assert cache.get("b") == "2"
        assert cache.get("c") is None

    def test_invalidate_by_pattern(self, cache: PrefetchCache) -> None:
        """Test invalidation by key prefix pattern."""
        for i in range(5):
            cache.set(f"prefix:key{i}", f"value{i}")
            cache.set(f"other:key{i}", f"value{i}")

        count = cache.invalidate_by_pattern("prefix:")
        assert count == 5

        # Other keys should remain
        assert cache.get("other:key0") == "value0"

    def test_clear(self, cache: PrefetchCache) -> None:
        """Test clear removes all entries and resets stats."""
        for i in range(5):
            cache.set(f"key{i}", i)
        cache.clear()

        stats = cache.stats()
        assert stats["entries"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_cleanup_expired(self, cache: PrefetchCache) -> None:
        """Test cleanup of expired entries."""
        for i in range(5):
            cache.set(f"expire{i}", f"value{i}", ttl_seconds=0.01)
        cache.set("keep", "value", ttl_seconds=3600)

        time.sleep(0.02)
        cleaned = cache.cleanup_expired()
        assert cleaned == 5

        assert cache.get("keep") == "value"

    def test_eviction_on_maxsize(self) -> None:
        """Test that oldest entries are evicted when maxsize is exceeded."""
        cache = PrefetchCache(maxsize=3)

        cache.set("key0", "val0", ttl_seconds=100)
        cache.set("key1", "val1", ttl_seconds=50)
        cache.set("key2", "val2", ttl_seconds=200)

        # Full at 3. Adding a 4th should evict the one with soonest expiry (key1, ttl=50).
        cache.set("key3", "val3", ttl_seconds=150)

        assert cache.stats()["entries"] == 3
        assert cache.stats()["evictions"] == 1
        # key1 had the soonest expiry, so it should be evicted
        assert cache.get("key1") is None
        assert cache.get("key0") is not None
        assert cache.get("key3") is not None

    def test_stats(self, cache: PrefetchCache) -> None:
        """Test stats returns correct flat dict."""
        cache.set("test1", "value1")
        cache.set("test2", "value2")
        cache.get("test1")  # hit
        cache.get("test2")  # hit
        cache.get("nonexistent")  # miss

        stats = cache.stats()
        assert stats["entries"] == 2
        assert stats["maxsize"] == 100
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2.0 / 3.0)

    def test_stats_evictions_tracked(self) -> None:
        """Test that evictions are tracked in stats."""
        cache = PrefetchCache(maxsize=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # should evict one

        stats = cache.stats()
        assert stats["evictions"] == 1

    def test_thread_safety(self) -> None:
        """Test thread-safe operations."""
        cache = PrefetchCache(maxsize=1000)
        errors: list[Exception] = []

        def writer(start: int) -> None:
            try:
                for i in range(100):
                    cache.set(f"key{start + i}", start + i)
            except Exception as e:
                errors.append(e)

        def reader(start: int) -> None:
            try:
                for i in range(100):
                    cache.get(f"key{start + i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(100,)),
            threading.Thread(target=reader, args=(0,)),
            threading.Thread(target=reader, args=(100,)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_backwards_compat_kwargs(self) -> None:
        """Test that old MultiTierCache kwargs are accepted without error."""
        cache = PrefetchCache(
            l1_maxsize=50,
            l1_max_bytes=1024,
            l2_db_path="/tmp/test.db",
            l3_cache_dir="/tmp/l3",
            l3_max_bytes=1024,
            auto_promote=True,
            promote_threshold=3,
        )
        # l1_maxsize should be used as the actual maxsize
        assert cache.stats()["maxsize"] == 50


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0

    def test_hit_rate(self) -> None:
        """Test hit rate calculation."""
        stats = CacheStats(hits=80, misses=20)
        assert stats.hit_rate == pytest.approx(0.8)

    def test_hit_rate_zero_total(self) -> None:
        """Test hit rate with zero lookups."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_all_hits(self) -> None:
        """Test hit rate with 100% hits."""
        stats = CacheStats(hits=50, misses=0)
        assert stats.hit_rate == 1.0
