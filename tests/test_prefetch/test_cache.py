"""Tests for the multi-tier cache system."""

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from jarvis.prefetch.cache import (
    CacheEntry,
    CacheStats,
    CacheTier,
    L1Cache,
    L2Cache,
    L3Cache,
    MultiTierCache,
)


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_entry_not_expired(self) -> None:
        """Test entry is not expired when fresh."""
        entry = CacheEntry(
            key="test",
            value="value",
            tier=CacheTier.L1,
            ttl_seconds=300,
        )
        assert not entry.is_expired

    def test_entry_expired(self) -> None:
        """Test entry is expired after TTL."""
        entry = CacheEntry(
            key="test",
            value="value",
            tier=CacheTier.L1,
            ttl_seconds=0.01,  # Very short TTL
            created_at=time.time() - 1,  # Created 1 second ago
        )
        assert entry.is_expired

    def test_touch_updates_metadata(self) -> None:
        """Test touch updates access time and count."""
        entry = CacheEntry(key="test", value="value", tier=CacheTier.L1)
        original_count = entry.access_count
        original_time = entry.accessed_at

        time.sleep(0.01)
        entry.touch()

        assert entry.access_count == original_count + 1
        assert entry.accessed_at > original_time


class TestL1Cache:
    """Tests for L1 (in-memory) cache."""

    def test_get_set(self) -> None:
        """Test basic get/set operations."""
        cache = L1Cache(maxsize=100)
        entry = CacheEntry(key="test", value="value", tier=CacheTier.L1)

        cache.set("test", entry)
        result = cache.get("test")

        assert result is not None
        assert result.value == "value"

    def test_get_nonexistent(self) -> None:
        """Test get returns None for missing keys."""
        cache = L1Cache()
        assert cache.get("nonexistent") is None

    def test_expiration(self) -> None:
        """Test entries expire based on TTL."""
        cache = L1Cache()
        entry = CacheEntry(
            key="test",
            value="value",
            tier=CacheTier.L1,
            ttl_seconds=0.01,
        )
        cache.set("test", entry)

        # Should be available immediately
        assert cache.get("test") is not None

        # Should expire after TTL
        time.sleep(0.1)
        assert cache.get("test") is None

    def test_lru_eviction(self) -> None:
        """Test LRU eviction when at capacity."""
        cache = L1Cache(maxsize=3)

        for i in range(3):
            entry = CacheEntry(key=f"key{i}", value=i, tier=CacheTier.L1)
            cache.set(f"key{i}", entry)

        # Access key0 to make it recently used
        cache.get("key0")

        # Add new entry, should evict key1 (least recently used)
        entry = CacheEntry(key="key3", value=3, tier=CacheTier.L1)
        cache.set("key3", entry)

        assert cache.get("key0") is not None  # Recently used
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") is not None  # Not evicted
        assert cache.get("key3") is not None  # New entry

    def test_remove(self) -> None:
        """Test remove operation."""
        cache = L1Cache()
        entry = CacheEntry(key="test", value="value", tier=CacheTier.L1)
        cache.set("test", entry)

        assert cache.remove("test")
        assert cache.get("test") is None
        assert not cache.remove("test")  # Already removed

    def test_clear(self) -> None:
        """Test clear operation."""
        cache = L1Cache()
        for i in range(5):
            entry = CacheEntry(key=f"key{i}", value=i, tier=CacheTier.L1)
            cache.set(f"key{i}", entry)

        cache.clear()
        assert len(cache.keys()) == 0

    def test_stats(self) -> None:
        """Test stats reporting."""
        cache = L1Cache(maxsize=100, max_bytes=1024)
        entry = CacheEntry(key="test", value="value", tier=CacheTier.L1, size_bytes=10)
        cache.set("test", entry)

        stats = cache.stats()
        assert stats["entries"] == 1
        assert stats["maxsize"] == 100
        assert stats["bytes"] == 10

    def test_thread_safety(self) -> None:
        """Test thread-safe operations."""
        cache = L1Cache(maxsize=1000)
        errors: list[Exception] = []

        def writer(start: int) -> None:
            try:
                for i in range(100):
                    entry = CacheEntry(
                        key=f"key{start + i}",
                        value=start + i,
                        tier=CacheTier.L1,
                    )
                    cache.set(f"key{start + i}", entry)
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


class TestL2Cache:
    """Tests for L2 (SQLite) cache."""

    @pytest.fixture
    def cache(self, tmp_path: Path) -> L2Cache:
        """Create a L2 cache with temp database."""
        return L2Cache(db_path=tmp_path / "cache.db")

    def test_get_set(self, cache: L2Cache) -> None:
        """Test basic get/set operations."""
        entry = CacheEntry(key="test", value={"data": "value"}, tier=CacheTier.L2)
        cache.set("test", entry)
        result = cache.get("test")

        assert result is not None
        assert result.value == {"data": "value"}

    def test_json_serialization(self, cache: L2Cache) -> None:
        """Test JSON value serialization."""
        entry = CacheEntry(
            key="json_test",
            value={"nested": {"list": [1, 2, 3]}},
            tier=CacheTier.L2,
        )
        cache.set("json_test", entry)
        result = cache.get("json_test")

        assert result is not None
        assert result.value == {"nested": {"list": [1, 2, 3]}}

    def test_numpy_serialization(self, cache: L2Cache) -> None:
        """Test numpy array serialization."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        entry = CacheEntry(key="numpy_test", value=arr, tier=CacheTier.L2)
        cache.set("numpy_test", entry)
        result = cache.get("numpy_test")

        assert result is not None
        assert np.array_equal(result.value, arr)

    def test_expiration(self, cache: L2Cache) -> None:
        """Test entry expiration."""
        entry = CacheEntry(
            key="expire_test",
            value="value",
            tier=CacheTier.L2,
            ttl_seconds=0.01,
        )
        cache.set("expire_test", entry)

        time.sleep(0.02)
        assert cache.get("expire_test") is None

    def test_remove_expired(self, cache: L2Cache) -> None:
        """Test batch removal of expired entries."""
        # Add some entries with short TTL
        for i in range(5):
            entry = CacheEntry(
                key=f"expire{i}",
                value=i,
                tier=CacheTier.L2,
                ttl_seconds=0.01,
            )
            cache.set(f"expire{i}", entry)

        time.sleep(0.02)
        removed = cache.remove_expired()
        assert removed == 5

    def test_remove_by_tag(self, cache: L2Cache) -> None:
        """Test removal by tag."""
        for i in range(5):
            entry = CacheEntry(
                key=f"tagged{i}",
                value=i,
                tier=CacheTier.L2,
                tags=["test_tag"] if i % 2 == 0 else [],
            )
            cache.set(f"tagged{i}", entry)

        removed = cache.remove_by_tag("test_tag")
        assert removed == 3  # tagged0, tagged2, tagged4

    def test_stats(self, cache: L2Cache) -> None:
        """Test stats reporting."""
        for i in range(5):
            entry = CacheEntry(
                key=f"stats{i}",
                value=f"value{i}",
                tier=CacheTier.L2,
                size_bytes=100,
            )
            cache.set(f"stats{i}", entry)

        stats = cache.stats()
        assert stats["entries"] == 5


class TestL3Cache:
    """Tests for L3 (disk) cache."""

    @pytest.fixture
    def cache(self, tmp_path: Path) -> L3Cache:
        """Create a L3 cache with temp directory."""
        return L3Cache(cache_dir=tmp_path / "l3_cache", max_bytes=10 * 1024 * 1024)

    def test_get_set(self, cache: L3Cache) -> None:
        """Test basic get/set operations."""
        entry = CacheEntry(key="test", value="large value", tier=CacheTier.L3)
        cache.set("test", entry)
        result = cache.get("test")

        assert result is not None
        assert result.value == "large value"

    def test_numpy_serialization(self, cache: L3Cache) -> None:
        """Test numpy array serialization."""
        arr = np.random.RandomState(42).randn(100, 100).astype(np.float32)
        entry = CacheEntry(key="large_array", value=arr, tier=CacheTier.L3)
        cache.set("large_array", entry)
        result = cache.get("large_array")

        assert result is not None
        assert np.allclose(result.value, arr)

    def test_expiration(self, cache: L3Cache) -> None:
        """Test entry expiration."""
        entry = CacheEntry(
            key="expire_test",
            value="value",
            tier=CacheTier.L3,
            ttl_seconds=0.01,
        )
        cache.set("expire_test", entry)

        time.sleep(0.02)
        assert cache.get("expire_test") is None

    def test_eviction_on_size_limit(self, tmp_path: Path) -> None:
        """Test eviction when size limit reached."""
        cache = L3Cache(cache_dir=tmp_path / "small_cache", max_bytes=1000)

        # Add entries that exceed size limit
        for i in range(10):
            entry = CacheEntry(
                key=f"entry{i}",
                value="x" * 200,  # ~200 bytes each
                tier=CacheTier.L3,
                size_bytes=200,
            )
            cache.set(f"entry{i}", entry)

        # Should have evicted old entries
        stats = cache.stats()
        assert stats["total_bytes"] <= 1000


class TestMultiTierCache:
    """Tests for the multi-tier cache."""

    @pytest.fixture
    def cache(self, tmp_path: Path) -> MultiTierCache:
        """Create a multi-tier cache with temp storage."""
        return MultiTierCache(
            l1_maxsize=100,
            l1_max_bytes=10 * 1024 * 1024,
            l2_db_path=tmp_path / "l2.db",
            l3_cache_dir=tmp_path / "l3",
            auto_promote=True,
            promote_threshold=2,
        )

    def test_set_to_different_tiers(self, cache: MultiTierCache) -> None:
        """Test setting values to different tiers."""
        cache.set("l1_key", "l1_value", tier=CacheTier.L1)
        cache.set("l2_key", "l2_value", tier=CacheTier.L2)
        cache.set("l3_key", "l3_value", tier=CacheTier.L3)

        assert cache.get("l1_key") == "l1_value"
        assert cache.get("l2_key") == "l2_value"
        assert cache.get("l3_key") == "l3_value"

    def test_cascading_lookup(self, cache: MultiTierCache) -> None:
        """Test that get checks all tiers."""
        # Set only in L2
        cache.set("l2_only", "value", tier=CacheTier.L2)

        # Should find in L2
        result = cache.get("l2_only")
        assert result == "value"

        stats = cache.stats()
        assert stats["misses"]["l1"] == 1
        assert stats["hits"]["l2"] == 1

    def test_auto_promotion(self, cache: MultiTierCache) -> None:
        """Test automatic tier promotion on frequent access."""
        cache.set("promote_me", "value", tier=CacheTier.L2)

        # Access multiple times to trigger promotion
        for _ in range(3):
            cache.get("promote_me")

        stats = cache.stats()
        assert stats["promotions"] >= 1

    def test_remove(self, cache: MultiTierCache) -> None:
        """Test remove from all tiers."""
        cache.set("to_remove", "value", tier=CacheTier.L1)
        cache.set("to_remove", "value", tier=CacheTier.L2)

        removed = cache.remove("to_remove")
        assert removed

        assert cache.get("to_remove") is None

    def test_invalidate_by_tag(self, cache: MultiTierCache) -> None:
        """Test invalidation by tag."""
        for i in range(5):
            cache.set(
                f"tagged{i}",
                f"value{i}",
                tier=CacheTier.L1,
                tags=["common_tag"],
            )

        count = cache.invalidate_by_tag("common_tag")
        assert count == 5

    def test_invalidate_by_pattern(self, cache: MultiTierCache) -> None:
        """Test invalidation by key pattern."""
        for i in range(5):
            cache.set(f"prefix:key{i}", f"value{i}", tier=CacheTier.L1)
            cache.set(f"other:key{i}", f"value{i}", tier=CacheTier.L1)

        count = cache.invalidate_by_pattern("prefix:")
        assert count == 5

        # Other keys should remain
        assert cache.get("other:key0") == "value0"

    def test_cleanup_expired(self, cache: MultiTierCache) -> None:
        """Test cleanup of expired entries."""
        for i in range(5):
            cache.set(
                f"expire{i}",
                f"value{i}",
                tier=CacheTier.L1,
                ttl_seconds=0.01,
            )

        time.sleep(0.02)
        cleaned = cache.cleanup_expired()
        assert cleaned == 5

    def test_stats(self, cache: MultiTierCache) -> None:
        """Test comprehensive stats."""
        cache.set("test1", "value1", tier=CacheTier.L1)
        cache.set("test2", "value2", tier=CacheTier.L2)
        cache.get("test1")
        cache.get("test2")
        cache.get("nonexistent")

        stats = cache.stats()
        assert "l1" in stats
        assert "l2" in stats
        assert "l3" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rates" in stats


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_hit_rates(self) -> None:
        """Test hit rate calculations."""
        stats = CacheStats()
        stats.l1_hits = 80
        stats.l1_misses = 20

        assert stats.l1_hit_rate == 0.8

    def test_overall_hit_rate(self) -> None:
        """Test overall hit rate calculation."""
        stats = CacheStats()
        stats.l1_hits = 50
        stats.l2_hits = 30
        stats.l3_hits = 10
        stats.l3_misses = 10  # Total misses

        assert stats.overall_hit_rate == 0.9

    def test_zero_division(self) -> None:
        """Test hit rates with zero totals."""
        stats = CacheStats()
        assert stats.l1_hit_rate == 0.0
        assert stats.l2_hit_rate == 0.0
        assert stats.overall_hit_rate == 0.0
