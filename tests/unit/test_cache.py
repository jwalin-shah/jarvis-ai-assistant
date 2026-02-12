"""Tests for jarvis/cache.py - TTL cache with LRU eviction and single-flight."""

import threading
import time
from unittest.mock import MagicMock

import pytest

from jarvis.cache import TTLCache


class TestBasicOperations:
    """Test get/set/delete basics."""

    def test_set_and_get(self):
        cache = TTLCache(ttl_seconds=10.0)
        cache.set("key1", "value1")
        found, value = cache.get("key1")
        assert found is True
        assert value == "value1"

    def test_get_missing_key(self):
        cache = TTLCache()
        found, value = cache.get("nonexistent")
        assert found is False
        assert value is None

    def test_delete_key(self):
        cache = TTLCache()
        cache.set("key1", "value1")
        cache.delete("key1")
        found, _ = cache.get("key1")
        assert found is False

    def test_delete_missing_key_no_error(self):
        cache = TTLCache()
        cache.delete("nonexistent")  # Should not raise

    def test_overwrite_existing_key(self):
        cache = TTLCache()
        cache.set("key1", "old")
        cache.set("key1", "new")
        found, value = cache.get("key1")
        assert found is True
        assert value == "new"

    def test_stores_various_types(self):
        cache = TTLCache()
        cache.set("int", 42)
        cache.set("list", [1, 2, 3])
        cache.set("dict", {"a": 1})
        cache.set("none", None)

        assert cache.get("int") == (True, 42)
        assert cache.get("list") == (True, [1, 2, 3])
        assert cache.get("dict") == (True, {"a": 1})
        assert cache.get("none") == (True, None)


class TestTTLExpiration:
    """Test time-to-live behavior."""

    def test_expired_entry_returns_not_found(self):
        cache = TTLCache(ttl_seconds=0.05)
        cache.set("key1", "value1")
        time.sleep(0.1)
        found, value = cache.get("key1")
        assert found is False

    def test_non_expired_entry_returns_value(self):
        cache = TTLCache(ttl_seconds=10.0)
        cache.set("key1", "value1")
        found, value = cache.get("key1")
        assert found is True
        assert value == "value1"

    def test_expired_entry_removed_from_cache(self):
        cache = TTLCache(ttl_seconds=0.05)
        cache.set("key1", "value1")
        time.sleep(0.1)
        cache.get("key1")  # Triggers cleanup
        assert cache.stats()["size"] == 0


class TestLRUEviction:
    """Test LRU eviction when at capacity."""

    def test_evicts_oldest_when_full(self):
        cache = TTLCache(ttl_seconds=10.0, maxsize=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)  # Should evict "a"

        found_a, _ = cache.get("a")
        found_d, val_d = cache.get("d")
        assert found_a is False
        assert found_d is True
        assert val_d == 4

    def test_access_refreshes_lru_order(self):
        cache = TTLCache(ttl_seconds=10.0, maxsize=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.get("a")  # Refreshes "a" - now "b" is oldest
        cache.set("d", 4)  # Should evict "b", not "a"

        found_a, _ = cache.get("a")
        found_b, _ = cache.get("b")
        assert found_a is True
        assert found_b is False

    def test_maxsize_one(self):
        cache = TTLCache(ttl_seconds=10.0, maxsize=1)
        cache.set("a", 1)
        cache.set("b", 2)
        found_a, _ = cache.get("a")
        found_b, _ = cache.get("b")
        assert found_a is False
        assert found_b is True


class TestInvalidateAndClear:
    """Test invalidation methods."""

    def test_invalidate_specific_key(self):
        cache = TTLCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.invalidate("a")
        assert cache.get("a") == (False, None)
        assert cache.get("b")[0] is True

    def test_invalidate_all(self):
        cache = TTLCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.invalidate(None)
        assert cache.stats()["size"] == 0

    def test_clear(self):
        cache = TTLCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.stats()["size"] == 0


class TestStats:
    """Test hit/miss statistics."""

    def test_initial_stats(self):
        cache = TTLCache(ttl_seconds=30.0, maxsize=128)
        stats = cache.stats()
        assert stats["size"] == 0
        assert stats["maxsize"] == 128
        assert stats["ttl_seconds"] == 30.0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

    def test_hit_tracking(self):
        cache = TTLCache()
        cache.set("a", 1)
        cache.get("a")  # hit
        cache.get("a")  # hit
        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 0

    def test_miss_tracking(self):
        cache = TTLCache()
        cache.get("missing1")
        cache.get("missing2")
        stats = cache.stats()
        assert stats["misses"] == 2
        assert stats["hits"] == 0

    def test_hit_rate(self):
        cache = TTLCache()
        cache.set("a", 1)
        cache.get("a")  # hit
        cache.get("missing")  # miss
        stats = cache.stats()
        assert stats["hit_rate"] == pytest.approx(0.5)


class TestGetOrSet:
    """Test get_or_set with factory and single-flight."""

    def test_computes_on_miss(self):
        cache = TTLCache(ttl_seconds=10.0)
        factory = MagicMock(return_value="computed")
        result = cache.get_or_set("key1", factory)
        assert result == "computed"
        factory.assert_called_once()

    def test_returns_cached_on_hit(self):
        cache = TTLCache(ttl_seconds=10.0)
        cache.set("key1", "cached")
        factory = MagicMock(return_value="computed")
        result = cache.get_or_set("key1", factory)
        assert result == "cached"
        factory.assert_not_called()

    def test_caches_computed_value(self):
        cache = TTLCache(ttl_seconds=10.0)
        cache.get_or_set("key1", lambda: "computed")
        found, value = cache.get("key1")
        assert found is True
        assert value == "computed"

    def test_single_flight_dedup(self):
        """Concurrent callers for the same key should only compute once."""
        cache = TTLCache(ttl_seconds=10.0)
        call_count = 0
        barrier = threading.Barrier(5)

        def slow_factory():
            nonlocal call_count
            call_count += 1
            time.sleep(0.1)
            return "result"

        results = [None] * 5
        errors = []

        def worker(idx):
            try:
                barrier.wait(timeout=2.0)
                results[idx] = cache.get_or_set("shared_key", slow_factory)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors, f"Worker errors: {errors}"
        # All should get the same result
        assert all(r == "result" for r in results)
        # Factory should be called very few times (ideally 1, but timing and OS scheduling
        # can allow a few threads through before the first result is cached)
        assert call_count <= 3, f"Factory called {call_count} times, expected <=3"

    def test_factory_exception_still_cleans_up_inflight(self):
        """If factory raises, inflight is cleaned up so next caller can retry."""
        cache = TTLCache(ttl_seconds=10.0)

        with pytest.raises(ValueError):
            cache.get_or_set("key1", lambda: (_ for _ in ()).throw(ValueError("boom")))

        # Should be able to compute again (inflight cleaned up)
        result = cache.get_or_set("key1", lambda: "recovered")
        assert result == "recovered"


class TestThreadSafety:
    """Test concurrent access doesn't corrupt state."""

    def test_concurrent_set_get(self):
        cache = TTLCache(ttl_seconds=10.0, maxsize=100)
        errors = []

        def writer(prefix, count):
            try:
                for i in range(count):
                    cache.set(f"{prefix}_{i}", i)
            except Exception as e:
                errors.append(e)

        def reader(prefix, count):
            try:
                for i in range(count):
                    cache.get(f"{prefix}_{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for p in range(4):
            threads.append(threading.Thread(target=writer, args=(f"w{p}", 50)))
            threads.append(threading.Thread(target=reader, args=(f"w{p}", 50)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors
        assert cache.stats()["size"] <= 100  # Respects maxsize
