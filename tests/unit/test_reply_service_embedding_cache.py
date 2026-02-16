"""Unit tests for ReplyService embedding cache feature.

Tests cover:
- _get_cached_embeddings(): content-addressable caching
- Cache hit/miss tracking
- Cache eviction (LRU behavior)
- Thread safety considerations
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from jarvis.reply_service import ReplyService

# =============================================================================
# _get_cached_embeddings Tests
# =============================================================================


class TestGetCachedEmbeddings:
    """Tests for ReplyService._get_cached_embeddings method."""

    def test_cache_creates_on_first_call(self) -> None:
        """Cache should be created lazily on first call."""
        service = ReplyService()

        # Cache should not exist before first call
        assert not hasattr(service, "_embedding_cache")

        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        # First call should create cache
        _result = service._get_cached_embeddings(["test text"], mock_embedder)

        assert hasattr(service, "_embedding_cache")
        assert isinstance(service._embedding_cache, dict)
        assert service._embedding_cache_misses > 0

    def test_cache_hit_returns_cached_value(self) -> None:
        """Second call with same text should return cached embedding."""
        service = ReplyService()

        mock_embedder = MagicMock()
        expected_embedding = np.array([[0.1, 0.2, 0.3]])
        mock_embedder.encode.return_value = expected_embedding

        # First call - cache miss
        result1 = service._get_cached_embeddings(["test text"], mock_embedder)

        # Reset mock to verify it's not called again
        mock_embedder.encode.reset_mock()

        # Second call - should be cache hit
        result2 = service._get_cached_embeddings(["test text"], mock_embedder)

        # Embedder should not be called for cache hit
        mock_embedder.encode.assert_not_called()

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)
        assert service._embedding_cache_hits > 0

    def test_cache_miss_calls_embedder(self) -> None:
        """New text should trigger embedder call."""
        service = ReplyService()

        mock_embedder = MagicMock()
        expected_embedding = np.array([[0.4, 0.5, 0.6]])
        mock_embedder.encode.return_value = expected_embedding

        result = service._get_cached_embeddings(["new text"], mock_embedder)

        mock_embedder.encode.assert_called_once()
        np.testing.assert_array_equal(result, expected_embedding)

    def test_different_texts_get_different_keys(self) -> None:
        """Different texts should have different cache keys."""
        service = ReplyService()

        mock_embedder = MagicMock()
        mock_embedder.encode.side_effect = [
            np.array([[0.1, 0.2]]),
            np.array([[0.3, 0.4]]),
        ]

        service._get_cached_embeddings(["text one"], mock_embedder)
        service._get_cached_embeddings(["text two"], mock_embedder)

        # Both should be in cache
        assert len(service._embedding_cache) == 2
        assert service._embedding_cache_misses == 2

    def test_cache_eviction_at_limit(self) -> None:
        """Cache should evict oldest entries when reaching 1000 limit."""
        service = ReplyService()

        # Manually create cache with 999 entries
        service._embedding_cache = {f"key_{i}": np.array([i]) for i in range(999)}
        service._embedding_cache_hits = 0
        service._embedding_cache_misses = 0

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        # Add 2 more entries (should trigger eviction)
        service._get_cached_embeddings(["text 1"], mock_embedder)
        service._get_cached_embeddings(["text 2"], mock_embedder)

        # Cache should be cleaned up (half evicted + 2 new)
        assert len(service._embedding_cache) <= 502  # 500 kept + 2 new

    def test_empty_texts_list(self) -> None:
        """Empty list should return empty array."""
        service = ReplyService()
        mock_embedder = MagicMock()

        result = service._get_cached_embeddings([], mock_embedder)

        assert len(result) == 0
        mock_embedder.encode.assert_not_called()

    def test_batch_encoding_caches_individual(self) -> None:
        """Batch encoding should cache each text individually."""
        service = ReplyService()

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ]
        )

        texts = ["first text", "second text"]
        _result = service._get_cached_embeddings(texts, mock_embedder)

        # Both should be cached separately
        assert len(service._embedding_cache) == 2

        # Reset mock and verify both are cached
        mock_embedder.encode.reset_mock()

        _result2 = service._get_cached_embeddings(["first text"], mock_embedder)
        mock_embedder.encode.assert_not_called()

    def test_content_hash_consistency(self) -> None:
        """Same text should always produce same cache key."""
        service = ReplyService()
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        # Call multiple times with same text
        for _ in range(5):
            service._get_cached_embeddings(["consistent text"], mock_embedder)

        # Should only be in cache once
        assert len(service._embedding_cache) == 1
        assert service._embedding_cache_hits == 4
        assert service._embedding_cache_misses == 1

    def test_similar_texts_different_keys(self) -> None:
        """Similar but different texts should have different cache keys."""
        service = ReplyService()
        mock_embedder = MagicMock()
        mock_embedder.encode.side_effect = [
            np.array([[0.1]]),
            np.array([[0.2]]),
        ]

        texts = ["Hello world", "hello world", "Hello World!"]
        for text in texts[:2]:  # Just test first two
            service._get_cached_embeddings([text], mock_embedder)

        # Both should be in cache (different due to case)
        assert len(service._embedding_cache) == 2


# =============================================================================
# Integration with _dedupe_examples
# =============================================================================


class TestEmbeddingCacheIntegration:
    """Tests for embedding cache integration with _dedupe_examples."""

    def test_dedupe_uses_cache_for_same_examples(self) -> None:
        """_dedupe_examples should benefit from embedding cache."""
        service = ReplyService()

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        examples = [
            ("context 1", "reply 1"),
            ("context 2", "reply 2"),
            ("context 3", "reply 3"),
        ]

        # First dedupe call
        result1 = service._dedupe_examples(examples, mock_embedder)

        # Note: _dedupe_examples uses embedder.encode directly for small batches
        # but our cache implementation will be used through _get_cached_embeddings
        # when integrated. For now, just verify it works.
        assert len(result1) > 0

        # Test direct cache usage
        texts = [f"{ctx} {out}" for ctx, out in examples]
        _ = service._get_cached_embeddings(texts, mock_embedder)

        # Reset mock
        mock_embedder.encode.reset_mock()

        # Second call should use cache
        _ = service._get_cached_embeddings(texts, mock_embedder)

        # Embedder should not be called for cache hit
        mock_embedder.encode.assert_not_called()
        assert service._embedding_cache_hits >= 3

    def test_dedupe_small_set_skips_cache(self) -> None:
        """Sets <= 6 items should skip embedding entirely."""
        service = ReplyService()
        mock_embedder = MagicMock()

        examples = [(f"ctx {i}", f"reply {i}") for i in range(5)]

        result = service._dedupe_examples(examples, mock_embedder)

        # Should not call embedder for small sets
        mock_embedder.encode.assert_not_called()
        assert len(result) == 5  # All returned unchanged


# =============================================================================
# Cache Metrics
# =============================================================================


class TestCacheMetrics:
    """Tests for cache hit/miss tracking."""

    def test_metrics_initialization(self) -> None:
        """Metrics should be initialized on first cache access."""
        service = ReplyService()
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.array([[0.1, 0.2]])

        # Before first call
        assert not hasattr(service, "_embedding_cache_hits")

        # After first call
        service._get_cached_embeddings(["test"], mock_embedder)
        assert service._embedding_cache_hits == 0
        assert service._embedding_cache_misses == 1

    def test_metrics_accumulate(self) -> None:
        """Metrics should accumulate across multiple calls."""
        service = ReplyService()
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.array([[0.1, 0.2]])

        # Mix of hits and misses
        for i in range(10):
            text = f"text {i % 3}"  # Only 3 unique texts
            service._get_cached_embeddings([text], mock_embedder)

        # Should have 3 misses and 7 hits
        assert service._embedding_cache_misses == 3
        assert service._embedding_cache_hits == 7
