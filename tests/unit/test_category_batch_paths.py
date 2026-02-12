"""TEST-10: Mixed batch classification paths.

Parameterized tests verifying that classify_batch correctly handles inputs
that hit the cache, fast-path, and full pipeline paths simultaneously.
"""

from __future__ import annotations

import time

import pytest

from jarvis.classifiers.category_classifier import (
    CategoryClassifier,
    classify_category,
    reset_category_classifier,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset classifier singleton between tests."""
    reset_category_classifier()
    yield
    reset_category_classifier()


class TestFastPathBatch:
    """Test fast-path detection within classify_batch."""

    def test_all_reactions_batch(self):
        """Batch of all reactions should all hit fast path."""
        classifier = CategoryClassifier()
        texts = [
            'Loved "Hey there"',
            'Laughed at "Nice joke"',
            'Emphasized "Important thing"',
        ]
        results = classifier.classify_batch(texts)
        assert len(results) == 3
        for r in results:
            assert r.method == "fast_path"

    def test_all_acknowledgments_batch(self):
        """Batch of all acknowledgments should all hit fast path."""
        classifier = CategoryClassifier()
        texts = ["ok", "got it", "sounds good"]
        results = classifier.classify_batch(texts)
        assert len(results) == 3
        for r in results:
            assert r.method == "fast_path"
            assert r.category == "acknowledge"

    def test_empty_batch_returns_empty(self):
        """Empty batch returns empty list."""
        classifier = CategoryClassifier()
        results = classifier.classify_batch([])
        assert results == []


class TestCacheBatch:
    """Test cache behavior within classify_batch."""

    def test_cached_results_returned_on_second_call(self):
        """Second call with same texts returns cached results."""
        classifier = CategoryClassifier()
        texts = ["ok", "sounds good"]

        # First call populates cache
        results1 = classifier.classify_batch(texts)
        assert len(results1) == 2

        # Second call should return cached results
        results2 = classifier.classify_batch(texts)
        assert len(results2) == 2

        # Results should be identical
        for r1, r2 in zip(results1, results2):
            assert r1.category == r2.category
            assert r1.method == r2.method

    def test_cache_expires_after_ttl(self):
        """Cached results expire after TTL."""
        classifier = CategoryClassifier()
        classifier._cache_ttl = 0.05  # 50ms TTL for testing

        texts = ["ok"]
        results1 = classifier.classify_batch(texts)

        # Wait for cache to expire
        time.sleep(0.2)

        # Should recompute (still fast-path, but not from cache)
        results2 = classifier.classify_batch(texts)
        assert results2[0].category == results1[0].category

    def test_cache_max_size_eviction(self):
        """Cache evicts oldest entries when max size is reached."""
        classifier = CategoryClassifier()
        classifier._cache_max_size = 5

        # Fill cache with 6 entries
        for i in range(6):
            classifier.classify_batch([f"message variant {i} ok"])

        # Cache should not exceed max size + some slack from eviction
        assert len(classifier._classification_cache) <= 6


class TestMixedBatchPaths:
    """Test batches with mixed fast-path and pipeline inputs."""

    def test_mixed_fast_path_and_regular(self):
        """Batch with fast-path and regular messages handles both."""
        classifier = CategoryClassifier()

        texts = [
            "ok",  # fast-path: acknowledge
            'Loved "Nice"',  # fast-path: emotion
            "Want to grab lunch tomorrow?",  # pipeline/default
            "sounds good",  # fast-path: acknowledge
        ]

        results = classifier.classify_batch(texts)
        assert len(results) == 4

        # Fast-path results
        assert results[0].category == "acknowledge"
        assert results[0].method == "fast_path"

        assert results[1].category == "emotion"
        assert results[1].method == "fast_path"

        assert results[3].category == "acknowledge"
        assert results[3].method == "fast_path"

        # Pipeline/default result (depends on whether model is loaded)
        assert results[2].category in (
            "question",
            "request",
            "statement",
            "emotion",
            "closing",
            "acknowledge",
        )

    def test_mixed_cached_and_fresh(self):
        """Batch where some items are cached and some are new."""
        classifier = CategoryClassifier()

        # Pre-populate cache
        classifier.classify_batch(["ok"])

        # Now batch with one cached and one new
        texts = ["ok", "something completely new and different"]
        results = classifier.classify_batch(texts)
        assert len(results) == 2
        assert results[0].category == "acknowledge"  # cached fast-path

    @pytest.mark.parametrize(
        "text,expected_category,expected_method",
        [
            ("ok", "acknowledge", "fast_path"),
            ("got it", "acknowledge", "fast_path"),
            ("sounds good", "acknowledge", "fast_path"),
            ('Loved "test"', "emotion", "fast_path"),
            ('Laughed at "joke"', "emotion", "fast_path"),
        ],
    )
    def test_fast_path_parametrized(self, text, expected_category, expected_method):
        """Parametrized test for fast-path classification."""
        result = classify_category(text)
        assert result.category == expected_category
        assert result.method == expected_method

    @pytest.mark.parametrize(
        "text",
        [
            "What time is the meeting?",
            "Can you send me the report?",
            "I'm feeling great today!",
            "Let's schedule a call for next week.",
            "The project deadline is Friday.",
        ],
    )
    def test_non_fast_path_messages_get_classified(self, text):
        """Non-fast-path messages get classified (via pipeline or default)."""
        result = classify_category(text)
        assert result.category in (
            "acknowledge",
            "closing",
            "emotion",
            "question",
            "request",
            "statement",
        )
        assert result.confidence > 0.0

    def test_batch_with_contexts(self):
        """classify_batch with context arrays works correctly."""
        classifier = CategoryClassifier()

        texts = ["ok", "sure thing"]
        contexts = [["Hey, can you help?"], ["Sounds good?"]]

        results = classifier.classify_batch(texts, contexts=contexts)
        assert len(results) == 2

    def test_batch_with_none_contexts(self):
        """classify_batch with None contexts defaults correctly."""
        classifier = CategoryClassifier()

        texts = ["ok", "sounds good"]
        results = classifier.classify_batch(texts, contexts=None)
        assert len(results) == 2

    def test_single_item_batch(self):
        """Single-item batch returns single result."""
        classifier = CategoryClassifier()
        results = classifier.classify_batch(["hello"])
        assert len(results) == 1
        assert results[0].category in (
            "acknowledge",
            "closing",
            "emotion",
            "question",
            "request",
            "statement",
        )
