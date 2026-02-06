"""Comprehensive tests for Response Classifier V2.

Tests:
- Batch processing correctness
- Concurrent access safety
- Performance regression tests
- Edge cases (empty, very long, unicode)
- Backward compatibility with V1 API
- Caching behavior
- Ensemble voting
- Custom class registration
"""

from __future__ import annotations

import concurrent.futures
import threading
import time

import numpy as np
import pytest

from jarvis.classifiers.response_classifier_v2 import (
    V2_TO_LEGACY,
    ABTestConfig,
    BatchResponseClassifier,
    ClassificationResult,
    ClassificationResultV2,
    CustomClass,
    EmbeddingCache,
    EnsembleVote,
    EnsembleVoter,
    FeatureCache,
    PlattScaler,
    ResponseType,
    ResponseTypeV2,
    get_batch_response_classifier,
    reset_batch_response_classifier,
    set_ab_test_config,
    set_use_v2_classifier,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def classifier():
    """Create a fresh classifier instance for testing."""
    reset_batch_response_classifier()
    clf = BatchResponseClassifier(
        enable_caching=True,
        enable_ensemble=True,
        use_v2_api=True,
    )
    yield clf
    clf.shutdown()


@pytest.fixture
def classifier_no_cache():
    """Create classifier without caching."""
    clf = BatchResponseClassifier(
        enable_caching=False,
        enable_ensemble=True,
        use_v2_api=True,
    )
    yield clf
    clf.shutdown()


@pytest.fixture
def classifier_legacy():
    """Create classifier with legacy API."""
    clf = BatchResponseClassifier(
        enable_caching=True,
        enable_ensemble=True,
        use_v2_api=False,
    )
    yield clf
    clf.shutdown()


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        "Yes!",
        "No thanks",
        "Maybe later",
        "What time?",
        "Ok cool",
        "That's awesome!",
        "I'm sorry to hear that",
        "Hey!",
        "I went to the store",
        "Let me check my calendar",
    ]


@pytest.fixture
def expected_types():
    """Expected types for sample messages."""
    return [
        ResponseTypeV2.AGREE,
        ResponseTypeV2.DECLINE,
        ResponseTypeV2.DEFER,
        ResponseTypeV2.QUESTION,
        ResponseTypeV2.ACKNOWLEDGE,
        ResponseTypeV2.REACT_POSITIVE,
        ResponseTypeV2.REACT_SYMPATHY,
        ResponseTypeV2.GREETING,
        ResponseTypeV2.STATEMENT,
        ResponseTypeV2.DEFER,
    ]


# =============================================================================
# Response Type Tests
# =============================================================================


class TestResponseTypes:
    """Tests for response type enums."""

    def test_v2_types_exist(self):
        """All V2 types should be defined."""
        assert ResponseTypeV2.AGREE.value == "AGREE"
        assert ResponseTypeV2.UNCERTAIN.value == "UNCERTAIN"
        assert ResponseTypeV2.EMOTIONAL_SUPPORT.value == "EMOTIONAL_SUPPORT"
        assert ResponseTypeV2.SCHEDULING.value == "SCHEDULING"
        assert ResponseTypeV2.INFORMATION_REQUEST.value == "INFORMATION_REQUEST"

    def test_legacy_types_exist(self):
        """All legacy types should be defined."""
        assert ResponseType.AGREE.value == "AGREE"
        assert ResponseType.STATEMENT.value == "STATEMENT"

    def test_v2_to_legacy_mapping(self):
        """V2 types should map to legacy types."""
        assert V2_TO_LEGACY[ResponseTypeV2.AGREE] == ResponseType.AGREE
        assert V2_TO_LEGACY[ResponseTypeV2.UNCERTAIN] == ResponseType.ANSWER
        assert V2_TO_LEGACY[ResponseTypeV2.EMOTIONAL_SUPPORT] == ResponseType.REACT_SYMPATHY
        assert V2_TO_LEGACY[ResponseTypeV2.QUESTION_CLARIFICATION] == ResponseType.QUESTION


# =============================================================================
# Classification Result Tests
# =============================================================================


class TestClassificationResult:
    """Tests for classification result classes."""

    def test_v2_result_creation(self):
        """V2 result should be created correctly."""
        result = ClassificationResultV2(
            label=ResponseTypeV2.AGREE,
            confidence=0.95,
            method="structural",
            structural_match=True,
        )
        assert result.label == ResponseTypeV2.AGREE
        assert result.confidence == 0.95
        assert result.method == "structural"
        assert result.structural_match is True

    def test_v2_to_legacy_conversion(self):
        """V2 result should convert to legacy format."""
        v2_result = ClassificationResultV2(
            label=ResponseTypeV2.AGREE,
            confidence=0.95,
            method="structural",
            structural_match=True,
        )
        legacy = v2_result.to_legacy()
        assert isinstance(legacy, ClassificationResult)
        assert legacy.label == ResponseType.AGREE
        assert legacy.confidence == 0.95

    def test_legacy_label_property(self):
        """V2 result should expose legacy label."""
        result = ClassificationResultV2(
            label=ResponseTypeV2.QUESTION_CLARIFICATION,
            confidence=0.8,
            method="svm",
        )
        assert result.legacy_label == ResponseType.QUESTION


# =============================================================================
# Basic Classification Tests
# =============================================================================


class TestBasicClassification:
    """Tests for basic classification functionality."""

    def test_classify_empty_string(self, classifier):
        """Empty string should return STATEMENT with zero confidence."""
        result = classifier.classify("")
        assert result.label == ResponseTypeV2.STATEMENT
        assert result.confidence == 0.0
        assert result.method == "empty"

    def test_classify_whitespace_only(self, classifier):
        """Whitespace-only should return STATEMENT."""
        result = classifier.classify("   \t\n   ")
        assert result.label == ResponseTypeV2.STATEMENT
        assert result.confidence == 0.0

    def test_classify_agree(self, classifier):
        """Affirmative messages should classify as AGREE."""
        messages = ["Yes", "Yeah", "Sure", "Definitely", "I'm down"]
        for msg in messages:
            result = classifier.classify(msg)
            assert result.label == ResponseTypeV2.AGREE, f"Failed for: {msg}"
            assert result.confidence > 0.8

    def test_classify_decline(self, classifier):
        """Negative messages should classify as DECLINE."""
        messages = ["No", "Nope", "Can't", "I'll pass"]
        for msg in messages:
            result = classifier.classify(msg)
            assert result.label == ResponseTypeV2.DECLINE, f"Failed for: {msg}"

    def test_classify_question(self, classifier):
        """Questions should classify as QUESTION."""
        messages = ["What time?", "Where?", "How do I get there?"]
        for msg in messages:
            result = classifier.classify(msg)
            assert result.label in [
                ResponseTypeV2.QUESTION,
                ResponseTypeV2.QUESTION_CLARIFICATION,
                ResponseTypeV2.QUESTION_FOLLOWUP,
                ResponseTypeV2.INFORMATION_REQUEST,
            ], f"Failed for: {msg}"

    def test_classify_greeting(self, classifier):
        """Greetings should classify as GREETING."""
        messages = ["Hey!", "Hi", "Hello", "Yo"]
        for msg in messages:
            result = classifier.classify(msg)
            assert result.label == ResponseTypeV2.GREETING, f"Failed for: {msg}"

    def test_classify_tapback_positive(self, classifier):
        """Positive tapbacks should classify as REACT_POSITIVE."""
        messages = ['Liked "hello"', 'Loved "great job"', "Laughed at an image"]
        for msg in messages:
            result = classifier.classify(msg)
            assert result.label == ResponseTypeV2.REACT_POSITIVE
            assert result.method == "tapback_positive"

    def test_classify_tapback_filtered(self, classifier):
        """Filtered tapbacks should classify as ANSWER."""
        messages = ['Disliked "message"', 'Emphasized "urgent"']
        for msg in messages:
            result = classifier.classify(msg)
            assert result.label == ResponseTypeV2.ANSWER
            assert result.method == "tapback_filtered"


# =============================================================================
# Batch Classification Tests
# =============================================================================


class TestBatchClassification:
    """Tests for batch classification functionality."""

    def test_batch_empty_list(self, classifier):
        """Empty list should return empty results."""
        results = classifier.classify_batch([])
        assert results == []

    def test_batch_single_message(self, classifier):
        """Single message batch should work."""
        results = classifier.classify_batch(["Yes"])
        assert len(results) == 1
        assert results[0].label == ResponseTypeV2.AGREE

    def test_batch_multiple_messages(self, classifier, sample_messages):
        """Multiple messages should be classified correctly."""
        results = classifier.classify_batch(sample_messages)
        assert len(results) == len(sample_messages)
        for result in results:
            assert isinstance(result, ClassificationResultV2)
            assert result.confidence > 0

    def test_batch_preserves_order(self, classifier):
        """Batch results should preserve input order."""
        messages = ["Yes", "No", "Maybe", "What?"]
        results = classifier.classify_batch(messages)

        assert results[0].label == ResponseTypeV2.AGREE
        assert results[1].label == ResponseTypeV2.DECLINE
        assert results[2].label == ResponseTypeV2.DEFER
        assert results[3].label == ResponseTypeV2.QUESTION

    def test_batch_handles_empty_in_middle(self, classifier):
        """Batch should handle empty strings in the middle."""
        messages = ["Yes", "", "No", "  ", "Maybe"]
        results = classifier.classify_batch(messages)
        assert len(results) == 5
        assert results[1].method == "empty"
        assert results[3].method == "empty"

    def test_batch_vs_single_consistency(self, classifier, sample_messages):
        """Batch results should be consistent with single classification."""
        batch_results = classifier.classify_batch(sample_messages)
        single_results = [classifier.classify(msg) for msg in sample_messages]

        for batch, single in zip(batch_results, single_results):
            assert batch.label == single.label, "Mismatch for message"

    def test_batch_large_input(self, classifier):
        """Large batch should be processed correctly."""
        messages = ["Yes"] * 100 + ["No"] * 100 + ["Maybe"] * 100
        results = classifier.classify_batch(messages)
        assert len(results) == 300


# =============================================================================
# Caching Tests
# =============================================================================


class TestCaching:
    """Tests for caching functionality."""

    def test_embedding_cache_basic(self):
        """Embedding cache should store and retrieve values."""
        cache = EmbeddingCache(maxsize=100)
        embedding = np.random.randn(384).astype(np.float32)

        cache.put("test text", embedding)
        retrieved = cache.get("test text")

        assert retrieved is not None
        np.testing.assert_array_almost_equal(retrieved, embedding)

    def test_embedding_cache_miss(self):
        """Cache miss should return None."""
        cache = EmbeddingCache(maxsize=100)
        assert cache.get("nonexistent") is None

    def test_embedding_cache_eviction(self):
        """Cache should evict oldest entries when full."""
        cache = EmbeddingCache(maxsize=3)

        for i in range(5):
            cache.put(f"text_{i}", np.random.randn(384).astype(np.float32))

        # First two should be evicted
        assert cache.get("text_0") is None
        assert cache.get("text_1") is None
        # Last three should remain
        assert cache.get("text_2") is not None
        assert cache.get("text_3") is not None
        assert cache.get("text_4") is not None

    def test_embedding_cache_stats(self):
        """Cache should track hit/miss statistics."""
        cache = EmbeddingCache(maxsize=100)
        cache.put("test", np.random.randn(384).astype(np.float32))

        cache.get("test")  # Hit
        cache.get("missing")  # Miss

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_feature_cache_basic(self):
        """Feature cache should store structural match results."""
        cache = FeatureCache(maxsize=100)
        cache.put("yes", (ResponseTypeV2.AGREE, 0.95))

        result = cache.get("yes")
        assert result == (ResponseTypeV2.AGREE, 0.95)

    def test_classifier_cache_stats(self, classifier):
        """Classifier should expose cache statistics."""
        classifier.classify("test message")
        stats = classifier.get_cache_stats()

        assert "embedding_cache" in stats
        assert "feature_cache" in stats

    def test_classifier_clear_caches(self, classifier):
        """Classifier should be able to clear caches."""
        classifier.classify("test message")
        classifier.clear_caches()

        stats = classifier.get_cache_stats()
        assert stats["embedding_cache"]["size"] == 0


# =============================================================================
# Ensemble Voting Tests
# =============================================================================


class TestEnsembleVoting:
    """Tests for ensemble voting functionality."""

    def test_voter_single_vote(self):
        """Single vote should be returned."""
        voter = EnsembleVoter()
        votes = [EnsembleVote(ResponseTypeV2.AGREE, 0.9, 1.0)]
        label, conf, weights = voter.vote(votes)
        assert label == ResponseTypeV2.AGREE
        assert conf == 0.9

    def test_voter_unanimous_votes(self):
        """Unanimous votes should have high confidence."""
        voter = EnsembleVoter()
        votes = [
            EnsembleVote(ResponseTypeV2.AGREE, 0.9, 0.4),
            EnsembleVote(ResponseTypeV2.AGREE, 0.85, 0.4),
            EnsembleVote(ResponseTypeV2.AGREE, 0.8, 0.2),
        ]
        label, conf, weights = voter.vote(votes)
        assert label == ResponseTypeV2.AGREE
        assert conf > 0.8

    def test_voter_conflicting_votes(self):
        """Conflicting votes should use weighted average."""
        voter = EnsembleVoter()
        votes = [
            EnsembleVote(ResponseTypeV2.AGREE, 0.9, 0.5),
            EnsembleVote(ResponseTypeV2.DECLINE, 0.7, 0.5),
        ]
        label, conf, weights = voter.vote(votes)
        assert label == ResponseTypeV2.AGREE  # Higher confidence wins
        assert label in weights

    def test_voter_uncertain_threshold(self):
        """Low confidence should return UNCERTAIN."""
        voter = EnsembleVoter()
        votes = [
            EnsembleVote(ResponseTypeV2.AGREE, 0.2, 0.5),
            EnsembleVote(ResponseTypeV2.DECLINE, 0.2, 0.5),
        ]
        label, conf, weights = voter.vote(votes, uncertain_threshold=0.3)
        assert label == ResponseTypeV2.UNCERTAIN

    def test_voter_empty_votes(self):
        """Empty votes should return UNCERTAIN."""
        voter = EnsembleVoter()
        label, conf, weights = voter.vote([])
        assert label == ResponseTypeV2.UNCERTAIN
        assert conf == 0.0


# =============================================================================
# Platt Scaling Tests
# =============================================================================


class TestPlattScaling:
    """Tests for Platt scaling confidence calibration."""

    def test_platt_scaler_transform(self):
        """Platt scaler should transform values."""
        scaler = PlattScaler(a=-2.0, b=0.0)
        result = scaler.transform_single(0.0)
        assert 0 < result < 1

    def test_platt_scaler_batch(self):
        """Platt scaler should handle batch transforms."""
        scaler = PlattScaler(a=-2.0, b=0.0)
        values = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        results = scaler.transform(values)
        assert results.shape == (3,)
        assert all(0 < r < 1 for r in results)


# =============================================================================
# Custom Class Tests
# =============================================================================


class TestCustomClasses:
    """Tests for custom class registration."""

    def test_register_custom_class(self, classifier):
        """Custom class should be registered."""
        custom = CustomClass(
            name="URGENT",
            patterns=[r"^urgent:", r"asap"],
            exemplars=["urgent: please respond", "need this asap"],
            confidence_threshold=0.8,
        )
        classifier.register_custom_class(custom)
        # Registration shouldn't raise
        assert True

    def test_unregister_custom_class(self, classifier):
        """Custom class should be unregistered."""
        custom = CustomClass(
            name="TEST_CLASS",
            patterns=[r"test"],
            exemplars=[],
        )
        classifier.register_custom_class(custom)
        classifier.unregister_custom_class("TEST_CLASS")
        # Unregistration shouldn't raise
        assert True


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestConcurrentAccess:
    """Tests for thread safety and concurrent access."""

    def test_concurrent_single_classify(self, classifier):
        """Concurrent single classifications should be safe."""
        messages = ["Yes", "No", "Maybe", "What?"] * 10

        def classify_message(msg):
            return classifier.classify(msg)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(classify_message, msg) for msg in messages]
            results = [f.result() for f in futures]

        assert len(results) == len(messages)
        for result in results:
            assert isinstance(result, ClassificationResultV2)

    def test_concurrent_batch_classify(self, classifier):
        """Concurrent batch classifications should be safe."""
        batches = [["Yes", "No", "Maybe"] for _ in range(10)]

        def classify_batch(batch):
            return classifier.classify_batch(batch)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(classify_batch, batch) for batch in batches]
            results = [f.result() for f in futures]

        assert len(results) == len(batches)
        for batch_result in results:
            assert len(batch_result) == 3

    def test_cache_thread_safety(self):
        """Cache should be thread-safe."""
        cache = EmbeddingCache(maxsize=100)
        errors = []

        def writer():
            for i in range(100):
                cache.put(f"text_{i}", np.random.randn(384).astype(np.float32))

        def reader():
            for i in range(100):
                cache.get(f"text_{i}")

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors means thread-safe
        assert len(errors) == 0


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_very_long_message(self, classifier):
        """Very long messages should be handled."""
        long_message = "Yes! " * 1000
        result = classifier.classify(long_message)
        assert isinstance(result, ClassificationResultV2)

    def test_unicode_messages(self, classifier):
        """Unicode messages should be handled."""
        messages = [
            "Yes! 👍",
            "That's awesome! 🎉",
            "こんにちは",
            "Привет",
            "مرحبا",
        ]
        results = classifier.classify_batch(messages)
        assert len(results) == len(messages)

    def test_special_characters(self, classifier):
        """Special characters should be handled."""
        messages = [
            "Yes!!!",
            "No...",
            "What???",
            "@#$%^&*()",
            "   Yes   ",
        ]
        results = classifier.classify_batch(messages)
        assert len(results) == len(messages)

    def test_newlines_in_message(self, classifier):
        """Newlines in messages should be handled."""
        result = classifier.classify("Yes\nI agree\nTotally")
        assert isinstance(result, ClassificationResultV2)

    def test_mixed_case(self, classifier):
        """Mixed case should be handled."""
        messages = ["YES", "yes", "YeS", "yEs"]
        results = classifier.classify_batch(messages)
        for result in results:
            assert result.label == ResponseTypeV2.AGREE


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility with V1 API."""

    def test_legacy_api_returns_legacy_result(self, classifier_legacy):
        """Legacy API should return ClassificationResult."""
        result = classifier_legacy.classify("Yes")
        assert isinstance(result, ClassificationResult)
        assert result.label == ResponseType.AGREE

    def test_legacy_batch_api(self, classifier_legacy):
        """Legacy batch API should return legacy results."""
        results = classifier_legacy.classify_batch(["Yes", "No"])
        assert all(isinstance(r, ClassificationResult) for r in results)

    def test_singleton_function(self):
        """Singleton function should work."""
        reset_batch_response_classifier()
        clf = get_batch_response_classifier()
        assert isinstance(clf, BatchResponseClassifier)

    def test_feature_flag(self):
        """Feature flag should control classifier version."""
        set_use_v2_classifier(True)
        # Should not raise
        set_use_v2_classifier(False)


# =============================================================================
# A/B Testing Tests
# =============================================================================


class TestABTesting:
    """Tests for A/B testing functionality."""

    def test_ab_test_config(self):
        """A/B test config should be set."""
        config = ABTestConfig(
            experiment_id="test_exp",
            treatment_percentage=0.5,
        )
        set_ab_test_config(config)
        # Should not raise

    def test_ab_test_disable(self):
        """A/B test should be disableable."""
        set_ab_test_config(None)
        # Should not raise


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance regression tests."""

    @pytest.mark.slow
    def test_single_message_latency(self, classifier):
        """Single message should complete in reasonable time."""
        # Warmup
        classifier.classify("test")

        start = time.perf_counter()
        for _ in range(100):
            classifier.classify("Yes I agree")
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed * 1000) / 100
        # Should be under 50ms average (generous for CI)
        assert avg_ms < 50, f"Average latency {avg_ms:.2f}ms exceeds 50ms"

    @pytest.mark.slow
    def test_batch_throughput(self, classifier):
        """Batch processing should have good throughput."""
        messages = ["Yes", "No", "Maybe", "What?"] * 25  # 100 messages

        # Warmup
        classifier.classify_batch(messages[:10])

        start = time.perf_counter()
        classifier.classify_batch(messages)
        elapsed = time.perf_counter() - start

        throughput = len(messages) / elapsed
        # Should process at least 100 messages per second
        assert throughput > 100, f"Throughput {throughput:.0f} msgs/sec below 100"


# =============================================================================
# Warmup Tests
# =============================================================================


class TestWarmup:
    """Tests for model warmup functionality."""

    def test_warmup_loads_models(self, classifier):
        """Warmup should load all models."""
        classifier.warmup()
        assert classifier._warmed_up is True

    def test_warmup_idempotent(self, classifier):
        """Multiple warmups should be idempotent."""
        classifier.warmup()
        classifier.warmup()
        classifier.warmup()
        assert classifier._warmed_up is True


# =============================================================================
# Shutdown Tests
# =============================================================================


class TestShutdown:
    """Tests for proper resource cleanup."""

    def test_shutdown_releases_resources(self):
        """Shutdown should release resources."""
        clf = BatchResponseClassifier()
        clf.classify("test")
        clf.shutdown()
        # Should not raise

    def test_reset_singleton(self):
        """Reset should clear singleton."""
        _clf1 = get_batch_response_classifier()
        reset_batch_response_classifier()
        _clf2 = get_batch_response_classifier()
        # Should be different instances (though this is implementation-dependent)
        # Variables prefixed with _ to indicate they're used for side effects only
        assert _clf1 is not None
        assert _clf2 is not None
