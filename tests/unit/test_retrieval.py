"""Tests for jarvis.retrieval module."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from jarvis.response_classifier import COMMITMENT_RESPONSE_TYPES, ResponseType
from jarvis.retrieval import (
    BM25IndexManager,
    CrossEncoderReranker,
    MultiTypeExamples,
    TypedExample,
    TypedRetriever,
    apply_temporal_decay_to_results,
    compute_temporal_decay,
    get_bm25_index,
    get_reranker,
    get_typed_retriever,
    hybrid_search_with_bm25,
    reciprocal_rank_fusion,
    reset_bm25_index,
    reset_reranker,
    reset_typed_retriever,
)


class TestTypedExample:
    """Tests for TypedExample dataclass."""

    def test_basic_creation(self):
        """Can create TypedExample with required fields."""
        example = TypedExample(
            trigger_text="Want to grab lunch?",
            response_text="Yeah I'm down!",
            response_type=ResponseType.AGREE,
            similarity=0.95,
            confidence=0.9,
        )
        assert example.trigger_text == "Want to grab lunch?"
        assert example.response_text == "Yeah I'm down!"
        assert example.response_type == ResponseType.AGREE
        assert example.similarity == 0.95
        assert example.confidence == 0.9
        assert example.pair_id is None

    def test_with_pair_id(self):
        """Can create with optional pair_id."""
        example = TypedExample(
            trigger_text="Test?",
            response_text="Yes",
            response_type=ResponseType.AGREE,
            similarity=0.8,
            confidence=0.7,
            pair_id=123,
        )
        assert example.pair_id == 123


class TestMultiTypeExamples:
    """Tests for MultiTypeExamples dataclass."""

    def test_basic_creation(self):
        """Can create MultiTypeExamples with required fields."""
        examples = MultiTypeExamples(
            query_trigger="Want to hang out?",
            trigger_da="commitment",
            examples_by_type={},
        )
        assert examples.query_trigger == "Want to hang out?"
        assert examples.trigger_da == "commitment"
        assert examples.examples_by_type == {}

    def test_get_examples_empty(self):
        """get_examples returns empty list for missing type."""
        examples = MultiTypeExamples(
            query_trigger="Test?",
            trigger_da="commitment",
            examples_by_type={},
        )
        result = examples.get_examples(ResponseType.AGREE)
        assert result == []

    def test_get_examples_with_data(self):
        """get_examples returns examples for existing type."""
        agree_examples = [
            TypedExample("Q?", "Yes!", ResponseType.AGREE, 0.9, 0.8),
        ]
        examples = MultiTypeExamples(
            query_trigger="Test?",
            trigger_da="commitment",
            examples_by_type={ResponseType.AGREE: agree_examples},
        )
        result = examples.get_examples(ResponseType.AGREE)
        assert len(result) == 1
        assert result[0].response_text == "Yes!"

    def test_has_examples_true(self):
        """has_examples returns True when type has examples."""
        agree_examples = [
            TypedExample("Q?", "Yes!", ResponseType.AGREE, 0.9, 0.8),
        ]
        examples = MultiTypeExamples(
            query_trigger="Test?",
            trigger_da="commitment",
            examples_by_type={ResponseType.AGREE: agree_examples},
        )
        assert examples.has_examples(ResponseType.AGREE) is True

    def test_has_examples_false_missing(self):
        """has_examples returns False for missing type."""
        examples = MultiTypeExamples(
            query_trigger="Test?",
            trigger_da="commitment",
            examples_by_type={},
        )
        assert examples.has_examples(ResponseType.AGREE) is False

    def test_has_examples_false_empty(self):
        """has_examples returns False for empty list."""
        examples = MultiTypeExamples(
            query_trigger="Test?",
            trigger_da="commitment",
            examples_by_type={ResponseType.AGREE: []},
        )
        assert examples.has_examples(ResponseType.AGREE) is False

    def test_available_types_property(self):
        """available_types returns types with examples."""
        examples = MultiTypeExamples(
            query_trigger="Test?",
            trigger_da="commitment",
            examples_by_type={
                ResponseType.AGREE: [TypedExample("Q?", "Yes!", ResponseType.AGREE, 0.9, 0.8)],
                ResponseType.DECLINE: [TypedExample("Q?", "No", ResponseType.DECLINE, 0.8, 0.7)],
                ResponseType.DEFER: [],  # Empty - shouldn't be included
            },
        )
        available = examples.available_types
        assert ResponseType.AGREE in available
        assert ResponseType.DECLINE in available
        assert ResponseType.DEFER not in available


class TestTypedRetriever:
    """Tests for TypedRetriever class."""

    def test_init_default(self):
        """Can create with default settings."""
        retriever = TypedRetriever()
        assert retriever._db is None  # Lazy loaded
        assert retriever._index_searcher is None  # Lazy loaded
        assert retriever._trigger_classifier is None  # Lazy loaded

    @patch("jarvis.retrieval.get_db")
    def test_db_property_lazy_loads(self, mock_get_db):
        """db property lazy loads database."""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        retriever = TypedRetriever()
        db = retriever.db

        mock_get_db.assert_called_once()
        mock_db.init_schema.assert_called_once()
        assert db == mock_db

    @patch("jarvis.retrieval.get_trigger_classifier")
    def test_trigger_classifier_property_lazy_loads(self, mock_get_classifier):
        """trigger_classifier property lazy loads classifier."""
        mock_classifier = MagicMock()
        mock_get_classifier.return_value = mock_classifier

        retriever = TypedRetriever()
        classifier = retriever.trigger_classifier

        mock_get_classifier.assert_called_once()
        assert classifier == mock_classifier

    @patch("jarvis.retrieval.get_trigger_classifier")
    def test_classify_trigger_returns_type_and_confidence(self, mock_get_classifier):
        """classify_trigger returns (type, confidence) tuple."""
        mock_classifier = MagicMock()
        mock_result = MagicMock()
        mock_result.trigger_type.value = "commitment"
        mock_result.confidence = 0.95
        mock_classifier.classify.return_value = mock_result
        mock_get_classifier.return_value = mock_classifier

        retriever = TypedRetriever()
        trigger_da, confidence = retriever.classify_trigger("want to hang out?")

        assert trigger_da == "commitment"
        assert confidence == 0.95

    def test_classify_trigger_no_classifier(self):
        """classify_trigger returns (None, 0.0) when classifier unavailable."""
        retriever = TypedRetriever()
        retriever._trigger_classifier = None

        # Patch to return None
        with patch.object(
            TypedRetriever, "trigger_classifier", new_callable=lambda: property(lambda self: None)
        ):
            retriever2 = TypedRetriever()
            # Force _trigger_classifier to None to simulate unavailable
            retriever2._trigger_classifier = None

            # Create a new retriever that returns None for trigger_classifier
            with patch(
                "jarvis.retrieval.get_trigger_classifier", side_effect=Exception("Not available")
            ):
                r = TypedRetriever()
                result = r.classify_trigger("test")
                assert result == (None, 0.0)


class TestGetTypedExamples:
    """Tests for get_typed_examples method."""

    def test_invalid_response_type_string(self):
        """Invalid response type string returns empty list."""
        retriever = TypedRetriever()
        result = retriever.get_typed_examples(
            trigger="test?",
            target_response_type="INVALID_TYPE",
        )
        assert result == []

    def test_valid_response_type_enum(self):
        """Can pass ResponseType enum directly."""
        with patch.object(TypedRetriever, "index_searcher") as mock_searcher:
            mock_searcher.search_with_pairs.return_value = []

            retriever = TypedRetriever()
            result = retriever.get_typed_examples(
                trigger="test?",
                target_response_type=ResponseType.AGREE,
            )
            assert result == []

    def test_filters_by_response_type(self):
        """Results are filtered by target response type."""
        with patch.object(TypedRetriever, "index_searcher") as mock_searcher:
            # Return mixed results
            mock_searcher.search_with_pairs.return_value = [
                {
                    "pair_id": 1,
                    "trigger_text": "Q1?",
                    "response_text": "Yes",
                    "response_da_type": "AGREE",
                    "response_da_conf": 0.9,
                    "similarity": 0.95,
                    "quality_score": 0.8,
                },
                {
                    "pair_id": 2,
                    "trigger_text": "Q2?",
                    "response_text": "No",
                    "response_da_type": "DECLINE",
                    "response_da_conf": 0.9,
                    "similarity": 0.90,
                    "quality_score": 0.8,
                },
            ]

            retriever = TypedRetriever()
            result = retriever.get_typed_examples(
                trigger="test?",
                target_response_type=ResponseType.AGREE,
            )

            # Should only return AGREE examples
            assert len(result) == 1
            assert result[0].response_type == ResponseType.AGREE

    def test_respects_min_quality(self):
        """Results below min_quality are filtered out."""
        with patch.object(TypedRetriever, "index_searcher") as mock_searcher:
            mock_searcher.search_with_pairs.return_value = [
                {
                    "pair_id": 1,
                    "trigger_text": "Q?",
                    "response_text": "Yes",
                    "response_da_type": "AGREE",
                    "response_da_conf": 0.9,
                    "similarity": 0.95,
                    "quality_score": 0.3,  # Below threshold
                },
            ]

            retriever = TypedRetriever()
            result = retriever.get_typed_examples(
                trigger="test?",
                target_response_type=ResponseType.AGREE,
                min_quality=0.5,
            )

            assert len(result) == 0

    def test_limits_to_k_results(self):
        """Results are limited to k."""
        with patch.object(TypedRetriever, "index_searcher") as mock_searcher:
            # Return many results
            mock_searcher.search_with_pairs.return_value = [
                {
                    "pair_id": i,
                    "trigger_text": f"Q{i}?",
                    "response_text": "Yes",
                    "response_da_type": "AGREE",
                    "response_da_conf": 0.9,
                    "similarity": 0.9 - i * 0.01,
                    "quality_score": 0.8,
                }
                for i in range(10)
            ]

            retriever = TypedRetriever()
            result = retriever.get_typed_examples(
                trigger="test?",
                target_response_type=ResponseType.AGREE,
                k=3,
            )

            assert len(result) == 3


class TestGetExamplesForCommitment:
    """Tests for get_examples_for_commitment method."""

    @patch("jarvis.embedding_adapter.get_embedder")
    def test_returns_multi_type_examples(self, mock_get_embedder):
        """Returns MultiTypeExamples with examples by type."""
        with patch.object(TypedRetriever, "index_searcher") as mock_searcher:
            with patch.object(TypedRetriever, "classify_trigger") as mock_classify:
                mock_classify.return_value = ("commitment", 0.9)

                # Mock embedder
                mock_emb = MagicMock()
                mock_get_embedder.return_value = mock_emb

                # Return examples for different types
                mock_searcher.search_with_pairs.return_value = [
                    {
                        "pair_id": 1,
                        "trigger_text": "Q?",
                        "response_text": "Yes",
                        "response_da_type": "AGREE",
                        "response_da_conf": 0.9,
                        "similarity": 0.95,
                        "quality_score": 0.8,
                    },
                    {
                        "pair_id": 2,
                        "trigger_text": "Q?",
                        "response_text": "No",
                        "response_da_type": "DECLINE",
                        "response_da_conf": 0.9,
                        "similarity": 0.90,
                        "quality_score": 0.8,
                    },
                    {
                        "pair_id": 3,
                        "trigger_text": "Q?",
                        "response_text": "Maybe",
                        "response_da_type": "DEFER",
                        "response_da_conf": 0.9,
                        "similarity": 0.85,
                        "quality_score": 0.8,
                    },
                ]

                retriever = TypedRetriever()
                result = retriever.get_examples_for_commitment("want to hang?")

                assert isinstance(result, MultiTypeExamples)
                assert result.trigger_da == "commitment"
                assert result.has_examples(ResponseType.AGREE)
                assert result.has_examples(ResponseType.DECLINE)
                assert result.has_examples(ResponseType.DEFER)

    @patch("jarvis.embedding_adapter.get_embedder")
    def test_uses_provided_trigger_da(self, mock_get_embedder):
        """Uses provided trigger_da instead of re-classifying."""
        with patch.object(TypedRetriever, "index_searcher") as mock_searcher:
            with patch.object(TypedRetriever, "classify_trigger") as mock_classify:
                mock_searcher.search_with_pairs.return_value = []
                mock_emb = MagicMock()
                mock_get_embedder.return_value = mock_emb

                retriever = TypedRetriever()
                result = retriever.get_examples_for_commitment(
                    "test?",
                    trigger_da="commitment",  # Pre-classified
                )

                # classify_trigger should NOT be called
                mock_classify.assert_not_called()
                assert result.trigger_da == "commitment"


class TestSingletonFactory:
    """Tests for singleton factory functions."""

    def test_get_typed_retriever_returns_singleton(self):
        """get_typed_retriever returns same instance."""
        reset_typed_retriever()

        r1 = get_typed_retriever()
        r2 = get_typed_retriever()

        assert r1 is r2

    def test_reset_typed_retriever_clears_singleton(self):
        """reset_typed_retriever clears the singleton."""
        r1 = get_typed_retriever()
        reset_typed_retriever()
        r2 = get_typed_retriever()

        assert r1 is not r2


class TestCommitmentResponseTypes:
    """Tests related to commitment response types."""

    def test_commitment_types_used_for_filtering(self):
        """COMMITMENT_RESPONSE_TYPES used correctly."""
        assert ResponseType.AGREE in COMMITMENT_RESPONSE_TYPES
        assert ResponseType.DECLINE in COMMITMENT_RESPONSE_TYPES
        assert ResponseType.DEFER in COMMITMENT_RESPONSE_TYPES

        # Non-commitment types should not be included
        assert ResponseType.QUESTION not in COMMITMENT_RESPONSE_TYPES
        assert ResponseType.STATEMENT not in COMMITMENT_RESPONSE_TYPES


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_search_results(self):
        """Empty search results handled gracefully."""
        with patch.object(TypedRetriever, "index_searcher") as mock_searcher:
            mock_searcher.search_with_pairs.return_value = []

            retriever = TypedRetriever()
            result = retriever.get_typed_examples(
                trigger="test?",
                target_response_type=ResponseType.AGREE,
            )

            assert result == []

    def test_search_exception_handled(self):
        """Search exceptions handled gracefully."""
        with patch.object(TypedRetriever, "index_searcher") as mock_searcher:
            mock_searcher.search_with_pairs.side_effect = Exception("Search failed")

            retriever = TypedRetriever()
            result = retriever.get_typed_examples(
                trigger="test?",
                target_response_type=ResponseType.AGREE,
            )

            assert result == []

    def test_missing_pair_id_skipped(self):
        """Results without pair_id are skipped."""
        with patch.object(TypedRetriever, "index_searcher") as mock_searcher:
            mock_searcher.search_with_pairs.return_value = [
                {
                    # No pair_id
                    "trigger_text": "Q?",
                    "response_text": "Yes",
                    "response_da_type": "AGREE",
                    "similarity": 0.95,
                },
            ]

            retriever = TypedRetriever()
            result = retriever.get_typed_examples(
                trigger="test?",
                target_response_type=ResponseType.AGREE,
            )

            assert result == []


# =============================================================================
# Temporal Decay Tests
# =============================================================================


class TestComputeTemporalDecay:
    """Tests for compute_temporal_decay function."""

    def test_no_timestamp_returns_one(self):
        """None timestamp returns 1.0 (no decay)."""
        result = compute_temporal_decay(None)
        assert result == 1.0

    def test_current_timestamp_returns_one(self):
        """Current timestamp returns 1.0."""
        result = compute_temporal_decay(datetime.now())
        assert result == 1.0

    def test_future_timestamp_returns_one(self):
        """Future timestamp returns 1.0."""
        future = datetime.now() + timedelta(days=30)
        result = compute_temporal_decay(future)
        assert result == 1.0

    def test_half_life_decay(self):
        """Message exactly half_life_days old returns ~0.5."""
        half_life = 365.0
        old_date = datetime.now() - timedelta(days=365)
        result = compute_temporal_decay(old_date, half_life_days=half_life)
        assert abs(result - 0.5) < 0.01  # Allow small float error

    def test_double_half_life_decay(self):
        """Message 2x half_life_days old returns ~0.25."""
        half_life = 365.0
        old_date = datetime.now() - timedelta(days=730)
        result = compute_temporal_decay(old_date, half_life_days=half_life)
        assert abs(result - 0.25) < 0.01

    def test_min_score_clamp(self):
        """Very old messages are clamped to min_score."""
        very_old = datetime.now() - timedelta(days=3650)  # 10 years
        result = compute_temporal_decay(very_old, half_life_days=365.0, min_score=0.1)
        assert result == 0.1

    def test_custom_half_life(self):
        """Custom half_life works correctly."""
        # With 30 day half life, 30 days old = 0.5
        old_date = datetime.now() - timedelta(days=30)
        result = compute_temporal_decay(old_date, half_life_days=30.0)
        assert abs(result - 0.5) < 0.01

    def test_zero_min_score(self):
        """min_score=0 allows full decay."""
        very_old = datetime.now() - timedelta(days=3650)
        result = compute_temporal_decay(very_old, half_life_days=365.0, min_score=0.0)
        assert result < 0.01  # Should be very small


class TestApplyTemporalDecayToResults:
    """Tests for apply_temporal_decay_to_results function."""

    @patch("jarvis.retrieval.get_config")
    def test_adds_temporal_score(self, mock_get_config):
        """Adds temporal_score key to results."""
        mock_config = MagicMock()
        mock_config.retrieval.temporal_decay_enabled = True
        mock_config.retrieval.temporal_half_life_days = 365.0
        mock_config.retrieval.temporal_min_score = 0.1
        mock_get_config.return_value = mock_config

        results = [
            {"similarity": 0.9, "source_timestamp": datetime.now()},
            {"similarity": 0.8, "source_timestamp": datetime.now() - timedelta(days=365)},
        ]

        updated = apply_temporal_decay_to_results(results)

        assert "temporal_score" in updated[0]
        assert "temporal_decay" in updated[0]
        # Recent message should have higher temporal_score
        assert updated[0]["temporal_score"] > updated[1]["temporal_score"]

    @patch("jarvis.retrieval.get_config")
    def test_sorts_by_temporal_score(self, mock_get_config):
        """Results are sorted by temporal_score descending."""
        mock_config = MagicMock()
        mock_config.retrieval.temporal_decay_enabled = True
        mock_config.retrieval.temporal_half_life_days = 30.0
        mock_config.retrieval.temporal_min_score = 0.1
        mock_get_config.return_value = mock_config

        # Higher similarity but older message vs lower similarity but recent
        results = [
            {"similarity": 0.95, "source_timestamp": datetime.now() - timedelta(days=90)},
            {"similarity": 0.7, "source_timestamp": datetime.now()},
        ]

        updated = apply_temporal_decay_to_results(results)

        # Recent message should be first despite lower similarity
        assert updated[0]["similarity"] == 0.7

    @patch("jarvis.retrieval.get_config")
    def test_disabled_returns_original_score(self, mock_get_config):
        """When disabled, temporal_score equals similarity."""
        mock_config = MagicMock()
        mock_config.retrieval.temporal_decay_enabled = False
        mock_get_config.return_value = mock_config

        results = [
            {"similarity": 0.9, "source_timestamp": datetime.now() - timedelta(days=1000)},
        ]

        updated = apply_temporal_decay_to_results(results)

        assert updated[0]["temporal_score"] == 0.9


# =============================================================================
# Reciprocal Rank Fusion Tests
# =============================================================================


class TestReciprocalRankFusion:
    """Tests for reciprocal_rank_fusion function."""

    def test_single_ranking(self):
        """Single ranking returns items in same order."""
        ranking = [(1, 0.9), (2, 0.8), (3, 0.7)]
        result = reciprocal_rank_fusion([ranking])

        assert result[0][0] == 1  # First item
        assert result[1][0] == 2  # Second item
        assert result[2][0] == 3  # Third item

    def test_two_rankings_fusion(self):
        """Two rankings are fused correctly."""
        r1 = [(1, 0.9), (2, 0.8), (3, 0.7)]
        r2 = [(2, 0.95), (1, 0.85), (4, 0.6)]

        result = reciprocal_rank_fusion([r1, r2])

        # Item 2 is ranked 1st in r2 and 2nd in r1 - should score high
        # Item 1 is ranked 1st in r1 and 2nd in r2 - should also score high
        result_ids = [item[0] for item in result]
        assert 1 in result_ids
        assert 2 in result_ids
        # Both rankings' items should be in result
        assert 4 in result_ids

    def test_empty_rankings(self):
        """Empty rankings return empty list."""
        result = reciprocal_rank_fusion([])
        assert result == []

    def test_rrf_k_parameter(self):
        """Different k values affect scores."""
        ranking = [(1, 0.9), (2, 0.8)]

        # Lower k gives higher scores to top ranks
        result_low_k = reciprocal_rank_fusion([ranking], k=1)
        # Higher k spreads scores more evenly
        result_high_k = reciprocal_rank_fusion([ranking], k=100)

        # With k=1: score of rank 1 = 1/(1+1) = 0.5
        # With k=100: score of rank 1 = 1/(100+1) ≈ 0.01
        assert result_low_k[0][1] > result_high_k[0][1]


# =============================================================================
# BM25 Index Manager Tests
# =============================================================================


class TestBM25IndexManager:
    """Tests for BM25IndexManager class."""

    def test_init_not_built(self):
        """New instance is not built."""
        manager = BM25IndexManager()
        assert manager.is_built is False

    def test_build_index(self):
        """Can build index from pairs."""
        manager = BM25IndexManager()
        pairs = [
            {"pair_id": 1, "trigger_text": "hello world"},
            {"pair_id": 2, "trigger_text": "goodbye world"},
            {"pair_id": 3, "trigger_text": "hello there"},
        ]
        manager.build_index(pairs)

        assert manager.is_built is True

    def test_search_after_build(self):
        """Can search after building."""
        manager = BM25IndexManager()
        # Need enough diverse documents for BM25 IDF to produce non-zero scores
        pairs = [
            {"pair_id": 1, "trigger_text": "hello world greeting"},
            {"pair_id": 2, "trigger_text": "goodbye cruel world"},
            {"pair_id": 3, "trigger_text": "something different here"},
            {"pair_id": 4, "trigger_text": "another unique text"},
            {"pair_id": 5, "trigger_text": "more documents needed"},
        ]
        manager.build_index(pairs)

        results = manager.search("hello", k=2)

        # Should find document with "hello"
        assert len(results) >= 1
        result_ids = [r[0] for r in results]
        assert 1 in result_ids  # Document 1 contains "hello"

    def test_search_before_build_returns_empty(self):
        """Search before build returns empty list."""
        manager = BM25IndexManager()
        results = manager.search("hello")
        assert results == []

    def test_search_no_matches_returns_empty(self):
        """Search with no matches returns empty list."""
        manager = BM25IndexManager()
        pairs = [
            {"pair_id": 1, "trigger_text": "hello world"},
        ]
        manager.build_index(pairs)

        results = manager.search("xyz123nonexistent")
        assert results == []

    def test_clear_resets_state(self):
        """Clear resets the index."""
        manager = BM25IndexManager()
        pairs = [{"pair_id": 1, "trigger_text": "hello world"}]
        manager.build_index(pairs)

        assert manager.is_built is True
        manager.clear()
        assert manager.is_built is False

    def test_empty_pairs_warning(self):
        """Empty pairs logs warning."""
        manager = BM25IndexManager()
        manager.build_index([])
        assert manager.is_built is False


class TestBM25Singleton:
    """Tests for BM25 singleton functions."""

    def test_get_bm25_index_returns_singleton(self):
        """get_bm25_index returns same instance."""
        reset_bm25_index()

        idx1 = get_bm25_index()
        idx2 = get_bm25_index()

        assert idx1 is idx2

    def test_reset_bm25_index_clears_singleton(self):
        """reset_bm25_index clears the singleton."""
        idx1 = get_bm25_index()
        reset_bm25_index()
        idx2 = get_bm25_index()

        assert idx1 is not idx2


# =============================================================================
# Hybrid Search Tests
# =============================================================================


class TestHybridSearchWithBM25:
    """Tests for hybrid_search_with_bm25 function."""

    @patch("jarvis.retrieval.get_config")
    def test_disabled_returns_faiss_only(self, mock_get_config):
        """When BM25 disabled, returns FAISS results only."""
        mock_config = MagicMock()
        mock_config.retrieval.bm25_enabled = False
        mock_get_config.return_value = mock_config

        faiss_results = [
            {"pair_id": 1, "similarity": 0.9},
            {"pair_id": 2, "similarity": 0.8},
        ]
        bm25_index = BM25IndexManager()

        result = hybrid_search_with_bm25("test", faiss_results, bm25_index, k=5)

        assert result == faiss_results[:5]

    @patch("jarvis.retrieval.get_config")
    def test_unbuilt_index_returns_faiss_only(self, mock_get_config):
        """When BM25 index not built, returns FAISS results only."""
        mock_config = MagicMock()
        mock_config.retrieval.bm25_enabled = True
        mock_get_config.return_value = mock_config

        faiss_results = [
            {"pair_id": 1, "similarity": 0.9},
        ]
        bm25_index = BM25IndexManager()  # Not built

        result = hybrid_search_with_bm25("test", faiss_results, bm25_index, k=5)

        assert result == faiss_results

    @patch("jarvis.retrieval.get_config")
    def test_fusion_adds_rrf_score(self, mock_get_config):
        """Fusion adds rrf_score to results."""
        mock_config = MagicMock()
        mock_config.retrieval.bm25_enabled = True
        mock_config.retrieval.rrf_k = 60
        mock_get_config.return_value = mock_config

        faiss_results = [
            {"pair_id": 1, "similarity": 0.9, "trigger_text": "hello world"},
            {"pair_id": 2, "similarity": 0.8, "trigger_text": "goodbye world"},
        ]

        bm25_index = BM25IndexManager()
        bm25_index.build_index([
            {"pair_id": 1, "trigger_text": "hello world"},
            {"pair_id": 2, "trigger_text": "goodbye world"},
        ])

        result = hybrid_search_with_bm25("hello", faiss_results, bm25_index, k=5)

        assert len(result) > 0
        assert "rrf_score" in result[0]


# =============================================================================
# Cross-Encoder Reranker Tests
# =============================================================================


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker class."""

    def test_init_default_model(self):
        """Initialize with default model from config."""
        with patch("jarvis.retrieval.get_config") as mock_config:
            mock_config.return_value.retrieval.rerank_model = "test-model"
            reranker = CrossEncoderReranker()
            assert reranker._model_name == "test-model"

    def test_init_custom_model(self):
        """Initialize with custom model."""
        with patch("jarvis.retrieval.get_config"):
            reranker = CrossEncoderReranker(model_name="custom-model")
            assert reranker._model_name == "custom-model"

    def test_rerank_empty_candidates(self):
        """Rerank with empty candidates returns empty list."""
        with patch("jarvis.retrieval.get_config") as mock_config:
            mock_config.return_value.retrieval.rerank_model = "test-model"
            reranker = CrossEncoderReranker()
            result = reranker.rerank("query", [])
            assert result == []

    def test_rerank_model_load_failure_returns_original(self):
        """If model fails to load, returns original candidates."""
        with patch("jarvis.retrieval.get_config") as mock_config:
            mock_config.return_value.retrieval.rerank_model = "nonexistent-model"
            reranker = CrossEncoderReranker()

            candidates = [
                {"trigger_text": "test1", "score": 0.9},
                {"trigger_text": "test2", "score": 0.8},
            ]

            # Mock the load to fail
            with patch.object(reranker, "_load_model", side_effect=Exception("Load failed")):
                result = reranker.rerank("query", candidates)

            # Should return original candidates
            assert len(result) == 2

    def test_rerank_respects_top_k(self):
        """Rerank respects top_k parameter."""
        with patch("jarvis.retrieval.get_config") as mock_config:
            mock_config.return_value.retrieval.rerank_model = "test-model"
            reranker = CrossEncoderReranker()

            candidates = [
                {"trigger_text": f"test{i}", "score": 0.9 - i * 0.1} for i in range(10)
            ]

            with patch.object(reranker, "_load_model", side_effect=Exception("Load failed")):
                result = reranker.rerank("query", candidates, top_k=3)

            assert len(result) == 3


class TestRerankerSingleton:
    """Tests for reranker singleton functions."""

    def test_get_reranker_returns_singleton(self):
        """get_reranker returns same instance."""
        reset_reranker()

        r1 = get_reranker()
        r2 = get_reranker()

        assert r1 is r2

    def test_reset_reranker_clears_singleton(self):
        """reset_reranker clears the singleton."""
        r1 = get_reranker()
        reset_reranker()
        r2 = get_reranker()

        assert r1 is not r2
