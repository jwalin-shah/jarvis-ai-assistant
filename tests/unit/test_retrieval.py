"""Tests for jarvis.retrieval module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from jarvis.response_classifier import COMMITMENT_RESPONSE_TYPES, ResponseType
from jarvis.retrieval import (
    MultiTypeExamples,
    TypedExample,
    TypedRetriever,
    get_typed_retriever,
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
                ResponseType.AGREE: [
                    TypedExample("Q?", "Yes!", ResponseType.AGREE, 0.9, 0.8)
                ],
                ResponseType.DECLINE: [
                    TypedExample("Q?", "No", ResponseType.DECLINE, 0.8, 0.7)
                ],
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
