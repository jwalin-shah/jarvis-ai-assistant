"""Tests for jarvis.retrieval module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from jarvis.response_types import COMMITMENT_RESPONSE_TYPES, ResponseType
from jarvis.search.retrieval import (
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


class TestTypedRetriever:
    """Tests for TypedRetriever class."""

    def test_init_default(self):
        """Can create with default settings."""
        retriever = TypedRetriever()
        assert retriever._db is None  # Lazy loaded
        assert retriever._vec_searcher is None  # Lazy loaded
        assert retriever._trigger_classifier is None  # Lazy loaded

    @patch("jarvis.search.retrieval.get_db")
    def test_db_property_lazy_loads(self, mock_get_db):
        """db property lazy loads database."""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        retriever = TypedRetriever()
        db = retriever.db

        mock_get_db.assert_called_once()
        mock_db.init_schema.assert_called_once()
        assert db == mock_db

    @patch("jarvis.search.retrieval.get_trigger_classifier")
    def test_trigger_classifier_property_lazy_loads(self, mock_get_classifier):
        """trigger_classifier property lazy loads classifier."""
        mock_classifier = MagicMock()
        mock_get_classifier.return_value = mock_classifier

        retriever = TypedRetriever()
        classifier = retriever.trigger_classifier

        mock_get_classifier.assert_called_once()
        assert classifier == mock_classifier

    def test_classify_trigger_no_classifier(self):
        """classify_trigger returns (None, 0.0) when classifier unavailable."""
        retriever = TypedRetriever()
        retriever._trigger_classifier = None

        with patch(
            "jarvis.search.retrieval.get_trigger_classifier", side_effect=Exception("Not available")
        ):
            result = retriever.classify_trigger("test")
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

    def test_filters_by_response_type(self):
        """Results are filtered by target response type."""
        with patch.object(TypedRetriever, "vec_searcher") as mock_searcher:
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
        with patch.object(TypedRetriever, "vec_searcher") as mock_searcher:
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


class TestGetExamplesForCommitment:
    """Tests for get_examples_for_commitment method."""

    @patch("jarvis.embedding_adapter.get_embedder")
    def test_returns_multi_type_examples(self, mock_get_embedder):
        """Returns MultiTypeExamples with examples by type."""
        with patch.object(TypedRetriever, "vec_searcher") as mock_searcher:
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
        assert ResponseType.QUESTION not in COMMITMENT_RESPONSE_TYPES
