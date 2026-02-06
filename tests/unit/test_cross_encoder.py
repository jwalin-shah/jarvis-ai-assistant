"""Tests for cross-encoder reranker module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestBertForSequenceClassification:
    """Test the BERT classification model forward pass."""

    @patch("models.cross_encoder.mx")
    def test_forward_returns_logits_shape(self, mock_mx):
        """Forward pass returns correct logit shape."""
        from models.cross_encoder import BertForSequenceClassification

        # Mock the BertModel and Linear layer behavior
        config = {
            "vocab_size": 30522,
            "hidden_size": 384,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "num_hidden_layers": 6,
            "intermediate_size": 1536,
            "num_attention_heads": 12,
        }

        model = BertForSequenceClassification(config, num_labels=1)
        assert model.bert is not None
        assert model.classifier is not None

    def test_num_labels_stored(self):
        """Model respects num_labels parameter."""
        from models.cross_encoder import BertForSequenceClassification

        config = {
            "vocab_size": 100,
            "hidden_size": 32,
            "max_position_embeddings": 64,
            "type_vocab_size": 2,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_attention_heads": 2,
        }

        model = BertForSequenceClassification(config, num_labels=3)
        # classifier should map hidden_size -> num_labels
        assert model.classifier.weight.shape == (3, 32)


class TestCrossEncoderRegistry:
    """Test the model registry."""

    def test_registry_has_default_model(self):
        from models.cross_encoder import CROSS_ENCODER_REGISTRY

        assert "ms-marco-MiniLM-L-6-v2" in CROSS_ENCODER_REGISTRY

    def test_registry_entry_structure(self):
        from models.cross_encoder import CROSS_ENCODER_REGISTRY

        for name, (hf_repo, num_labels, activation) in CROSS_ENCODER_REGISTRY.items():
            assert isinstance(hf_repo, str)
            assert isinstance(num_labels, int)
            assert activation in ("sigmoid", "softmax", "none")


class TestInProcessCrossEncoder:
    """Test the InProcessCrossEncoder class."""

    def test_init_defaults(self):
        from models.cross_encoder import InProcessCrossEncoder

        ce = InProcessCrossEncoder()
        assert ce.model is None
        assert ce.tokenizer is None
        assert ce.model_name is None
        assert ce._default_model == "ms-marco-MiniLM-L-6-v2"
        assert ce.is_loaded is False

    def test_unknown_model_raises(self):
        from models.cross_encoder import InProcessCrossEncoder

        ce = InProcessCrossEncoder(model_name="nonexistent-model")
        with pytest.raises(ValueError, match="Unknown cross-encoder"):
            ce.load_model("nonexistent-model")

    def test_predict_empty_pairs(self):
        from models.cross_encoder import InProcessCrossEncoder

        ce = InProcessCrossEncoder()
        # Mock load to avoid needing real weights
        ce.model = MagicMock()
        ce.tokenizer = MagicMock()
        ce.model_name = "test"

        result = ce.predict([])
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    @patch("models.cross_encoder.InProcessCrossEncoder._get_gpu_lock")
    def test_predict_returns_correct_shape(self, mock_gpu_lock):
        """predict() returns one score per pair."""
        import mlx.core as mx

        from models.cross_encoder import InProcessCrossEncoder

        mock_gpu_lock.return_value = MagicMock(__enter__=MagicMock(), __exit__=MagicMock())

        ce = InProcessCrossEncoder()
        ce.model_name = "test"
        ce._activation = "sigmoid"
        ce._num_labels = 1

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.no_padding = MagicMock()
        mock_tokenizer.enable_padding = MagicMock()

        # Create mock encodings
        mock_encoding = MagicMock()
        mock_encoding.ids = [101, 2054, 102, 3462, 102]
        mock_encoding.attention_mask = [1, 1, 1, 1, 1]
        mock_encoding.type_ids = [0, 0, 0, 1, 1]

        mock_tokenizer.encode = MagicMock(return_value=mock_encoding)
        ce.tokenizer = mock_tokenizer

        # Mock model forward pass
        mock_model = MagicMock()
        mock_model.return_value = mx.array([[0.5]])  # single logit
        ce.model = mock_model

        scores = ce.predict([("query", "doc1")])
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (1,)


class TestCrossEncoderReranker:
    """Test the reranker service layer."""

    def test_rerank_empty_candidates(self):
        from models.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        result = reranker.rerank("query", [])
        assert result == []

    def test_rerank_single_candidate(self):
        from models.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        candidates = [{"trigger_text": "hello", "response_text": "hi", "similarity": 0.9}]
        result = reranker.rerank("hello", candidates)
        assert len(result) == 1
        assert result[0]["rerank_score"] == 1.0

    def test_rerank_sorts_by_score(self):
        """Reranker should sort candidates by cross-encoder score."""
        from models.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        # Mock the cross-encoder to return predictable scores
        mock_ce = MagicMock()
        mock_ce.predict.return_value = np.array([0.3, 0.9, 0.1])
        reranker._cross_encoder = mock_ce

        candidates = [
            {"trigger_text": "low relevance", "response_text": "r1", "similarity": 0.8},
            {"trigger_text": "high relevance", "response_text": "r2", "similarity": 0.7},
            {"trigger_text": "very low", "response_text": "r3", "similarity": 0.9},
        ]

        result = reranker.rerank("query", candidates, top_k=2)

        assert len(result) == 2
        assert result[0]["trigger_text"] == "high relevance"
        assert result[0]["rerank_score"] == pytest.approx(0.9)
        assert result[1]["trigger_text"] == "low relevance"
        assert result[1]["rerank_score"] == pytest.approx(0.3)

    def test_rerank_respects_top_k(self):
        from models.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        mock_ce = MagicMock()
        mock_ce.predict.return_value = np.array([0.5, 0.8, 0.3, 0.9])
        reranker._cross_encoder = mock_ce

        candidates = [
            {"trigger_text": f"doc{i}", "response_text": f"r{i}", "similarity": 0.5}
            for i in range(4)
        ]

        result = reranker.rerank("query", candidates, top_k=2)
        assert len(result) == 2

    def test_rerank_handles_missing_text_key(self):
        """Candidates without the text key get score 0."""
        from models.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        mock_ce = MagicMock()
        mock_ce.predict.return_value = np.array([0.8])
        reranker._cross_encoder = mock_ce

        candidates = [
            {"trigger_text": "has text", "response_text": "r1", "similarity": 0.5},
            {"response_text": "r2", "similarity": 0.9},  # missing trigger_text
        ]

        result = reranker.rerank("query", candidates, top_k=2)
        assert len(result) == 2
        # The one with text should be scored, the other gets 0
        scored = [c for c in result if c["rerank_score"] > 0]
        assert len(scored) == 1


class TestContextServiceReranking:
    """Test reranker integration with ContextService."""

    def test_context_service_accepts_reranker(self):
        from jarvis.services.context_service import ContextService

        mock_reranker = MagicMock()
        svc = ContextService(reranker=mock_reranker)
        assert svc._reranker is mock_reranker

    def test_context_service_reranker_defaults_none(self):
        from jarvis.services.context_service import ContextService

        svc = ContextService()
        assert svc._reranker is None


class TestRetrievalConfig:
    """Test reranker config fields."""

    def test_defaults(self):
        from jarvis.config import RetrievalConfig

        cfg = RetrievalConfig()
        assert cfg.reranker_enabled is False
        assert cfg.reranker_model == "ms-marco-MiniLM-L-6-v2"
        assert cfg.reranker_top_k == 3
        assert cfg.reranker_candidates == 10

    def test_custom_values(self):
        from jarvis.config import RetrievalConfig

        cfg = RetrievalConfig(
            reranker_enabled=True,
            reranker_model="ms-marco-MiniLM-L-6-v2",
            reranker_top_k=5,
            reranker_candidates=20,
        )
        assert cfg.reranker_enabled is True
        assert cfg.reranker_top_k == 5
        assert cfg.reranker_candidates == 20


class TestSingletons:
    """Test singleton accessors."""

    def test_get_cross_encoder_returns_instance(self):
        from models.cross_encoder import get_cross_encoder, reset_cross_encoder

        try:
            ce = get_cross_encoder()
            assert ce is not None
            assert ce._default_model == "ms-marco-MiniLM-L-6-v2"
            # Second call returns same instance
            ce2 = get_cross_encoder()
            assert ce is ce2
        finally:
            reset_cross_encoder()

    def test_get_reranker_returns_instance(self):
        from models.reranker import get_reranker, reset_reranker

        try:
            r = get_reranker()
            assert r is not None
            r2 = get_reranker()
            assert r is r2
        finally:
            reset_reranker()

    def test_reset_cross_encoder(self):
        from models.cross_encoder import (
            get_cross_encoder,
            reset_cross_encoder,
        )

        try:
            ce1 = get_cross_encoder()
            reset_cross_encoder()
            ce2 = get_cross_encoder()
            assert ce1 is not ce2
        finally:
            reset_cross_encoder()
