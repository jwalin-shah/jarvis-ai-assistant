"""Tests for jarvis/embedding_adapter.py - Unified embedding interface."""

import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from jarvis.embedding_adapter import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_REGISTRY,
    CachedEmbedder,
    MLXEmbedder,
    get_model_info,
)


class TestGetModelInfo:
    """Test model registry lookups."""

    def test_known_model(self):
        hf_id, mlx_name = get_model_info("bge-small")
        assert hf_id == "BAAI/bge-small-en-v1.5"
        assert mlx_name == "bge-small"

    def test_all_registered_models(self):
        for name in EMBEDDING_MODEL_REGISTRY:
            hf_id, mlx_name = get_model_info(name)
            assert hf_id, f"Empty HuggingFace ID for {name}"
            assert mlx_name, f"Empty MLX name for {name}"

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding model"):
            get_model_info("nonexistent-model")


class TestMLXEmbedder:
    """Test MLXEmbedder with mocked MLX backend."""

    @pytest.fixture()
    def mock_embedder(self):
        """Create an MLXEmbedder with mocked internals."""
        embedder = MLXEmbedder()
        mock_mlx = MagicMock()
        mock_mlx.encode.return_value = np.random.randn(1, EMBEDDING_DIM).astype(np.float32)
        embedder._mlx_embedder = mock_mlx
        embedder._model_name = "bge-small"
        embedder._init_attempted = True
        return embedder

    def test_encode_single_string(self, mock_embedder):
        mock_embedder._mlx_embedder.encode.return_value = np.ones(
            (1, EMBEDDING_DIM), dtype=np.float32
        )
        result = mock_embedder.encode("hello")
        assert result.shape == (1, EMBEDDING_DIM)
        # Single string gets wrapped in list
        mock_embedder._mlx_embedder.encode.assert_called_once()

    def test_encode_list_of_strings(self, mock_embedder):
        texts = ["hello", "world"]
        mock_embedder._mlx_embedder.encode.return_value = np.ones(
            (2, EMBEDDING_DIM), dtype=np.float32
        )
        result = mock_embedder.encode(texts)
        assert result.shape == (2, EMBEDDING_DIM)

    def test_encode_empty_list(self, mock_embedder):
        result = mock_embedder.encode([])
        assert result.shape == (0, EMBEDDING_DIM)
        mock_embedder._mlx_embedder.encode.assert_not_called()

    def test_embedding_dim_property(self, mock_embedder):
        assert mock_embedder.embedding_dim == EMBEDDING_DIM

    def test_backend_property(self, mock_embedder):
        assert mock_embedder.backend == "mlx"

    def test_model_name_property(self, mock_embedder):
        assert mock_embedder.model_name == "bge-small"

    def test_normalize_embeddings_alias(self, mock_embedder):
        """normalize_embeddings param should override normalize."""
        mock_embedder._mlx_embedder.encode.return_value = np.ones(
            (1, EMBEDDING_DIM), dtype=np.float32
        )
        mock_embedder.encode("test", normalize_embeddings=False)
        call_kwargs = mock_embedder._mlx_embedder.encode.call_args[1]
        assert call_kwargs["normalize"] is False

    def test_is_available_true(self, mock_embedder):
        assert mock_embedder.is_available() is True

    def test_is_available_false_when_no_backend(self):
        embedder = MLXEmbedder()
        embedder._init_attempted = True
        embedder._mlx_embedder = None
        assert embedder.is_available() is False

    def test_unload_resets_state(self, mock_embedder):
        mock_embedder.unload()
        assert mock_embedder._init_attempted is False


class TestCachedEmbedder:
    """Test CachedEmbedder caching behavior."""

    @pytest.fixture()
    def cached_embedder(self):
        """Create a CachedEmbedder with mocked base."""
        base = MagicMock(spec=MLXEmbedder)
        base.backend = "mlx"
        base.embedding_dim = EMBEDDING_DIM
        base.model_name = "bge-small"
        base.is_available.return_value = True
        base.encode.return_value = np.random.randn(1, EMBEDDING_DIM).astype(np.float32)
        return CachedEmbedder(base, maxsize=100)

    def test_cache_miss_calls_base(self, cached_embedder):
        cached_embedder.encode("hello")
        cached_embedder.base.encode.assert_called_once()

    def test_cache_hit_skips_base(self, cached_embedder):
        vec = np.ones((1, EMBEDDING_DIM), dtype=np.float32)
        cached_embedder.base.encode.return_value = vec
        cached_embedder.encode("hello")
        cached_embedder.encode("hello")  # Should hit cache
        assert cached_embedder.base.encode.call_count == 1

    def test_different_texts_both_computed(self, cached_embedder):
        cached_embedder.base.encode.side_effect = [
            np.ones((1, EMBEDDING_DIM), dtype=np.float32),
            np.zeros((1, EMBEDDING_DIM), dtype=np.float32),
        ]
        cached_embedder.encode("hello")
        cached_embedder.encode("world")
        assert cached_embedder.base.encode.call_count == 2

    def test_batch_encode_caches_individual(self, cached_embedder):
        """Batch encoding should cache each text individually."""
        batch_result = np.random.randn(2, EMBEDDING_DIM).astype(np.float32)
        cached_embedder.base.encode.return_value = batch_result
        cached_embedder.encode(["hello", "world"])

        # Second call with one cached text should only compute the new one
        new_result = np.random.randn(1, EMBEDDING_DIM).astype(np.float32)
        cached_embedder.base.encode.return_value = new_result
        cached_embedder.encode(["hello", "new_text"])

        # Second batch call: "hello" cached, only "new_text" computed
        second_call_texts = cached_embedder.base.encode.call_args[0][0]
        assert second_call_texts == ["new_text"]

    def test_empty_list_returns_empty(self, cached_embedder):
        result = cached_embedder.encode([])
        assert result.shape == (0, EMBEDDING_DIM)
        cached_embedder.base.encode.assert_not_called()

    def test_lru_eviction(self):
        base = MagicMock(spec=MLXEmbedder)
        base.encode.return_value = np.ones((1, EMBEDDING_DIM), dtype=np.float32)
        cached = CachedEmbedder(base, maxsize=2)

        cached.encode("a")
        cached.encode("b")
        cached.encode("c")  # Evicts "a"

        base.encode.reset_mock()
        base.encode.return_value = np.ones((1, EMBEDDING_DIM), dtype=np.float32)

        # "a" was evicted, should call base again
        cached.encode("a")
        assert base.encode.call_count == 1

    def test_proxy_properties(self, cached_embedder):
        assert cached_embedder.backend == "mlx"
        assert cached_embedder.embedding_dim == EMBEDDING_DIM
        assert cached_embedder.model_name == "bge-small"

    def test_computation_counter(self, cached_embedder):
        assert cached_embedder.embedding_computations == 0
        cached_embedder.encode("hello")
        assert cached_embedder.embedding_computations == 1
        cached_embedder.encode("hello")  # cache hit
        assert cached_embedder.embedding_computations == 1

    def test_thread_safety(self, cached_embedder):
        """Concurrent encode calls should not corrupt cache."""
        cached_embedder.base.encode.return_value = np.ones((1, EMBEDDING_DIM), dtype=np.float32)
        errors = []

        def worker(text_id):
            try:
                for _ in range(20):
                    cached_embedder.encode(f"text_{text_id}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors

    def test_unload_clears_cache(self, cached_embedder):
        cached_embedder.encode("hello")
        cached_embedder.unload()
        cached_embedder.base.unload.assert_called_once()
