"""Tests for jarvis.search.vec_search module."""

from __future__ import annotations

import numpy as np

from jarvis.search.vec_search import VecSearcher, VecSearchResult


class TestBinarizeEmbedding:
    """Tests for VecSearcher._binarize_embedding()."""

    def test_positive_values_become_ones(self):
        """Positive values are packed as 1 bits."""
        emb = np.ones(384, dtype=np.float32)
        result = VecSearcher._binarize_embedding(emb)
        # All 1s -> every byte is 0xFF
        assert result == b"\xff" * 48

    def test_negative_values_become_zeros(self):
        """Negative/zero values are packed as 0 bits."""
        emb = np.full(384, -0.5, dtype=np.float32)
        result = VecSearcher._binarize_embedding(emb)
        assert result == b"\x00" * 48

    def test_zero_values_become_zeros(self):
        """Zero values are packed as 0 bits (> 0, not >= 0)."""
        emb = np.zeros(384, dtype=np.float32)
        result = VecSearcher._binarize_embedding(emb)
        assert result == b"\x00" * 48

    def test_output_length(self):
        """384 dims -> 48 bytes (384/8)."""
        emb = np.random.RandomState(42).randn(384).astype(np.float32)
        result = VecSearcher._binarize_embedding(emb)
        assert len(result) == 48

    def test_mixed_values(self):
        """Mixed positive/negative values pack correctly."""
        # First 8 dims: [+, -, +, -, +, -, +, -] -> 10101010 -> 0xAA
        emb = np.zeros(384, dtype=np.float32)
        for i in range(0, 8, 2):
            emb[i] = 1.0
        for i in range(1, 8, 2):
            emb[i] = -1.0
        result = VecSearcher._binarize_embedding(emb)
        assert result[0] == 0xAA


class TestQuantizeEmbedding:
    """Tests for VecSearcher._quantize_embedding()."""

    def test_normalized_range(self):
        """Normalized [-1, 1] embedding maps to int8 [-127, 127]."""
        emb = np.array([1.0, -1.0, 0.0, 0.5], dtype=np.float32)
        result = VecSearcher._quantize_embedding(None, emb)  # type: ignore[arg-type]
        arr = np.frombuffer(result, dtype=np.int8)
        assert arr[0] == 127
        assert arr[1] == -127
        assert arr[2] == 0
        assert arr[3] == 63  # 0.5 * 127 = 63.5 -> 63


class TestDistanceToSimilarity:
    """Tests for VecSearcher._distance_to_similarity()."""

    def test_zero_distance_is_perfect(self):
        """Distance 0 maps to similarity 1.0."""
        assert VecSearcher._distance_to_similarity(0.0) == 1.0

    def test_large_distance_clamps_to_zero(self):
        """Very large distance clamps to 0.0."""
        result = VecSearcher._distance_to_similarity(1000.0)
        assert result == 0.0

    def test_moderate_distance(self):
        """Moderate distance gives valid similarity in (0, 1)."""
        result = VecSearcher._distance_to_similarity(50.0)
        assert 0.0 < result < 1.0


class TestVecSearchResult:
    """Tests for VecSearchResult dataclass."""

    def test_basic_creation(self):
        """Can create result with all fields."""
        r = VecSearchResult(
            rowid=42,
            distance=10.0,
            score=0.95,
            context_text="hello",
            reply_text="hi there",
        )
        assert r.rowid == 42
        assert r.score == 0.95
        assert r.context_text == "hello"
        assert r.reply_text == "hi there"

    def test_default_none_fields(self):
        """Optional fields default to None."""
        r = VecSearchResult(rowid=1, distance=0.0, score=1.0)
        assert r.chat_id is None
        assert r.context_text is None
        assert r.reply_text is None
