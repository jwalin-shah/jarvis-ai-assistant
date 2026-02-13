"""Hardware tests for MLX model inference.

These tests require:
- macOS with Apple Silicon
- MLX installed and functional
- Model files downloaded

Run with: pytest tests/hardware/ -v
Skip in CI: pytest -m "not hardware"
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import hardware_required, slow_test


@pytest.mark.hardware
@hardware_required(
    min_ram_gb=8,
    requires_apple_silicon=True,
    requires_model="lfm-1.2b-soc-fused",
)
@slow_test(timeout=30)
def test_mlx_model_generation():
    """Test actual MLX model generation."""
    from models.loader import MLXModelLoader

    loader = MLXModelLoader()
    loader.load()

    try:
        result = loader.generate_sync(
            prompt="Say hello briefly:",
            max_tokens=10,
            temperature=0.1,
        )
        assert result.text
        assert result.tokens_generated > 0
    finally:
        loader.unload()


@pytest.mark.hardware
@hardware_required(min_ram_gb=8, requires_apple_silicon=True)
def test_mlx_embedding_quality():
    """Test embedding quality with real MLX inference."""
    from jarvis.embedding_adapter import get_embedder

    embedder = get_embedder()

    # Test semantic similarity using cosine similarity on full sentences
    # (single words like "king"/"queen" are too short for BERT to distinguish well)
    emb1 = embedder.encode("The weather is sunny and warm today").flatten()
    emb2 = embedder.encode("It's a beautiful day with clear skies").flatten()
    emb3 = embedder.encode("The stock market crashed yesterday").flatten()

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # Similar sentences should have higher cosine similarity than unrelated ones
    sim_related = cosine_sim(emb1, emb2)
    sim_unrelated = cosine_sim(emb1, emb3)

    assert sim_related > sim_unrelated, (
        f"Embeddings should capture semantics: related={sim_related:.4f} "
        f"vs unrelated={sim_unrelated:.4f}"
    )


@pytest.mark.hardware
@hardware_required(min_ram_gb=8, requires_apple_silicon=True)
@slow_test(timeout=10)
def test_generation_latency_budget():
    """Test that generation meets latency budget."""
    import time

    from contracts.models import GenerationRequest
    from models.loader import MLXModelLoader

    loader = MLXModelLoader()
    if not loader.is_loaded():
        loader.load()

    try:
        request = GenerationRequest(prompt="Hi", max_tokens=5)

        # Warm up
        loader.generate_sync(prompt=request.prompt, max_tokens=request.max_tokens)

        # Measure
        start = time.perf_counter()
        loader.generate_sync(prompt=request.prompt, max_tokens=request.max_tokens)
        latency = (time.perf_counter() - start) * 1000

        print(f"\nGeneration latency: {latency:.2f}ms")
        assert latency < 2000.0, f"Generation too slow: {latency:.2f}ms"
    finally:
        loader.unload()
