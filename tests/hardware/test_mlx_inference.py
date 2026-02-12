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

    # Test semantic similarity
    emb1 = embedder.encode("king")
    emb2 = embedder.encode("queen")
    emb3 = embedder.encode("apple")

    # King and queen should be more similar than king and apple
    sim_king_queen = np.dot(emb1.flatten(), emb2.flatten())
    sim_king_apple = np.dot(emb1.flatten(), emb3.flatten())

    assert sim_king_queen > sim_king_apple, "Embeddings should capture semantics"


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
