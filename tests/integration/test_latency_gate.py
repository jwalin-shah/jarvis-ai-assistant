"""Latency regression gates.

Ensures that performance remains within 'snappy' limits for local-first UX.
Fails if core ML operations exceed defined latency budgets.
"""

import time

import psutil
import pytest

from jarvis.embedding_adapter import get_embedder

# Latency Budgets (ms)
EMBEDDING_LATENCY_THRESHOLD_MS = 200.0

# Skip on low-memory machines (< 8 GB available)
_MIN_MEMORY_GB = 8
_available_gb = psutil.virtual_memory().total / (1024**3)
_low_memory = _available_gb < _MIN_MEMORY_GB


@pytest.mark.skipif(
    _low_memory, reason=f"Low memory ({_available_gb:.1f} GB < {_MIN_MEMORY_GB} GB)"
)
@pytest.mark.skipif(not get_embedder().is_available(), reason="Embedder not available")
def test_embedding_latency_gate():
    """Gate: Single text embedding must be under budget."""
    embedder = get_embedder()
    text = "This is a sample sentence for latency benchmarking."

    # Warm up
    embedder.encode(text)

    start = time.perf_counter()
    embedder.encode(text)
    latency = (time.perf_counter() - start) * 1000

    print(f"Embedding Latency: {latency:.2f}ms")
    assert latency < EMBEDDING_LATENCY_THRESHOLD_MS, f"Embedding too slow: {latency:.2f}ms"


@pytest.mark.real_model
@pytest.mark.skipif(
    _low_memory, reason=f"Low memory ({_available_gb:.1f} GB < {_MIN_MEMORY_GB} GB)"
)
def test_generation_latency_gate():
    """Gate: Generation (warm start) must be under budget."""
    from jarvis.contracts.models import GenerationRequest
    from models.generator import MLXGenerator
    from models.loader import MLXModelLoader

    loader = MLXModelLoader()
    try:
        if not loader.is_loaded():
            loader.load()
    except Exception as e:
        if "Insufficient memory" in str(e):
            pytest.skip(f"Skipping latency gate due to insufficient memory: {e}")
        raise e

    generator = MLXGenerator(loader=loader, skip_templates=True)
    request = GenerationRequest(prompt="Hi", max_tokens=5)

    # Warm up the model/cache
    generator.generate(request)

    start = time.perf_counter()
    generator.generate(request)
    latency = (time.perf_counter() - start) * 1000

    print(f"Generation Latency (Warm): {latency:.2f}ms")
    assert latency < 2000.0, f"Generation too slow: {latency:.2f}ms"
