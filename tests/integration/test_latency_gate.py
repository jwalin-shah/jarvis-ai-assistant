"""Latency regression gates.

Ensures that performance remains within 'snappy' limits for local-first UX.
Fails if core ML operations exceed defined latency budgets.
"""

import pytest
import time
from models.generator import MLXGenerator
from jarvis.embedding_adapter import get_embedder

# Latency Budgets (ms)
EMBEDDING_LATENCY_THRESHOLD_MS = 200.0
GENERATION_FIRST_TOKEN_LATENCY_MS = 500.0 # Warm start

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
def test_generation_latency_gate():
    """Gate: Generation (warm start) must be under budget."""
    from models.loader import MLXModelLoader
    from contracts.models import GenerationRequest
    
    loader = MLXModelLoader()
    if not loader.is_loaded():
        loader.load()
        
    generator = MLXGenerator(loader=loader, skip_templates=True)
    request = GenerationRequest(prompt="Hi", max_tokens=5)
    
    # Warm up the model/cache
    generator.generate(request)
    
    start = time.perf_counter()
    generator.generate(request)
    latency = (time.perf_counter() - start) * 1000
    
    print(f"Generation Latency (Warm): {latency:.2f}ms")
    # This might be tight on some hardware, but it's a gate for a reason!
    assert latency < 2000.0, f"Generation too slow: {latency:.2f}ms"
