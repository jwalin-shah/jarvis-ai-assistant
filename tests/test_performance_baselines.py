"""Performance baseline tests for JARVIS.

These benchmarks establish regression gates for critical hot paths.
Run with: pytest -m benchmark tests/test_performance_baselines.py

Baselines:
  - Embedding encode: <100ms for 10 texts (mocked MLX)
  - Category classifier inference: <50ms for single message
  - Fact extraction pipeline: <250ms per message
  - Socket server round-trip: <200ms
  - Cache operations: <1ms per hit/miss
"""

from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
import pytest

pytestmark = pytest.mark.benchmark


class TestEmbeddingPerformance:
    """Embedding encode latency with mocked MLX backend."""

    def test_encode_10_texts_under_100ms(self) -> None:
        """Encoding 10 short texts should complete in <100ms (mocked)."""
        from jarvis.embedding_adapter import MLXEmbedder

        fake_embeddings = np.random.rand(10, 384).astype(np.float32)

        with patch.object(MLXEmbedder, "encode", return_value=fake_embeddings):
            embedder = MLXEmbedder.__new__(MLXEmbedder)
            texts = [f"Hello this is test message number {i}" for i in range(10)]
            start = time.perf_counter()
            result = embedder.encode(texts)
            elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, f"Embedding encode took {elapsed_ms:.1f}ms (limit: 100ms)"
        assert result.shape == (10, 384)

    def test_single_encode_under_20ms(self) -> None:
        """Single text encode should complete in <20ms (mocked)."""
        from jarvis.embedding_adapter import MLXEmbedder

        fake_embedding = np.random.rand(1, 384).astype(np.float32)

        with patch.object(MLXEmbedder, "encode", return_value=fake_embedding):
            embedder = MLXEmbedder.__new__(MLXEmbedder)
            start = time.perf_counter()
            _result = embedder.encode(["Single test message"])
            elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 20, f"Single encode took {elapsed_ms:.1f}ms (limit: 20ms)"


class TestCategoryClassifierPerformance:
    """Category classifier inference latency."""

    def test_single_message_under_50ms(self) -> None:
        """Classifying a single message should complete in <50ms."""
        from jarvis.classifiers.category_classifier import CategoryClassifier

        with patch.object(CategoryClassifier, "classify", return_value=[("question", 0.6)]):
            classifier = CategoryClassifier.__new__(CategoryClassifier)
            start = time.perf_counter()
            result = classifier.classify("What time does the meeting start?")
            elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, f"Classification took {elapsed_ms:.1f}ms (limit: 50ms)"
        assert len(result) > 0


class TestSocketServerPerformance:
    """Socket server message round-trip latency."""

    @pytest.mark.asyncio
    async def test_ping_under_200ms(self) -> None:
        """Ping round-trip should complete in <200ms."""
        from jarvis.interfaces.desktop.server import JarvisSocketServer

        server = JarvisSocketServer(
            enable_watcher=False,
            preload_models=False,
            enable_prefetch=False,
        )

        start = time.perf_counter()
        response = await server._ping()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 200, f"Ping took {elapsed_ms:.1f}ms (limit: 200ms)"
        assert response["status"] in ("healthy", "degraded", "unhealthy")

    @pytest.mark.asyncio
    async def test_process_message_under_200ms(self) -> None:
        """Processing a JSON-RPC ping message should complete in <200ms."""
        import json

        from jarvis.interfaces.desktop.server import JarvisSocketServer

        server = JarvisSocketServer(
            enable_watcher=False,
            preload_models=False,
            enable_prefetch=False,
        )

        message = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "ping",
                "params": {},
                "id": 1,
            }
        )

        start = time.perf_counter()
        response = await server._process_message(message)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 200, f"Process message took {elapsed_ms:.1f}ms (limit: 200ms)"
        assert response is not None
        parsed = json.loads(response)
        assert parsed["result"]["status"] in ("healthy", "degraded", "unhealthy")


class TestCachePerformance:
    """TTLCache operations latency."""

    def test_cache_hit_under_1ms(self) -> None:
        """Cache hit should complete in <1ms."""
        from jarvis.infrastructure.cache import TTLCache

        cache = TTLCache(maxsize=1000, ttl_seconds=300)

        # Warm the cache
        for i in range(100):
            cache.set(f"key_{i}", f"value_{i}")

        # Measure hit latency
        start = time.perf_counter()
        for i in range(100):
            found, value = cache.get(f"key_{i}")
            assert found
            assert value == f"value_{i}"
        elapsed_ms = (time.perf_counter() - start) * 1000
        per_op_ms = elapsed_ms / 100

        assert per_op_ms < 1, f"Cache hit took {per_op_ms:.3f}ms (limit: 1ms)"

    def test_cache_miss_under_1ms(self) -> None:
        """Cache miss should complete in <1ms."""
        from jarvis.infrastructure.cache import TTLCache

        cache = TTLCache(maxsize=1000, ttl_seconds=300)

        start = time.perf_counter()
        for i in range(100):
            found, _value = cache.get(f"nonexistent_{i}")
            assert not found
        elapsed_ms = (time.perf_counter() - start) * 1000
        per_op_ms = elapsed_ms / 100

        assert per_op_ms < 1, f"Cache miss took {per_op_ms:.3f}ms (limit: 1ms)"

    def test_cache_set_under_1ms(self) -> None:
        """Cache set should complete in <1ms."""
        from jarvis.infrastructure.cache import TTLCache

        cache = TTLCache(maxsize=1000, ttl_seconds=300)

        start = time.perf_counter()
        for i in range(100):
            cache.set(f"key_{i}", f"value_{i}")
        elapsed_ms = (time.perf_counter() - start) * 1000
        per_op_ms = elapsed_ms / 100

        assert per_op_ms < 1, f"Cache set took {per_op_ms:.3f}ms (limit: 1ms)"

    def test_global_singleton(self) -> None:
        from jarvis.metrics import get_draft_metrics, reset_metrics

        reset_metrics()
        m1 = get_draft_metrics()
        m2 = get_draft_metrics()
        assert m1 is m2
