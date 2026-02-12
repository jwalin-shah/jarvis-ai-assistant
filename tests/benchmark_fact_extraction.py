"""Real MLX performance benchmarks for fact extraction pipeline.

Runs against actual MLX models on Apple Silicon. Each benchmark prints
timing results with flush=True and logs memory via jarvis/utils/memory.py.

Run:
    uv run python -m pytest tests/benchmark_fact_extraction.py -v -s
    # Or just the benchmarks:
    uv run python -m pytest tests/benchmark_fact_extraction.py -v -s -k benchmark

Requires: MLX, Apple Silicon, ~384MB BERT model weights.
"""

from __future__ import annotations

import gc
import time

import pytest

# ---------------------------------------------------------------------------
# Skip entire module if MLX/BERT not available
# ---------------------------------------------------------------------------


def _can_load_embedder() -> bool:
    try:
        from jarvis.embedding_adapter import get_embedder

        embedder = get_embedder()
        result = embedder.encode(["test"])
        return result.shape[1] == 384
    except Exception:
        return False


_embedder_available = None


def embedder_available() -> bool:
    global _embedder_available
    if _embedder_available is None:
        _embedder_available = _can_load_embedder()
    return _embedder_available


requires_mlx = pytest.mark.skipif(
    "not embedder_available()",
    reason="MLX BERT embedder not available",
)


def _get_embedder():
    from jarvis.embedding_adapter import get_embedder

    return get_embedder()


def _get_memory_mb() -> float:
    """Get current process RSS in MB."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _log_memory(label: str) -> float:
    mb = _get_memory_mb()
    print(f"  [MEM] {label}: {mb:.1f} MB RSS", flush=True)
    return mb


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@requires_mlx
class TestBERTBenchmarks:
    """BERT embedding performance benchmarks."""

    def test_benchmark_bert_single_encode(self):
        """Single text encode: target <50ms."""
        embedder = _get_embedder()

        # Warmup
        embedder.encode(["warmup text"])

        start = time.perf_counter()
        result = embedder.encode(["I live in Austin and work at Google"])
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"\n  BERT single encode: {elapsed_ms:.1f}ms", flush=True)
        assert result.shape == (1, 384)
        assert elapsed_ms < 200, f"Too slow: {elapsed_ms:.1f}ms (target <50ms, hard limit 200ms)"

    def test_benchmark_bert_batch_10(self):
        """Batch of 10 texts: target <150ms (15ms/text)."""
        embedder = _get_embedder()
        texts = [f"Test sentence number {i} with some context" for i in range(10)]

        # Warmup
        embedder.encode(["warmup"])

        start = time.perf_counter()
        result = embedder.encode(texts)
        elapsed_ms = (time.perf_counter() - start) * 1000
        per_text = elapsed_ms / 10

        print(f"\n  BERT batch 10: {elapsed_ms:.1f}ms total, {per_text:.1f}ms/text", flush=True)
        assert result.shape == (10, 384)
        assert elapsed_ms < 500, f"Too slow: {elapsed_ms:.1f}ms (target <150ms, hard limit 500ms)"

    def test_benchmark_bert_batch_100(self):
        """Batch of 100 texts: target <800ms (8ms/text)."""
        embedder = _get_embedder()
        texts = [f"Message number {i}: I like going to the park with friends" for i in range(100)]

        # Warmup
        embedder.encode(["warmup"])

        mem_before = _log_memory("before batch-100")
        start = time.perf_counter()
        result = embedder.encode(texts)
        elapsed_ms = (time.perf_counter() - start) * 1000
        mem_after = _log_memory("after batch-100")
        per_text = elapsed_ms / 100

        print(
            f"\n  BERT batch 100: {elapsed_ms:.1f}ms total, {per_text:.1f}ms/text, "
            f"mem delta: {mem_after - mem_before:.1f} MB",
            flush=True,
        )
        assert result.shape == (100, 384)
        assert elapsed_ms < 5000, f"Too slow: {elapsed_ms:.1f}ms (target <800ms, hard limit 5000ms)"

    def test_benchmark_bert_model_load_cold(self):
        """Cold model load: target <2000ms.

        Note: This reloads the embedder which may or may not trigger actual model reload
        depending on singleton caching.
        """
        start = time.perf_counter()
        embedder = _get_embedder()
        # Force a single encode to ensure model is fully loaded
        embedder.encode(["model load test"])
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"\n  BERT model load (warm cache): {elapsed_ms:.1f}ms", flush=True)
        # Warm cache should be very fast; cold loads can take 2-5s
        assert elapsed_ms < 10000, f"Model load too slow: {elapsed_ms:.1f}ms"


@requires_mlx
class TestFactExtractionBenchmarks:
    """Fact extraction pipeline benchmarks."""

    def test_benchmark_fact_extraction_single(self):
        """Single message extraction: target <250ms."""
        from jarvis.contacts.fact_extractor import FactExtractor

        ext = FactExtractor()

        # Warmup
        ext.extract_facts([{"text": "warmup message for the extractor"}], "c1")

        start = time.perf_counter()
        facts = ext.extract_facts(
            [{"text": "I live in Austin and my sister Sarah works at Google"}], "c1"
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"\n  Fact extraction single: {elapsed_ms:.1f}ms, {len(facts)} facts", flush=True)
        assert elapsed_ms < 500, f"Too slow: {elapsed_ms:.1f}ms (target <250ms)"

    def test_benchmark_message_gate_predict(self):
        """MessageGate prediction: target <5ms (no model = instant)."""
        from jarvis.contacts.fact_filter import MessageGate

        gate = MessageGate(model_path="/nonexistent")  # No model = fast fallback

        start = time.perf_counter()
        for _ in range(1000):
            gate.predict_score("I love sushi and live in Austin")
        elapsed_ms = (time.perf_counter() - start) * 1000
        per_call = elapsed_ms / 1000

        print(
            f"\n  MessageGate predict (no model): {elapsed_ms:.1f}ms / 1000 calls, "
            f"{per_call:.3f}ms/call",
            flush=True,
        )
        assert per_call < 5, f"Too slow: {per_call:.3f}ms/call (target <5ms)"


@requires_mlx
class TestMemoryBenchmarks:
    """Memory pressure tracking during batch operations."""

    def test_benchmark_memory_pressure_batch(self):
        """Monitor memory during batch encode, peak RSS should stay <3GB."""
        embedder = _get_embedder()

        # Baseline
        gc.collect()
        mem_baseline = _log_memory("baseline")

        # Batch encode 200 texts
        texts = [f"Message {i}: various content about life, work, and food" for i in range(200)]
        start = time.perf_counter()
        result = embedder.encode(texts)
        elapsed_ms = (time.perf_counter() - start) * 1000

        mem_peak = _log_memory("after batch-200")
        mem_delta = mem_peak - mem_baseline

        print(
            f"\n  Batch 200: {elapsed_ms:.1f}ms, mem delta: {mem_delta:.1f} MB, "
            f"peak RSS: {mem_peak:.1f} MB",
            flush=True,
        )
        assert result.shape == (200, 384)
        assert mem_peak < 3000, f"Memory too high: {mem_peak:.1f} MB (target <3000 MB)"

    def test_benchmark_memory_snapshot(self):
        """Verify memory tracking utility works."""
        from jarvis.utils.memory import get_memory_info

        info = get_memory_info()
        print(
            f"\n  Memory info: RSS={info.rss_mb:.1f}MB, "
            f"VMS={info.vms_mb:.1f}MB, "
            f"swap={info.swap_used_mb:.1f}MB",
            flush=True,
        )
        assert info.rss_mb > 0
        assert info.vms_mb > 0
