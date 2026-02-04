#!/usr/bin/env python3
"""FAISS Index Benchmark - Compare V1 vs V2 Performance.

Measures:
    - Insert throughput (vectors/second)
    - Query latency (ms/query)
    - Memory usage (MB)
    - Shard management overhead (v2 only)

Usage:
    uv run python benchmarks/index_benchmark.py
    uv run python benchmarks/index_benchmark.py --vectors 100000
    uv run python benchmarks/index_benchmark.py --output results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    name: str
    version: str
    num_vectors: int

    # Insert metrics
    insert_total_time_s: float
    insert_throughput_vps: float  # vectors per second
    insert_memory_mb: float

    # Query metrics
    query_latencies_ms: list[float]
    query_mean_ms: float
    query_p50_ms: float
    query_p95_ms: float
    query_p99_ms: float
    query_throughput_qps: float  # queries per second

    # Index metrics
    index_size_bytes: int
    num_shards: int = 1
    compression_ratio: float = 1.0

    # Additional info
    index_type: str = "unknown"
    timestamp: str = ""


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024  # Convert KB to MB on macOS


def create_mock_pairs(num_pairs: int, seed: int = 42) -> list[Any]:
    """Create mock Pair objects for benchmarking.

    Args:
        num_pairs: Number of pairs to create.
        seed: Random seed for reproducibility.

    Returns:
        List of mock Pair objects.
    """
    np.random.seed(seed)
    base_time = datetime.now()

    # Pre-generate random data for performance
    templates = [
        "What is {}?",
        "How do I {}?",
        "Can you help me with {}?",
        "I need to {}",
        "Please explain {}",
        "Tell me about {}",
        "Why does {} happen?",
        "When should I {}?",
        "Where can I find {}?",
        "Who can help with {}?",
    ]

    topics = [
        "python programming",
        "machine learning",
        "data science",
        "web development",
        "API design",
        "database optimization",
        "cloud computing",
        "DevOps",
        "testing",
        "debugging",
        "performance tuning",
        "security",
        "scalability",
        "architecture",
    ]

    pairs = []
    for i in range(num_pairs):
        pair = MagicMock()
        pair.id = i + 1
        template = templates[i % len(templates)]
        topic = topics[i % len(topics)]
        pair.trigger_text = template.format(f"{topic} {i}")
        pair.response_text = f"Here's information about {topic} (item {i})"
        pair.chat_id = f"chat_{i % 100}"
        pair.source_timestamp = base_time - timedelta(days=i % 365)
        pair.quality_score = 0.7 + (i % 3) * 0.1
        pairs.append(pair)

    return pairs


def create_mock_db(pairs: list[Any]) -> MagicMock:
    """Create a mock JarvisDB.

    Args:
        pairs: Pairs to include in mock.

    Returns:
        Mock database.
    """
    db = MagicMock()
    pairs_by_id = {p.id: p for p in pairs}
    db.get_pairs_by_ids.return_value = pairs_by_id
    db.get_all_pairs.return_value = pairs
    db.get_training_pairs.return_value = pairs
    db.get_active_index.return_value = None
    db.add_embeddings_bulk = MagicMock()
    db.clear_embeddings = MagicMock()
    db.add_index_version = MagicMock()
    return db


def create_mock_embedder(dimension: int = 384) -> MagicMock:
    """Create a mock embedder that generates deterministic embeddings.

    Args:
        dimension: Embedding dimension.

    Returns:
        Mock embedder.
    """
    embedder = MagicMock()

    def encode_func(texts: list[str], normalize: bool = True) -> np.ndarray:
        embeddings = np.array(
            [np.random.RandomState(hash(t) % (2**31)).randn(dimension) for t in texts],
            dtype=np.float32,
        )
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
        return embeddings

    embedder.encode = encode_func
    return embedder


def benchmark_v1_index(
    pairs: list[Any],
    mock_db: MagicMock,
    mock_embedder: MagicMock,
    temp_dir: Path,
    num_queries: int = 100,
) -> BenchmarkResult:
    """Benchmark V1 IncrementalTriggerIndex.

    Args:
        pairs: Pairs to index.
        mock_db: Mock database.
        mock_embedder: Mock embedder.
        temp_dir: Temporary directory.
        num_queries: Number of search queries to run.

    Returns:
        BenchmarkResult with metrics.
    """
    from unittest.mock import patch

    # Import V1 index
    from jarvis.index import (
        IncrementalIndexConfig,
        IncrementalTriggerIndex,
        reset_incremental_index,
    )

    reset_incremental_index()

    config = IncrementalIndexConfig(
        indexes_dir=temp_dir / "indexes" / "triggers",
        index_type="ivfpq_4x",
        min_vectors_for_compression=100,
        auto_save=True,
    )

    gc.collect()
    start_memory = get_memory_usage_mb()

    # Create index with mocked embedder
    with patch("jarvis.index.get_embedder", return_value=mock_embedder):
        with patch("jarvis.index.get_configured_model_name", return_value="benchmark"):
            index = IncrementalTriggerIndex(mock_db, config)

            # Benchmark inserts
            start_time = time.perf_counter()
            added = index.add_pairs(pairs)
            insert_time = time.perf_counter() - start_time

    insert_memory = get_memory_usage_mb() - start_memory

    # Get index size
    index_path = index._get_index_path()
    index_size = index_path.stat().st_size if index_path.exists() else 0

    # Generate queries
    np.random.seed(123)
    query_indices = np.random.choice(len(pairs), num_queries, replace=False)
    queries = [pairs[i].trigger_text for i in query_indices]

    # Benchmark queries
    query_latencies = []
    with patch("jarvis.index.get_embedder", return_value=mock_embedder):
        for query in queries:
            start_time = time.perf_counter()
            index.search(query, k=10, threshold=0.0)
            latency_ms = (time.perf_counter() - start_time) * 1000
            query_latencies.append(latency_ms)

    # Calculate statistics
    query_mean = statistics.mean(query_latencies)
    query_latencies_sorted = sorted(query_latencies)
    p50_idx = int(len(query_latencies) * 0.5)
    p95_idx = int(len(query_latencies) * 0.95)
    p99_idx = int(len(query_latencies) * 0.99)

    reset_incremental_index()

    return BenchmarkResult(
        name="V1 IncrementalTriggerIndex",
        version="v1",
        num_vectors=added,
        insert_total_time_s=insert_time,
        insert_throughput_vps=added / insert_time if insert_time > 0 else 0,
        insert_memory_mb=insert_memory,
        query_latencies_ms=query_latencies,
        query_mean_ms=query_mean,
        query_p50_ms=query_latencies_sorted[p50_idx],
        query_p95_ms=query_latencies_sorted[p95_idx],
        query_p99_ms=query_latencies_sorted[min(p99_idx, len(query_latencies) - 1)],
        query_throughput_qps=num_queries / sum(query_latencies) * 1000,
        index_size_bytes=index_size,
        num_shards=1,
        index_type="ivfpq_4x",
        timestamp=datetime.now().isoformat(),
    )


def benchmark_v2_index(
    pairs: list[Any],
    mock_db: MagicMock,
    mock_embedder: MagicMock,
    temp_dir: Path,
    num_queries: int = 100,
) -> BenchmarkResult:
    """Benchmark V2 ShardedTriggerIndex.

    Args:
        pairs: Pairs to index.
        mock_db: Mock database.
        mock_embedder: Mock embedder.
        temp_dir: Temporary directory.
        num_queries: Number of search queries to run.

    Returns:
        BenchmarkResult with metrics.
    """
    from unittest.mock import patch

    # Import V2 index
    from jarvis.index_v2 import (
        ShardedIndexConfig,
        ShardedTriggerIndex,
        reset_sharded_index,
    )

    reset_sharded_index()

    config = ShardedIndexConfig(
        indexes_dir=temp_dir / "indexes_v2",
        batch_size=512,
        index_type="ivfpq_4x",
        min_vectors_for_compression=100,
        enable_journaling=True,
        enable_checksums=True,
        background_warming=False,
        max_parallel_searches=4,
    )

    gc.collect()
    start_memory = get_memory_usage_mb()

    # Create index with mocked embedder
    with patch("jarvis.index_v2.get_embedder", return_value=mock_embedder):
        with patch("jarvis.index_v2.get_configured_model_name", return_value="benchmark"):
            index = ShardedTriggerIndex(mock_db, config)

            # Benchmark inserts
            start_time = time.perf_counter()
            added = index.add_pairs(pairs)
            insert_time = time.perf_counter() - start_time

    insert_memory = get_memory_usage_mb() - start_memory

    # Get index stats
    stats = index.get_stats()
    index_size = stats.total_size_bytes
    num_shards = stats.total_shards

    # Generate queries
    np.random.seed(123)
    query_indices = np.random.choice(len(pairs), num_queries, replace=False)
    queries = [pairs[i].trigger_text for i in query_indices]

    # Benchmark queries
    query_latencies = []
    with patch("jarvis.index_v2.get_embedder", return_value=mock_embedder):
        for query in queries:
            start_time = time.perf_counter()
            index.search(query, k=10, threshold=0.0)
            latency_ms = (time.perf_counter() - start_time) * 1000
            query_latencies.append(latency_ms)

    # Calculate statistics
    query_mean = statistics.mean(query_latencies)
    query_latencies_sorted = sorted(query_latencies)
    p50_idx = int(len(query_latencies) * 0.5)
    p95_idx = int(len(query_latencies) * 0.95)
    p99_idx = int(len(query_latencies) * 0.99)

    index.close()
    reset_sharded_index()

    return BenchmarkResult(
        name="V2 ShardedTriggerIndex",
        version="v2",
        num_vectors=added,
        insert_total_time_s=insert_time,
        insert_throughput_vps=added / insert_time if insert_time > 0 else 0,
        insert_memory_mb=insert_memory,
        query_latencies_ms=query_latencies,
        query_mean_ms=query_mean,
        query_p50_ms=query_latencies_sorted[p50_idx],
        query_p95_ms=query_latencies_sorted[p95_idx],
        query_p99_ms=query_latencies_sorted[min(p99_idx, len(query_latencies) - 1)],
        query_throughput_qps=num_queries / sum(query_latencies) * 1000,
        index_size_bytes=index_size,
        num_shards=num_shards,
        index_type="ivfpq_4x",
        timestamp=datetime.now().isoformat(),
    )


def print_results(v1_result: BenchmarkResult, v2_result: BenchmarkResult) -> None:
    """Print benchmark comparison.

    Args:
        v1_result: V1 benchmark result.
        v2_result: V2 benchmark result.
    """
    print("\n" + "=" * 80)
    print("FAISS INDEX BENCHMARK RESULTS")
    print("=" * 80)

    print(f"\nDataset size: {v1_result.num_vectors:,} vectors")

    print("\n" + "-" * 40)
    print("INSERT PERFORMANCE")
    print("-" * 40)
    print(f"{'Metric':<30} {'V1':>15} {'V2':>15} {'Change':>15}")
    print("-" * 75)

    insert_speedup = v2_result.insert_throughput_vps / v1_result.insert_throughput_vps
    print(
        f"{'Total time (s)':<30} {v1_result.insert_total_time_s:>15.2f} {v2_result.insert_total_time_s:>15.2f} {(v2_result.insert_total_time_s / v1_result.insert_total_time_s - 1) * 100:>14.1f}%"
    )
    print(
        f"{'Throughput (vectors/s)':<30} {v1_result.insert_throughput_vps:>15.0f} {v2_result.insert_throughput_vps:>15.0f} {(insert_speedup - 1) * 100:>14.1f}%"
    )
    print(
        f"{'Memory (MB)':<30} {v1_result.insert_memory_mb:>15.1f} {v2_result.insert_memory_mb:>15.1f} {(v2_result.insert_memory_mb / v1_result.insert_memory_mb - 1) * 100:>14.1f}%"
    )
    print(
        f"{'Index size (MB)':<30} {v1_result.index_size_bytes / 1024 / 1024:>15.2f} {v2_result.index_size_bytes / 1024 / 1024:>15.2f} {(v2_result.index_size_bytes / v1_result.index_size_bytes - 1) * 100:>14.1f}%"
    )

    print("\n" + "-" * 40)
    print("QUERY PERFORMANCE")
    print("-" * 40)
    print(f"{'Metric':<30} {'V1':>15} {'V2':>15} {'Change':>15}")
    print("-" * 75)

    query_speedup = v1_result.query_mean_ms / v2_result.query_mean_ms
    print(
        f"{'Mean latency (ms)':<30} {v1_result.query_mean_ms:>15.2f} {v2_result.query_mean_ms:>15.2f} {(1 - query_speedup) * 100:>14.1f}%"
    )
    print(
        f"{'P50 latency (ms)':<30} {v1_result.query_p50_ms:>15.2f} {v2_result.query_p50_ms:>15.2f} {(v2_result.query_p50_ms / v1_result.query_p50_ms - 1) * 100:>14.1f}%"
    )
    print(
        f"{'P95 latency (ms)':<30} {v1_result.query_p95_ms:>15.2f} {v2_result.query_p95_ms:>15.2f} {(v2_result.query_p95_ms / v1_result.query_p95_ms - 1) * 100:>14.1f}%"
    )
    print(
        f"{'P99 latency (ms)':<30} {v1_result.query_p99_ms:>15.2f} {v2_result.query_p99_ms:>15.2f} {(v2_result.query_p99_ms / v1_result.query_p99_ms - 1) * 100:>14.1f}%"
    )
    print(
        f"{'Throughput (queries/s)':<30} {v1_result.query_throughput_qps:>15.0f} {v2_result.query_throughput_qps:>15.0f} {(v2_result.query_throughput_qps / v1_result.query_throughput_qps - 1) * 100:>14.1f}%"
    )

    print("\n" + "-" * 40)
    print("INDEX STRUCTURE")
    print("-" * 40)
    print(f"V1: Single index ({v1_result.index_type})")
    print(f"V2: {v2_result.num_shards} shards ({v2_result.index_type})")

    print("\n" + "-" * 40)
    print("SUMMARY")
    print("-" * 40)

    if insert_speedup > 1:
        print(f"Insert throughput: V2 is {insert_speedup:.1f}x faster")
    else:
        print(f"Insert throughput: V1 is {1 / insert_speedup:.1f}x faster")

    if query_speedup > 1:
        print(f"Query latency: V2 is {query_speedup:.1f}x faster")
    else:
        print(f"Query latency: V1 is {1 / query_speedup:.1f}x faster")

    print("=" * 80)


def run_benchmark(
    num_vectors: int = 10000,
    num_queries: int = 100,
    output_file: str | None = None,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """Run full benchmark comparison.

    Args:
        num_vectors: Number of vectors to index.
        num_queries: Number of search queries.
        output_file: Optional file to write results.

    Returns:
        Tuple of (v1_result, v2_result).
    """
    print(f"Creating {num_vectors:,} mock pairs...")
    pairs = create_mock_pairs(num_vectors)
    mock_db = create_mock_db(pairs)
    mock_embedder = create_mock_embedder()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        print("\nRunning V1 benchmark...")
        v1_result = benchmark_v1_index(pairs, mock_db, mock_embedder, temp_path, num_queries)

        # Clean up between benchmarks
        gc.collect()

        print("Running V2 benchmark...")
        v2_result = benchmark_v2_index(pairs, mock_db, mock_embedder, temp_path, num_queries)

    print_results(v1_result, v2_result)

    if output_file:
        results = {
            "v1": {k: v for k, v in asdict(v1_result).items() if k != "query_latencies_ms"},
            "v2": {k: v for k, v in asdict(v2_result).items() if k != "query_latencies_ms"},
            "config": {
                "num_vectors": num_vectors,
                "num_queries": num_queries,
            },
        }
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults written to: {output_file}")

    return v1_result, v2_result


def run_scaling_benchmark(
    vector_counts: list[int] | None = None,
    output_file: str | None = None,
) -> dict[str, Any]:
    """Run benchmark at multiple scales.

    Args:
        vector_counts: List of vector counts to test.
        output_file: Optional file to write results.

    Returns:
        Dict with all results.
    """
    if vector_counts is None:
        vector_counts = [1000, 5000, 10000, 25000, 50000]

    results = {
        "scaling": [],
        "timestamp": datetime.now().isoformat(),
    }

    for count in vector_counts:
        print(f"\n{'=' * 60}")
        print(f"Running benchmark with {count:,} vectors")
        print("=" * 60)

        v1_result, v2_result = run_benchmark(
            num_vectors=count,
            num_queries=min(100, count // 10),
        )

        results["scaling"].append(
            {
                "num_vectors": count,
                "v1_insert_throughput": v1_result.insert_throughput_vps,
                "v2_insert_throughput": v2_result.insert_throughput_vps,
                "v1_query_latency_ms": v1_result.query_mean_ms,
                "v2_query_latency_ms": v2_result.query_mean_ms,
                "v1_index_size_mb": v1_result.index_size_bytes / 1024 / 1024,
                "v2_index_size_mb": v2_result.index_size_bytes / 1024 / 1024,
                "v2_num_shards": v2_result.num_shards,
            }
        )

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nScaling results written to: {output_file}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark FAISS Index V1 vs V2 Performance")
    parser.add_argument(
        "--vectors",
        type=int,
        default=10000,
        help="Number of vectors to index (default: 10000)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=100,
        help="Number of search queries (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Run scaling benchmark at multiple sizes",
    )

    args = parser.parse_args()

    if args.scaling:
        run_scaling_benchmark(output_file=args.output)
    else:
        run_benchmark(
            num_vectors=args.vectors,
            num_queries=args.queries,
            output_file=args.output,
        )


if __name__ == "__main__":
    main()
