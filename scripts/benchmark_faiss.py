#!/usr/bin/env python3
"""Benchmark FAISS index types for semantic search.

Compares IndexFlatIP, IndexIVFFlat, and IndexIVFPQ on actual message embeddings.
Measures: memory, build time, search speed, and recall quality.

Usage:
    uv run python scripts/benchmark_faiss.py
"""

import gc
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_index_size_mb(index) -> float:
    """Get FAISS index size in MB by serializing it."""

    import faiss

    # Serialize index to bytes to measure actual size
    writer = faiss.VectorIOWriter()
    faiss.write_index(index, writer)
    return len(faiss.vector_to_array(writer.data)) / (1024 * 1024)


def generate_test_embeddings(n_vectors: int, dim: int = 384) -> np.ndarray:
    """Generate random test embeddings (normalized like real embeddings)."""
    print(f"Generating {n_vectors:,} test embeddings ({dim} dimensions)...")
    embeddings = np.random.randn(n_vectors, dim).astype(np.float32)
    # Normalize like real embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings


def benchmark_index(
    name: str,
    index,
    embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10,
) -> dict:
    """Benchmark a FAISS index.

    Args:
        name: Index name for display
        index: FAISS index object
        embeddings: All embeddings to index
        queries: Query embeddings
        ground_truth: True nearest neighbor IDs for recall calculation
        k: Number of neighbors to retrieve

    Returns:
        Dictionary with benchmark results
    """

    # Build index
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {name}")
    print(f"{'=' * 60}")

    build_start = time.perf_counter()

    if hasattr(index, "train") and not index.is_trained:
        print("  Training index...")
        # IVF indexes need training on a sample
        train_size = min(len(embeddings), 50000)
        index.train(embeddings[:train_size])

    print("  Adding vectors...")
    index.add(embeddings)

    build_time = time.perf_counter() - build_start

    # Measure actual index size by serialization
    mem_used = get_index_size_mb(index)

    # Search benchmark
    print("  Running search benchmark...")
    n_queries = len(queries)

    # Warm up
    index.search(queries[:10], k)

    # Timed search
    search_start = time.perf_counter()
    distances, indices = index.search(queries, k)
    search_time = time.perf_counter() - search_start

    # Calculate recall (how many true neighbors are found)
    recall_sum = 0
    for i in range(n_queries):
        true_neighbors = set(ground_truth[i])
        found_neighbors = set(indices[i])
        recall_sum += len(true_neighbors & found_neighbors) / k
    recall = recall_sum / n_queries

    results = {
        "name": name,
        "build_time_sec": round(build_time, 2),
        "memory_mb": round(mem_used, 1),
        "search_time_ms": round(search_time / n_queries * 1000, 2),
        "total_search_ms": round(search_time * 1000, 1),
        "recall_at_k": round(recall * 100, 1),
        "queries_per_sec": round(n_queries / search_time, 1),
    }

    print(f"  Build time:     {results['build_time_sec']:.2f}s")
    print(f"  Memory used:    {results['memory_mb']:.1f} MB")
    print(f"  Search time:    {results['search_time_ms']:.2f}ms per query")
    print(f"  Recall@{k}:      {results['recall_at_k']:.1f}%")
    print(f"  Throughput:     {results['queries_per_sec']:.1f} queries/sec")

    return results


def main():
    import faiss

    print("FAISS Index Benchmark")
    print("=" * 60)

    # Configuration
    n_vectors = 400_000  # Simulating 400K messages
    dim = 384  # BGE-small embedding dimension
    n_queries = 100
    k = 10  # Top-k neighbors

    # For IVF indexes
    n_clusters = 1024  # sqrt(n_vectors) is a good starting point

    # For PQ
    n_subquantizers = 48  # Must divide dim (384 / 48 = 8)
    n_bits = 8  # Bits per subquantizer

    print(f"Vectors:    {n_vectors:,}")
    print(f"Dimensions: {dim}")
    print(f"Queries:    {n_queries}")
    print(f"Top-k:      {k}")

    # Generate test data
    embeddings = generate_test_embeddings(n_vectors, dim)
    queries = generate_test_embeddings(n_queries, dim)

    # Compute ground truth with brute force
    print("\nComputing ground truth (brute force)...")
    flat_index = faiss.IndexFlatIP(dim)
    flat_index.add(embeddings)
    _, ground_truth = flat_index.search(queries, k)

    results = []

    # 1. IndexFlatIP (brute force baseline)
    print("\n" + "=" * 60)
    index_flat = faiss.IndexFlatIP(dim)
    results.append(
        benchmark_index(
            "IndexFlatIP (brute force)",
            index_flat,
            embeddings.copy(),
            queries,
            ground_truth,
            k,
        )
    )
    del index_flat
    gc.collect()

    # 2. IndexIVFFlat (higher nprobe for better recall)
    quantizer = faiss.IndexFlatIP(dim)
    index_ivf = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
    index_ivf.nprobe = 128  # Higher nprobe = better recall (search more clusters)
    results.append(
        benchmark_index(
            f"IndexIVFFlat (nprobe=128, {n_clusters} clusters)",
            index_ivf,
            embeddings.copy(),
            queries,
            ground_truth,
            k,
        )
    )
    del index_ivf, quantizer
    gc.collect()

    # 3. IndexIVFPQ
    quantizer = faiss.IndexFlatIP(dim)
    index_ivfpq = faiss.IndexIVFPQ(
        quantizer, dim, n_clusters, n_subquantizers, n_bits, faiss.METRIC_INNER_PRODUCT
    )
    index_ivfpq.nprobe = 128
    results.append(
        benchmark_index(
            f"IndexIVFPQ (nprobe=128, {n_subquantizers}x{n_bits}bit)",
            index_ivfpq,
            embeddings.copy(),
            queries,
            ground_truth,
            k,
        )
    )
    del index_ivfpq, quantizer
    gc.collect()

    # 4. IndexIVFPQ with more aggressive compression
    quantizer = faiss.IndexFlatIP(dim)
    index_ivfpq_small = faiss.IndexIVFPQ(
        quantizer,
        dim,
        n_clusters,
        24,
        n_bits,
        faiss.METRIC_INNER_PRODUCT,  # 24 subquantizers
    )
    index_ivfpq_small.nprobe = 128
    results.append(
        benchmark_index(
            f"IndexIVFPQ (nprobe=128, 24x{n_bits}bit, more compressed)",
            index_ivfpq_small,
            embeddings.copy(),
            queries,
            ground_truth,
            k,
        )
    )
    del index_ivfpq_small, quantizer
    gc.collect()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Index':<45} {'Memory':>10} {'Search':>10} {'Recall':>10}")
    print("-" * 75)
    for r in results:
        print(
            f"{r['name']:<45} {r['memory_mb']:>8.1f}MB {r['search_time_ms']:>8.2f}ms {r['recall_at_k']:>8.1f}%"
        )

    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    ivfpq = next(r for r in results if "IVFPQ" in r["name"] and "48x" in r["name"])
    flat = next(r for r in results if "Flat" in r["name"] and "IVF" not in r["name"])

    memory_saved = flat["memory_mb"] - ivfpq["memory_mb"]
    recall_lost = flat["recall_at_k"] - ivfpq["recall_at_k"]
    speedup = flat["search_time_ms"] / ivfpq["search_time_ms"]

    print(
        f"IndexIVFPQ saves {memory_saved:.0f}MB ({memory_saved / flat['memory_mb'] * 100:.0f}% reduction)"
    )
    print(
        f"Recall drops by {recall_lost:.1f}% (from {flat['recall_at_k']:.1f}% to {ivfpq['recall_at_k']:.1f}%)"
    )
    print(f"Search is {speedup:.1f}x faster")

    if ivfpq["recall_at_k"] >= 90:
        print("\n→ IVFPQ is recommended for your use case (good recall, major memory savings)")
    else:
        print("\n→ Consider IVFFlat if recall is critical, IVFPQ if memory matters more")


if __name__ == "__main__":
    main()
