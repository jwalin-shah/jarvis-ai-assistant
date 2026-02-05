#!/usr/bin/env python3
"""Benchmark FAISS index types with REAL message embeddings.

Embeds actual messages from chat.db and tests index quality.

Usage:
    uv run python scripts/benchmark_faiss_real.py --limit 10000
    uv run python scripts/benchmark_faiss_real.py --limit 50000
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_index_size_mb(index) -> float:
    """Get FAISS index size in MB by serializing it."""
    import faiss

    writer = faiss.VectorIOWriter()
    faiss.write_index(index, writer)
    return len(faiss.vector_to_array(writer.data)) / (1024 * 1024)


def get_real_embeddings(limit: int) -> tuple[np.ndarray, list[str]]:
    """Get embeddings for real messages from chat.db.

    Returns:
        Tuple of (embeddings array, list of message texts)
    """
    from integrations.imessage import ChatDBReader
    from jarvis.embedding_adapter import get_embedder

    print(f"Loading up to {limit:,} messages from chat.db...")

    with ChatDBReader() as reader:
        # Get ALL conversations
        conversations = reader.get_conversations(limit=500)
        print(f"Found {len(conversations)} conversations")

        all_messages = []
        for i, conv in enumerate(conversations):
            if len(all_messages) >= limit:
                break
            msgs = reader.get_messages(conv.chat_id, limit=min(5000, limit - len(all_messages)))
            all_messages.extend(msgs)
            if i % 20 == 0:
                n_convos = len(conversations)
                n_msgs = len(all_messages)
                print(f"  Progress: {i + 1}/{n_convos} convos, {n_msgs:,} messages...")

    # Filter to messages with meaningful text
    texts = []
    for msg in all_messages:
        if msg.text and len(msg.text.strip()) >= 5:
            texts.append(msg.text.strip())
            if len(texts) >= limit:
                break

    print(f"Found {len(texts):,} messages with text content")

    if len(texts) < 100:
        print("ERROR: Not enough messages to benchmark. Need at least 100.")
        sys.exit(1)

    # Embed messages
    print(f"Embedding {len(texts):,} messages with BGE-small...")
    embedder = get_embedder()

    batch_size = 256
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = embedder.encode(batch, normalize=True)
        all_embeddings.append(embeddings)
        if (i // batch_size) % 10 == 0:
            print(f"  Embedded {min(i + batch_size, len(texts)):,}/{len(texts):,}...")

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"Embedding complete: shape {embeddings.shape}")

    return embeddings, texts


def benchmark_index(
    name: str,
    index,
    embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10,
) -> dict:
    """Benchmark a FAISS index."""

    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {name}")
    print(f"{'=' * 60}")

    build_start = time.perf_counter()

    if hasattr(index, "train") and not index.is_trained:
        print("  Training index...")
        train_size = min(len(embeddings), 50000)
        index.train(embeddings[:train_size])

    print("  Adding vectors...")
    index.add(embeddings)

    build_time = time.perf_counter() - build_start
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

    # Calculate recall
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
        "search_time_ms": round(search_time / n_queries * 1000, 3),
        "recall_at_k": round(recall * 100, 1),
        "queries_per_sec": round(n_queries / search_time, 1),
    }

    print(f"  Build time:     {results['build_time_sec']:.2f}s")
    print(f"  Index size:     {results['memory_mb']:.1f} MB")
    print(f"  Search time:    {results['search_time_ms']:.3f}ms per query")
    print(f"  Recall@{k}:      {results['recall_at_k']:.1f}%")

    return results


def main():
    import faiss

    parser = argparse.ArgumentParser(description="Benchmark FAISS with real embeddings")
    parser.add_argument("--limit", type=int, default=10000, help="Max messages to embed")
    args = parser.parse_args()

    print("=" * 60)
    print("FAISS Benchmark with REAL Message Embeddings")
    print("=" * 60)

    # Get real embeddings
    embeddings, texts = get_real_embeddings(args.limit)

    n_vectors = len(embeddings)
    dim = embeddings.shape[1]
    k = 10
    n_queries = min(100, n_vectors // 10)

    # Use random subset as queries
    query_indices = np.random.choice(n_vectors, n_queries, replace=False)
    queries = embeddings[query_indices]

    # Scale clusters based on data size
    n_clusters = min(1024, int(np.sqrt(n_vectors) * 4))

    print(f"\nVectors:    {n_vectors:,}")
    print(f"Dimensions: {dim}")
    print(f"Queries:    {n_queries}")
    print(f"Clusters:   {n_clusters}")

    # Compute ground truth with brute force
    print("\nComputing ground truth...")
    flat_index = faiss.IndexFlatIP(dim)
    flat_index.add(embeddings)
    _, ground_truth = flat_index.search(queries, k)

    results = []

    # 1. IndexFlatIP (brute force baseline)
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

    # 2. IndexIVFFlat
    quantizer = faiss.IndexFlatIP(dim)
    index_ivf = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
    index_ivf.nprobe = max(16, n_clusters // 8)  # Search ~12.5% of clusters
    results.append(
        benchmark_index(
            f"IndexIVFFlat (nprobe={index_ivf.nprobe})",
            index_ivf,
            embeddings.copy(),
            queries,
            ground_truth,
            k,
        )
    )
    del index_ivf, quantizer
    gc.collect()

    # All compression levels to test
    # For dim=384, valid subquantizers are divisors: 384, 192, 128, 96, 64, 48, 32, 24
    # Compression = 1536 bytes (raw) / subquantizers bytes
    compression_configs = [
        (384, "4x"),  # 384 bytes/vec, highest quality PQ
        (192, "8x"),  # 192 bytes/vec
        (128, "12x"),  # 128 bytes/vec
        (96, "16x"),  # 96 bytes/vec
        (64, "24x"),  # 64 bytes/vec
        (48, "32x"),  # 48 bytes/vec
    ]

    nprobe = max(16, n_clusters // 8)

    for n_subq, compression_label in compression_configs:
        quantizer = faiss.IndexFlatIP(dim)
        index_pq = faiss.IndexIVFPQ(
            quantizer, dim, n_clusters, n_subq, 8, faiss.METRIC_INNER_PRODUCT
        )
        index_pq.nprobe = nprobe
        results.append(
            benchmark_index(
                f"IVFPQ ({n_subq}x8bit, {compression_label})",
                index_pq,
                embeddings.copy(),
                queries,
                ground_truth,
                k,
            )
        )
        del index_pq, quantizer
        gc.collect()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Index':<40} {'Size':>10} {'Search':>12} {'Recall@10':>10}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['name']:<40} {r['memory_mb']:>8.1f}MB "
            f"{r['search_time_ms']:>10.3f}ms {r['recall_at_k']:>9.1f}%"
        )

    # Compression stats
    flat_size = results[0]["memory_mb"]
    print("\n" + "=" * 60)
    print("COMPRESSION ANALYSIS")
    print("=" * 60)
    for r in results[1:]:
        compression = flat_size / r["memory_mb"] if r["memory_mb"] > 0 else 0
        savings = flat_size - r["memory_mb"]
        print(f"{r['name']:<40}")
        print(f"  Compression: {compression:.1f}x ({savings:.1f}MB saved)")
        print(f"  Recall:      {r['recall_at_k']:.1f}%")
        print()


if __name__ == "__main__":
    main()
