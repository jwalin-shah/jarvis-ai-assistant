#!/usr/bin/env python3
"""Cluster response types to discover natural categories.

Uses UMAP + HDBSCAN pipeline (same approach as BERTopic):
1. Embed responses (we already have bge-small embeddings)
2. UMAP dimensionality reduction (384 → 5 dims)
3. HDBSCAN clustering (finds natural clusters + outliers)

Uses training pairs only (respects holdout split).

Usage:
    uv run python -m scripts.cluster_response_types
    uv run python -m scripts.cluster_response_types --min-cluster-size 100
    uv run python -m scripts.cluster_response_types --compare-kmeans
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Output directory
OUTPUT_DIR = Path.home() / ".jarvis" / "response_clusters"


def batch_encode(texts: list[str], batch_size: int = 500) -> np.ndarray:
    """Encode texts in batches with progress logging."""
    from jarvis.embedding_adapter import get_embedder

    embedder = get_embedder()
    all_embeddings = []

    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        embeddings = embedder.encode(batch, normalize=True)
        all_embeddings.append(embeddings)

        progress = min(i + batch_size, total)
        if progress % 5000 == 0 or progress == total:
            logger.info(f"  Encoded {progress}/{total} ({100*progress/total:.1f}%)")

    return np.vstack(all_embeddings).astype(np.float32)


def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 5,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    metric: str = "cosine",
) -> np.ndarray:
    """Reduce dimensionality with UMAP."""
    from umap import UMAP

    logger.info(f"Reducing {embeddings.shape[1]} dims → {n_components} dims with UMAP...")

    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
        verbose=True,
    )

    start = time.time()
    reduced = reducer.fit_transform(embeddings)
    elapsed = time.time() - start

    logger.info(f"  UMAP completed in {elapsed:.1f}s")

    return reduced


def cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 100,
    min_samples: int = 10,
) -> tuple[np.ndarray, object]:
    """Cluster with HDBSCAN (density-based, finds natural clusters)."""
    from hdbscan import HDBSCAN

    logger.info(f"Clustering with HDBSCAN (min_cluster_size={min_cluster_size})...")

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",  # Excess of Mass
    )

    start = time.time()
    labels = clusterer.fit_predict(embeddings)
    elapsed = time.time() - start

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    logger.info(f"  HDBSCAN completed in {elapsed:.1f}s")
    logger.info(
        f"  Found {n_clusters} clusters + {n_noise} noise ({100*n_noise/len(labels):.1f}%)"
    )

    return labels, clusterer


def cluster_kmeans(
    embeddings: np.ndarray,
    n_clusters: int = 15,
) -> tuple[np.ndarray, object]:
    """Cluster with K-means for comparison."""
    from sklearn.cluster import MiniBatchKMeans

    logger.info(f"Clustering with K-means (k={n_clusters}) for comparison...")

    clusterer = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=1000,
        n_init=10,
    )

    start = time.time()
    labels = clusterer.fit_predict(embeddings)
    elapsed = time.time() - start

    logger.info(f"  K-means completed in {elapsed:.1f}s")

    return labels, clusterer


def analyze_clusters(
    responses: list[str],
    labels: np.ndarray,
    n_samples: int = 15,
) -> dict[int, dict]:
    """Analyze each cluster by sampling responses."""
    cluster_info = {}

    unique_labels = sorted(set(labels))

    for cluster_id in unique_labels:
        # Get responses in this cluster
        mask = labels == cluster_id
        cluster_responses = [r for r, m in zip(responses, mask) if m]

        if not cluster_responses:
            continue

        # Sample responses
        sample_size = min(n_samples, len(cluster_responses))
        np.random.seed(42 + cluster_id)
        sample_indices = np.random.choice(len(cluster_responses), sample_size, replace=False)
        samples = [cluster_responses[i] for i in sample_indices]

        # Compute stats
        lengths = [len(r.split()) for r in cluster_responses]

        # Find common starting words/phrases
        starts = Counter()
        for r in cluster_responses:
            words = r.lower().split()[:2]
            if words:
                starts[" ".join(words)] += 1

        # Find common patterns
        has_question = sum(1 for r in cluster_responses if "?" in r)
        has_emoji = sum(1 for r in cluster_responses if any(ord(c) > 127 for c in r))
        lol_patterns = ["lol", "haha", "lmao"]
        has_lol = sum(1 for r in cluster_responses if any(x in r.lower() for x in lol_patterns))

        cluster_info[cluster_id] = {
            "count": len(cluster_responses),
            "pct": 100 * len(cluster_responses) / len(responses),
            "avg_words": np.mean(lengths),
            "median_words": np.median(lengths),
            "common_starts": starts.most_common(10),
            "pct_questions": 100 * has_question / len(cluster_responses),
            "pct_emoji": 100 * has_emoji / len(cluster_responses),
            "pct_lol": 100 * has_lol / len(cluster_responses),
            "samples": samples,
        }

    return cluster_info


def suggest_cluster_names(cluster_info: dict[int, dict]) -> dict[int, str]:
    """Suggest names for clusters based on patterns."""
    names = {}

    for cluster_id, info in cluster_info.items():
        if cluster_id == -1:
            names[cluster_id] = "NOISE/OUTLIERS"
            continue

        samples = info["samples"]
        sample_text = " ".join(samples).lower()
        avg_words = info["avg_words"]
        pct_questions = info["pct_questions"]
        pct_lol = info["pct_lol"]

        # Use patterns to name
        if pct_questions > 40:
            names[cluster_id] = "QUESTION"
        elif pct_lol > 30:
            names[cluster_id] = "REACTION_FUNNY"
        elif any(s in sample_text for s in ["yeah", "yes", "sure", "down", "sounds good"]):
            if avg_words < 4:
                names[cluster_id] = "AGREE_SHORT"
            else:
                names[cluster_id] = "AGREE_ELABORATE"
        elif any(s in sample_text for s in ["can't", "cant", "no ", "nope", "sorry", "busy"]):
            names[cluster_id] = "DECLINE"
        elif any(s in sample_text for s in ["maybe", "let me check", "i'll see", "get back"]):
            names[cluster_id] = "DEFER/UNCERTAIN"
        elif "idk" in sample_text:
            names[cluster_id] = "DEFER/UNCERTAIN"
        elif any(s in sample_text for s in ["ok", "okay", "k ", "bet", "got it", "cool"]):
            names[cluster_id] = "ACKNOWLEDGE"
        elif any(s in sample_text for s in ["thanks", "thank you", "ty", "thx", "appreciate"]):
            names[cluster_id] = "THANKS"
        elif any(s in sample_text for s in ["nice", "congrats", "awesome", "great", "amazing"]):
            names[cluster_id] = "POSITIVE_REACT"
        elif any(s in sample_text for s in ["damn", "bruh", "bro", "dude", "omg", "wtf"]):
            names[cluster_id] = "EXCLAMATION"
        elif any(s in sample_text for s in ["see you", "later", "bye", "night", "gn", "gm"]):
            names[cluster_id] = "GREETING/FAREWELL"
        elif any(s in sample_text for s in ["i think", "i feel", "i was", "i'm"]):
            names[cluster_id] = "PERSONAL_STATEMENT"
        elif avg_words > 15:
            names[cluster_id] = "LONG_RESPONSE"
        else:
            names[cluster_id] = f"CLUSTER_{cluster_id}"

    return names


def print_cluster_report(
    cluster_info: dict[int, dict],
    cluster_names: dict[int, str],
    method: str = "HDBSCAN",
) -> None:
    """Print detailed cluster analysis."""
    print("\n" + "=" * 70)
    print(f"RESPONSE TYPE CLUSTERING REPORT ({method})")
    print("=" * 70)

    # Sort by size (noise last)
    sorted_clusters = sorted(
        cluster_info.items(),
        key=lambda x: (x[0] == -1, -x[1]["count"])
    )

    total = sum(info["count"] for info in cluster_info.values())

    print(f"\n{'ID':<4} {'Name':<20} {'Count':>8} {'%':>6} {'Words':>6} {'?%':>5} {'LOL%':>5}")
    print("-" * 60)

    for cluster_id, info in sorted_clusters:
        name = cluster_names.get(cluster_id, f"CLUSTER_{cluster_id}")
        print(f"{cluster_id:<4} {name:<20} {info['count']:>8} {info['pct']:>5.1f}% "
              f"{info['avg_words']:>6.1f} {info['pct_questions']:>4.0f}% {info['pct_lol']:>4.0f}%")

    print("-" * 60)
    print(f"{'':4} {'TOTAL':<20} {total:>8}")

    print("\n" + "=" * 70)
    print("CLUSTER SAMPLES (Top clusters)")
    print("=" * 70)

    # Show top clusters (skip noise if too big)
    shown = 0
    for cluster_id, info in sorted_clusters:
        if shown >= 10:
            break
        if cluster_id == -1 and info["pct"] > 20:
            print(f"\n--- NOISE ({info['count']} points, {info['pct']:.1f}%) - skipping ---")
            continue

        name = cluster_names.get(cluster_id, f"CLUSTER_{cluster_id}")
        print(f"\n--- {name} (Cluster {cluster_id}, n={info['count']}, {info['pct']:.1f}%) ---")
        print(
            f"Avg words: {info['avg_words']:.1f}, "
            f"Questions: {info['pct_questions']:.0f}%, LOL: {info['pct_lol']:.0f}%"
        )
        print("Samples:")
        for i, sample in enumerate(info["samples"][:8]):
            display = sample[:70] + "..." if len(sample) > 70 else sample
            print(f"  {i+1}. {display}")
        shown += 1


def main():
    parser = argparse.ArgumentParser(
        description="Cluster response types with UMAP + HDBSCAN"
    )
    parser.add_argument(
        "--min-cluster-size", type=int, default=100, help="HDBSCAN min cluster size"
    )
    parser.add_argument("--umap-dims", type=int, default=5, help="UMAP output dimensions")
    parser.add_argument("--batch-size", type=int, default=500, help="Encoding batch size")
    parser.add_argument(
        "--compare-kmeans", action="store_true", help="Also run K-means for comparison"
    )
    parser.add_argument("--kmeans-k", type=int, default=15, help="K for K-means comparison")
    parser.add_argument("--save", action="store_true", help="Save cluster model and labels")
    parser.add_argument("--output", type=str, help="Output JSON file for cluster info")
    args = parser.parse_args()

    from jarvis.db import get_db

    db = get_db()

    # Step 1: Load ALL training pairs
    logger.info("Loading training pairs...")
    training_pairs = db.get_training_pairs()  # No limit now

    if not training_pairs:
        logger.error("No training pairs found. Run build_training_index.py first.")
        return

    logger.info(f"Loaded {len(training_pairs)} training pairs")

    # Step 2: Extract responses
    responses = [p.response_text for p in training_pairs]
    logger.info(f"Extracted {len(responses)} responses")

    # Step 3: Load or encode responses
    embeddings_cache = OUTPUT_DIR / "response_embeddings.npy"

    if embeddings_cache.exists():
        logger.info(f"Loading cached embeddings from {embeddings_cache}...")
        embeddings = np.load(embeddings_cache)
        if len(embeddings) != len(responses):
            logger.warning(
                f"Cache size mismatch ({len(embeddings)} vs {len(responses)}), re-encoding"
            )
            embeddings = None
        else:
            logger.info(f"Loaded {len(embeddings)} cached embeddings")
    else:
        embeddings = None

    if embeddings is None:
        logger.info("Encoding responses...")
        start = time.time()
        embeddings = batch_encode(responses, batch_size=args.batch_size)
        encode_time = time.time() - start
        logger.info(f"Encoded in {encode_time:.1f}s ({len(responses)/encode_time:.0f} texts/sec)")

        # Save for future use
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_cache, embeddings)
        logger.info(f"Cached embeddings to {embeddings_cache}")

    # Step 4: UMAP dimensionality reduction
    reduced = reduce_dimensions(embeddings, n_components=args.umap_dims)

    # Step 5: HDBSCAN clustering
    labels_hdbscan, clusterer = cluster_hdbscan(reduced, min_cluster_size=args.min_cluster_size)

    # Step 6: Analyze and report
    logger.info("Analyzing HDBSCAN clusters...")
    cluster_info = analyze_clusters(responses, labels_hdbscan)
    cluster_names = suggest_cluster_names(cluster_info)
    print_cluster_report(cluster_info, cluster_names, method="HDBSCAN")

    # Step 7: Compare with K-means if requested
    if args.compare_kmeans:
        labels_kmeans, _ = cluster_kmeans(reduced, n_clusters=args.kmeans_k)
        kmeans_info = analyze_clusters(responses, labels_kmeans)
        kmeans_names = suggest_cluster_names(kmeans_info)
        print_cluster_report(kmeans_info, kmeans_names, method=f"K-means (k={args.kmeans_k})")

    # Step 8: Save if requested
    if args.save or args.output:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Save reduced embeddings
        reduced_path = OUTPUT_DIR / "reduced_embeddings.npy"
        np.save(reduced_path, reduced)
        logger.info(f"Saved reduced embeddings to {reduced_path}")

        # Save cluster info
        info_path = args.output or (OUTPUT_DIR / "cluster_info.json")

        serializable_info = {}
        for cid, info in cluster_info.items():
            serializable_info[str(cid)] = {
                "name": cluster_names.get(cid, f"CLUSTER_{cid}"),
                "count": int(info["count"]),
                "pct": float(info["pct"]),
                "avg_words": float(info["avg_words"]),
                "pct_questions": float(info["pct_questions"]),
                "pct_lol": float(info["pct_lol"]),
                "common_starts": [[s, c] for s, c in info["common_starts"]],
                "samples": info["samples"][:10],
            }

        with open(info_path, "w") as f:
            json.dump(serializable_info, f, indent=2)
        logger.info(f"Saved cluster info to {info_path}")

        # Save labels
        labels_path = OUTPUT_DIR / "hdbscan_labels.npy"
        np.save(labels_path, labels_hdbscan)
        logger.info(f"Saved HDBSCAN labels to {labels_path}")


if __name__ == "__main__":
    main()
