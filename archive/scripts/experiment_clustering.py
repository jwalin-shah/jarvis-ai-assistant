#!/usr/bin/env python3
"""Unsupervised clustering experiment to discover natural message categories.

This script:
1. Pulls ALL messages from iMessage DB directly
2. Separates incoming (from others) vs outgoing (from me)
3. Embeds with multiple encoder models (BGE, Arctic families)
4. Clusters using multiple algorithms to find natural groupings
5. Prints samples from each cluster for inspection

Clustering algorithms:
- K-Means: Fast, spherical clusters, specify k (MLX GPU-accelerated available)
- GMM: Soft assignments (probabilities), shows ambiguous messages
- Spectral: Finds non-spherical shapes, uses Nyström approx for large datasets
- HDBSCAN: Auto-detects k, handles noise/outliers (slowest)

The goal: discover what categories ACTUALLY exist in the data,
rather than forcing pre-defined labels.

Usage:
    # Quick comparison with K-Means (uses cached embeddings if available)
    uv run python -m scripts.experiment_clustering --k 5 10 15

    # Use direct mode (doesn't need embed server running):
    uv run python -m scripts.experiment_clustering --k 5 10 15 --direct

    # All algorithms except HDBSCAN
    uv run python -m scripts.experiment_clustering --all-algorithms --k 5 10 15

    # HDBSCAN only (slowest but auto-detects k)
    uv run python -m scripts.experiment_clustering --hdbscan

    # Quick model subset for faster iteration
    uv run python -m scripts.experiment_clustering --models-quick --k 5 10

    # Full benchmark: all models, all algorithms
    uv run python -m scripts.experiment_clustering --all-algorithms --k 5 10 15 20
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time

# Add scripts dir to path for direct embedder import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

# MLX is used by MinimalEmbedder for embeddings (not for K-means - sklearn is faster at this scale)
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

# HDBSCAN is optional - better clustering but slower
try:
    import hdbscan

    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    hdbscan = None

# UMAP is optional - helps with high-dim clustering
try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    umap = None

# BERTopic is optional - topic modeling with interpretable labels
try:
    from bertopic import BERTopic

    HAS_BERTOPIC = True
except ImportError:
    HAS_BERTOPIC = False
    BERTopic = None

from sklearn.metrics import silhouette_score

from jarvis.text_normalizer import normalize_for_task

# =============================================================================
# Constants
# =============================================================================

CHAT_DB_PATH = Path.home() / "Library/Messages/chat.db"
# Socket paths for production vs minimal server
SOCKET_PATH_PRODUCTION = "/tmp/jarvis-embed.sock"  # Production service (mlx-embedding-models)
SOCKET_PATH_MINIMAL = "/tmp/jarvis-embed-minimal.sock"  # Our minimal server

# Available embedding models for clustering experiments
MODELS = [
    # BGE family
    "bge-small",  # 12 layers, 384 dim, 193MB, 5363 msg/s
    "bge-base",  # 12 layers, 768 dim, 483MB, 2081 msg/s
    "bge-large",  # 24 layers, 1024 dim, 1346MB, 629 msg/s
    # Arctic family (Snowflake) - similar architecture to BGE
    "arctic-xs",  # 6 layers, 384 dim, 149MB, 7722 msg/s (fastest)
    "arctic-s",  # 12 layers, 384 dim, 193MB, 5398 msg/s
    "arctic-m",  # 12 layers, 768 dim, 482MB, 2056 msg/s
    "arctic-l",  # 24 layers, 1024 dim, 1345MB, 628 msg/s
]

# Quick subset for faster iteration
MODELS_QUICK = ["bge-small", "arctic-xs", "arctic-m"]

# Best performers based on silhouette scores (use these for algorithm comparison)
MODELS_BEST = ["arctic-m", "bge-small"]

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ClusterResult:
    """Result from clustering a set of messages."""

    model_name: str
    message_type: str  # "incoming" or "outgoing"
    k: int
    silhouette: float
    cluster_sizes: list[int]
    cluster_samples: dict[int, list[str]]  # cluster_id -> sample texts
    embedding_time: float
    cluster_time: float


# =============================================================================
# Text Processing
# =============================================================================


def extract_text_from_attributed_body(blob: bytes) -> str | None:
    """Extract plain text from attributedBody blob using the shared parser."""
    if not blob:
        return None
    try:
        from integrations.imessage.parser import parse_attributed_body

        return parse_attributed_body(blob)
    except Exception:
        # Keep the script resilient; normalization will drop artifacts anyway.
        return None


# =============================================================================
# Database Access
# =============================================================================


def get_all_messages(
    limit: int | None = None,
    min_length: int = 3,
    max_length: int = 500,
) -> tuple[list[str], list[str]]:
    """Pull all messages from iMessage DB.

    Args:
        limit: Maximum messages per type (incoming/outgoing). None for all.
        min_length: Minimum text length to include.
        max_length: Maximum text length to include.

    Returns:
        Tuple of (incoming_messages, outgoing_messages)
    """
    if not CHAT_DB_PATH.exists():
        raise FileNotFoundError(f"iMessage database not found: {CHAT_DB_PATH}")

    # Connect read-only
    conn = sqlite3.connect(f"file:{CHAT_DB_PATH}?mode=ro", uri=True)
    cursor = conn.cursor()

    # Query all messages
    query = """
        SELECT
            message.text,
            message.attributedBody,
            message.is_from_me
        FROM message
        WHERE (message.text IS NOT NULL AND message.text != '')
           OR message.attributedBody IS NOT NULL
        ORDER BY message.date DESC
    """

    if limit:
        query += f" LIMIT {limit * 3}"  # Get extra since we'll filter

    cursor.execute(query)

    incoming = []
    outgoing = []
    seen = set()  # Dedupe

    for text, attributed_body, is_from_me in cursor.fetchall():
        # Get text from either field
        if not text and attributed_body:
            text = extract_text_from_attributed_body(attributed_body)

        if not text:
            continue

        # Normalize
        text = normalize_for_task(text, "topic_modeling")
        if not text:
            continue

        # Length filter
        if len(text) < min_length or len(text) > max_length:
            continue

        # Dedupe
        if text in seen:
            continue
        seen.add(text)

        # Categorize
        if is_from_me:
            if limit is None or len(outgoing) < limit:
                outgoing.append(text)
        else:
            if limit is None or len(incoming) < limit:
                incoming.append(text)

        # Early exit if we have enough
        if limit and len(incoming) >= limit and len(outgoing) >= limit:
            break

    conn.close()
    return incoming, outgoing


# =============================================================================
# Embedding (via MLX socket server)
# =============================================================================


def embed_texts(
    texts: list[str],
    model: str,
    message_type: str,
    batch_size: int | None = None,
    cache_dir: Path | None = None,
    use_socket: bool = True,
    socket_path: str | None = None,
    force_embed: bool = False,
) -> np.ndarray:
    """Embed texts using socket server or direct calls. Always uses cache if available.

    Args:
        texts: Texts to embed.
        model: Model name (bge-small, arctic-m, etc.)
        message_type: "incoming" or "outgoing" - used for stable cache naming.
        batch_size: Batch size for requests. If None, auto-selects based on model size.
        cache_dir: Directory to cache embeddings. If None, keeps in RAM.
        use_socket: If True (default), use production socket server. If False, use direct calls.
        socket_path: Socket path to use. If None, uses SOCKET_PATH_PRODUCTION.
        force_embed: If True, re-embed even if cache exists (default False).

    Returns:
        Embeddings array of shape (n_texts, embedding_dim)
    """
    if not texts:
        return np.array([])

    # Stable cache file: model + message_type (no fragile hashing)
    cache_file = None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{model}_{message_type}.npy"

        if cache_file.exists() and not force_embed:
            cached = np.load(cache_file)
            print(f"      Loaded {len(cached):,} cached embeddings from {cache_file.name}")
            if len(cached) != len(texts):
                print(
                    f"      WARNING: Cache has {len(cached)} embeddings "
                    f"but {len(texts)} texts provided"
                )
                print(
                    "               Using cached embeddings anyway "
                    "(text count mismatch is OK for experiments)"
                )
            return cached

    if not force_embed and cache_file:
        print(f"      No cache found at {cache_file.name}, embedding {len(texts):,} texts...")

    start = time.time()

    # Auto-select batch size based on model
    if batch_size is None:
        if model in ("bge-large", "arctic-l"):
            batch_size = 256
        elif model in ("bge-base", "arctic-m"):
            batch_size = 512
        else:
            batch_size = 1024

    if use_socket:
        sock_path = socket_path or SOCKET_PATH_PRODUCTION
        embeddings = _embed_texts_socket(texts, model, batch_size, cache_dir, start, sock_path)
    else:
        embeddings = _embed_texts_direct(texts, model, batch_size, cache_dir, start)

    # Save to stable cache
    if cache_file:
        np.save(cache_file, embeddings)
        print(f"      Saved {len(embeddings):,} embeddings to {cache_file.name}")

    return embeddings


# Global embedder instance for direct calls (avoids reload per batch)
_embedder = None


def _get_embedder():
    """Get or create global embedder instance."""
    global _embedder
    if _embedder is None:
        from minimal_mlx_embed_server import MinimalEmbedder

        _embedder = MinimalEmbedder()
    return _embedder


def _embed_texts_direct(
    texts: list[str],
    model: str,
    batch_size: int,
    cache_dir: Path | None,
    start: float,
) -> np.ndarray:
    """Embed using direct Python calls (MinimalEmbedder).

    MinimalEmbedder now handles length-sorting internally for optimal performance.
    """
    embedder = _get_embedder()
    embedder.load_model(model)

    # MinimalEmbedder.encode() handles length-sorting internally
    embeddings = embedder.encode(texts, normalize=True, batch_size=batch_size)

    elapsed = time.time() - start
    rate = len(texts) / elapsed if elapsed > 0 else 0
    print(f"      {len(texts):,}/{len(texts):,} (100%) - {rate:,.0f}/s")

    # Optionally save to disk
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        emb_file = cache_dir / f"embeddings_{model}_{len(texts)}.npy"
        np.save(emb_file, embeddings)

    return embeddings


def _embed_texts_socket(
    texts: list[str],
    model: str,
    batch_size: int,
    cache_dir: Path | None,
    start: float,
    socket_path: str = SOCKET_PATH_PRODUCTION,
) -> np.ndarray:
    """Embed using production socket server (fast, uses mlx-embedding-models)."""
    import base64
    import socket as sock_module

    def send_request(sock, batch_texts):
        request = {
            "jsonrpc": "2.0",
            "method": "embed",
            "params": {"texts": batch_texts, "model": model, "normalize": True, "binary": True},
            "id": 1,
        }
        sock.sendall(json.dumps(request).encode() + b"\n")

        response = b""
        while b"\n" not in response:
            chunk = sock.recv(262144)
            if not chunk:
                raise ConnectionError("Server closed connection")
            response += chunk

        result = json.loads(response.decode())
        if "error" in result:
            raise RuntimeError(f"Embed error: {result['error']}")
        return result["result"]

    sock = sock_module.socket(sock_module.AF_UNIX, sock_module.SOCK_STREAM)
    sock.settimeout(300)
    sock.connect(socket_path)

    try:
        first_batch = texts[: min(batch_size, len(texts))]
        result = send_request(sock, first_batch)

        dim = result["dimension"]
        # Handle both binary (our minimal server) and JSON (production server) responses
        if "embeddings_b64" in result:
            emb_bytes = base64.b64decode(result["embeddings_b64"])
            first_emb = np.frombuffer(emb_bytes, dtype=np.float32).reshape(-1, dim)
        else:
            first_emb = np.array(result["embeddings"], dtype=np.float32)

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            emb_file = cache_dir / f"embeddings_{model}_{len(texts)}.npy"
            embeddings = np.memmap(emb_file, dtype=np.float32, mode="w+", shape=(len(texts), dim))
        else:
            embeddings = np.empty((len(texts), dim), dtype=np.float32)

        embeddings[: len(first_batch)] = first_emb
        processed = len(first_batch)

        for i in range(len(first_batch), len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = send_request(sock, batch)

            # Handle both binary and JSON responses
            if "embeddings_b64" in result:
                emb_bytes = base64.b64decode(result["embeddings_b64"])
                batch_emb = np.frombuffer(emb_bytes, dtype=np.float32).reshape(-1, dim)
            else:
                batch_emb = np.array(result["embeddings"], dtype=np.float32)
            embeddings[i : i + len(batch)] = batch_emb
            processed += len(batch)

            progress_interval = max(len(texts) // 10, batch_size)
            if (i + batch_size) % progress_interval < batch_size:
                elapsed = time.time() - start
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (len(texts) - processed) / rate if rate > 0 else 0
                pct = 100 * processed / len(texts)
                print(
                    f"      {processed:,}/{len(texts):,} ({pct:.0f}%) - "
                    f"{rate:,.0f}/s, ETA {eta:.0f}s"
                )

        if cache_dir:
            embeddings.flush()

    finally:
        sock.close()

    return embeddings


# =============================================================================
# Clustering
# =============================================================================


def cluster_kmeans(
    texts: list[str],
    embeddings: np.ndarray,
    k: int,
    n_samples: int = 15,
) -> tuple[float, list[int], dict[int, list[str]], np.ndarray]:
    """Cluster with K-Means.

    Args:
        texts: Original texts.
        embeddings: Embedding vectors.
        k: Number of clusters.
        n_samples: Samples per cluster.

    Returns:
        Tuple of (silhouette_score, cluster_sizes, cluster_samples, labels)
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    # Silhouette score (sample for speed if large)
    if len(embeddings) > 10000:
        sample_idx = np.random.choice(len(embeddings), 10000, replace=False)
        sil = silhouette_score(embeddings[sample_idx], labels[sample_idx])
    else:
        sil = silhouette_score(embeddings, labels)

    # Cluster sizes
    unique_labels = sorted(set(labels))
    sizes = [int((labels == i).sum()) for i in unique_labels]

    # Get samples closest to each centroid (most representative)
    samples = {}
    for cluster_id in unique_labels:
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_embeddings = embeddings[cluster_mask]

        if len(cluster_indices) == 0:
            samples[cluster_id] = []
            continue

        centroid = centroids[cluster_id]
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

        n = min(n_samples, len(cluster_indices))
        closest_idx = np.argsort(distances)[:n]
        samples[cluster_id] = [texts[cluster_indices[i]] for i in closest_idx]

    return sil, sizes, samples, labels


def cluster_hdbscan(
    texts: list[str],
    embeddings: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int = 10,
    n_samples: int = 15,
    use_umap: bool = True,
    umap_dims: int = 50,
) -> tuple[float, list[int], dict[int, list[str]], np.ndarray, int]:
    """Cluster with HDBSCAN (auto-detects number of clusters).

    Args:
        texts: Original texts.
        embeddings: Embedding vectors.
        min_cluster_size: Minimum cluster size for HDBSCAN.
        min_samples: Min samples for core point in HDBSCAN.
        n_samples: Samples to extract per cluster.
        use_umap: Whether to reduce dimensions with UMAP first.
        umap_dims: Target dimensions for UMAP.

    Returns:
        Tuple of (silhouette, sizes, samples, labels, n_clusters)
    """
    if not HAS_HDBSCAN:
        raise ImportError("hdbscan not installed. Run: uv pip install hdbscan")

    data = embeddings

    # Optional UMAP dimensionality reduction (helps HDBSCAN)
    if use_umap and HAS_UMAP and embeddings.shape[1] > umap_dims:
        print(f"      Reducing {embeddings.shape[1]}D -> {umap_dims}D with UMAP...")
        reducer = umap.UMAP(
            n_components=umap_dims,
            metric="cosine",
            random_state=42,
            n_neighbors=15,
            min_dist=0.0,
        )
        data = reducer.fit_transform(embeddings)

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",  # Excess of Mass - better for varying densities
        prediction_data=True,
    )
    labels = clusterer.fit_predict(data)

    # Count clusters (excluding noise label -1)
    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l >= 0])
    n_noise = int((labels == -1).sum())

    noise_pct = 100 * n_noise / len(labels)
    print(f"      Found {n_clusters} clusters, {n_noise} noise points ({noise_pct:.1f}%)")

    # Silhouette (exclude noise points)
    non_noise_mask = labels >= 0
    if non_noise_mask.sum() > 100 and n_clusters > 1:
        sample_size = min(10000, non_noise_mask.sum())
        non_noise_indices = np.where(non_noise_mask)[0]
        sample_idx = np.random.choice(non_noise_indices, sample_size, replace=False)
        sil = silhouette_score(data[sample_idx], labels[sample_idx])
    else:
        sil = 0.0

    # Cluster sizes (including noise as -1)
    sizes = [int((labels == l).sum()) for l in unique_labels]

    # Get representative samples per cluster
    samples = {}
    for cluster_id in unique_labels:
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            samples[cluster_id] = []
            continue

        # For HDBSCAN, use probabilities to find most confident members
        if hasattr(clusterer, "probabilities_") and cluster_id >= 0:
            cluster_probs = clusterer.probabilities_[cluster_mask]
            n = min(n_samples, len(cluster_indices))
            top_idx = np.argsort(cluster_probs)[-n:][::-1]
            samples[cluster_id] = [texts[cluster_indices[i]] for i in top_idx]
        else:
            # Fallback: random sample
            n = min(n_samples, len(cluster_indices))
            idx = np.random.choice(len(cluster_indices), n, replace=False)
            samples[cluster_id] = [texts[cluster_indices[i]] for i in idx]

    return sil, sizes, samples, labels, n_clusters


def cluster_gmm(
    texts: list[str],
    embeddings: np.ndarray,
    k: int,
    n_samples: int = 15,
) -> tuple[float, list[int], dict[int, list[str]], np.ndarray, np.ndarray]:
    """Cluster with Gaussian Mixture Model.

    GMM provides soft assignments (probabilities) showing how confident
    each assignment is, useful for finding ambiguous messages.

    Args:
        texts: Original texts.
        embeddings: Embedding vectors.
        k: Number of clusters (components).
        n_samples: Samples per cluster.

    Returns:
        Tuple of (silhouette, sizes, samples, labels, probabilities)
        where probabilities is (n_samples, k) soft assignments.
    """
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=42,
        n_init=3,
        max_iter=200,
    )
    labels = gmm.fit_predict(embeddings)
    probabilities = gmm.predict_proba(embeddings)

    # Silhouette score (sample for speed if large)
    if len(embeddings) > 10000:
        sample_idx = np.random.choice(len(embeddings), 10000, replace=False)
        sil = silhouette_score(embeddings[sample_idx], labels[sample_idx])
    else:
        sil = silhouette_score(embeddings, labels)

    # Cluster sizes
    unique_labels = sorted(set(labels))
    sizes = [int((labels == i).sum()) for i in unique_labels]

    # Get samples with highest confidence for each cluster
    samples = {}
    for cluster_id in unique_labels:
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            samples[cluster_id] = []
            continue

        # Use probability of assigned cluster as confidence
        cluster_probs = probabilities[cluster_indices, cluster_id]
        n = min(n_samples, len(cluster_indices))
        top_idx = np.argsort(cluster_probs)[-n:][::-1]
        samples[cluster_id] = [texts[cluster_indices[i]] for i in top_idx]

    return sil, sizes, samples, labels, probabilities


def cluster_spectral(
    texts: list[str],
    embeddings: np.ndarray,
    k: int,
    n_samples: int = 15,
    max_samples: int = 10000,
    n_neighbors: int = 10,
) -> tuple[float, list[int], dict[int, list[str]], np.ndarray]:
    """Cluster with Spectral Clustering.

    Uses k-nearest neighbors affinity for scalability (avoids O(n²) full matrix).
    For large datasets, uses Nyström approximation by sampling.

    Args:
        texts: Original texts.
        embeddings: Embedding vectors.
        k: Number of clusters.
        n_samples: Samples per cluster to return.
        max_samples: If dataset larger, sample for fitting then assign rest.
        n_neighbors: Number of neighbors for sparse affinity matrix.

    Returns:
        Tuple of (silhouette, sizes, samples, labels)
    """
    from sklearn.cluster import SpectralClustering
    from sklearn.neighbors import NearestNeighbors

    n_total = len(embeddings)
    use_sampling = n_total > max_samples

    if use_sampling:
        # Nyström-style approximation: fit on sample, assign rest via nearest neighbor
        print(f"      Large dataset ({n_total}), sampling {max_samples} for spectral...")
        sample_idx = np.random.choice(n_total, max_samples, replace=False)
        sample_embeddings = embeddings[sample_idx]
        [texts[i] for i in sample_idx]

        # Fit spectral on sample
        spectral = SpectralClustering(
            n_clusters=k,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            random_state=42,
            assign_labels="kmeans",
        )
        sample_labels = spectral.fit_predict(sample_embeddings)

        # Assign remaining points to nearest sampled point's cluster
        print(f"      Assigning remaining {n_total - max_samples} points...")
        nn = NearestNeighbors(n_neighbors=1, metric="cosine")
        nn.fit(sample_embeddings)
        _, indices = nn.kneighbors(embeddings)
        labels = sample_labels[indices.flatten()]

        # Use sample for silhouette (full would be expensive)
        sil = silhouette_score(sample_embeddings, sample_labels)
    else:
        # Fit directly with sparse affinity
        spectral = SpectralClustering(
            n_clusters=k,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            random_state=42,
            assign_labels="kmeans",
        )
        labels = spectral.fit_predict(embeddings)

        # Silhouette score
        if n_total > 10000:
            sample_idx = np.random.choice(n_total, 10000, replace=False)
            sil = silhouette_score(embeddings[sample_idx], labels[sample_idx])
        else:
            sil = silhouette_score(embeddings, labels)

    # Cluster sizes
    unique_labels = sorted(set(labels))
    sizes = [int((labels == i).sum()) for i in unique_labels]

    # Get representative samples (using centroid distance)
    samples = {}
    for cluster_id in unique_labels:
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_embeddings = embeddings[cluster_mask]

        if len(cluster_indices) == 0:
            samples[cluster_id] = []
            continue

        # Compute centroid and find closest points
        centroid = cluster_embeddings.mean(axis=0)
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

        n = min(n_samples, len(cluster_indices))
        closest_idx = np.argsort(distances)[:n]
        samples[cluster_id] = [texts[cluster_indices[i]] for i in closest_idx]

    return sil, sizes, samples, labels


def cluster_and_analyze(
    texts: list[str],
    embeddings: np.ndarray,
    k: int,
    n_samples: int = 15,
) -> tuple[float, list[int], dict[int, list[str]]]:
    """Cluster embeddings and extract samples from each cluster (K-Means wrapper).

    Args:
        texts: Original texts.
        embeddings: Embedding vectors.
        k: Number of clusters.
        n_samples: Number of samples to extract per cluster.

    Returns:
        Tuple of (silhouette_score, cluster_sizes, cluster_samples)
    """
    sil, sizes, samples, _ = cluster_kmeans(texts, embeddings, k, n_samples)
    return sil, sizes, samples


def cluster_bertopic(
    texts: list[str],
    embeddings: np.ndarray,
    min_topic_size: int = 50,
    n_samples: int = 15,
    balanced: bool = False,
) -> tuple[
    float, list[int], list[int], dict[int, list[str]], np.ndarray, int, dict[int, list[str]]
]:
    """Topic modeling with BERTopic (auto-detects number of topics).

    BERTopic combines:
    - Pre-computed embeddings (we provide these)
    - HDBSCAN for clustering (auto-determines K)
    - c-TF-IDF for topic representation (interpretable keywords)

    Args:
        texts: Original texts.
        embeddings: Embedding vectors.
        min_topic_size: Minimum topic size for HDBSCAN.
        n_samples: Samples per topic.
        balanced: If True, use balanced config (more topics, fewer outliers).

    Returns:
        Tuple of (silhouette, sizes, topic_ids, samples, labels, n_topics, topic_words)
        where topic_words maps topic_id -> list of top keywords.
    """
    if not HAS_BERTOPIC:
        raise ImportError("bertopic not installed. Run: uv pip install bertopic")

    # Configure HDBSCAN for BERTopic
    hdbscan_model = None
    if HAS_HDBSCAN:
        if balanced:
            # Balanced config: more topics, fewer outliers (good for casual chat)
            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=min_topic_size,
                min_samples=3,  # Lower = more permissive clustering
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            )
        else:
            # Default: conservative clustering
            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=min_topic_size,
                min_samples=10,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            )

    # Configure UMAP for dimensionality reduction
    umap_model = None
    if HAS_UMAP:
        if balanced:
            # Balanced config: preserve more dimensions for better topic separation
            umap_model = umap.UMAP(
                n_neighbors=50,  # More neighbors = smoother manifold
                n_components=25,  # More dims = preserve more info from 768-dim embeddings
                min_dist=0.2,  # Spread points more = less dense clusters
                metric="cosine",
                random_state=42,
            )
        else:
            # Default: aggressive reduction
            umap_model = umap.UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric="cosine",
                random_state=42,
            )

    # Vectorizer with stopwords to avoid "the/and/you" topics
    stop_words = set(ENGLISH_STOP_WORDS)
    stop_words.update({"u", "ur", "im", "idk", "lol", "lmao", "tbh", "ngl"})
    vectorizer_model = CountVectorizer(stop_words=list(stop_words), ngram_range=(1, 2))

    # Create BERTopic model (we skip embedding since we provide pre-computed)
    topic_model = BERTopic(
        embedding_model=None,  # We provide embeddings
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        nr_topics="auto",  # Let it find optimal number
        verbose=False,
    )

    # Fit with pre-computed embeddings
    topics, probs = topic_model.fit_transform(texts, embeddings)
    labels = np.array(topics)

    # Get topic info
    topic_info = topic_model.get_topic_info()
    n_topics = len([t for t in topic_info["Topic"] if t != -1])  # Exclude outlier topic
    n_outliers = int((labels == -1).sum())

    outlier_pct = 100 * n_outliers / len(labels)
    print(f"      Found {n_topics} topics, {n_outliers} outliers ({outlier_pct:.1f}%)")

    # Silhouette score (exclude outliers). Prefer UMAP space if available.
    non_outlier_mask = labels >= 0
    reduced_embeddings = None
    if hasattr(topic_model, "umap_model") and topic_model.umap_model is not None:
        reduced_embeddings = getattr(topic_model.umap_model, "embedding_", None)
        if (
            isinstance(reduced_embeddings, np.ndarray)
            and reduced_embeddings.shape[0] != embeddings.shape[0]
        ):
            reduced_embeddings = None

    if non_outlier_mask.sum() > 100 and n_topics > 1:
        sample_size = min(10000, non_outlier_mask.sum())
        non_outlier_indices = np.where(non_outlier_mask)[0]
        sample_idx = np.random.choice(non_outlier_indices, sample_size, replace=False)
        sil_embeddings = reduced_embeddings if reduced_embeddings is not None else embeddings
        sil = silhouette_score(sil_embeddings[sample_idx], labels[sample_idx])
    else:
        sil = 0.0

    # Get topic sizes
    unique_topics = sorted(set(labels))
    sizes = [int((labels == t).sum()) for t in unique_topics]

    # Get representative samples and keywords per topic
    samples = {}
    topic_words = {}
    for topic_id in unique_topics:
        topic_mask = labels == topic_id
        topic_indices = np.where(topic_mask)[0]

        if len(topic_indices) == 0:
            samples[topic_id] = []
            topic_words[topic_id] = []
            continue

        # Get top keywords for this topic
        if topic_id >= 0:
            words = topic_model.get_topic(topic_id)
            topic_words[topic_id] = [w[0] for w in words[:10]] if words else []
        else:
            topic_words[topic_id] = ["outliers"]

        # Get representative documents (BERTopic has this built-in)
        if topic_id >= 0:
            try:
                repr_docs = topic_model.get_representative_docs(topic_id)
                samples[topic_id] = repr_docs[:n_samples] if repr_docs else []
            except Exception:
                # Fallback: random sample
                n = min(n_samples, len(topic_indices))
                idx = np.random.choice(len(topic_indices), n, replace=False)
                samples[topic_id] = [texts[topic_indices[i]] for i in idx]
        else:
            # For outliers, just take random samples
            n = min(n_samples, len(topic_indices))
            idx = np.random.choice(len(topic_indices), n, replace=False)
            samples[topic_id] = [texts[topic_indices[i]] for i in idx]

    return sil, sizes, unique_topics, samples, labels, n_topics, topic_words


def run_clustering_experiment(
    texts: list[str],
    message_type: str,
    models: list[str],
    k_values: list[int],
    use_hdbscan: bool = False,
    hdbscan_min_cluster_size: int = 50,
    use_gmm: bool = False,
    use_spectral: bool = False,
    spectral_max_samples: int = 10000,
    spectral_neighbors: int = 10,
    use_bertopic: bool = False,
    bertopic_min_topic_size: int = 50,
    bertopic_balanced: bool = False,
    bertopic_sweep: bool = False,
    no_disk_stream: bool = False,
    use_socket: bool = True,
    force_embed: bool = False,
    embed_only: bool = False,
) -> list[ClusterResult]:
    """Run clustering experiment across models and K values.

    Args:
        texts: Messages to cluster.
        message_type: "incoming" or "outgoing"
        models: List of model names to try.
        k_values: List of K values to try (ignored if use_hdbscan=True or use_bertopic=True).
        use_hdbscan: If True, use HDBSCAN (auto-detects K).
        hdbscan_min_cluster_size: Min cluster size for HDBSCAN.
        use_gmm: If True, also run GMM clustering.
        use_spectral: If True, also run Spectral clustering.
        spectral_max_samples: Max samples for spectral (uses Nyström approx if exceeded).
        spectral_neighbors: Number of neighbors for spectral affinity.
        use_bertopic: If True, run BERTopic topic modeling (auto-detects topics).
        bertopic_min_topic_size: Min topic size for BERTopic.
        bertopic_balanced: If True, use balanced config (more topics, fewer outliers).
        bertopic_sweep: If True, run multiple BERTopic configs for comparison.
        no_disk_stream: If True, keep embeddings in RAM instead of streaming to disk.
        use_socket: If True (default), use production socket server (fast).
            If False, use direct calls.

    Returns:
        List of ClusterResult objects.
    """
    results = []

    # Create cache directory for embeddings (stream to disk)
    cache_dir = Path("results/clustering/embed_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        # Determine batch size for this model
        # MinimalEmbedder handles length-sorting internally, so we can use larger batches
        # These are optimized for 8GB M1/M2 systems
        if model in ("bge-large", "arctic-l"):
            batch_size = 128  # Large models - 1.3GB weights, ~200/s
        elif model in ("bge-base", "arctic-m"):
            batch_size = 256  # Medium models - 500MB weights, ~800/s
        else:
            batch_size = 512  # Small models - <200MB weights, ~3000/s

        stream_msg = "in RAM" if no_disk_stream else "cached"
        mode_msg = "socket" if use_socket else "direct"
        print(f"\n  [{model}] {message_type} - {len(texts):,} texts ({stream_msg}, {mode_msg})...")
        start = time.time()
        embeddings = embed_texts(
            texts,
            model,
            message_type,
            batch_size=batch_size,
            cache_dir=None if no_disk_stream else cache_dir,
            use_socket=use_socket,
            force_embed=force_embed,
        )
        embed_time = time.time() - start
        if embed_time > 1:  # Only show rate if we actually embedded
            rate = len(texts) / embed_time
            print(f"    Embedded in {embed_time:.1f}s ({rate:,.0f} texts/sec)")

        # Skip clustering if embed_only
        if embed_only:
            print("    Skipping clustering (--embed-only)")
            continue

        if use_hdbscan:
            # HDBSCAN auto-detects K
            print(f"    Clustering with HDBSCAN (min_cluster_size={hdbscan_min_cluster_size})...")
            start = time.time()
            sil, sizes, samples, labels, n_clusters = cluster_hdbscan(
                texts,
                embeddings,
                min_cluster_size=hdbscan_min_cluster_size,
                use_umap=HAS_UMAP,
            )
            cluster_time = time.time() - start

            result = ClusterResult(
                model_name=model,
                message_type=message_type,
                k=n_clusters,  # Auto-detected
                silhouette=sil,
                cluster_sizes=sizes,
                cluster_samples=samples,
                embedding_time=embed_time,
                cluster_time=cluster_time,
            )
            result.algorithm = "hdbscan"
            results.append(result)
            print(f"      Auto K={n_clusters}, Silhouette: {sil:.3f}")

        else:
            # Run K-Means for each k value
            for k in k_values:
                print(f"    Clustering with K-Means K={k}...")
                start = time.time()
                sil, sizes, samples, _ = cluster_kmeans(texts, embeddings, k)
                cluster_time = time.time() - start

                result = ClusterResult(
                    model_name=model,
                    message_type=message_type,
                    k=k,
                    silhouette=sil,
                    cluster_sizes=sizes,
                    cluster_samples=samples,
                    embedding_time=embed_time,
                    cluster_time=cluster_time,
                )
                result.algorithm = "kmeans"
                results.append(result)
                print(f"      Silhouette: {sil:.3f}, Sizes: {sorted(sizes, reverse=True)[:5]}...")

            # Run GMM if requested
            if use_gmm:
                for k in k_values:
                    print(f"    Clustering with GMM K={k}...")
                    start = time.time()
                    sil, sizes, samples, labels, probs = cluster_gmm(texts, embeddings, k)
                    cluster_time = time.time() - start

                    # Report ambiguous samples (low max probability)
                    max_probs = probs.max(axis=1)
                    n_ambiguous = (max_probs < 0.7).sum()
                    pct_ambiguous = 100 * n_ambiguous / len(texts)

                    result = ClusterResult(
                        model_name=model,
                        message_type=message_type,
                        k=k,
                        silhouette=sil,
                        cluster_sizes=sizes,
                        cluster_samples=samples,
                        embedding_time=embed_time,
                        cluster_time=cluster_time,
                    )
                    result.algorithm = "gmm"
                    result.ambiguous_pct = pct_ambiguous
                    results.append(result)
                    print(
                        f"      Silhouette: {sil:.3f}, Ambiguous (<70% conf): {pct_ambiguous:.1f}%"
                    )

            # Run Spectral if requested
            if use_spectral:
                for k in k_values:
                    approx = "(Nyström approx)" if len(texts) > spectral_max_samples else ""
                    print(f"    Clustering with Spectral K={k} {approx}...")
                    start = time.time()
                    sil, sizes, samples, labels = cluster_spectral(
                        texts,
                        embeddings,
                        k,
                        max_samples=spectral_max_samples,
                        n_neighbors=spectral_neighbors,
                    )
                    cluster_time = time.time() - start

                    result = ClusterResult(
                        model_name=model,
                        message_type=message_type,
                        k=k,
                        silhouette=sil,
                        cluster_sizes=sizes,
                        cluster_samples=samples,
                        embedding_time=embed_time,
                        cluster_time=cluster_time,
                    )
                    result.algorithm = "spectral"
                    results.append(result)
                    print(
                        f"      Silhouette: {sil:.3f}, Sizes: {sorted(sizes, reverse=True)[:5]}..."
                    )

        # Run BERTopic if requested (auto-detects number of topics)
        if use_bertopic:
            # Define configs to run
            if bertopic_sweep:
                # Sweep: run multiple configs for comparison
                configs = [
                    {"name": "default", "min_size": bertopic_min_topic_size, "balanced": False},
                    {"name": "balanced", "min_size": bertopic_min_topic_size, "balanced": True},
                    {"name": "aggressive", "min_size": 25, "balanced": True},
                ]
            else:
                # Single config
                configs = [
                    {
                        "name": "balanced" if bertopic_balanced else "default",
                        "min_size": bertopic_min_topic_size,
                        "balanced": bertopic_balanced,
                    }
                ]

            for cfg in configs:
                cfg_name = cfg["name"]
                cfg_min_size = cfg["min_size"]
                cfg_balanced = cfg["balanced"]
                print(
                    f"    BERTopic [{cfg_name}] "
                    f"(min_topic_size={cfg_min_size}, balanced={cfg_balanced})..."
                )
                start = time.time()
                sil, sizes, topic_ids, samples, labels, n_topics, topic_words = cluster_bertopic(
                    texts,
                    embeddings,
                    min_topic_size=cfg_min_size,
                    balanced=cfg_balanced,
                )
                cluster_time = time.time() - start

                result = ClusterResult(
                    model_name=model,
                    message_type=message_type,
                    k=n_topics,  # Auto-detected
                    silhouette=sil,
                    cluster_sizes=sizes,
                    cluster_samples=samples,
                    embedding_time=embed_time,
                    cluster_time=cluster_time,
                )
                result.algorithm = f"bertopic-{cfg_name}"
                result.topic_words = topic_words
                result.topic_ids = topic_ids
                results.append(result)
                # Print top topics with keywords
                sorted_topics = sorted(
                    [(t, s) for t, s in zip(topic_ids, sizes) if t >= 0],
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]
                for topic_id, size in sorted_topics:
                    keywords = topic_words.get(topic_id, [])[:5]
                    print(f"      Topic {topic_id} ({size} msgs): {', '.join(keywords)}")

    return results


def print_cluster_samples(result: ClusterResult, max_clusters: int = 10) -> None:
    """Print samples from each cluster for inspection."""
    algo = getattr(result, "algorithm", "kmeans")
    print(f"\n{'=' * 70}")
    print(f"Model: {result.model_name} | Type: {result.message_type} | Algo: {algo} | K={result.k}")
    print(f"Silhouette: {result.silhouette:.3f}")
    if algo == "gmm":
        print(f"Ambiguous (<70% confidence): {getattr(result, 'ambiguous_pct', 0):.1f}%")
    print(f"{'=' * 70}")

    # Sort clusters by size (handle BERTopic topic ids)
    if algo.startswith("bertopic") and hasattr(result, "topic_ids"):
        sorted_clusters = sorted(
            [(t, s) for t, s in zip(result.topic_ids, result.cluster_sizes)],
            key=lambda x: x[1],
            reverse=True,
        )
    else:
        sorted_clusters = sorted(enumerate(result.cluster_sizes), key=lambda x: x[1], reverse=True)

    for cluster_id, size in sorted_clusters[:max_clusters]:
        samples = result.cluster_samples.get(cluster_id, [])
        print(f"\n--- Cluster {cluster_id} ({size} messages) ---")
        for i, sample in enumerate(samples[:10]):
            # Truncate long messages
            display = sample[:80] + "..." if len(sample) > 80 else sample
            print(f"  {i + 1}. {display}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Cluster messages to find natural categories")
    parser.add_argument(
        "--limit", type=int, default=None, help="Max messages per type (default: no limit)"
    )
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 15], help="K values to try")
    parser.add_argument("--models", nargs="+", default=MODELS, help="Models to test")
    parser.add_argument(
        "--models-quick",
        action="store_true",
        help="Use quick model subset (bge-small, arctic-xs, arctic-m)",
    )
    parser.add_argument(
        "--models-best", action="store_true", help="Use best performers only (arctic-m, bge-small)"
    )
    parser.add_argument("--type", choices=["incoming", "outgoing", "both"], default="both")
    parser.add_argument("--output", type=Path, default=Path("results/clustering"))
    parser.add_argument("--print-samples", action="store_true", help="Print cluster samples")

    # Clustering algorithm flags
    parser.add_argument(
        "--hdbscan", action="store_true", help="Use HDBSCAN (auto-detects K, exclusive)"
    )
    parser.add_argument("--min-cluster-size", type=int, default=50, help="HDBSCAN min cluster size")
    parser.add_argument(
        "--gmm", action="store_true", help="Also run GMM clustering (soft assignments)"
    )
    parser.add_argument("--spectral", action="store_true", help="Also run Spectral clustering")
    parser.add_argument(
        "--spectral-max-samples",
        type=int,
        default=10000,
        help="Max samples for spectral (uses Nyström approx if exceeded)",
    )
    parser.add_argument(
        "--spectral-neighbors",
        type=int,
        default=10,
        help="Number of neighbors for spectral affinity matrix",
    )
    parser.add_argument(
        "--bertopic", action="store_true", help="Run BERTopic topic modeling (auto-detects topics)"
    )
    parser.add_argument("--min-topic-size", type=int, default=50, help="BERTopic min topic size")
    parser.add_argument(
        "--bertopic-balanced",
        action="store_true",
        help="Use balanced BERTopic config (UMAP: n_neighbors=50, n_components=25)",
    )
    parser.add_argument(
        "--bertopic-sweep",
        action="store_true",
        help="Run multiple BERTopic configs: default, balanced, aggressive (min_size=25)",
    )
    parser.add_argument(
        "--all-algorithms",
        action="store_true",
        help="Run all algorithms: K-Means, GMM, Spectral (not HDBSCAN)",
    )
    parser.add_argument(
        "--auto-k",
        action="store_true",
        help="Run auto-K algorithms only: HDBSCAN + BERTopic (skip K-Means)",
    )
    parser.add_argument(
        "--no-disk-stream",
        action="store_true",
        help="Keep embeddings in RAM instead of caching to disk",
    )
    parser.add_argument(
        "--force-embed",
        action="store_true",
        help="Re-embed even if cache exists (default: always use cache)",
    )
    parser.add_argument(
        "--embed-only", action="store_true", help="Only embed texts and cache, skip clustering"
    )
    parser.add_argument(
        "--socket",
        action="store_true",
        default=False,
        help="Use production socket server (fastest, but higher memory)",
    )
    parser.add_argument(
        "--direct",
        dest="socket",
        action="store_false",
        help="Use direct MinimalEmbedder calls (default, low memory)",
    )
    args = parser.parse_args()

    # Handle model selection flags
    if args.models_quick:
        args.models = MODELS_QUICK
    elif args.models_best:
        args.models = MODELS_BEST

    # Handle --all-algorithms
    if args.all_algorithms:
        args.gmm = True
        args.spectral = True

    # Handle --auto-k (auto-detect number of clusters)
    if args.auto_k:
        args.hdbscan = True
        args.bertopic = True

    print("=" * 70)
    print("UNSUPERVISED CLUSTERING EXPERIMENT")
    print("=" * 70)
    print(f"Models: {args.models}")

    algorithms = []
    if args.hdbscan:
        algorithms.append(f"HDBSCAN (auto K, min_cluster_size={args.min_cluster_size})")
    if args.bertopic:
        algorithms.append(f"BERTopic (auto topics, min_topic_size={args.min_topic_size})")
    if not args.hdbscan:  # K-Means and friends require specifying K
        algorithms.append(f"K-Means K={args.k}")
        if args.gmm:
            algorithms.append(f"GMM K={args.k}")
        if args.spectral:
            approx = (
                f"Nyström>{args.spectral_max_samples}"
                if args.limit > args.spectral_max_samples
                else "full"
            )
            algorithms.append(f"Spectral K={args.k} ({approx})")

    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Message limit: {args.limit}")
    embed_mode = "socket (production)" if args.socket else "direct (MinimalEmbedder)"
    print(f"Embedding mode: {embed_mode}")

    # Check socket server if using socket mode
    if args.socket:
        import socket as sock_module

        sock = sock_module.socket(sock_module.AF_UNIX, sock_module.SOCK_STREAM)
        try:
            sock.connect(SOCKET_PATH_PRODUCTION)
            sock.close()
            print(f"Socket server: connected ({SOCKET_PATH_PRODUCTION})")
        except (ConnectionRefusedError, FileNotFoundError):
            print(f"\nERROR: Socket server not running at {SOCKET_PATH_PRODUCTION}")
            print("Start it with: cd ~/.jarvis/mlx-embed-service && uv run python server.py")
            print("Or use --direct mode to skip the server requirement.")
            sys.exit(1)

    print("\nAvailable backends:")
    print(f"  MLX: {'yes' if HAS_MLX else 'no'}")
    print(f"  HDBSCAN: {'yes' if HAS_HDBSCAN else 'no'}")
    print(f"  UMAP: {'yes' if HAS_UMAP else 'no'}")
    print(f"  BERTopic: {'yes' if HAS_BERTOPIC else 'no'}")

    # Pull messages
    print("\n[1] Pulling messages from iMessage DB...")
    start = time.time()
    incoming, outgoing = get_all_messages(limit=args.limit)
    print(f"    Incoming: {len(incoming)}")
    print(f"    Outgoing: {len(outgoing)}")
    print(f"    Time: {time.time() - start:.1f}s")

    all_results = []

    # Cluster incoming
    if args.type in ("incoming", "both") and incoming:
        print("\n[2] Clustering INCOMING messages...")
        results = run_clustering_experiment(
            incoming,
            "incoming",
            args.models,
            args.k,
            use_hdbscan=args.hdbscan,
            hdbscan_min_cluster_size=args.min_cluster_size,
            use_gmm=args.gmm,
            use_spectral=args.spectral,
            spectral_max_samples=args.spectral_max_samples,
            spectral_neighbors=args.spectral_neighbors,
            use_bertopic=args.bertopic,
            bertopic_min_topic_size=args.min_topic_size,
            bertopic_balanced=args.bertopic_balanced,
            bertopic_sweep=args.bertopic_sweep,
            no_disk_stream=args.no_disk_stream,
            use_socket=args.socket,
            force_embed=args.force_embed,
            embed_only=args.embed_only,
        )
        all_results.extend(results)

    # Cluster outgoing
    if args.type in ("outgoing", "both") and outgoing:
        print("\n[3] Clustering OUTGOING messages...")
        results = run_clustering_experiment(
            outgoing,
            "outgoing",
            args.models,
            args.k,
            use_hdbscan=args.hdbscan,
            hdbscan_min_cluster_size=args.min_cluster_size,
            use_gmm=args.gmm,
            use_spectral=args.spectral,
            spectral_max_samples=args.spectral_max_samples,
            spectral_neighbors=args.spectral_neighbors,
            use_bertopic=args.bertopic,
            bertopic_min_topic_size=args.min_topic_size,
            bertopic_balanced=args.bertopic_balanced,
            bertopic_sweep=args.bertopic_sweep,
            no_disk_stream=args.no_disk_stream,
            use_socket=args.socket,
            force_embed=args.force_embed,
            embed_only=args.embed_only,
        )
        all_results.extend(results)

    # Print samples if requested
    if args.print_samples:
        for result in all_results:
            print_cluster_samples(result)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"{'Model':<12} {'Type':<10} {'Algo':<10} {'K':>4} "
        f"{'Silhouette':>10} {'Embed(s)':>10} {'Cluster(s)':>10}"
    )
    print("-" * 80)
    for r in all_results:
        algo = getattr(r, "algorithm", "kmeans")
        ambig = f" ({getattr(r, 'ambiguous_pct', 0):.0f}%amb)" if algo == "gmm" else ""
        print(
            f"{r.model_name:<12} {r.message_type:<10} {algo:<10} {r.k:>4} "
            f"{r.silhouette:>10.3f} {r.embedding_time:>10.1f} {r.cluster_time:>10.2f}{ambig}"
        )

    # Find best silhouette per type and algorithm
    print("\n" + "=" * 70)
    print("BEST CONFIGURATIONS")
    print("=" * 70)
    for msg_type in ["incoming", "outgoing"]:
        type_results = [r for r in all_results if r.message_type == msg_type]
        if not type_results:
            continue

        print(f"\n{msg_type.upper()}:")

        # Group by algorithm
        algorithms_seen = set(getattr(r, "algorithm", "kmeans") for r in type_results)
        for algo in sorted(algorithms_seen):
            algo_results = [r for r in type_results if getattr(r, "algorithm", "kmeans") == algo]
            if algo_results:
                best = max(algo_results, key=lambda x: x.silhouette)
                extra = ""
                if algo == "gmm":
                    extra = f", {getattr(best, 'ambiguous_pct', 0):.1f}% ambiguous"
                elif algo.startswith("bertopic"):
                    n_topics = len(getattr(best, "topic_words", {}))
                    extra = f", {n_topics} topics discovered"
                print(
                    f"  {algo.upper()}: {best.model_name} K={best.k} "
                    f"(silhouette={best.silhouette:.3f}{extra})"
                )

        # Print samples for overall best
        best_overall = max(type_results, key=lambda x: x.silhouette)
        print_cluster_samples(best_overall, max_clusters=best_overall.k)

    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / f"clustering_results_{int(time.time())}.json"

    results_data = [
        {
            "model": r.model_name,
            "type": r.message_type,
            "algorithm": getattr(r, "algorithm", "kmeans"),
            "k": r.k,
            "silhouette": r.silhouette,
            "ambiguous_pct": getattr(r, "ambiguous_pct", None),
            "sizes": [int(x) for x in r.cluster_sizes],
            "samples": {str(k): v for k, v in r.cluster_samples.items()},  # JSON needs string keys
            "topic_words": {str(k): v for k, v in getattr(r, "topic_words", {}).items()},
            "topic_ids": [int(x) for x in getattr(r, "topic_ids", [])]
            if getattr(r, "topic_ids", None) is not None
            else None,
            "embed_time": r.embedding_time,
            "cluster_time": r.cluster_time,
        }
        for r in all_results
    ]
    output_file.write_text(json.dumps(results_data, indent=2))
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
