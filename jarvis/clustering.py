"""Cluster Analysis Pipeline for Message Pairs.

Runs clustering on pairs in the database to discover intent patterns.
Clusters are stored in the database and linked to pairs for faster retrieval.

This module bridges the gap between:
- Template mining (benchmarks/templates/mine.py) which outputs to JSON
- The clusters table in jarvis.db which is used by the router

Usage:
    from jarvis.clustering import run_cluster_analysis, ClusterAnalyzer

    # Quick analysis with defaults
    stats = run_cluster_analysis()

    # Full control
    analyzer = ClusterAnalyzer(db=my_db, embedder=my_embedder)
    stats = analyzer.analyze(min_cluster_size=5, n_clusters=20)
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from jarvis.db import JarvisDB

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_N_CLUSTERS = 20  # Number of intent clusters to discover
MIN_CLUSTER_SIZE = 5  # Minimum pairs per cluster to keep
MIN_PAIRS_FOR_CLUSTERING = 50  # Minimum pairs needed to run clustering
SAMPLE_SIZE_PER_CLUSTER = 5  # Number of example messages per cluster


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ClusteringStats:
    """Statistics from a clustering run."""

    total_pairs: int
    pairs_clustered: int
    clusters_created: int
    noise_count: int  # Pairs not assigned to any cluster
    largest_cluster_size: int
    smallest_cluster_size: int
    avg_cluster_size: float


# =============================================================================
# Cluster Analyzer
# =============================================================================


class ClusterAnalyzer:
    """Analyzes pairs in the database to discover intent clusters.

    Uses KMeans clustering on trigger embeddings to group similar intents.
    Results are stored in the clusters table and linked to pairs.
    """

    def __init__(
        self,
        db: JarvisDB | None = None,
        embedder: Any | None = None,
    ) -> None:
        """Initialize the cluster analyzer.

        Args:
            db: Database instance. Uses default if None.
            embedder: Embedder with encode() method. Uses default if None.
        """
        self._db = db
        self._embedder = embedder

    @property
    def db(self) -> JarvisDB:
        """Get or create the database instance."""
        if self._db is None:
            from jarvis.db import get_db

            self._db = get_db()
        return self._db

    @property
    def embedder(self) -> Any:
        """Get or create the embedder instance."""
        if self._embedder is None:
            from jarvis.embedding_adapter import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    def analyze(
        self,
        n_clusters: int = DEFAULT_N_CLUSTERS,
        min_cluster_size: int = MIN_CLUSTER_SIZE,
        min_quality: float = 0.3,
    ) -> ClusteringStats:
        """Run cluster analysis on pairs in the database.

        Args:
            n_clusters: Target number of clusters to create.
            min_cluster_size: Minimum pairs per cluster to keep.
            min_quality: Minimum quality score for pairs to include.

        Returns:
            ClusteringStats with analysis results.
        """
        # Get all quality pairs from the database
        logger.info("Loading pairs from database (min_quality=%.2f)", min_quality)
        pairs = self.db.get_training_pairs(min_quality=min_quality)

        if len(pairs) < MIN_PAIRS_FOR_CLUSTERING:
            logger.warning(
                "Not enough pairs for clustering (%d < %d). Skipping.",
                len(pairs),
                MIN_PAIRS_FOR_CLUSTERING,
            )
            return ClusteringStats(
                total_pairs=len(pairs),
                pairs_clustered=0,
                clusters_created=0,
                noise_count=len(pairs),
                largest_cluster_size=0,
                smallest_cluster_size=0,
                avg_cluster_size=0.0,
            )

        logger.info("Loaded %d pairs for clustering", len(pairs))

        # Extract trigger texts and pair IDs
        pair_ids = [p.id for p in pairs if p.id is not None]
        trigger_texts = [p.trigger_text for p in pairs if p.id is not None]

        if not trigger_texts:
            logger.error("No valid pairs found (all have None IDs)")
            return ClusteringStats(
                total_pairs=len(pairs),
                pairs_clustered=0,
                clusters_created=0,
                noise_count=len(pairs),
                largest_cluster_size=0,
                smallest_cluster_size=0,
                avg_cluster_size=0.0,
            )

        # Compute embeddings
        logger.info("Computing embeddings for %d triggers", len(trigger_texts))
        embeddings = self.embedder.encode(trigger_texts, normalize=True)
        embeddings = embeddings.astype(np.float32)

        # Run clustering
        logger.info("Running KMeans clustering with n_clusters=%d", n_clusters)
        labels, centroids = self._cluster_embeddings(embeddings, n_clusters)

        # Analyze clusters and filter by size
        cluster_counts = Counter(labels)
        valid_clusters = {
            cid: count
            for cid, count in cluster_counts.items()
            if cid >= 0 and count >= min_cluster_size
        }

        if not valid_clusters:
            logger.warning("No clusters met the minimum size requirement (%d)", min_cluster_size)
            return ClusteringStats(
                total_pairs=len(pairs),
                pairs_clustered=0,
                clusters_created=0,
                noise_count=len(pairs),
                largest_cluster_size=0,
                smallest_cluster_size=0,
                avg_cluster_size=0.0,
            )

        logger.info(
            "Found %d valid clusters (filtered from %d)",
            len(valid_clusters),
            len(set(labels)),
        )

        # Clear existing clusters and create new ones
        self.db.clear_clusters()

        # Create cluster entries in database
        cluster_id_map: dict[int, int] = {}  # kmeans_label -> db_cluster_id
        for kmeans_label, count in sorted(valid_clusters.items(), key=lambda x: -x[1]):
            # Get sample triggers for this cluster
            cluster_indices = [i for i, l in enumerate(labels) if l == kmeans_label]
            sample_indices = cluster_indices[:SAMPLE_SIZE_PER_CLUSTER]
            sample_triggers = [trigger_texts[i] for i in sample_indices]

            # Get sample responses
            sample_responses = [pairs[i].response_text for i in sample_indices if i < len(pairs)]

            # Generate cluster name from most common words
            cluster_name = self._generate_cluster_name(sample_triggers, kmeans_label)

            # Add cluster to database
            cluster = self.db.add_cluster(
                name=cluster_name,
                description=f"Auto-discovered cluster with {count} pairs",
                example_triggers=sample_triggers,
                example_responses=sample_responses,
            )

            if cluster.id is not None:
                cluster_id_map[kmeans_label] = cluster.id
                logger.debug(
                    "Created cluster %s (id=%d, count=%d)", cluster_name, cluster.id, count
                )

        # Update pair cluster assignments
        assignments: list[tuple[int, int]] = []
        noise_count = 0

        for i, (pair_id, kmeans_label) in enumerate(zip(pair_ids, labels)):
            if kmeans_label in cluster_id_map:
                db_cluster_id = cluster_id_map[kmeans_label]
                assignments.append((pair_id, db_cluster_id))
            else:
                # Not in a valid cluster (too small or noise)
                noise_count += 1

        if assignments:
            updated = self.db.update_cluster_assignments(assignments)
            logger.info("Updated %d pair cluster assignments", updated)

        # Calculate statistics
        cluster_sizes = list(valid_clusters.values())
        stats = ClusteringStats(
            total_pairs=len(pairs),
            pairs_clustered=len(assignments),
            clusters_created=len(cluster_id_map),
            noise_count=noise_count,
            largest_cluster_size=max(cluster_sizes),
            smallest_cluster_size=min(cluster_sizes),
            avg_cluster_size=sum(cluster_sizes) / len(cluster_sizes),
        )

        logger.info(
            "Clustering complete: %d clusters, %d pairs clustered, %d noise",
            stats.clusters_created,
            stats.pairs_clustered,
            stats.noise_count,
        )

        return stats

    def _cluster_embeddings(
        self,
        embeddings: np.ndarray,
        n_clusters: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cluster embeddings using KMeans.

        Args:
            embeddings: Embedding matrix (n_samples x embedding_dim).
            n_clusters: Number of clusters to create.

        Returns:
            Tuple of (cluster_labels, cluster_centroids).
        """
        from sklearn.cluster import KMeans

        # Adjust clusters if not enough data
        n_samples = len(embeddings)
        effective_clusters = min(n_clusters, n_samples // MIN_CLUSTER_SIZE)
        effective_clusters = max(1, effective_clusters)

        if effective_clusters != n_clusters:
            logger.info(
                "Adjusted n_clusters from %d to %d based on data size",
                n_clusters,
                effective_clusters,
            )

        kmeans = KMeans(
            n_clusters=effective_clusters,
            random_state=42,
            n_init=10,
            max_iter=300,
            n_jobs=-1,  # Use all CPU cores for parallelization
        )
        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_

        return labels, centroids

    def _generate_cluster_name(self, sample_triggers: list[str], cluster_id: int) -> str:
        """Generate a name for a cluster based on sample triggers.

        Args:
            sample_triggers: Sample trigger texts from this cluster.
            cluster_id: Numeric cluster ID for fallback naming.

        Returns:
            Human-readable cluster name.
        """
        # Stop words to filter out
        stop_words = frozenset(
            {
                "the",
                "a",
                "an",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "must",
                "shall",
                "can",
                "need",
                "dare",
                "to",
                "of",
                "in",
                "for",
                "on",
                "with",
                "at",
                "by",
                "from",
                "as",
                "into",
                "through",
                "during",
                "before",
                "after",
                "above",
                "below",
                "between",
                "under",
                "again",
                "further",
                "then",
                "once",
                "here",
                "there",
                "when",
                "where",
                "why",
                "how",
                "all",
                "each",
                "few",
                "more",
                "most",
                "other",
                "some",
                "such",
                "no",
                "nor",
                "not",
                "only",
                "own",
                "same",
                "so",
                "than",
                "too",
                "very",
                "just",
                "and",
                "but",
                "if",
                "or",
                "because",
                "until",
                "while",
                "this",
                "that",
                "these",
                "those",
                "am",
                "i",
                "me",
                "my",
                "you",
                "your",
                "he",
                "him",
                "his",
                "she",
                "her",
                "it",
                "its",
                "we",
                "us",
                "our",
                "they",
                "them",
                "their",
                "what",
                "which",
                "who",
                "whom",
                "okay",
                "ok",
                "yeah",
                "yes",
                "no",
            }
        )

        # Combine all triggers and extract words
        combined = " ".join(sample_triggers).lower()
        words = combined.split()
        filtered_words = [w for w in words if len(w) > 2 and w not in stop_words]

        if not filtered_words:
            return f"CLUSTER_{cluster_id}"

        # Get most common words
        word_counts = Counter(filtered_words)
        top_words = [word for word, _ in word_counts.most_common(3)]

        # Create name from top words, include cluster_id for uniqueness
        base_name = "_".join(top_words[:2]).upper()
        if not base_name:
            return f"CLUSTER_{cluster_id}"

        # Always include cluster_id to ensure uniqueness
        return f"{base_name}_{cluster_id}"


# =============================================================================
# Convenience Functions
# =============================================================================


def run_cluster_analysis(
    n_clusters: int = DEFAULT_N_CLUSTERS,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    min_quality: float = 0.3,
) -> ClusteringStats:
    """Run cluster analysis on pairs in the database.

    Convenience function that creates a ClusterAnalyzer and runs analysis.

    Args:
        n_clusters: Target number of clusters to create.
        min_cluster_size: Minimum pairs per cluster to keep.
        min_quality: Minimum quality score for pairs to include.

    Returns:
        ClusteringStats with analysis results.
    """
    analyzer = ClusterAnalyzer()
    return analyzer.analyze(
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        min_quality=min_quality,
    )


def get_cluster_stats() -> dict[str, Any]:
    """Get current cluster statistics from the database.

    Returns:
        Dict with cluster statistics.
    """
    from jarvis.db import get_db

    db = get_db()
    clusters = db.list_clusters()
    stats = db.get_stats()

    return {
        "total_clusters": len(clusters),
        "cluster_names": [c.name for c in clusters],
        "total_pairs": stats.get("pairs", 0),
        "pairs_with_clusters": stats.get("pairs_with_clusters", 0),
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ClusterAnalyzer",
    "ClusteringStats",
    "run_cluster_analysis",
    "get_cluster_stats",
    "DEFAULT_N_CLUSTERS",
    "MIN_CLUSTER_SIZE",
]
