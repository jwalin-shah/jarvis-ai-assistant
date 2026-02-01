"""Response Clustering - Group similar responses into intent clusters.

NOTE: This module requires the optional 'hdbscan' dependency which is not
installed by default. Install with: pip install hdbscan

The router now uses a simpler top-K selection approach that doesn't require
clustering. This module is optional and primarily useful for:
- Analyzing your response patterns
- Mining templates from historical data
- Advanced customization

Uses HDBSCAN to discover natural clusters in your response patterns:
- ACCEPT_INVITATION: "sounds good", "I'm down", "let's do it"
- DECLINE_POLITELY: "can't today", "maybe next time"
- CONFIRM_ARRIVAL: "omw", "be there soon", "5 min"

Usage:
    pip install hdbscan                # Install optional dependency
    jarvis db cluster                  # Auto-cluster responses
    jarvis db label-cluster 0 GREETING # Name a cluster
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    """Configuration for response clustering."""

    # HDBSCAN parameters
    min_cluster_size: int = 10  # Minimum responses to form a cluster
    min_samples: int = 5  # Core point density requirement
    metric: str = "euclidean"  # Distance metric for normalized embeddings

    # Number of example responses to keep per cluster
    num_examples: int = 5


@dataclass
class ClusterResult:
    """Result of clustering operation."""

    cluster_id: int
    name: str | None
    size: int
    example_triggers: list[str]
    example_responses: list[str]
    pair_ids: list[int]


class ResponseClusterer:
    """Clusters response patterns using HDBSCAN.

    Uses sentence embeddings to find semantic groups in your responses,
    then extracts representative examples for each cluster.
    """

    def __init__(self, config: ClusterConfig | None = None) -> None:
        """Initialize clusterer with configuration.

        Args:
            config: Clustering configuration. Uses defaults if None.
        """
        self.config = config or ClusterConfig()
        self._embedder = None
        self._cluster_labels: list[int] | None = None

    @property
    def embedder(self) -> Any:
        """Get the unified embedder."""
        if self._embedder is None:
            from jarvis.embedding_adapter import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    def cluster_responses(
        self,
        pairs: list[dict[str, Any]],
        progress_callback: Any | None = None,
    ) -> list[ClusterResult]:
        """Cluster response texts and return cluster information.

        Args:
            pairs: List of pair dictionaries with 'id', 'trigger_text', 'response_text'.
            progress_callback: Optional callback(stage, progress, message).

        Returns:
            List of ClusterResult objects.
        """
        if not pairs:
            return []

        # Import HDBSCAN (optional dependency)
        try:
            from hdbscan import HDBSCAN
        except ImportError:
            raise ImportError(
                "HDBSCAN is required for clustering. Install with: pip install hdbscan\n"
                "Note: The router now uses top-K selection which doesn't require clustering.\n"
                "Clustering is optional and only needed for advanced response pattern analysis."
            )

        # Stage 1: Extract response texts
        if progress_callback:
            progress_callback("encoding", 0.0, "Extracting response texts...")

        responses = [p["response_text"] for p in pairs]
        pair_ids = [p["id"] for p in pairs]
        triggers = [p["trigger_text"] for p in pairs]

        # Stage 2: Compute embeddings
        if progress_callback:
            progress_callback("encoding", 0.2, f"Encoding {len(responses)} responses...")

        logger.info("Computing embeddings for %d responses...", len(responses))
        embeddings = self.embedder.encode(responses, normalize=True)

        # Stage 3: Run HDBSCAN clustering
        if progress_callback:
            progress_callback("clustering", 0.5, "Running HDBSCAN clustering...")

        logger.info(
            "Clustering with min_cluster_size=%d, min_samples=%d",
            self.config.min_cluster_size,
            self.config.min_samples,
        )

        clusterer = HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            min_samples=self.config.min_samples,
            metric=self.config.metric,
        )
        labels = clusterer.fit_predict(embeddings.astype(np.float32))

        # Store labels for later retrieval
        self._cluster_labels = labels.tolist()

        # Stage 4: Extract cluster information
        if progress_callback:
            progress_callback("extracting", 0.8, "Extracting cluster information...")

        # Get unique cluster labels (excluding -1 which is noise)
        unique_labels = set(labels)
        unique_labels.discard(-1)

        logger.info(
            "Found %d clusters (plus %d noise points)",
            len(unique_labels),
            sum(1 for label in labels if label == -1),
        )

        # Build cluster results
        results: list[ClusterResult] = []

        for cluster_label in sorted(unique_labels):
            # Get indices for this cluster
            indices = [i for i, label in enumerate(labels) if label == cluster_label]

            # Get responses, triggers, and pair IDs for this cluster
            cluster_responses = [responses[i] for i in indices]
            cluster_triggers = [triggers[i] for i in indices]
            cluster_pair_ids = [pair_ids[i] for i in indices]

            # Find representative examples (diverse, not just first N)
            example_indices = self._select_diverse_examples(
                embeddings[indices],
                self.config.num_examples,
            )

            example_triggers = [cluster_triggers[i] for i in example_indices]
            example_responses = [cluster_responses[i] for i in example_indices]

            results.append(
                ClusterResult(
                    cluster_id=int(cluster_label),
                    name=None,  # To be labeled later
                    size=len(indices),
                    example_triggers=example_triggers,
                    example_responses=example_responses,
                    pair_ids=cluster_pair_ids,
                )
            )

        if progress_callback:
            progress_callback("done", 1.0, f"Found {len(results)} clusters")

        return results

    def _select_diverse_examples(
        self,
        embeddings: np.ndarray,
        num_examples: int,
    ) -> list[int]:
        """Select diverse examples from a cluster using maximal marginal relevance.

        Args:
            embeddings: Embeddings for cluster members.
            num_examples: Number of examples to select.

        Returns:
            Indices of selected examples.
        """
        if len(embeddings) <= num_examples:
            return list(range(len(embeddings)))

        n = len(embeddings)

        # Use farthest point sampling for diversity
        # Vectorized: track min distance to any selected point for each candidate
        selected = [0]  # Start with first point

        # Initialize min distances from all points to the first selected point
        # This avoids O(n²) nested loop by incrementally updating distances
        min_distances = np.linalg.norm(embeddings - embeddings[0], axis=1)
        min_distances[0] = -1  # Mark first point as selected (negative = selected)

        while len(selected) < num_examples:
            # Find point with maximum min-distance to selected set
            best_idx = int(np.argmax(min_distances))

            if min_distances[best_idx] <= 0:
                break  # All points selected or no valid candidates

            selected.append(best_idx)

            # Update min distances: for each point, check if new selected point is closer
            new_distances = np.linalg.norm(embeddings - embeddings[best_idx], axis=1)
            min_distances = np.minimum(min_distances, new_distances)
            min_distances[best_idx] = -1  # Mark as selected

        return selected

    def get_cluster_labels(self) -> list[int]:
        """Get the cluster labels from the last clustering run.

        Returns:
            List of cluster labels (-1 for noise).

        Raises:
            RuntimeError: If cluster_responses() has not been called yet.
        """
        if self._cluster_labels is None:
            raise RuntimeError("Call cluster_responses first")
        return self._cluster_labels


def suggest_cluster_names(
    cluster_results: list[ClusterResult],
    use_llm: bool = False,
) -> dict[int, str]:
    """Suggest names for clusters based on their content.

    Args:
        cluster_results: List of ClusterResult objects.
        use_llm: Whether to use LLM for suggestions (future).

    Returns:
        Dictionary mapping cluster_id to suggested name.
    """
    suggestions: dict[int, str] = {}

    # Pattern-based name suggestions
    patterns = {
        "ACCEPT_INVITATION": [
            "sounds good",
            "i'm down",
            "let's do it",
            "i'm in",
            "count me in",
            "works for me",
            "that works",
            "sure thing",
            "absolutely",
            "definitely",
        ],
        "DECLINE_POLITELY": [
            "can't today",
            "maybe next time",
            "not today",
            "sorry can't",
            "rain check",
            "busy",
            "another time",
            "not this time",
        ],
        "CONFIRM_ARRIVAL": [
            "omw",
            "on my way",
            "be there soon",
            "almost there",
            "5 min",
            "leaving now",
            "heading out",
            "pulling up",
        ],
        "GREETING": [
            "hey",
            "hi",
            "hello",
            "what's up",
            "yo",
            "sup",
            "hiya",
        ],
        "ACKNOWLEDGE": [
            "haha",
            "lol",
            "lmao",
            "nice",
            "cool",
            "great",
            "awesome",
            "perfect",
            "sweet",
            "got it",
        ],
        "ASK_TIME": [
            "when",
            "what time",
            "how long",
            "how soon",
        ],
        "ASK_LOCATION": [
            "where",
            "what address",
            "location",
        ],
        "CONFIRM_UNDERSTANDING": [
            "makes sense",
            "understood",
            "got it",
            "roger",
            "copy that",
        ],
        "EXPRESS_THANKS": [
            "thanks",
            "thank you",
            "appreciate",
            "thx",
            "ty",
        ],
        "QUESTION_CLARIFICATION": [
            "what do you mean",
            "could you explain",
            "can you clarify",
            "?",
        ],
    }

    for result in cluster_results:
        # Combine examples for matching
        all_text = " ".join(result.example_responses).lower()

        best_match = None
        best_score = 0

        for name, keywords in patterns.items():
            score = sum(1 for kw in keywords if kw.lower() in all_text)
            if score > best_score:
                best_score = score
                best_match = name

        if best_match and best_score >= 2:
            suggestions[result.cluster_id] = best_match
        else:
            # Generic name based on cluster ID
            suggestions[result.cluster_id] = f"CLUSTER_{result.cluster_id}"

    return suggestions


def save_cluster_results(
    results: list[ClusterResult],
    labels: np.ndarray,
    output_path: Path,
) -> None:
    """Save clustering results to a JSON file.

    Args:
        results: List of ClusterResult objects.
        labels: Array of cluster labels for all pairs.
        output_path: Path to save results.
    """
    data = {
        "clusters": [
            {
                "cluster_id": r.cluster_id,
                "name": r.name,
                "size": r.size,
                "example_triggers": r.example_triggers,
                "example_responses": r.example_responses,
                "pair_ids": r.pair_ids,
            }
            for r in results
        ],
        "labels": labels.tolist(),
        "noise_count": sum(1 for label in labels if label == -1),
        "total_pairs": len(labels),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Saved clustering results to %s", output_path)


def cluster_and_store(
    jarvis_db: Any,
    config: ClusterConfig | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Run clustering on all pairs and store results in the database.

    Args:
        jarvis_db: JarvisDB instance.
        config: Clustering configuration.
        progress_callback: Optional progress callback.

    Returns:
        Statistics about the clustering operation.
    """
    stats = {
        "pairs_processed": 0,
        "clusters_found": 0,
        "noise_pairs": 0,
        "clusters_created": [],
    }

    # Get all pairs from database
    pairs = jarvis_db.get_all_pairs()
    if not pairs:
        logger.warning("No pairs found in database")
        return stats

    # Convert to dict format
    pair_dicts = [
        {
            "id": p.id,
            "trigger_text": p.trigger_text,
            "response_text": p.response_text,
        }
        for p in pairs
    ]

    stats["pairs_processed"] = len(pair_dicts)

    # Run clustering
    clusterer = ResponseClusterer(config)
    results = clusterer.cluster_responses(pair_dicts, progress_callback)

    stats["clusters_found"] = len(results)

    # Get suggested names
    name_suggestions = suggest_cluster_names(results)

    # Clear existing clusters and store new ones
    jarvis_db.clear_clusters()

    for result in results:
        suggested_name = name_suggestions.get(result.cluster_id, f"CLUSTER_{result.cluster_id}")

        # Create cluster in database
        cluster = jarvis_db.add_cluster(
            name=suggested_name,
            description=None,
            example_triggers=result.example_triggers,
            example_responses=result.example_responses,
        )

        stats["clusters_created"].append(
            {
                "id": cluster.id,
                "name": cluster.name,
                "size": result.size,
            }
        )

        # Note: pair_ids -> cluster mapping will be done in index building
        # because we need FAISS IDs to link everything together

    # Count noise (pairs not in any cluster)
    clustered_pair_ids = set()
    for result in results:
        clustered_pair_ids.update(result.pair_ids)

    stats["noise_pairs"] = len(pair_dicts) - len(clustered_pair_ids)

    return stats
