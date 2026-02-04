"""Community detection for relationship graphs.

Provides Louvain and other clustering algorithms for identifying
contact communities in the relationship network.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

from jarvis.graph.builder import GraphData

logger = logging.getLogger(__name__)

# Cluster color palette (visually distinct colors)
CLUSTER_COLORS = [
    "#FF6B6B",  # Coral
    "#4ECDC4",  # Teal
    "#45B7D1",  # Sky Blue
    "#96CEB4",  # Sage
    "#FFEAA7",  # Yellow
    "#DDA0DD",  # Plum
    "#98D8C8",  # Mint
    "#F7DC6F",  # Gold
    "#BB8FCE",  # Purple
    "#85C1E9",  # Light Blue
    "#F8B500",  # Amber
    "#00CED1",  # Dark Cyan
    "#FF7F50",  # Coral
    "#9FE2BF",  # Sea Green
    "#DE3163",  # Cerise
    "#6495ED",  # Cornflower
]


@dataclass
class ClusterResult:
    """Result of community detection.

    Attributes:
        clusters: Mapping of node ID to cluster ID
        modularity: Modularity score of the clustering
        num_clusters: Number of clusters found
        cluster_sizes: Size of each cluster
        cluster_labels: Optional labels for clusters
    """

    clusters: dict[str, int] = field(default_factory=dict)
    modularity: float = 0.0
    num_clusters: int = 0
    cluster_sizes: dict[int, int] = field(default_factory=dict)
    cluster_labels: dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "clusters": self.clusters,
            "modularity": self.modularity,
            "num_clusters": self.num_clusters,
            "cluster_sizes": self.cluster_sizes,
            "cluster_labels": self.cluster_labels,
        }


def get_cluster_colors(num_clusters: int) -> list[str]:
    """Get a list of colors for cluster visualization.

    Args:
        num_clusters: Number of clusters

    Returns:
        List of hex color strings
    """
    if num_clusters <= len(CLUSTER_COLORS):
        return CLUSTER_COLORS[:num_clusters]

    # Generate additional colors by adjusting hue
    colors = list(CLUSTER_COLORS)
    while len(colors) < num_clusters:
        # Generate a random but visually distinct color
        hue = (len(colors) * 0.618033988749895) % 1.0  # Golden ratio
        saturation = 0.6 + random.random() * 0.2
        lightness = 0.5 + random.random() * 0.2

        # HSL to RGB conversion
        c = (1 - abs(2 * lightness - 1)) * saturation
        x = c * (1 - abs((hue * 6) % 2 - 1))
        m = lightness - c / 2

        if hue < 1 / 6:
            r, g, b = c, x, 0
        elif hue < 2 / 6:
            r, g, b = x, c, 0
        elif hue < 3 / 6:
            r, g, b = 0, c, x
        elif hue < 4 / 6:
            r, g, b = 0, x, c
        elif hue < 5 / 6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        r = int((r + m) * 255)
        g = int((g + m) * 255)
        b = int((b + m) * 255)

        colors.append(f"#{r:02x}{g:02x}{b:02x}")

    return colors


class LouvainClustering:
    """Louvain algorithm for community detection.

    The Louvain method optimizes modularity through a two-phase
    iterative process:
    1. Local optimization: Move nodes to neighboring communities
    2. Aggregation: Build a new graph with communities as nodes

    References:
        Blondel et al., "Fast unfolding of communities in large networks"
    """

    def __init__(self, resolution: float = 1.0, random_seed: int | None = None) -> None:
        """Initialize the Louvain clustering.

        Args:
            resolution: Resolution parameter (higher = more clusters)
            random_seed: Seed for reproducibility
        """
        self.resolution = resolution
        if random_seed is not None:
            random.seed(random_seed)

    def detect(self, graph: GraphData) -> ClusterResult:
        """Detect communities in the graph.

        Args:
            graph: Input graph data

        Returns:
            ClusterResult with community assignments
        """
        if not graph.nodes or not graph.edges:
            # Each node is its own cluster
            clusters = {n.id: i for i, n in enumerate(graph.nodes)}
            return ClusterResult(
                clusters=clusters,
                modularity=0.0,
                num_clusters=len(clusters),
                cluster_sizes={i: 1 for i in range(len(clusters))},
            )

        # Build adjacency matrix representation
        node_ids = [n.id for n in graph.nodes]
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        n = len(node_ids)

        # Initialize weights matrix
        weights: dict[tuple[int, int], float] = {}
        degrees: list[float] = [0.0] * n
        total_weight = 0.0

        for edge in graph.edges:
            if edge.source not in id_to_idx or edge.target not in id_to_idx:
                continue

            i = id_to_idx[edge.source]
            j = id_to_idx[edge.target]
            w = edge.weight

            weights[(i, j)] = weights.get((i, j), 0) + w
            weights[(j, i)] = weights.get((j, i), 0) + w

            degrees[i] += w
            degrees[j] += w
            total_weight += 2 * w

        if total_weight == 0:
            clusters = {nid: i for i, nid in enumerate(node_ids)}
            return ClusterResult(
                clusters=clusters,
                modularity=0.0,
                num_clusters=n,
                cluster_sizes={i: 1 for i in range(n)},
            )

        # Initialize each node in its own community
        community = list(range(n))
        community_weight: dict[int, float] = {i: degrees[i] for i in range(n)}
        community_internal: dict[int, float] = {i: 0.0 for i in range(n)}

        # Phase 1: Local optimization
        improved = True
        iteration = 0
        max_iterations = 100

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # Randomize node order
            nodes_order = list(range(n))
            random.shuffle(nodes_order)

            for node in nodes_order:
                current_comm = community[node]
                node_degree = degrees[node]

                # Remove node from current community
                community_weight[current_comm] -= node_degree
                sum_to_current = sum(
                    weights.get((node, j), 0) for j in range(n) if community[j] == current_comm
                )
                community_internal[current_comm] -= sum_to_current

                # Find best community
                best_comm = current_comm
                best_gain = 0.0

                # Check neighboring communities
                neighbor_comms: set[int] = set()
                for j in range(n):
                    if weights.get((node, j), 0) > 0:
                        neighbor_comms.add(community[j])

                for comm in neighbor_comms:
                    sum_to_comm = sum(
                        weights.get((node, j), 0) for j in range(n) if community[j] == comm
                    )

                    # Calculate modularity gain
                    delta_q = (
                        sum_to_comm / total_weight
                        - self.resolution * node_degree * community_weight[comm] / (total_weight**2)
                    )

                    if delta_q > best_gain:
                        best_gain = delta_q
                        best_comm = comm

                # Move node to best community
                community[node] = best_comm
                community_weight[best_comm] += node_degree
                sum_to_best = sum(
                    weights.get((node, j), 0) for j in range(n) if community[j] == best_comm
                )
                community_internal[best_comm] += sum_to_best

                if best_comm != current_comm:
                    improved = True

        # Renumber communities to be contiguous
        unique_comms = sorted(set(community))
        comm_map = {c: i for i, c in enumerate(unique_comms)}
        community = [comm_map[c] for c in community]

        # Calculate final modularity
        modularity = 0.0
        for i in range(n):
            for j in range(n):
                if community[i] == community[j]:
                    actual = weights.get((i, j), 0)
                    expected = degrees[i] * degrees[j] / total_weight
                    modularity += actual - self.resolution * expected

        modularity /= total_weight

        # Build result
        clusters = {node_ids[i]: community[i] for i in range(n)}
        num_clusters = len(unique_comms)

        # Count cluster sizes
        cluster_sizes: dict[int, int] = {}
        for comm in community:
            cluster_sizes[comm] = cluster_sizes.get(comm, 0) + 1

        # Generate cluster labels based on dominant relationship type
        cluster_labels: dict[int, str] = {}
        cluster_relationships: dict[int, dict[str, int]] = {i: {} for i in range(num_clusters)}

        for node in graph.nodes:
            comm = clusters.get(node.id, 0)
            rel_type = node.relationship_type
            cluster_relationships[comm][rel_type] = cluster_relationships[comm].get(rel_type, 0) + 1

        for comm, rel_counts in cluster_relationships.items():
            if rel_counts:
                dominant = max(rel_counts.keys(), key=lambda k: rel_counts[k])
                cluster_labels[comm] = dominant.title()
            else:
                cluster_labels[comm] = f"Cluster {comm + 1}"

        return ClusterResult(
            clusters=clusters,
            modularity=round(modularity, 4),
            num_clusters=num_clusters,
            cluster_sizes=cluster_sizes,
            cluster_labels=cluster_labels,
        )


def detect_communities(
    graph: GraphData,
    resolution: float = 1.0,
    apply_colors: bool = True,
) -> tuple[GraphData, ClusterResult]:
    """Detect communities and optionally apply cluster colors.

    Args:
        graph: Input graph data
        resolution: Louvain resolution parameter
        apply_colors: Whether to update node colors

    Returns:
        Tuple of (updated GraphData, ClusterResult)
    """
    clustering = LouvainClustering(resolution=resolution)
    result = clustering.detect(graph)

    # Apply cluster IDs to nodes
    for node in graph.nodes:
        node.cluster_id = result.clusters.get(node.id)

    # Optionally apply cluster colors
    if apply_colors:
        colors = get_cluster_colors(result.num_clusters)
        for node in graph.nodes:
            if node.cluster_id is not None:
                node.color = colors[node.cluster_id % len(colors)]

    graph.metadata["clustering"] = {
        "algorithm": "louvain",
        "resolution": resolution,
        "modularity": result.modularity,
        "num_clusters": result.num_clusters,
    }

    return graph, result


def cluster_by_relationship(graph: GraphData) -> tuple[GraphData, ClusterResult]:
    """Cluster nodes by their relationship type.

    Simple clustering that groups contacts by their relationship
    category (family, friend, work, etc.).

    Args:
        graph: Input graph data

    Returns:
        Tuple of (updated GraphData, ClusterResult)
    """
    # Group by relationship type
    type_to_cluster: dict[str, int] = {}
    cluster_to_type: dict[int, str] = {}
    clusters: dict[str, int] = {}

    for node in graph.nodes:
        rel_type = node.relationship_type
        if rel_type not in type_to_cluster:
            cluster_id = len(type_to_cluster)
            type_to_cluster[rel_type] = cluster_id
            cluster_to_type[cluster_id] = rel_type

        clusters[node.id] = type_to_cluster[rel_type]
        node.cluster_id = clusters[node.id]

    # Count sizes
    cluster_sizes: dict[int, int] = {}
    for cluster_id in clusters.values():
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

    # Generate labels
    cluster_labels = {cid: ctype.title() for cid, ctype in cluster_to_type.items()}

    result = ClusterResult(
        clusters=clusters,
        modularity=0.0,  # Not applicable
        num_clusters=len(type_to_cluster),
        cluster_sizes=cluster_sizes,
        cluster_labels=cluster_labels,
    )

    graph.metadata["clustering"] = {
        "algorithm": "relationship_type",
        "num_clusters": result.num_clusters,
    }

    return graph, result
