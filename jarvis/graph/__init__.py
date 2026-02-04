"""Graph data module for relationship visualization.

Provides graph construction, layout algorithms, community detection,
and export functionality for contact relationship networks.
"""

from jarvis.graph.builder import (
    GraphBuilder,
    GraphData,
    GraphEdge,
    GraphNode,
    build_ego_graph,
    build_network_graph,
)
from jarvis.graph.clustering import (
    ClusterResult,
    detect_communities,
    get_cluster_colors,
)
from jarvis.graph.export import (
    export_to_graphml,
    export_to_html,
    export_to_json,
    export_to_svg,
)
from jarvis.graph.layout import (
    LayoutEngine,
    compute_force_layout,
    compute_hierarchical_layout,
)

__all__ = [
    # Builder
    "GraphBuilder",
    "GraphData",
    "GraphEdge",
    "GraphNode",
    "build_network_graph",
    "build_ego_graph",
    # Clustering
    "ClusterResult",
    "detect_communities",
    "get_cluster_colors",
    # Layout
    "LayoutEngine",
    "compute_force_layout",
    "compute_hierarchical_layout",
    # Export
    "export_to_json",
    "export_to_graphml",
    "export_to_html",
    "export_to_svg",
]
