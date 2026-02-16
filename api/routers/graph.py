"""Graph visualization API endpoints."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Query

from api.schemas.graph import (
    ClusterResultSchema,
    ContactProfileDetailSchema,
    ExportGraphRequest,
    ExportGraphResponse,
    GraphDataSchema,
    GraphEvolutionResponse,
    KnowledgeGraphSchema,
)
from api.services.graph_service import (
    build_ego_graph,
    build_knowledge_graph,
    build_network_graph,
    compute_clusters,
    compute_graph_evolution,
    compute_graph_stats,
    export_graph_data,
    load_contact_profile,
)

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get(
    "/network",
    response_model=GraphDataSchema,
    summary="Get full network graph",
    responses={
        200: {
            "description": "Network graph data",
            "content": {
                "application/json": {
                    "example": {
                        "nodes": [
                            {
                                "id": "abc123",
                                "label": "John Doe",
                                "size": 24.0,
                                "color": "#4ECDC4",
                                "relationship_type": "friend",
                                "message_count": 150,
                            }
                        ],
                        "edges": [
                            {
                                "source": "me",
                                "target": "abc123",
                                "weight": 0.75,
                                "message_count": 150,
                            }
                        ],
                        "metadata": {"total_contacts": 50},
                    }
                }
            },
        },
    },
)
def get_network_graph(
    include_relationships: list[str] | None = Query(
        default=None,
        description="Filter by relationship types",
    ),
    min_messages: int = Query(default=1, ge=0, description="Minimum messages"),
    days_back: int | None = Query(default=None, ge=1, description="Days of history"),
    max_nodes: int = Query(default=100, ge=1, le=500, description="Max nodes"),
    layout: Literal["force", "hierarchical", "radial"] = Query(
        default="force",
        description="Layout algorithm",
    ),
    include_clusters: bool = Query(default=True, description="Run clustering"),
    width: int = Query(default=800, ge=100, le=4000, description="Layout width"),
    height: int = Query(default=600, ge=100, le=4000, description="Layout height"),
) -> GraphDataSchema:
    """Get the full relationship network graph."""
    return build_network_graph(
        include_relationships=include_relationships,
        min_messages=min_messages,
        days_back=days_back,
        max_nodes=max_nodes,
        layout=layout,
        include_clusters=include_clusters,
        width=width,
        height=height,
    )


@router.get(
    "/ego/{contact_id}",
    response_model=GraphDataSchema,
    summary="Get ego-centric graph",
    responses={
        200: {"description": "Ego network centered on contact"},
        404: {"description": "Contact not found"},
    },
)
def get_ego_graph(
    contact_id: str,
    depth: int = Query(default=1, ge=1, le=3, description="Hops from center"),
    max_neighbors: int = Query(default=20, ge=1, le=100, description="Max neighbors"),
    layout: Literal["force", "radial"] = Query(default="radial", description="Layout"),
    width: int = Query(default=800, ge=100, le=4000, description="Layout width"),
    height: int = Query(default=600, ge=100, le=4000, description="Layout height"),
) -> GraphDataSchema:
    """Get an ego-centric graph centered on a specific contact."""
    return build_ego_graph(
        contact_id=contact_id,
        depth=depth,
        max_neighbors=max_neighbors,
        layout=layout,
        width=width,
        height=height,
    )


@router.get(
    "/contact/{contact_id}",
    response_model=ContactProfileDetailSchema,
    summary="Get contact profile with facts",
    responses={
        200: {"description": "Contact profile with knowledge graph facts"},
        404: {"description": "Contact not found"},
    },
)
def get_contact_profile(contact_id: str) -> ContactProfileDetailSchema:
    """Get detailed contact profile including extracted facts."""
    return load_contact_profile(contact_id)


@router.get(
    "/clusters",
    response_model=ClusterResultSchema,
    summary="Get cluster assignments",
    responses={
        200: {
            "description": "Community detection results",
            "content": {
                "application/json": {
                    "example": {
                        "clusters": {"abc123": 0, "def456": 1},
                        "modularity": 0.42,
                        "num_clusters": 3,
                        "cluster_sizes": {"0": 5, "1": 3, "2": 4},
                        "cluster_labels": {"0": "Family", "1": "Work", "2": "Friend"},
                    }
                }
            },
        },
    },
)
def get_clusters(
    max_nodes: int = Query(default=100, ge=1, le=500, description="Max nodes"),
    resolution: float = Query(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Clustering resolution (higher = more clusters)",
    ),
) -> ClusterResultSchema:
    """Get community assignments for contacts."""
    return compute_clusters(max_nodes=max_nodes, resolution=resolution)


@router.get(
    "/evolution",
    response_model=GraphEvolutionResponse,
    summary="Get temporal graph evolution",
    responses={
        200: {"description": "Graph snapshots over time"},
        400: {"description": "Invalid date range"},
    },
)
def get_graph_evolution(
    from_date: str = Query(description="Start date (YYYY-MM-DD)"),
    to_date: str = Query(description="End date (YYYY-MM-DD)"),
    interval: Literal["day", "week", "month"] = Query(
        default="week",
        description="Snapshot interval",
    ),
    max_nodes: int = Query(default=50, ge=1, le=200, description="Max nodes per snapshot"),
) -> GraphEvolutionResponse:
    """Get graph evolution over a time period."""
    return compute_graph_evolution(
        from_date=from_date,
        to_date=to_date,
        interval=interval,
        max_nodes=max_nodes,
    )


@router.post(
    "/export",
    response_model=ExportGraphResponse,
    summary="Export graph to file format",
    responses={
        200: {"description": "Exported graph data"},
    },
)
def export_graph(
    request: ExportGraphRequest,
    max_nodes: int = Query(default=100, ge=1, le=500, description="Max nodes"),
) -> ExportGraphResponse:
    """Export relationship graph in supported formats."""
    return export_graph_data(request=request, max_nodes=max_nodes)


@router.get(
    "/stats",
    summary="Get graph statistics",
    responses={
        200: {
            "description": "Network statistics",
            "content": {
                "application/json": {
                    "example": {
                        "total_contacts": 50,
                        "total_messages": 5000,
                        "avg_messages_per_contact": 100,
                        "relationship_distribution": {"friend": 20, "family": 10},
                        "cluster_count": 5,
                    }
                }
            },
        },
    },
)
def get_graph_stats(
    max_nodes: int = Query(default=200, ge=1, le=500, description="Max nodes to analyze"),
) -> dict[str, Any]:
    """Get summary statistics for the relationship network."""
    return compute_graph_stats(max_nodes=max_nodes)


@router.get(
    "/knowledge",
    response_model=KnowledgeGraphSchema,
    summary="Get knowledge graph with facts as entity nodes",
    responses={
        200: {
            "description": "Knowledge graph with contacts and fact entities",
            "content": {
                "application/json": {
                    "example": {
                        "nodes": [
                            {
                                "id": "alice",
                                "label": "Alice Smith",
                                "node_type": "contact",
                                "relationship_type": "friend",
                                "color": "#4ECDC4",
                                "size": 24,
                            },
                            {
                                "id": "entity:san francisco",
                                "label": "San Francisco",
                                "node_type": "entity",
                                "category": "location",
                                "color": "#4ECDC4",
                                "size": 8,
                            },
                        ],
                        "edges": [
                            {
                                "source": "alice",
                                "target": "entity:san francisco",
                                "edge_type": "lives_in",
                                "label": "lives in",
                                "weight": 0.9,
                                "category": "location",
                            }
                        ],
                        "metadata": {
                            "total_nodes": 10,
                            "total_edges": 8,
                            "contact_count": 3,
                            "entity_count": 7,
                        },
                    }
                }
            },
        },
    },
)
def get_knowledge_graph() -> KnowledgeGraphSchema:
    """Get knowledge graph with contacts and fact entities."""
    return build_knowledge_graph()
