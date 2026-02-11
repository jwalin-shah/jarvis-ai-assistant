"""Graph visualization API endpoints.

Provides endpoints for building and retrieving relationship network graphs,
ego-centric views, cluster analysis, and temporal evolution.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query

from api.schemas.graph import (
    ClusterResultSchema,
    ContactFactSchema,
    ContactProfileDetailSchema,
    ExportGraphRequest,
    ExportGraphResponse,
    GraphDataSchema,
    GraphEdgeSchema,
    GraphEvolutionResponse,
    GraphEvolutionSnapshot,
    GraphNodeSchema,
)
from jarvis.errors import GraphError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["graph"])


def _graph_to_schema(graph_data: Any) -> GraphDataSchema:
    """Convert internal GraphData to API schema."""
    from jarvis.graph.builder import GraphData

    if not isinstance(graph_data, GraphData):
        raise ValueError("Expected GraphData instance")

    nodes = [
        GraphNodeSchema(
            id=n.id,
            label=n.label,
            size=n.size,
            color=n.color,
            relationship_type=n.relationship_type,
            message_count=n.message_count,
            last_contact=n.last_contact,
            sentiment_score=n.sentiment_score,
            response_time_avg=n.response_time_avg,
            x=n.x,
            y=n.y,
            cluster_id=n.cluster_id,
            metadata=n.metadata,
        )
        for n in graph_data.nodes
    ]

    edges = [
        GraphEdgeSchema(
            source=e.source,
            target=e.target,
            weight=e.weight,
            message_count=e.message_count,
            sentiment=e.sentiment,
            last_interaction=e.last_interaction,
            bidirectional=e.bidirectional,
        )
        for e in graph_data.edges
    ]

    return GraphDataSchema(
        nodes=nodes,
        edges=edges,
        metadata=graph_data.metadata,
    )


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
    """Get the full relationship network graph.

    Returns a graph of all contacts with nodes sized by message frequency
    and colored by relationship type. Supports various layout algorithms
    and optional community detection.

    **Layout algorithms:**
    - `force`: Force-directed spring layout (best for general networks)
    - `hierarchical`: Tree-like layout from center
    - `radial`: Concentric circles from center

    **Filtering:**
    - `include_relationships`: Only include specific relationship types
    - `min_messages`: Exclude contacts with fewer messages
    - `days_back`: Only consider recent messages
    """
    try:
        from jarvis.graph import (
            build_network_graph,
            detect_communities,
        )
        from jarvis.graph.layout import LayoutConfig, LayoutEngine

        # Calculate since date
        since = None
        if days_back:
            since = datetime.now() - timedelta(days=days_back)

        # Build the graph
        graph = build_network_graph(
            include_relationships=include_relationships,
            min_messages=min_messages,
            since=since,
            max_nodes=max_nodes,
        )

        # Apply clustering
        if include_clusters and graph.nodes:
            graph, cluster_result = detect_communities(graph, apply_colors=True)

        # Apply layout
        config = LayoutConfig(width=width, height=height)
        engine = LayoutEngine(config)

        if layout == "force":
            graph = engine.force_directed(graph)
        elif layout == "hierarchical":
            graph = engine.hierarchical(graph)
        elif layout == "radial":
            graph = engine.radial(graph)

        return _graph_to_schema(graph)

    except Exception as e:
        logger.exception("Error building network graph")
        raise GraphError("Failed to build network graph", cause=e)


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
    """Get an ego-centric graph centered on a specific contact.

    Returns a graph showing the specified contact and their connections
    up to the given depth. Useful for exploring a single contact's
    network neighborhood.

    Args:
        contact_id: Contact identifier (phone, email, or chat_id)
        depth: Number of hops from center (1-3)
        max_neighbors: Maximum neighbors to include
        layout: Layout algorithm (radial recommended for ego graphs)
    """
    try:
        from jarvis.graph import build_ego_graph
        from jarvis.graph.layout import LayoutConfig, LayoutEngine

        # Build ego graph
        graph = build_ego_graph(
            contact_id=contact_id,
            depth=depth,
            max_neighbors=max_neighbors,
        )

        if not graph.nodes:
            raise HTTPException(status_code=404, detail="Contact not found")

        # Apply layout
        config = LayoutConfig(width=width, height=height)
        engine = LayoutEngine(config)

        if layout == "radial":
            # Find the ego node for centering
            ego_id = graph.metadata.get("center")
            if ego_id:
                from jarvis.graph.builder import _hash_id

                center_id = _hash_id(ego_id)
            else:
                center_id = None
            graph = engine.radial(graph, center_id=center_id)
        else:
            graph = engine.force_directed(graph)

        return _graph_to_schema(graph)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error building ego graph for %s", contact_id)
        raise GraphError(
            f"Failed to build ego graph for contact: {contact_id}",
            contact_id=contact_id,
            cause=e,
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
    """Get a detailed contact profile combining relationship data and knowledge graph facts.

    Returns profile metadata (relationship, formality, style guide, topics)
    plus structured facts from the knowledge graph (location, work, preferences, etc.).
    """
    try:
        from jarvis.contacts.contact_profile import (
            ContactProfile,
        )
        from jarvis.contacts.contact_profile import (
            load_profile as load_contact_profile,
        )
        from jarvis.relationships import generate_style_guide, load_profile

        # Try to load ContactProfile (has formality, topics, etc.)
        cp: ContactProfile | None = None
        try:
            cp = load_contact_profile(contact_id)
        except Exception:
            pass

        # Try to load RelationshipProfile (has style guide data)
        style_guide = ""
        avg_response_time: float | None = None
        try:
            rp = load_profile(contact_id)
            if rp:
                style_guide = generate_style_guide(rp)
                avg_response_time = rp.response_patterns.avg_response_time_minutes
        except Exception:
            pass

        # Load facts from DB
        facts: list[ContactFactSchema] = []
        try:
            from jarvis.db import get_db

            db = get_db()
            with db.connection() as conn:
                rows = conn.execute(
                    "SELECT category, subject, predicate, value, confidence "
                    "FROM contact_facts WHERE contact_id = ? ORDER BY confidence DESC",
                    (contact_id,),
                ).fetchall()
                for row in rows:
                    r = (
                        dict(row)
                        if hasattr(row, "keys")
                        else {
                            "category": row[0],
                            "subject": row[1],
                            "predicate": row[2],
                            "value": row[3],
                            "confidence": row[4],
                        }
                    )
                    facts.append(
                        ContactFactSchema(
                            category=r["category"],
                            subject=r["subject"],
                            predicate=r["predicate"],
                            value=r["value"],
                            confidence=r["confidence"],
                        )
                    )
        except Exception as e:
            logger.debug("Could not load contact facts: %s", e)

        if cp is None and not facts and style_guide == "":
            raise HTTPException(status_code=404, detail="Contact not found")

        return ContactProfileDetailSchema(
            contact_id=contact_id,
            contact_name=cp.contact_name if cp else None,
            relationship=cp.relationship if cp else "unknown",
            formality=cp.formality if cp else "casual",
            formality_score=cp.formality_score if cp else 0.5,
            style_guide=style_guide,
            message_count=cp.message_count if cp else 0,
            avg_message_length=cp.avg_message_length if cp else 0.0,
            avg_response_time_minutes=avg_response_time,
            top_topics=cp.top_topics if cp else [],
            facts=facts,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error loading contact profile for %s", contact_id)
        raise GraphError(
            f"Failed to load contact profile: {contact_id}",
            contact_id=contact_id,
            cause=e,
        )


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
    """Get community/cluster assignments for contacts.

    Uses the Louvain algorithm to detect communities in the
    relationship network. Returns cluster assignments with
    labels based on dominant relationship types.

    **Resolution parameter:**
    - 0.5: Fewer, larger clusters
    - 1.0: Default balance
    - 2.0+: More, smaller clusters
    """
    try:
        from jarvis.graph import build_network_graph, detect_communities

        # Build graph
        graph = build_network_graph(max_nodes=max_nodes)

        # Run clustering
        _, result = detect_communities(graph, resolution=resolution)

        return ClusterResultSchema(
            clusters=result.clusters,
            modularity=result.modularity,
            num_clusters=result.num_clusters,
            cluster_sizes={str(k): v for k, v in result.cluster_sizes.items()},
            cluster_labels={str(k): v for k, v in result.cluster_labels.items()},
        )

    except Exception as e:
        logger.exception("Error computing clusters")
        raise GraphError("Failed to compute graph clusters", cause=e)


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
    """Get graph evolution over a time period.

    Returns a series of graph snapshots showing how the relationship
    network changed over time. Useful for visualizing communication
    pattern changes.

    **Intervals:**
    - `day`: Daily snapshots (max 30 days recommended)
    - `week`: Weekly snapshots (max 52 weeks)
    - `month`: Monthly snapshots (max 24 months)
    """
    try:
        # Parse dates
        try:
            start = datetime.fromisoformat(from_date)
            end = datetime.fromisoformat(to_date)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

        if end <= start:
            raise HTTPException(status_code=400, detail="End date must be after start date")

        # Determine delta
        if interval == "day":
            delta = timedelta(days=1)
            max_snapshots = 30
        elif interval == "week":
            delta = timedelta(weeks=1)
            max_snapshots = 52
        else:
            delta = timedelta(days=30)
            max_snapshots = 24

        # Generate snapshots
        from jarvis.graph import build_network_graph, compute_force_layout

        snapshots: list[GraphEvolutionSnapshot] = []
        current = start

        while current <= end and len(snapshots) < max_snapshots:
            # Build graph for messages up to this point
            graph = build_network_graph(
                since=start,
                until=current,
                max_nodes=max_nodes,
            )

            # Filter to messages before current date
            # (simplified - in production would filter at source)

            if graph.nodes:
                graph = compute_force_layout(graph)

            snapshot = GraphEvolutionSnapshot(
                timestamp=current.isoformat(),
                graph=_graph_to_schema(graph),
                metrics={
                    "node_count": graph.node_count,
                    "edge_count": graph.edge_count,
                    "avg_messages": (
                        sum(n.message_count for n in graph.nodes) / len(graph.nodes)
                        if graph.nodes
                        else 0
                    ),
                },
            )
            snapshots.append(snapshot)
            current += delta

        return GraphEvolutionResponse(
            from_date=from_date,
            to_date=to_date,
            interval=interval,
            snapshots=snapshots,
            total_snapshots=len(snapshots),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error computing graph evolution")
        raise GraphError("Failed to compute graph evolution", cause=e)


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
    """Export the relationship graph to various formats.

    **Formats:**
    - `json`: JSON format for programmatic use
    - `graphml`: GraphML XML format for Gephi, yEd
    - `svg`: Static SVG image
    - `html`: Interactive HTML with D3.js
    """
    try:
        from jarvis.graph import (
            build_network_graph,
            compute_force_layout,
            export_to_graphml,
            export_to_html,
            export_to_json,
            export_to_svg,
        )

        # Build graph
        graph = build_network_graph(max_nodes=max_nodes)

        if request.include_layout and graph.nodes:
            graph = compute_force_layout(
                graph,
                width=request.width,
                height=request.height,
            )

        # Export
        if request.format == "json":
            data = export_to_json(graph)
            filename = "relationship_graph.json"
        elif request.format == "graphml":
            data = export_to_graphml(graph)
            filename = "relationship_graph.graphml"
        elif request.format == "svg":
            data = export_to_svg(
                graph,
                width=request.width,
                height=request.height,
            )
            filename = "relationship_graph.svg"
        else:  # html
            data = export_to_html(
                graph,
                width=request.width,
                height=request.height,
            )
            filename = "relationship_graph.html"

        return ExportGraphResponse(
            format=request.format,
            filename=filename,
            data=data,
            size_bytes=len(data.encode("utf-8")),
        )

    except Exception as e:
        logger.exception("Error exporting graph")
        raise GraphError("Failed to export graph", cause=e)


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
    """Get statistics about the relationship network.

    Returns summary statistics including contact counts,
    message totals, relationship distribution, and clustering info.
    """
    try:
        from jarvis.graph import build_network_graph, detect_communities

        # Build graph
        graph = build_network_graph(max_nodes=max_nodes)

        # Get relationship distribution
        rel_dist: dict[str, int] = {}
        for node in graph.nodes:
            rel_type = node.relationship_type
            rel_dist[rel_type] = rel_dist.get(rel_type, 0) + 1

        # Run clustering for cluster count
        if graph.nodes:
            _, cluster_result = detect_communities(graph)
            cluster_count = cluster_result.num_clusters
        else:
            cluster_count = 0

        # Calculate stats
        total_messages = sum(n.message_count for n in graph.nodes)
        avg_messages = total_messages / len(graph.nodes) if graph.nodes else 0

        # Find most active contact
        most_active = max(graph.nodes, key=lambda n: n.message_count) if graph.nodes else None

        return {
            "total_contacts": len(graph.nodes),
            "total_messages": total_messages,
            "avg_messages_per_contact": round(avg_messages, 1),
            "relationship_distribution": rel_dist,
            "cluster_count": cluster_count,
            "most_active_contact": most_active.label if most_active else None,
            "most_active_messages": most_active.message_count if most_active else 0,
            "generated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.exception("Error computing graph stats")
        raise GraphError("Failed to compute graph statistics", cause=e)
