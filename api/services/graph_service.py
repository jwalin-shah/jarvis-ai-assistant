"""Service-layer logic for graph API endpoints."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Literal

from fastapi import HTTPException

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
    KnowledgeEdgeSchema,
    KnowledgeGraphSchema,
    KnowledgeNodeSchema,
)
from jarvis.core.exceptions import GraphError

logger = logging.getLogger(__name__)


def graph_to_schema(graph_data: object) -> GraphDataSchema:
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

    return GraphDataSchema(nodes=nodes, edges=edges, metadata=graph_data.metadata)


def build_network_graph(
    *,
    include_relationships: list[str] | None,
    min_messages: int,
    days_back: int | None,
    max_nodes: int,
    layout: Literal["force", "hierarchical", "radial"],
    include_clusters: bool,
    width: int,
    height: int,
) -> GraphDataSchema:
    """Build full network graph and map to API schema."""
    try:
        from jarvis.graph import build_network_graph as _build_network_graph
        from jarvis.graph import detect_communities
        from jarvis.graph.layout import LayoutConfig, LayoutEngine

        since = datetime.now() - timedelta(days=days_back) if days_back else None
        graph = _build_network_graph(
            include_relationships=include_relationships,
            min_messages=min_messages,
            since=since,
            max_nodes=max_nodes,
        )

        if include_clusters and graph.nodes:
            graph, _ = detect_communities(graph, apply_colors=True)

        config = LayoutConfig(width=width, height=height)
        engine = LayoutEngine(config)
        if layout == "force":
            graph = engine.force_directed(graph)
        elif layout == "hierarchical":
            graph = engine.hierarchical(graph)
        else:
            graph = engine.radial(graph)

        return graph_to_schema(graph)
    except Exception as e:
        logger.exception("Error building network graph")
        raise GraphError("Failed to build network graph", cause=e)


def build_ego_graph(
    *,
    contact_id: str,
    depth: int,
    max_neighbors: int,
    layout: Literal["force", "radial"],
    width: int,
    height: int,
) -> GraphDataSchema:
    """Build ego graph centered on a contact."""
    try:
        from jarvis.graph import build_ego_graph as _build_ego_graph
        from jarvis.graph.layout import LayoutConfig, LayoutEngine

        graph = _build_ego_graph(contact_id=contact_id, depth=depth, max_neighbors=max_neighbors)

        if not graph.nodes:
            raise HTTPException(status_code=404, detail="Contact not found")

        config = LayoutConfig(width=width, height=height)
        engine = LayoutEngine(config)

        if layout == "radial":
            ego_id = graph.metadata.get("center")
            center_id = None
            if ego_id:
                from jarvis.graph.builder import _hash_id

                center_id = _hash_id(ego_id)
            graph = engine.radial(graph, center_id=center_id)
        else:
            graph = engine.force_directed(graph)

        return graph_to_schema(graph)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error building ego graph for %s", contact_id)
        raise GraphError(
            f"Failed to build ego graph for contact: {contact_id}",
            contact_id=contact_id,
            cause=e,
        )


def load_contact_profile(contact_id: str) -> ContactProfileDetailSchema:
    """Load contact profile details including structured facts."""
    try:
        from jarvis.contacts.contact_profile import ContactProfile
        from jarvis.contacts.contact_profile import load_profile as load_contact_profile
        from jarvis.relationships import generate_style_guide, load_profile

        cp: ContactProfile | None = None
        try:
            cp = load_contact_profile(contact_id)
        except Exception as e:
            logger.debug(f"Failed to load contact profile for {contact_id}: {e}")

        style_guide = ""
        avg_response_time: float | None = None
        try:
            rp = load_profile(contact_id)
            if rp:
                style_guide = generate_style_guide(rp)
                avg_response_time = rp.response_patterns.avg_response_time_minutes  # type: ignore[attr-defined]
        except Exception:  # nosec B110
            pass

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


def compute_clusters(*, max_nodes: int, resolution: float) -> ClusterResultSchema:
    """Compute community assignments for the relationship graph."""
    try:
        from jarvis.graph import build_network_graph, detect_communities

        graph = build_network_graph(max_nodes=max_nodes)
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


def compute_graph_evolution(
    *,
    from_date: str,
    to_date: str,
    interval: Literal["day", "week", "month"],
    max_nodes: int,
) -> GraphEvolutionResponse:
    """Compute graph snapshots over time."""
    try:
        try:
            start = datetime.fromisoformat(from_date)
            end = datetime.fromisoformat(to_date)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

        if end <= start:
            raise HTTPException(status_code=400, detail="End date must be after start date")

        if interval == "day":
            delta = timedelta(days=1)
            max_snapshots = 30
        elif interval == "week":
            delta = timedelta(weeks=1)
            max_snapshots = 52
        else:
            delta = timedelta(days=30)
            max_snapshots = 24

        from jarvis.graph import build_network_graph, compute_force_layout

        snapshots: list[GraphEvolutionSnapshot] = []
        current = start
        while current <= end and len(snapshots) < max_snapshots:
            graph = build_network_graph(since=start, until=current, max_nodes=max_nodes)
            if graph.nodes:
                graph = compute_force_layout(graph)

            snapshot = GraphEvolutionSnapshot(
                timestamp=current.isoformat(),
                graph=graph_to_schema(graph),
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


def export_graph_data(*, request: ExportGraphRequest, max_nodes: int) -> ExportGraphResponse:
    """Export graph to supported formats."""
    try:
        from jarvis.graph import (
            build_network_graph,
            compute_force_layout,
            export_to_graphml,
            export_to_html,
            export_to_json,
            export_to_svg,
        )

        graph = build_network_graph(max_nodes=max_nodes)
        if request.include_layout and graph.nodes:
            graph = compute_force_layout(graph, width=request.width, height=request.height)

        if request.format == "json":
            data = export_to_json(graph)
            filename = "relationship_graph.json"
        elif request.format == "graphml":
            data = export_to_graphml(graph)
            filename = "relationship_graph.graphml"
        elif request.format == "svg":
            data = export_to_svg(graph, width=request.width, height=request.height)
            filename = "relationship_graph.svg"
        else:
            data = export_to_html(graph, width=request.width, height=request.height)
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


def compute_graph_stats(*, max_nodes: int) -> dict[str, object]:
    """Compute summary statistics for relationship graph."""
    try:
        from jarvis.graph import build_network_graph, detect_communities

        graph = build_network_graph(max_nodes=max_nodes)

        rel_dist: dict[str, int] = {}
        for node in graph.nodes:
            rel_type = node.relationship_type
            rel_dist[rel_type] = rel_dist.get(rel_type, 0) + 1

        if graph.nodes:
            _, cluster_result = detect_communities(graph)
            cluster_count = cluster_result.num_clusters
        else:
            cluster_count = 0

        total_messages = sum(n.message_count for n in graph.nodes)
        avg_messages = total_messages / len(graph.nodes) if graph.nodes else 0
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


def build_knowledge_graph() -> KnowledgeGraphSchema:
    """Build and map contact/entity knowledge graph to API schema."""
    try:
        from jarvis.graph import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.build_from_db()
        data = kg.to_graph_data()

        nodes = [
            KnowledgeNodeSchema(
                id=n["id"],
                label=n["label"],
                node_type=n.get("node_type", "contact"),
                category=n.get("category", ""),
                color=n.get("color", "#8E8E93"),
                size=n.get("size", 12.0),
                relationship_type=n.get("relationship_type", "unknown"),
                message_count=n.get("message_count", 0),
                metadata=n.get("metadata", {}),
            )
            for n in data.nodes
        ]

        edges = [
            KnowledgeEdgeSchema(
                source=e["source"],
                target=e["target"],
                edge_type=e.get("edge_type", ""),
                label=e.get("label", ""),
                weight=e.get("weight", 1.0),
                category=e.get("category", ""),
            )
            for e in data.edges
        ]

        return KnowledgeGraphSchema(nodes=nodes, edges=edges, metadata=data.metadata)
    except Exception as e:
        logger.exception("Error building knowledge graph")
        raise GraphError("Failed to build knowledge graph", cause=e)
