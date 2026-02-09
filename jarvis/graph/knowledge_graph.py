"""Knowledge graph built from contact facts and profiles.

Builds a NetworkX graph with two node types:
- Contact nodes: from contact_profiles table
- Entity nodes: unique subjects from contact_facts table

Edges:
- Contact↔Contact: interaction edges (from builder.py)
- Contact→Entity: fact edges (predicate as type)
- Inferred Contact↔Contact via shared entities
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Entity node colors by fact category
ENTITY_COLORS: dict[str, str] = {
    "relationship": "#DDA0DD",
    "location": "#4ECDC4",
    "work": "#45B7D1",
    "preference": "#F7DC6F",
    "event": "#FF7F50",
    "default": "#8E8E93",
}


@dataclass
class KnowledgeNode:
    """A node in the knowledge graph."""

    id: str
    label: str
    node_type: str  # "contact" or "entity"
    category: str = ""  # For entity nodes: relationship, location, work, etc.
    color: str = "#8E8E93"
    size: float = 12.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEdge:
    """An edge in the knowledge graph."""

    source: str
    target: str
    edge_type: str  # predicate: lives_in, works_at, is_family_of, etc.
    label: str = ""
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGraphData:
    """Serializable knowledge graph data for API/frontend."""

    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """Knowledge graph built from contact facts and profiles."""

    def __init__(self) -> None:
        self._nx = None
        self.graph = None

    def _ensure_nx(self) -> Any:
        """Lazy-import networkx."""
        if self._nx is None:
            import networkx as nx

            self._nx = nx
            self.graph = nx.MultiDiGraph()
        return self._nx

    def build_from_db(self) -> None:
        """Load contact_profiles + contact_facts into the graph."""
        nx = self._ensure_nx()
        self.graph = nx.MultiDiGraph()

        try:
            from jarvis.db import get_db

            db = get_db()
        except Exception as e:
            logger.error("Cannot connect to DB: %s", e)
            return

        with db.connection() as conn:
            # Load contact profiles as contact nodes
            profiles = conn.execute(
                "SELECT contact_id, contact_name, relationship, message_count FROM contact_profiles"
            ).fetchall()

            for row in profiles:
                p = dict(row) if hasattr(row, "keys") else row
                cid = p["contact_id"] if isinstance(p, dict) else p[0]
                name = (p["contact_name"] if isinstance(p, dict) else p[1]) or cid[:12]
                rel = (p["relationship"] if isinstance(p, dict) else p[2]) or "unknown"
                msgs = (p["message_count"] if isinstance(p, dict) else p[3]) or 0

                self.graph.add_node(
                    cid,
                    label=name,
                    node_type="contact",
                    relationship=rel,
                    message_count=msgs,
                    color="#4ECDC4",
                    size=max(12, min(40, 12 + msgs * 0.02)),
                )

            # Load facts and create entity nodes + edges
            facts = conn.execute(
                "SELECT contact_id, category, subject, predicate, value, confidence "
                "FROM contact_facts ORDER BY confidence DESC"
            ).fetchall()

            entity_ids: dict[str, str] = {}  # normalized_subject -> node_id

            for row in facts:
                f = dict(row) if hasattr(row, "keys") else row
                cid = f["contact_id"] if isinstance(f, dict) else f[0]
                cat = f["category"] if isinstance(f, dict) else f[1]
                subj = f["subject"] if isinstance(f, dict) else f[2]
                pred = f["predicate"] if isinstance(f, dict) else f[3]
                val = f["value"] if isinstance(f, dict) else f[4]
                conf = f["confidence"] if isinstance(f, dict) else f[5]

                # Create entity node if not exists
                subj_key = subj.lower().strip()
                if subj_key not in entity_ids:
                    eid = f"entity:{subj_key}"
                    entity_ids[subj_key] = eid
                    self.graph.add_node(
                        eid,
                        label=subj,
                        node_type="entity",
                        category=cat,
                        color=ENTITY_COLORS.get(cat, ENTITY_COLORS["default"]),
                        size=8,
                    )

                # Add edge from contact to entity
                eid = entity_ids[subj_key]
                edge_label = pred.replace("_", " ")
                if val:
                    edge_label += f" ({val})"
                self.graph.add_edge(
                    cid,
                    eid,
                    edge_type=pred,
                    label=edge_label,
                    weight=conf,
                    category=cat,
                )

        logger.info(
            "Knowledge graph built: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def to_graph_data(self) -> KnowledgeGraphData:
        """Convert to serializable format for API."""
        if self.graph is None:
            return KnowledgeGraphData()

        nodes = []
        for nid, attrs in self.graph.nodes(data=True):
            nodes.append(
                {
                    "id": nid,
                    "label": attrs.get("label", nid),
                    "node_type": attrs.get("node_type", "contact"),
                    "category": attrs.get("category", ""),
                    "color": attrs.get("color", "#8E8E93"),
                    "size": attrs.get("size", 12),
                    "relationship_type": attrs.get("relationship", "unknown"),
                    "message_count": attrs.get("message_count", 0),
                    "metadata": {
                        k: v
                        for k, v in attrs.items()
                        if k not in ("label", "node_type", "category", "color", "size")
                    },
                }
            )

        edges = []
        for src, tgt, attrs in self.graph.edges(data=True):
            edges.append(
                {
                    "source": src,
                    "target": tgt,
                    "edge_type": attrs.get("edge_type", ""),
                    "label": attrs.get("label", ""),
                    "weight": attrs.get("weight", 1.0),
                    "category": attrs.get("category", ""),
                }
            )

        return KnowledgeGraphData(
            nodes=nodes,
            edges=edges,
            metadata={
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "contact_count": sum(1 for n in nodes if n["node_type"] == "contact"),
                "entity_count": sum(1 for n in nodes if n["node_type"] == "entity"),
            },
        )

    def query_contact(self, contact_id: str) -> dict[str, Any]:
        """Get all facts and connections for one contact."""
        if self.graph is None or contact_id not in self.graph:
            return {"contact_id": contact_id, "facts": [], "connections": []}

        facts = []
        connections = []
        for _, tgt, attrs in self.graph.edges(contact_id, data=True):
            edge_data = {
                "target": tgt,
                "target_label": self.graph.nodes[tgt].get("label", tgt),
                "edge_type": attrs.get("edge_type", ""),
                "label": attrs.get("label", ""),
                "category": attrs.get("category", ""),
                "weight": attrs.get("weight", 1.0),
            }
            target_type = self.graph.nodes[tgt].get("node_type", "")
            if target_type == "entity":
                facts.append(edge_data)
            else:
                connections.append(edge_data)

        return {
            "contact_id": contact_id,
            "label": self.graph.nodes[contact_id].get("label", contact_id),
            "facts": facts,
            "connections": connections,
        }

    def find_connections(self, entity: str) -> list[str]:
        """Which contacts are connected to this entity?"""
        if self.graph is None:
            return []
        eid = f"entity:{entity.lower().strip()}"
        if eid not in self.graph:
            return []
        # Find all contacts that have edges to this entity
        contacts = []
        for src, tgt, _ in self.graph.in_edges(eid, data=True):
            if self.graph.nodes[src].get("node_type") == "contact":
                contacts.append(src)
        return contacts

    def search_facts(self, query: str) -> list[dict[str, Any]]:
        """Search facts across all contacts."""
        if self.graph is None:
            return []

        query_lower = query.lower()
        results = []
        for src, tgt, attrs in self.graph.edges(data=True):
            label = attrs.get("label", "").lower()
            tgt_label = self.graph.nodes.get(tgt, {}).get("label", "").lower()
            if query_lower in label or query_lower in tgt_label:
                results.append(
                    {
                        "source": src,
                        "source_label": self.graph.nodes[src].get("label", src),
                        "target": tgt,
                        "target_label": self.graph.nodes[tgt].get("label", tgt),
                        "edge_type": attrs.get("edge_type", ""),
                        "label": attrs.get("label", ""),
                        "weight": attrs.get("weight", 1.0),
                    }
                )
        return results[:50]
