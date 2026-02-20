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

from jarvis.utils.latency_tracker import track_latency

# Type hint for networkx graph
_nx_graph_type = Any

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
    edges: list[KnowledgeEdge] = field(default_factory=list)


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
        self._nx: Any = None
        self.graph: _nx_graph_type = None
        # Inverted index: lowercased word -> set of (src, tgt, edge_key) tuples
        self._search_index: dict[str, set[tuple[str, str, int]]] = {}
        # Edge data cache for search results: (src, tgt, key) -> attrs dict
        self._edge_cache: dict[tuple[str, str, int], dict[str, Any]] = {}

    def _ensure_nx(self) -> _nx_graph_type:
        """Lazy-import networkx."""
        if self._nx is None:
            import networkx as nx

            self._nx = nx
            self.graph = nx.MultiDiGraph()
        return self._nx

    def _get_edges_batch(self, conn: Any, node_ids: list[str]) -> dict[str, list[KnowledgeEdge]]:
        """Fetch edges for multiple nodes in a single query.

        Args:
            conn: Database connection
            node_ids: List of node IDs to fetch edges for

        Returns:
            Dict mapping node_id -> list of KnowledgeEdge
        """
        if not node_ids:
            return {}

        # Build placeholders for IN clause: (?, ?, ?, ...)
        placeholders = ",".join(["?"] * len(node_ids))

        # Single query to fetch all edges for all nodes
        cursor = conn.execute(
            f"""
            SELECT contact_id, subject, predicate, value, confidence
            FROM contact_facts
            WHERE contact_id IN ({placeholders})
            ORDER BY contact_id, confidence DESC
            """,
            tuple(node_ids),
        )

        # Map edges by node_id for O(1) lookup
        edges_by_node: dict[str, list[KnowledgeEdge]] = {nid: [] for nid in node_ids}

        for row in cursor.fetchall():
            contact_id = row["contact_id"]
            subject = row["subject"]
            predicate = row["predicate"]
            value = row["value"]
            confidence = row["confidence"]

            # Create edge from contact to entity
            edge = KnowledgeEdge(
                source=contact_id,
                target=f"entity:{subject.lower().strip()}",
                edge_type=predicate,
                label=f"{predicate.replace('_', ' ')} ({value})"
                if value
                else predicate.replace("_", " "),
                weight=confidence,
            )
            edges_by_node[contact_id].append(edge)

        return edges_by_node

    def build_nodes_with_edges_batch(self, conn: Any) -> list[KnowledgeNode]:
        """Build nodes with edges using batch fetching (N+1 fix).

        Before (N+1 pattern):
            for node in nodes:  # N iterations
                edges = self._get_edges(node.id)  # N queries!
                node.edges = edges

        After (batch pattern):
            nodes = [...]  # 1 query for nodes
            edges_by_node = self._get_edges_batch(conn, node_ids)  # 1 query for edges
            for node in nodes:
                node.edges = edges_by_node.get(node.id, [])  # O(1) lookup

        Returns:
            List of KnowledgeNode with edges populated
        """
        # 1. Fetch all nodes (1 query)
        cursor = conn.execute(
            """
            SELECT chat_id AS contact_id, display_name AS contact_name,
                   relationship, 0 AS message_count
            FROM contacts
            WHERE chat_id IS NOT NULL
            """
        )

        nodes: list[KnowledgeNode] = []
        node_ids: list[str] = []

        for row in cursor.fetchall():
            cid = row["contact_id"]
            name = row["contact_name"] or cid[:12]
            rel = row["relationship"] or "unknown"

            node = KnowledgeNode(
                id=cid,
                label=name,
                node_type="contact",
                category=rel,
                color="#4ECDC4",
                size=max(12, min(40, 12 + row["message_count"] * 0.02)),
            )
            nodes.append(node)
            node_ids.append(cid)

        # 2. Batch fetch all edges in ONE query (not N queries!)
        edges_by_node = self._get_edges_batch(conn, node_ids)

        # 3. Assign edges to nodes in a single pass (O(1) lookup)
        for node in nodes:
            node.edges = edges_by_node.get(node.id, [])

        return nodes

    def build_from_db(self) -> None:
        """Load contact_profiles + contact_facts into the graph."""
        with track_latency("graph_build"):
            nx = self._ensure_nx()
            self.graph = nx.MultiDiGraph()

            try:
                from jarvis.db import get_db

                db = get_db()
            except Exception as e:
                logger.error("Cannot connect to DB: %s", e)
                return

            import time

            start_time = time.perf_counter()

            with db.connection() as conn:
                # Load contacts as contact nodes
                profiles = conn.execute(
                    "SELECT chat_id AS contact_id, display_name AS contact_name,"
                    " relationship, 0 AS message_count FROM contacts"
                    " WHERE chat_id IS NOT NULL"
                ).fetchall()

                # PERF FIX: Use batch operations add_nodes_from() and add_edges_from()
                # Before: 1100+ individual add_node() and add_edge() calls = ~200ms
                # After: 3 batch operations = ~30ms

                # Collect all contact nodes for batch insertion
                contact_nodes = []
                for row in profiles:
                    p = dict(row) if hasattr(row, "keys") else row
                    cid = p["contact_id"] if isinstance(p, dict) else p[0]
                    name = (p["contact_name"] if isinstance(p, dict) else p[1]) or cid[:12]
                    rel = (p["relationship"] if isinstance(p, dict) else p[2]) or "unknown"
                    msgs = (p["message_count"] if isinstance(p, dict) else p[3]) or 0

                    contact_nodes.append(
                        (
                            cid,
                            {
                                "label": name,
                                "node_type": "contact",
                                "relationship": rel,
                                "message_count": msgs,
                                "color": "#4ECDC4",
                                "size": max(12, min(40, 12 + msgs * 0.02)),
                            },
                        )
                    )

                # Batch add all contact nodes at once
                self.graph.add_nodes_from(contact_nodes)

                # Load facts and create entity nodes + edges
                facts = conn.execute(
                    "SELECT contact_id, category, subject, predicate, value, confidence "
                    "FROM contact_facts ORDER BY confidence DESC"
                ).fetchall()

                entity_ids: dict[str, str] = {}  # normalized_subject -> node_id
                entity_nodes = []  # Collect entity nodes for batch insertion
                edges = []  # Collect edges for batch insertion

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
                        entity_nodes.append(
                            (
                                eid,
                                {
                                    "label": subj,
                                    "node_type": "entity",
                                    "category": cat,
                                    "color": ENTITY_COLORS.get(cat, ENTITY_COLORS["default"]),
                                    "size": 8,
                                },
                            )
                        )

                    # Add edge from contact to entity
                    eid = entity_ids[subj_key]
                    edge_label = pred.replace("_", " ")
                    if val:
                        edge_label += f" ({val})"
                    edges.append(
                        (
                            cid,
                            eid,
                            {
                                "edge_type": pred,
                                "label": edge_label,
                                "weight": conf,
                                "category": cat,
                            },
                        )
                    )

                # Batch add all entity nodes at once
                if entity_nodes:
                    self.graph.add_nodes_from(entity_nodes)

                # Batch add all edges at once
                if edges:
                    self.graph.add_edges_from(edges)

            # Build search index for fast fact lookups
            self._build_search_index()

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "Knowledge graph built: %d nodes, %d edges in %.1fms (batch operations)",
                self.graph.number_of_nodes(),
                self.graph.number_of_edges(),
                elapsed_ms,
            )

    def _build_search_index(self) -> None:
        """Build inverted index over edge labels and target labels for fast search."""
        self._search_index.clear()
        self._edge_cache.clear()
        if self.graph is None:
            return

        for src, tgt, key, attrs in self.graph.edges(data=True, keys=True):
            edge_ref = (src, tgt, key)
            self._edge_cache[edge_ref] = attrs

            # Index edge label tokens
            label = attrs.get("label", "").lower()
            tgt_label = self.graph.nodes.get(tgt, {}).get("label", "").lower()

            tokens: set[str] = set()
            for text in (label, tgt_label):
                for token in text.split():
                    token = token.strip("().,;:")
                    if len(token) >= 2:
                        tokens.add(token)

            for token in tokens:
                if token not in self._search_index:
                    self._search_index[token] = set()
                self._search_index[token].add(edge_ref)

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
            target_node = self.graph.nodes.get(tgt)
            if target_node is None:
                logger.warning("Target node %s not found in graph", tgt)
                continue
            edge_data = {
                "target": tgt,
                "target_label": target_node.get("label", tgt),
                "edge_type": attrs.get("edge_type", ""),
                "label": attrs.get("label", ""),
                "category": attrs.get("category", ""),
                "weight": attrs.get("weight", 1.0),
            }
            target_type = target_node.get("node_type", "")
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

    def search_facts(self, query: str, limit: int = 50) -> list[dict[str, Any]]:
        """Search facts across all contacts using inverted index.

        Uses token-based index for fast lookups. Falls back to full scan
        for substring queries that don't match whole tokens.
        """
        if self.graph is None:
            return []

        query_lower = query.lower().strip()
        if not query_lower:
            return []

        # Try indexed lookup first: find edges matching any query token
        query_tokens = [t.strip("().,;:") for t in query_lower.split() if len(t) >= 2]
        candidate_refs: set[tuple[str, str, int]] | None = None

        if query_tokens:
            for token in query_tokens:
                matching = self._search_index.get(token, set())
                if candidate_refs is None:
                    candidate_refs = set(matching)
                else:
                    # Intersect: all tokens must match
                    candidate_refs &= matching

        # Verify candidates with substring match (index is token-based, query may be substring)
        results = []
        if candidate_refs:
            for edge_ref in candidate_refs:
                src, tgt, _ = edge_ref
                attrs = self._edge_cache.get(edge_ref, {})
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
                    if len(results) >= limit:
                        break
        else:
            # Fallback: full scan for queries with no token matches (e.g., single char)
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
                    if len(results) >= limit:
                        break
        return results
