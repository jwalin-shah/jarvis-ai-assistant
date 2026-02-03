"""Graph builder for constructing relationship networks.

Builds graph data structures from contact and relationship information,
supporting full network graphs and ego-centric views.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Relationship type colors (Apple-inspired palette)
RELATIONSHIP_COLORS: dict[str, str] = {
    "family": "#FF6B6B",  # Coral red
    "friend": "#4ECDC4",  # Teal
    "work": "#45B7D1",  # Sky blue
    "acquaintance": "#96CEB4",  # Sage
    "romantic": "#DDA0DD",  # Plum
    "professional": "#6495ED",  # Cornflower
    "unknown": "#8E8E93",  # Gray
}

# Node size ranges
MIN_NODE_SIZE = 8
MAX_NODE_SIZE = 40
DEFAULT_NODE_SIZE = 16


@dataclass
class GraphNode:
    """A node in the relationship graph representing a contact.

    Attributes:
        id: Unique identifier (hashed contact ID)
        label: Display name for the node
        size: Node radius based on message frequency
        color: Node color based on relationship type
        relationship_type: Type of relationship (family, friend, work, etc.)
        message_count: Total messages exchanged
        last_contact: ISO timestamp of last message
        sentiment_score: Average sentiment (-1 to 1)
        response_time_avg: Average response time in minutes
        x: X position (set by layout algorithm)
        y: Y position (set by layout algorithm)
        cluster_id: Community/cluster assignment
        metadata: Additional data for tooltips
    """

    id: str
    label: str
    size: float = DEFAULT_NODE_SIZE
    color: str = RELATIONSHIP_COLORS["unknown"]
    relationship_type: str = "unknown"
    message_count: int = 0
    last_contact: str | None = None
    sentiment_score: float = 0.0
    response_time_avg: float | None = None
    x: float | None = None
    y: float | None = None
    cluster_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "label": self.label,
            "size": self.size,
            "color": self.color,
            "relationship_type": self.relationship_type,
            "message_count": self.message_count,
            "last_contact": self.last_contact,
            "sentiment_score": self.sentiment_score,
            "response_time_avg": self.response_time_avg,
            "x": self.x,
            "y": self.y,
            "cluster_id": self.cluster_id,
            "metadata": self.metadata,
        }


@dataclass
class GraphEdge:
    """An edge in the relationship graph representing interaction.

    Attributes:
        source: Source node ID
        target: Target node ID
        weight: Edge weight based on interaction strength (0-1)
        message_count: Total messages between nodes
        sentiment: Average sentiment of interactions
        last_interaction: ISO timestamp of last interaction
        bidirectional: Whether communication is two-way
    """

    source: str
    target: str
    weight: float = 0.5
    message_count: int = 0
    sentiment: float = 0.0
    last_interaction: str | None = None
    bidirectional: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "message_count": self.message_count,
            "sentiment": self.sentiment,
            "last_interaction": self.last_interaction,
            "bidirectional": self.bidirectional,
        }


@dataclass
class GraphData:
    """Complete graph data structure.

    Attributes:
        nodes: List of graph nodes
        edges: List of graph edges
        metadata: Graph-level metadata
    """

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "metadata": self.metadata,
        }

    @property
    def node_count(self) -> int:
        """Number of nodes in the graph."""
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        return len(self.edges)


def _hash_id(contact_id: str) -> str:
    """Create a stable hash for contact ID."""
    return hashlib.sha256(contact_id.encode()).hexdigest()[:16]


def _compute_node_size(
    message_count: int,
    min_count: int,
    max_count: int,
) -> float:
    """Compute node size based on message count using log scale."""
    if max_count == min_count:
        return DEFAULT_NODE_SIZE

    # Use log scale to prevent huge nodes
    log_min = math.log1p(min_count)
    log_max = math.log1p(max_count)
    log_val = math.log1p(message_count)

    if log_max == log_min:
        normalized = 0.5
    else:
        normalized = (log_val - log_min) / (log_max - log_min)

    return MIN_NODE_SIZE + (MAX_NODE_SIZE - MIN_NODE_SIZE) * normalized


def _compute_edge_weight(
    message_count: int,
    max_messages: int,
    recency_days: int | None = None,
) -> float:
    """Compute edge weight from message count and recency."""
    if max_messages == 0:
        return 0.1

    # Base weight from message count (log scale)
    base_weight = math.log1p(message_count) / math.log1p(max_messages)

    # Apply recency decay if available
    if recency_days is not None and recency_days > 0:
        decay = math.exp(-recency_days / 90)  # 90-day half-life
        weight = 0.7 * base_weight + 0.3 * decay
    else:
        weight = base_weight

    return max(0.05, min(1.0, weight))


class GraphBuilder:
    """Builder for constructing relationship graphs.

    Provides methods for building full network graphs and ego-centric
    views from contact and message data.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the graph builder.

        Args:
            db_path: Path to jarvis.db (uses default if None)
        """
        from jarvis.db import JARVIS_DB_PATH

        self.db_path = db_path or JARVIS_DB_PATH
        self._contact_cache: dict[str, dict[str, Any]] = {}
        self._message_stats_cache: dict[str, dict[str, Any]] = {}

    def _get_contacts(self) -> list[dict[str, Any]]:
        """Fetch contacts from database."""
        from jarvis.db import JarvisDB

        contacts = []
        try:
            db = JarvisDB(self.db_path)
            with db.connect() as conn:
                cursor = conn.execute("""
                    SELECT id, chat_id, display_name, phone_or_email,
                           relationship, style_notes, handles_json,
                           created_at, updated_at
                    FROM contacts
                    WHERE display_name IS NOT NULL
                    ORDER BY display_name
                """)
                for row in cursor.fetchall():
                    contacts.append({
                        "id": row[0],
                        "chat_id": row[1],
                        "display_name": row[2],
                        "phone_or_email": row[3],
                        "relationship": row[4] or "unknown",
                        "style_notes": row[5],
                        "handles_json": row[6],
                        "created_at": row[7],
                        "updated_at": row[8],
                    })
        except Exception as e:
            logger.warning(f"Error fetching contacts: {e}")

        return contacts

    def _get_message_stats(
        self,
        chat_id: str | None = None,
        since: datetime | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Get message statistics per contact."""
        from integrations.imessage import ChatDBReader

        stats: dict[str, dict[str, Any]] = {}

        try:
            reader = ChatDBReader()
            conversations = reader.get_conversations(limit=500)

            for conv in conversations:
                cid = conv.chat_id
                if chat_id and cid != chat_id:
                    continue

                # Get message counts
                messages = reader.get_messages(cid, limit=500)
                if since:
                    messages = [m for m in messages if m.date >= since]

                sent_count = sum(1 for m in messages if m.is_from_me)
                received_count = len(messages) - sent_count

                # Calculate sentiment from messages
                sentiment_sum = 0.0
                sentiment_count = 0
                for msg in messages:
                    if msg.text:
                        # Simple sentiment heuristic
                        text = msg.text.lower()
                        positive = sum(
                            1 for w in ["thanks", "love", "great", "awesome", "happy"]
                            if w in text
                        )
                        negative = sum(
                            1 for w in ["sorry", "sad", "angry", "hate", "bad"]
                            if w in text
                        )
                        if positive + negative > 0:
                            sentiment_sum += (positive - negative) / (positive + negative)
                            sentiment_count += 1

                avg_sentiment = sentiment_sum / sentiment_count if sentiment_count > 0 else 0.0

                # Get last message date
                last_date = max((m.date for m in messages), default=None)

                # Store participant data
                for participant in conv.participants:
                    if participant not in stats:
                        stats[participant] = {
                            "message_count": 0,
                            "sent_count": 0,
                            "received_count": 0,
                            "sentiment": 0.0,
                            "sentiment_samples": 0,
                            "last_contact": None,
                            "chat_ids": [],
                        }

                    stats[participant]["message_count"] += len(messages)
                    stats[participant]["sent_count"] += sent_count
                    stats[participant]["received_count"] += received_count
                    stats[participant]["chat_ids"].append(cid)

                    # Update sentiment average
                    old_samples = stats[participant]["sentiment_samples"]
                    if sentiment_count > 0:
                        total = old_samples + sentiment_count
                        old_weight = old_samples / total if total > 0 else 0
                        new_weight = sentiment_count / total if total > 0 else 0
                        stats[participant]["sentiment"] = (
                            stats[participant]["sentiment"] * old_weight
                            + avg_sentiment * new_weight
                        )
                        stats[participant]["sentiment_samples"] = total

                    # Update last contact
                    if last_date:
                        last_str = last_date.isoformat()
                        if (
                            stats[participant]["last_contact"] is None
                            or last_str > stats[participant]["last_contact"]
                        ):
                            stats[participant]["last_contact"] = last_str

        except Exception as e:
            logger.warning(f"Error getting message stats: {e}")

        return stats

    def build_network(
        self,
        include_relationships: list[str] | None = None,
        min_messages: int = 1,
        since: datetime | None = None,
        max_nodes: int = 100,
    ) -> GraphData:
        """Build a full network graph of all contacts.

        Args:
            include_relationships: Filter by relationship types (None = all)
            min_messages: Minimum message count to include
            since: Only include messages after this date
            max_nodes: Maximum number of nodes to include

        Returns:
            GraphData with nodes and edges
        """
        contacts = self._get_contacts()
        stats = self._get_message_stats(since=since)

        # Build contact lookup by various identifiers
        contact_lookup: dict[str, dict[str, Any]] = {}
        for contact in contacts:
            if contact["chat_id"]:
                contact_lookup[contact["chat_id"]] = contact
            if contact["phone_or_email"]:
                contact_lookup[contact["phone_or_email"]] = contact

        # Create nodes
        nodes: list[GraphNode] = []
        node_ids: set[str] = set()

        # Calculate message count range
        message_counts = [
            s.get("message_count", 0)
            for s in stats.values()
            if s.get("message_count", 0) >= min_messages
        ]
        min_count = min(message_counts) if message_counts else 0
        max_count = max(message_counts) if message_counts else 1

        for identifier, stat in stats.items():
            msg_count = stat.get("message_count", 0)
            if msg_count < min_messages:
                continue

            # Look up contact info
            contact = contact_lookup.get(identifier, {})
            relationship = contact.get("relationship", "unknown")

            if include_relationships and relationship not in include_relationships:
                continue

            display_name = contact.get("display_name") or identifier
            node_id = _hash_id(identifier)

            if node_id in node_ids:
                continue

            node = GraphNode(
                id=node_id,
                label=display_name,
                size=_compute_node_size(msg_count, min_count, max_count),
                color=RELATIONSHIP_COLORS.get(relationship, RELATIONSHIP_COLORS["unknown"]),
                relationship_type=relationship,
                message_count=msg_count,
                last_contact=stat.get("last_contact"),
                sentiment_score=stat.get("sentiment", 0.0),
                metadata={
                    "identifier": identifier,
                    "sent": stat.get("sent_count", 0),
                    "received": stat.get("received_count", 0),
                },
            )
            nodes.append(node)
            node_ids.add(node_id)

            if len(nodes) >= max_nodes:
                break

        # Sort by message count and limit
        nodes.sort(key=lambda n: n.message_count, reverse=True)
        nodes = nodes[:max_nodes]
        node_ids = {n.id for n in nodes}

        # Create edges based on shared conversations/groups
        edges: list[GraphEdge] = []
        edge_set: set[tuple[str, str]] = set()

        # Add "me" as central node
        me_node = GraphNode(
            id="me",
            label="Me",
            size=MAX_NODE_SIZE,
            color="#007AFF",  # iOS blue
            relationship_type="self",
            message_count=sum(n.message_count for n in nodes),
        )
        nodes.insert(0, me_node)
        node_ids.add("me")

        # Create edges from me to each contact
        max_messages = max((n.message_count for n in nodes if n.id != "me"), default=1)
        for node in nodes:
            if node.id == "me":
                continue

            # Calculate recency
            recency_days = None
            if node.last_contact:
                try:
                    last_dt = datetime.fromisoformat(node.last_contact.replace("Z", "+00:00"))
                    recency_days = (datetime.now() - last_dt.replace(tzinfo=None)).days
                except ValueError:
                    pass

            edge = GraphEdge(
                source="me",
                target=node.id,
                weight=_compute_edge_weight(node.message_count, max_messages, recency_days),
                message_count=node.message_count,
                sentiment=node.sentiment_score,
                last_interaction=node.last_contact,
            )
            edges.append(edge)
            edge_set.add(("me", node.id))

        # TODO: Add edges between contacts who appear in same group chats

        return GraphData(
            nodes=nodes,
            edges=edges,
            metadata={
                "total_contacts": len(contacts),
                "filtered_nodes": len(nodes),
                "generated_at": datetime.now().isoformat(),
                "min_messages": min_messages,
            },
        )

    def build_ego_graph(
        self,
        contact_id: str,
        depth: int = 1,
        max_neighbors: int = 20,
    ) -> GraphData:
        """Build an ego-centric graph centered on a specific contact.

        Args:
            contact_id: The central contact ID
            depth: How many hops from center to include
            max_neighbors: Maximum neighbors to include

        Returns:
            GraphData centered on the specified contact
        """
        # Build full network first
        full_graph = self.build_network(max_nodes=200)

        # Find the target node
        target_hash = _hash_id(contact_id)
        target_node = None
        for node in full_graph.nodes:
            if node.id == target_hash or node.metadata.get("identifier") == contact_id:
                target_node = node
                break

        if target_node is None:
            # Create a minimal graph with just the contact
            return GraphData(
                nodes=[
                    GraphNode(
                        id=target_hash,
                        label=contact_id,
                        size=DEFAULT_NODE_SIZE,
                    )
                ],
                edges=[],
                metadata={"center": contact_id, "depth": depth},
            )

        # Collect nodes within depth
        included_nodes: dict[str, GraphNode] = {target_node.id: target_node}
        included_edges: list[GraphEdge] = []

        # BFS from target
        current_level = {target_node.id}
        for _ in range(depth):
            next_level: set[str] = set()
            for edge in full_graph.edges:
                if edge.source in current_level and edge.target not in included_nodes:
                    next_level.add(edge.target)
                    # Find the node
                    for node in full_graph.nodes:
                        if node.id == edge.target:
                            included_nodes[node.id] = node
                            break
                    included_edges.append(edge)

                elif edge.target in current_level and edge.source not in included_nodes:
                    next_level.add(edge.source)
                    for node in full_graph.nodes:
                        if node.id == edge.source:
                            included_nodes[node.id] = node
                            break
                    included_edges.append(edge)

            current_level = next_level
            if len(included_nodes) >= max_neighbors + 1:
                break

        # Sort by message count and limit
        nodes_list = sorted(
            included_nodes.values(),
            key=lambda n: n.message_count,
            reverse=True,
        )[:max_neighbors + 1]
        node_ids = {n.id for n in nodes_list}

        # Filter edges to only included nodes
        filtered_edges = [
            e for e in included_edges
            if e.source in node_ids and e.target in node_ids
        ]

        return GraphData(
            nodes=nodes_list,
            edges=filtered_edges,
            metadata={
                "center": contact_id,
                "depth": depth,
                "generated_at": datetime.now().isoformat(),
            },
        )


def build_network_graph(
    include_relationships: list[str] | None = None,
    min_messages: int = 1,
    since: datetime | None = None,
    max_nodes: int = 100,
) -> GraphData:
    """Convenience function to build a network graph.

    Args:
        include_relationships: Filter by relationship types
        min_messages: Minimum message count
        since: Only include messages after this date
        max_nodes: Maximum nodes to include

    Returns:
        GraphData with the network
    """
    builder = GraphBuilder()
    return builder.build_network(
        include_relationships=include_relationships,
        min_messages=min_messages,
        since=since,
        max_nodes=max_nodes,
    )


def build_ego_graph(
    contact_id: str,
    depth: int = 1,
    max_neighbors: int = 20,
) -> GraphData:
    """Convenience function to build an ego graph.

    Args:
        contact_id: The central contact
        depth: Number of hops
        max_neighbors: Max neighbors

    Returns:
        GraphData centered on the contact
    """
    builder = GraphBuilder()
    return builder.build_ego_graph(
        contact_id=contact_id,
        depth=depth,
        max_neighbors=max_neighbors,
    )
