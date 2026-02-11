"""Graph builder for constructing relationship networks.

Builds graph data structures from contact and relationship information,
supporting full network graphs and ego-centric views.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

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

# Sentiment lexicon for casual texting/iMessage
POSITIVE_WORDS: set[str] = {
    # Common positive words
    "love",
    "thanks",
    "thank",
    "great",
    "awesome",
    "happy",
    "good",
    "best",
    "perfect",
    "amazing",
    "wonderful",
    "excellent",
    "fantastic",
    "nice",
    "beautiful",
    "brilliant",
    "excited",
    "fun",
    "cool",
    "glad",
    "appreciate",
    "yes",
    "yay",
    "yeah",
    "yep",
    "sweet",
    "congrats",
    "congratulations",
    "celebrate",
    "lovely",
    "enjoy",
    "enjoyed",
    "delighted",
    "pleased",
    "proud",
    "success",
    "successful",
    "win",
    "won",
    "victory",
    "blessed",
    "grateful",
    "ideal",
    "outstanding",
    "superb",
    "fabulous",
    "marvelous",
    "delightful",
    "incredible",
    "adore",
    "treasure",
    "cherish",
    "joy",
    "joyful",
    # Texting slang
    "lol",
    "lmao",
    "rofl",
    "haha",
    "hehe",
    "omg",
    "wow",
    "yesss",
    "yayyy",
    "woohoo",
    "yasss",
    "lit",
    "fire",
    "dope",
    "sick",
    "rad",
    "stellar",
    "legit",
    "boss",
    "btw",
    "tbh",
    "imo",
    "imho",
    "ngl",
    "fr",
    "bet",
    "vibes",
    "vibe",
    "mood",
    # Positive emojis
    "\U0001f600",
    "\U0001f603",
    "\U0001f604",
    "\U0001f601",
    "\U0001f60a",
    "\U0001f60d",
    "\U0001f970",
    "\U0001f618",
    "\U0001f917",
    "\U0001f929",
    "\U0001f60e",
    "\U0001f64c",
    "\U0001f44d",
    "\U0001f44f",
    "\U0001f4aa",
    "\U0001f389",
    "\U0001f38a",
    "\u2764\ufe0f",
    "\U0001f495",
    "\U0001f496",
    "\U0001f497",
    "\U0001f499",
    "\U0001f49a",
    "\U0001f49b",
    "\U0001f9e1",
    "\U0001f49c",
    "\U0001f90d",
    "\u2728",
    "\u2b50",
    "\U0001f31f",
    "\U0001f4af",
    "\U0001f525",
    "\U0001f44c",
    "\U0001f64f",
    "\U0001f602",
    "\U0001f923",
    "\u263a\ufe0f",
    "\U0001f60c",
    "\U0001f973",
    "\U0001f607",
}

NEGATIVE_WORDS: set[str] = {
    # Common negative words
    "sorry",
    "sad",
    "angry",
    "hate",
    "bad",
    "terrible",
    "awful",
    "horrible",
    "worst",
    "upset",
    "annoyed",
    "frustrated",
    "disappointed",
    "disappointing",
    "sucks",
    "sucked",
    "wrong",
    "problem",
    "issue",
    "unfortunately",
    "miss",
    "missed",
    "cancel",
    "cancelled",
    "worry",
    "worried",
    "stress",
    "stressed",
    "difficult",
    "hard",
    "tough",
    "struggle",
    "struggling",
    "fail",
    "failed",
    "failure",
    "lost",
    "lose",
    "losing",
    "sick",
    "ill",
    "hurt",
    "pain",
    "painful",
    "crying",
    "depressed",
    "anxious",
    "nervous",
    "scared",
    "afraid",
    "fear",
    "disaster",
    "crisis",
    "emergency",
    "broke",
    "broken",
    "damage",
    "damaged",
    "hopeless",
    "helpless",
    "miserable",
    "pathetic",
    "nightmare",
    "regret",
    "unfortunate",
    "dreadful",
    "disgusting",
    "worthless",
    "boring",
    "bored",
    "tired",
    "exhausted",
    "overwhelmed",
    "confused",
    "stupid",
    "ridiculous",
    "annoying",
    "irritated",
    # Texting slang
    "ugh",
    "smh",
    "wtf",
    "omfg",
    "nah",
    "nope",
    "meh",
    "blah",
    "yikes",
    "oof",
    "rip",
    "bruh",
    "fml",
    "smfh",
    "smdh",
    # Negative emojis
    "\U0001f622",
    "\U0001f62d",
    "\U0001f614",
    "\U0001f61e",
    "\U0001f61f",
    "\U0001f615",
    "\U0001f641",
    "\u2639\ufe0f",
    "\U0001f623",
    "\U0001f616",
    "\U0001f62b",
    "\U0001f629",
    "\U0001f624",
    "\U0001f620",
    "\U0001f621",
    "\U0001f92c",
    "\U0001f494",
    "\U0001f630",
    "\U0001f628",
    "\U0001f631",
    "\U0001f633",
    "\U0001f62c",
    "\U0001f926",
    "\U0001f937",
    "\U0001f612",
    "\U0001f644",
    "\U0001f611",
    "\U0001f62a",
}

INTENSIFIERS: set[str] = {
    "very",
    "really",
    "so",
    "super",
    "extremely",
    "totally",
    "absolutely",
    "completely",
    "incredibly",
    "particularly",
    "especially",
    "exceptionally",
    "remarkably",
    "quite",
    "utterly",
    "truly",
    "genuinely",
    "seriously",
    "definitely",
    "literally",
}

# Intensifier boost factor
INTENSIFIER_BOOST = 1.5


def _compute_sentiment_score(text: str) -> float:
    """Compute sentiment score using lexicon-based approach.

    Args:
        text: Message text to analyze

    Returns:
        Sentiment score from -1.0 (negative) to 1.0 (positive)
    """
    if not text:
        return 0.0

    # Tokenize and lowercase
    tokens = text.lower().split()
    if not tokens:
        return 0.0

    positive_score = 0.0
    negative_score = 0.0

    # Find intensifier positions for context-aware boosting
    intensifier_positions: set[int] = set()
    for i, token in enumerate(tokens):
        if token in INTENSIFIERS:
            intensifier_positions.add(i)

    # Score each token using set membership (O(1) per lookup)
    for i, token in enumerate(tokens):
        boost = INTENSIFIER_BOOST if (i - 1) in intensifier_positions else 1.0

        if token in POSITIVE_WORDS:
            positive_score += boost

        if token in NEGATIVE_WORDS:
            negative_score += boost

    # Normalize to -1 to 1 range
    total_score = positive_score + negative_score
    if total_score == 0:
        return 0.0

    # Calculate polarity
    polarity = (positive_score - negative_score) / total_score
    return max(-1.0, min(1.0, polarity))


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
            with db.connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT id, chat_id, display_name, phone_or_email,
                           relationship, style_notes, handles_json,
                           created_at, updated_at
                    FROM contacts
                    WHERE display_name IS NOT NULL
                    ORDER BY display_name
                """
                )
                for row in cursor.fetchall():
                    contacts.append(
                        {
                            "id": row[0],
                            "chat_id": row[1],
                            "display_name": row[2],
                            "phone_or_email": row[3],
                            "relationship": row[4] or "unknown",
                            "style_notes": row[5],
                            "handles_json": row[6],
                            "created_at": row[7],
                            "updated_at": row[8],
                        }
                    )
        except Exception as e:
            logger.warning("Error fetching contacts: %s", e)

        return contacts

    def _get_message_stats(
        self,
        chat_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Get message statistics per contact."""
        from integrations.imessage import ChatDBReader

        stats: dict[str, dict[str, Any]] = {}

        try:
            reader = ChatDBReader()
            conversations = reader.get_conversations(limit=500)

            # Filter conversations if specific chat_id requested
            if chat_id:
                conversations = [c for c in conversations if c.chat_id == chat_id]

            # Collect all chat_ids and fetch messages in a single batch query
            chat_ids = [conv.chat_id for conv in conversations]
            messages_by_chat: dict[str, list] = self._batch_get_messages(
                reader, chat_ids, limit_per_chat=500, after=since, before=until
            )

            for conv in conversations:
                cid = conv.chat_id
                messages = messages_by_chat.get(cid, [])

                sent_count = sum(1 for m in messages if m.is_from_me)
                received_count = len(messages) - sent_count

                # Calculate sentiment from messages using lexicon
                sentiment_sum = 0.0
                sentiment_count = 0
                for msg in messages:
                    if msg.text:
                        score = _compute_sentiment_score(msg.text)
                        if score != 0.0:
                            sentiment_sum += score
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
            logger.warning("Error getting message stats: %s", e)

        return stats

    @staticmethod
    def _batch_get_messages(
        reader: Any,
        chat_ids: list[str],
        limit_per_chat: int = 500,
        after: datetime | None = None,
        before: datetime | None = None,
    ) -> dict[str, list]:
        """Fetch messages for multiple chat_ids in a single SQL query.

        Returns a dict mapping chat_id -> list of Message objects.
        Falls back to per-chat queries if the batch query fails.
        """
        if not chat_ids:
            return {}

        from contracts.imessage import Message
        from integrations.imessage.parser import (
            datetime_to_apple_timestamp,
            parse_apple_timestamp,
        )

        messages_by_chat: dict[str, list] = defaultdict(list)

        try:
            with reader._connection_context() as conn:
                placeholders = ",".join("?" * len(chat_ids))
                params: list[Any] = list(chat_ids)

                date_clauses = ""
                if after is not None:
                    params.append(datetime_to_apple_timestamp(after))
                    date_clauses += "AND message.date > ?"
                if before is not None:
                    params.append(datetime_to_apple_timestamp(before))
                    date_clauses += " AND message.date < ?"

                query = f"""
                    SELECT
                        chat.guid as chat_id,
                        message.ROWID as msg_id,
                        message.text,
                        message.is_from_me,
                        message.date,
                        handle.id as sender,
                        ROW_NUMBER() OVER (
                            PARTITION BY chat.guid ORDER BY message.date DESC
                        ) as rn
                    FROM chat
                    JOIN chat_message_join ON chat.ROWID = chat_message_join.chat_id
                    JOIN message ON chat_message_join.message_id = message.ROWID
                    LEFT JOIN handle ON message.handle_id = handle.ROWID
                    WHERE chat.guid IN ({placeholders})
                    {date_clauses}
                    ORDER BY chat.guid, message.date DESC
                """

                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()

                for row in rows:
                    row_chat_id = row[0]
                    rn = row[6]
                    if rn > limit_per_chat:
                        continue

                    msg_date = parse_apple_timestamp(row[4]) if row[4] else None

                    msg = Message(
                        id=row[1],
                        chat_id=row_chat_id,
                        sender=row[5] or "",
                        sender_name=None,
                        text=row[2] or "",
                        date=msg_date or datetime.now(),
                        is_from_me=bool(row[3]),
                    )
                    messages_by_chat[row_chat_id].append(msg)

        except Exception as e:
            logger.warning("Batch message fetch failed, falling back to per-chat: %s", e)
            for cid in chat_ids:
                try:
                    messages_by_chat[cid] = reader.get_messages(
                        cid, limit=limit_per_chat, after=after
                    )
                except Exception:
                    messages_by_chat[cid] = []

        return dict(messages_by_chat)

    def build_network(
        self,
        include_relationships: list[str] | None = None,
        min_messages: int = 1,
        since: datetime | None = None,
        until: datetime | None = None,
        max_nodes: int = 100,
    ) -> GraphData:
        """Build a full network graph of all contacts.

        Args:
            include_relationships: Filter by relationship types (None = all)
            min_messages: Minimum message count to include
            since: Only include messages after this date
            until: Only include messages before this date
            max_nodes: Maximum number of nodes to include

        Returns:
            GraphData with nodes and edges
        """
        contacts = self._get_contacts()
        stats = self._get_message_stats(since=since, until=until)

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

        # Future enhancement: Add edges between contacts who appear in same group chats

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

        # Index nodes by ID and identifier for O(1) lookup
        target_hash = _hash_id(contact_id)
        nodes_by_id = {n.id: n for n in full_graph.nodes}
        nodes_by_ident = {
            n.metadata.get("identifier"): n
            for n in full_graph.nodes
            if n.metadata.get("identifier")
        }
        target_node = nodes_by_id.get(target_hash) or nodes_by_ident.get(contact_id)

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

        # BFS from target (uses nodes_by_id dict for O(1) lookups)
        current_level = {target_node.id}
        for _ in range(depth):
            next_level: set[str] = set()
            for edge in full_graph.edges:
                if edge.source in current_level and edge.target not in included_nodes:
                    next_level.add(edge.target)
                    if edge.target in nodes_by_id:
                        included_nodes[edge.target] = nodes_by_id[edge.target]
                    included_edges.append(edge)

                elif edge.target in current_level and edge.source not in included_nodes:
                    next_level.add(edge.source)
                    if edge.source in nodes_by_id:
                        included_nodes[edge.source] = nodes_by_id[edge.source]
                    included_edges.append(edge)

            current_level = next_level
            if len(included_nodes) >= max_neighbors + 1:
                break

        # Sort by message count and limit
        nodes_list = sorted(
            included_nodes.values(),
            key=lambda n: n.message_count,
            reverse=True,
        )[: max_neighbors + 1]
        node_ids = {n.id for n in nodes_list}

        # Filter edges to only included nodes
        filtered_edges = [
            e for e in included_edges if e.source in node_ids and e.target in node_ids
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
    until: datetime | None = None,
    max_nodes: int = 100,
) -> GraphData:
    """Convenience function to build a network graph.

    Args:
        include_relationships: Filter by relationship types
        min_messages: Minimum message count
        since: Only include messages after this date
        until: Only include messages before this date
        max_nodes: Maximum nodes to include

    Returns:
        GraphData with the network
    """
    builder = GraphBuilder()
    return builder.build_network(
        include_relationships=include_relationships,
        min_messages=min_messages,
        since=since,
        until=until,
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
