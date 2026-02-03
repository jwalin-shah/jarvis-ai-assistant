"""Embedding-based Relationship Profiling.

Uses vector embeddings to analyze communication patterns:
- Topic clusters from message embeddings
- Semantic similarity between your style and theirs
- Communication dynamics (who initiates, response patterns)
- Topic evolution over time

Complements the keyword-based profiling in relationships.py with
learned representations that capture semantic meaning.

Profile storage: ~/.jarvis/embedding_profiles/{contact_hash}.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

EMBEDDING_PROFILES_DIR = Path.home() / ".jarvis" / "embedding_profiles"
MIN_MESSAGES_FOR_EMBEDDING_PROFILE = 30  # Need enough data for meaningful clusters
PROFILE_VERSION = "1.0.0"

# Clustering config
DEFAULT_NUM_CLUSTERS = 5  # Number of topic clusters to extract
MIN_CLUSTER_SIZE = 3  # Minimum messages per cluster


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TopicCluster:
    """A cluster of semantically similar messages representing a topic.

    Attributes:
        cluster_id: Unique identifier for the cluster.
        centroid: The average embedding vector (serialized as list).
        sample_messages: Representative messages from this cluster.
        message_count: Number of messages in the cluster.
        from_me_ratio: Ratio of messages from you vs them (0-1).
        label: Optional human-readable label.
    """

    cluster_id: int
    centroid: list[float]
    sample_messages: list[str]
    message_count: int
    from_me_ratio: float
    label: str | None = None


@dataclass
class CommunicationDynamics:
    """Semantic analysis of communication patterns.

    Attributes:
        your_avg_embedding: Average embedding of your messages.
        their_avg_embedding: Average embedding of their messages.
        style_similarity: Cosine similarity between your and their avg embeddings.
        initiation_ratio: Ratio of conversations you initiate (0-1).
        topic_diversity: Entropy of topic cluster distribution.
        response_semantic_shift: How much your responses differ from triggers.
    """

    your_avg_embedding: list[float]
    their_avg_embedding: list[float]
    style_similarity: float
    initiation_ratio: float
    topic_diversity: float
    response_semantic_shift: float


@dataclass
class EmbeddingProfile:
    """Complete embedding-based relationship profile.

    Attributes:
        contact_id: Hashed contact identifier.
        contact_name: Display name if available.
        topic_clusters: List of discovered topic clusters.
        dynamics: Communication dynamics analysis.
        message_count: Total messages analyzed.
        embedding_model: Model used for embeddings.
        last_updated: ISO timestamp of last update.
        version: Profile format version.
        relationship_type: Classified relationship (family, close friend, coworker, etc.).
        relationship_confidence: Confidence score for the classification (0-1).
    """

    contact_id: str
    contact_name: str | None = None
    topic_clusters: list[TopicCluster] = field(default_factory=list)
    dynamics: CommunicationDynamics | None = None
    message_count: int = 0
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    last_updated: str = ""
    version: str = PROFILE_VERSION
    relationship_type: str | None = None
    relationship_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to dictionary for serialization."""
        return {
            "contact_id": self.contact_id,
            "contact_name": self.contact_name,
            "topic_clusters": [asdict(tc) for tc in self.topic_clusters],
            "dynamics": asdict(self.dynamics) if self.dynamics else None,
            "message_count": self.message_count,
            "embedding_model": self.embedding_model,
            "last_updated": self.last_updated,
            "version": self.version,
            "relationship_type": self.relationship_type,
            "relationship_confidence": self.relationship_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingProfile:
        """Create profile from dictionary."""
        topic_clusters = [TopicCluster(**tc) for tc in data.get("topic_clusters", [])]
        dynamics_data = data.get("dynamics")
        dynamics = CommunicationDynamics(**dynamics_data) if dynamics_data else None

        return cls(
            contact_id=data["contact_id"],
            contact_name=data.get("contact_name"),
            topic_clusters=topic_clusters,
            dynamics=dynamics,
            message_count=data.get("message_count", 0),
            embedding_model=data.get("embedding_model", "BAAI/bge-small-en-v1.5"),
            last_updated=data.get("last_updated", ""),
            version=data.get("version", PROFILE_VERSION),
            relationship_type=data.get("relationship_type"),
            relationship_confidence=data.get("relationship_confidence"),
        )


# =============================================================================
# Embedding Analysis Functions
# =============================================================================


def _hash_contact_id(contact_id: str) -> str:
    """Create a stable hash for contact ID storage."""
    from jarvis.contact_utils import hash_contact_id

    return hash_contact_id(contact_id)


def _compute_embeddings(
    texts: list[str],
    embedder: Any,
    batch_size: int = 32,
) -> np.ndarray:
    """Compute normalized embeddings for a list of texts.

    Args:
        texts: List of text strings to embed.
        embedder: Embedding model with encode() method.
        batch_size: Batch size for encoding.

    Returns:
        Normalized embedding matrix (n_texts x embedding_dim).
    """
    if not texts:
        return np.array([])

    embeddings = embedder.encode(texts, normalize=True)
    return embeddings.astype(np.float32)


def _cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int = DEFAULT_NUM_CLUSTERS,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster embeddings using K-means.

    Args:
        embeddings: Embedding matrix (n_samples x embedding_dim).
        n_clusters: Number of clusters to create.
        min_cluster_size: Minimum samples per cluster.

    Returns:
        Tuple of (cluster_labels, cluster_centroids).
    """
    from sklearn.cluster import KMeans

    # Adjust clusters if not enough data
    n_samples = len(embeddings)
    effective_clusters = min(n_clusters, n_samples // min_cluster_size)
    effective_clusters = max(1, effective_clusters)

    kmeans = KMeans(
        n_clusters=effective_clusters,
        random_state=42,
        n_init=10,
        max_iter=300,
        n_jobs=-1,  # Use all CPU cores for parallelization
    )
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    return labels, centroids


def _compute_entropy(distribution: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution."""
    distribution = np.array(distribution)
    distribution = distribution / distribution.sum()  # Normalize
    # Filter out zeros to avoid log(0)
    distribution = distribution[distribution > 0]
    return float(-np.sum(distribution * np.log2(distribution)))


def _extract_sample_messages(
    messages: list[str],
    labels: np.ndarray,
    cluster_id: int,
    n_samples: int = 3,
) -> list[str]:
    """Extract representative sample messages from a cluster.

    Args:
        messages: All message texts.
        labels: Cluster labels for each message.
        cluster_id: Which cluster to sample from.
        n_samples: Number of samples to extract.

    Returns:
        List of sample message texts.
    """
    cluster_indices = np.where(labels == cluster_id)[0]
    if len(cluster_indices) == 0:
        return []

    # Take evenly spaced samples
    step = max(1, len(cluster_indices) // n_samples)
    sample_indices = cluster_indices[::step][:n_samples]

    return [messages[i][:200] for i in sample_indices]  # Truncate long messages


def build_embedding_profile(
    contact_id: str,
    messages: list[Any],
    embedder: Any,
    contact_name: str | None = None,
    n_clusters: int = DEFAULT_NUM_CLUSTERS,
) -> EmbeddingProfile:
    """Build an embedding-based relationship profile.

    Args:
        contact_id: Unique identifier for the contact.
        messages: List of Message objects with .text and .is_from_me attributes.
        embedder: Embedding model with encode() method.
        contact_name: Optional display name for the contact.
        n_clusters: Number of topic clusters to extract.

    Returns:
        EmbeddingProfile with semantic analysis.
    """
    hashed_id = _hash_contact_id(contact_id)

    # Filter messages with text
    valid_messages = [m for m in messages if m.text and len(m.text.strip()) > 2]

    if len(valid_messages) < MIN_MESSAGES_FOR_EMBEDDING_PROFILE:
        logger.info(
            "Not enough messages for embedding profile (%d < %d)",
            len(valid_messages),
            MIN_MESSAGES_FOR_EMBEDDING_PROFILE,
        )
        return EmbeddingProfile(
            contact_id=hashed_id,
            contact_name=contact_name,
            message_count=len(valid_messages),
            last_updated=datetime.now().isoformat(),
        )

    # Extract texts and metadata
    texts = [m.text for m in valid_messages]
    is_from_me = [m.is_from_me for m in valid_messages]

    # Compute embeddings
    logger.info("Computing embeddings for %d messages", len(texts))
    embeddings = _compute_embeddings(texts, embedder)

    # Cluster messages by topic
    logger.info("Clustering into %d topic clusters", n_clusters)
    labels, centroids = _cluster_embeddings(embeddings, n_clusters)

    # Log cluster distribution for debugging
    from collections import Counter

    cluster_distribution = Counter(labels)
    logger.debug(
        "Cluster distribution: %s (total centroids: %d)",
        dict(cluster_distribution.most_common(10)),
        len(centroids),
    )

    # Build topic clusters
    topic_clusters = []
    skipped_small_clusters = 0
    for cluster_id in range(len(centroids)):
        cluster_mask = labels == cluster_id
        cluster_count = cluster_mask.sum()

        if cluster_count < MIN_CLUSTER_SIZE:
            skipped_small_clusters += 1
            logger.debug(
                "Skipping cluster %d: only %d messages (min: %d)",
                cluster_id,
                cluster_count,
                MIN_CLUSTER_SIZE,
            )
            continue

        # Compute ratio of "from me" messages in this cluster
        cluster_from_me = np.array(is_from_me)[cluster_mask]
        from_me_ratio = cluster_from_me.sum() / len(cluster_from_me)

        # Get sample messages
        samples = _extract_sample_messages(texts, labels, cluster_id)

        topic_clusters.append(
            TopicCluster(
                cluster_id=cluster_id,
                centroid=centroids[cluster_id].tolist(),
                sample_messages=samples,
                message_count=int(cluster_count),
                from_me_ratio=round(float(from_me_ratio), 3),
            )
        )

    # Log clustering summary
    logger.info(
        "Built %d topic clusters (skipped %d small clusters with < %d messages)",
        len(topic_clusters),
        skipped_small_clusters,
        MIN_CLUSTER_SIZE,
    )

    # Compute communication dynamics
    my_mask = np.array(is_from_me)
    their_mask = ~my_mask

    if my_mask.sum() > 0 and their_mask.sum() > 0:
        my_embeddings = embeddings[my_mask]
        their_embeddings = embeddings[their_mask]

        my_avg = my_embeddings.mean(axis=0)
        their_avg = their_embeddings.mean(axis=0)

        # Style similarity (cosine since embeddings are normalized)
        style_sim = float(np.dot(my_avg, their_avg))

        # Topic diversity (entropy of cluster distribution)
        cluster_counts = np.bincount(labels, minlength=len(centroids))
        topic_diversity = _compute_entropy(cluster_counts)

        # Response semantic shift (avg distance between consecutive messages)
        response_shifts = []
        for i in range(1, len(embeddings)):
            if is_from_me[i] != is_from_me[i - 1]:
                # This is a response - compute semantic distance
                dist = 1 - float(np.dot(embeddings[i], embeddings[i - 1]))
                response_shifts.append(dist)

        avg_shift = sum(response_shifts) / len(response_shifts) if response_shifts else 0.0

        # Initiation ratio (who starts conversations)
        # Simple heuristic: first message in each "session" (gap > 1 hour)
        initiations_me = 0
        initiations_them = 0
        prev_time = None

        for m in valid_messages:
            if prev_time is None or (m.date - prev_time).total_seconds() > 3600:
                # New conversation session
                if m.is_from_me:
                    initiations_me += 1
                else:
                    initiations_them += 1
            prev_time = m.date

        total_initiations = initiations_me + initiations_them
        initiation_ratio = initiations_me / total_initiations if total_initiations > 0 else 0.5

        dynamics = CommunicationDynamics(
            your_avg_embedding=my_avg.tolist(),
            their_avg_embedding=their_avg.tolist(),
            style_similarity=round(style_sim, 3),
            initiation_ratio=round(float(initiation_ratio), 3),
            topic_diversity=round(topic_diversity, 3),
            response_semantic_shift=round(avg_shift, 3),
        )
    else:
        dynamics = None

    return EmbeddingProfile(
        contact_id=hashed_id,
        contact_name=contact_name,
        topic_clusters=topic_clusters,
        dynamics=dynamics,
        message_count=len(valid_messages),
        last_updated=datetime.now().isoformat(),
    )


# =============================================================================
# Profile Storage Functions
# =============================================================================


def _ensure_profiles_dir() -> Path:
    """Ensure the embedding profiles directory exists."""
    EMBEDDING_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    return EMBEDDING_PROFILES_DIR


def _get_profile_path(contact_id: str) -> Path:
    """Get the file path for a contact's embedding profile."""
    hashed_id = _hash_contact_id(contact_id)
    return _ensure_profiles_dir() / f"{hashed_id}.json"


def save_embedding_profile(profile: EmbeddingProfile) -> bool:
    """Save an embedding profile to disk.

    Args:
        profile: The EmbeddingProfile to save.

    Returns:
        True if saved successfully.
    """
    try:
        profile_path = _ensure_profiles_dir() / f"{profile.contact_id}.json"
        with profile_path.open("w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, indent=2)
        return True
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to save embedding profile: %s", e)
        return False


def load_embedding_profile(contact_id: str) -> EmbeddingProfile | None:
    """Load an embedding profile from disk.

    Args:
        contact_id: The contact identifier (will be hashed).

    Returns:
        EmbeddingProfile if found, None otherwise.
    """
    profile_path = _get_profile_path(contact_id)

    if not profile_path.exists():
        return None

    try:
        with profile_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return EmbeddingProfile.from_dict(data)
    except (OSError, json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to load embedding profile: %s", e)
        return None


def delete_embedding_profile(contact_id: str) -> bool:
    """Delete an embedding profile from disk."""
    profile_path = _get_profile_path(contact_id)
    try:
        if profile_path.exists():
            profile_path.unlink()
        return True
    except OSError:
        return False


def list_embedding_profiles() -> list[str]:
    """List all saved embedding profile IDs."""
    try:
        _ensure_profiles_dir()
        return [p.stem for p in EMBEDDING_PROFILES_DIR.glob("*.json")]
    except OSError:
        return []


# =============================================================================
# Profile Generation from Database
# =============================================================================


def build_profiles_for_all_contacts(
    db: Any,
    embedder: Any,
    imessage_reader: Any | None = None,
    min_messages: int = MIN_MESSAGES_FOR_EMBEDDING_PROFILE,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build embedding profiles for all contacts in the database.

    Args:
        db: JarvisDB instance.
        embedder: Embedding model with encode() method.
        imessage_reader: Optional ChatDBReader for fetching message history.
        min_messages: Minimum messages required for a profile.
        limit: Maximum number of contacts to process.

    Returns:
        Statistics about the profile building.
    """
    contacts = db.list_contacts(limit=limit or 1000)
    logger.info("Building embedding profiles for %d contacts", len(contacts))

    stats = {
        "contacts_processed": 0,
        "profiles_created": 0,
        "profiles_skipped": 0,
        "total_messages_analyzed": 0,
        "errors": [],
    }

    for contact in contacts:
        if not contact.chat_id and not contact.id:
            stats["profiles_skipped"] += 1
            continue

        try:
            messages = []

            # Try iMessage reader first
            if imessage_reader and contact.chat_id:
                try:
                    messages = imessage_reader.get_messages(
                        contact.chat_id,
                        limit=5000,
                    )
                except Exception as e:
                    logger.debug("iMessage failed for %s: %s", contact.display_name, e)

            # Fallback: use pairs from database if iMessage didn't work
            if len(messages) < min_messages and contact.id:
                pairs = db.get_pairs(contact_id=contact.id, limit=5000)
                messages = []
                for p in pairs:
                    if p.trigger_timestamp and p.response_timestamp:
                        messages.append(_MockMessage(p.trigger_text, False, p.trigger_timestamp))
                        messages.append(_MockMessage(p.response_text, True, p.response_timestamp))

            if len(messages) < min_messages:
                logger.debug(
                    "Skipping %s: only %d messages",
                    contact.display_name,
                    len(messages),
                )
                stats["profiles_skipped"] += 1
                continue

            # Build profile
            profile = build_embedding_profile(
                contact_id=str(contact.id),
                messages=messages,
                embedder=embedder,
                contact_name=contact.display_name,
            )

            # Save profile
            if save_embedding_profile(profile):
                stats["profiles_created"] += 1
                stats["total_messages_analyzed"] += profile.message_count
                logger.info(
                    "Created profile for %s (%d messages, %d clusters)",
                    contact.display_name,
                    profile.message_count,
                    len(profile.topic_clusters),
                )
            else:
                stats["errors"].append(f"Failed to save profile for {contact.display_name}")

            stats["contacts_processed"] += 1

        except Exception as e:
            logger.warning("Error processing %s: %s", contact.display_name, e)
            stats["errors"].append(f"{contact.display_name}: {str(e)}")
            stats["contacts_processed"] += 1

    return stats


@dataclass
class _MockMessage:
    """Mock message object for when we only have pairs."""

    text: str
    is_from_me: bool
    date: datetime


# =============================================================================
# Profile Usage for Reply Generation
# =============================================================================


def get_relevant_topic_cluster(
    incoming_message: str,
    profile: EmbeddingProfile,
    embedder: Any,
    threshold: float = 0.5,
) -> TopicCluster | None:
    """Find the most relevant topic cluster for an incoming message.

    Args:
        incoming_message: The message to match.
        profile: The contact's embedding profile.
        embedder: Embedding model.
        threshold: Minimum similarity to consider a match.

    Returns:
        The most similar TopicCluster, or None if below threshold.
    """
    if not profile.topic_clusters:
        return None

    # Embed the incoming message
    msg_embedding = embedder.encode([incoming_message], normalize=True)[0]

    # Find most similar cluster
    best_cluster = None
    best_similarity = threshold

    for cluster in profile.topic_clusters:
        centroid = np.array(cluster.centroid)
        similarity = float(np.dot(msg_embedding, centroid))

        if similarity > best_similarity:
            best_similarity = similarity
            best_cluster = cluster

    return best_cluster


def build_profiles_for_all_chats(
    db: Any,
    embedder: Any,
    min_pairs: int = 30,
    limit: int | None = None,
    classify_relationships: bool = True,
    include_groups: bool = True,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Build embedding profiles for ALL chat_ids in the database.

    Unlike build_profiles_for_all_contacts, this works with ANY chat
    (phone number or group chat), not just saved contacts.

    For group chats, we still build embedding profiles (topic clusters,
    communication dynamics, your response patterns) but skip relationship
    classification since groups contain multiple people.

    Args:
        db: JarvisDB instance.
        embedder: Embedding model with encode() method.
        min_pairs: Minimum pairs required for a profile.
        limit: Maximum number of chats to process.
        classify_relationships: If True, also run relationship classifier (1:1 only).
        include_groups: If True, also build profiles for group chats.
        force_rebuild: If False, skip chats that already have profiles.

    Returns:
        Statistics about the profile building.
    """
    from collections import defaultdict

    # Get all pairs grouped by chat_id
    all_pairs = db.get_training_pairs(min_quality=0.0)
    pairs_by_chat: dict[str, list] = defaultdict(list)
    group_chats: set[str] = set()

    for p in all_pairs:
        if not p.chat_id:
            continue
        if p.is_group:
            if include_groups:
                pairs_by_chat[p.chat_id].append(p)
                group_chats.add(p.chat_id)
        else:
            pairs_by_chat[p.chat_id].append(p)

    # Sort by pair count descending
    sorted_chats = sorted(
        pairs_by_chat.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )

    if limit:
        sorted_chats = sorted_chats[:limit]

    n_groups = sum(1 for chat_id, _ in sorted_chats if chat_id in group_chats)
    n_direct = len(sorted_chats) - n_groups
    logger.info(
        "Building embedding profiles for %d chats (%d direct, %d groups)",
        len(sorted_chats),
        n_direct,
        n_groups,
    )

    # Initialize relationship classifier if needed
    relationship_classifier = None
    if classify_relationships:
        try:
            from jarvis.relationship_classifier import RelationshipClassifier

            relationship_classifier = RelationshipClassifier()
            logger.info("Relationship classifier initialized")
        except Exception as e:
            logger.warning("Failed to initialize relationship classifier: %s", e)

    stats = {
        "chats_processed": 0,
        "profiles_created": 0,
        "profiles_skipped": 0,
        "profiles_skipped_existing": 0,
        "group_profiles_created": 0,
        "total_messages_analyzed": 0,
        "relationships_classified": 0,
        "errors": [],
    }

    for chat_id, pairs in sorted_chats:
        if len(pairs) < min_pairs:
            stats["profiles_skipped"] += 1
            continue

        is_group = chat_id in group_chats

        # Skip if profile already exists (unless force_rebuild)
        if not force_rebuild:
            existing = load_embedding_profile(chat_id)
            if existing and existing.message_count > 0:
                stats["profiles_skipped_existing"] += 1
                logger.debug("Skipping %s - profile already exists", chat_id[:20])
                continue

        try:
            # Convert pairs to mock messages
            messages = []
            for p in pairs:
                # Use source_timestamp if available, otherwise use a default
                timestamp = p.source_timestamp or datetime.now()
                messages.append(_MockMessage(p.trigger_text, False, timestamp))
                messages.append(_MockMessage(p.response_text, True, timestamp))

            # Build embedding profile
            profile = build_embedding_profile(
                contact_id=chat_id,
                messages=messages,
                embedder=embedder,
                contact_name=chat_id,  # Use chat_id as name
            )

            # Mark as group chat if applicable
            if is_group:
                profile.relationship_type = "group"
                profile.relationship_confidence = 1.0

            # Classify relationship type (skip for groups - can't classify a group)
            elif relationship_classifier:
                try:
                    result = relationship_classifier.classify_contact(chat_id)
                    profile.relationship_type = result.relationship
                    profile.relationship_confidence = result.confidence
                    stats["relationships_classified"] += 1
                    logger.debug(
                        "Classified %s as %s (%.0f%%)",
                        chat_id[:15],
                        result.relationship,
                        result.confidence * 100,
                    )
                except Exception as e:
                    logger.debug("Failed to classify relationship for %s: %s", chat_id, e)

            # Save profile
            if save_embedding_profile(profile):
                stats["profiles_created"] += 1
                if is_group:
                    stats["group_profiles_created"] += 1
                stats["total_messages_analyzed"] += profile.message_count
                logger.info(
                    "Created profile for %s (%d msgs, %d clusters, rel=%s)",
                    chat_id[:20],
                    profile.message_count,
                    len(profile.topic_clusters),
                    profile.relationship_type or "unknown",
                )
            else:
                stats["errors"].append(f"Failed to save profile for {chat_id}")

            stats["chats_processed"] += 1

        except Exception as e:
            logger.warning("Error processing %s: %s", chat_id, e)
            stats["errors"].append(f"{chat_id}: {str(e)}")
            stats["chats_processed"] += 1

    return stats


def generate_embedding_style_guide(profile: EmbeddingProfile) -> str:
    """Generate style guidance based on embedding analysis.

    Args:
        profile: The embedding profile.

    Returns:
        Natural language description of communication patterns.
    """
    parts = []

    # Relationship type (most important for tone)
    if profile.relationship_type:
        rel = profile.relationship_type
        if rel == "family":
            parts.append("This is a family member - use warm, caring tone")
        elif rel == "close friend":
            parts.append("This is a close friend - casual, relaxed tone is appropriate")
        elif rel == "romantic partner":
            parts.append("This is a romantic partner - affectionate tone")
        elif rel == "coworker":
            parts.append("This is a coworker - professional but friendly tone")
        elif rel == "acquaintance":
            parts.append("This is an acquaintance - polite, somewhat formal tone")
        else:
            parts.append(f"Relationship: {rel}")

    if not profile.dynamics:
        if parts:
            return ". ".join(parts) + "."
        return "Limited embedding data. Using default patterns."

    dynamics = profile.dynamics

    # Style similarity guidance
    if dynamics.style_similarity > 0.8:
        parts.append("you communicate very similarly")
    elif dynamics.style_similarity > 0.6:
        parts.append("your styles are moderately aligned")
    else:
        parts.append("you have different communication styles")

    # Initiation guidance
    if dynamics.initiation_ratio > 0.7:
        parts.append("you typically start conversations")
    elif dynamics.initiation_ratio < 0.3:
        parts.append("they usually initiate")
    else:
        parts.append("conversation initiation is balanced")

    # Topic diversity
    if dynamics.topic_diversity > 2.0:
        parts.append("you discuss a wide variety of topics")
    elif dynamics.topic_diversity < 1.0:
        parts.append("conversations tend to focus on specific topics")

    # Response shift
    if dynamics.response_semantic_shift > 0.4:
        parts.append("your responses often shift the topic")
    elif dynamics.response_semantic_shift < 0.2:
        parts.append("you tend to stay on topic in responses")

    # Topic clusters
    if profile.topic_clusters:
        top_clusters = sorted(
            profile.topic_clusters,
            key=lambda c: c.message_count,
            reverse=True,
        )[:3]

        for cluster in top_clusters:
            if cluster.sample_messages:
                sample = cluster.sample_messages[0][:50]
                parts.append(f'common topic example: "{sample}..."')

    return ". ".join(parts) + "." if parts else "No patterns detected."


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data classes
    "TopicCluster",
    "CommunicationDynamics",
    "EmbeddingProfile",
    # Profile building
    "build_embedding_profile",
    "build_profiles_for_all_contacts",
    "build_profiles_for_all_chats",
    # Storage
    "save_embedding_profile",
    "load_embedding_profile",
    "delete_embedding_profile",
    "list_embedding_profiles",
    # Usage
    "get_relevant_topic_cluster",
    "generate_embedding_style_guide",
    # Constants
    "MIN_MESSAGES_FOR_EMBEDDING_PROFILE",
]
