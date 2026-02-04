"""Topic Discovery - Learn topics per contact from their conversation history.

Discovers topics dynamically per contact using embedding clustering,
then uses those topics for real-time message classification and chunk detection.

Flow:
    OFFLINE (once per contact):
        1. Get all messages for contact
        2. Use existing embeddings from DB
        3. Cluster with HDBSCAN → discover N topics
        4. Compute centroids + extract keywords
        5. Save to DB

    ONLINE (real-time):
        1. Embed new message (via MLX server)
        2. Cosine similarity to contact's centroids
        3. Assign topic → detect if topic changed → chunk boundary
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from jarvis.ner_client import Entity, get_entities_batch, is_service_running

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# Default weights for hybrid similarity scoring
DEFAULT_COSINE_WEIGHT: float = 0.7
DEFAULT_ENTITY_WEIGHT: float = 0.3


@dataclass
class DiscoveredTopic:
    """A topic discovered from a contact's conversation history."""

    topic_id: int
    centroid: NDArray[np.float32]  # 384-dim embedding
    keywords: list[str]  # Representative words/phrases
    message_count: int  # How many messages in this topic
    representative_text: str  # Most representative message (for labeling)
    top_entities: dict[str, list[str]] = field(default_factory=dict)  # {"PERSON": ["Jake"]}
    entity_density: float = 0.0  # Entities per message in cluster

    def to_dict(self) -> dict[str, Any]:
        """Serialize for DB storage."""
        return {
            "topic_id": self.topic_id,
            "centroid": self.centroid.tolist(),
            "keywords": self.keywords,
            "message_count": self.message_count,
            "representative_text": self.representative_text,
            "top_entities": self.top_entities,
            "entity_density": self.entity_density,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscoveredTopic:
        """Deserialize from DB."""
        return cls(
            topic_id=data["topic_id"],
            centroid=np.array(data["centroid"], dtype=np.float32),
            keywords=data["keywords"],
            message_count=data["message_count"],
            representative_text=data["representative_text"],
            top_entities=data.get("top_entities", {}),
            entity_density=data.get("entity_density", 0.0),
        )


@dataclass
class TopicAssignment:
    """Topic assignment for a single message."""

    topic_id: int
    confidence: float  # Cosine similarity to centroid
    is_chunk_start: bool  # True if topic changed from previous message


@dataclass
class ContactTopics:
    """All discovered topics for a contact."""

    contact_id: str
    topics: list[DiscoveredTopic] = field(default_factory=list)
    noise_count: int = 0  # Messages that didn't cluster

    def get_centroid_matrix(self) -> NDArray[np.float32]:
        """Get all centroids as a matrix for fast batch classification."""
        if not self.topics:
            return np.array([], dtype=np.float32)
        return np.vstack([t.centroid for t in self.topics])

    def classify(
        self,
        embedding: NDArray[np.float32],
        entities: list[Entity] | None = None,
        cosine_weight: float = DEFAULT_COSINE_WEIGHT,
        entity_weight: float = DEFAULT_ENTITY_WEIGHT,
    ) -> TopicAssignment | None:
        """Classify a single message embedding to a topic.

        When entities are provided, uses hybrid scoring:
        combined_score = cosine_weight * cosine_sim + entity_weight * entity_overlap

        Args:
            embedding: Message embedding (384-dim).
            entities: Optional list of entities extracted from the message.
            cosine_weight: Weight for cosine similarity (default DEFAULT_COSINE_WEIGHT).
            entity_weight: Weight for entity overlap (default DEFAULT_ENTITY_WEIGHT).

        Returns:
            TopicAssignment with topic and confidence.
        """
        if not self.topics:
            return None

        # Normalize input
        embedding = embedding / np.linalg.norm(embedding)

        # Compute cosine similarities to all centroids
        centroids = self.get_centroid_matrix()
        # Centroids should already be normalized, but ensure it
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids_norm = centroids / norms

        cosine_similarities = centroids_norm @ embedding

        if entities is not None and any(t.top_entities for t in self.topics):
            # Entity-aware classification
            message_entity_set = _entities_to_label_set(entities)
            entity_overlaps = np.zeros(len(self.topics), dtype=np.float32)

            for i, topic in enumerate(self.topics):
                if topic.top_entities and message_entity_set:
                    # Build topic entity set from top_entities (with token expansion)
                    topic_entity_set: set[str] = set()
                    for label, texts in topic.top_entities.items():
                        for text in texts:
                            topic_entity_set.add(f"{label}:{text}")
                            # Also add tokens for partial matching
                            for token in text.split():
                                if len(token) > 2:
                                    topic_entity_set.add(f"{label}:{token}")

                    # Overlap coefficient (better than Jaccard for size differences)
                    entity_overlaps[i] = _compute_overlap_coefficient(
                        message_entity_set, topic_entity_set
                    )

            # Combined score
            combined_scores = (
                cosine_weight * cosine_similarities + entity_weight * entity_overlaps
            )
            best_idx = int(np.argmax(combined_scores))
            best_score = float(combined_scores[best_idx])
        else:
            # Cosine-only classification (fallback)
            best_idx = int(np.argmax(cosine_similarities))
            best_score = float(cosine_similarities[best_idx])

        return TopicAssignment(
            topic_id=self.topics[best_idx].topic_id,
            confidence=best_score,
            is_chunk_start=False,  # Caller sets this based on previous
        )


# --- Entity-based helper functions ---


def _entities_to_label_set(entities: list[Entity]) -> set[str]:
    """Convert entities to normalized label set for overlap similarity.

    Includes both full entity text and individual tokens (>2 chars) for
    fuzzy matching. This helps match "Jake Smith" with "Jake".

    Args:
        entities: List of Entity objects.

    Returns:
        Set of strings like {"PERSON:jake", "PERSON:jake smith", "ORG:google"}.
    """
    result: set[str] = set()
    for e in entities:
        # Add full name
        result.add(f"{e.label}:{e.text.lower()}")
        # Also add individual tokens for partial matching
        for token in e.text.lower().split():
            if len(token) > 2:
                result.add(f"{e.label}:{token}")
    return result


def _compute_overlap_coefficient(set_a: set[str], set_b: set[str]) -> float:
    """Compute overlap coefficient between two sets.

    overlap = |A ∩ B| / min(|A|, |B|)

    This is better than Jaccard when one set is a subset of another
    (e.g., message has 2 entities, topic has 10).

    Args:
        set_a: First set.
        set_b: Second set.

    Returns:
        Overlap coefficient (0.0 to 1.0).
    """
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    min_size = min(len(set_a), len(set_b))
    return intersection / min_size if min_size > 0 else 0.0


def _compute_entity_overlap_matrix(
    entity_sets: list[set[str]],
    n: int,
) -> NDArray[np.float32]:
    """Compute NxN overlap coefficient matrix from entity sets.

    Uses overlap coefficient (intersection / min size) instead of Jaccard
    because it handles size differences better.

    Optimized with early termination for empty sets.

    Args:
        entity_sets: List of entity label sets per message.
        n: Number of messages.

    Returns:
        NxN similarity matrix (float32).
    """
    matrix = np.zeros((n, n), dtype=np.float32)

    # Pre-compute set sizes for early termination
    sizes = np.array([len(s) for s in entity_sets], dtype=np.int32)

    # Find indices of non-empty sets for efficient iteration
    non_empty_indices = np.where(sizes > 0)[0]

    # Only iterate over non-empty sets
    for idx_i, i in enumerate(non_empty_indices):
        set_i = entity_sets[i]
        size_i = sizes[i]
        for j in non_empty_indices[idx_i + 1:]:
            set_j = entity_sets[j]
            intersection = len(set_i & set_j)
            if intersection > 0:
                min_size = min(size_i, sizes[j])
                overlap = intersection / min_size
                matrix[i, j] = overlap
                matrix[j, i] = overlap

    return matrix


def _compute_entity_overlap_chunk(
    entity_sets: list[set[str]],
    i_start: int,
    i_end: int,
    j_start: int,
    j_end: int,
) -> NDArray[np.float32]:
    """Compute a chunk of the overlap matrix without pre-computing the full matrix.

    Args:
        entity_sets: Full list of entity label sets per message.
        i_start: Row start index.
        i_end: Row end index.
        j_start: Column start index.
        j_end: Column end index.

    Returns:
        Chunk of the similarity matrix (i_end - i_start, j_end - j_start).
    """
    chunk_rows = i_end - i_start
    chunk_cols = j_end - j_start
    chunk = np.zeros((chunk_rows, chunk_cols), dtype=np.float32)

    for local_i, global_i in enumerate(range(i_start, i_end)):
        set_i = entity_sets[global_i]
        if not set_i:
            continue
        size_i = len(set_i)
        for local_j, global_j in enumerate(range(j_start, j_end)):
            if global_i == global_j:
                continue  # Skip diagonal
            set_j = entity_sets[global_j]
            if not set_j:
                continue
            intersection = len(set_i & set_j)
            if intersection > 0:
                min_size = min(size_i, len(set_j))
                chunk[local_i, local_j] = intersection / min_size

    return chunk


# Keep old name as alias for backward compatibility with tests
def _compute_entity_jaccard_matrix(
    entity_sets: list[set[str]],
    n: int,
) -> NDArray[np.float32]:
    """Compute NxN overlap coefficient matrix from entity sets.

    Note: This function now uses overlap coefficient instead of Jaccard
    for better handling of size differences. The name is kept for
    backward compatibility.

    Args:
        entity_sets: List of entity label sets per message.
        n: Number of messages.

    Returns:
        NxN similarity matrix (float32).
    """
    return _compute_entity_overlap_matrix(entity_sets, n)


def _compute_combined_distance_matrix(
    embeddings_norm: NDArray[np.float32],
    entity_sets: list[set[str]],
    cosine_weight: float = DEFAULT_COSINE_WEIGHT,
    entity_weight: float = DEFAULT_ENTITY_WEIGHT,
) -> NDArray[np.float32]:
    """Compute combined distance matrix: 1 - (cosine_weight*cosine + entity_weight*overlap).

    Args:
        embeddings_norm: Normalized embeddings (N, D).
        entity_sets: Entity label sets per message.
        cosine_weight: Weight for cosine similarity (default DEFAULT_COSINE_WEIGHT).
        entity_weight: Weight for entity overlap similarity (default DEFAULT_ENTITY_WEIGHT).

    Returns:
        NxN distance matrix for HDBSCAN precomputed metric.
    """
    n = len(embeddings_norm)

    # Cosine similarity matrix
    cosine_sim = embeddings_norm @ embeddings_norm.T

    # Entity overlap similarity matrix
    overlap_sim = _compute_entity_overlap_matrix(entity_sets, n)

    # Combined similarity and convert to distance
    combined_sim = cosine_weight * cosine_sim + entity_weight * overlap_sim
    distance_matrix = 1.0 - combined_sim

    # Ensure diagonal is 0 and matrix is non-negative
    np.fill_diagonal(distance_matrix, 0.0)
    np.maximum(distance_matrix, 0.0, out=distance_matrix)

    return distance_matrix.astype(np.float32)


def _compute_distance_matrix_chunked(
    embeddings_norm: NDArray[np.float32],
    entity_sets: list[set[str]],
    cosine_weight: float = DEFAULT_COSINE_WEIGHT,
    entity_weight: float = DEFAULT_ENTITY_WEIGHT,
    chunk_size: int = 2000,
) -> NDArray[np.float32]:
    """Compute combined distance matrix in chunks for memory efficiency.

    For n > 5000, full NxN matrix can exceed 400MB. This computes both
    cosine and overlap similarity per-chunk to avoid pre-computing full matrices.

    Args:
        embeddings_norm: Normalized embeddings (N, D).
        entity_sets: Entity label sets per message.
        cosine_weight: Weight for cosine similarity (default DEFAULT_COSINE_WEIGHT).
        entity_weight: Weight for entity overlap similarity (default DEFAULT_ENTITY_WEIGHT).
        chunk_size: Size of chunks to process.

    Returns:
        NxN distance matrix for HDBSCAN precomputed metric.
    """
    n = len(embeddings_norm)
    distance_matrix = np.zeros((n, n), dtype=np.float32)

    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        for j_start in range(0, n, chunk_size):
            j_end = min(j_start + chunk_size, n)

            # Cosine similarity for chunk
            cosine_chunk = embeddings_norm[i_start:i_end] @ embeddings_norm[j_start:j_end].T

            # Compute overlap similarity for this chunk only (not full matrix)
            overlap_chunk = _compute_entity_overlap_chunk(
                entity_sets, i_start, i_end, j_start, j_end
            )

            # Combined distance
            combined_sim = cosine_weight * cosine_chunk + entity_weight * overlap_chunk
            distance_matrix[i_start:i_end, j_start:j_end] = 1.0 - combined_sim

    # Ensure diagonal is 0 and matrix is non-negative
    np.fill_diagonal(distance_matrix, 0.0)
    np.maximum(distance_matrix, 0.0, out=distance_matrix)

    return distance_matrix


def _extract_entity_metadata(
    cluster_entities: list[list[Entity]],
    top_k: int = 5,
) -> tuple[dict[str, list[str]], float]:
    """Extract top entities per label type and entity density for a cluster.

    Args:
        cluster_entities: List of entity lists, one per message in cluster.
        top_k: Number of top entities to keep per label type.

    Returns:
        Tuple of (top_entities dict, entity_density).
    """
    from collections import Counter

    label_counts: dict[str, Counter[str]] = {}

    total_entities = 0
    for entities in cluster_entities:
        total_entities += len(entities)
        for e in entities:
            if e.label not in label_counts:
                label_counts[e.label] = Counter()
            label_counts[e.label][e.text.lower()] += 1

    top_entities: dict[str, list[str]] = {}
    for label, counts in label_counts.items():
        top_entities[label] = [text for text, _ in counts.most_common(top_k)]

    entity_density = total_entities / len(cluster_entities) if cluster_entities else 0.0

    return top_entities, entity_density


class TopicDiscovery:
    """Discovers and manages topics per contact."""

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int = 3,
        max_topics: int = 20,
        noise_threshold: float = 0.3,
    ):
        """Initialize topic discovery.

        Args:
            min_cluster_size: Minimum messages to form a topic
            min_samples: HDBSCAN min_samples parameter
            max_topics: Maximum topics per contact (prevents over-fragmentation)
            noise_threshold: If >this fraction is noise, reduce min_cluster_size
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.max_topics = max_topics
        self.noise_threshold = noise_threshold

    def _compute_adaptive_cluster_size(self, n_messages: int) -> int:
        """Compute adaptive min_cluster_size based on message count.

        Larger conversations need larger minimum clusters to avoid
        over-fragmentation, while smaller conversations need smaller
        clusters to capture subtopics.

        Args:
            n_messages: Number of messages for the contact.

        Returns:
            Adapted min_cluster_size.
        """
        if n_messages < 50:
            return 3
        elif n_messages < 200:
            return 5
        elif n_messages < 500:
            return 8
        else:
            return min(15, n_messages // 50)

    def discover_topics(
        self,
        contact_id: str,
        embeddings: NDArray[np.float32],
        texts: list[str],
    ) -> ContactTopics:
        """Discover topics from a contact's message embeddings.

        Uses hybrid clustering when NER service is available:
        combined_similarity = 0.7 * cosine + 0.3 * entity_jaccard

        Args:
            contact_id: The contact identifier
            embeddings: (N, 384) array of message embeddings
            texts: Corresponding message texts (for keyword extraction)

        Returns:
            ContactTopics with discovered topics and centroids
        """
        # Use adaptive cluster size based on conversation volume
        adaptive_min_cluster = self._compute_adaptive_cluster_size(len(embeddings))
        effective_min_cluster = min(self.min_cluster_size, adaptive_min_cluster)

        if len(embeddings) < effective_min_cluster:
            logger.debug(
                f"Contact {contact_id}: only {len(embeddings)} messages, "
                f"need {effective_min_cluster} for clustering"
            )
            return ContactTopics(contact_id=contact_id, noise_count=len(embeddings))

        # Normalize embeddings for cosine distance
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        embeddings_norm = embeddings / norms

        n = len(embeddings)

        # Try to get entities for hybrid clustering
        entities_per_message: list[list[Entity]] | None = None
        entity_sets: list[set[str]] | None = None
        use_hybrid = False

        if is_service_running():
            try:
                entities_per_message = get_entities_batch(texts)
                entity_sets = [_entities_to_label_set(ents) for ents in entities_per_message]
                use_hybrid = True
                logger.debug(f"Contact {contact_id}: using hybrid clustering with entities")
            except Exception as e:
                logger.warning(f"NER extraction failed, falling back to cosine-only: {e}")

        # Cluster with HDBSCAN
        from sklearn.cluster import HDBSCAN

        if use_hybrid and entity_sets is not None:
            # Compute combined distance matrix
            if n > 5000:
                distance_matrix = _compute_distance_matrix_chunked(
                    embeddings_norm, entity_sets
                )
            else:
                distance_matrix = _compute_combined_distance_matrix(
                    embeddings_norm, entity_sets
                )

            clusterer = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric="precomputed",
                cluster_selection_method="eom",
            )
            labels = clusterer.fit_predict(distance_matrix)
        else:
            # Fallback: cosine-only clustering
            clusterer = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric="cosine",
                cluster_selection_method="eom",
            )
            labels = clusterer.fit_predict(embeddings_norm)

        # Check noise ratio
        noise_mask = labels == -1
        noise_ratio = noise_mask.sum() / len(labels)

        if noise_ratio > self.noise_threshold and self.min_cluster_size > 3:
            # Too much noise - retry with smaller clusters
            logger.debug(
                f"Contact {contact_id}: {noise_ratio:.0%} noise, "
                f"retrying with smaller clusters"
            )
            if use_hybrid and entity_sets is not None:
                clusterer = HDBSCAN(
                    min_cluster_size=max(3, self.min_cluster_size - 2),
                    min_samples=max(2, self.min_samples - 1),
                    metric="precomputed",
                    cluster_selection_method="eom",
                )
                labels = clusterer.fit_predict(distance_matrix)
            else:
                clusterer = HDBSCAN(
                    min_cluster_size=max(3, self.min_cluster_size - 2),
                    min_samples=max(2, self.min_samples - 1),
                    metric="cosine",
                    cluster_selection_method="eom",
                )
                labels = clusterer.fit_predict(embeddings_norm)
            noise_mask = labels == -1

        # Extract topics
        unique_labels = set(labels) - {-1}

        # Limit to max_topics (keep largest clusters)
        if len(unique_labels) > self.max_topics:
            label_counts = {
                label: int((labels == label).sum())
                for label in unique_labels
            }
            sorted_labels = sorted(
                label_counts, key=lambda x: label_counts[x], reverse=True
            )
            unique_labels = set(sorted_labels[: self.max_topics])

        topics = []
        for topic_idx, label in enumerate(sorted(unique_labels)):
            mask = labels == label
            cluster_embeddings = embeddings_norm[mask]
            cluster_texts = [t for t, m in zip(texts, mask) if m]

            # Compute centroid (normalized)
            centroid = cluster_embeddings.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)

            # Find most representative message (closest to centroid)
            similarities = cluster_embeddings @ centroid
            rep_idx = int(np.argmax(similarities))
            representative = cluster_texts[rep_idx][:100]  # Truncate for storage

            # Extract keywords (simple: most common words)
            keywords = self._extract_keywords(cluster_texts, top_k=5)

            # Extract entity metadata if available
            top_entities: dict[str, list[str]] = {}
            entity_density = 0.0
            if entities_per_message is not None:
                cluster_entities = [e for e, m in zip(entities_per_message, mask) if m]
                top_entities, entity_density = _extract_entity_metadata(cluster_entities)

            topics.append(
                DiscoveredTopic(
                    topic_id=topic_idx,
                    centroid=centroid.astype(np.float32),
                    keywords=keywords,
                    message_count=int(mask.sum()),
                    representative_text=representative,
                    top_entities=top_entities,
                    entity_density=entity_density,
                )
            )

        logger.info(
            f"Contact {contact_id}: discovered {len(topics)} topics "
            f"from {len(embeddings)} messages ({noise_mask.sum()} noise)"
            + (", hybrid mode" if use_hybrid else "")
        )

        return ContactTopics(
            contact_id=contact_id,
            topics=topics,
            noise_count=int(noise_mask.sum()),
        )

    def _extract_keywords(self, texts: list[str], top_k: int = 5) -> list[str]:
        """Extract representative keywords from a cluster of texts."""
        import re
        from collections import Counter

        # Simple word frequency (could use TF-IDF for better results)
        word_counts: Counter[str] = Counter()
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "i", "you", "he", "she", "it", "we", "they", "me", "him",
            "her", "us", "them", "my", "your", "his", "its", "our",
            "their", "this", "that", "these", "those", "and", "or",
            "but", "if", "then", "so", "than", "too", "very", "just",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "about", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "again", "further",
            "once", "here", "there", "when", "where", "why", "how",
            "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "than",
            "what", "which", "who", "whom", "lol", "lmao", "haha",
            "yeah", "yea", "ya", "ok", "okay", "like", "gonna", "wanna",
            "gotta", "kinda", "sorta", "im", "dont", "cant", "wont",
        }

        for text in texts:
            words = re.findall(r"\b[a-z]{3,}\b", text.lower())
            for word in words:
                if word not in stopwords:
                    word_counts[word] += 1

        return [word for word, _ in word_counts.most_common(top_k)]

    def classify_message(
        self,
        contact_topics: ContactTopics,
        embedding: NDArray[np.float32],
        entities: list[Entity] | None = None,
        previous_topic_id: int | None = None,
    ) -> TopicAssignment | None:
        """Classify a message and detect chunk boundaries.

        Args:
            contact_topics: The contact's discovered topics
            embedding: Message embedding (384-dim)
            entities: Optional entities extracted from the message (for hybrid scoring)
            previous_topic_id: Topic of the previous message (for chunk detection)

        Returns:
            TopicAssignment with topic and chunk boundary flag
        """
        assignment = contact_topics.classify(embedding, entities=entities)
        if assignment is None:
            return None

        # Detect chunk boundary (topic changed)
        if previous_topic_id is not None:
            assignment.is_chunk_start = assignment.topic_id != previous_topic_id

        return assignment

    def classify_conversation(
        self,
        contact_topics: ContactTopics,
        embeddings: NDArray[np.float32],
        min_confidence: float = 0.3,
        entities_list: list[list[Entity]] | None = None,
    ) -> list[TopicAssignment]:
        """Classify a sequence of messages and detect chunk boundaries.

        Args:
            contact_topics: The contact's discovered topics
            embeddings: (N, 384) array of message embeddings
            min_confidence: Minimum confidence to assign a topic
            entities_list: Optional list of entities per message for hybrid scoring.
                           Must have same length as embeddings if provided.

        Returns:
            List of TopicAssignments with chunk boundaries marked
        """
        if not contact_topics.topics or len(embeddings) == 0:
            return []

        # Validate entities_list length if provided
        if entities_list is not None and len(entities_list) != len(embeddings):
            logger.warning(
                f"entities_list length ({len(entities_list)}) != embeddings length "
                f"({len(embeddings)}), ignoring entities"
            )
            entities_list = None

        assignments = []
        previous_topic_id: int | None = None

        for i, emb in enumerate(embeddings):
            entities = entities_list[i] if entities_list is not None else None
            assignment = self.classify_message(
                contact_topics, emb, entities=entities, previous_topic_id=previous_topic_id
            )

            if assignment is None:
                continue

            # First message is always a chunk start
            if i == 0:
                assignment.is_chunk_start = True

            # Low confidence = possible topic shift (treat as chunk boundary)
            if assignment.confidence < min_confidence:
                assignment.is_chunk_start = True

            assignments.append(assignment)
            previous_topic_id = assignment.topic_id

        return assignments


# Singleton instance
_discovery: TopicDiscovery | None = None


def get_topic_discovery() -> TopicDiscovery:
    """Get the singleton TopicDiscovery instance."""
    global _discovery
    if _discovery is None:
        _discovery = TopicDiscovery()
    return _discovery
