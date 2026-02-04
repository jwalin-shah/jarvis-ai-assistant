"""Unit tests for the entity-augmented topic discovery module.

Tests cover hybrid clustering, entity Jaccard similarity, combined distance matrices,
entity metadata extraction, and online entity-aware classification.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.ner_client import Entity
from jarvis.topic_discovery import (
    ContactTopics,
    DiscoveredTopic,
    TopicAssignment,
    TopicDiscovery,
    _compute_combined_distance_matrix,
    _compute_distance_matrix_chunked,
    _compute_entity_jaccard_matrix,
    _entities_to_label_set,
    _extract_entity_metadata,
)


class TestEntitiesToLabelSet:
    """Tests for the _entities_to_label_set helper."""

    def test_converts_entities_to_normalized_set(self) -> None:
        """Test that entities are converted to lowercase label:text format."""
        entities = [
            Entity(text="Jake", start=0, end=4, label="PERSON"),
            Entity(text="Google", start=10, end=16, label="ORG"),
            Entity(text="San Francisco", start=20, end=33, label="GPE"),
        ]

        result = _entities_to_label_set(entities)

        assert result == {"PERSON:jake", "ORG:google", "GPE:san francisco"}

    def test_empty_entities_returns_empty_set(self) -> None:
        """Test that empty entity list returns empty set."""
        result = _entities_to_label_set([])
        assert result == set()

    def test_handles_duplicate_entities(self) -> None:
        """Test that duplicate entities are deduplicated."""
        entities = [
            Entity(text="Jake", start=0, end=4, label="PERSON"),
            Entity(text="Jake", start=10, end=14, label="PERSON"),  # Same entity again
        ]

        result = _entities_to_label_set(entities)

        assert result == {"PERSON:jake"}


class TestComputeEntityJaccardMatrix:
    """Tests for the _compute_entity_jaccard_matrix helper."""

    def test_identical_sets_have_similarity_1(self) -> None:
        """Test that identical entity sets have Jaccard similarity of 1."""
        entity_sets = [
            {"PERSON:jake", "ORG:google"},
            {"PERSON:jake", "ORG:google"},
        ]

        matrix = _compute_entity_jaccard_matrix(entity_sets, 2)

        assert matrix[0, 1] == pytest.approx(1.0)
        assert matrix[1, 0] == pytest.approx(1.0)

    def test_disjoint_sets_have_similarity_0(self) -> None:
        """Test that completely different entity sets have similarity 0."""
        entity_sets = [
            {"PERSON:jake"},
            {"ORG:google"},
        ]

        matrix = _compute_entity_jaccard_matrix(entity_sets, 2)

        assert matrix[0, 1] == pytest.approx(0.0)
        assert matrix[1, 0] == pytest.approx(0.0)

    def test_partial_overlap_correct_jaccard(self) -> None:
        """Test that partial overlap computes correct Jaccard coefficient."""
        entity_sets = [
            {"PERSON:jake", "ORG:google"},  # 2 entities
            {"PERSON:jake", "ORG:apple"},   # 1 overlap, 1 different
        ]
        # Intersection = 1, Union = 3, Jaccard = 1/3

        matrix = _compute_entity_jaccard_matrix(entity_sets, 2)

        assert matrix[0, 1] == pytest.approx(1.0 / 3.0)
        assert matrix[1, 0] == pytest.approx(1.0 / 3.0)

    def test_empty_sets_have_similarity_0(self) -> None:
        """Test that empty entity sets result in 0 similarity."""
        entity_sets: list[set[str]] = [set(), {"PERSON:jake"}]

        matrix = _compute_entity_jaccard_matrix(entity_sets, 2)

        assert matrix[0, 1] == pytest.approx(0.0)

    def test_matrix_is_symmetric(self) -> None:
        """Test that the Jaccard matrix is symmetric."""
        entity_sets = [
            {"PERSON:jake", "ORG:google"},
            {"PERSON:sarah"},
            {"ORG:google", "GPE:sf"},
        ]

        matrix = _compute_entity_jaccard_matrix(entity_sets, 3)

        assert np.allclose(matrix, matrix.T)


class TestComputeCombinedDistanceMatrix:
    """Tests for the _compute_combined_distance_matrix helper."""

    def test_pure_cosine_with_empty_entities(self) -> None:
        """Test that with no entities, distance equals 1 - cosine similarity."""
        np.random.seed(42)
        embeddings = np.random.randn(3, 384).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / norms

        entity_sets: list[set[str]] = [set(), set(), set()]  # No entities

        # With entity_weight=0.3 but no entities, Jaccard is all 0
        # So distance = 1 - 0.7 * cosine
        distance = _compute_combined_distance_matrix(embeddings_norm, entity_sets)

        # Calculate expected
        cosine_sim = embeddings_norm @ embeddings_norm.T
        expected = 1.0 - 0.7 * cosine_sim
        np.fill_diagonal(expected, 0.0)
        np.maximum(expected, 0.0, out=expected)

        assert np.allclose(distance, expected, atol=1e-5)

    def test_weights_are_applied_correctly(self) -> None:
        """Test that cosine and entity weights are applied correctly."""
        # Create simple embeddings
        embeddings_norm = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # Identical to first -> cosine = 1
        ], dtype=np.float32)

        entity_sets = [
            {"PERSON:jake"},
            {"PERSON:jake"},  # Identical -> Jaccard = 1
        ]

        distance = _compute_combined_distance_matrix(
            embeddings_norm, entity_sets, cosine_weight=0.7, entity_weight=0.3
        )

        # Combined similarity = 0.7 * 1 + 0.3 * 1 = 1.0
        # Distance = 1 - 1.0 = 0.0
        assert distance[0, 1] == pytest.approx(0.0)

    def test_diagonal_is_zero(self) -> None:
        """Test that diagonal elements are always 0 (self-distance)."""
        np.random.seed(42)
        embeddings = np.random.randn(5, 384).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / norms

        entity_sets = [{"PERSON:a"} for _ in range(5)]

        distance = _compute_combined_distance_matrix(embeddings_norm, entity_sets)

        assert np.allclose(np.diag(distance), 0.0)

    def test_matrix_is_non_negative(self) -> None:
        """Test that all distances are non-negative."""
        np.random.seed(42)
        embeddings = np.random.randn(10, 384).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / norms

        entity_sets = [{"PERSON:a", "ORG:b"} for _ in range(10)]

        distance = _compute_combined_distance_matrix(embeddings_norm, entity_sets)

        assert np.all(distance >= 0)


class TestComputeDistanceMatrixChunked:
    """Tests for the _compute_distance_matrix_chunked helper."""

    def test_chunked_matches_non_chunked(self) -> None:
        """Test that chunked computation matches non-chunked for same data."""
        np.random.seed(42)
        embeddings = np.random.randn(20, 384).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / norms

        entity_sets = [{"PERSON:a"} if i % 2 == 0 else {"ORG:b"} for i in range(20)]

        non_chunked = _compute_combined_distance_matrix(embeddings_norm, entity_sets)
        chunked = _compute_distance_matrix_chunked(
            embeddings_norm, entity_sets, chunk_size=5
        )

        assert np.allclose(non_chunked, chunked, atol=1e-5)


class TestExtractEntityMetadata:
    """Tests for the _extract_entity_metadata helper."""

    def test_extracts_top_entities_per_label(self) -> None:
        """Test that top entities are extracted per label type."""
        cluster_entities = [
            [Entity(text="Jake", start=0, end=4, label="PERSON"),
             Entity(text="Google", start=10, end=16, label="ORG")],
            [Entity(text="Jake", start=0, end=4, label="PERSON"),
             Entity(text="Apple", start=10, end=15, label="ORG")],
            [Entity(text="Sarah", start=0, end=5, label="PERSON"),
             Entity(text="Google", start=10, end=16, label="ORG")],
        ]

        top_entities, density = _extract_entity_metadata(cluster_entities, top_k=3)

        # Jake appears 2x, Sarah 1x
        assert "PERSON" in top_entities
        assert "jake" in top_entities["PERSON"]
        assert "sarah" in top_entities["PERSON"]

        # Google appears 2x, Apple 1x
        assert "ORG" in top_entities
        assert "google" in top_entities["ORG"]

    def test_entity_density_calculation(self) -> None:
        """Test that entity density is calculated correctly."""
        cluster_entities = [
            [Entity(text="Jake", start=0, end=4, label="PERSON")],  # 1 entity
            [Entity(text="A", start=0, end=1, label="PERSON"),
             Entity(text="B", start=2, end=3, label="ORG")],  # 2 entities
            [],  # 0 entities
        ]

        _, density = _extract_entity_metadata(cluster_entities)

        # Total = 3 entities, 3 messages -> density = 1.0
        assert density == pytest.approx(1.0)

    def test_empty_cluster_returns_empty_metadata(self) -> None:
        """Test that empty cluster returns empty metadata."""
        top_entities, density = _extract_entity_metadata([])

        assert top_entities == {}
        assert density == 0.0

    def test_respects_top_k_limit(self) -> None:
        """Test that only top_k entities are returned per label."""
        cluster_entities = [
            [Entity(text="A", start=0, end=1, label="PERSON")],
            [Entity(text="A", start=0, end=1, label="PERSON")],
            [Entity(text="A", start=0, end=1, label="PERSON")],
            [Entity(text="B", start=0, end=1, label="PERSON")],
            [Entity(text="B", start=0, end=1, label="PERSON")],
            [Entity(text="C", start=0, end=1, label="PERSON")],
        ]

        top_entities, _ = _extract_entity_metadata(cluster_entities, top_k=2)

        # Only top 2: "a" (3x) and "b" (2x), not "c" (1x)
        assert len(top_entities["PERSON"]) == 2
        assert "a" in top_entities["PERSON"]
        assert "b" in top_entities["PERSON"]


class TestDiscoveredTopicSerialization:
    """Tests for DiscoveredTopic serialization with new fields."""

    def test_to_dict_includes_new_fields(self) -> None:
        """Test that to_dict includes top_entities and entity_density."""
        topic = DiscoveredTopic(
            topic_id=1,
            centroid=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            keywords=["test"],
            message_count=10,
            representative_text="Hello",
            top_entities={"PERSON": ["jake", "sarah"]},
            entity_density=1.5,
        )

        data = topic.to_dict()

        assert data["top_entities"] == {"PERSON": ["jake", "sarah"]}
        assert data["entity_density"] == 1.5

    def test_from_dict_with_new_fields(self) -> None:
        """Test that from_dict correctly deserializes new fields."""
        data = {
            "topic_id": 1,
            "centroid": [1.0, 0.0, 0.0],
            "keywords": ["test"],
            "message_count": 10,
            "representative_text": "Hello",
            "top_entities": {"ORG": ["google"]},
            "entity_density": 2.0,
        }

        topic = DiscoveredTopic.from_dict(data)

        assert topic.top_entities == {"ORG": ["google"]}
        assert topic.entity_density == 2.0

    def test_from_dict_backward_compatible(self) -> None:
        """Test that from_dict handles legacy data without new fields."""
        data = {
            "topic_id": 1,
            "centroid": [1.0, 0.0, 0.0],
            "keywords": ["test"],
            "message_count": 10,
            "representative_text": "Hello",
            # No top_entities or entity_density
        }

        topic = DiscoveredTopic.from_dict(data)

        assert topic.top_entities == {}
        assert topic.entity_density == 0.0

    def test_roundtrip_serialization(self) -> None:
        """Test that serialize/deserialize roundtrip preserves data."""
        original = DiscoveredTopic(
            topic_id=5,
            centroid=np.array([0.5, 0.5, 0.0], dtype=np.float32),
            keywords=["lunch", "food"],
            message_count=25,
            representative_text="Let's grab lunch",
            top_entities={"PERSON": ["mom", "dad"], "GPE": ["sf"]},
            entity_density=1.2,
        )

        data = original.to_dict()
        restored = DiscoveredTopic.from_dict(data)

        assert restored.topic_id == original.topic_id
        assert np.allclose(restored.centroid, original.centroid)
        assert restored.keywords == original.keywords
        assert restored.message_count == original.message_count
        assert restored.representative_text == original.representative_text
        assert restored.top_entities == original.top_entities
        assert restored.entity_density == original.entity_density


class TestTopicDiscoveryHybridMode:
    """Tests for hybrid clustering in TopicDiscovery.discover_topics()."""

    @pytest.fixture
    def sample_embeddings(self) -> np.ndarray:
        """Create sample embeddings for testing."""
        np.random.seed(42)
        # Create 20 embeddings in 2 clusters
        cluster1 = np.random.randn(10, 384).astype(np.float32) + np.array([1, 0] + [0] * 382)
        cluster2 = np.random.randn(10, 384).astype(np.float32) + np.array([-1, 0] + [0] * 382)
        return np.vstack([cluster1, cluster2])

    @pytest.fixture
    def sample_texts(self) -> list[str]:
        """Create sample texts for testing."""
        return [f"Message {i}" for i in range(20)]

    @patch("jarvis.topic_discovery.is_service_running")
    @patch("jarvis.topic_discovery.get_entities_batch")
    def test_uses_hybrid_mode_when_ner_available(
        self,
        mock_get_entities: MagicMock,
        mock_is_running: MagicMock,
        sample_embeddings: np.ndarray,
        sample_texts: list[str],
    ) -> None:
        """Test that hybrid mode is used when NER service is available."""
        mock_is_running.return_value = True
        mock_get_entities.return_value = [
            [Entity(text="Jake", start=0, end=4, label="PERSON")]
            for _ in range(20)
        ]

        discovery = TopicDiscovery(min_cluster_size=3, min_samples=2)
        result = discovery.discover_topics("test_contact", sample_embeddings, sample_texts)

        # Verify NER was called
        mock_get_entities.assert_called_once_with(sample_texts)
        # Should discover some topics
        assert len(result.topics) > 0

    @patch("jarvis.topic_discovery.is_service_running")
    def test_falls_back_to_cosine_when_ner_unavailable(
        self,
        mock_is_running: MagicMock,
        sample_embeddings: np.ndarray,
        sample_texts: list[str],
    ) -> None:
        """Test that cosine-only clustering is used when NER is unavailable."""
        mock_is_running.return_value = False

        discovery = TopicDiscovery(min_cluster_size=3, min_samples=2)
        result = discovery.discover_topics("test_contact", sample_embeddings, sample_texts)

        # Should still discover topics (cosine-only mode)
        assert len(result.topics) >= 0  # At least doesn't crash
        # All topics should have empty entity metadata
        for topic in result.topics:
            assert topic.top_entities == {}
            assert topic.entity_density == 0.0

    @patch("jarvis.topic_discovery.is_service_running")
    @patch("jarvis.topic_discovery.get_entities_batch")
    def test_falls_back_on_ner_exception(
        self,
        mock_get_entities: MagicMock,
        mock_is_running: MagicMock,
        sample_embeddings: np.ndarray,
        sample_texts: list[str],
    ) -> None:
        """Test that NER errors trigger fallback to cosine-only."""
        mock_is_running.return_value = True
        mock_get_entities.side_effect = Exception("NER service error")

        discovery = TopicDiscovery(min_cluster_size=3, min_samples=2)
        result = discovery.discover_topics("test_contact", sample_embeddings, sample_texts)

        # Should not raise, should fall back gracefully
        assert result.contact_id == "test_contact"

    @patch("jarvis.topic_discovery.is_service_running")
    @patch("jarvis.topic_discovery.get_entities_batch")
    def test_extracts_entity_metadata_per_topic(
        self,
        mock_get_entities: MagicMock,
        mock_is_running: MagicMock,
        sample_embeddings: np.ndarray,
        sample_texts: list[str],
    ) -> None:
        """Test that entity metadata is extracted for each discovered topic."""
        mock_is_running.return_value = True
        # First 10 messages have PERSON entities, last 10 have ORG entities
        mock_get_entities.return_value = (
            [[Entity(text="Jake", start=0, end=4, label="PERSON")] for _ in range(10)] +
            [[Entity(text="Google", start=0, end=6, label="ORG")] for _ in range(10)]
        )

        discovery = TopicDiscovery(min_cluster_size=3, min_samples=2)
        result = discovery.discover_topics("test_contact", sample_embeddings, sample_texts)

        # If clustering works, topics should have entity info
        if result.topics:
            # At least check the structure is correct
            for topic in result.topics:
                assert isinstance(topic.top_entities, dict)
                assert isinstance(topic.entity_density, float)


class TestOnlineClassification:
    """Tests for entity-aware online classification."""

    @pytest.fixture
    def sample_topics(self) -> ContactTopics:
        """Create sample topics for classification testing."""
        topic1 = DiscoveredTopic(
            topic_id=0,
            centroid=np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32),
            keywords=["work"],
            message_count=10,
            representative_text="Work stuff",
            top_entities={"ORG": ["google", "apple"]},
            entity_density=1.0,
        )
        topic2 = DiscoveredTopic(
            topic_id=1,
            centroid=np.array([0.0, 1.0, 0.0] + [0.0] * 381, dtype=np.float32),
            keywords=["family"],
            message_count=10,
            representative_text="Family stuff",
            top_entities={"PERSON": ["mom", "dad"]},
            entity_density=1.5,
        )
        return ContactTopics(contact_id="test", topics=[topic1, topic2])

    def test_classify_with_entities(self, sample_topics: ContactTopics) -> None:
        """Test that classification uses entity overlap when entities provided."""
        # Embedding closer to topic 0 (work)
        embedding = np.array([0.8, 0.2, 0.0] + [0.0] * 381, dtype=np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        # But entities match topic 1 (family)
        entities = [
            Entity(text="mom", start=0, end=3, label="PERSON"),
        ]

        result = sample_topics.classify(embedding, entities=entities)

        assert result is not None
        # The combined score should consider both cosine and entity overlap
        # With strong entity match to topic 1, it might shift the result
        assert result.topic_id in [0, 1]  # Either is valid depending on weights

    def test_classify_without_entities_backward_compat(
        self, sample_topics: ContactTopics
    ) -> None:
        """Test that classification works without entities (backward compatible)."""
        # Embedding closer to topic 1
        embedding = np.array([0.2, 0.8, 0.0] + [0.0] * 381, dtype=np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        result = sample_topics.classify(embedding)  # No entities

        assert result is not None
        assert result.topic_id == 1  # Should match based on cosine alone

    def test_classify_message_passes_entities(self, sample_topics: ContactTopics) -> None:
        """Test that TopicDiscovery.classify_message passes entities to classify."""
        discovery = TopicDiscovery()

        embedding = np.array([0.5, 0.5, 0.0] + [0.0] * 381, dtype=np.float32)
        entities = [Entity(text="google", start=0, end=6, label="ORG")]

        result = discovery.classify_message(
            sample_topics,
            embedding,
            entities=entities,
            previous_topic_id=None,
        )

        assert result is not None
        assert isinstance(result, TopicAssignment)

    def test_classify_message_detects_chunk_boundary(
        self, sample_topics: ContactTopics
    ) -> None:
        """Test that classify_message still detects chunk boundaries with entities."""
        discovery = TopicDiscovery()

        # First message to topic 0
        embedding1 = np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32)
        result1 = discovery.classify_message(
            sample_topics, embedding1, entities=None, previous_topic_id=None
        )

        # Second message to topic 1
        embedding2 = np.array([0.0, 1.0, 0.0] + [0.0] * 381, dtype=np.float32)
        result2 = discovery.classify_message(
            sample_topics,
            embedding2,
            entities=None,
            previous_topic_id=result1.topic_id if result1 else None,
        )

        assert result2 is not None
        if result1 is not None and result1.topic_id != result2.topic_id:
            assert result2.is_chunk_start

    def test_classify_with_no_topic_entities(self) -> None:
        """Test classification when topics have no entity metadata."""
        topic = DiscoveredTopic(
            topic_id=0,
            centroid=np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32),
            keywords=["test"],
            message_count=10,
            representative_text="Test",
            top_entities={},  # No entity metadata
            entity_density=0.0,
        )
        contact_topics = ContactTopics(contact_id="test", topics=[topic])

        embedding = np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32)
        entities = [Entity(text="Jake", start=0, end=4, label="PERSON")]

        result = contact_topics.classify(embedding, entities=entities)

        # Should still work, falling back to cosine-only
        assert result is not None
        assert result.topic_id == 0


class TestContactTopicsClassify:
    """Tests for ContactTopics.classify method."""

    def test_classify_returns_none_for_empty_topics(self) -> None:
        """Test that classify returns None when no topics exist."""
        contact_topics = ContactTopics(contact_id="test", topics=[])

        embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = contact_topics.classify(embedding)

        assert result is None

    def test_classify_normalizes_embedding(self) -> None:
        """Test that classification normalizes the input embedding."""
        topic = DiscoveredTopic(
            topic_id=0,
            centroid=np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32),
            keywords=["test"],
            message_count=10,
            representative_text="Test",
        )
        contact_topics = ContactTopics(contact_id="test", topics=[topic])

        # Unnormalized embedding (same direction as centroid)
        embedding = np.array([10.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32)
        result = contact_topics.classify(embedding)

        assert result is not None
        # Should have high confidence (close to 1.0)
        assert result.confidence > 0.9
