"""Tests for topic_segmenter module."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.ner_client import Entity
from jarvis.topic_segmenter import (
    SegmentBoundary,
    SegmentBoundaryReason,
    SegmentMessage,
    TopicSegment,
    TopicSegmenter,
    _aggregate_entities,
    _compute_jaccard,
    _entities_to_label_set,
    reset_segmenter,
    segment_conversation,
    segment_for_extraction,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_message():
    """Factory for creating mock Message objects."""

    def _make_message(
        text: str,
        is_from_me: bool = False,
        date: datetime | None = None,
        msg_id: int = 1,
    ):
        msg = MagicMock()
        msg.text = text
        msg.is_from_me = is_from_me
        msg.date = date or datetime.now()
        msg.id = msg_id
        return msg

    return _make_message


@pytest.fixture
def mock_embedder():
    """Create a mock embedder that returns consistent embeddings."""
    embedder = MagicMock()
    embedder.encode.side_effect = lambda texts, normalize=True: np.random.randn(
        len(texts), 384
    ).astype(np.float32)
    return embedder


@pytest.fixture
def sample_entities():
    """Sample entities for testing."""
    return [
        Entity(text="Jake", start=0, end=4, label="PERSON"),
        Entity(text="San Francisco", start=20, end=33, label="GPE"),
    ]


@pytest.fixture
def conversation_messages(mock_message):
    """Create a sample conversation with clear topic shifts."""
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    return [
        # Topic 1: Lunch plans
        mock_message("Hey, want to grab lunch?", False, base_time, 1),
        mock_message("Sure, where?", True, base_time + timedelta(minutes=1), 2),
        mock_message("How about sushi?", False, base_time + timedelta(minutes=2), 3),
        mock_message("Sounds good!", True, base_time + timedelta(minutes=3), 4),
        # Topic 2: Weekend plans (topic shift marker)
        mock_message("btw, what are you doing this weekend?", False, base_time + timedelta(minutes=5), 5),
        mock_message("Not sure yet", True, base_time + timedelta(minutes=6), 6),
        mock_message("There's a concert on Saturday", False, base_time + timedelta(minutes=7), 7),
        # Topic 3: After large time gap
        mock_message("Hey how did the concert go?", False, base_time + timedelta(hours=5), 8),
        mock_message("It was amazing!", True, base_time + timedelta(hours=5, minutes=1), 9),
    ]


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestEntitiesHelpers:
    """Test entity helper functions."""

    def test_entities_to_label_set_basic(self, sample_entities):
        """Test converting entities to label set."""
        result = _entities_to_label_set(sample_entities)

        assert "PERSON:jake" in result
        assert "GPE:san francisco" in result
        # Tokens should also be included
        assert "GPE:san" in result
        assert "GPE:francisco" in result

    def test_entities_to_label_set_empty(self):
        """Test with empty entity list."""
        result = _entities_to_label_set([])
        assert result == set()

    def test_entities_to_label_set_short_tokens_excluded(self):
        """Test that short tokens (<= 2 chars) are excluded."""
        entities = [Entity(text="A B", start=0, end=3, label="PERSON")]
        result = _entities_to_label_set(entities)

        # "A" and "B" should NOT be in result (too short)
        assert "PERSON:a" not in result
        assert "PERSON:b" not in result
        # But full text should be
        assert "PERSON:a b" in result

    def test_compute_jaccard_basic(self):
        """Test Jaccard similarity computation."""
        set_a = {"a", "b", "c"}
        set_b = {"b", "c", "d"}
        # Intersection: {b, c} = 2, Union: {a, b, c, d} = 4
        assert _compute_jaccard(set_a, set_b) == 0.5

    def test_compute_jaccard_identical(self):
        """Test Jaccard with identical sets."""
        set_a = {"a", "b", "c"}
        assert _compute_jaccard(set_a, set_a) == 1.0

    def test_compute_jaccard_disjoint(self):
        """Test Jaccard with disjoint sets."""
        set_a = {"a", "b"}
        set_b = {"c", "d"}
        assert _compute_jaccard(set_a, set_b) == 0.0

    def test_compute_jaccard_empty(self):
        """Test Jaccard with empty sets."""
        assert _compute_jaccard(set(), set()) == 0.0
        assert _compute_jaccard({"a"}, set()) == 0.0

    def test_aggregate_entities(self, sample_entities):
        """Test entity aggregation from messages."""
        messages = [
            SegmentMessage(
                text="test",
                timestamp=datetime.now(),
                is_from_me=False,
                entities=sample_entities,
            ),
            SegmentMessage(
                text="test2",
                timestamp=datetime.now(),
                is_from_me=True,
                entities=[Entity(text="Jake", start=0, end=4, label="PERSON")],
            ),
        ]

        result = _aggregate_entities(messages)

        assert "PERSON" in result
        assert "jake" in result["PERSON"]  # Lowercase
        assert "GPE" in result


# =============================================================================
# Test Data Classes
# =============================================================================


class TestSegmentMessage:
    """Test SegmentMessage dataclass."""

    def test_segment_message_creation(self):
        """Test creating a SegmentMessage."""
        msg = SegmentMessage(
            text="Hello",
            timestamp=datetime(2024, 1, 1),
            is_from_me=False,
        )
        assert msg.text == "Hello"
        assert msg.embedding is None
        assert msg.entities == []

    def test_segment_message_with_embedding(self):
        """Test SegmentMessage with embedding."""
        embedding = np.random.randn(384).astype(np.float32)
        msg = SegmentMessage(
            text="Hello",
            timestamp=datetime(2024, 1, 1),
            is_from_me=False,
            embedding=embedding,
        )
        assert msg.embedding is not None
        assert msg.embedding.shape == (384,)


class TestTopicSegment:
    """Test TopicSegment dataclass."""

    def test_topic_segment_text_property(self):
        """Test combined text property."""
        messages = [
            SegmentMessage(text="Hello", timestamp=datetime.now(), is_from_me=False),
            SegmentMessage(text="World", timestamp=datetime.now(), is_from_me=True),
        ]
        segment = TopicSegment(
            segment_id="test",
            messages=messages,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 1, 1),
        )

        assert segment.text == "Hello\nWorld"

    def test_topic_segment_message_count(self):
        """Test message count property."""
        messages = [
            SegmentMessage(text="Hello", timestamp=datetime.now(), is_from_me=False),
            SegmentMessage(text="World", timestamp=datetime.now(), is_from_me=True),
        ]
        segment = TopicSegment(
            segment_id="test",
            messages=messages,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 1, 1),
        )

        assert segment.message_count == 2

    def test_topic_segment_duration(self):
        """Test duration property."""
        segment = TopicSegment(
            segment_id="test",
            messages=[],
            start_time=datetime(2024, 1, 1, 10, 0),
            end_time=datetime(2024, 1, 1, 10, 30),
        )

        assert segment.duration_seconds == 1800  # 30 minutes


# =============================================================================
# Test TopicSegmenter
# =============================================================================


class TestTopicSegmenter:
    """Test TopicSegmenter class."""

    def test_init_defaults(self):
        """Test default initialization."""
        segmenter = TopicSegmenter()
        assert segmenter.window_size == 3
        assert segmenter.similarity_threshold == 0.55
        assert segmenter.boundary_threshold == 0.5

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        segmenter = TopicSegmenter(
            window_size=5,
            similarity_threshold=0.6,
            boundary_threshold=0.4,
        )
        assert segmenter.window_size == 5
        assert segmenter.similarity_threshold == 0.6
        assert segmenter.boundary_threshold == 0.4

    def test_segment_empty(self):
        """Test segmenting empty message list."""
        segmenter = TopicSegmenter()
        result = segmenter.segment([])
        assert result == []

    def test_segment_single_message(self, mock_message, mock_embedder):
        """Test segmenting a single message."""
        mock_embedder.encode.return_value = np.random.randn(1, 384).astype(np.float32)

        segmenter = TopicSegmenter(embedder=mock_embedder)
        messages = [mock_message("Hello world", False, datetime(2024, 1, 1))]
        result = segmenter.segment(messages)

        assert len(result) == 1
        assert len(result[0].messages) == 1

    def test_segment_detects_time_gap_boundary(self, mock_message, mock_embedder):
        """Test that large time gaps create boundaries."""
        # Return similar embeddings so only time gap triggers boundary
        mock_embedder.encode.return_value = np.ones((3, 384), dtype=np.float32)

        segmenter = TopicSegmenter(embedder=mock_embedder, time_gap_minutes=30.0)
        base_time = datetime(2024, 1, 1, 10, 0)
        messages = [
            mock_message("Hello", False, base_time, 1),
            mock_message("World", True, base_time + timedelta(minutes=5), 2),
            # Large gap
            mock_message("Later message", False, base_time + timedelta(hours=2), 3),
        ]
        result = segmenter.segment(messages)

        # Should have at least 2 segments due to time gap
        assert len(result) >= 2

    def test_segment_detects_topic_shift_marker(self, mock_message, mock_embedder):
        """Test that topic shift markers create boundaries."""
        mock_embedder.encode.return_value = np.ones((2, 384), dtype=np.float32)

        segmenter = TopicSegmenter(
            embedder=mock_embedder, use_topic_shift_markers=True, topic_shift_weight=0.6
        )
        base_time = datetime(2024, 1, 1, 10, 0)
        messages = [
            mock_message("Let's discuss the project", False, base_time, 1),
            mock_message("btw, did you see the game last night?", False, base_time + timedelta(minutes=1), 2),
        ]
        result = segmenter.segment(messages)

        # "btw" should trigger a boundary
        assert len(result) >= 2

    def test_merge_small_segments(self, mock_message, mock_embedder):
        """Test that small segments get merged."""
        mock_embedder.encode.return_value = np.random.randn(3, 384).astype(np.float32)

        # Set high min_segment_messages to force merging
        segmenter = TopicSegmenter(embedder=mock_embedder, min_segment_messages=5)
        base_time = datetime(2024, 1, 1, 10, 0)
        messages = [
            mock_message("A", False, base_time, 1),
            mock_message("B", True, base_time + timedelta(minutes=1), 2),
            mock_message("C", False, base_time + timedelta(minutes=2), 3),
        ]
        result = segmenter.segment(messages)

        # Should merge into a single segment since all are < min_segment_messages
        assert len(result) == 1


class TestBoundaryScoreComputation:
    """Test boundary score computation."""

    def test_compute_window_centroids(self):
        """Test sliding window centroid computation."""
        segmenter = TopicSegmenter(window_size=2)

        # Create messages with known embeddings
        embedding1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        embedding2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        messages = [
            SegmentMessage(
                text="A", timestamp=datetime.now(), is_from_me=False, embedding=embedding1
            ),
            SegmentMessage(
                text="B", timestamp=datetime.now(), is_from_me=False, embedding=embedding2
            ),
        ]

        centroids = segmenter._compute_window_centroids(messages)

        assert centroids is not None
        assert centroids.shape == (2, 3)
        # First centroid is just the first embedding (normalized)
        # Second centroid is mean of both (normalized)

    def test_compute_boundary_scores_empty(self):
        """Test boundary computation with empty/single message."""
        segmenter = TopicSegmenter()

        # Empty
        assert segmenter._compute_boundary_scores([]) == []

        # Single message
        msg = SegmentMessage(text="A", timestamp=datetime.now(), is_from_me=False)
        assert segmenter._compute_boundary_scores([msg]) == []


# =============================================================================
# Test Public API
# =============================================================================


class TestPublicAPI:
    """Test public API functions."""

    def test_reset_segmenter(self):
        """Test resetting singleton segmenter."""
        reset_segmenter()
        # Should not raise
        reset_segmenter()

    @patch("jarvis.topic_segmenter.get_segmenter")
    def test_segment_conversation(self, mock_get_segmenter, mock_message):
        """Test segment_conversation convenience function."""
        mock_segmenter = MagicMock()
        mock_segmenter.segment.return_value = []
        mock_get_segmenter.return_value = mock_segmenter

        messages = [mock_message("Hello", False)]
        result = segment_conversation(messages, contact_id="test")

        mock_segmenter.segment.assert_called_once_with(messages, "test")

    @patch("jarvis.topic_segmenter.segment_conversation")
    def test_segment_for_extraction(self, mock_segment_conversation, mock_message):
        """Test segment_for_extraction returns message groups."""
        base_time = datetime(2024, 1, 1, 10, 0)
        msg1 = mock_message("Hello", False, base_time, 1)
        msg2 = mock_message("World", True, base_time + timedelta(minutes=1), 2)
        messages = [msg1, msg2]

        # Mock segment_conversation to return TopicSegments
        segment = TopicSegment(
            segment_id="test",
            messages=[
                SegmentMessage(text="Hello", timestamp=base_time, is_from_me=False),
                SegmentMessage(text="World", timestamp=base_time + timedelta(minutes=1), is_from_me=True),
            ],
            start_time=base_time,
            end_time=base_time + timedelta(minutes=1),
        )
        mock_segment_conversation.return_value = [segment]

        result = segment_for_extraction(messages)

        assert len(result) == 1
        assert len(result[0]) == 2


# =============================================================================
# Test CorefResolver
# =============================================================================


class TestCorefResolver:
    """Test coreference resolver."""

    def test_coref_resolver_not_available_without_fastcoref(self):
        """Test that resolver returns None when fastcoref not installed."""
        from jarvis.coref_resolver import reset_coref_resolver

        reset_coref_resolver()

        # Mock ImportError for fastcoref
        with patch.dict("sys.modules", {"fastcoref": None}):
            from jarvis.coref_resolver import CorefResolver

            resolver = CorefResolver()

            # Should gracefully handle missing module
            result = resolver.resolve("Jake said he would come")
            # Returns original text when not available
            assert result == "Jake said he would come"

    def test_coref_resolver_resolve_batch_empty(self):
        """Test batch resolution with empty list."""
        from jarvis.coref_resolver import CorefResolver

        resolver = CorefResolver()
        result = resolver.resolve_batch([])
        assert result == []

    def test_coref_resolver_unload(self):
        """Test unloading resolver."""
        from jarvis.coref_resolver import CorefResolver

        resolver = CorefResolver()
        resolver.unload()  # Should not raise
        assert not resolver._available


# =============================================================================
# Test Integration with Extract
# =============================================================================


class TestExtractIntegration:
    """Test integration with extraction pipeline."""

    def test_group_into_turns_semantic_exists(self):
        """Test that semantic bundling method exists on TurnBasedExtractor."""
        from jarvis.extract import TurnBasedExtractor

        # The method should exist
        assert hasattr(TurnBasedExtractor, "_group_into_turns_semantic")

    def test_group_into_turns_time_based_fallback_exists(self):
        """Test that time-based fallback method exists on TurnBasedExtractor."""
        from jarvis.extract import TurnBasedExtractor

        # The method should exist
        assert hasattr(TurnBasedExtractor, "_group_into_turns_time_based")

    def test_time_based_fallback_works(self, mock_message):
        """Test that time-based fallback produces correct turns."""
        from jarvis.extract import ExtractionConfig, TurnBasedExtractor

        base_time = datetime(2024, 1, 1, 10, 0)
        msg1 = mock_message("Hello there friend", False, base_time, 1)
        msg2 = mock_message("How are you doing today", True, base_time + timedelta(minutes=1), 2)
        msg1.attachments = []
        msg2.attachments = []
        msg1.is_system_message = False
        msg2.is_system_message = False

        extractor = TurnBasedExtractor(ExtractionConfig())
        turns = extractor._group_into_turns_time_based([msg1, msg2])

        # Should get 2 turns (different speakers)
        assert len(turns) == 2
        assert turns[0].is_from_me is False
        assert turns[1].is_from_me is True

    def test_segmentation_config_checked(self, mock_message):
        """Test that _group_into_turns checks segmentation config."""
        from jarvis.extract import ExtractionConfig, TurnBasedExtractor

        base_time = datetime(2024, 1, 1, 10, 0)
        msg1 = mock_message("Hello there friend", False, base_time, 1)
        msg1.attachments = []
        msg1.is_system_message = False

        # Mock config with segmentation disabled - should use time-based
        mock_config = MagicMock()
        mock_config.segmentation.enabled = False

        extractor = TurnBasedExtractor(ExtractionConfig())

        with patch("jarvis.config.get_config", return_value=mock_config):
            # Should not raise and should produce turns
            turns = extractor._group_into_turns([msg1])
            assert len(turns) == 1
