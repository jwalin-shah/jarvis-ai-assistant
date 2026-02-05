"""Unit tests for the unified_relationship module.

Tests cover the UnifiedProfileBuilder, formality calculation, and profile serialization.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from jarvis.unified_relationship import (
    CASUAL_INDICATORS,
    FORMAL_INDICATORS,
    CommunicationStyle,
    RelationshipType,
    TopicProfile,
    UnifiedProfileBuilder,
    UnifiedRelationshipProfile,
    build_unified_profile,
    compute_unified_formality,
    unified_to_legacy,
)


class FakeMessage:
    """Fake message object for testing."""

    def __init__(
        self,
        text: str,
        is_from_me: bool = True,
        date: datetime | None = None,
    ):
        self.id = 1
        self.chat_id = "test_chat"
        self.sender = "test_sender"
        self.sender_name = None
        self.text = text
        self.date = date or datetime.now()
        self.is_from_me = is_from_me
        self.attachments = []
        self.reply_to_id = None
        self.reactions = []
        self.date_delivered = None
        self.date_read = None
        self.is_system_message = False


class TestComputeUnifiedFormality:
    """Tests for compute_unified_formality function."""

    def test_casual_messages_low_formality(self) -> None:
        """Test that casual messages result in low formality score."""
        messages = [
            FakeMessage("lol thats hilarious"),
            FakeMessage("omg dude wanna hang out"),
            FakeMessage("yep sounds good lmao"),
            FakeMessage("gonna be late btw"),
        ]

        formality = compute_unified_formality(messages)

        # Should be low (casual)
        assert formality < 0.4

    def test_formal_messages_high_formality(self) -> None:
        """Test that formal messages result in high formality score."""
        messages = [
            FakeMessage("Dear Mr. Smith, please find the attached report."),
            FakeMessage("I appreciate your prompt response regarding the meeting."),
            FakeMessage("Thank you for confirming the scheduled deadline."),
            FakeMessage("Please kindly review the attached documents."),
        ]

        formality = compute_unified_formality(messages)

        # Should be high (formal)
        assert formality > 0.6

    def test_mixed_messages_moderate_formality(self) -> None:
        """Test that mixed messages result in moderate formality."""
        messages = [
            FakeMessage("Hey, can we meet tomorrow?"),
            FakeMessage("Sure, that works for me."),
            FakeMessage("Thanks!"),
            FakeMessage("See you then."),
        ]

        formality = compute_unified_formality(messages)

        # Should be moderate
        assert 0.3 < formality < 0.7

    def test_empty_messages_returns_default(self) -> None:
        """Test that empty message list returns 0.5 (default)."""
        formality = compute_unified_formality([])
        assert formality == 0.5

    def test_emoji_increases_casualness(self) -> None:
        """Test that emoji usage increases casualness (lowers formality)."""
        without_emoji = [FakeMessage("Sounds good")]
        with_emoji = [FakeMessage("Sounds good! 😊")]

        formality_without = compute_unified_formality(without_emoji)
        formality_with = compute_unified_formality(with_emoji)

        # Emoji should lower formality
        assert formality_with < formality_without

    def test_laplace_smoothing_prevents_extremes(self) -> None:
        """Test that Laplace smoothing prevents 0.0 or 1.0 scores."""
        casual_msgs = [FakeMessage("lol")]
        formal_msgs = [FakeMessage("Please review this.")]

        casual_formality = compute_unified_formality(casual_msgs)
        formal_formality = compute_unified_formality(formal_msgs)

        # Neither should hit absolute extremes due to Laplace smoothing
        assert 0.0 < casual_formality < 1.0
        assert 0.0 < formal_formality < 1.0


class TestUnifiedProfileBuilder:
    """Tests for UnifiedProfileBuilder class."""

    def test_build_minimal_profile_insufficient_messages(self) -> None:
        """Test that insufficient messages returns minimal profile."""
        builder = UnifiedProfileBuilder(min_messages=10)
        messages = [FakeMessage(f"Message {i}") for i in range(5)]

        profile = builder.build_profile("test_contact", messages, "Test Contact")

        assert profile.contact_id == "test_contact"
        assert profile.contact_name == "Test Contact"
        assert profile.relationship_type.category == "unknown"
        assert profile.relationship_type.confidence == 0.0
        assert profile.message_count == 5

    @patch("jarvis.topic_chunker.chunk_conversation")
    def test_build_profile_style_analysis(
        self, mock_chunk: MagicMock
    ) -> None:
        """Test that style analysis is performed correctly."""
        # Mock topic chunking to return empty
        mock_chunk.return_value = []

        builder = UnifiedProfileBuilder(min_messages=5)
        messages = [
            FakeMessage("Hey! How are you? 😊"),
            FakeMessage("I'm doing great thanks!!"),
            FakeMessage("Wanna grab lunch?"),
            FakeMessage("Sounds good!"),
            FakeMessage("See ya later!"),
        ]

        profile = builder.build_profile("test_contact", messages)

        # Style should be analyzed
        assert profile.style.formality < 0.5  # Casual messages
        assert profile.style.emoji_frequency > 0  # Has emoji
        assert profile.style.exclamation_frequency > 0  # Has exclamations

    @patch("jarvis.topic_chunker.chunk_conversation")
    def test_build_profile_analyzes_my_messages(
        self, mock_chunk: MagicMock
    ) -> None:
        """Test that style analysis focuses on 'my' messages."""
        mock_chunk.return_value = []

        builder = UnifiedProfileBuilder(min_messages=5)
        messages = [
            FakeMessage("hey what's up", is_from_me=True),
            FakeMessage("Dear Sir, please confirm.", is_from_me=False),
            FakeMessage("lol sounds good", is_from_me=True),
            FakeMessage("I appreciate your response.", is_from_me=False),
            FakeMessage("cool see ya!", is_from_me=True),
        ]

        profile = builder.build_profile("test_contact", messages)

        # Style should reflect MY messages (casual), not theirs (formal)
        assert profile.style.formality < 0.5


class TestUnifiedRelationshipProfileSerialization:
    """Tests for UnifiedRelationshipProfile serialization."""

    def test_to_dict_includes_all_fields(self) -> None:
        """Test that to_dict includes all profile fields."""
        profile = UnifiedRelationshipProfile(
            contact_id="test123",
            contact_name="Test Contact",
            relationship_type=RelationshipType(
                category="close friend",
                confidence=0.85,
                signals={"close friend": 0.85, "coworker": 0.15},
            ),
            style=CommunicationStyle(
                formality=0.3,
                emoji_frequency=1.5,
                avg_message_length=45.0,
                typical_length="medium",
                uses_caps=False,
                exclamation_frequency=0.8,
                question_frequency=0.2,
                greeting_style=["hey", "hi"],
                signoff_style=["thanks"],
                common_phrases=["sounds good"],
            ),
            topics=TopicProfile(
                segment_count=10,
                distribution={"Planning": 0.5, "Food": 0.3},
                top_topics=["Planning", "Food"],
                entities={"PERSON": ["jake"]},
                avg_segment_duration_minutes=12.5,
            ),
            message_count=500,
            last_updated="2024-01-15T10:30:00",
        )

        data = profile.to_dict()

        assert data["contact_id"] == "test123"
        assert data["contact_name"] == "Test Contact"
        assert data["relationship_type"]["category"] == "close friend"
        assert data["relationship_type"]["confidence"] == 0.85
        assert data["style"]["formality"] == 0.3
        assert data["style"]["emoji_frequency"] == 1.5
        assert data["topics"]["segment_count"] == 10
        assert data["topics"]["top_topics"] == ["Planning", "Food"]
        assert data["message_count"] == 500
        assert data["version"] == "2.0.0"

    def test_from_dict_reconstructs_profile(self) -> None:
        """Test that from_dict correctly reconstructs a profile."""
        data = {
            "contact_id": "test123",
            "contact_name": "Test Contact",
            "relationship_type": {
                "category": "family",
                "confidence": 0.9,
                "signals": {"family": 0.9},
            },
            "style": {
                "formality": 0.6,
                "emoji_frequency": 0.5,
                "avg_message_length": 60.0,
                "typical_length": "medium",
                "uses_caps": True,
                "exclamation_frequency": 0.3,
                "question_frequency": 0.4,
                "greeting_style": ["hi"],
                "signoff_style": ["bye"],
                "common_phrases": ["love you"],
            },
            "topics": {
                "segment_count": 5,
                "distribution": {"Family": 0.8},
                "top_topics": ["Family"],
                "entities": {"PERSON": ["mom", "dad"]},
                "avg_segment_duration_minutes": 20.0,
            },
            "message_count": 300,
            "last_updated": "2024-01-15T10:30:00",
            "version": "2.0.0",
        }

        profile = UnifiedRelationshipProfile.from_dict(data)

        assert profile.contact_id == "test123"
        assert profile.relationship_type.category == "family"
        assert profile.relationship_type.confidence == 0.9
        assert profile.style.formality == 0.6
        assert profile.style.uses_caps is True
        assert profile.topics.segment_count == 5
        assert profile.topics.entities == {"PERSON": ["mom", "dad"]}

    def test_roundtrip_serialization(self) -> None:
        """Test that serialize/deserialize roundtrip preserves data."""
        original = UnifiedRelationshipProfile(
            contact_id="roundtrip_test",
            contact_name="Roundtrip Contact",
            relationship_type=RelationshipType(
                category="coworker",
                confidence=0.7,
                signals={"coworker": 0.7},
            ),
            style=CommunicationStyle(
                formality=0.8,
                emoji_frequency=0.1,
                avg_message_length=100.0,
                typical_length="long",
                uses_caps=False,
                exclamation_frequency=0.1,
                question_frequency=0.3,
                greeting_style=["hello"],
                signoff_style=["regards"],
                common_phrases=["as discussed"],
            ),
            topics=TopicProfile(
                segment_count=8,
                distribution={"Work": 0.6, "Meeting": 0.4},
                top_topics=["Work", "Meeting"],
                entities={"ORG": ["google"]},
                avg_segment_duration_minutes=15.0,
            ),
            message_count=200,
            last_updated="2024-01-15T10:30:00",
        )

        data = original.to_dict()
        restored = UnifiedRelationshipProfile.from_dict(data)

        assert restored.contact_id == original.contact_id
        assert restored.relationship_type.category == original.relationship_type.category
        assert restored.style.formality == original.style.formality
        assert restored.topics.segment_count == original.topics.segment_count
        assert restored.version == original.version


class TestUnifiedToLegacy:
    """Tests for unified_to_legacy conversion."""

    def test_converts_to_legacy_format(self) -> None:
        """Test that unified_to_legacy produces correct legacy format."""
        profile = UnifiedRelationshipProfile(
            contact_id="legacy_test",
            contact_name="Legacy Contact",
            relationship_type=RelationshipType(
                category="close friend",
                confidence=0.85,
            ),
            style=CommunicationStyle(
                formality=0.3,
                emoji_frequency=1.5,
                avg_message_length=45.0,
                typical_length="medium",
                uses_caps=False,
                exclamation_frequency=0.8,
                question_frequency=0.2,
                greeting_style=["hey"],
                signoff_style=["thanks"],
                common_phrases=["sounds good"],
            ),
            topics=TopicProfile(
                segment_count=10,
                distribution={"Planning": 0.5},
                top_topics=["Planning"],
                entities={},
                avg_segment_duration_minutes=12.0,
            ),
            message_count=100,
            last_updated="2024-01-15T10:30:00",
        )

        legacy = unified_to_legacy(profile)

        # Check legacy format structure
        assert legacy["contact_id"] == "legacy_test"
        assert "tone_profile" in legacy
        assert legacy["tone_profile"]["formality_score"] == 0.3
        assert legacy["tone_profile"]["emoji_frequency"] == 1.5
        assert "topic_distribution" in legacy
        assert legacy["topic_distribution"]["topics"] == {"Planning": 0.5}
        assert "response_patterns" in legacy
        assert legacy["response_patterns"]["typical_response_length"] == "medium"
        assert legacy["response_patterns"]["greeting_style"] == ["hey"]


class TestBuildUnifiedProfile:
    """Tests for build_unified_profile convenience function."""

    @patch("jarvis.topic_chunker.chunk_conversation")
    def test_builds_profile_with_default_settings(
        self, mock_chunk: MagicMock
    ) -> None:
        """Test that build_unified_profile works with default settings."""
        mock_chunk.return_value = []

        messages = [FakeMessage(f"Message {i}") for i in range(15)]

        profile = build_unified_profile("test_contact", messages, "Test")

        assert profile.contact_id == "test_contact"
        assert profile.contact_name == "Test"
        assert profile.message_count == 15


class TestIndicatorSets:
    """Tests to verify indicator sets are properly defined."""

    def test_casual_indicators_not_empty(self) -> None:
        """Test that casual indicators set is populated."""
        assert len(CASUAL_INDICATORS) > 20
        assert "lol" in CASUAL_INDICATORS
        assert "haha" in CASUAL_INDICATORS
        assert "gonna" in CASUAL_INDICATORS

    def test_formal_indicators_not_empty(self) -> None:
        """Test that formal indicators set is populated."""
        assert len(FORMAL_INDICATORS) > 10
        assert "please" in FORMAL_INDICATORS
        assert "regards" in FORMAL_INDICATORS
        assert "thank you" in FORMAL_INDICATORS

    def test_no_overlap_between_casual_and_formal(self) -> None:
        """Test that casual and formal indicators don't overlap."""
        overlap = CASUAL_INDICATORS & FORMAL_INDICATORS
        assert len(overlap) == 0, f"Unexpected overlap: {overlap}"
