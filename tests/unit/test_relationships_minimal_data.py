"""TEST-09: Relationship profiles with <20 messages.

Verifies graceful handling when building profiles with minimal data.
The system requires MIN_MESSAGES_FOR_PROFILE (20) messages for reliable
analysis. Tests verify what happens with fewer messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class MockMessage:
    """Mock message for testing."""

    text: str
    is_from_me: bool
    date: datetime
    attachments: list = None

    def __post_init__(self):
        if self.attachments is None:
            self.attachments = []


def create_messages(count: int) -> list[MockMessage]:
    """Create a list of mock messages."""
    base_time = datetime(2024, 1, 1, 10, 0)
    messages = []
    for i in range(count):
        messages.append(
            MockMessage(
                text=f"Message number {i}",
                is_from_me=(i % 2 == 0),
                date=base_time + timedelta(minutes=i * 5),
            )
        )
    return messages


class TestRelationshipProfileMinimalData:
    """Test behavior with fewer than MIN_MESSAGES_FOR_PROFILE messages."""

    def test_min_messages_constant_is_20(self):
        """Verify the constant value."""
        from jarvis.relationships import MIN_MESSAGES_FOR_PROFILE

        assert MIN_MESSAGES_FOR_PROFILE == 20

    def test_build_profile_with_zero_messages(self):
        """Building a profile with 0 messages returns minimal profile."""
        from jarvis.relationships import build_relationship_profile

        result = build_relationship_profile("empty_contact", [])
        # Returns a minimal profile (not None) with defaults
        assert result is not None
        assert result.message_count == 0

    def test_build_profile_with_one_message(self):
        """Building a profile with 1 message returns minimal profile."""
        from jarvis.relationships import build_relationship_profile

        messages = create_messages(1)
        result = build_relationship_profile("sparse_contact", messages)
        assert result is not None
        assert result.message_count == 1

    def test_build_profile_with_19_messages(self):
        """Building a profile with 19 messages (under threshold) returns minimal profile."""
        from jarvis.relationships import build_relationship_profile

        messages = create_messages(19)
        result = build_relationship_profile("almost_enough", messages)
        assert result is not None
        assert result.message_count == 19

    def test_build_profile_with_exactly_20_messages(self):
        """Building a profile with exactly 20 messages succeeds with full analysis."""
        from jarvis.relationships import build_relationship_profile

        messages = create_messages(20)
        result = build_relationship_profile("just_enough", messages)
        assert result is not None
        # Full analysis should populate tone_profile
        assert result.tone_profile is not None

    def test_build_profile_with_30_messages(self):
        """Building a profile with 30 messages (above threshold) succeeds."""
        from jarvis.relationships import build_relationship_profile

        messages = create_messages(30)
        result = build_relationship_profile("plenty", messages)
        assert result is not None
        assert result.tone_profile is not None

    def test_profile_needs_refresh_no_existing_profile(self):
        """profile_needs_refresh returns True when profile has no last_updated."""
        from jarvis.relationships import RelationshipProfile, profile_needs_refresh

        # Create a profile with empty last_updated (simulates "never updated")
        profile = RelationshipProfile(contact_id="test", last_updated="")
        result = profile_needs_refresh(profile)
        assert result is True

    def test_generate_style_guide_with_minimal_profile(self):
        """Style guide can be generated even with minimal data."""
        from jarvis.relationships import (
            RelationshipProfile,
            ResponsePatterns,
            ToneProfile,
            TopicDistribution,
            generate_style_guide,
        )

        profile = RelationshipProfile(
            contact_id="minimal",
            contact_name="Test",
            tone_profile=ToneProfile(
                formality_score=0.5,
                emoji_frequency=0.0,
                exclamation_frequency=0.0,
                question_frequency=0.0,
                avg_message_length=10.0,
                uses_caps=False,
            ),
            topic_distribution=TopicDistribution(
                topics={},
                top_topics=[],
            ),
            response_patterns=ResponsePatterns(
                avg_response_time_minutes=5.0,
            ),
            message_count=20,
        )

        guide = generate_style_guide(profile)
        assert isinstance(guide, str)
        assert len(guide) > 0

    def test_select_matching_examples_empty_history(self):
        """select_matching_examples with empty example lists returns empty."""
        from jarvis.relationships import RelationshipProfile, select_matching_examples

        profile = RelationshipProfile(contact_id="test")
        result = select_matching_examples(
            profile=profile,
            casual_examples=[],
            professional_examples=[],
        )
        assert result == []

    def test_get_voice_guidance_with_default_profile(self):
        """get_voice_guidance returns dict with guidance parameters."""
        from jarvis.relationships import RelationshipProfile, get_voice_guidance

        profile = RelationshipProfile(contact_id="test")
        guidance = get_voice_guidance(profile)
        assert isinstance(guidance, dict)
        assert "formality" in guidance

    def test_load_profile_nonexistent_contact(self):
        """load_profile for nonexistent contact returns None."""
        from jarvis.relationships import load_profile

        # Use a contact_id that definitely doesn't have a saved profile
        result = load_profile("nonexistent_contact_abc123_xyz")
        assert result is None

    def test_delete_profile_nonexistent_is_safe(self):
        """delete_profile for nonexistent contact doesn't raise."""
        from jarvis.relationships import delete_profile

        # Should not raise
        delete_profile("nonexistent_contact_abc123_xyz")
