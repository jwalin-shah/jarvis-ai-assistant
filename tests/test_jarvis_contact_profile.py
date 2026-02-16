from datetime import datetime
from unittest.mock import patch

import numpy as np
import pytest

from contracts.imessage import Message
from jarvis.contacts.contact_profile import (
    ContactProfile,
    ContactProfileBuilder,
    Fact,
    format_style_guide,
    get_contact_profile,
    invalidate_profile_cache,
    load_profile,
    save_profile,
)


@pytest.fixture
def mock_messages():
    return [
        Message(
            id=i,
            chat_id="chat123",
            sender="me" if i % 2 == 0 else "them",
            sender_name="Them",
            text=f"Message {i} content. Hi!",
            date=datetime.now(),
            is_from_me=(i % 2 == 0),
        )
        for i in range(20)
    ]


def test_fact_dataclass():
    fact = Fact(
        category="work",
        subject="Google",
        predicate="works_at",
        value="Software Engineer",
        source_text="I work at Google as a Software Engineer",
        confidence=0.9,
        contact_id="contact1",
    )
    assert fact.category == "work"
    assert fact.subject == "Google"
    assert fact.confidence == 0.9


def test_contact_profile_to_from_dict():
    profile = ContactProfile(
        contact_id="chat123",
        contact_name="Test User",
        relationship="friend",
        formality="casual",
        message_count=100,
    )
    data = profile.to_dict()
    assert data["contact_id"] == "chat123"
    assert data["relationship"] == "friend"

    new_profile = ContactProfile.from_dict(data)
    assert new_profile.contact_id == "chat123"
    assert new_profile.relationship == "friend"
    assert new_profile.message_count == 100


def test_profile_builder_min_messages():
    builder = ContactProfileBuilder(min_messages=10)
    messages = [
        Message(
            id=1,
            chat_id="c1",
            sender="me",
            sender_name="me",
            text="hi",
            date=datetime.now(),
            is_from_me=True,
        )
    ]
    profile = builder.build_profile("c1", messages)
    assert profile.message_count == 1
    assert profile.relationship == "unknown"  # Default when too few messages


@patch("jarvis.contacts.contact_profile.ContactProfileBuilder._classify_relationship")
def test_profile_builder_full(mock_classify, mock_messages):
    mock_classify.return_value = ("family", 0.95)

    # Add some diverse messages to test style analysis
    mock_messages[0].text = "lol omg that is so funny!!! 😂"
    mock_messages[2].text = "u rly coming to the party rn?"
    mock_messages[4].text = "hi"
    mock_messages[6].text = "see you later"  # Changed to avoid "talk later" vs "later" ambiguity

    # Clear out the "Message X content" from other "me" messages to avoid noise
    for i in range(8, 20, 2):
        mock_messages[i].text = "just another message"

    builder = ContactProfileBuilder(min_messages=10)
    profile = builder.build_profile("chat123", mock_messages, contact_name="Test User")

    assert profile.contact_id == "chat123"
    assert profile.relationship == "family"
    assert profile.relationship_confidence == 0.95
    assert profile.message_count == 20
    assert profile.my_message_count == 10
    assert profile.uses_abbreviations is True
    assert "lol" in profile.common_abbreviations or "rn" in profile.common_abbreviations
    assert profile.emoji_frequency > 0
    assert profile.exclamation_frequency > 0
    assert "hi" in profile.greeting_style
    assert "later" in profile.signoff_style or "see you" in profile.signoff_style


def test_compute_formality():
    builder = ContactProfileBuilder()

    formal_msgs = [
        Message(
            id=1,
            chat_id="c1",
            sender="me",
            sender_name="me",
            text="Dear Mr. Smith, Please find the attached document regarding our meeting.",
            date=datetime.now(),
            is_from_me=True,
        ),
        Message(
            id=2,
            chat_id="c1",
            sender="me",
            sender_name="me",
            text="Sincerely, Jarvis.",
            date=datetime.now(),
            is_from_me=True,
        ),
    ]
    formal_score = builder._compute_formality(formal_msgs)

    casual_msgs = [
        Message(
            id=1,
            chat_id="c1",
            sender="me",
            sender_name="me",
            text="lol u coming??",
            date=datetime.now(),
            is_from_me=True,
        ),
        Message(
            id=2,
            chat_id="c1",
            sender="me",
            sender_name="me",
            text="omg so funny!!! 😂😂😂",
            date=datetime.now(),
            is_from_me=True,
        ),
    ]
    casual_score = builder._compute_formality(casual_msgs)

    assert formal_score > casual_score


def test_format_style_guide():
    profile = ContactProfile(
        contact_id="c1",
        relationship="friend",
        formality="casual",
        avg_message_length=45.2,
        uses_abbreviations=True,
        common_abbreviations=["lol", "u", "rn"],
        emoji_frequency=0.5,
        message_count=50,
    )
    guide = format_style_guide(profile)
    assert "casual" in guide
    assert "friend" in guide
    assert "45 chars" in guide
    assert "lol" in guide


def test_format_style_guide_low_history():
    profile = ContactProfile(
        contact_id="c1", message_count=4
    )  # Must be < MIN_MESSAGES_FOR_PROFILE (5)
    guide = format_style_guide(profile)
    assert "Limited message history" in guide


@patch("jarvis.contacts.contact_profile.PROFILES_DIR")
def test_save_load_profile(mock_profiles_dir, tmp_path):
    mock_profiles_dir.mkdir.return_value = None
    # We need to mock _get_profile_path to use tmp_path
    with patch("jarvis.contacts.contact_profile._get_profile_path") as mock_path:
        test_path = tmp_path / "test_profile.json"
        mock_path.return_value = test_path

        profile = ContactProfile(contact_id="test_contact", relationship="colleague")
        assert save_profile(profile) is True
        assert test_path.exists()

        loaded = load_profile("test_contact")
        assert loaded.contact_id == "test_contact"
        assert loaded.relationship == "colleague"


def test_get_contact_profile_caching():
    invalidate_profile_cache()
    with patch("jarvis.contacts.contact_profile.load_profile") as mock_load:
        profile = ContactProfile(contact_id="cached_contact")
        mock_load.return_value = profile

        # First call: load from disk
        res1 = get_contact_profile("cached_contact")
        assert res1 == profile
        assert mock_load.call_count == 1

        # Second call: from cache
        res2 = get_contact_profile("cached_contact")
        assert res2 == profile
        assert mock_load.call_count == 1

        invalidate_profile_cache()
        get_contact_profile("cached_contact")
        assert mock_load.call_count == 2


def test_discover_topics(mock_messages):
    """Test _discover_topics handles edge cases correctly."""
    builder = ContactProfileBuilder()

    # When embeddings is None, should return empty list
    assert builder._discover_topics("c1", mock_messages, None) == []

    # When embeddings provided but no segments in DB, falls back to message texts
    embeddings = np.random.randn(40, 384).astype(np.float32)
    long_messages = mock_messages * 2
    result = builder._discover_topics("c1", long_messages, embeddings)
    # Returns message texts that are long enough (>20 chars)
    assert isinstance(result, list)
    # The fallback returns up to 10 long messages
    assert len(result) <= 10
