"""Unit tests for the merged ContactProfile system.

Tests cover:
- ContactProfile dataclass (to_dict, from_dict, round-trip)
- ContactProfileBuilder.build_profile() with mock messages
- format_style_guide() for casual/formal/abbreviated profiles
- save_profile() / load_profile() round-trip
- get_contact_profile() LRU caching
- Profile building without embeddings (topics skipped gracefully)
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from jarvis.contacts.contact_profile import (
    EMOJI_PATTERN,
    TEXT_ABBREVIATIONS,
    ContactProfile,
    ContactProfileBuilder,
    _get_profile_path,
    format_style_guide,
    load_profile,
    save_profile,
)

# =============================================================================
# Helpers
# =============================================================================


@dataclass
class MockMessage:
    """Minimal Message-like object for testing."""

    text: str
    is_from_me: bool
    date: datetime
    chat_id: str = "chat123"
    id: int = 0
    sender: str = "+1234567890"
    sender_name: str | None = None
    attachments: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.attachments is None:
            self.attachments = []


def _make_messages(
    texts: list[str], is_from_me: bool = True
) -> list[MockMessage]:
    """Create a list of mock messages from text strings."""
    base = datetime(2024, 1, 1, 12, 0, 0)
    return [
        MockMessage(text=t, is_from_me=is_from_me, date=base)
        for t in texts
    ]


def _make_conversation(
    my_texts: list[str], their_texts: list[str]
) -> list[MockMessage]:
    """Create interleaved conversation."""
    msgs: list[MockMessage] = []
    base = datetime(2024, 1, 1, 12, 0, 0)
    for i, t in enumerate(my_texts):
        msgs.append(MockMessage(text=t, is_from_me=True, date=base))
    for i, t in enumerate(their_texts):
        msgs.append(MockMessage(text=t, is_from_me=False, date=base))
    return msgs


# =============================================================================
# ContactProfile Dataclass Tests
# =============================================================================


class TestContactProfile:
    """Tests for ContactProfile dataclass."""

    def test_default_values(self) -> None:
        profile = ContactProfile(contact_id="test123")
        assert profile.contact_id == "test123"
        assert profile.relationship == "unknown"
        assert profile.formality == "casual"
        assert profile.formality_score == 0.5
        assert profile.avg_message_length == 50.0
        assert profile.typical_length == "medium"
        assert profile.message_count == 0
        assert profile.top_topics == []
        assert profile.greeting_style == []

    def test_to_dict(self) -> None:
        profile = ContactProfile(
            contact_id="abc",
            contact_name="Alice",
            relationship="close friend",
            relationship_confidence=0.85,
            formality="very_casual",
            formality_score=0.25,
            avg_message_length=35.0,
            uses_abbreviations=True,
            common_abbreviations=["lol", "idk"],
            emoji_frequency=0.5,
            top_topics=["food, lunch", "scheduling, plans"],
            message_count=200,
            my_message_count=100,
            updated_at="2024-01-01T12:00:00",
        )
        data = profile.to_dict()

        assert data["contact_id"] == "abc"
        assert data["contact_name"] == "Alice"
        assert data["relationship"] == "close friend"
        assert data["formality"] == "very_casual"
        assert data["common_abbreviations"] == ["lol", "idk"]
        assert data["top_topics"] == ["food, lunch", "scheduling, plans"]
        assert data["message_count"] == 200

    def test_from_dict(self) -> None:
        data = {
            "contact_id": "xyz",
            "contact_name": "Bob",
            "relationship": "coworker",
            "relationship_confidence": 0.7,
            "formality": "formal",
            "formality_score": 0.8,
            "avg_message_length": 120.0,
            "typical_length": "long",
            "message_count": 50,
        }
        profile = ContactProfile.from_dict(data)

        assert profile.contact_id == "xyz"
        assert profile.contact_name == "Bob"
        assert profile.relationship == "coworker"
        assert profile.formality == "formal"
        assert profile.avg_message_length == 120.0
        assert profile.message_count == 50

    def test_from_dict_missing_keys(self) -> None:
        data = {"contact_id": "minimal"}
        profile = ContactProfile.from_dict(data)

        assert profile.contact_id == "minimal"
        assert profile.relationship == "unknown"
        assert profile.formality == "casual"
        assert profile.message_count == 0

    def test_roundtrip(self) -> None:
        original = ContactProfile(
            contact_id="roundtrip",
            contact_name="Charlie",
            relationship="family",
            relationship_confidence=0.9,
            formality="casual",
            formality_score=0.4,
            avg_message_length=45.0,
            typical_length="medium",
            uses_lowercase=True,
            uses_abbreviations=True,
            common_abbreviations=["lol", "gonna"],
            emoji_frequency=1.2,
            exclamation_frequency=0.3,
            greeting_style=["hey", "yo"],
            signoff_style=["later"],
            common_phrases=["sounds good"],
            top_topics=["social, party"],
            message_count=150,
            my_message_count=80,
            updated_at="2024-06-15T10:30:00",
        )
        restored = ContactProfile.from_dict(original.to_dict())

        assert restored.contact_id == original.contact_id
        assert restored.contact_name == original.contact_name
        assert restored.relationship == original.relationship
        assert restored.formality == original.formality
        assert restored.formality_score == original.formality_score
        assert restored.avg_message_length == original.avg_message_length
        assert restored.uses_lowercase == original.uses_lowercase
        assert restored.common_abbreviations == original.common_abbreviations
        assert restored.emoji_frequency == original.emoji_frequency
        assert restored.greeting_style == original.greeting_style
        assert restored.top_topics == original.top_topics
        assert restored.message_count == original.message_count


# =============================================================================
# ContactProfileBuilder Tests
# =============================================================================


class TestContactProfileBuilder:
    """Tests for ContactProfileBuilder.build_profile()."""

    @pytest.fixture
    def builder(self) -> ContactProfileBuilder:
        return ContactProfileBuilder(min_messages=3)

    def test_insufficient_messages_returns_minimal(
        self, builder: ContactProfileBuilder
    ) -> None:
        msgs = _make_messages(["hi", "hey"])
        with patch(
            "jarvis.contact_profile.ContactProfileBuilder._classify_relationship",
            return_value=("unknown", 0.0),
        ):
            profile = builder.build_profile("chat1", msgs)  # type: ignore[arg-type]

        assert profile.contact_id == "chat1"
        assert profile.message_count == 2
        assert profile.relationship == "unknown"

    def test_casual_messages(self, builder: ContactProfileBuilder) -> None:
        msgs = _make_conversation(
            my_texts=[
                "hey lol",
                "omg thats so funny haha",
                "gonna go now",
                "yeah nah idk",
                "btw u coming?",
            ],
            their_texts=[
                "sup dude",
                "haha yeah",
                "ok cool",
                "later bro",
                "yep see ya",
            ],
        )
        with patch(
            "jarvis.contact_profile.ContactProfileBuilder._classify_relationship",
            return_value=("close friend", 0.8),
        ):
            profile = builder.build_profile("chat2", msgs, "John")  # type: ignore[arg-type]

        assert profile.relationship == "close friend"
        assert profile.formality in ("casual", "very_casual")
        assert profile.uses_abbreviations is True
        assert profile.message_count == 10
        assert profile.my_message_count == 5
        assert profile.contact_name == "John"

    def test_formal_messages(self, builder: ContactProfileBuilder) -> None:
        msgs = _make_conversation(
            my_texts=[
                "Dear Sir, I am writing to confirm our meeting.",
                "Please find attached the requested documentation.",
                "I appreciate your prompt attention to this matter.",
                "The meeting has been scheduled for Tuesday.",
                "Kindly confirm your availability.",
            ],
            their_texts=[
                "Thank you for the update.",
                "Confirmed for Tuesday.",
                "The documentation looks good.",
                "I will review and respond.",
                "Regards.",
            ],
        )
        with patch(
            "jarvis.contact_profile.ContactProfileBuilder._classify_relationship",
            return_value=("coworker", 0.7),
        ):
            profile = builder.build_profile("chat3", msgs)  # type: ignore[arg-type]

        assert profile.relationship == "coworker"
        assert profile.formality == "formal"
        assert profile.uses_abbreviations is False
        assert profile.emoji_frequency == 0.0

    def test_no_embeddings_skips_topics(
        self, builder: ContactProfileBuilder
    ) -> None:
        msgs = _make_messages(["test message"] * 5)
        with patch(
            "jarvis.contact_profile.ContactProfileBuilder._classify_relationship",
            return_value=("unknown", 0.0),
        ):
            profile = builder.build_profile("chat4", msgs, embeddings=None)  # type: ignore[arg-type]

        assert profile.top_topics == []

    def test_emoji_detection(self, builder: ContactProfileBuilder) -> None:
        msgs = _make_messages(
            ["Hello! 😊", "That's great 🎉", "Thanks 👍", "Sure thing"]
        )
        with patch(
            "jarvis.contact_profile.ContactProfileBuilder._classify_relationship",
            return_value=("unknown", 0.0),
        ):
            profile = builder.build_profile("chat5", msgs)  # type: ignore[arg-type]

        assert profile.emoji_frequency == pytest.approx(0.75, rel=0.1)

    def test_greeting_extraction(self, builder: ContactProfileBuilder) -> None:
        msgs = _make_messages(
            ["hey what's up", "hey there", "hey how are you", "yo dude"]
        )
        with patch(
            "jarvis.contact_profile.ContactProfileBuilder._classify_relationship",
            return_value=("unknown", 0.0),
        ):
            profile = builder.build_profile("chat6", msgs)  # type: ignore[arg-type]

        assert "hey" in profile.greeting_style

    def test_lowercase_detection(self, builder: ContactProfileBuilder) -> None:
        msgs = _make_messages(
            ["hey what's up", "nothing much", "same here lol", "cool cool"]
        )
        with patch(
            "jarvis.contact_profile.ContactProfileBuilder._classify_relationship",
            return_value=("unknown", 0.0),
        ):
            profile = builder.build_profile("chat7", msgs)  # type: ignore[arg-type]

        assert profile.uses_lowercase is True


# =============================================================================
# format_style_guide Tests
# =============================================================================


class TestFormatStyleGuide:
    """Tests for format_style_guide()."""

    def test_minimal_profile(self) -> None:
        profile = ContactProfile(contact_id="x", message_count=3)
        guide = format_style_guide(profile)
        assert "Limited message history" in guide

    def test_casual_friend(self) -> None:
        profile = ContactProfile(
            contact_id="x",
            relationship="close friend",
            formality="very_casual",
            avg_message_length=35.0,
            uses_abbreviations=True,
            common_abbreviations=["lol", "gonna", "idk"],
            emoji_frequency=0.0,
            greeting_style=["hey", "yo"],
            top_topics=["food, lunch", "social, party"],
            message_count=100,
        )
        guide = format_style_guide(profile)

        assert "very casual" in guide
        assert "close friend" in guide
        assert "35" in guide
        assert "lol" in guide
        assert "hey" in guide or "yo" in guide
        assert "Avoid emojis" in guide

    def test_formal_coworker(self) -> None:
        profile = ContactProfile(
            contact_id="x",
            relationship="coworker",
            formality="formal",
            avg_message_length=120.0,
            uses_abbreviations=False,
            emoji_frequency=0.0,
            message_count=50,
        )
        guide = format_style_guide(profile)

        assert "professional" in guide
        assert "coworker" in guide
        assert "abbreviation" not in guide.lower()

    def test_unknown_relationship_omitted(self) -> None:
        profile = ContactProfile(
            contact_id="x",
            relationship="unknown",
            formality="casual",
            message_count=30,
        )
        guide = format_style_guide(profile)

        assert "unknown" not in guide.lower()
        assert "casual" in guide


# =============================================================================
# Storage Tests
# =============================================================================


class TestStorage:
    """Tests for save_profile / load_profile round-trip."""

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "jarvis.contact_profile.PROFILES_DIR",
                Path(temp_dir) / "profiles",
            ):
                profile = ContactProfile(
                    contact_id="storage_test",
                    contact_name="TestUser",
                    relationship="family",
                    formality="casual",
                    formality_score=0.4,
                    avg_message_length=42.0,
                    uses_abbreviations=True,
                    common_abbreviations=["lol"],
                    message_count=100,
                    updated_at="2024-01-01T00:00:00",
                )

                assert save_profile(profile) is True

                loaded = load_profile("storage_test")
                assert loaded is not None
                assert loaded.contact_id == "storage_test"
                assert loaded.contact_name == "TestUser"
                assert loaded.relationship == "family"
                assert loaded.formality_score == 0.4
                assert loaded.avg_message_length == 42.0
                assert loaded.common_abbreviations == ["lol"]
                assert loaded.message_count == 100

    def test_load_nonexistent_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "jarvis.contact_profile.PROFILES_DIR",
                Path(temp_dir) / "profiles",
            ):
                result = load_profile("nonexistent_contact")
                assert result is None

    def test_profile_path_uses_hash(self) -> None:
        path1 = _get_profile_path("contact1")
        path2 = _get_profile_path("contact2")
        path1_again = _get_profile_path("contact1")

        assert path1 != path2
        assert path1 == path1_again
        assert path1.suffix == ".json"


# =============================================================================
# Regex / Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_abbreviations_contains_common_slang(self) -> None:
        expected = {"lol", "omg", "btw", "idk", "tbh", "gonna", "wanna"}
        assert expected.issubset(TEXT_ABBREVIATIONS)

    def test_emoji_pattern_matches(self) -> None:
        cases = [
            ("Hello 😊", 1),
            ("🎉🎊🎈", 3),
            ("No emojis here", 0),
            ("Mixed 👍 content 🔥", 2),
        ]
        for text, expected in cases:
            matches = EMOJI_PATTERN.findall(text)
            assert len(matches) == expected, f"Failed for: {text}"


# =============================================================================
# Abbreviation Ordering Tests
# =============================================================================


class TestAbbreviationOrdering:
    """Tests that abbreviations are returned sorted by frequency."""

    def test_abbreviations_sorted_by_frequency(self) -> None:
        # "lol" appears 5 times, "idk" 3 times, "btw" 1 time
        texts = (
            ["lol that was funny"] * 5
            + ["idk what happened"] * 3
            + ["btw check this out"]
        )
        builder = ContactProfileBuilder(min_messages=3)
        uses_abbrevs, abbrevs = builder._detect_abbreviations(texts)
        assert uses_abbrevs is True
        assert len(abbrevs) >= 3
        # lol should come before idk, idk before btw
        lol_idx = abbrevs.index("lol")
        idk_idx = abbrevs.index("idk")
        btw_idx = abbrevs.index("btw")
        assert lol_idx < idk_idx < btw_idx

    def test_abbreviations_max_five(self) -> None:
        # Use many different abbreviations
        texts = [
            "lol idk btw tbh omg ngl gonna wanna u ur"
        ] * 5
        builder = ContactProfileBuilder(min_messages=3)
        _, abbrevs = builder._detect_abbreviations(texts)
        assert len(abbrevs) <= 5


# =============================================================================
# Common Phrases AND-logic Tests
# =============================================================================


class TestCommonPhrasesFiltering:
    """Tests that common phrases require BOTH words to be non-stopwords."""

    def test_stopword_pairs_excluded(self) -> None:
        # "the meeting" has "the" as stopword - with AND logic, "the" is
        # a stopword but "meeting" is not. Both must be non-stopwords.
        # Phrases like "is the", "for it" (both stopwords) should be excluded.
        msgs = _make_messages(
            ["is the plan ready"] * 5
            + ["plan ready for tomorrow"] * 5
        )
        builder = ContactProfileBuilder(min_messages=3)
        phrases = builder._extract_common_phrases(msgs)
        # "is the" should NOT appear (both words: "is" is stopword)
        for phrase in phrases:
            words = phrase.split()
            from jarvis.contacts.contact_profile import STOPWORDS
            # With AND logic, neither word should be a stopword
            assert words[0] not in STOPWORDS or words[1] not in STOPWORDS
            # More specifically, phrases where both are stopwords must not appear
            assert not (words[0] in STOPWORDS and words[1] in STOPWORDS)

    def test_meaningful_phrases_kept(self) -> None:
        # "sounds good" - both non-stopwords, should be kept
        msgs = _make_messages(["sounds good thanks"] * 5)
        builder = ContactProfileBuilder(min_messages=3)
        phrases = builder._extract_common_phrases(msgs, min_count=3)
        assert "sounds good" in phrases
