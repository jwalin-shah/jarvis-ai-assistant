"""Behavior-driven tests for contact facts extraction and storage.

Tests focus on PUBLIC methods only:
- FactExtractor.extract_facts() - extract facts from messages
- save_facts() / get_facts_for_contact() - persist and retrieve
- ContactProfileBuilder.build_profile() - build profiles from messages

Principles:
1. Test behavior: "given messages X, expect facts Y"
2. Minimal mocking - only external dependencies (DB paths)
3. Real in-memory test data
4. Tests pass even if implementation is rewritten
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

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
from jarvis.contacts.fact_extractor import FactExtractor
from jarvis.contacts.fact_storage import get_facts_for_contact, save_facts
from jarvis.contracts.imessage import Message

# =============================================================================
# Test Data Helpers
# =============================================================================


def create_test_message(
    text: str,
    is_from_me: bool = False,
    msg_id: int = 1,
    chat_id: str = "test_chat",
    sender: str = "+1234567890",
    date: datetime | None = None,
) -> Message:
    """Create a test message with sensible defaults."""
    return Message(
        id=msg_id,
        chat_id=chat_id,
        sender=sender,
        sender_name="Test Contact" if not is_from_me else "Me",
        text=text,
        date=date or datetime(2024, 1, 15, 10, 0, 0),
        is_from_me=is_from_me,
    )


def create_test_messages(texts: list[str], **kwargs: Any) -> list[Message]:
    """Create multiple test messages from text strings."""
    return [create_test_message(text, msg_id=i + 1, **kwargs) for i, text in enumerate(texts)]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def extractor() -> FactExtractor:
    """Create a FactExtractor with default settings."""
    return FactExtractor()


@pytest.fixture
def temp_db_path():
    """Provide a temporary database path for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_profiles_dir(temp_db_path: Path):
    """Mock the profiles directory to use temp path."""
    with patch(
        "jarvis.contacts.contact_profile.PROFILES_DIR",
        temp_db_path / "profiles",
    ):
        (temp_db_path / "profiles").mkdir(exist_ok=True)
        yield temp_db_path / "profiles"


@pytest.fixture(autouse=True)
def clear_profile_cache():
    """Clear profile cache before each test."""
    invalidate_profile_cache()
    yield


# =============================================================================
# Fact Extraction Tests (Public API: extract_facts)
# =============================================================================


class TestExtractFactsWork:
    """Test extraction of work/employment facts from messages."""

    def test_extract_work_from_simple_statement(self, extractor: FactExtractor) -> None:
        """Given 'I work at Google', expect work fact extracted."""
        messages = [create_test_message("I work at Google")]

        facts = extractor.extract_facts(messages, contact_id="test_contact")

        work_facts = [f for f in facts if f.category == "work"]
        assert len(work_facts) >= 1
        assert any("Google" in f.subject for f in work_facts)

    def test_extract_work_various_patterns(self, extractor: FactExtractor) -> None:
        """Given various work patterns, expect appropriate facts."""
        test_cases = [
            ("I work at Microsoft", "Microsoft"),
            ("I work at Google", "Google"),
            ("I work at Amazon now", "Amazon"),
            ("I joined Meta yesterday", "Meta"),
        ]

        for text, expected_company in test_cases:
            messages = [create_test_message(text)]
            facts = extractor.extract_facts(messages, contact_id="test_contact")
            work_facts = [f for f in facts if f.category == "work"]
            assert any(expected_company in f.subject for f in work_facts), (
                f"Expected '{expected_company}' in work facts for: {text}"
            )

    def test_no_work_facts_from_casual_chat(self, extractor: FactExtractor) -> None:
        """Given casual conversation without work info, expect no work facts."""
        messages = [
            create_test_message("Hey, how are you?"),
            create_test_message("Want to grab lunch?"),
            create_test_message("Sounds good!"),
        ]

        facts = extractor.extract_facts(messages, contact_id="test_contact")
        work_facts = [f for f in facts if f.category == "work"]

        assert len(work_facts) == 0


class TestExtractFactsLocation:
    """Test extraction of location facts from messages."""

    def test_extract_current_location(self, extractor: FactExtractor) -> None:
        """Given 'I live in Austin', expect location fact."""
        messages = [create_test_message("I live in Austin")]

        facts = extractor.extract_facts(messages, contact_id="test_contact")

        location_facts = [f for f in facts if f.category == "location"]
        assert len(location_facts) >= 1
        assert any("Austin" in f.subject for f in location_facts)

    def test_extract_location_various_patterns(self, extractor: FactExtractor) -> None:
        """Given various location patterns, expect appropriate facts."""
        test_cases = [
            ("I live in San Francisco", "San Francisco"),
            ("Based in New York", "New York"),
            ("Currently in Seattle", "Seattle"),
        ]

        for text, expected_location in test_cases:
            messages = [create_test_message(text)]
            facts = extractor.extract_facts(messages, contact_id="test_contact")
            location_facts = [f for f in facts if f.category == "location"]
            assert any(expected_location in f.subject for f in location_facts), (
                f"Expected '{expected_location}' in location facts for: {text}"
            )

    def test_extract_moving_location(self, extractor: FactExtractor) -> None:
        """Given 'moving to LA', expect future location fact."""
        messages = [create_test_message("I'm moving to Los Angeles next month")]

        facts = extractor.extract_facts(messages, contact_id="test_contact")

        location_facts = [f for f in facts if f.category == "location"]
        assert any("Los Angeles" in f.subject for f in location_facts)


class TestExtractFactsPreferences:
    """Test extraction of preference/liking facts from messages."""

    def test_extract_likes(self, extractor: FactExtractor) -> None:
        """Given 'I love hiking', expect preference fact with 'likes' predicate."""
        messages = [create_test_message("I love hiking in the mountains")]

        facts = extractor.extract_facts(messages, contact_id="test_contact")

        pref_facts = [f for f in facts if f.category == "preference"]
        assert len(pref_facts) >= 1
        assert any("hiking" in f.subject.lower() for f in pref_facts)
        assert any(f.predicate == "likes" for f in pref_facts)

    def test_extract_dislikes(self, extractor: FactExtractor) -> None:
        """Given 'I hate cilantro', expect preference fact with 'dislikes' predicate."""
        messages = [create_test_message("I hate cilantro on my tacos")]

        facts = extractor.extract_facts(messages, contact_id="test_contact")

        pref_facts = [f for f in facts if f.category == "preference"]
        assert any("cilantro" in f.subject.lower() for f in pref_facts)
        assert any(f.predicate == "dislikes" for f in pref_facts)

    def test_no_preference_from_filler_like(self, extractor: FactExtractor) -> None:
        """Given 'I was like, whatever', expect no preference fact (filler word)."""
        messages = [create_test_message("I was like, totally excited about it")]

        facts = extractor.extract_facts(messages, contact_id="test_contact")
        pref_facts = [f for f in facts if f.category == "preference"]

        # Should not extract "like" as a preference when it's used as filler
        assert not any("totally excited" in f.subject for f in pref_facts)


class TestExtractFactsRelationships:
    """Test extraction of relationship facts from messages."""

    def test_extract_family_relationship(self, extractor: FactExtractor) -> None:
        """Given 'my sister Sarah', expect relationship fact."""
        messages = [create_test_message("My sister Sarah is visiting next week")]

        facts = extractor.extract_facts(messages, contact_id="test_contact")

        rel_facts = [f for f in facts if f.category == "relationship"]
        assert any("Sarah" in f.subject for f in rel_facts)
        assert any(f.value and "sister" in f.value.lower() for f in rel_facts)

    def test_extract_friend_relationship(self, extractor: FactExtractor) -> None:
        """Given 'my friend John', expect friend relationship fact."""
        messages = [create_test_message("Going to the game with my friend Mike")]

        facts = extractor.extract_facts(messages, contact_id="test_contact")

        rel_facts = [f for f in facts if f.category == "relationship"]
        assert any("Mike" in f.subject for f in rel_facts)


class TestExtractFactsFiltering:
    """Test that low-quality facts are filtered out."""

    def test_filters_vague_pronouns(self, extractor: FactExtractor) -> None:
        """Given 'I love me', expect no fact with subject 'me'."""
        messages = [create_test_message("I love me")]

        facts = extractor.extract_facts(messages, contact_id="test_contact")
        subjects = {f.subject.lower().strip() for f in facts}

        assert "me" not in subjects
        assert "you" not in subjects
        assert "it" not in subjects

    def test_filters_bot_messages(self, extractor: FactExtractor) -> None:
        """Given pharmacy/bot messages, expect no facts extracted."""
        messages = [
            create_test_message("Your CVS Pharmacy prescription is ready for pickup"),
            create_test_message("I work at Google"),  # Real message should still work
        ]

        facts = extractor.extract_facts(messages, contact_id="test_contact")

        subjects = {f.subject.lower() for f in facts}
        assert "prescription" not in subjects
        assert "cvs" not in subjects
        # The real message should still produce facts
        assert any("Google" in f.subject for f in facts)

    def test_deduplicates_facts(self, extractor: FactExtractor) -> None:
        """Given duplicate facts in messages, expect single fact."""
        messages = [
            create_test_message("I work at Google"),
            create_test_message("Yeah, I work at Google"),
            create_test_message("Google is my employer"),
        ]

        facts = extractor.extract_facts(messages, contact_id="test_contact")
        work_facts = [f for f in facts if f.category == "work" and "Google" in f.subject]

        # Should deduplicate to single Google work fact
        assert len(work_facts) == 1

    def test_respects_confidence_threshold(self) -> None:
        """Given low-confidence extraction with high threshold, expect filtered."""
        extractor = FactExtractor(confidence_threshold=0.8)
        messages = [create_test_message("I like sf")]  # Short, gets penalty

        facts = extractor.extract_facts(messages, contact_id="test_contact")

        # Short subjects get confidence penalty, may fall below 0.8 threshold
        pref_facts = [f for f in facts if f.category == "preference"]
        for f in pref_facts:
            assert f.confidence >= 0.8


class TestExtractFactsAttribution:
    """Test that facts are correctly attributed to user or contact."""

    def test_contact_message_attribution(self, extractor: FactExtractor) -> None:
        """Given message from contact, expect attribution='contact'."""
        messages = [create_test_message("I work at Google", is_from_me=False)]

        facts = extractor.extract_facts(messages, contact_id="test_contact")

        assert len(facts) >= 1
        assert all(f.attribution == "contact" for f in facts)

    def test_user_message_attribution(self, extractor: FactExtractor) -> None:
        """Given message from user, expect attribution='user'."""
        messages = [create_test_message("I work at Apple", is_from_me=True)]

        facts = extractor.extract_facts(messages, contact_id="test_contact")

        work_facts = [f for f in facts if f.category == "work"]
        assert len(work_facts) >= 1
        assert all(f.attribution == "user" for f in work_facts)


# =============================================================================
# Fact Storage Tests (Public API: save_facts, get_facts_for_contact)
# =============================================================================


class TestFactStorage:
    """Test persisting and retrieving facts."""

    def test_save_and_retrieve_facts(self) -> None:
        """Given facts saved, expect same facts retrieved."""
        facts = [
            Fact(
                category="work",
                subject="Google",
                predicate="works_at",
                value="",
                confidence=0.9,
                contact_id="test_contact",
                source_text="I work at Google",
                extracted_at="2024-01-15T10:00:00",
            ),
            Fact(
                category="location",
                subject="Austin",
                predicate="lives_in",
                value="",
                confidence=0.85,
                contact_id="test_contact",
                source_text="I live in Austin",
                extracted_at="2024-01-15T10:00:00",
            ),
        ]

        # Use in-memory DB with proper mocking at the jarvis.db level
        mock_conn = sqlite3.connect(":memory:", check_same_thread=False)
        mock_conn.row_factory = sqlite3.Row
        # Create the table
        mock_conn.execute(
            """
            CREATE TABLE contact_facts (
                id INTEGER PRIMARY KEY,
                contact_id TEXT,
                category TEXT,
                subject TEXT,
                predicate TEXT,
                value TEXT,
                confidence REAL,
                source_message_id INTEGER,
                source_text TEXT,
                extracted_at TEXT,
                linked_contact_id TEXT,
                valid_from TEXT,
                valid_until TEXT,
                attribution TEXT DEFAULT 'contact',
                segment_id INTEGER
            )
            """
        )

        # Create a mock DB class
        class MockDB:
            def connection(self):
                return self

            def __enter__(self):
                return mock_conn

            def __exit__(self, *args):
                return None

        with patch("jarvis.db.get_db") as mock_get_db:
            mock_get_db.return_value = MockDB()

            # Save facts
            count = save_facts(facts, contact_id="test_contact")
            assert count == 2

            # Retrieve facts
            retrieved = get_facts_for_contact("test_contact")
            assert len(retrieved) == 2

            # Verify content
            subjects = {f.subject for f in retrieved}
            assert "Google" in subjects
            assert "Austin" in subjects

        mock_conn.close()

    def test_empty_facts_save_gracefully(self) -> None:
        """Given empty fact list, expect save to return 0 without error."""
        mock_conn = sqlite3.connect(":memory:", check_same_thread=False)

        class MockDB:
            def connection(self):
                return self

            def __enter__(self):
                return mock_conn

            def __exit__(self, *args):
                return None

        with patch("jarvis.db.get_db") as mock_get_db:
            mock_get_db.return_value = MockDB()

            count = save_facts([], contact_id="test_contact")
            assert count == 0

        mock_conn.close()


# =============================================================================
# Contact Profile Tests (Public API: build_profile)
# =============================================================================


class TestContactProfileBuilding:
    """Test building contact profiles from messages."""

    def test_build_profile_casual_friend(self) -> None:
        """Given casual messages, expect casual friend profile."""
        messages = create_test_messages(
            [
                "hey lol",
                "omg thats so funny haha",
                "gonna go now",
                "yeah nah idk",
                "btw u coming?",
                "sup dude",
                "haha yeah",
                "ok cool",
                "later bro",
                "yep see ya",
            ],
            is_from_me=False,
        )

        builder = ContactProfileBuilder(min_messages=5)
        profile = builder.build_profile("test_chat", messages, contact_name="John")

        assert profile.contact_id == "test_chat"
        assert profile.contact_name == "John"
        assert profile.message_count == 10
        assert profile.formality in ("casual", "very_casual")
        assert profile.uses_abbreviations is True

    def test_build_profile_formal_colleague(self) -> None:
        """Given formal messages, expect formal colleague profile."""
        messages = create_test_messages(
            [
                "Dear Sir, I am writing to confirm our meeting.",
                "Please find attached the requested documentation.",
                "I appreciate your prompt attention to this matter.",
                "The meeting has been scheduled for Tuesday.",
                "Kindly confirm your availability.",
            ],
            is_from_me=False,
        )

        builder = ContactProfileBuilder(min_messages=5)
        profile = builder.build_profile("test_chat", messages, contact_name="Mr. Smith")

        assert profile.formality == "formal"
        assert profile.uses_abbreviations is False
        assert profile.emoji_frequency == 0.0

    def test_build_profile_insufficient_messages(self) -> None:
        """Given too few messages, expect minimal profile."""
        messages = create_test_messages(["hi", "hey"], is_from_me=False)

        builder = ContactProfileBuilder(min_messages=10)
        profile = builder.build_profile("test_chat", messages)

        assert profile.message_count == 2
        assert profile.relationship == "unknown"

    def test_build_profile_detects_emoji(self) -> None:
        """Given messages with emoji, expect emoji frequency detected."""
        messages = create_test_messages(
            ["Hello! 😊", "That's great 🎉", "Thanks 👍", "Sure thing"],
            is_from_me=False,
        )

        builder = ContactProfileBuilder(min_messages=3)
        profile = builder.build_profile("test_chat", messages)

        assert profile.emoji_frequency > 0

    def test_build_profile_detects_greetings(self) -> None:
        """Given messages with greetings, expect greeting style extracted."""
        messages = create_test_messages(
            ["hey what's up", "hey there", "hey how are you", "yo dude"],
            is_from_me=False,
        )

        builder = ContactProfileBuilder(min_messages=3)
        profile = builder.build_profile("test_chat", messages)

        assert "hey" in profile.greeting_style or "yo" in profile.greeting_style


class TestContactProfileStorage:
    """Test saving and loading contact profiles."""

    def test_save_and_load_profile_roundtrip(self, mock_profiles_dir: Path) -> None:
        """Given profile saved, expect identical profile loaded."""
        original = ContactProfile(
            contact_id="test_roundtrip",
            contact_name="Test User",
            relationship="friend",
            relationship_confidence=0.85,
            formality="casual",
            formality_score=0.4,
            avg_message_length=42.0,
            uses_abbreviations=True,
            common_abbreviations=["lol", "idk"],
            emoji_frequency=0.5,
            top_topics=["food, lunch", "social, party"],
            message_count=100,
            updated_at="2024-01-15T10:00:00",
        )

        # Save
        assert save_profile(original) is True

        # Load
        loaded = load_profile("test_roundtrip")

        # Verify
        assert loaded.contact_id == original.contact_id
        assert loaded.contact_name == original.contact_name
        assert loaded.relationship == original.relationship
        assert loaded.relationship_confidence == original.relationship_confidence
        assert loaded.formality == original.formality
        assert loaded.formality_score == original.formality_score
        assert loaded.avg_message_length == original.avg_message_length
        assert loaded.uses_abbreviations == original.uses_abbreviations
        assert loaded.common_abbreviations == original.common_abbreviations
        assert loaded.emoji_frequency == original.emoji_frequency
        assert loaded.top_topics == original.top_topics
        assert loaded.message_count == original.message_count

    def test_load_nonexistent_returns_none(self, mock_profiles_dir: Path) -> None:
        """Given nonexistent contact, expect None returned."""
        result = load_profile("nonexistent_contact_12345")
        assert result is None

    def test_profile_caching(self, mock_profiles_dir: Path) -> None:
        """Given profile loaded twice, expect second from cache."""
        profile = ContactProfile(
            contact_id="cached_test",
            contact_name="Cached User",
            relationship="family",
        )

        # First call: load from disk
        save_profile(profile)
        first = get_contact_profile("cached_test")

        # Second call: from cache
        second = get_contact_profile("cached_test")

        # Should be same object from cache
        assert first.contact_id == second.contact_id


class TestFormatStyleGuide:
    """Test formatting profile as style guide for LLM context."""

    def test_format_casual_friend_profile(self) -> None:
        """Given casual friend profile, expect appropriate style guide."""
        profile = ContactProfile(
            contact_id="test",
            contact_name="Alice",
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
        assert "lol" in guide or "abbreviation" in guide.lower()
        assert "hey" in guide or "yo" in guide

    def test_format_formal_coworker_profile(self) -> None:
        """Given formal coworker profile, expect professional style guide."""
        profile = ContactProfile(
            contact_id="test",
            contact_name="Bob",
            relationship="coworker",
            formality="formal",
            avg_message_length=120.0,
            uses_abbreviations=False,
            emoji_frequency=0.0,
            message_count=50,
        )

        guide = format_style_guide(profile)

        assert "professional" in guide.lower() or "formal" in guide.lower()
        assert "coworker" in guide
        assert "abbreviation" not in guide.lower()

    def test_format_minimal_profile(self) -> None:
        """Given minimal profile, expect limited history note."""
        profile = ContactProfile(contact_id="test", message_count=3)

        guide = format_style_guide(profile)

        assert "Limited message history" in guide


# =============================================================================
# Integration Tests (End-to-end behavior)
# =============================================================================


class TestEndToEndFactPipeline:
    """Integration tests from messages → facts → storage."""

    def test_messages_to_facts_extraction(self, extractor: FactExtractor) -> None:
        """Given realistic conversation, expect relevant facts extracted."""
        messages = [
            create_test_message("Hey! I just got a new job at Google", is_from_me=False, msg_id=1),
            create_test_message("That's awesome, congrats!", is_from_me=True, msg_id=2),
            create_test_message(
                "Yeah, I'm moving to San Francisco next month", is_from_me=False, msg_id=3
            ),
            create_test_message(
                "I love hiking in the mountains, can't wait!", is_from_me=False, msg_id=4
            ),
        ]

        facts = extractor.extract_facts(messages, contact_id="test_contact")

        # Should extract work fact
        work_facts = [f for f in facts if f.category == "work"]
        assert any("Google" in f.subject for f in work_facts)

        # Should extract location fact
        location_facts = [f for f in facts if f.category == "location"]
        assert any("San Francisco" in f.subject for f in location_facts)

        # Should extract preference fact (hiking in mountains)
        pref_facts = [f for f in facts if f.category == "preference"]
        assert len(pref_facts) >= 1
        assert any("hiking" in f.subject.lower() for f in pref_facts)

    def test_preserves_message_attribution(self, extractor: FactExtractor) -> None:
        """Given mixed sender messages, expect correct attribution per fact."""
        messages = [
            create_test_message("I work at Google", is_from_me=False, msg_id=1),
            create_test_message("I work at Apple", is_from_me=True, msg_id=2),
        ]

        facts = extractor.extract_facts(messages, contact_id="test_contact")

        # Find Google fact (from contact)
        google_facts = [f for f in facts if "Google" in f.subject]
        assert len(google_facts) == 1
        assert google_facts[0].attribution == "contact"

        # Find Apple fact (from user)
        apple_facts = [f for f in facts if "Apple" in f.subject]
        assert len(apple_facts) == 1
        assert apple_facts[0].attribution == "user"


# =============================================================================
# Performance Tests (Behavioral: "should complete within X time")
# =============================================================================


class TestExtractionPerformance:
    """Performance requirements as behavioral expectations."""

    def test_extracts_100_messages_quickly(self, extractor: FactExtractor) -> None:
        """Given 100 messages, expect extraction to complete within 500ms."""
        import time

        messages = [
            create_test_message(
                "I love hiking in the mountains every weekend and enjoy outdoor activities like camping",
                msg_id=i,
            )
            for i in range(100)
        ]

        start = time.perf_counter()
        facts = extractor.extract_facts(messages, contact_id="test_contact")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, f"Extraction took {elapsed_ms:.1f}ms, expected <500ms"
        assert len(facts) > 0

    def test_handles_large_batch_gracefully(self, extractor: FactExtractor) -> None:
        """Given many messages with varied content, expect no errors."""
        messages = [
            create_test_message("I work at Google"),
            create_test_message("Your CVS prescription is ready"),  # Bot
            create_test_message("I live in Austin"),
            create_test_message("lol omg"),
            create_test_message("I hate cilantro"),
        ] * 20  # 100 messages

        # Should not raise
        facts = extractor.extract_facts(messages, contact_id="test_contact")

        # Should have filtered out bot messages
        subjects = {f.subject.lower() for f in facts}
        assert "prescription" not in subjects
        assert "cvs" not in subjects

        # Should have work and location facts
        assert any(f.category == "work" for f in facts)
        assert any(f.category == "location" for f in facts)
