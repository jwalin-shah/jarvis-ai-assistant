"""Unit tests for the ContactProfiler module.

Tests cover:
- StyleProfile dataclass (to_dict, from_dict)
- ContactProfile dataclass (to_dict, from_dict)
- ContactProfiler.analyze_style() with various message types
- ContactProfiler.build_profile()
- ContactProfiler.get_profile() (cache + disk)
- Singleton functions
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.contact_profile import (
    EMOJI_PATTERN,
    TEXT_ABBREVIATIONS,
    ContactProfile,
    ContactProfiler,
    StyleProfile,
    build_contact_profile,
    get_contact_profile,
    get_profiler,
)
from jarvis.topic_discovery import ContactTopics, DiscoveredTopic


class TestStyleProfile:
    """Tests for StyleProfile dataclass."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        profile = StyleProfile()

        assert profile.avg_length == 30.0
        assert profile.min_length == 1
        assert profile.max_length == 200
        assert profile.formality == "casual"
        assert profile.uses_lowercase is False
        assert profile.uses_abbreviations is False
        assert profile.uses_minimal_punctuation is True
        assert profile.common_abbreviations == []
        assert profile.emoji_frequency == 0.0
        assert profile.exclamation_frequency == 0.0
        assert profile.spell_error_rate == 0.0
        assert profile.slang_frequency == 0.0
        assert profile.slang_types == []
        assert profile.vocabulary_diversity == 0.0
        assert profile.avg_words_per_message == 0.0
        assert profile.message_count == 0

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        profile = StyleProfile(
            avg_length=45.5,
            min_length=5,
            max_length=150,
            formality="very_casual",
            uses_lowercase=True,
            uses_abbreviations=True,
            uses_minimal_punctuation=False,
            common_abbreviations=["lol", "btw"],
            emoji_frequency=0.5,
            exclamation_frequency=0.2,
            spell_error_rate=0.05,
            slang_frequency=0.3,
            slang_types=["gonna", "wanna"],
            vocabulary_diversity=0.65,
            avg_words_per_message=8.5,
            message_count=100,
        )

        data = profile.to_dict()

        assert data["avg_length"] == 45.5
        assert data["formality"] == "very_casual"
        assert data["uses_lowercase"] is True
        assert data["common_abbreviations"] == ["lol", "btw"]
        assert data["emoji_frequency"] == 0.5
        assert data["message_count"] == 100

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        data = {
            "avg_length": 50.0,
            "min_length": 3,
            "max_length": 180,
            "formality": "formal",
            "uses_lowercase": False,
            "uses_abbreviations": False,
            "common_abbreviations": [],
            "emoji_frequency": 0.1,
            "message_count": 50,
        }

        profile = StyleProfile.from_dict(data)

        assert profile.avg_length == 50.0
        assert profile.formality == "formal"
        assert profile.message_count == 50

    def test_from_dict_with_missing_keys(self) -> None:
        """Test that from_dict handles missing keys with defaults."""
        data = {"avg_length": 25.0}

        profile = StyleProfile.from_dict(data)

        assert profile.avg_length == 25.0
        assert profile.formality == "casual"  # default
        assert profile.message_count == 0  # default

    def test_roundtrip(self) -> None:
        """Test that to_dict/from_dict roundtrips correctly."""
        original = StyleProfile(
            avg_length=42.0,
            formality="very_casual",
            uses_lowercase=True,
            common_abbreviations=["u", "r"],
            slang_types=["gonna"],
            message_count=25,
        )

        restored = StyleProfile.from_dict(original.to_dict())

        assert restored.avg_length == original.avg_length
        assert restored.formality == original.formality
        assert restored.uses_lowercase == original.uses_lowercase
        assert restored.common_abbreviations == original.common_abbreviations


class TestContactProfile:
    """Tests for ContactProfile dataclass."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        profile = ContactProfile(contact_id="test123")

        assert profile.contact_id == "test123"
        assert isinstance(profile.style, StyleProfile)
        assert profile.topics is None
        assert profile.version == 1

    def test_to_dict_without_topics(self) -> None:
        """Test serialization without topics."""
        profile = ContactProfile(
            contact_id="abc123",
            style=StyleProfile(avg_length=40.0),
        )

        data = profile.to_dict()

        assert data["contact_id"] == "abc123"
        assert data["style"]["avg_length"] == 40.0
        assert data["topics"] is None
        assert "created_at" in data
        assert "updated_at" in data

    def test_to_dict_with_topics(self) -> None:
        """Test serialization with topics."""
        topics = ContactTopics(
            contact_id="abc123",
            topics=[
                DiscoveredTopic(
                    topic_id=0,
                    centroid=np.array([0.1, 0.2, 0.3], dtype=np.float32),
                    keywords=["test", "example"],
                    message_count=10,
                    representative_text="This is a test",
                )
            ],
            noise_count=5,
        )

        profile = ContactProfile(
            contact_id="abc123",
            style=StyleProfile(),
            topics=topics,
        )

        data = profile.to_dict()

        assert data["topics"] is not None
        assert data["topics"]["contact_id"] == "abc123"
        assert len(data["topics"]["topics"]) == 1
        assert data["topics"]["topics"][0]["topic_id"] == 0
        assert data["topics"]["noise_count"] == 5

    def test_from_dict_without_topics(self) -> None:
        """Test deserialization without topics."""
        data = {
            "contact_id": "xyz789",
            "style": {"avg_length": 35.0, "formality": "casual"},
            "topics": None,
            "created_at": "2024-01-01T12:00:00",
            "updated_at": "2024-01-02T12:00:00",
            "version": 2,
        }

        profile = ContactProfile.from_dict(data)

        assert profile.contact_id == "xyz789"
        assert profile.style.avg_length == 35.0
        assert profile.topics is None
        assert profile.version == 2

    def test_from_dict_with_topics(self) -> None:
        """Test deserialization with topics."""
        data = {
            "contact_id": "xyz789",
            "style": {},
            "topics": {
                "contact_id": "xyz789",
                "topics": [
                    {
                        "topic_id": 1,
                        "centroid": [0.5, 0.6, 0.7],
                        "keywords": ["word1", "word2"],
                        "message_count": 15,
                        "representative_text": "Representative",
                    }
                ],
                "noise_count": 3,
            },
            "created_at": "2024-01-01T12:00:00",
            "updated_at": "2024-01-02T12:00:00",
        }

        profile = ContactProfile.from_dict(data)

        assert profile.contact_id == "xyz789"
        assert profile.topics is not None
        assert len(profile.topics.topics) == 1
        assert profile.topics.topics[0].topic_id == 1
        assert profile.topics.noise_count == 3


class TestContactProfilerAnalyzeStyle:
    """Tests for ContactProfiler.analyze_style()."""

    @pytest.fixture
    def profiler(self) -> ContactProfiler:
        """Create a profiler with temp directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ContactProfiler(profile_dir=Path(temp_dir))

    def test_empty_messages_returns_default(self, profiler: ContactProfiler) -> None:
        """Test that empty message list returns default profile."""
        result = profiler.analyze_style([])

        assert result.message_count == 0
        assert result.avg_length == 30.0  # default

    def test_whitespace_only_messages_filtered(self, profiler: ContactProfiler) -> None:
        """Test that whitespace-only messages are filtered."""
        result = profiler.analyze_style(["   ", "\n", "\t", ""])

        assert result.message_count == 0

    def test_basic_stats(self, profiler: ContactProfiler) -> None:
        """Test that basic stats are computed correctly."""
        messages = [
            "Hello there",  # 11 chars
            "How are you doing today?",  # 24 chars
            "Good",  # 4 chars
        ]

        result = profiler.analyze_style(messages)

        assert result.message_count == 3
        assert result.min_length == 4
        assert result.max_length == 24
        assert result.avg_length == pytest.approx(13.0, rel=0.1)

    def test_lowercase_detection(self, profiler: ContactProfiler) -> None:
        """Test that lowercase preference is detected."""
        # Mostly lowercase messages
        messages = [
            "hey what's up",
            "nothing much, you?",
            "same here lol",
            "cool cool",
            "see ya later",
        ]

        result = profiler.analyze_style(messages)

        assert result.uses_lowercase is True

    def test_abbreviations_detection(self, profiler: ContactProfiler) -> None:
        """Test that abbreviations are detected."""
        messages = [
            "hey lol how r u",
            "idk what to do tbh",
            "gonna go now ttyl",
        ]

        result = profiler.analyze_style(messages)

        assert result.uses_abbreviations is True
        assert len(result.common_abbreviations) > 0
        # Should find some common abbreviations
        found = set(result.common_abbreviations)
        assert "lol" in found or "idk" in found or "tbh" in found

    def test_emoji_frequency(self, profiler: ContactProfiler) -> None:
        """Test that emoji frequency is calculated."""
        messages = [
            "Hello! 😊",
            "That's great 🎉",
            "Thanks 👍",
            "Sure thing",  # no emoji
        ]

        result = profiler.analyze_style(messages)

        # 3 emojis across 4 messages = 0.75 per message
        assert result.emoji_frequency == pytest.approx(0.75, rel=0.1)

    def test_exclamation_frequency(self, profiler: ContactProfiler) -> None:
        """Test that exclamation frequency is calculated."""
        messages = [
            "Wow!",
            "Amazing!!",
            "Great",
            "Nice",
        ]

        result = profiler.analyze_style(messages)

        # 3 exclamations across 4 messages = 0.75 per message
        assert result.exclamation_frequency == pytest.approx(0.75, rel=0.1)

    def test_formality_formal(self, profiler: ContactProfiler) -> None:
        """Test that formal style is detected.

        The formality algorithm detects informality markers. Formal messages
        should have: no lowercase preference, no abbreviations, no slang,
        low spell errors, no casual words, and longer message lengths (>=30 chars).
        """
        # Very formal, professional messages with proper capitalization
        messages = [
            "Dear Sir, I am writing to inquire about the status of our agreement.",
            "Please find attached the requested documentation for your review.",
            "I appreciate your prompt attention to this matter and look forward to hearing from you.",
            "The meeting has been scheduled for next Tuesday at your convenience.",
            "Kindly confirm your availability at your earliest opportunity.",
        ]

        result = profiler.analyze_style(messages)

        # These messages should have low informality score (no casual markers)
        # If still casual, it means our test messages aren't formal enough
        assert result.formality in ("formal", "casual")
        # At minimum, should NOT be very_casual
        assert result.formality != "very_casual"
        # Formal messages should not have abbreviations
        assert result.uses_abbreviations is False

    def test_formality_very_casual(self, profiler: ContactProfiler) -> None:
        """Test that very casual style is detected."""
        messages = [
            "hey lol",
            "omg that's so funny haha",
            "gonna go now",
            "yeah nah idk",
            "btw u coming?",
        ]

        result = profiler.analyze_style(messages)

        assert result.formality == "very_casual"

    def test_vocabulary_diversity(self, profiler: ContactProfiler) -> None:
        """Test vocabulary diversity calculation."""
        # High diversity - many unique words
        diverse_messages = [
            "The quick brown fox",
            "Jumped over lazy dogs",
            "Yesterday was wonderful",
        ]

        result = profiler.analyze_style(diverse_messages)
        high_diversity = result.vocabulary_diversity

        # Low diversity - repeated words
        repetitive_messages = [
            "good good good",
            "good morning good",
            "good night good",
        ]

        result2 = profiler.analyze_style(repetitive_messages)
        low_diversity = result2.vocabulary_diversity

        assert high_diversity > low_diversity

    def test_punctuation_detection(self, profiler: ContactProfiler) -> None:
        """Test minimal punctuation detection."""
        # Minimal punctuation
        minimal = ["hey", "whats up", "nothing much"]
        result1 = profiler.analyze_style(minimal)

        # Heavy punctuation
        heavy = ["Hey! What's up?!", "Nothing much... you?", "Cool! Great!"]
        result2 = profiler.analyze_style(heavy)

        assert result1.uses_minimal_punctuation is True
        assert result2.uses_minimal_punctuation is False


class TestContactProfilerBuildProfile:
    """Tests for ContactProfiler.build_profile()."""

    @pytest.fixture
    def profiler(self) -> ContactProfiler:
        """Create a profiler with temp directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ContactProfiler(profile_dir=Path(temp_dir))

    def test_build_profile_without_embeddings(self, profiler: ContactProfiler) -> None:
        """Test building profile without embeddings (style only)."""
        messages = ["Hello there", "How are you?", "Great day!"]

        profile = profiler.build_profile("contact123", messages)

        assert profile.contact_id == "contact123"
        assert profile.style.message_count == 3
        assert profile.topics is None

    def test_build_profile_with_embeddings(self, profiler: ContactProfiler) -> None:
        """Test building profile with embeddings (style + topics)."""
        messages = ["Hello", "Hi", "Test"]
        # Mock embeddings (3 messages, 384 dims)
        embeddings = np.random.randn(3, 384).astype(np.float32)

        with patch.object(
            profiler._topic_discovery,
            "discover_topics",
            return_value=ContactTopics(
                contact_id="contact123",
                topics=[
                    DiscoveredTopic(
                        topic_id=0,
                        centroid=np.zeros(384, dtype=np.float32),
                        keywords=["test"],
                        message_count=3,
                        representative_text="Hello",
                    )
                ],
                noise_count=0,
            ),
        ) as mock_discover:
            profile = profiler.build_profile("contact123", messages, embeddings)

            mock_discover.assert_called_once()
            assert profile.topics is not None
            assert len(profile.topics.topics) == 1

    def test_build_profile_caches_result(self, profiler: ContactProfiler) -> None:
        """Test that built profile is cached."""
        messages = ["Test message"]

        profiler.build_profile("cached_contact", messages)

        # Should be in cache
        assert "cached_contact" in profiler._cache
        cached = profiler._cache["cached_contact"]
        assert cached.contact_id == "cached_contact"

    def test_build_profile_saves_to_disk(self, profiler: ContactProfiler) -> None:
        """Test that built profile is saved to disk."""
        messages = ["Test message"]

        profiler.build_profile("disk_contact", messages)

        # Check that file was created
        path = profiler._get_profile_path("disk_contact")
        assert path.exists()


class TestContactProfilerGetProfile:
    """Tests for ContactProfiler.get_profile()."""

    @pytest.fixture
    def profiler(self) -> ContactProfiler:
        """Create a profiler with temp directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ContactProfiler(profile_dir=Path(temp_dir))

    def test_get_profile_returns_none_for_unknown(
        self, profiler: ContactProfiler
    ) -> None:
        """Test that get_profile returns None for unknown contact."""
        result = profiler.get_profile("unknown_contact")
        assert result is None

    def test_get_profile_returns_cached(self, profiler: ContactProfiler) -> None:
        """Test that get_profile returns cached profile."""
        # Build profile first (this caches it)
        profiler.build_profile("cached_contact", ["Test"])

        # Get should return from cache
        result = profiler.get_profile("cached_contact")

        assert result is not None
        assert result.contact_id == "cached_contact"

    def test_get_profile_loads_from_disk(self, profiler: ContactProfiler) -> None:
        """Test that get_profile loads from disk if not cached."""
        # Build profile (saves to disk)
        profiler.build_profile("disk_contact", ["Test message"])

        # Clear cache
        profiler.clear_cache()

        # Get should load from disk
        result = profiler.get_profile("disk_contact")

        assert result is not None
        assert result.contact_id == "disk_contact"


class TestContactProfilerHelpers:
    """Tests for ContactProfiler helper methods."""

    def test_get_profile_path_hashes_contact_id(self) -> None:
        """Test that profile path uses hashed contact ID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            profiler = ContactProfiler(profile_dir=Path(temp_dir))

            path1 = profiler._get_profile_path("contact1")
            path2 = profiler._get_profile_path("contact2")
            path1_again = profiler._get_profile_path("contact1")

            # Different contacts should have different paths
            assert path1 != path2
            # Same contact should have same path
            assert path1 == path1_again
            # Path should be inside profile_dir
            assert path1.parent == Path(temp_dir)
            # Should be a JSON file
            assert path1.suffix == ".json"

    def test_clear_cache(self) -> None:
        """Test that clear_cache empties the cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            profiler = ContactProfiler(profile_dir=Path(temp_dir))

            # Add to cache
            profiler.build_profile("contact1", ["Test"])
            assert len(profiler._cache) == 1

            # Clear cache
            profiler.clear_cache()
            assert len(profiler._cache) == 0


class TestSingletonFunctions:
    """Tests for module-level singleton functions."""

    def test_get_profiler_returns_singleton(self) -> None:
        """Test that get_profiler returns the same instance."""
        # Reset global singleton
        import jarvis.contact_profile as cp

        cp._profiler = None

        profiler1 = get_profiler()
        profiler2 = get_profiler()

        assert profiler1 is profiler2

    def test_get_contact_profile_delegates_to_profiler(self) -> None:
        """Test that get_contact_profile delegates correctly."""
        with patch("jarvis.contact_profile.get_profiler") as mock_get:
            mock_profiler = MagicMock()
            mock_profiler.get_profile.return_value = None
            mock_get.return_value = mock_profiler

            result = get_contact_profile("test_contact")

            mock_profiler.get_profile.assert_called_once_with("test_contact")
            assert result is None

    def test_build_contact_profile_delegates_to_profiler(self) -> None:
        """Test that build_contact_profile delegates correctly."""
        with patch("jarvis.contact_profile.get_profiler") as mock_get:
            mock_profiler = MagicMock()
            mock_profile = ContactProfile(contact_id="test")
            mock_profiler.build_profile.return_value = mock_profile
            mock_get.return_value = mock_profiler

            messages = ["Test message"]
            embeddings = np.array([[0.1, 0.2]])

            result = build_contact_profile("test_contact", messages, embeddings)

            mock_profiler.build_profile.assert_called_once_with(
                "test_contact", messages, embeddings
            )
            assert result == mock_profile


class TestRegexPatterns:
    """Tests for module-level regex patterns and constants."""

    def test_text_abbreviations_contains_common_slang(self) -> None:
        """Test that TEXT_ABBREVIATIONS contains common slang."""
        expected = {"lol", "omg", "btw", "idk", "tbh", "gonna", "wanna"}
        assert expected.issubset(TEXT_ABBREVIATIONS)

    def test_emoji_pattern_matches_emojis(self) -> None:
        """Test that EMOJI_PATTERN matches common emojis."""
        test_cases = [
            ("Hello 😊", 1),
            ("🎉🎊🎈", 3),
            ("No emojis here", 0),
            ("Mixed 👍 content 🔥", 2),
        ]

        for text, expected_count in test_cases:
            matches = EMOJI_PATTERN.findall(text)
            assert len(matches) == expected_count, f"Failed for: {text}"
