"""Contact Profile - Unified per-contact style and topic analysis.

Consolidates all per-contact learned data:
- Writing style (length, formality, abbreviations, emoji usage)
- Discovered topics (centroids, keywords)
- Updated during extraction, cached for generation

This replaces the on-the-fly style analysis in prompts.py with
a cached, pre-computed profile that's updated incrementally.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np

from jarvis.topic_discovery import ContactTopics, DiscoveredTopic, TopicDiscovery

logger = logging.getLogger(__name__)

# Common text abbreviations (from prompts.py, centralized here)
TEXT_ABBREVIATIONS = frozenset({
    "u", "ur", "r", "n", "y", "k", "ok", "kk", "pls", "plz", "thx",
    "ty", "np", "yw", "idk", "idc", "imo", "imho", "tbh", "ngl",
    "fr", "rn", "atm", "btw", "fyi", "lmk", "hmu", "wbu", "hbu",
    "omg", "omw", "otw", "brb", "brt", "ttyl", "gtg", "g2g",
    "lol", "lmao", "lmfao", "rofl", "jk", "jfc", "smh", "nvm",
    "bc", "cuz", "tho", "rly", "sry", "prob", "def", "obvi",
    "whatev", "watever", "w/e", "w/o", "b4", "2day", "2morrow",
    "2nite", "l8r", "l8", "gr8", "m8", "str8", "h8", "w8",
    "gonna", "wanna", "gotta", "kinda", "sorta", "tryna",
    "boutta", "finna", "shoulda", "coulda", "woulda",
    "yea", "yeh", "ya", "yup", "yep", "nah", "nope",
    "aight", "ight", "bet", "facts", "cap", "nocap", "lowkey",
    "highkey", "deadass", "sus", "slay", "fire", "lit", "goat",
})

# Emoji pattern
EMOJI_PATTERN = re.compile(
    r"[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0001F600-\U0001F64F"
    r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]"
)


@dataclass
class StyleProfile:
    """Writing style characteristics for a contact.

    Uses hybrid approach: regex patterns + features from preprocessing
    (spell check, slang detection, NER) for richer style detection.
    """

    # Length statistics
    avg_length: float = 30.0
    min_length: int = 1
    max_length: int = 200

    # Formality (derived from all features)
    formality: Literal["formal", "casual", "very_casual"] = "casual"

    # Basic patterns (regex-based)
    uses_lowercase: bool = False
    uses_abbreviations: bool = False
    uses_minimal_punctuation: bool = True
    common_abbreviations: list[str] = field(default_factory=list)

    # Frequencies
    emoji_frequency: float = 0.0  # per message
    exclamation_frequency: float = 0.0  # per message

    # From spell checker (preprocessing)
    spell_error_rate: float = 0.0  # % of words with corrections
    # High = casual/fast typer, Low = careful writer

    # From slang detection (preprocessing)
    slang_frequency: float = 0.0  # slang words per message
    slang_types: list[str] = field(default_factory=list)  # which slang they use

    # Vocabulary richness
    vocabulary_diversity: float = 0.0  # unique words / total words
    avg_words_per_message: float = 0.0

    # Message count used for analysis
    message_count: int = 0

    def to_dict(self) -> dict:
        return {
            "avg_length": self.avg_length,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "formality": self.formality,
            "uses_lowercase": self.uses_lowercase,
            "uses_abbreviations": self.uses_abbreviations,
            "uses_minimal_punctuation": self.uses_minimal_punctuation,
            "common_abbreviations": self.common_abbreviations,
            "emoji_frequency": self.emoji_frequency,
            "exclamation_frequency": self.exclamation_frequency,
            "spell_error_rate": self.spell_error_rate,
            "slang_frequency": self.slang_frequency,
            "slang_types": self.slang_types,
            "vocabulary_diversity": self.vocabulary_diversity,
            "avg_words_per_message": self.avg_words_per_message,
            "message_count": self.message_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StyleProfile:
        return cls(
            avg_length=data.get("avg_length", 30.0),
            min_length=data.get("min_length", 1),
            max_length=data.get("max_length", 200),
            formality=data.get("formality", "casual"),
            uses_lowercase=data.get("uses_lowercase", False),
            uses_abbreviations=data.get("uses_abbreviations", False),
            uses_minimal_punctuation=data.get("uses_minimal_punctuation", True),
            common_abbreviations=data.get("common_abbreviations", []),
            emoji_frequency=data.get("emoji_frequency", 0.0),
            exclamation_frequency=data.get("exclamation_frequency", 0.0),
            spell_error_rate=data.get("spell_error_rate", 0.0),
            slang_frequency=data.get("slang_frequency", 0.0),
            slang_types=data.get("slang_types", []),
            vocabulary_diversity=data.get("vocabulary_diversity", 0.0),
            avg_words_per_message=data.get("avg_words_per_message", 0.0),
            message_count=data.get("message_count", 0),
        )


@dataclass
class ContactProfile:
    """Complete profile for a contact - style + topics."""

    contact_id: str
    style: StyleProfile = field(default_factory=StyleProfile)
    topics: ContactTopics | None = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "contact_id": self.contact_id,
            "style": self.style.to_dict(),
            "topics": {
                "contact_id": self.topics.contact_id,
                "topics": [t.to_dict() for t in self.topics.topics],
                "noise_count": self.topics.noise_count,
            } if self.topics else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ContactProfile:
        """Deserialize from storage."""
        topics_data = data.get("topics")
        topics = None
        if topics_data:
            topics = ContactTopics(
                contact_id=topics_data["contact_id"],
                topics=[DiscoveredTopic.from_dict(t) for t in topics_data["topics"]],
                noise_count=topics_data.get("noise_count", 0),
            )

        created = data.get("created_at")
        updated = data.get("updated_at")
        return cls(
            contact_id=data["contact_id"],
            style=StyleProfile.from_dict(data.get("style", {})),
            topics=topics,
            created_at=datetime.fromisoformat(created) if created else datetime.now(),
            updated_at=datetime.fromisoformat(updated) if updated else datetime.now(),
            version=data.get("version", 1),
        )


class ContactProfiler:
    """Builds and manages contact profiles."""

    def __init__(self, profile_dir: Path | None = None):
        """Initialize the profiler.

        Args:
            profile_dir: Directory to store profiles. Defaults to ~/.jarvis/profiles/
        """
        if profile_dir is None:
            profile_dir = Path.home() / ".jarvis" / "profiles"
        self.profile_dir = profile_dir
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._topic_discovery = TopicDiscovery()
        self._cache: dict[str, ContactProfile] = {}

    def analyze_style(self, messages: list[str]) -> StyleProfile:
        """Analyze writing style from messages.

        Uses hybrid approach:
        1. Regex patterns (fast, interpretable)
        2. Features from preprocessing (spell check, slang detection)

        This leverages our existing preprocessing pipeline for richer style signals.
        """
        if not messages:
            return StyleProfile()

        # Filter to non-empty
        messages = [m for m in messages if m and m.strip()]
        if not messages:
            return StyleProfile()

        # === BASIC STATS ===
        lengths = [len(m) for m in messages]
        avg_length = sum(lengths) / len(lengths)
        min_length = min(lengths)
        max_length = max(lengths)

        # Word counts
        all_words = []
        for msg in messages:
            all_words.extend(re.findall(r"\b\w+\b", msg.lower()))
        total_words = len(all_words)
        unique_words = len(set(all_words))
        vocabulary_diversity = unique_words / max(total_words, 1)
        avg_words = total_words / len(messages)

        # === REGEX-BASED PATTERNS ===

        # Case analysis
        lowercase_count = 0
        for msg in messages:
            letters = re.sub(r"[^a-zA-Z]", "", msg)
            if letters:
                ratio = sum(1 for c in letters if c.islower()) / len(letters)
                if ratio > 0.9:
                    lowercase_count += 1
        uses_lowercase = lowercase_count / len(messages) > 0.7

        # Abbreviations (from our slang list)
        words_set = set(all_words)
        found_abbrevs = list(words_set & TEXT_ABBREVIATIONS)
        uses_abbreviations = len(found_abbrevs) >= 2

        # Punctuation
        total_chars = sum(len(m) for m in messages)
        total_exclamations = sum(m.count("!") for m in messages)
        total_periods = sum(m.count(".") for m in messages)
        excl_density = total_exclamations / max(total_chars, 1)
        period_density = total_periods / max(total_chars, 1)
        uses_minimal_punctuation = excl_density < 0.02 and period_density < 0.03

        # Emoji frequency
        emoji_count = sum(len(EMOJI_PATTERN.findall(m)) for m in messages)
        emoji_freq = emoji_count / len(messages)
        excl_freq = total_exclamations / len(messages)

        # === FROM PREPROCESSING: SLANG DETECTION ===
        slang_count = 0
        slang_found: set[str] = set()
        try:
            from jarvis.slang import SLANG_MAP
            for word in all_words:
                if word in SLANG_MAP:
                    slang_count += 1
                    slang_found.add(word)
        except ImportError:
            pass
        slang_freq = slang_count / len(messages) if messages else 0

        # === FROM PREPROCESSING: SPELL CHECK ===
        spell_errors = 0
        try:
            from jarvis.text_normalizer import _get_spell_checker
            checker = _get_spell_checker()
            if checker:
                for word in all_words:
                    if len(word) >= 3:  # Skip short words
                        suggestions = checker.lookup(word, verbosity=0, max_edit_distance=1)
                        if suggestions and suggestions[0].term != word:
                            spell_errors += 1
        except Exception:
            pass
        spell_error_rate = spell_errors / max(total_words, 1)

        # === FORMALITY (using all signals) ===
        casual_words = {"lol", "haha", "omg", "btw", "gonna", "wanna", "yeah", "nah", "bro", "dude"}
        casual_count = len(words_set & casual_words)

        # Enhanced formality detection using multiple signals
        informality_score = 0
        if uses_lowercase:
            informality_score += 2
        if uses_abbreviations:
            informality_score += 2
        if slang_freq > 0.5:
            informality_score += 2
        if spell_error_rate > 0.1:
            informality_score += 1
        if casual_count >= 2:
            informality_score += 2
        if avg_length < 30:
            informality_score += 1

        if informality_score >= 5:
            formality: Literal["formal", "casual", "very_casual"] = "very_casual"
        elif informality_score >= 2:
            formality = "casual"
        else:
            formality = "formal"

        return StyleProfile(
            avg_length=round(avg_length, 1),
            min_length=min_length,
            max_length=max_length,
            formality=formality,
            uses_lowercase=uses_lowercase,
            uses_abbreviations=uses_abbreviations,
            uses_minimal_punctuation=uses_minimal_punctuation,
            common_abbreviations=found_abbrevs[:5],
            emoji_frequency=round(emoji_freq, 2),
            exclamation_frequency=round(excl_freq, 2),
            spell_error_rate=round(spell_error_rate, 3),
            slang_frequency=round(slang_freq, 2),
            slang_types=list(slang_found)[:10],
            vocabulary_diversity=round(vocabulary_diversity, 2),
            avg_words_per_message=round(avg_words, 1),
            message_count=len(messages),
        )

    def build_profile(
        self,
        contact_id: str,
        messages: list[str],
        embeddings: np.ndarray | None = None,
    ) -> ContactProfile:
        """Build a complete profile for a contact.

        Args:
            contact_id: Unique contact identifier
            messages: List of message texts from this contact
            embeddings: Pre-computed embeddings (N, 384). If None, topics not computed.

        Returns:
            ContactProfile with style and topics
        """
        # Analyze style
        style = self.analyze_style(messages)

        # Discover topics if embeddings provided
        topics = None
        if embeddings is not None and len(embeddings) > 0:
            topics = self._topic_discovery.discover_topics(
                contact_id=contact_id,
                embeddings=embeddings,
                texts=messages,
            )

        profile = ContactProfile(
            contact_id=contact_id,
            style=style,
            topics=topics,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # Cache and save
        self._cache[contact_id] = profile
        self._save_profile(profile)

        return profile

    def get_profile(self, contact_id: str) -> ContactProfile | None:
        """Get a cached or stored profile."""
        # Check cache
        if contact_id in self._cache:
            return self._cache[contact_id]

        # Load from disk
        profile = self._load_profile(contact_id)
        if profile:
            self._cache[contact_id] = profile
        return profile

    def _get_profile_path(self, contact_id: str) -> Path:
        """Get the file path for a contact's profile."""
        # Hash the contact_id for privacy
        import hashlib
        hashed = hashlib.sha256(contact_id.encode()).hexdigest()[:16]
        return self.profile_dir / f"{hashed}.json"

    def _save_profile(self, profile: ContactProfile) -> None:
        """Save profile to disk."""
        path = self._get_profile_path(profile.contact_id)
        try:
            with open(path, "w") as f:
                json.dump(profile.to_dict(), f, indent=2)
            logger.debug(f"Saved profile for {profile.contact_id[:8]}...")
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")

    def _load_profile(self, contact_id: str) -> ContactProfile | None:
        """Load profile from disk."""
        path = self._get_profile_path(contact_id)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return ContactProfile.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load profile: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()


# Singleton
_profiler: ContactProfiler | None = None


def get_profiler() -> ContactProfiler:
    """Get the singleton ContactProfiler."""
    global _profiler
    if _profiler is None:
        _profiler = ContactProfiler()
    return _profiler


def get_contact_profile(contact_id: str) -> ContactProfile | None:
    """Convenience function to get a contact's profile."""
    return get_profiler().get_profile(contact_id)


def build_contact_profile(
    contact_id: str,
    messages: list[str],
    embeddings: np.ndarray | None = None,
) -> ContactProfile:
    """Convenience function to build a contact's profile."""
    return get_profiler().build_profile(contact_id, messages, embeddings)
