"""Relationship classifier using embeddings and message patterns.

Automatically classifies relationships between contacts based on
message patterns and semantic analysis. Works on ALL messages with
a contact (not just extracted pairs).

Usage:
    from jarvis.classifiers.relationship_classifier import RelationshipClassifier

    classifier = RelationshipClassifier()

    # Classify a contact
    result = classifier.classify_contact("chat1234")
    print(result.relationship)  # e.g., "close friend", "family", "coworker"

    # Classify from messages directly
    result = classifier.classify_messages(messages)

    # Auto-classify all contacts
    results = classifier.classify_all_contacts()
"""

from __future__ import annotations

import contextlib
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from jarvis.contracts.pipeline import Relationship as ContractRelationship

# Cache TTL for classification results (5 minutes)
_CLASSIFICATION_CACHE_TTL_SECONDS = 300
_classification_cache: dict[str, tuple[float, ClassificationResult]] = {}
_cache_lock = threading.Lock()
_CLASSIFICATION_CACHE_MAX_SIZE = 500

logger = logging.getLogger(__name__)

# Pre-compiled pattern for display name tokenization
_NAME_SPLIT_RE = re.compile(r"[\s\-_]+")


# =============================================================================
# Helper Functions
# =============================================================================


def _is_emoji(text: str) -> bool:
    """Check if text contains emoji characters using unicode ranges.

    Checks against common emoji ranges to determine if the text
    should be classified as emoji rather than regular text.

    Args:
        text: Text to check

    Returns:
        True if any character in text is an emoji.
    """
    for char in text:
        code = ord(char)
        # Emoji ranges: 0x1F300-0x1F9FF covers most emoji
        # 0x2600-0x27BF covers miscellaneous symbols including emoji
        # 0x1F000-0x1F02F covers mahjong and dominoes
        if 0x1F300 <= code <= 0x1F9FF or 0x2600 <= code <= 0x27BF or 0x1F000 <= code <= 0x1F02F:
            return True
    return False


# =============================================================================
# Relationship Categories
# =============================================================================

RELATIONSHIP_CATEGORIES = {
    "family": {
        "label": "family",
        "keywords": [
            "mom",
            "dad",
            "mother",
            "father",
            "sister",
            "brother",
            "grandma",
            "grandpa",
            "aunt",
            "uncle",
            "cousin",
            "family",
            "love you",
            "miss you",
            "thanksgiving",
            "christmas",
            "birthday",
            "come home",
            "dinner at home",
            "parents",
            "son",
            "daughter",
            "kid",
            "kids",
        ],
        "anti_keywords": [  # Keywords that suggest NOT this category
            "meeting",
            "deadline",
            "project",
            "sprint",
            "standup",
        ],
        "patterns": {
            "time_any_hour": True,  # Family texts at any hour
            "high_emoji_rate": True,  # Lots of hearts, etc.
            "medium_frequency": True,  # Regular but not constant
        },
    },
    "close friend": {
        "label": "close friend",
        "keywords": [
            "dude",
            "bro",
            "lol",
            "lmao",
            "haha",
            "omg",
            "hangout",
            "drinks",
            "party",
            "weekend",
            "chill",
            "wanna",
            "gonna",
            "miss you",
            "bestie",
            "squad",
            "crew",
            "yo",
            "bruh",
            "bar",
            "club",
            "concert",
            "game",
            "watch the game",
        ],
        "anti_keywords": [
            "deadline",
            "sprint",
            "standup",
            "sync",
            "meeting",
        ],
        "patterns": {
            "casual_language": True,
            "late_night_messages": True,
            "high_emoji_rate": True,
        },
    },
    "romantic partner": {
        "label": "romantic partner",
        "keywords": [
            "love you",
            "miss you so much",
            "baby",
            "babe",
            "honey",
            "sweetheart",
            "date night",
            "can't wait to see you",
            "thinking of you",
            "good morning",
            "good night",
            "sleep well",
            "dream of me",
            "❤️",
            "😘",
            "💕",
            "💗",
            "💞",
            "xoxo",
            "kisses",
        ],
        "anti_keywords": [],
        "patterns": {
            "very_high_frequency": True,
            "time_any_hour": True,
            "high_emoji_rate": True,
            "long_messages": True,
        },
    },
    "coworker": {
        "label": "coworker",
        "keywords": [
            "meeting",
            "deadline",
            "project",
            "office",
            "presentation",
            "client",
            "slack",
            "zoom",
            "standup",
            "sprint",
            "manager",
            "boss",
            "team",
            "eod",
            "eow",
            "fyi",
            "asap",
            "sync",
            "review",
            "pr",
            "merge",
            "deploy",
        ],
        "anti_keywords": [
            "love you",
            "miss you",
            "party",
            "drinks",
            "hangout",
        ],
        "patterns": {
            "work_hours_only": True,
            "formal_language": True,
            "low_emoji_rate": True,
        },
    },
    "acquaintance": {
        "label": "acquaintance",
        "keywords": [
            "long time",
            "nice to meet",
            "good seeing you",
            "let's catch up",
        ],
        "anti_keywords": [],
        "patterns": {
            "low_frequency": True,
            "short_messages": True,
            "formal_language": True,
        },
    },
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ClassificationResult:
    """Result of relationship classification."""

    chat_id: str
    display_name: str
    relationship: str
    confidence: float
    signals: dict[str, float] = field(default_factory=dict)
    sample_size: int = 0
    features: dict[str, float] = field(default_factory=dict)

    def to_contract_relationship(self) -> ContractRelationship:
        """Convert to pipeline Relationship contract type."""
        return ContractRelationship(
            source_entity=self.display_name or self.chat_id,
            target_entity="user",
            relation_type=self.relationship,
            confidence=self.confidence,
        )


@dataclass
class MessageFeatures:
    """Extracted features from message history."""

    avg_message_length: float
    message_frequency_per_day: float
    emoji_rate: float
    avg_hour_of_day: float
    hour_variance: float  # High variance = texts at any hour
    formal_language_score: float
    keyword_matches: dict[str, int] = field(default_factory=dict)
    anti_keyword_matches: dict[str, int] = field(default_factory=dict)
    total_messages: int = 0
    date_range_days: int = 0
    my_messages_pct: float = 0.5  # % of messages I sent


@dataclass
class ChatMessage:
    """Lightweight message for classification."""

    text: str
    is_from_me: bool
    date: datetime | None
    chat_id: str


# =============================================================================
# Classifier
# =============================================================================


class RelationshipClassifier:
    """Classifies relationships based on message patterns.

    Uses a combination of:
    - Keyword matching
    - Time patterns (when messages are sent)
    - Message style (formal vs casual)
    - Response patterns

    Does NOT use LLM - fast and deterministic.
    """

    def __init__(
        self,
        chat_db_path: Path | None = None,
        min_messages: int = 10,
    ):
        """Initialize the classifier.

        Args:
            chat_db_path: Path to iMessage chat.db. Uses default if None.
            min_messages: Minimum messages needed for classification.
        """
        self.chat_db_path = chat_db_path or Path.home() / "Library/Messages/chat.db"
        self.min_messages = min_messages

        # Pre-compile keyword/anti-keyword patterns per category
        # Emoji keywords are escaped and compiled into the same regex pattern
        # as text keywords, avoiding a separate per-emoji string search loop.
        self._keyword_patterns: dict[str, re.Pattern[str] | None] = {}
        self._anti_keyword_patterns: dict[str, re.Pattern[str] | None] = {}

        for cat, info in RELATIONSHIP_CATEGORIES.items():
            all_kw_parts = []
            for kw in info["keywords"]:
                kw_lower = kw.lower()
                if _is_emoji(kw_lower):
                    all_kw_parts.append(re.escape(kw_lower))
                else:
                    all_kw_parts.append(rf"\b{re.escape(kw_lower)}\b")
            self._keyword_patterns[cat] = (
                re.compile("|".join(all_kw_parts)) if all_kw_parts else None
            )

            all_akw_parts = []
            for kw in info.get("anti_keywords", []):
                kw_lower = kw.lower()
                if _is_emoji(kw_lower):
                    all_akw_parts.append(re.escape(kw_lower))
                else:
                    all_akw_parts.append(rf"\b{re.escape(kw_lower)}\b")
            self._anti_keyword_patterns[cat] = (
                re.compile("|".join(all_akw_parts)) if all_akw_parts else None
            )

    def _get_messages_for_chat(
        self,
        chat_id: str,
        limit: int = 500,
    ) -> list[ChatMessage]:
        """Get messages for a specific chat from chat.db.

        Args:
            chat_id: The chat identifier.
            limit: Maximum messages to retrieve.

        Returns:
            List of ChatMessage objects.
        """
        if not self.chat_db_path.exists():
            logger.warning("Chat database not found: %s", self.chat_db_path)
            return []

        messages = []
        try:
            with contextlib.closing(
                sqlite3.connect(
                    f"file:{self.chat_db_path}?mode=ro",
                    uri=True,
                    timeout=5.0,
                )
            ) as conn:
                conn.row_factory = sqlite3.Row

                # Query messages for this chat
                query = """
                    SELECT
                        m.text,
                        m.is_from_me,
                        m.date as date_int,
                        c.chat_identifier
                    FROM message m
                    JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
                    JOIN chat c ON cmj.chat_id = c.ROWID
                    WHERE c.chat_identifier = ?
                        AND m.text IS NOT NULL
                        AND m.text != ''
                    ORDER BY m.date DESC
                    LIMIT ?
                """

                cursor = conn.execute(query, (chat_id, limit))

                for row in cursor:
                    # Convert macOS timestamp to datetime
                    date_int = row["date_int"]
                    if date_int:
                        # macOS uses nanoseconds since 2001-01-01
                        timestamp = date_int / 1_000_000_000 + 978307200
                        date = datetime.fromtimestamp(timestamp, tz=UTC)
                    else:
                        date = None

                    messages.append(
                        ChatMessage(
                            text=row["text"] or "",
                            is_from_me=bool(row["is_from_me"]),
                            date=date,
                            chat_id=row["chat_identifier"],
                        )
                    )

        except Exception as e:
            logger.error("Failed to get messages for chat %s: %s", chat_id, e)

        return messages

    def _get_messages_for_chats_batch(
        self,
        chat_ids: list[str],
        limit_per_chat: int = 500,
    ) -> dict[str, list[ChatMessage]]:
        """Get messages for multiple chats in ONE query.

        Args:
            chat_ids: List of chat identifiers.
            limit_per_chat: Maximum messages per chat.

        Returns:
            Dict of chat_id -> list of ChatMessage objects.
        """
        if not chat_ids or not self.chat_db_path.exists():
            return {cid: [] for cid in chat_ids}

        messages_by_chat: dict[str, list[ChatMessage]] = {cid: [] for cid in chat_ids}
        try:
            with contextlib.closing(
                sqlite3.connect(
                    f"file:{self.chat_db_path}?mode=ro",
                    uri=True,
                    timeout=5.0,
                )
            ) as conn:
                conn.row_factory = sqlite3.Row

                # Use ROW_NUMBER to limit per chat within a single query
                placeholders = ",".join("?" for _ in chat_ids)
                query = f"""
                    WITH ranked AS (
                        SELECT
                            m.text,
                            m.is_from_me,
                            m.date as date_int,
                            c.chat_identifier,
                            ROW_NUMBER() OVER (
                                PARTITION BY c.chat_identifier
                                ORDER BY m.date DESC
                            ) as rn
                        FROM message m
                        JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
                        JOIN chat c ON cmj.chat_id = c.ROWID
                        WHERE c.chat_identifier IN ({placeholders})
                            AND m.text IS NOT NULL
                            AND m.text != ''
                    )
                    SELECT text, is_from_me, date_int, chat_identifier
                    FROM ranked
                    WHERE rn <= ?
                """

                cursor = conn.execute(query, (*chat_ids, limit_per_chat))

                for row in cursor:
                    date_int = row["date_int"]
                    if date_int:
                        timestamp = date_int / 1_000_000_000 + 978307200
                        date = datetime.fromtimestamp(timestamp, tz=UTC)
                    else:
                        date = None

                    chat_id = row["chat_identifier"]
                    messages_by_chat[chat_id].append(
                        ChatMessage(
                            text=row["text"] or "",
                            is_from_me=bool(row["is_from_me"]),
                            date=date,
                            chat_id=chat_id,
                        )
                    )

        except Exception as e:
            logger.error("Failed to batch-get messages for %d chats: %s", len(chat_ids), e)

        return messages_by_chat

    def _get_non_group_chats(self) -> list[tuple[str, str]]:
        """Get all non-group chats (1:1 conversations).

        Returns:
            List of (chat_id, display_name) tuples.
        """
        if not self.chat_db_path.exists():
            return []

        chats = []
        try:
            with contextlib.closing(
                sqlite3.connect(
                    f"file:{self.chat_db_path}?mode=ro",
                    uri=True,
                    timeout=5.0,
                )
            ) as conn:
                conn.row_factory = sqlite3.Row

                # Get 1:1 chats: phone numbers/emails that aren't group chats
                # Group chats have identifiers starting with "chat" and multiple handles
                # Uses CTEs instead of correlated subqueries to avoid per-row scans
                query = """
                    WITH handle_counts AS (
                        SELECT chat_id, COUNT(*) as cnt
                        FROM chat_handle_join
                        GROUP BY chat_id
                    ),
                    msg_counts AS (
                        SELECT chat_id, COUNT(*) as cnt
                        FROM chat_message_join
                        GROUP BY chat_id
                    )
                    SELECT
                        c.chat_identifier,
                        c.display_name,
                        COALESCE(mc.cnt, 0) as msg_count
                    FROM chat c
                    JOIN handle_counts hc ON hc.chat_id = c.ROWID AND hc.cnt = 1
                    LEFT JOIN msg_counts mc ON mc.chat_id = c.ROWID
                    ORDER BY msg_count DESC
                """

                cursor = conn.execute(query)

                for row in cursor:
                    chat_id = row["chat_identifier"]
                    display_name = row["display_name"] or chat_id
                    chats.append((chat_id, display_name))

        except Exception as e:
            logger.error("Failed to get chats: %s", e)

        return chats

    def _extract_features(self, messages: list[ChatMessage]) -> MessageFeatures:
        """Extract features from message history.

        Args:
            messages: List of messages from a conversation.

        Returns:
            MessageFeatures with computed statistics.
        """
        if not messages:
            return MessageFeatures(
                avg_message_length=0,
                message_frequency_per_day=0,
                emoji_rate=0,
                avg_hour_of_day=12,
                hour_variance=0,
                formal_language_score=0.5,
            )

        # Calculate basic stats
        lengths = []
        hours = []
        emoji_count = 0
        total_chars = 0
        formal_indicators = 0
        informal_indicators = 0
        my_messages = 0
        keyword_matches: dict[str, int] = {cat: 0 for cat in RELATIONSHIP_CATEGORIES}
        anti_keyword_matches: dict[str, int] = {cat: 0 for cat in RELATIONSHIP_CATEGORIES}

        for msg in messages:
            text = msg.text or ""
            lengths.append(len(text))
            total_chars += len(text)

            if msg.is_from_me:
                my_messages += 1

            # Extract hour if date available
            if msg.date:
                hours.append(msg.date.hour)

            # Count emojis using robust detection
            emoji_chars = sum(1 for c in text if _is_emoji(c))
            emoji_count += emoji_chars

            # Formal vs informal indicators
            text_lower = text.lower()
            if any(w in text_lower for w in ["please", "thank you", "regards", "sincerely"]):
                formal_indicators += 1
            informal_words = ["lol", "lmao", "haha", "omg", "u ", " ur ", "gonna", "wanna"]
            if any(w in text_lower for w in informal_words):
                informal_indicators += 1

            # Keyword matching using pre-compiled patterns (includes emojis)
            for cat in RELATIONSHIP_CATEGORIES:
                pat = self._keyword_patterns[cat]
                if pat:
                    keyword_matches[cat] += len(pat.findall(text_lower))

                anti_pat = self._anti_keyword_patterns[cat]
                if anti_pat:
                    anti_keyword_matches[cat] += len(anti_pat.findall(text_lower))

        # Calculate date range
        dates = [m.date for m in messages if m.date]
        if len(dates) >= 2:
            date_range = (max(dates) - min(dates)).days
        else:
            date_range = 1

        avg_length = sum(lengths) / len(lengths) if lengths else 0
        freq_per_day = len(messages) / max(date_range, 1)
        emoji_rate = emoji_count / max(total_chars, 1)
        avg_hour = sum(hours) / len(hours) if hours else 12
        hour_var = float(np.var(hours)) if len(hours) > 1 else 0

        # Formal language score (0-1, higher = more formal)
        total_indicators = formal_indicators + informal_indicators
        formal_score = formal_indicators / max(total_indicators, 1) if total_indicators else 0.5

        my_pct = my_messages / len(messages) if messages else 0.5

        return MessageFeatures(
            avg_message_length=avg_length,
            message_frequency_per_day=freq_per_day,
            emoji_rate=emoji_rate,
            avg_hour_of_day=avg_hour,
            hour_variance=hour_var,
            formal_language_score=formal_score,
            keyword_matches=keyword_matches,
            anti_keyword_matches=anti_keyword_matches,
            total_messages=len(messages),
            date_range_days=date_range,
            my_messages_pct=my_pct,
        )

    def _compute_scores(self, features: MessageFeatures) -> dict[str, float]:
        """Compute scores for each relationship category.

        Args:
            features: Extracted message features.

        Returns:
            Dictionary of category -> score.
        """
        scores: dict[str, float] = {}

        for cat, info in RELATIONSHIP_CATEGORIES.items():
            score = 0.0

            # Keyword match score (normalized by message count)
            kw_divisor = max(features.total_messages / 50, 5)
            keyword_score = min(features.keyword_matches.get(cat, 0) / kw_divisor, 1.0)
            score += keyword_score * 0.4  # 40% weight

            # Anti-keyword penalty
            anti_score = features.anti_keyword_matches.get(cat, 0)
            if anti_score > 0:
                score -= min(anti_score / 5, 0.3)  # Max 30% penalty

            # Pattern matching
            patterns = info.get("patterns", {})

            # Time patterns
            if patterns.get("work_hours_only"):
                in_work_hours = 9 <= features.avg_hour_of_day <= 17
                low_variance = features.hour_variance < 10
                work_hour_score = 1.0 if (in_work_hours and low_variance) else 0.0
                score += work_hour_score * 0.2
            elif patterns.get("time_any_hour"):
                any_hour_score = min(features.hour_variance / 30, 1.0)
                score += any_hour_score * 0.15
            elif patterns.get("late_night_messages"):
                is_late_night = features.avg_hour_of_day >= 20 or features.avg_hour_of_day <= 2
                late_score = 1.0 if is_late_night else 0.0
                score += late_score * 0.15

            # Emoji rate
            if patterns.get("high_emoji_rate"):
                emoji_score = min(features.emoji_rate * 50, 1.0)
                score += emoji_score * 0.15
            elif patterns.get("low_emoji_rate"):
                emoji_score = 1.0 - min(features.emoji_rate * 50, 1.0)
                score += emoji_score * 0.15

            # Formality
            if patterns.get("formal_language"):
                score += features.formal_language_score * 0.15
            elif patterns.get("casual_language"):
                score += (1.0 - features.formal_language_score) * 0.15

            # Frequency
            if patterns.get("very_high_frequency"):
                freq_score = min(features.message_frequency_per_day / 5, 1.0)
                score += freq_score * 0.1
            elif patterns.get("medium_frequency"):
                # Score 1.0 when 0.5-3 msgs/day, tapering to 0 outside
                freq = features.message_frequency_per_day
                if 0.5 <= freq <= 3.0:
                    freq_score = 1.0
                elif freq < 0.5:
                    freq_score = max(freq / 0.5, 0.0)
                else:
                    freq_score = max(1.0 - (freq - 3.0) / 3.0, 0.0)
                score += freq_score * 0.1
            elif patterns.get("low_frequency"):
                freq_score = 1.0 if features.message_frequency_per_day < 0.5 else 0.0
                score += freq_score * 0.1

            scores[cat] = max(score, 0.0)

        return scores

    # Display name -> relationship mappings (checked case-insensitively)
    _DISPLAY_NAME_SIGNALS: dict[str, list[str]] = {
        "family": [
            "mom",
            "dad",
            "mother",
            "father",
            "mama",
            "papa",
            "sis",
            "bro",
            "brother",
            "sister",
            "grandma",
            "grandpa",
            "grandmother",
            "grandfather",
            "aunt",
            "uncle",
            "cousin",
            "wife",
            "husband",
            # Indian family terms
            "nana",
            "nani",
            "dadi",
            "dada",
            "masi",
            "mausi",
            "chacha",
            "bua",
            "taya",
        ],
        "romantic partner": [
            "babe",
            "baby",
            "bae",
            "love",
            "honey",
            "sweetheart",
            "hubby",
            "wifey",
        ],
    }

    def _score_display_name(self, display_name: str) -> dict[str, float]:
        """Score relationship categories based on contact display name.

        If the display name contains a known relationship keyword (e.g. "Mom",
        "Gopal Mama"), returns a bonus score for the matching category.

        Args:
            display_name: The contact's display name.

        Returns:
            Dict of category -> bonus score (0.0 if no match).
        """
        bonuses: dict[str, float] = {}
        if not display_name:
            return bonuses

        name_lower = display_name.lower()
        name_tokens = set(_NAME_SPLIT_RE.split(name_lower))

        for category, keywords in self._DISPLAY_NAME_SIGNALS.items():
            for kw in keywords:
                if kw in name_tokens:
                    bonuses[category] = 0.5
                    break

        return bonuses

    def classify_messages(
        self,
        messages: list[ChatMessage],
        chat_id: str = "",
        display_name: str = "",
    ) -> ClassificationResult:
        """Classify relationship from a list of messages.

        Args:
            messages: List of ChatMessage objects.
            chat_id: Optional chat identifier.
            display_name: Optional display name.

        Returns:
            ClassificationResult with relationship type and confidence.
        """
        if len(messages) < self.min_messages:
            return ClassificationResult(
                chat_id=chat_id,
                display_name=display_name,
                relationship="unknown",
                confidence=0.0,
                sample_size=len(messages),
            )

        features = self._extract_features(messages)
        scores = self._compute_scores(features)

        # Apply display name signal as a strong prior
        name_bonuses = self._score_display_name(display_name)
        for cat, bonus in name_bonuses.items():
            if cat in scores:
                scores[cat] += bonus

        # Find best match
        best_category = max(scores, key=lambda k: scores[k])
        best_score = scores[best_category]

        # Normalize confidence (0-1)
        confidence = min(best_score / 0.6, 1.0)

        return ClassificationResult(
            chat_id=chat_id,
            display_name=display_name,
            relationship=best_category,
            confidence=round(confidence, 2),
            signals=scores,
            sample_size=len(messages),
            features={
                "avg_length": features.avg_message_length,
                "freq_per_day": features.message_frequency_per_day,
                "emoji_rate": features.emoji_rate,
                "avg_hour": features.avg_hour_of_day,
                "formality": features.formal_language_score,
            },
        )

    def classify_contact(self, chat_id: str, display_name: str = "") -> ClassificationResult:
        """Classify a specific contact by chat_id.

        Uses TTL-based caching to avoid recomputing classification for the same contact.

        Args:
            chat_id: The chat identifier (phone/email).
            display_name: Optional display name.

        Returns:
            ClassificationResult with relationship type and confidence.
        """
        # Check cache first
        now = time.time()
        with _cache_lock:
            if chat_id in _classification_cache:
                cached_time, cached_result = _classification_cache[chat_id]
                if now - cached_time < _CLASSIFICATION_CACHE_TTL_SECONDS:
                    logger.debug("Classification cache hit for %s", chat_id)
                    return cached_result

        messages = self._get_messages_for_chat(chat_id)
        result = self.classify_messages(messages, chat_id, display_name)

        # Update cache (evict oldest if at capacity)
        with _cache_lock:
            if len(_classification_cache) >= _CLASSIFICATION_CACHE_MAX_SIZE:
                # Evict oldest entry
                oldest_key = min(_classification_cache, key=lambda k: _classification_cache[k][0])
                del _classification_cache[oldest_key]
            _classification_cache[chat_id] = (now, result)

        return result

    def classify_all_contacts(
        self,
        limit: int = 100,
        min_confidence: float = 0.3,
    ) -> list[ClassificationResult]:
        """Classify all 1:1 contacts (not groups).

        Args:
            limit: Maximum number of contacts to classify.
            min_confidence: Minimum confidence to include.

        Returns:
            List of ClassificationResult objects.
        """
        chats = self._get_non_group_chats()[:limit]
        return self.classify_contacts_batch(chats, min_confidence=min_confidence)

    def classify_contacts_batch(
        self,
        chat_ids: list[tuple[str, str]],
        min_confidence: float = 0.3,
    ) -> list[ClassificationResult]:
        """Classify multiple contacts with a single batched DB query.

        Fetches all messages for all chat_ids in ONE query, then classifies
        each contact from memory. Much faster than N individual DB queries.

        Args:
            chat_ids: List of (chat_id, display_name) tuples.
            min_confidence: Minimum confidence to include in results.

        Returns:
            List of ClassificationResult objects (filtered by confidence).
        """
        if not chat_ids:
            return []

        # Check cache for already-classified contacts, collect uncached ones
        now = time.time()
        results: list[ClassificationResult] = []
        uncached: list[tuple[str, str]] = []

        with _cache_lock:
            for chat_id, display_name in chat_ids:
                if chat_id in _classification_cache:
                    cached_time, cached_result = _classification_cache[chat_id]
                    if now - cached_time < _CLASSIFICATION_CACHE_TTL_SECONDS:
                        if cached_result.confidence >= min_confidence:
                            results.append(cached_result)
                        continue
                uncached.append((chat_id, display_name))

        # Batch-fetch messages for all uncached chats in ONE query
        if uncached:
            uncached_ids = [cid for cid, _ in uncached]
            messages_by_chat = self._get_messages_for_chats_batch(uncached_ids)

            for chat_id, display_name in uncached:
                messages = messages_by_chat.get(chat_id, [])
                result = self.classify_messages(messages, chat_id, display_name)

                # Update cache
                with _cache_lock:
                    if len(_classification_cache) >= _CLASSIFICATION_CACHE_MAX_SIZE:
                        oldest_key = min(
                            _classification_cache, key=lambda k: _classification_cache[k][0]
                        )
                        del _classification_cache[oldest_key]
                    _classification_cache[chat_id] = (now, result)

                if result.confidence >= min_confidence:
                    results.append(result)

        # Sort by confidence
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def evaluate(
        self,
        ground_truth: dict[str, str],
    ) -> dict[str, float]:
        """Evaluate classifier against ground truth labels.

        Args:
            ground_truth: Dict of chat_id -> correct relationship.

        Returns:
            Evaluation metrics (accuracy, per-category precision/recall).
        """
        correct = 0
        total = 0
        per_category: dict[str, dict[str, int]] = {
            cat: {"tp": 0, "fp": 0, "fn": 0} for cat in RELATIONSHIP_CATEGORIES
        }

        for chat_id, true_label in ground_truth.items():
            result = self.classify_contact(chat_id)
            predicted = result.relationship

            if predicted == true_label:
                correct += 1
                per_category[true_label]["tp"] += 1
            else:
                if predicted in per_category:
                    per_category[predicted]["fp"] += 1
                if true_label in per_category:
                    per_category[true_label]["fn"] += 1

            total += 1

        accuracy = correct / total if total > 0 else 0

        # Calculate per-category metrics
        metrics = {"accuracy": accuracy}
        for cat, counts in per_category.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics[f"{cat}_precision"] = precision
            metrics[f"{cat}_recall"] = recall

        return metrics


# =============================================================================
# Convenience Functions
# =============================================================================

# Module-level singleton instance to avoid repeated initialization
_classifier_instance: RelationshipClassifier | None = None
_classifier_lock = threading.Lock()


def _get_classifier() -> RelationshipClassifier:
    """Get or create the singleton RelationshipClassifier instance.

    Ensures only one classifier is instantiated, reducing memory overhead
    from repeated pattern compilation and initialization.

    Returns:
        Singleton RelationshipClassifier instance.
    """
    global _classifier_instance
    if _classifier_instance is None:
        with _classifier_lock:
            if _classifier_instance is None:
                _classifier_instance = RelationshipClassifier()
    return _classifier_instance


def clear_classification_cache() -> None:
    """Clear the classification result cache.

    Useful for testing or when message history changes significantly.
    """
    with _cache_lock:
        _classification_cache.clear()


def suggest_relationship(chat_id: str) -> str:
    """Get relationship suggestion for a contact.

    Args:
        chat_id: The chat identifier.

    Returns:
        Relationship label string.
    """
    classifier = _get_classifier()
    result = classifier.classify_contact(chat_id)
    if result.confidence < 0.3:
        return "unknown"
    return result.relationship


def classify_top_contacts(n: int = 20) -> None:
    """Classify and print top N contacts.

    Args:
        n: Number of contacts to classify.
    """
    classifier = _get_classifier()
    results = classifier.classify_all_contacts(limit=n)

    print(f"\n{'Contact':<25} {'Relationship':<18} {'Confidence':<12} {'Messages':<10}")
    print("-" * 70)

    for r in results:
        name = r.display_name[:24] if r.display_name else r.chat_id[:24]
        print(f"{name:<25} {r.relationship:<18} {r.confidence:<12.2f} {r.sample_size:<10}")


if __name__ == "__main__":
    classify_top_contacts(20)
