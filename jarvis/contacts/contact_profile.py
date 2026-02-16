"""Contact Profile - Unified per-contact style, relationship, and topic analysis.

Merges three previously disconnected profile systems into one:
- Style analysis (from old ContactProfiler.analyze_style)
- Relationship classification (from RelationshipClassifier)
- Topic discovery (from TopicDiscovery via HDBSCAN)

Built during preprocessing, cached for generation.
Storage: ~/.jarvis/profiles/{hashed_id}.json
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from jarvis.contacts.contact_utils import hash_contact_id

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from contracts.imessage import Message


# =============================================================================
# Fact Data Class (used by fact_extractor, fact_storage, knowledge_graph)
# =============================================================================


@dataclass
class Fact:
    """A structured fact extracted from a message."""

    category: str  # relationship, location, work, preference, event
    subject: str  # entity name (e.g., "Sarah", "Austin", "Google")
    predicate: str  # relationship type (e.g., "is_family_of", "lives_in", "works_at")
    value: str = ""  # optional detail (e.g., "sister")
    source_text: str = ""  # the message text this was extracted from
    confidence: float = 0.8  # 0-1 confidence score
    contact_id: str = ""  # which contact this fact belongs to
    extracted_at: str = ""  # ISO timestamp
    source_message_id: int | None = None  # optional message ROWID
    linked_contact_id: str | None = None  # resolved contact ID (from NER person linking)
    # Temporal validity (for facts that change over time)
    valid_from: str | None = None  # When this fact became true (ISO timestamp)
    valid_until: str | None = None  # When this fact stopped being true (ISO timestamp)
    attribution: str = "contact"  # "contact", "user", or "third_party"

    def to_searchable_text(self) -> str:
        """Convert fact to a searchable text string for embedding.

        Format: "{predicate}: {subject} ({value})"
        """
        text = f"{self.predicate}: {self.subject}"
        if self.value:
            text += f" ({self.value})"
        return text


logger = logging.getLogger(__name__)

# Storage
PROFILES_DIR = Path.home() / ".jarvis" / "profiles"

# Minimum messages for reliable analysis
MIN_MESSAGES_FOR_PROFILE = 5

# Common text abbreviations
TEXT_ABBREVIATIONS = frozenset(
    {
        "u",
        "ur",
        "r",
        "n",
        "y",
        "k",
        "ok",
        "kk",
        "pls",
        "plz",
        "thx",
        "ty",
        "np",
        "yw",
        "idk",
        "idc",
        "imo",
        "imho",
        "tbh",
        "ngl",
        "fr",
        "rn",
        "atm",
        "btw",
        "fyi",
        "lmk",
        "hmu",
        "wbu",
        "hbu",
        "omg",
        "omw",
        "otw",
        "brb",
        "brt",
        "ttyl",
        "gtg",
        "g2g",
        "lol",
        "lmao",
        "lmfao",
        "rofl",
        "jk",
        "jfc",
        "smh",
        "nvm",
        "bc",
        "cuz",
        "tho",
        "rly",
        "sry",
        "prob",
        "def",
        "obvi",
        "whatev",
        "watever",
        "w/e",
        "w/o",
        "b4",
        "2day",
        "2morrow",
        "2nite",
        "l8r",
        "l8",
        "gr8",
        "m8",
        "str8",
        "h8",
        "w8",
        "gonna",
        "wanna",
        "gotta",
        "kinda",
        "sorta",
        "tryna",
        "boutta",
        "finna",
        "shoulda",
        "coulda",
        "woulda",
        "yea",
        "yeh",
        "ya",
        "yup",
        "yep",
        "nah",
        "nope",
        "aight",
        "ight",
        "bet",
        "facts",
        "cap",
        "nocap",
        "lowkey",
        "highkey",
        "deadass",
        "sus",
        "slay",
        "fire",
        "lit",
        "goat",
    }
)

EMOJI_PATTERN = re.compile(
    r"[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0001F600-\U0001F64F"
    r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]"
)

# Pre-compiled patterns for per-message loops (avoid recompilation overhead)
_WORD_RE = re.compile(r"\b\w+\b")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z]")

# Formality indicators (from unified_relationship.py)
CASUAL_INDICATORS = frozenset(
    {
        "lol",
        "haha",
        "hehe",
        "lmao",
        "omg",
        "btw",
        "brb",
        "ttyl",
        "idk",
        "ikr",
        "nvm",
        "tbh",
        "imo",
        "np",
        "k",
        "kk",
        "yeah",
        "yep",
        "nope",
        "gonna",
        "wanna",
        "gotta",
        "cuz",
        "bc",
        "u",
        "ur",
        "r",
        "thx",
        "ty",
        "pls",
        "plz",
        "yo",
        "dude",
        "bro",
        "sis",
        "fam",
        "smh",
        "ngl",
        "rn",
        "af",
        "lowkey",
        "highkey",
        "ong",
        "fr",
        "periodt",
        "slay",
        "based",
        "cap",
        "nocap",
    }
)

FORMAL_INDICATORS = frozenset(
    {
        "regards",
        "sincerely",
        "please",
        "kindly",
        "thank you",
        "appreciate",
        "regarding",
        "attached",
        "discussed",
        "confirmed",
        "scheduled",
        "deadline",
        "meeting",
        "mr",
        "mrs",
        "ms",
        "dr",
        "hello",
        "dear",
        "cordially",
        "per",
        "pursuant",
        "hereby",
        "accordingly",
        "respectfully",
    }
)

GREETING_PATTERNS = frozenset(
    {
        "hi",
        "hey",
        "hello",
        "yo",
        "sup",
        "whats up",
        "hola",
        "good morning",
        "morning",
        "afternoon",
        "evening",
        "howdy",
        "hiya",
        "heya",
    }
)

SIGNOFF_PATTERNS = frozenset(
    {
        "bye",
        "goodbye",
        "later",
        "cya",
        "see ya",
        "see you",
        "ttyl",
        "talk later",
        "take care",
        "night",
        "goodnight",
        "gn",
        "peace",
        "cheers",
        "thanks",
        "thank you",
        "thx",
        "ty",
    }
)

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "my",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "and",
        "or",
        "but",
        "if",
        "so",
        "that",
        "this",
        "just",
        "not",
        "no",
        "our",
        "your",
        "their",
        "have",
        "has",
        "had",
    }
)


# =============================================================================
# Data Class
# =============================================================================


@dataclass
class ContactProfile:
    """Complete profile for a contact - relationship + style + topics."""

    contact_id: str
    contact_name: str | None = None

    # Relationship (from RelationshipClassifier)
    relationship: str = "unknown"
    relationship_confidence: float = 0.0
    relationship_reasoning: str | None = None

    # Style (computed from MY messages to this contact)
    formality: str = "casual"  # formal, casual, very_casual
    formality_score: float = 0.5  # 0.0 (casual) to 1.0 (formal)
    avg_message_length: float = 50.0
    typical_length: str = "medium"  # short/medium/long
    uses_lowercase: bool = False
    uses_abbreviations: bool = False
    common_abbreviations: list[str] = field(default_factory=list)
    emoji_frequency: float = 0.0
    exclamation_frequency: float = 0.0

    # Conversation patterns (from MY messages)
    greeting_style: list[str] = field(default_factory=list)
    signoff_style: list[str] = field(default_factory=list)
    common_phrases: list[str] = field(default_factory=list)

    # Topics (from HDBSCAN on both parties' messages)
    top_topics: list[str] = field(default_factory=list)

    # Facts (from instruction_extractor)
    extracted_facts: list[dict[str, Any]] = field(default_factory=list)

    # Metadata
    message_count: int = 0
    my_message_count: int = 0
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "contact_id": self.contact_id,
            "contact_name": self.contact_name,
            "relationship": self.relationship,
            "relationship_confidence": self.relationship_confidence,
            "relationship_reasoning": self.relationship_reasoning,
            "formality": self.formality,
            "formality_score": self.formality_score,
            "avg_message_length": self.avg_message_length,
            "typical_length": self.typical_length,
            "uses_lowercase": self.uses_lowercase,
            "uses_abbreviations": self.uses_abbreviations,
            "common_abbreviations": self.common_abbreviations,
            "emoji_frequency": self.emoji_frequency,
            "exclamation_frequency": self.exclamation_frequency,
            "greeting_style": self.greeting_style,
            "signoff_style": self.signoff_style,
            "common_phrases": self.common_phrases,
            "top_topics": self.top_topics,
            "extracted_facts": self.extracted_facts,
            "message_count": self.message_count,
            "my_message_count": self.my_message_count,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContactProfile:
        """Deserialize from storage."""
        return cls(
            contact_id=data["contact_id"],
            contact_name=data.get("contact_name"),
            relationship=data.get("relationship", "unknown"),
            relationship_confidence=data.get("relationship_confidence", 0.0),
            relationship_reasoning=data.get("relationship_reasoning"),
            formality=data.get("formality", "casual"),
            formality_score=data.get("formality_score", 0.5),
            avg_message_length=data.get("avg_message_length", 50.0),
            typical_length=data.get("typical_length", "medium"),
            uses_lowercase=data.get("uses_lowercase", False),
            uses_abbreviations=data.get("uses_abbreviations", False),
            common_abbreviations=data.get("common_abbreviations", []),
            emoji_frequency=data.get("emoji_frequency", 0.0),
            exclamation_frequency=data.get("exclamation_frequency", 0.0),
            greeting_style=data.get("greeting_style", []),
            signoff_style=data.get("signoff_style", []),
            common_phrases=data.get("common_phrases", []),
            top_topics=data.get("top_topics", []),
            extracted_facts=data.get("extracted_facts", []),
            message_count=data.get("message_count", 0),
            my_message_count=data.get("my_message_count", 0),
            updated_at=data.get("updated_at", ""),
        )


# =============================================================================
# Profile Builder
# =============================================================================


class ContactProfileBuilder:
    """Builds ContactProfile from messages.

    Combines:
    - RelationshipClassifier for relationship type
    - Laplace-smoothed formality scoring
    - Style analysis (abbreviations, emoji, length, lowercase)
    - Greeting/signoff/phrase extraction
    - HDBSCAN topic discovery (optional, requires embeddings)
    """

    def __init__(self, min_messages: int = MIN_MESSAGES_FOR_PROFILE) -> None:
        self.min_messages = min_messages

    @staticmethod
    def _normalize_relationship_label(label: str) -> str:
        """Normalize relationship labels from classifiers/LLM output."""
        normalized = (label or "").strip().lower()
        # Remove common markdown emphasis/noise from model output.
        normalized = normalized.strip("*_` ")
        return normalized or "unknown"

    def build_profile(
        self,
        contact_id: str,
        messages: list[Message],
        contact_name: str | None = None,
        embeddings: NDArray[np.float32] | None = None,
    ) -> ContactProfile:
        """Build a complete profile for a contact."""
        now = datetime.now().isoformat()

        # IMPROVED: Resolve name if missing
        if not contact_name or contact_name in ["None", "Unknown", "Contact"]:
            from integrations.imessage.reader import ChatDBReader

            try:
                # Use cached reader logic
                with ChatDBReader() as reader:
                    conv = reader.get_conversation(contact_id)
                    if conv and conv.display_name:
                        contact_name = conv.display_name
            except Exception:  # nosec B110
                pass

        if len(messages) < self.min_messages:
            return ContactProfile(
                contact_id=contact_id,
                contact_name=contact_name,
                message_count=len(messages),
                my_message_count=sum(1 for m in messages if m.is_from_me),
                updated_at=now,
            )

        # Split messages
        my_messages = [m for m in messages if m.is_from_me]
        analyze_msgs = my_messages if my_messages else messages

        # Relationship classification
        relationship, rel_confidence = self._classify_relationship(
            contact_id, messages, contact_name
        )
        relationship = self._normalize_relationship_label(relationship)

        # LLM Refinement
        relationship_reasoning = None
        # Keep high-confidence classifier output stable; only refine uncertain cases.
        if len(messages) >= 3 and rel_confidence < 0.95:
            logger.info(
                "Refining relationship for %s (current: %s, confidence: %.2f)",
                contact_name or contact_id, relationship, rel_confidence
            )
            relationship, rel_confidence, relationship_reasoning = (
                self._refine_relationship_with_llm(
                    messages, contact_name or "Contact", relationship, rel_confidence
                )
            )
            relationship = self._normalize_relationship_label(relationship)
            logger.info(
                "LLM refined relationship for %s to: %s (confidence: %.2f)",
                contact_name or contact_id, relationship, rel_confidence
            )

        # Formality (Laplace-smoothed)
        formality_score = self._compute_formality(analyze_msgs)
        if formality_score < 0.35:
            formality: Literal["formal", "casual", "very_casual"] = "very_casual"
        elif formality_score < 0.55:
            formality = "casual"
        else:
            formality = "formal"

        # Style stats
        my_texts = [m.text for m in analyze_msgs if m.text]
        avg_length, typical_length = self._compute_length(my_texts)
        uses_lowercase = self._detect_lowercase(my_texts)
        uses_abbreviations, common_abbreviations = self._detect_abbreviations(my_texts)
        emoji_freq = self._compute_emoji_frequency(my_texts)
        excl_freq = self._compute_exclamation_frequency(my_texts)

        # Conversation patterns
        greeting_style = self._extract_greetings(analyze_msgs)
        signoff_style = self._extract_signoffs(analyze_msgs)
        common_phrases = self._extract_common_phrases(analyze_msgs)

        # Topics (optional)
        top_topics = self._discover_topics(contact_id, messages, embeddings)

        # Extracted Facts (from DB)
        extracted_facts = self._fetch_db_facts(contact_id)

        return ContactProfile(
            contact_id=contact_id,
            contact_name=contact_name,
            relationship=relationship,
            relationship_confidence=rel_confidence,
            relationship_reasoning=relationship_reasoning,
            formality=formality,
            formality_score=round(formality_score, 3),
            avg_message_length=round(avg_length, 1),
            typical_length=typical_length,
            uses_lowercase=uses_lowercase,
            uses_abbreviations=uses_abbreviations,
            common_abbreviations=common_abbreviations,
            emoji_frequency=round(emoji_freq, 2),
            exclamation_frequency=round(excl_freq, 2),
            greeting_style=greeting_style,
            signoff_style=signoff_style,
            common_phrases=common_phrases,
            top_topics=top_topics,
            extracted_facts=extracted_facts,
            message_count=len(messages),
            my_message_count=len(my_messages),
            updated_at=now,
        )

    # --- Facts ---

    def _fetch_db_facts(self, contact_id: str) -> list[dict[str, Any]]:
        """Fetch facts for this contact from the database."""
        try:
            from jarvis.db import get_db

            db = get_db()
            with db.connection() as conn:
                rows = conn.execute(
                    """SELECT category, subject, predicate, value, confidence,
                        extracted_at FROM contact_facts WHERE contact_id = ?
                        ORDER BY extracted_at DESC""",
                    (contact_id,),
                ).fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.warning("Failed to fetch facts for profile %s: %s", contact_id, e)
            return []

    # --- Relationship ---

    def _refine_relationship_with_llm(
        self, messages: list[Message], contact_name: str, base_rel: str, base_conf: float
    ) -> tuple[str, float, str]:
        """Use LLM to refine the relationship type and provide reasoning."""
        try:
            from models.loader import get_model

            loader = get_model()

            if not loader.is_loaded():
                return base_rel, base_conf, "Rule-based analysis."

            # Format a small sample of the chat
            sample = messages[-15:]  # Last 15 messages
            turns = []
            curr_sender = "User" if sample[0].is_from_me else contact_name
            curr_msgs = []
            for m in sample:
                sender = "User" if m.is_from_me else contact_name
                if sender == curr_sender:
                    curr_msgs.append(m.text or "")
                else:
                    if curr_msgs:
                        turns.append(f"{curr_sender}: {' '.join(curr_msgs)}")
                    curr_sender = sender
                    curr_msgs = [m.text or ""]
            if curr_msgs:
                turns.append(f"{curr_sender}: {' '.join(curr_msgs)}")
            chat_text = "\n".join(turns)

            prompt = f"""Analyze this chat and determine the relationship between
User and {contact_name}.
Categories: family, close friend, coworker, acquaintance, romantic partner.

Chat:
{chat_text}

Rules:
1. Pick the best category.
2. Provide 1 sentence of reasoning.
3. Output format: Category | Confidence (0-1) | Reasoning

Result:"""

            res = loader.generate_sync(prompt=prompt, max_tokens=60, temperature=0.0)
            text = res.text.strip()

            if "|" in text:
                parts = text.split("|")
                if len(parts) >= 3:
                    rel = parts[0].strip().lower()
                    # Extract float confidence
                    conf_str = re.search(r"[\d\.]+", parts[1])
                    conf = float(conf_str.group()) if conf_str else base_conf
                    reason = parts[2].strip()
                    return rel, conf, reason

        except Exception as e:
            logger.debug("LLM relationship refinement failed: %s", e)

        return base_rel, base_conf, "Rule-based analysis."

    def _classify_relationship(
        self,
        contact_id: str,
        messages: list[Message],
        contact_name: str | None = None,
    ) -> tuple[str, float]:
        """Classify relationship using RelationshipClassifier."""
        try:
            from jarvis.classifiers.relationship_classifier import (
                ChatMessage,
                RelationshipClassifier,
            )

            chat_messages = [
                ChatMessage(
                    text=m.text or "",
                    is_from_me=m.is_from_me,
                    date=m.date,
                    chat_id=contact_id,
                )
                for m in messages
            ]

            classifier = RelationshipClassifier(min_messages=self.min_messages)
            result = classifier.classify_messages(chat_messages, contact_id, contact_name or "")
            return result.relationship, result.confidence
        except Exception as e:
            logger.warning("Relationship classification failed: %s", e)
            return "unknown", 0.0

    # --- Formality (Laplace-smoothed) ---

    def _compute_formality(self, messages: list[Message]) -> float:
        """Compute formality score from 0.0 (casual) to 1.0 (formal).

        Uses Laplace smoothing: (formal + 1) / (total + 2).
        """
        if not messages:
            return 0.5

        casual_score = 0.0
        formal_score = 0.0

        for msg in messages:
            if not msg.text:
                continue
            text = msg.text.strip()
            if not text:
                continue

            text_lower = text.lower()
            words = set(_WORD_RE.findall(text_lower))

            casual_score += len(words & CASUAL_INDICATORS)
            formal_score += len(words & FORMAL_INDICATORS)

            # Emoji = casual
            emojis = EMOJI_PATTERN.findall(text)
            if emojis:
                casual_score += min(len(emojis) * 0.5, 2.0)

            # Multiple exclamations = casual
            if text.count("!") > 1:
                casual_score += 1.0

            # Short messages = casual, long = formal
            if len(text) <= 10:
                casual_score += 0.5
            elif len(text) >= 80:
                formal_score += 0.5

            # Capitalization and punctuation
            if text and text[0].isupper():
                formal_score += 0.3
            if text.endswith((".", "?", "!")):
                formal_score += 0.2

        total = formal_score + casual_score
        return (formal_score + 1.0) / (total + 2.0)

    # --- Style stats ---

    @staticmethod
    def _compute_length(
        texts: list[str],
    ) -> tuple[float, Literal["short", "medium", "long"]]:
        if not texts:
            return 50.0, "medium"
        lengths = [len(t) for t in texts]
        avg = sum(lengths) / len(lengths)
        if avg <= 30:
            category: Literal["short", "medium", "long"] = "short"
        elif avg <= 100:
            category = "medium"
        else:
            category = "long"
        return avg, category

    @staticmethod
    def _detect_lowercase(texts: list[str]) -> bool:
        if not texts:
            return False
        lowercase_count = 0
        for text in texts:
            letters = _NON_ALPHA_RE.sub("", text)
            if letters:
                ratio = sum(1 for c in letters if c.islower()) / len(letters)
                if ratio > 0.8:
                    lowercase_count += 1
        return lowercase_count / len(texts) > 0.6

    @staticmethod
    def _detect_abbreviations(texts: list[str]) -> tuple[bool, list[str]]:
        if not texts:
            return False, []
        word_counts: Counter[str] = Counter()
        for text in texts:
            word_counts.update(_WORD_RE.findall(text.lower()))
        found = {w: word_counts[w] for w in word_counts if w in TEXT_ABBREVIATIONS}
        sorted_abbrevs = sorted(found, key=found.get, reverse=True)  # type: ignore[arg-type]
        return len(found) >= 2, sorted_abbrevs[:5]

    @staticmethod
    def _compute_emoji_frequency(texts: list[str]) -> float:
        if not texts:
            return 0.0
        count = sum(len(EMOJI_PATTERN.findall(t)) for t in texts)
        return count / len(texts)

    @staticmethod
    def _compute_exclamation_frequency(texts: list[str]) -> float:
        if not texts:
            return 0.0
        count = sum(t.count("!") for t in texts)
        return count / len(texts)

    # --- Conversation patterns ---

    @staticmethod
    def _extract_greetings(messages: list[Message]) -> list[str]:
        greetings_found: Counter[str] = Counter()
        for msg in messages:
            if not msg.text:
                continue
            first_words = " ".join(msg.text.lower().strip().split()[:3])
            for pattern in GREETING_PATTERNS:
                if first_words.startswith(pattern):
                    greetings_found[pattern] += 1
                    break
        return [g for g, _ in greetings_found.most_common(3)]

    @staticmethod
    def _extract_signoffs(messages: list[Message]) -> list[str]:
        signoffs_found: Counter[str] = Counter()
        for msg in messages:
            if not msg.text:
                continue
            last_words = " ".join(msg.text.lower().strip().split()[-3:])
            for pattern in SIGNOFF_PATTERNS:
                if pattern in last_words:
                    signoffs_found[pattern] += 1
                    break
        return [s for s, _ in signoffs_found.most_common(3)]

    @staticmethod
    def _extract_common_phrases(messages: list[Message], min_count: int = 3) -> list[str]:
        phrase_counter: Counter[str] = Counter()
        for msg in messages:
            if not msg.text or len(msg.text) < 5:
                continue
            words = _WORD_RE.findall(msg.text.lower())
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                if w1 not in STOPWORDS and w2 not in STOPWORDS:
                    phrase = f"{w1} {w2}"
                    if len(phrase) >= 5:
                        phrase_counter[phrase] += 1
        return [p for p, c in phrase_counter.most_common(5) if c >= min_count]

    # --- Topics (optional) ---

    @staticmethod
    def _discover_topics(
        contact_id: str,
        messages: list[Message],
        embeddings: NDArray[np.float32] | None,
    ) -> list[str]:
        """Run HDBSCAN topic discovery if embeddings provided and enough messages."""
        if embeddings is None or len(embeddings) < 30:
            return []

        try:
            from jarvis.db import get_db
            from jarvis.search.vec_search import get_vec_searcher
            from jarvis.topics.topic_discovery import TopicDiscovery

            db = get_db()
            searcher = get_vec_searcher()

            # 1. Fetch pre-computed segment data for this contact
            with db.connection() as conn:
                rows = conn.execute(
                    """
                    SELECT topic_label, keywords_json, entities_json, vec_chunk_rowid
                    FROM conversation_segments
                    WHERE chat_id = ? AND vec_chunk_rowid IS NOT NULL
                    ORDER BY start_time DESC
                    LIMIT 200
                    """,
                    (contact_id,),
                ).fetchall()

            if not rows or len(rows) < 3:
                # Fall back to raw message labels if no segments found
                labels = [m.text for m in messages if m.text and len(m.text) > 20][:10]
                return [l[:30] for l in labels] if labels else []

            # 2. Extract labels and fetch centroids from vector searcher
            chunk_ids = [row["vec_chunk_rowid"] for row in rows]
            id_to_centroid = searcher.get_embeddings_by_ids(chunk_ids)

            if not id_to_centroid:
                # Fall back to stored topic labels
                unique_labels = list({r["topic_label"] for r in rows if r["topic_label"]})
                return unique_labels[:3]

            # 3. Cluster segment centroids to find "Global Topics"
            centroids = np.array(
                [id_to_centroid[cid] for cid in chunk_ids if cid in id_to_centroid],
                dtype=np.float32,
            )
            # For keywords, use the stored topic labels and keywords
            texts = []
            for r in rows:
                kws = json.loads(r["keywords_json"]) if r["keywords_json"] else []
                label = r["topic_label"] or ""
                texts.append(f"{label} {' '.join(kws)}")

            if len(centroids) < 5:
                # Too few for HDBSCAN, just return unique labels
                unique_labels = list({r["topic_label"] for r in rows if r["topic_label"]})
                return unique_labels[:3]

            discovery = TopicDiscovery()
            result = discovery.discover_topics(
                contact_id=contact_id,
                embeddings=centroids,
                texts=texts,
                min_cluster_size=max(2, len(centroids) // 10),
                min_samples=1,
            )

            # 4. Extract keyword labels from top clusters
            sorted_topics = sorted(result.topics, key=lambda t: t.message_count, reverse=True)
            return [", ".join(t.keywords[:3]) for t in sorted_topics[:3]]
        except Exception as e:
            logger.warning("Segment-based topic discovery failed for %s: %s", contact_id, e)
            return []


# =============================================================================
# Style Guide Generation
# =============================================================================


def format_style_guide(profile: ContactProfile) -> str:
    """Convert profile to natural language style guide for LLM prompts.

    Returns a concise, actionable description the LLM can follow.
    """
    if profile.message_count < MIN_MESSAGES_FOR_PROFILE:
        return (
            f"Limited message history ({profile.message_count} messages). "
            "Using default casual tone."
        )

    parts: list[str] = []

    # Relationship + formality
    tone_label = {
        "very_casual": "very casual",
        "casual": "casual, friendly",
        "formal": "professional, polished",
    }.get(profile.formality, "casual, friendly")

    if profile.relationship != "unknown":
        parts.append(f"Use a {tone_label} tone. This is a {profile.relationship.replace('_', ' ')}")
    else:
        parts.append(f"Use a {tone_label} tone")

    # Message length
    parts.append(f"Your typical message length: {profile.avg_message_length:.0f} chars")

    # Abbreviations
    if profile.uses_abbreviations and profile.common_abbreviations:
        abbrevs = ", ".join(profile.common_abbreviations[:4])
        parts.append(f"Use abbreviations like: {abbrevs}")

    # Emoji guidance
    if profile.emoji_frequency > 1.0:
        parts.append("Use emojis liberally")
    elif profile.emoji_frequency > 0.3:
        parts.append("Use emojis occasionally")
    elif profile.emoji_frequency < 0.05:
        parts.append("Avoid emojis")

    # Greetings
    if profile.greeting_style:
        greetings = ", ".join(f'"{g}"' for g in profile.greeting_style[:2])
        parts.append(f"Common greetings: {greetings}")

    # Topics
    if profile.top_topics:
        topics = ", ".join(profile.top_topics[:3])
        parts.append(f"Common topics: {topics}")

    return "\n- ".join([""] + parts).strip("- \n") if len(parts) > 1 else parts[0]


# =============================================================================
# Storage
# =============================================================================


def _ensure_profiles_dir() -> Path:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    return PROFILES_DIR


def _get_profile_path(contact_id: str) -> Path:
    hashed = hash_contact_id(contact_id)
    return _ensure_profiles_dir() / f"{hashed}.json"


def _json_serial(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def save_profile(profile: ContactProfile) -> bool:
    """Save profile to disk."""
    path = _get_profile_path(profile.contact_id)
    try:
        with open(path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2, default=_json_serial)
        logger.debug("Saved profile for %s", profile.contact_id[:16])
        return True
    except Exception as e:
        logger.error("Failed to save profile: %s", e)
        return False


def load_profile(contact_id: str) -> ContactProfile | None:
    """Load profile from disk."""
    path = _get_profile_path(contact_id)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return ContactProfile.from_dict(data)
    except Exception as e:
        logger.error("Failed to load profile: %s", e)
        return None


# LRU cache for hot-path access during generation
@lru_cache(maxsize=128)
def _cached_load(contact_id: str) -> ContactProfile | None:
    return load_profile(contact_id)


def get_contact_profile(contact_id: str) -> ContactProfile | None:
    """Get a contact profile with LRU caching.

    Returns cached profile if available, otherwise loads from disk.
    """
    return _cached_load(contact_id)


def invalidate_profile_cache() -> None:
    """Clear the entire profile LRU cache.

    Note: lru_cache does not support per-key invalidation,
    so this always clears the entire cache.
    """
    _cached_load.cache_clear()


def update_preference_tables(profile: ContactProfile, messages: list[Message]) -> None:
    """Update contact_style_targets and contact_timing_prefs tables.

    Computes derived style and timing preferences from the profile and message
    history, then persists them to the SQL preference tables.
    """
    from jarvis.analytics.engine import get_analytics_engine
    from jarvis.db import get_db

    db = get_db()
    now = datetime.now()

    # 1. Update contact_style_targets
    # Compute greeting rate as (unique greetings found) / (total messages)
    greeting_rate = (
        len(profile.greeting_style) / profile.message_count
        if profile.message_count > 0 and profile.greeting_style
        else 0.0
    )

    with db.connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO contact_style_targets
            (contact_id, median_reply_length, emoji_rate, greeting_rate, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                profile.contact_id,
                profile.avg_message_length,  # Use avg as proxy for median
                profile.emoji_frequency,
                greeting_rate,
                now,
            ),
        )

    # 2. Update contact_timing_prefs
    engine = get_analytics_engine()
    contact_analytics = engine.compute_contact_analytics(
        messages, profile.contact_id, profile.contact_name
    )
    hourly, daily, _, _ = engine.compute_time_distributions(messages)

    # Infer preferred hours (top 5 by volume)
    pref_hours = sorted(hourly.keys(), key=lambda h: hourly[h], reverse=True)[:5]
    # Infer quiet hours (bottom 6 by volume - usually late night/early morning)
    q_hours = sorted(hourly.keys(), key=lambda h: hourly[h])[:6]
    # Infer optimal weekdays (above average volume)
    avg_daily = sum(daily.values()) / len(daily) if daily else 0
    opt_days = [d for d, count in daily.items() if count > avg_daily]

    last_interaction = messages[-1].date if messages else None

    with db.connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO contact_timing_prefs
            (contact_id, timezone, quiet_hours_json, preferred_hours_json,
             optimal_weekdays_json, avg_response_time_mins, last_interaction, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile.contact_id,
                None,  # Timezone inference requires more logic
                json.dumps(q_hours),
                json.dumps(pref_hours),
                json.dumps(opt_days),
                contact_analytics.avg_response_time_minutes,
                last_interaction,
                now,
            ),
        )
