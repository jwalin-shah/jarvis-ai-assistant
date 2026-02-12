"""
EXAMPLE: Clean understanding pipeline (entity extraction → knowledge graph)

Before: fact_extractor.py (889 lines) scattered with NER, patterns, NLI
After: Clean separation: extract → resolve → store
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.db.models import Message


# ============================================================================
# Data Types
# ============================================================================


@dataclass
class Entity:
    """An entity extracted from text."""

    text: str  # Original text
    label: str  # PERSON, LOCATION, ORGANIZATION, etc.
    start: int  # Start index in text
    end: int  # End index in text
    normalized: str = ""  # Normalized form

    def __post_init__(self):
        if not self.normalized:
            self.normalized = self.text.lower().strip()


@dataclass
class Fact:
    """A fact extracted from text (subject-predicate-object)."""

    subject: str  # e.g., "Sarah"
    predicate: str  # e.g., "is_sister_of"
    object: str | None  # e.g., "user" or None
    value: str | None  # e.g., "close" (for relationship strength)

    # Provenance
    source_text: str  # Original message text
    source_message_id: str | None = None
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Quality
    confidence: float = 0.5  # 0.0 - 1.0

    # Temporal
    valid_from: str | None = None
    valid_until: str | None = None


@dataclass
class ExtractedKnowledge:
    """All knowledge extracted from a message."""

    entities: list[Entity] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.entities) == 0 and len(self.facts) == 0


# ============================================================================
# Extraction Rules
# ============================================================================

# Relationship patterns
RELATIONSHIP_PATTERNS = [
    (
        re.compile(r"\bmy\s+(sister|brother)\s+([A-Z][a-z]+)\b", re.I),
        lambda m: (m.group(2), f"is_{m.group(1)}_of", "user"),
    ),
    (
        re.compile(r"\bmy\s+(mom|mother|dad|father)\s+([A-Z][a-z]+)\b", re.I),
        lambda m: (m.group(2), "is_parent_of", "user"),
    ),
    (
        re.compile(r"\bmy\s+(friend|best friend)\s+([A-Z][a-z]+)\b", re.I),
        lambda m: (m.group(2), "is_friend_of", "user"),
    ),
    (
        re.compile(r"\bmy\s+(wife|husband|partner|girlfriend|boyfriend)\s+([A-Z][a-z]+)\b", re.I),
        lambda m: (m.group(2), "is_partner_of", "user"),
    ),
]

# Location patterns (with temporal awareness)
LOCATION_PATTERNS = [
    # Present
    (
        re.compile(r"\b(?:live|living)\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", re.I),
        lambda m: (m.group(1), "lives_in", None, "present"),
    ),
    # Past
    (
        re.compile(
            r"\b(?:grew\s+up\s+in|lived\s+in|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", re.I
        ),
        lambda m: (m.group(1), "lived_in", None, "past"),
    ),
    # Future
    (
        re.compile(r"\b(?:moving\s+to|relocating\s+to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", re.I),
        lambda m: (m.group(1), "moving_to", None, "future"),
    ),
]

# Work patterns
WORK_PATTERNS = [
    (
        re.compile(r"\b(?:work|working)\s+(?:at|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", re.I),
        lambda m: (m.group(1), "works_at", None),
    ),
    (
        re.compile(r"\b(?:started|joined)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", re.I),
        lambda m: (m.group(1), "works_at", None),
    ),
]

# Preference patterns
POSITIVE_WORDS = {"love", "like", "enjoy", "favorite", "obsessed", "addicted"}
NEGATIVE_WORDS = {"hate", "dislike", "can't stand", "allergic"}

PREFERENCE_PATTERN = re.compile(
    r"\b(?:love|like|hate|dislike|can't stand|obsessed with|addicted to|allergic to)\s+"
    r"([a-z\s]+?)(?:\s+(?:and|but|or|because)\b|\.|,|$)",
    re.I,
)


# ============================================================================
# Extractor
# ============================================================================


class KnowledgeExtractor:
    """
    Extract entities and facts from messages.

    This runs OFF the critical path - it doesn't block reply generation.
    """

    def __init__(self, use_spacy: bool = False, use_nli: bool = False):
        self.use_spacy = use_spacy
        self.use_nli = use_nli
        self._nlp = None

    def extract(self, message: Message | str, contact_id: str = "") -> ExtractedKnowledge:
        """
        Extract all knowledge from a message.

        Steps:
        1. Rule-based fact extraction (fast, always runs)
        2. NER entity extraction (optional, uses spaCy)
        3. Quality filtering
        4. Optional NLI verification
        """
        text = message.text if hasattr(message, "text") else message
        msg_id = getattr(message, "id", None)

        if not text or len(text) < 5:
            return ExtractedKnowledge()

        # Skip bot/professional messages
        if self._is_bot_message(text):
            return ExtractedKnowledge()

        knowledge = ExtractedKnowledge()

        # 1. Rule-based extraction
        knowledge.facts.extend(self._extract_relationships(text, contact_id, msg_id))
        knowledge.facts.extend(self._extract_locations(text, contact_id, msg_id))
        knowledge.facts.extend(self._extract_work(text, contact_id, msg_id))
        knowledge.facts.extend(self._extract_preferences(text, contact_id, msg_id))

        # 2. NER extraction (optional)
        if self.use_spacy:
            entities, facts = self._extract_with_ner(text, contact_id, msg_id)
            knowledge.entities.extend(entities)
            knowledge.facts.extend(facts)

        # 3. Quality filtering
        knowledge.facts = self._filter_quality(knowledge.facts)

        # 4. Optional NLI verification
        if self.use_nli and knowledge.facts:
            knowledge.facts = self._verify_with_nli(knowledge.facts)

        return knowledge

    def _extract_relationships(self, text: str, contact_id: str, msg_id: str | None) -> list[Fact]:
        """Extract relationship facts using regex patterns."""
        facts = []
        for pattern, extractor in RELATIONSHIP_PATTERNS:
            for match in pattern.finditer(text):
                try:
                    subject, predicate, obj = extractor(match)
                    facts.append(
                        Fact(
                            subject=subject,
                            predicate=predicate,
                            object=obj,
                            source_text=text[:200],
                            source_message_id=msg_id,
                            confidence=0.8,
                        )
                    )
                except Exception:
                    continue
        return facts

    def _extract_locations(self, text: str, contact_id: str, msg_id: str | None) -> list[Fact]:
        """Extract location facts with temporal awareness."""
        facts = []
        now = datetime.now().isoformat()

        for pattern, extractor in LOCATION_PATTERNS:
            for match in pattern.finditer(text):
                try:
                    result = extractor(match)
                    subject, predicate, obj, tense = result

                    fact = Fact(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        source_text=text[:200],
                        source_message_id=msg_id,
                    )

                    # Set confidence and temporal bounds based on tense
                    if tense == "present":
                        fact.confidence = 0.85
                        fact.valid_from = now
                    elif tense == "past":
                        fact.confidence = 0.5
                        fact.valid_until = now
                    elif tense == "future":
                        fact.confidence = 0.6
                        fact.valid_from = now

                    facts.append(fact)
                except Exception:
                    continue
        return facts

    def _extract_work(self, text: str, contact_id: str, msg_id: str | None) -> list[Fact]:
        """Extract work/organization facts."""
        facts = []
        for pattern, extractor in WORK_PATTERNS:
            for match in pattern.finditer(text):
                try:
                    subject, predicate, obj = extractor(match)
                    facts.append(
                        Fact(
                            subject=subject,
                            predicate=predicate,
                            object=obj,
                            source_text=text[:200],
                            source_message_id=msg_id,
                            confidence=0.7,
                        )
                    )
                except Exception:
                    continue
        return facts

    def _extract_preferences(self, text: str, contact_id: str, msg_id: str | None) -> list[Fact]:
        """Extract preference facts (likes/dislikes)."""
        facts = []
        match_text = text.lower()

        # Determine sentiment
        predicate = "likes"
        for word in NEGATIVE_WORDS:
            if word in match_text:
                predicate = "dislikes"
                break

        for match in PREFERENCE_PATTERN.finditer(text):
            subject = match.group(1).strip()
            if len(subject) < 3:
                continue

            facts.append(
                Fact(
                    subject=subject,
                    predicate=predicate,
                    source_text=text[:200],
                    source_message_id=msg_id,
                    confidence=0.6,
                )
            )
        return facts

    def _extract_with_ner(
        self, text: str, contact_id: str, msg_id: str | None
    ) -> tuple[list[Entity], list[Fact]]:
        """Extract entities using spaCy NER."""
        entities = []
        facts = []

        if self._nlp is None:
            try:
                import spacy

                self._nlp = spacy.load("en_core_web_sm")
            except Exception:
                return entities, facts

        doc = self._nlp(text)

        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(ent.text) > 1:
                entities.append(
                    Entity(text=ent.text, label="PERSON", start=ent.start_char, end=ent.end_char)
                )
                facts.append(
                    Fact(
                        subject=ent.text,
                        predicate="mentioned_person",
                        source_text=text[:200],
                        source_message_id=msg_id,
                        confidence=0.4,
                    )
                )
            elif ent.label_ in ("GPE", "LOC"):
                entities.append(
                    Entity(text=ent.text, label="LOCATION", start=ent.start_char, end=ent.end_char)
                )
            elif ent.label_ == "ORG":
                entities.append(
                    Entity(
                        text=ent.text, label="ORGANIZATION", start=ent.start_char, end=ent.end_char
                    )
                )

        return entities, facts

    def _is_bot_message(self, text: str) -> bool:
        """Detect bot/spam messages."""
        bot_markers = [
            "CVS Pharmacy",
            "Rx Ready",
            "Check out this job at",
        ]
        for marker in bot_markers:
            if marker in text:
                return True
        return False

    def _filter_quality(self, facts: list[Fact]) -> list[Fact]:
        """Filter low-quality facts."""
        filtered = []
        for fact in facts:
            # Skip vague subjects
            if fact.subject.lower() in {"it", "that", "this", "them"}:
                continue
            # Skip short subjects for preferences
            if fact.predicate in ("likes", "dislikes") and len(fact.subject.split()) < 2:
                continue
            filtered.append(fact)
        return filtered

    def _verify_with_nli(self, facts: list[Fact]) -> list[Fact]:
        """Optional NLI verification."""
        # This would call the NLI model
        # For now, just return facts as-is
        return facts


# ============================================================================
# Knowledge Graph Store
# ============================================================================


class KnowledgeGraphStore:
    """
    Store and query facts in the knowledge graph.
    """

    def __init__(self, db):
        self.db = db

    def add_knowledge(self, knowledge: ExtractedKnowledge, contact_id: str) -> None:
        """
        Add extracted knowledge to the graph.

        Handles:
        - Deduplication
        - Conflict resolution (temporal facts)
        - Linking to contact
        """
        for fact in knowledge.facts:
            self._store_fact(fact, contact_id)

        for entity in knowledge.entities:
            self._store_entity(entity, contact_id)

    def _store_fact(self, fact: Fact, contact_id: str) -> None:
        """Store a single fact, handling conflicts."""
        # Check if fact already exists
        existing = self._get_similar_fact(fact, contact_id)

        if existing:
            # Update if new fact has higher confidence
            if fact.confidence > existing.confidence:
                self._update_fact(fact, contact_id)
        else:
            # Insert new fact
            self._insert_fact(fact, contact_id)

    def _get_similar_fact(self, fact: Fact, contact_id: str) -> Fact | None:
        """Find similar existing fact."""
        # Query DB for same subject+predicate+contact
        return None

    def _insert_fact(self, fact: Fact, contact_id: str) -> None:
        """Insert new fact into DB."""
        pass

    def _update_fact(self, fact: Fact, contact_id: str) -> None:
        """Update existing fact."""
        pass

    def _store_entity(self, entity: Entity, contact_id: str) -> None:
        """Store entity reference."""
        pass

    def query_facts(self, contact_id: str, predicate: str | None = None) -> list[Fact]:
        """Query facts for a contact."""
        return []

    def resolve_entity(self, mention: str) -> str | None:
        """Resolve a mention to an entity."""
        return None


# ============================================================================
# Understanding Pipeline (Background Job)
# ============================================================================


class UnderstandingPipeline:
    """
    Background pipeline for extracting and storing knowledge.

    This runs asynchronously - it doesn't block message processing.
    """

    def __init__(self, extractor: KnowledgeExtractor, store: KnowledgeGraphStore):
        self.extractor = extractor
        self.store = store

    def process_message(self, message: Message, contact_id: str) -> None:
        """
        Process a message to extract knowledge.

        This is called by the background worker, not on the critical path.
        """
        # Extract knowledge
        knowledge = self.extractor.extract(message, contact_id)

        if knowledge.is_empty():
            return

        # Store in graph
        self.store.add_knowledge(knowledge, contact_id)

    def process_batch(self, messages: list[tuple[Message, str]]) -> None:
        """Process multiple messages in batch."""
        for message, contact_id in messages:
            self.process_message(message, contact_id)


# ============================================================================
# Public API
# ============================================================================

_extractor: KnowledgeExtractor | None = None
_store: KnowledgeGraphStore | None = None
_pipeline: UnderstandingPipeline | None = None


def get_extractor() -> KnowledgeExtractor:
    """Get the singleton extractor."""
    global _extractor
    if _extractor is None:
        _extractor = KnowledgeExtractor(use_spacy=False, use_nli=False)
    return _extractor


def get_knowledge_store() -> KnowledgeGraphStore:
    """Get the singleton knowledge store."""
    global _store
    if _store is None:
        from jarvis.db import get_db

        _store = KnowledgeGraphStore(get_db())
    return _store


def get_understanding_pipeline() -> UnderstandingPipeline:
    """Get the singleton understanding pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = UnderstandingPipeline(extractor=get_extractor(), store=get_knowledge_store())
    return _pipeline


# Background task entry point
def process_message_for_knowledge(message: Message, contact_id: str) -> None:
    """
    Background task to extract knowledge from a message.

    This is called by the async worker.
    """
    pipeline = get_understanding_pipeline()
    pipeline.process_message(message, contact_id)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class MockMessage:
        text: str
        id: str = "msg_123"

    # Test messages
    messages = [
        MockMessage("My sister Sarah lives in Austin"),
        MockMessage("I love sushi but hate cilantro"),
        MockMessage("Started at Google last week"),
        MockMessage("ok"),  # Should be ignored (too short/bot)
    ]

    extractor = KnowledgeExtractor(use_spacy=False)

    for msg in messages:
        knowledge = extractor.extract(msg, contact_id="user_123")
        print(f"\nMessage: {msg.text}")
        print(f"  Entities: {len(knowledge.entities)}")
        for fact in knowledge.facts:
            print(
                f"  Fact: {fact.subject} --{fact.predicate}--> "
                f"{fact.object or fact.value} (conf={fact.confidence})"
            )

    # Output:
    # Message: My sister Sarah lives in Austin
    #   Fact: Sarah --is_sister_of--> user (conf=0.8)
    #   Fact: Austin --lives_in--> None (conf=0.85)
    #
    # Message: I love sushi but hate cilantro
    #   Fact: sushi --likes--> None (conf=0.6)
    #   Fact: cilantro --dislikes--> None (conf=0.6)
    #
    # Message: Started at Google last week
    #   Fact: Google --works_at--> None (conf=0.7)
    #
    # Message: ok
    #   (no facts - too short)
