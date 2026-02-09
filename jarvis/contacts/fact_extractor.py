"""Fact extraction pipeline for contact knowledge.

Extracts structured facts (relationships, locations, work, preferences, events)
from chat messages using NER + rule-based patterns + optional NLI verification.

Pipeline: NER → rule-based pre-filter → NLI verification → dedup against DB.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from jarvis.contacts.contact_profile import Fact

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Possessive relationship patterns: "my sister Sarah", "my friend John"
# Name group requires uppercase start (no IGNORECASE - "my/My" both matched by [Mm]y)
RELATIONSHIP_PATTERN = re.compile(
    r"\b[Mm]y\s+(sister|brother|mom|mother|dad|father|wife|husband|"
    r"girlfriend|boyfriend|partner|daughter|son|cousin|aunt|uncle|"
    r"[Gg]randma|[Gg]randmother|[Gg]randpa|[Gg]randfather|friend|best friend|"
    r"roommate|fiancée?|boss|coworker|colleague|neighbor)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
)

# Location patterns: "moved to Austin", "live in NYC", "going to Paris"
LOCATION_PATTERN = re.compile(
    r"\b(?:moved?\s+to|live[sd]?\s+in|living\s+in|from|based\s+in|"
    r"heading\s+to|going\s+to|visiting)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
)

# Work patterns: "work at Google", "started at Meta", "job at Apple"
WORK_PATTERN = re.compile(
    r"\b(?:work(?:s|ing|ed)?\s+(?:at|for)|job\s+at|started\s+at|"
    r"joined|hired\s+(?:at|by))\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
)

# Preference patterns: "hate cilantro", "love sushi", "allergic to"
# Stop at conjunctions (and, but, or) and punctuation to avoid over-capture
PREFERENCE_PATTERN = re.compile(
    r"\b(?:hate[sd]?|love[sd]?|can't\s+stand|allergic\s+to|"
    r"obsessed\s+with|addicted\s+to|favorite\s+\w+\s+is)\s+"
    r"(\w+(?:\s+(?!and\b|but\b|or\b)\w+){0,2})",
    re.IGNORECASE,
)

# Sentiment words for preference extraction
POSITIVE_PREF = {"love", "loves", "loved", "obsessed", "addicted", "favorite"}
NEGATIVE_PREF = {"hate", "hates", "hated", "can't stand", "allergic"}


@dataclass
class EntitySpan:
    """A named entity extracted from text."""

    text: str
    label: str  # PERSON, GPE, ORG, DATE, etc.
    start: int
    end: int


class FactExtractor:
    """Extracts structured facts from messages.

    Uses rule-based patterns as the primary extraction method.
    Optionally uses spaCy NER for entity detection and NLI for verification.
    """

    def __init__(
        self,
        entailment_threshold: float = 0.7,
        use_nli: bool = False,
    ) -> None:
        self.threshold = entailment_threshold
        self.use_nli = use_nli
        self._nlp = None

    def _get_nlp(self) -> Any:
        """Lazy-load spaCy model."""
        if self._nlp is None:
            try:
                import spacy

                self._nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.warning("spaCy not available, using regex-only extraction: %s", e)
                self._nlp = False  # sentinel: don't retry
        return self._nlp if self._nlp is not False else None

    def extract_facts(self, messages: list[Any], contact_id: str = "") -> list[Fact]:
        """Extract facts from a list of messages.

        Args:
            messages: Message objects with .text attribute, or dicts with "text" key.
            contact_id: ID of the contact these messages belong to.

        Returns:
            Deduplicated list of extracted facts.
        """
        now = datetime.now().isoformat()
        facts: list[Fact] = []

        for msg in messages:
            text = msg.get("text", "") if isinstance(msg, dict) else (getattr(msg, "text", None) or "")
            msg_id = msg.get("id") if isinstance(msg, dict) else getattr(msg, "id", None)
            if not text or len(text) < 5:
                continue

            extracted = self._extract_rule_based(text, contact_id, now)
            # Attach source message ID
            for fact in extracted:
                fact.source_message_id = msg_id
            facts.extend(extracted)

        # Deduplicate
        facts = self._deduplicate(facts)

        # NLI verification pass
        if self.use_nli and facts:
            facts = self._verify_facts_nli(facts)

        logger.info(
            "Extracted %d facts from %d messages for %s",
            len(facts),
            len(messages),
            contact_id[:16],
        )
        return facts

    def _verify_facts_nli(self, facts: list[Fact]) -> list[Fact]:
        """Filter facts by NLI entailment verification (batched)."""
        try:
            from jarvis.nlp.entailment import (
                fact_to_hypothesis,
                verify_entailment_batch,
            )

            # Build (premise, hypothesis) pairs for batch scoring
            pairs: list[tuple[str, str]] = []
            for fact in facts:
                hypothesis = fact_to_hypothesis(
                    fact.category, fact.subject, fact.predicate, fact.value
                )
                pairs.append((fact.source_text, hypothesis))

            results = verify_entailment_batch(pairs, threshold=self.threshold)

            verified: list[Fact] = []
            for fact, (is_entailed, score) in zip(facts, results):
                if is_entailed:
                    fact.confidence = score
                    verified.append(fact)
                else:
                    logger.debug(
                        "NLI rejected fact: %s/%s (score=%.2f)",
                        fact.category, fact.subject, score,
                    )
            return verified
        except Exception as e:
            logger.warning("NLI verification failed, returning unverified facts: %s", e)
            return facts

    def _extract_rule_based(self, text: str, contact_id: str, timestamp: str) -> list[Fact]:
        """Extract facts using regex patterns."""
        facts: list[Fact] = []

        # Relationship patterns
        for match in RELATIONSHIP_PATTERN.finditer(text):
            rel_type = match.group(1).lower()
            person = match.group(2)
            facts.append(
                Fact(
                    category="relationship",
                    subject=person,
                    predicate="is_family_of"
                    if rel_type
                    in (
                        "sister",
                        "brother",
                        "mom",
                        "mother",
                        "dad",
                        "father",
                        "wife",
                        "husband",
                        "daughter",
                        "son",
                        "cousin",
                        "aunt",
                        "uncle",
                        "grandma",
                        "grandmother",
                        "grandpa",
                        "grandfather",
                    )
                    else "is_friend_of"
                    if "friend" in rel_type
                    else "is_associated_with",
                    value=rel_type,
                    source_text=text[:200],
                    confidence=0.8,
                    contact_id=contact_id,
                    extracted_at=timestamp,
                )
            )

        # Location patterns
        for match in LOCATION_PATTERN.finditer(text):
            location = match.group(1)
            facts.append(
                Fact(
                    category="location",
                    subject=location,
                    predicate="lives_in",
                    source_text=text[:200],
                    confidence=0.7,
                    contact_id=contact_id,
                    extracted_at=timestamp,
                )
            )

        # Work patterns
        for match in WORK_PATTERN.finditer(text):
            org = match.group(1)
            facts.append(
                Fact(
                    category="work",
                    subject=org,
                    predicate="works_at",
                    source_text=text[:200],
                    confidence=0.7,
                    contact_id=contact_id,
                    extracted_at=timestamp,
                )
            )

        # Preference patterns
        for match in PREFERENCE_PATTERN.finditer(text):
            thing = match.group(1).strip()
            # Determine sentiment
            match_text = match.group(0).lower()
            if any(w in match_text for w in NEGATIVE_PREF):
                predicate = "dislikes"
            else:
                predicate = "likes"
            facts.append(
                Fact(
                    category="preference",
                    subject=thing,
                    predicate=predicate,
                    source_text=text[:200],
                    confidence=0.6,
                    contact_id=contact_id,
                    extracted_at=timestamp,
                )
            )

        return facts

    def _deduplicate(self, facts: list[Fact]) -> list[Fact]:
        """Remove duplicate facts based on (category, subject_lower, predicate)."""
        seen: set[tuple[str, str, str]] = set()
        unique: list[Fact] = []
        for fact in facts:
            key = (fact.category, fact.subject.lower().strip(), fact.predicate)
            if key not in seen:
                seen.add(key)
                unique.append(fact)
        return unique

    def extract_facts_with_ner(self, messages: list[Any], contact_id: str = "") -> list[Fact]:
        """Extract facts using spaCy NER + rule-based patterns.

        Falls back to pure rule-based if spaCy is unavailable.
        """
        nlp = self._get_nlp()
        if nlp is None:
            return self.extract_facts(messages, contact_id)

        now = datetime.now().isoformat()
        facts: list[Fact] = []

        # Process in chunks for memory efficiency
        chunk_size = 100
        texts = [getattr(m, "text", "") or "" for m in messages if getattr(m, "text", None)]

        for i in range(0, len(texts), chunk_size):
            chunk = texts[i : i + chunk_size]
            docs = list(nlp.pipe(chunk, batch_size=50))

            for doc in docs:
                text = doc.text
                # Rule-based first
                facts.extend(self._extract_rule_based(text, contact_id, now))

                # NER-enhanced: extract entities spaCy found
                for ent in doc.ents:
                    if ent.label_ == "PERSON" and len(ent.text) > 1:
                        # Check if already captured by relationship pattern
                        if not any(
                            f.subject.lower() == ent.text.lower()
                            for f in facts
                            if f.category == "relationship"
                        ):
                            facts.append(
                                Fact(
                                    category="relationship",
                                    subject=ent.text,
                                    predicate="mentioned_person",
                                    source_text=text[:200],
                                    confidence=0.4,
                                    contact_id=contact_id,
                                    extracted_at=now,
                                )
                            )
                    elif ent.label_ in ("GPE", "LOC") and len(ent.text) > 1:
                        if not any(
                            f.subject.lower() == ent.text.lower()
                            for f in facts
                            if f.category == "location"
                        ):
                            facts.append(
                                Fact(
                                    category="location",
                                    subject=ent.text,
                                    predicate="mentioned_location",
                                    source_text=text[:200],
                                    confidence=0.3,
                                    contact_id=contact_id,
                                    extracted_at=now,
                                )
                            )
                    elif ent.label_ == "ORG" and len(ent.text) > 1:
                        if not any(
                            f.subject.lower() == ent.text.lower()
                            for f in facts
                            if f.category == "work"
                        ):
                            facts.append(
                                Fact(
                                    category="work",
                                    subject=ent.text,
                                    predicate="mentioned_org",
                                    source_text=text[:200],
                                    confidence=0.3,
                                    contact_id=contact_id,
                                    extracted_at=now,
                                )
                            )

        return self._deduplicate(facts)
