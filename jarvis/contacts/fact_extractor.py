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
from jarvis.contacts.junk_filters import is_bot_message, is_professional_message
from jarvis.contracts.pipeline import Fact as ContractFact

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

# Temporal markers for fact dating
PAST_MARKERS = {"was", "were", "had", "used to", "grew up", "previously", "before", "last year"}
PRESENT_MARKERS = {"am", "is", "are", "currently", "now", "these days", "lately"}
FUTURE_MARKERS = {"will", "going to", "planning to", "moving to", "starting at", "next week"}

# Location patterns with temporal awareness
# Past: "moved from NYC", "grew up in Texas"
# Present: "live in Austin", "based in SF"
# Future: "moving to LA", "heading to Paris"
LOCATION_PAST_PATTERN = re.compile(
    r"\b(?:moved?\s+from|grew\s+up\s+in|was\s+based\s+in|lived\s+in|from)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
    re.IGNORECASE,
)

LOCATION_PRESENT_PATTERN = re.compile(
    r"\b(?:live[sd]?\s+in|living\s+in|based\s+in|currently\s+in)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
    re.IGNORECASE,
)

LOCATION_FUTURE_PATTERN = re.compile(
    r"\b(?:moving\s+to|relocating\s+to|heading\s+to|going\s+to|travel(?:ing|ed)?\s+to|"
    r"visiting)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
    re.IGNORECASE,
)

# Legacy pattern (keep for compatibility)
LOCATION_PATTERN = re.compile(
    r"\b(?:moved?\s+to|live[sd]?\s+in|living\s+in|from|based\s+in|"
    r"heading\s+to|going\s+to|visiting|relocated\s+to|travel(?:ing|ed)?\s+to)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
)

# Work patterns: "work at Google", "started at Meta", "job at Apple"
# Improved: better company name capture
WORK_PATTERN = re.compile(
    r"\b(?:work(?:s|ing|ed)?\s+(?:at|for)|job\s+(?:at|with)|started\s+(?:at|with)|"
    r"joined|hired\s+(?:at|by|with)|employed\s+(?:at|by))\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
)

# Preference patterns: "hate cilantro", "love sushi", "allergic to"
# Improved: capture more complete phrases (4-5 words max to avoid runaway)
PREFERENCE_PATTERN = re.compile(
    r"\b(?:hate[sd]?|love[sd]?|can't\s+stand|allergic\s+to|"
    r"obsessed\s+with|addicted\s+to|favorite\s+(?:food|thing|person|place|movie|band)\s+is|"
    r"dislike[sd]?|enjoy[s]?|prefer|like[sd]?)\s+"
    r"([A-Za-z\s]+?)(?:\s+(?:and|but|or|because|when|if|since)\b|\.|\,|!|\?|$)",
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
    Includes quality filters to reject bot messages, vague subjects, and short phrases.
    """

    def __init__(
        self,
        entailment_threshold: float = 0.7,
        use_nli: bool = True,  # Enable by default
        confidence_threshold: float = 0.5,
    ) -> None:
        self.threshold = entailment_threshold
        self.use_nli = use_nli
        self.confidence_threshold = confidence_threshold
        self._nlp = None
        self._contacts_cache: dict[str, Any] | None = None

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
            Deduplicated list of extracted facts (filtered by quality).
        """
        now = datetime.now().isoformat()
        facts: list[Fact] = []

        for msg in messages:
            text = (
                msg.get("text", "") if isinstance(msg, dict) else (getattr(msg, "text", None) or "")
            )
            msg_id = msg.get("id") if isinstance(msg, dict) else getattr(msg, "id", None)
            if not text or len(text) < 5:
                continue

            # Skip bot messages before extraction
            if self._is_bot_message(text, contact_id):
                logger.debug("Skipping bot message: %s", text[:50])
                continue

            # Skip professional messages (emails, formal correspondence)
            if self._is_professional_message(text):
                logger.debug("Skipping professional message: %s", text[:50])
                continue

            extracted = self._extract_rule_based(text, contact_id, now)
            # Attach source message ID
            for fact in extracted:
                fact.source_message_id = msg_id
            facts.extend(extracted)

        # Apply quality filters and recalibrate confidence
        facts = self._apply_quality_filters(facts)

        # Deduplicate
        facts = self._deduplicate(facts)

        # NLI verification pass
        if self.use_nli and facts:
            facts = self._verify_facts_nli(facts)

        logger.info(
            "Extracted %d facts from %d messages for %s (after quality filtering)",
            len(facts),
            len(messages),
            contact_id[:16],
        )
        return facts

    # =========================================================================
    # Quality Filtering Methods
    # =========================================================================

    def _is_like_filler_word(self, text: str, match_start: int, match_end: int) -> bool:
        """Detect if 'like' in this context is a filler word, not a preference verb.

        Filler patterns:
        - "it's like X" (that's like, it's like)
        - "that's like X"
        - "<verb> like X" where verb != like (looks like, feels like, sounds like)
        - "like okay", "like yeah", "like what"
        - Mid-sentence: "so like I was..." (conversational filler)
        """
        # Look at context around the match (50 chars before/after)
        context_start = max(0, match_start - 50)
        context_end = min(len(text), match_end + 50)
        context = text[context_start:context_end].lower()

        # Pattern: X is/was/feels like (not preference)
        if re.search(r"(it's|that's|feels?|looks?|sounds?)\s+like\b", context):
            return True

        # Pattern: like + discourse marker (okay, yeah, what, so, you know)
        if re.search(r"\blike\s+(okay|yeah|yeah?|what|so|you know|omg|lol)", context):
            return True

        # Pattern: mid-sentence filler "so like" or "and like"
        if re.search(r"(?:^|[,;])\s+like\s+", context):
            return True

        # Pattern: "I like" at sentence start is usually preference, allow it
        # Pattern: "I was like" at sentence start is likely filler, reject it
        if re.search(r"^i\s+was\s+like\b", context):
            return True

        return False

    def _is_professional_message(self, text: str) -> bool:
        """Delegate to shared junk filter. See jarvis.contacts.junk_filters."""
        return is_professional_message(text)

    def _is_coherent_subject(self, subject: str) -> bool:
        """Reject subjects that are vague pronouns or incomplete fragments.

        Checks:
        - Single pronouns: "it", "that", "this", "them", "there"
        - Pronoun phrases: "it in", "that in", "it there"
        - Incomplete infinitives: "to call this", "to have you"
        - Bare prepositions: "in august", "at night"
        - Too many abbreviations: "sm rn" (slang overload)
        - Malformed: missing spaces, consecutive capitals
        """
        subject_lower = subject.lower().strip()

        # Single pronouns or vague words
        vague_words = {
            "it",
            "that",
            "this",
            "them",
            "there",
            "those",
            "these",
            "what",
            "when",
            "where",
            "why",
            "how",
        }
        if subject_lower in vague_words:
            return False

        # Pronoun + preposition (fragment pattern) - but NOT full NP with content
        # "it in August" is bad, "cilantro in my food" is good
        if re.match(r"^(it|that|this|them|there)\s+(in|at|on|for|to)", subject_lower):
            return False

        # Incomplete infinitive phrase (to + verb + object cutoff)
        if re.match(r"^to\s+\w+\s+(this|that|these|those|me|you|him|her|it)$", subject_lower):
            return False

        # Bare time/location prepositions (nothing before the preposition)
        if re.match(
            r"^(in|at|on)\s+(august|spring|summer|winter|night|day|morning|afternoon)$",
            subject_lower,
        ):
            return False

        # Too many abbreviations (>50% of words are 1-2 chars)
        words = subject.split()
        if len(words) > 1:
            short_words = sum(
                1
                for w in words
                if len(w) <= 2 and w.lower() not in {"i", "a", "to", "of", "in", "at", "on"}
            )
            if len(words) >= 2 and short_words / len(words) > 0.5:
                return False

        # Malformed: word spacing issues (e.g., "ofmetal" instead of "of metal")
        # Check for lowercase after uppercase in non-standard way
        if re.search(r"[a-z]{2,}[A-Z][a-z]+", subject):
            return False

        # Must have at least 2 characters and contain a letter
        if len(subject) < 2 or not any(c.isalpha() for c in subject):
            return False

        return True

    def _is_bot_message(self, text: str, chat_id: str = "") -> bool:
        """Delegate to shared junk filter. See jarvis.contacts.junk_filters."""
        return is_bot_message(text, chat_id)

    def _is_vague_subject(self, subject: str) -> bool:
        """Reject vague subjects that are pronouns losing context.

        Pronouns to reject: me, you, that, this, it, them, he, she
        """
        vague_pronouns = {"me", "you", "that", "this", "it", "them", "he", "she"}
        return subject.lower().strip() in vague_pronouns

    def _is_too_short(self, category: str, subject: str) -> bool:
        """Reject facts with subjects too short for category.

        Category-specific thresholds:
        - preference: require min 3 words (context crucial: "driving in sf")
        - relationship: allow names (1+ word): "Sarah", "Mom"
        - work/location: allow names (1+ word for proper nouns)
        - event: require 2+ words

        Exceptions:
        - Single proper nouns are always OK (Company, Name, City)
        - Subjects starting with capital letter get leniency
        """
        word_count = len(subject.split())

        # Preference needs full context
        if category == "preference":
            return word_count < 3

        # Relationship allows single names
        if category == "relationship":
            return word_count < 1  # Always allow

        # Work/location allow single proper nouns (capitalized)
        if category in ("work", "location"):
            # Single word is OK if it's a proper noun (starts with capital)
            if word_count == 1:
                return not subject[0].isupper()  # OK if uppercase, too short if lowercase
            # Multi-word is always OK
            return False

        return False

    def _calculate_confidence(
        self,
        base_confidence: float,
        category: str,
        subject: str,
        is_vague: bool,
        is_short: bool,
    ) -> float:
        """Recalibrate confidence based on quality factors.

        Adjustments:
        - Vague subject: multiply by 0.5
        - Short phrase: multiply by 0.7
        - Rich context (4+ words): multiply by 1.1
        """
        adjusted = base_confidence

        if is_vague:
            adjusted *= 0.5
        elif is_short:
            adjusted *= 0.7

        # Bonus for rich context (4+ words)
        word_count = len(subject.split())
        if word_count >= 4:
            adjusted = min(adjusted * 1.1, 1.0)

        return adjusted

    def _apply_quality_filters(self, facts: list[Fact]) -> list[Fact]:
        """Filter facts by quality and recalibrate confidence.

        Returns only facts with confidence >= threshold after filtering.
        """
        filtered: list[Fact] = []

        for fact in facts:
            # Check coherence (new)
            is_coherent = self._is_coherent_subject(fact.subject)
            if not is_coherent:
                logger.debug(
                    "Rejecting incoherent subject: %s (category=%s)",
                    fact.subject,
                    fact.category,
                )
                continue

            # Check vague subject
            is_vague = self._is_vague_subject(fact.subject)
            if is_vague:
                logger.debug(
                    "Rejecting vague subject: %s (category=%s)",
                    fact.subject,
                    fact.category,
                )
                continue

            # Check short phrase
            is_short = self._is_too_short(fact.category, fact.subject)

            # Recalibrate confidence
            adjusted_confidence = self._calculate_confidence(
                fact.confidence,
                fact.category,
                fact.subject,
                is_vague,
                is_short,
            )

            # Only keep if confidence >= threshold
            if adjusted_confidence < self.confidence_threshold:
                logger.debug(
                    "Rejecting low-confidence fact: %s/%s (conf=%.2f, adjusted=%.2f)",
                    fact.category,
                    fact.subject,
                    fact.confidence,
                    adjusted_confidence,
                )
                continue

            # Update confidence and keep
            fact.confidence = adjusted_confidence
            filtered.append(fact)

        logger.debug(
            "Quality filtering: %d → %d facts (%.1f%% kept)",
            len(facts),
            len(filtered),
            (len(filtered) / len(facts) * 100) if facts else 0,
        )
        return filtered

    def _verify_facts_nli(self, facts: list[Fact]) -> list[Fact]:
        """Filter facts by NLI entailment verification (batched).

        Skips verification if the NLI model is not already loaded to avoid
        cold-loading a heavy model mid-extraction (adds 2-7s of latency).
        """
        try:
            from models.nli_cross_encoder import _nli_encoder

            # Skip NLI if model not already warm - cold load is too expensive
            # for inline extraction
            if _nli_encoder is None or not _nli_encoder.is_loaded():
                logger.debug("NLI model not warm, skipping verification")
                return facts

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
                        fact.category,
                        fact.subject,
                        score,
                    )
            return verified
        except Exception as e:
            logger.warning("NLI verification failed, returning unverified facts: %s", e)
            return facts

    def _clean_subject(self, subject: str) -> str:
        """Clean extracted subject: strip trailing prepositions, conjunctions, etc."""
        subject = subject.strip()

        # Remove trailing prepositions and conjunctions
        trailing_words = [
            r"\s+(and|but|or|because|when|if|since|unless|while|though|although)$",
            r"\s+(is|are|was|were|be|been)$",
        ]
        for pattern in trailing_words:
            subject = re.sub(pattern, "", subject, flags=re.IGNORECASE)

        # Remove trailing whitespace again
        return subject.strip()

    def _extract_rule_based(self, text: str, contact_id: str, timestamp: str) -> list[Fact]:
        """Extract facts using regex patterns."""
        facts: list[Fact] = []

        # Relationship patterns
        for match in RELATIONSHIP_PATTERN.finditer(text):
            rel_type = match.group(1).lower()
            person = match.group(2).strip()
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

        # Location patterns - temporal aware
        # Present locations (highest confidence - current residence)
        for match in LOCATION_PRESENT_PATTERN.finditer(text):
            location = self._clean_subject(match.group(1))
            if location:
                facts.append(
                    Fact(
                        category="location",
                        subject=location,
                        predicate="lives_in",
                        source_text=text[:200],
                        confidence=0.85,  # Higher confidence for present tense
                        contact_id=contact_id,
                        extracted_at=timestamp,
                        valid_from=timestamp,  # Current location
                    )
                )

        # Future locations (moving to)
        for match in LOCATION_FUTURE_PATTERN.finditer(text):
            location = self._clean_subject(match.group(1))
            if location:
                facts.append(
                    Fact(
                        category="location",
                        subject=location,
                        predicate="moving_to",
                        source_text=text[:200],
                        confidence=0.6,
                        contact_id=contact_id,
                        extracted_at=timestamp,
                        valid_from=timestamp,  # Will be valid from now
                    )
                )

        # Past locations (grew up in, moved from)
        for match in LOCATION_PAST_PATTERN.finditer(text):
            location = self._clean_subject(match.group(1))
            if location:
                facts.append(
                    Fact(
                        category="location",
                        subject=location,
                        predicate="lived_in",
                        source_text=text[:200],
                        confidence=0.5,  # Lower confidence for past
                        contact_id=contact_id,
                        extracted_at=timestamp,
                        valid_until=timestamp,  # No longer valid
                    )
                )

        # Legacy pattern (fallback)
        for match in LOCATION_PATTERN.finditer(text):
            location = self._clean_subject(match.group(1))
            if location:
                # Check if already extracted by temporal patterns
                already_extracted = any(
                    f.category == "location" and f.subject.lower() == location.lower()
                    for f in facts
                )
                if not already_extracted:
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
            org = self._clean_subject(match.group(1))
            if org:  # Only add if non-empty after cleaning
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
            # Skip if "like" is a filler word, not preference verb
            if "like" in match.group(0).lower() and self._is_like_filler_word(
                text, match.start(), match.end()
            ):
                logger.debug(f"Skipping filler 'like': {match.group(0)[:60]}...")
                continue

            thing = self._clean_subject(match.group(1))
            if not thing:  # Skip if empty after cleaning
                continue

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

    # =========================================================================
    # NER Person Extraction
    # =========================================================================

    def _extract_person_facts_ner(self, text: str, contact_id: str, timestamp: str) -> list[Fact]:
        """Extract PERSON entities from text using spaCy NER.

        Resolves person names to contacts and creates relationship facts.
        Returns empty list if spaCy unavailable.
        """
        nlp = self._get_nlp()
        if nlp is None:
            return []

        facts: list[Fact] = []
        try:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON" and len(ent.text) > 1:
                    person_name = ent.text.strip()

                    # Try to resolve to contact
                    _linked_contact_id = self._resolve_person_to_contact(person_name)

                    facts.append(
                        Fact(
                            category="relationship",
                            subject=person_name,
                            predicate="mentioned_person",
                            source_text=text[:200],
                            confidence=0.5,
                            contact_id=contact_id,
                            extracted_at=timestamp,
                        )
                    )
        except Exception as e:
            logger.warning("NER person extraction failed: %s", e)

        return facts

    def _resolve_person_to_contact(self, person_name: str) -> str | None:
        """Fuzzy match person name to contact in database.

        Uses token-based matching: require unique match with confidence >= 0.7
        or clear winner with 0.2+ gap over runner-up.

        Returns:
            Contact ID if match found, None otherwise.
        """
        try:
            from jarvis.db import get_db

            db = get_db()

            # Get all contacts
            with db.connection() as conn:
                cursor = conn.execute("SELECT id, display_name FROM contacts")
                contacts = cursor.fetchall()

            if not contacts:
                return None

            # Simple token-based fuzzy match
            name_tokens = set(person_name.lower().split())
            scores: dict[int, float] = {}

            for contact_id, display_name in contacts:
                contact_tokens = set(display_name.lower().split())
                if not contact_tokens:
                    continue

                # Jaccard similarity: intersection / union
                intersection = len(name_tokens & contact_tokens)
                union = len(name_tokens | contact_tokens)
                similarity = intersection / union if union > 0 else 0
                scores[contact_id] = similarity

            if not scores:
                return None

            # Sort by score descending
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_score = sorted_scores[0][1]

            # Require high confidence or clear winner
            if top_score < 0.7:
                return None

            # Check for clear winner (gap >= 0.2)
            if len(sorted_scores) > 1:
                runner_up_score = sorted_scores[1][1]
                if top_score - runner_up_score < 0.2:
                    return None  # ambiguous match

            return str(sorted_scores[0][0])

        except Exception as e:
            logger.warning("Person resolution failed: %s", e)
            return None


def fact_to_contract(fact: Fact) -> ContractFact:
    """Convert an internal Fact to the pipeline ContractFact type.

    Maps the richer internal Fact (with category, value, contact_id, etc.)
    to the minimal contract Fact(subject, predicate, object, confidence, source_text).
    """
    return ContractFact(
        subject=fact.subject,
        predicate=fact.predicate,
        object=fact.value or fact.category,
        confidence=fact.confidence,
        source_text=fact.source_text,
    )


def facts_to_contract(facts: list[Fact]) -> list[ContractFact]:
    """Convert a list of internal Facts to pipeline ContractFact types."""
    return [fact_to_contract(f) for f in facts]
