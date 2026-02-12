"""Fact extraction pipeline for contact knowledge.

Extracts structured facts (relationships, locations, work, preferences, events)
from chat messages using NER + rule-based patterns + optional NLI verification.

Pipeline: NER → rule-based pre-filter → NLI verification → dedup against DB.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from jarvis.contacts.attribution import AttributionResolver
from jarvis.contacts.contact_profile import Fact
from jarvis.contacts.junk_filters import (
    is_bot_message,
    is_code_message,
    is_professional_message,
    is_tapback_reaction,
)
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
    r"\b(?:[Mm]oved?\s+from|[Gg]rew\s+up\s+in|[Ww]as\s+based\s+in|[Ll]ived\s+in)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
)

LOCATION_PRESENT_PATTERN = re.compile(
    r"\b(?:[Ll]ive[sd]?\s+in|[Ll]iving\s+in|[Bb]ased\s+in|[Cc]urrently\s+in)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
)

LOCATION_FUTURE_PATTERN = re.compile(
    r"\b(?:[Mm]oving\s+to|[Rr]elocating\s+to)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
)

# Legacy pattern (keep for compatibility)
LOCATION_PATTERN = re.compile(
    r"\b(?:[Mm]oved?\s+to|[Ll]ive[sd]?\s+in|[Ll]iving\s+in|[Bb]ased\s+in|"
    r"[Rr]elocated\s+to)\s+"
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
    r"dislike[sd]?|enjoy[s]?|prefer)\s+"
    r"([A-Za-z\s]+?)(?:\s+(?:and|but|or|because|when|if|since)\b|\.|\,|!|\?|$)",
    re.IGNORECASE,
)

# Quality filter patterns (pre-compiled for hot-path usage)
_VAGUE_PRONOUN_RE = re.compile(r"^(it|that|this|them|there)\s+(in|at|on|for|to)")
_INCOMPLETE_INFINITIVE_RE = re.compile(
    r"^to\s+\w+\s+(this|that|these|those|me|you|him|her|it)$",
)
_BARE_PREPOSITION_RE = re.compile(
    r"^(in|at|on)\s+(august|spring|summer|winter|night|day|morning|afternoon)$",
)
_CAMELCASE_RE = re.compile(r"[a-z]{2,}[A-Z][a-z]+")
_TRAILING_WORD_PATTERNS = [
    re.compile(
        r"\s+(and|but|or|because|when|if|since|unless|while|though|although)$", re.IGNORECASE,
    ),
    re.compile(r"\s+(is|are|was|were|be|been)$", re.IGNORECASE),
]

# Sentiment words for preference extraction
POSITIVE_PREF = {"love", "loves", "loved", "obsessed", "addicted", "favorite"}
NEGATIVE_PREF = {"hate", "hates", "hated", "can't stand", "allergic"}

# Family relationship types for predicate resolution
_FAMILY_RELATIONS = frozenset(
    {
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
    }
)


@dataclass
class _RulePattern:
    """Descriptor for a simple rule-based extraction pattern.

    Each instance drives one ``finditer`` loop inside ``_extract_rule_based``.
    Patterns that need custom logic (relationship, preference) use dedicated handlers.
    """

    name: str
    pattern: re.Pattern[str]
    category: str
    predicate: str
    confidence: float
    subject_group: int  # regex group index for the extracted subject
    dedup: bool = False  # skip if (category, subject) already seen
    valid_from_ts: bool = False  # set valid_from=timestamp on the Fact
    valid_until_ts: bool = False  # set valid_until=timestamp on the Fact


# Registry of simple rule patterns (no custom handler needed).
# Order matters: temporal locations before legacy fallback.
_RULE_PATTERNS: list[_RulePattern] = [
    _RulePattern(
        name="location_present",
        pattern=LOCATION_PRESENT_PATTERN,
        category="location",
        predicate="lives_in",
        confidence=0.85,
        subject_group=1,
        valid_from_ts=True,
    ),
    _RulePattern(
        name="location_future",
        pattern=LOCATION_FUTURE_PATTERN,
        category="location",
        predicate="moving_to",
        confidence=0.6,
        subject_group=1,
        valid_from_ts=True,
    ),
    _RulePattern(
        name="location_past",
        pattern=LOCATION_PAST_PATTERN,
        category="location",
        predicate="lived_in",
        confidence=0.5,
        subject_group=1,
        valid_until_ts=True,
    ),
    _RulePattern(
        name="location_legacy",
        pattern=LOCATION_PATTERN,
        category="location",
        predicate="lives_in",
        confidence=0.7,
        subject_group=1,
        dedup=True,
    ),
    _RulePattern(
        name="work",
        pattern=WORK_PATTERN,
        category="work",
        predicate="works_at",
        confidence=0.7,
        subject_group=1,
        dedup=True,
    ),
]


@dataclass
class EntitySpan:
    """A named entity extracted from text."""

    text: str
    label: str  # PERSON, GPE, ORG, DATE, etc.
    start: int
    end: int


# Module-level spaCy singleton (shared across all FactExtractor instances)
_spacy_nlp: Any = None
_spacy_nlp_lock = threading.Lock()


def _get_shared_nlp() -> Any:
    """Return shared spaCy model, loading once on first call (thread-safe)."""
    global _spacy_nlp
    if _spacy_nlp is None:
        with _spacy_nlp_lock:
            if _spacy_nlp is None:
                try:
                    import spacy

                    _spacy_nlp = spacy.load("en_core_web_sm")
                except (ImportError, OSError) as e:
                    logger.warning("spaCy not available, using regex-only extraction: %s", e)
                    _spacy_nlp = False  # sentinel: don't retry
    return _spacy_nlp if _spacy_nlp is not False else None


# Module-level contacts cache with TTL (shared across all FactExtractor instances).
# Avoids repeated DB queries when multiple extractors are instantiated.
_contacts_cache_data: list[tuple[str, str, set[str]]] | None = None
_contacts_cache_time: float = 0.0
_contacts_cache_lock = threading.Lock()
_CONTACTS_CACHE_TTL = 300.0  # 5 minutes


def _get_cached_contacts() -> list[tuple[str, str, set[str]]]:
    """Return cached contacts list, refreshing from DB if TTL expired (thread-safe)."""
    global _contacts_cache_data, _contacts_cache_time
    now = time.monotonic()
    if _contacts_cache_data is not None and (now - _contacts_cache_time) < _CONTACTS_CACHE_TTL:
        return _contacts_cache_data

    with _contacts_cache_lock:
        # Double-check after acquiring lock
        now = time.monotonic()
        if (
            _contacts_cache_data is not None
            and (now - _contacts_cache_time) < _CONTACTS_CACHE_TTL
        ):
            return _contacts_cache_data

        try:
            from jarvis.db import get_db

            db = get_db()
            with db.connection() as conn:
                cursor = conn.execute("SELECT id, display_name FROM contacts")
                rows = cursor.fetchall()

            _contacts_cache_data = [
                (str(cid), name, set(name.lower().split())) for cid, name in rows if name
            ]
        except (OSError, sqlite3.Error, ImportError) as e:
            logger.debug("Contact resolution DB query failed: %s", e)
            _contacts_cache_data = []

        _contacts_cache_time = now
        return _contacts_cache_data


class FactExtractor:
    """Extracts structured facts from messages.

    Uses rule-based patterns as the primary extraction method.
    Optionally uses spaCy NER for entity detection and NLI for verification.
    Includes quality filters to reject bot messages, vague subjects, and short phrases.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self._attribution_resolver = AttributionResolver()

    def _get_nlp(self) -> Any:
        """Return shared spaCy model singleton."""
        return _get_shared_nlp()

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

            # Skip tapback reactions
            if is_tapback_reaction(text):
                continue

            # Skip code snippets sent via iMessage
            if is_code_message(text):
                logger.debug("Skipping code message: %s", text[:50])
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

        # Resolve attribution (who is each fact about?)
        self._resolve_attribution(facts, messages)

        # Deduplicate
        facts = self._deduplicate(facts)

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

    # Pre-compiled filler word patterns (avoid recompiling per match)
    _FILLER_PATTERNS = [
        re.compile(r"(it\u2019s|it's|that\u2019s|that's|feels?|looks?|sounds?)\s+like\b"),
        re.compile(r"\blike\s+(okay|yeah|yeah?|what|so|you know|omg|lol)"),
        re.compile(r"(?:^|[,;])\s+like\s+"),
        re.compile(r"^i\s+was\s+like\b"),
        # "like how", "like when", "like where", "like why", "like if", "like a", "like the"
        re.compile(r"\blike\s+(how|when|where|why|if|a\b|an?\b|the\b)"),
        # "I like just", "I like literally", "I like actually" (filler adverbs)
        re.compile(r"\bi\s+like\s+(just|already|really|literally|actually|basically)"),
        # "like 5 minutes", "like 3 people" (approximation)
        re.compile(r"\blike\s+\d"),
        # "not like that", "isn't like", "wasn't like"
        re.compile(r"\b(not|isn't|wasn't|ain't|don't|doesn't)\s+like\b"),
    ]

    def _is_like_filler_word(self, text: str, match_start: int, match_end: int) -> bool:
        """Detect if 'like' in this context is a filler word, not a preference verb."""
        # Look at context around the match (50 chars before/after)
        context_start = max(0, match_start - 50)
        context_end = min(len(text), match_end + 50)
        context = text[context_start:context_end].lower()

        for pat in self._FILLER_PATTERNS:
            if pat.search(context):
                return True

        return False

    # Words that indicate a captured clause fragment, not a noun phrase subject
    _PREF_REJECT_STARTS = re.compile(
        r"^(how|when|where|why|if|what|i|he|she|they|we|you)\b", re.IGNORECASE
    )
    _PREF_REJECT_VERBS = re.compile(
        r"\b(was|were|did|gonna|going to|realized|thought|said|told|know|knew)\b",
        re.IGNORECASE,
    )

    def _is_valid_preference_subject(self, subject: str) -> bool:
        """Reject preference subjects that are clause fragments, not noun phrases.

        Rejects subjects starting with interrogatives/pronouns or containing
        verb-like patterns that indicate a captured sentence fragment.
        """
        if self._PREF_REJECT_STARTS.match(subject.strip()):
            return False
        if self._PREF_REJECT_VERBS.search(subject):
            return False
        return True

    def _is_professional_message(self, text: str) -> bool:
        """Delegate to shared junk filter. See jarvis.contacts.junk_filters."""
        return is_professional_message(text)

    def _is_vague_pronoun(self, subject_lower: str) -> bool:
        """Check if subject is a vague pronoun or pronoun+preposition fragment."""
        vague_words = {
            "it", "that", "this", "them", "there",
            "those", "these", "what", "when", "where", "why", "how",
        }
        if subject_lower in vague_words:
            return True

        # Pronoun + preposition (fragment pattern)
        # "it in August" is bad, "cilantro in my food" is good
        if _VAGUE_PRONOUN_RE.match(subject_lower):
            return True

        return False

    def _is_incomplete_phrase(self, subject_lower: str) -> bool:
        """Check if subject is an incomplete infinitive or bare prepositional phrase."""
        # Incomplete infinitive phrase (to + verb + object cutoff)
        if _INCOMPLETE_INFINITIVE_RE.match(subject_lower):
            return True

        # Bare time/location prepositions (nothing before the preposition)
        if _BARE_PREPOSITION_RE.match(subject_lower):
            return True

        return False

    def _is_malformed(self, subject: str) -> bool:
        """Check if subject has too many abbreviations, bad spacing, or is too short."""
        # Too many abbreviations (>50% of words are 1-2 chars)
        words = subject.split()
        if len(words) > 1:
            short_words = sum(
                1
                for w in words
                if len(w) <= 2 and w.lower() not in {"i", "a", "to", "of", "in", "at", "on"}
            )
            if len(words) >= 2 and short_words / len(words) > 0.5:
                return True

        # Malformed: word spacing issues (e.g., "ofmetal" instead of "of metal")
        if _CAMELCASE_RE.search(subject):
            return True

        # Must have at least 2 characters and contain a letter
        if len(subject) < 2 or not any(c.isalpha() for c in subject):
            return True

        return False

    def _is_coherent_subject(self, subject: str) -> bool:
        """Reject subjects that are vague pronouns or incomplete fragments.

        Delegates to:
        - _is_vague_pronoun(): single pronouns, pronoun+preposition fragments
        - _is_incomplete_phrase(): incomplete infinitives, bare prepositions
        - _is_malformed(): abbreviation overload, bad spacing, too short
        """
        subject_lower = subject.lower().strip()

        if self._is_vague_pronoun(subject_lower):
            return False
        if self._is_incomplete_phrase(subject_lower):
            return False
        if self._is_malformed(subject):
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
            "Quality filtering: %d \u2192 %d facts (%.1f%% kept)",
            len(facts),
            len(filtered),
            (len(filtered) / len(facts) * 100) if facts else 0,
        )
        return filtered

    def _resolve_attribution(self, facts: list[Fact], messages: list[Any]) -> None:
        """Resolve attribution for each fact in-place.

        Builds a msg_id -> is_from_me lookup from messages, then runs the
        resolver on each fact.
        """
        # Build lookup: source_message_id -> is_from_me
        msg_lookup: dict[int | None, bool] = {}
        for msg in messages:
            if isinstance(msg, dict):
                msg_id = msg.get("id")
                is_from_me = bool(msg.get("is_from_me", False))
            else:
                msg_id = getattr(msg, "id", None)
                is_from_me = bool(getattr(msg, "is_from_me", False))
            if msg_id is not None:
                msg_lookup[msg_id] = is_from_me

        for fact in facts:
            is_from_me = msg_lookup.get(fact.source_message_id, False)
            fact.attribution = self._attribution_resolver.resolve(
                source_text=fact.source_text,
                subject=fact.subject,
                is_from_me=is_from_me,
                category=fact.category,
            )

    def _clean_subject(self, subject: str) -> str:
        """Clean extracted subject: strip trailing prepositions, conjunctions, etc."""
        subject = subject.strip()

        # Remove trailing prepositions and conjunctions
        for pattern in _TRAILING_WORD_PATTERNS:
            subject = pattern.sub("", subject)

        # Remove trailing whitespace again
        return subject.strip()

    def _extract_relationships(
        self, text: str, contact_id: str, timestamp: str,
    ) -> list[Fact]:
        """Extract relationship facts from possessive patterns (e.g. 'my sister Sarah')."""
        facts: list[Fact] = []
        for match in RELATIONSHIP_PATTERN.finditer(text):
            rel_type = match.group(1).lower()
            person = match.group(2).strip()
            if rel_type in _FAMILY_RELATIONS:
                predicate = "is_family_of"
            elif "friend" in rel_type:
                predicate = "is_friend_of"
            else:
                predicate = "is_associated_with"
            facts.append(
                Fact(
                    category="relationship",
                    subject=person,
                    predicate=predicate,
                    value=rel_type,
                    source_text=text[:200],
                    confidence=0.8,
                    contact_id=contact_id,
                    extracted_at=timestamp,
                )
            )
        return facts

    def _extract_locations_and_work(
        self,
        text: str,
        contact_id: str,
        timestamp: str,
        existing_facts: list[Fact],
    ) -> list[Fact]:
        """Extract location and work facts using registry-driven patterns."""
        facts: list[Fact] = []
        # Build dedup set lazily: patterns with dedup=False run first, then the
        # dedup set is built once before dedup=True patterns execute.
        _extracted: set[tuple[str, str]] | None = None

        for rule in _RULE_PATTERNS:
            # Build dedup set on first dedup-aware pattern
            if rule.dedup and _extracted is None:
                _extracted = {(f.category, f.subject.lower()) for f in existing_facts + facts}

            for match in rule.pattern.finditer(text):
                subject = self._clean_subject(match.group(rule.subject_group))
                if not subject:
                    continue

                # Dedup check
                if rule.dedup:
                    key = (rule.category, subject.lower())
                    if key in _extracted:
                        continue

                fact_kwargs: dict[str, Any] = {
                    "category": rule.category,
                    "subject": subject,
                    "predicate": rule.predicate,
                    "source_text": text[:200],
                    "confidence": rule.confidence,
                    "contact_id": contact_id,
                    "extracted_at": timestamp,
                }
                if rule.valid_from_ts:
                    fact_kwargs["valid_from"] = timestamp
                if rule.valid_until_ts:
                    fact_kwargs["valid_until"] = timestamp

                facts.append(Fact(**fact_kwargs))

                if rule.dedup:
                    _extracted.add(key)

        return facts

    def _extract_preferences(
        self,
        text: str,
        contact_id: str,
        timestamp: str,
        existing_facts: list[Fact],
    ) -> list[Fact]:
        """Extract preference facts with filler-word filtering and sentiment detection."""
        facts: list[Fact] = []
        _extracted = {(f.category, f.subject.lower()) for f in existing_facts}

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

            # Reject captured clause fragments (not noun phrases)
            if not self._is_valid_preference_subject(thing):
                logger.debug("Skipping invalid preference subject: %s", thing[:60])
                continue

            # Coherence check (same as quality filter, applied early)
            if not self._is_coherent_subject(thing):
                logger.debug("Skipping incoherent preference subject: %s", thing[:60])
                continue

            key = ("preference", thing.lower())
            if key not in _extracted:
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
                _extracted.add(key)

        return facts

    def _extract_rule_based(self, text: str, contact_id: str, timestamp: str) -> list[Fact]:
        """Extract facts using regex patterns.

        Delegates to three extraction phases:
        1. _extract_relationships(): possessive patterns (e.g. 'my sister Sarah')
        2. _extract_locations_and_work(): registry-driven location/work patterns
        3. _extract_preferences(): preference patterns with filler-word filtering
        """
        facts = self._extract_relationships(text, contact_id, timestamp)
        facts.extend(self._extract_locations_and_work(text, contact_id, timestamp, facts))
        facts.extend(self._extract_preferences(text, contact_id, timestamp, facts))
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

                # Build dedup set for O(1) lookups instead of O(n) scans
                _ner_seen = {(f.category, f.subject.lower()) for f in facts}

                # NER-enhanced: extract entities spaCy found
                for ent in doc.ents:
                    ent_lower = ent.text.lower()
                    if ent.label_ == "PERSON" and len(ent.text) > 1:
                        if ("relationship", ent_lower) not in _ner_seen:
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
                            _ner_seen.add(("relationship", ent_lower))
                    elif ent.label_ in ("GPE", "LOC") and len(ent.text) > 1:
                        if ("location", ent_lower) not in _ner_seen:
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
                            _ner_seen.add(("location", ent_lower))
                    elif ent.label_ == "ORG" and len(ent.text) > 1:
                        if ("work", ent_lower) not in _ner_seen:
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
                            _ner_seen.add(("work", ent_lower))

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
                    linked_contact_id = self._resolve_person_to_contact(person_name)

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
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.warning("NER person extraction failed: %s", e)

        return facts

    def _get_contacts_for_resolution(self) -> list[tuple[str, str, set[str]]]:
        """Load and cache contacts with pre-tokenized names for fuzzy matching.

        Uses module-level cache with 5-minute TTL (shared across instances).
        """
        return _get_cached_contacts()

    def _resolve_person_to_contact(self, person_name: str) -> str | None:
        """Fuzzy match person name to contact in database.

        Uses cached contacts with pre-tokenized names (single DB query per session).
        Requires unique match with confidence >= 0.7 or clear winner with 0.2+ gap.

        Returns:
            Contact ID if match found, None otherwise.
        """
        try:
            contacts = self._get_contacts_for_resolution()

            if not contacts:
                return None

            # Simple token-based fuzzy match
            name_tokens = set(person_name.lower().split())
            scores: dict[str, float] = {}

            for contact_id, display_name, contact_tokens in contacts:
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

        except (ValueError, KeyError, TypeError) as e:
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
