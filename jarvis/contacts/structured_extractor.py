"""Structured Fact Extractor - Produces proper (subject, predicate, object) triples.

Replaces the free-text extraction with structured triple extraction:
- Subject: Who the fact is about (contact name, "I", or entity)
- Predicate: Relationship type (lives_in, works_at, likes, etc.)
- Object: The value (Austin, Google, pizza, etc.)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from jarvis.contacts.contact_profile import Fact

logger = logging.getLogger(__name__)


# Common predicate patterns mapped to canonical forms
PREDICATE_PATTERNS = {
    # Location
    "lives_in": [r"live[s]? in", r"living in", r"from\s+\w", r"based in", r"located in"],
    "lived_in": [r"lived in", r"used to live", r"moved from", r"grew up in"],
    "moving_to": [r"moving to", r"moving in", r"relocating to", r"planning to move"],
    "from": [r"from\s+\w", r"originally from", r"hometown"],
    # Work
    "works_at": [r"work[s]? at", r"working at", r"works for", r"employed at", r"job at"],
    "worked_at": [r"worked at", r"used to work", r"previously worked", r"former"],
    "job_title": [r"is a[n]?\s+\w", r"works as", r"position as", r"role as"],
    "studies_at": [r"stud[iesy]{2,3} at", r"student at", r"attending", r"goes to"],
    # Preferences
    "likes": [r"like[s]?", r"enjoy[s]?", r"love[s]?", r"favorite", r"prefer[s]?"],
    "dislikes": [r"hate[s]?", r"dislike[s]?", r"can't stand", r"don't like"],
    "allergic_to": [r"allergic to"],
    # Health
    "has_condition": [r"has\s+\w", r"diagnosed with", r"suffers from", r"condition"],
    "takes_medication": [r"takes?\s+\w", r"medication", r"prescribed"],
    # Relationships
    "is_family_of": [r"(sister|brother|mom|dad|mother|father|cousin|aunt|uncle) of", r"related to"],
    "is_friend_of": [r"friend[s]? with", r"friends with", r"knows"],
    "is_partner_of": [r"dating", r"married to", r"partner of", r"boyfriend", r"girlfriend"],
    # Personal
    "birthday_is": [r"birthday", r"born on", r"turns?\s+\d"],
    "age_is": [r"age is", r"\d+ years old", r"turning\s+\d"],
    "has_pet": [r"has (a |an )?(dog|cat|pet|puppy|kitten)"],
}

# Category mapping from predicates
PREDICATE_CATEGORIES = {
    "lives_in": "location",
    "lived_in": "location",
    "moving_to": "location",
    "from": "location",
    "works_at": "work",
    "worked_at": "work",
    "job_title": "work",
    "studies_at": "education",
    "likes": "preference",
    "dislikes": "preference",
    "allergic_to": "health",
    "has_condition": "health",
    "takes_medication": "health",
    "is_family_of": "relationship",
    "is_friend_of": "relationship",
    "is_partner_of": "relationship",
    "birthday_is": "personal",
    "age_is": "personal",
    "has_pet": "personal",
}


def extract_predicate_from_text(text: str) -> tuple[str | None, str | None]:
    """Extract predicate and object from free-text fact.

    Args:
        text: Free-text fact like "lives in Austin and works at Google"

    Returns:
        Tuple of (predicate, object) or None if no match
    """
    text_lower = text.lower()

    for predicate, patterns in PREDICATE_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern + r"[\s:]+([^,.;]+)", text_lower)
            if match:
                object_value = match.group(1).strip()
                # Clean up object value
                object_value = re.sub(r"^(is|are|was|were)\s+", "", object_value)
                object_value = object_value.strip()
                if len(object_value) > 1:
                    return predicate, object_value

    return None, None  # No predicate found


def parse_free_text_fact(fact_text: str, subject_hint: str | None = None) -> list[dict[str, Any]]:
    """Parse free-text fact into structured triples.

    Handles complex facts like "lives in Austin and works at Google"
    and splits into multiple structured facts.

    Args:
        fact_text: Raw fact text from LLM extraction
        subject_hint: Expected subject name

    Returns:
        List of dicts with keys: subject, predicate, object, category
    """
    results = []

    # Split on conjunctions for compound facts
    # "lives in Austin and works at Google" -> two facts
    sentences = re.split(r"\s+(?:and|but|also)\s+|\s*[;,]\s*", fact_text)

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 3:
            continue

        # Try to extract structured triple
        predicate, obj = extract_predicate_from_text(sentence)

        if predicate:
            category = PREDICATE_CATEGORIES.get(predicate, "other")
            results.append(
                {
                    "subject": subject_hint or "Contact",
                    "predicate": predicate,
                    "object": obj or "",
                    "category": category,
                    "original": sentence,
                }
            )
        else:
            # Couldn't parse - store as "other" with note
            results.append(
                {
                    "subject": subject_hint or "Contact",
                    "predicate": "note",
                    "object": sentence,
                    "category": "other",
                    "original": sentence,
                }
            )

    return results


def restructure_existing_fact(fact: Fact) -> list[Fact]:
    """Convert an existing free-text Fact into structured Fact(s).

    Args:
        fact: Existing Fact with subject, value (free text), empty predicate

    Returns:
        List of new structured Facts (may be multiple if compound)
    """
    # Parse the free-text value
    parsed = parse_free_text_fact(fact.value, fact.subject)

    results = []
    for p in parsed:
        results.append(
            Fact(
                category=p["category"],
                subject=p["subject"],
                predicate=p["predicate"],
                value=p["object"],
                source_text=fact.source_text,
                confidence=fact.confidence,
                contact_id=fact.contact_id,
                source_message_id=fact.source_message_id,
                extracted_at=fact.extracted_at,
                attribution=fact.attribution,
            )
        )

    return results


def enhance_extractor_output(
    facts: list[Fact],
    contact_name: str,
    user_name: str,
) -> list[Fact]:
    """Post-process LLM-extracted facts to add structure.

    Takes free-text facts from instruction_extractor and:
    1. Parses compound facts ("lives in X and works at Y")
    2. Extracts proper predicates
    3. Cleans subject names
    4. Deduplicates within batch

    Args:
        facts: Raw facts from LLM extraction
        contact_name: Name of the contact
        user_name: Your name (Jwalin)

    Returns:
        Enhanced structured facts
    """
    enhanced = []
    seen = set()

    for fact in facts:
        # Fix subject names
        subject = fact.subject
        if subject.lower() in ["contact", "them", "they"]:
            subject = contact_name
        elif subject.lower() in ["me", "i", "myself"]:
            subject = user_name
        elif subject.lower() in ["user"]:
            subject = user_name

        # Parse the free-text value
        parsed = parse_free_text_fact(fact.value, subject)

        for p in parsed:
            # Create new structured fact
            new_fact = Fact(
                category=p["category"],
                subject=p["subject"],
                predicate=p["predicate"],
                value=p["object"],
                source_text=fact.source_text,
                confidence=fact.confidence,
                contact_id=fact.contact_id,
                source_message_id=fact.source_message_id,
                extracted_at=fact.extracted_at,
                attribution=fact.attribution,
            )

            # Deduplicate by key fields
            key = (
                new_fact.contact_id,
                new_fact.category,
                new_fact.subject,
                new_fact.predicate,
                new_fact.value,
            )
            if key not in seen:
                seen.add(key)
                enhanced.append(new_fact)

    return enhanced
