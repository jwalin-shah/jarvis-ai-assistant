"""Attribution resolver for fact extraction.

Determines WHO a fact is about: the contact, the user, or a third party.
Rule-based, zero ML, zero memory footprint.
"""

from __future__ import annotations

import re

# First-person pronouns (speaker talking about themselves)
_FIRST_PERSON = re.compile(r"\b(I|I'm|I've|I'll|I'd|my|me|mine|myself)\b", re.IGNORECASE)

# Third-person pronouns (speaker talking about someone else)
_THIRD_PERSON = re.compile(r"\b(he|she|they|his|her|their|him|them)\b", re.IGNORECASE)

# Possessive relationship pattern: "my sister Sarah", "my friend John"
_RELATION_PATTERN = re.compile(
    r"\b[Mm]y\s+(?:sister|brother|mom|mother|dad|father|wife|husband|"
    r"girlfriend|boyfriend|partner|daughter|son|cousin|aunt|uncle|"
    r"[Gg]randma|[Gg]randmother|[Gg]randpa|[Gg]randfather|friend|best friend|"
    r"roommate|fiancée?|boss|coworker|colleague|neighbor)\s+",
)


class AttributionResolver:
    """Resolve who a fact is about based on message context.

    Attribution values:
    - "contact": the contact is talking about themselves (default)
    - "user": the user (is_from_me=True) is talking about themselves
    - "third_party": either party is talking about someone else
    """

    def resolve(
        self,
        *,
        source_text: str,
        subject: str,
        is_from_me: bool,
        category: str = "",
    ) -> str:
        """Determine attribution for a single fact.

        Args:
            source_text: The original message text.
            subject: The extracted fact subject (e.g., "Sarah", "Austin").
            is_from_me: Whether the message was sent by the user.
            category: Fact category (relationship, location, etc.).

        Returns:
            "contact", "user", or "third_party".
        """
        if not source_text:
            return "contact"

        # 1. Relationship + possessive pattern ("my sister Sarah") -> third_party
        if _RELATION_PATTERN.search(source_text):
            return "third_party"

        # 2. Third-person pronoun near subject -> third_party
        #    Only if the subject is a person name (relationship category)
        if category == "relationship" and _THIRD_PERSON.search(source_text):
            return "third_party"

        # 3. is_from_me=True -> user talking about themselves
        if is_from_me:
            return "user"

        # 4. is_from_me=False -> contact talking about themselves
        return "contact"
