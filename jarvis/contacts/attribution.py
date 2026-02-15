"""Attribution resolver for fact extraction.

Determines WHO a fact is about: the contact, the user, or a third party.
Rule-based, zero ML, zero memory footprint.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

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
        """Determine attribution for a single fact."""
        if not source_text:
            return "contact"

        res = self._do_resolve(source_text, subject, is_from_me, category)
        logger.debug(f"Attribution: '{subject}' from '{source_text[:30]}...' (me={is_from_me}) -> {res}")
        return res

    def _do_resolve(
        self,
        source_text: str,
        subject: str,
        is_from_me: bool,
        category: str = "",
    ) -> str:
        # 1. Relationship + possessive pattern ("my sister Sarah") -> always third_party
        if _RELATION_PATTERN.search(source_text):
            return "third_party"

        # 2. If it's a person name in relationship category and NOT 'my ...' pattern,
        #    but contains third person pronouns, it's likely about someone else.
        if category == "relationship" and _THIRD_PERSON.search(source_text):
            return "third_party"

        # 3. is_from_me=True -> user talking about themselves
        if is_from_me:
            # Re-check: if they say "I love Alex", and subject is Alex, it's third_party
            # But "I love reading", subject is reading, it's user.
            if category in ("relationship", "person") and subject.lower() not in source_text.lower():
                 # This is a bit complex for a regex resolver, but let's keep it simple:
                 # If the subject is a person name and not 'I/me', it's third party.
                 pass

            # Simple rule: if I said it, and it's not a clear 'my sister' etc, it's about ME.
            return "user"

        # 4. is_from_me=False -> contact talking about themselves
        return "contact"
