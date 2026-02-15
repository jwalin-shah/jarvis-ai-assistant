"""Adapter for InstructionFactExtractor to work with the bakeoff evaluation system.

Maps structured Facts (Subject, Predicate, Value) to Candidates (span_text, label).
"""

from __future__ import annotations

import logging
from typing import Any

from jarvis.contacts.extractors.base import ExtractedCandidate, ExtractorAdapter, register_extractor

from jarvis.contacts.instruction_extractor import get_instruction_extractor
from jarvis.topics.topic_segmenter import TopicSegment

logger = logging.getLogger(__name__)

# Map Instruction predicates to entity labels for evaluation compatibility
PREDICATE_TO_LABEL = {
    "lives_in": "place",
    "lived_in": "place",
    "moving_to": "place",
    "from": "place",
    "located_at": "place",
    "works_at": "work",
    "worked_at": "work",
    "job_title": "occupation",
    "is_family_of": "family_member",
    "has_relative": "family_member",
    "is_friend_of": "person_name",  # Close enough
    "is_partner_of": "partner_name",
    "likes_food": "food_like",
    "dislikes_food": "food_dislike",
    "enjoys": "activity",
    "allergic_to": "allergy",
    "has_condition": "health_condition",
    "birthday_is": "birthday",
    "attends": "school",
    "has_pet": "pet_name",
    # Fallbacks
    "has_fact": "general_fact",
}


class InstructionExtractorAdapter(ExtractorAdapter):
    """Adapter for V5 InstructionFactExtractor."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__("instruction", config)
        self.model_tier = config.get("model_tier", "1.2b")
        self.extractor = get_instruction_extractor(tier=self.model_tier)

    def extract_from_text(
        self, text: str, message_id: int, is_from_me: bool = False, **kwargs
    ) -> list[ExtractedCandidate]:
        """Extract facts from a single message (treated as a 1-msg segment)."""
        if not text:
            return []

        # Mock a message object
        class MockMessage:
            def __init__(self, text, is_from_me):
                self.text = text
                self.is_from_me = is_from_me
                self.sender_name = "User" if is_from_me else "Contact"
                self.id = message_id

        # Wrap in a segment
        segment = TopicSegment(
            chat_id="eval_chat",
            contact_id="eval_contact",
            messages=[MockMessage(text, is_from_me)],
            start_time=0,
            end_time=0,
            message_count=1,
            segment_id=f"eval_{message_id}",
            text=text,
        )

        # Extract
        try:
            facts = self.extractor.extract_facts_from_segment(
                segment, contact_id="eval_contact", contact_name="Contact", user_name="Me"
            )
        except Exception as e:
            logger.error(f"Instruction extraction failed: {e}")
            return []

        # Convert Facts to ExtractedCandidates
        candidates = []
        for fact in facts:
            # Map predicate to entity label
            # Try exact match, then substring match
            label = PREDICATE_TO_LABEL.get(fact.predicate, "unknown")

            # Simple heuristic mapping if direct lookup fails
            if label == "unknown":
                if "loc" in fact.predicate or "place" in fact.predicate:
                    label = "place"
                elif "work" in fact.predicate or "job" in fact.predicate:
                    label = "work"
                elif "like" in fact.predicate or "love" in fact.predicate:
                    label = "activity"

            # Find offsets (simple substring search)
            # Note: Instruction model doesn't return offsets, so we guess the first occurrence
            start_char = text.find(fact.value)
            end_char = start_char + len(fact.value) if start_char != -1 else -1

            candidates.append(
                ExtractedCandidate(
                    span_text=fact.value,
                    span_label=label,
                    score=fact.confidence,
                    start_char=start_char,
                    end_char=end_char,
                    fact_type=f"{fact.category}.{fact.predicate}",
                    extractor_metadata={
                        "subject": fact.subject,
                        "predicate": fact.predicate,
                        "attribution": fact.attribution,
                    },
                )
            )

        return candidates


# Register
register_extractor("instruction", InstructionExtractorAdapter)
