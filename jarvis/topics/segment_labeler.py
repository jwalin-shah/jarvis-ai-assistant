"""Segment Labeler - Generates high-quality topic labels.

Produces human-readable labels like "Trip: Austin" or "Job Search: Sarah"
from topic segments by combining:
1. TF-IDF keywords (filtered by blacklist to remove noise)
2. Entity anchors (people, places, things mentioned)
3. Scoring to pick most specific anchor

Blacklist: Common noise words that TF-IDF picks up but aren't useful topics.
Examples: "lmao", "lol", "yeah", "damn", "basically", "cuz", "thing", "time"
These are filtered out so labels reflect actual content, not speech patterns.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jarvis.topics.topic_segmenter import TopicSegment

logger = logging.getLogger(__name__)


class SegmentLabeler:
    """Refines TopicSegment labels using semantic extraction."""

    def __init__(self) -> None:
        from jarvis.topics.entity_anchor import get_tracker

        self._tracker = get_tracker()

    def label_segment(self, segment: TopicSegment) -> str:
        """Generate a human-readable topic label using the LLM."""
        if not segment.text:
            return "General Chat"

        # Strategy: Use LLM to describe the conversation in 2-4 words
        try:
            from models.loader import get_model
            loader = get_model()
            
            # Use Turn-Based formatting
            turns = []
            if segment.messages:
                curr_sender = "User" if segment.messages[0].is_from_me else "Contact"
                curr_msgs = []
                for m in segment.messages:
                    sender = "User" if m.is_from_me else "Contact"
                    if sender == curr_sender:
                        curr_msgs.append(m.text or "")
                    else:
                        if curr_msgs: turns.append(f"{curr_sender}: {' '.join(curr_msgs)}")
                        curr_sender = sender
                        curr_msgs = [m.text or ""]
                if curr_msgs: turns.append(f"{curr_sender}: {' '.join(curr_msgs)}")
            
            chat_text = "\n".join(turns)
            
            prompt = f"Chat:\n{chat_text}\n\nWhat is the main topic of this chat? (2-4 words maximum). Examples: 'Weekend Plans', 'Job Interview', 'Moving to SF'.\nTopic:"
            
            res = loader.generate_sync(
                prompt=prompt, 
                max_tokens=15, 
                temperature=0.0,
                stop_sequences=["\n", ".", "Chat:"]
            )
            
            label = res.text.strip().strip('"').strip("'")
            # Cleanup common model output junk
            label = re.sub(r"^(The topic is|Topic:)\s*", "", label, flags=re.IGNORECASE)
            
            if len(label) > 3 and len(label) < 40:
                return label.title()
        except Exception as e:
            logger.debug("LLM labeling failed: %s", e)

        # Legacy fallback logic below
        from jarvis.text_normalizer import normalize_text

        # Get first message only
        if not segment.messages:
            return "General Chat"

        first_msg = segment.messages[0]
        first_text = first_msg.text or ""

        if not first_text:
            return "General Chat"

        # Normalize but don't expand slang (we want to detect entities from clean text)
        normalized = normalize_text(first_text)

        # Capitalize first letter - spaCy needs this to detect named entities
        if normalized:
            normalized = normalized[0].upper() + normalized[1:]

        # Extract entities WITH TYPES using spaCy
        doc = self._tracker.nlp(normalized)

        # Priority: PERSON > GPE (places) > ORG > other
        entity_priority = {"PERSON": 1, "GPE": 2, "ORG": 3, "LOC": 4, "FAC": 5}

        # Collect entities by type
        entities_by_type: dict[str, list[str]] = {}
        for ent in doc.ents:
            label = ent.label_
            if label in entity_priority:
                entities_by_type.setdefault(label, []).append(ent.text.title())

        # Return highest priority entity
        for label in ["PERSON", "GPE", "ORG", "LOC", "FAC"]:
            if entities_by_type.get(label):
                return entities_by_type[label][0]

        return "General Chat"
    def summarize_segment(self, segment: TopicSegment) -> str:
        """Generate a 1-sentence summary of the segment text."""
        if not segment.text or len(segment.text.split()) < 5:
            return "Short exchange."

        try:
            from models.loader import get_model
            loader = get_model()
            
            # Use same turn-based logic
            turns = []
            if segment.messages:
                curr_sender = "User" if segment.messages[0].is_from_me else "Contact"
                curr_msgs = []
                for m in segment.messages:
                    sender = "User" if m.is_from_me else "Contact"
                    if sender == curr_sender:
                        curr_msgs.append(m.text or "")
                    else:
                        if curr_msgs: turns.append(f"{curr_sender}: {' '.join(curr_msgs)}")
                        curr_sender = sender
                        curr_msgs = [m.text or ""]
                if curr_msgs: turns.append(f"{curr_sender}: {' '.join(curr_msgs)}")
            
            chat_text = "\n".join(turns)
            
            prompt = f"Chat:\n{chat_text}\n\nSummarize this chat in exactly one sentence:\nSummary:"
            
            res = loader.generate_sync(
                prompt=prompt, 
                max_tokens=60, 
                temperature=0.0,
                stop_sequences=["\n", "Chat:"]
            )
            
            summary = res.text.strip().strip('"').strip("'")
            if len(summary) > 10:
                return summary
        except Exception as e:
            logger.debug("LLM summarization failed: %s", e)

        # Fallback
        label = self.label_segment(segment)
        return f"{label} (Keywords: {', '.join(segment.keywords[:3])})"


_labeler: SegmentLabeler | None = None


def get_labeler() -> SegmentLabeler:
    global _labeler
    if _labeler is None:
        _labeler = SegmentLabeler()
    return _labeler


def reset_labeler() -> None:
    global _labeler
    _labeler = None
