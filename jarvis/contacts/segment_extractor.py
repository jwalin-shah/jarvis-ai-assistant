"""Extract facts from topic-segmented conversations.

Bridges TopicSegmenter output → CandidateExtractor using GLiNER.
This is the glue between the topic segmentation pipeline and the
fact extraction pipeline.

Usage:
    from jarvis.contacts.segment_extractor import extract_facts_from_segments
    from jarvis.topics.topic_segmenter import TopicSegmenter

    segmenter = TopicSegmenter(normalization_task="extraction")
    segments = segmenter.segment(messages)
    candidates = extract_facts_from_segments(segments, extractor)
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.contacts.candidate_extractor import CandidateExtractor, FactCandidate
    from jarvis.topics.topic_segmenter import TopicSegment

logger = logging.getLogger(__name__)

# Patterns indicating the subject is NOT the speaker (attribution guard)
_OTHERS_SUBJECT_RE = re.compile(
    r"\bmy\s+(friends?|buddys?|pals?|homies?|brothers?|sisters?|moms?|dads?|"
    r"mothers?|fathers?|aunts?|uncles?|cousins?|wife|husband|girlfriends?|"
    r"boyfriends?|partners?|coworkers?|roommates?|boss)\b"
    r".+\b(is|was|works?|became|got)\b",
    re.IGNORECASE,
)


def extract_facts_from_segments(
    segments: list[TopicSegment],
    candidate_extractor: CandidateExtractor,
) -> list[FactCandidate]:
    """Extract facts from pre-segmented, pre-normalized conversation data.

    For each segment:
    1. Runs GLiNER on each message's normalized text (already slang-expanded).
       GLiNER candidates go through entailment inside extract_candidates().
    2. Deduplicates candidates.

    Args:
        segments: TopicSegment list from TopicSegmenter.segment().
        candidate_extractor: CandidateExtractor instance (GLiNER + entailment).

    Returns:
        Deduplicated list of FactCandidate objects.
    """
    gliner_candidates: list[FactCandidate] = []

    for segment in segments:
        # Collect context texts for the segment (for GLiNER context window)
        segment_texts = [m.text for m in segment.messages if m.text]

        for idx, msg in enumerate(segment.messages):
            if not msg.text:
                continue

            # Build prev/next context from within the segment
            prev_msgs = segment_texts[max(0, idx - 2) : idx]
            next_msgs = segment_texts[idx + 1 : idx + 3]

            # Run GLiNER on normalized text (entailment runs inside)
            gliner_cands = candidate_extractor.extract_candidates(
                text=msg.text,
                message_id=getattr(msg, "message_id", None) or getattr(msg, "id", 0) or 0,
                is_from_me=msg.is_from_me,
                prev_messages=prev_msgs if prev_msgs else None,
                next_messages=next_msgs if next_msgs else None,
                use_gate=False,  # Already filtered by segmenter
            )
            gliner_candidates.extend(gliner_cands)

    # Attribution guard: reject facts where the subject is clearly not the speaker
    all_candidates = _filter_misattributed(gliner_candidates)

    return _deduplicate(all_candidates)


def _filter_misattributed(candidates: list[FactCandidate]) -> list[FactCandidate]:
    """Reject facts where the sentence subject is clearly not the speaker.

    Catches "my friend is a founder" being attributed as the user's job title.
    Only filters work.job_title and work.employer fact types since those are
    most prone to this error.
    """
    filtered: list[FactCandidate] = []
    for c in candidates:
        if c.fact_type in ("work.job_title", "work.employer") and c.source_text:
            if _OTHERS_SUBJECT_RE.search(c.source_text):
                logger.debug(
                    "Attribution rejected: '%s' (%s) - subject is not speaker",
                    c.span_text,
                    c.fact_type,
                )
                continue
        filtered.append(c)
    return filtered


def _deduplicate(candidates: list[FactCandidate]) -> list[FactCandidate]:
    """Deduplicate candidates by (message_id, span_text_lower, fact_type).

    Args:
        candidates: Raw candidate list (may have duplicates).

    Returns:
        Deduplicated list.
    """
    seen: dict[tuple[int, str, str], FactCandidate] = {}

    for c in candidates:
        key = (c.message_id, c.span_text.casefold(), c.fact_type)
        existing = seen.get(key)
        if existing is None:
            seen[key] = c
        elif c.gliner_score > existing.gliner_score:
            # Prefer the one with a higher GLiNER score
            seen[key] = c

    return list(seen.values())
