"""Collection helpers for the segment fact-extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jarvis.topics.topic_segmenter import TopicSegment


_LOW_INFO_VALUES = {
    "me",
    "you",
    "that",
    "this",
    "it",
    "they",
    "them",
    "him",
    "her",
}


@dataclass(frozen=True)
class FactCandidate:
    """Single fact candidate with segment context for NLI verification."""

    fact: Any
    segment_text: str
    segment_db_id: int


def should_verify_fact_value(value: str) -> bool:
    """Fast gate before NLI verification to avoid low-signal checks."""
    normalized = " ".join((value or "").strip().lower().split())
    if not normalized or normalized in _LOW_INFO_VALUES:
        return False
    if len(normalized) < 3:
        return False
    # Skip values with no letters (often timestamps/noise).
    if not any(ch.isalpha() for ch in normalized):
        return False
    return True


def segment_fingerprint(segment: TopicSegment) -> str:
    """Stable fingerprint used for deduplicating fact extraction work."""
    text = get_segment_text(segment)
    normalized = " ".join(text.lower().split())
    return sha1(normalized.encode("utf-8")).hexdigest()


def get_segment_text(segment: TopicSegment) -> str:
    """Return segment text, rebuilding from messages when precomputed text is empty."""
    text = getattr(segment, "text", "") or ""
    if text:
        return text
    return "\n".join((m.text or "") for m in getattr(segment, "messages", []))


def collect_fact_candidates(
    batch_segments: list[TopicSegment],
    batch_results: list[list[Any]],
    batch_db_ids: list[int],
) -> list[FactCandidate]:
    """Build fact candidates from batch extraction output."""
    candidates: list[FactCandidate] = []
    for segment_obj, segment_facts, segment_db_id in zip(
        batch_segments, batch_results, batch_db_ids
    ):
        if not segment_facts:
            continue
        fast_gated_facts = [
            fact
            for fact in segment_facts
            if should_verify_fact_value(getattr(fact, "value", ""))
        ]
        if not fast_gated_facts:
            continue
        segment_text = get_segment_text(segment_obj)
        for fact in fast_gated_facts:
            candidates.append(
                FactCandidate(
                    fact=fact,
                    segment_text=segment_text,
                    segment_db_id=segment_db_id,
                )
            )
    return candidates
