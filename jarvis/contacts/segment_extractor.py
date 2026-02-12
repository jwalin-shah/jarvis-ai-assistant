"""Extract facts from topic-segmented conversations.

Bridges TopicSegmenter output → CandidateExtractor, merging GLiNER candidates
with spaCy NER entities from the segmenter. This is the glue between the
topic segmentation pipeline and the fact extraction pipeline.

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

# Pre-compiled emoji detection regex (Supplementary Multilingual Plane)
_EMOJI_RE = re.compile(r"[\U0001F000-\U0001FFFF]")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")

# spaCy entity label → (default fact_type, default span_label for FactCandidate)
SPACY_LABEL_MAP: dict[str, tuple[str, str]] = {
    "PERSON": ("relationship.friend", "person_name"),
    "ORG": ("work.employer", "org"),
    "GPE": ("location.current", "place"),
    "LOC": ("location.current", "place"),
}

# Broader entailment hypotheses for spaCy entities.
# These are intentionally weaker than the GLiNER templates because spaCy gives
# us entity type but no semantic context. "mentions" instead of "lives in".
SPACY_HYPOTHESIS_TEMPLATES: dict[str, str] = {
    "relationship.friend": "The message mentions a person named {span}",
    "work.employer": "The message mentions an organization called {span}",
    "location.current": "The message mentions a place called {span}",
}

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
    2. Converts spaCy entities from the segment into FactCandidates.
    3. Runs spaCy candidates through entailment gate (same as GLiNER).
    4. Deduplicates across both sources.

    Args:
        segments: TopicSegment list from TopicSegmenter.segment().
        candidate_extractor: CandidateExtractor instance (GLiNER + entailment).

    Returns:
        Deduplicated list of FactCandidate objects.
    """
    gliner_candidates: list[FactCandidate] = []
    spacy_candidates: list[FactCandidate] = []

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
                message_id=msg.message_id or 0,
                is_from_me=msg.is_from_me,
                prev_messages=prev_msgs if prev_msgs else None,
                next_messages=next_msgs if next_msgs else None,
                use_gate=False,  # Already filtered by segmenter
            )
            gliner_candidates.extend(gliner_cands)

        # Convert spaCy entities from segment into FactCandidates
        spacy_cands = _spacy_entities_to_candidates(segment)
        spacy_candidates.extend(spacy_cands)

    # Filter noisy spaCy candidates, then run through entailment
    spacy_candidates = _filter_noisy_spacy(spacy_candidates)
    if spacy_candidates and candidate_extractor._use_entailment:
        spacy_candidates = _verify_spacy_entailment(spacy_candidates)

    all_candidates = gliner_candidates + spacy_candidates

    # Attribution guard: reject facts where the subject is clearly not the speaker
    all_candidates = _filter_misattributed(all_candidates)

    return _deduplicate(all_candidates)


def _spacy_entities_to_candidates(
    segment: TopicSegment,
) -> list[FactCandidate]:
    """Convert spaCy NER entities from a topic segment into FactCandidates.

    The segmenter aggregates entities per segment as:
        segment.entities = {"PERSON": ["jake"], "ORG": ["google"], ...}

    We also check individual message entities for character offsets.

    Args:
        segment: A TopicSegment with aggregated entities.

    Returns:
        List of FactCandidate objects from spaCy entities.
    """
    from jarvis.contacts.candidate_extractor import FactCandidate

    candidates: list[FactCandidate] = []

    # Walk individual messages for precise offsets and message attribution
    for msg in segment.messages:
        if not msg.entities:
            continue
        for entity in msg.entities:
            mapping = SPACY_LABEL_MAP.get(entity.label)
            if mapping is None:
                continue

            fact_type, span_label = mapping
            span_text = entity.text.strip()

            # Skip very short or vague spans
            if len(span_text) < 2:
                continue

            candidates.append(
                FactCandidate(
                    message_id=msg.message_id or 0,
                    span_text=span_text,
                    span_label=span_label,
                    gliner_score=0.0,  # spaCy doesn't give calibrated scores
                    fact_type=fact_type,
                    start_char=entity.start,
                    end_char=entity.end,
                    source_text=msg.text,
                    is_from_me=msg.is_from_me,
                )
            )

    return candidates


# Regex for all-caps phrases (3+ chars) that are likely emphasis, not entities
_ALL_CAPS_RE = re.compile(r"^[A-Z\s!?.,']{3,}$")


def _filter_noisy_spacy(candidates: list[FactCandidate]) -> list[FactCandidate]:
    """Filter noisy spaCy NER predictions using structural heuristics.

    Only uses surface-level patterns that generalize across any chat:
    - All-caps exclamations (COUGH UP, SEND ME) → emphasis, not entities
    - Emoji in span → NER tokenization artifact
    - Repeated characters (3+) → slang elongation (CASHHHH, Whyyyy)
    - All-lowercase single-word PERSON → common words misclassified
      (real names are capitalized; spaCy depends on this)
    """
    filtered: list[FactCandidate] = []
    for c in candidates:
        span = c.span_text.strip()

        # All-caps exclamations → not real entities
        if _ALL_CAPS_RE.match(span) and c.span_label in ("org", "person_name"):
            continue

        # Emoji in span → NER artifact
        if _EMOJI_RE.search(span):
            continue

        # Repeated characters (3+) → slang not entity (e.g. "CASHHHH", "Whyyyy")
        if _REPEATED_CHAR_RE.search(span):
            continue

        # All-lowercase single-word PERSON → almost always a common word
        # Real names are capitalized (spaCy depends on this)
        if c.span_label == "person_name" and span.islower() and " " not in span:
            continue

        filtered.append(c)

    n_dropped = len(candidates) - len(filtered)
    if n_dropped:
        logger.debug("spaCy noise filter: %d -> %d candidates", len(candidates), len(filtered))
    return filtered


def _verify_spacy_entailment(
    candidates: list[FactCandidate],
) -> list[FactCandidate]:
    """Run spaCy candidates through entailment with broader hypotheses.

    Uses "The message mentions X" instead of "The user lives in X" to avoid
    rejecting valid location/person mentions that aren't about residence/friendship.
    """
    if not candidates:
        return candidates

    from jarvis.nlp.entailment import verify_entailment_batch

    pairs: list[tuple[str, str]] = []
    for c in candidates:
        template = SPACY_HYPOTHESIS_TEMPLATES.get(c.fact_type)
        if template:
            hypothesis = template.format(span=c.span_text)
        else:
            hypothesis = f"The message mentions {c.span_text}"
        pairs.append((c.source_text, hypothesis))

    results = verify_entailment_batch(pairs, threshold=0.12)

    verified: list[FactCandidate] = []
    for candidate, (is_entailed, score) in zip(candidates, results):
        if is_entailed:
            # Keep gliner_score at 0 to indicate spaCy origin in dedup
            verified.append(candidate)
        else:
            logger.debug(
                "spaCy entailment rejected: '%s' (%s) score=%.3f",
                candidate.span_text,
                candidate.fact_type,
                score,
            )

    logger.debug("spaCy entailment: %d -> %d candidates", len(candidates), len(verified))
    return verified


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

    When both GLiNER and spaCy produce the same span, prefer GLiNER
    (it has a calibrated score and went through entailment).

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
            # Prefer the one with a real GLiNER score
            seen[key] = c

    return list(seen.values())
