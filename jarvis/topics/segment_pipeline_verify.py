"""Verification stage helpers for the segment fact-extraction pipeline."""

from __future__ import annotations

from typing import Any

from jarvis.topics.segment_pipeline_collect import FactCandidate


def verify_fact_candidates(
    candidates: list[FactCandidate],
    *,
    threshold: float = 0.05,
) -> tuple[list[Any], int]:
    """Run NLI verification and attach segment ids to accepted facts."""
    if not candidates:
        return [], 0

    from jarvis.contacts.fact_verifier import FactVerifier

    verifier = FactVerifier(threshold=threshold)
    verification_input = [(candidate.fact, candidate.segment_text) for candidate in candidates]
    verified_facts, rejected_count = verifier.verify_facts_batch(verification_input)

    fact_to_db_id = {id(candidate.fact): candidate.segment_db_id for candidate in candidates}
    for fact in verified_facts:
        segment_db_id = fact_to_db_id.get(id(fact))
        if segment_db_id is not None:
            setattr(fact, "_segment_db_id", segment_db_id)

    return verified_facts, rejected_count
