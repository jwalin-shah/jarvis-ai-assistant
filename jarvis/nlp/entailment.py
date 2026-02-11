"""Entailment verification for fact extraction using pure MLX NLI cross-encoder.

Uses cross-encoder/nli-deberta-v3-xsmall implemented in MLX (~22M params,
87.77% MNLI accuracy). Scores (premise, hypothesis) pairs for
contradiction/entailment/neutral.

All GPU operations go through MLXModelLoader._mlx_load_lock for thread safety.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def verify_entailment(
    premise: str,
    hypothesis: str,
    threshold: float = 0.6,
) -> tuple[bool, float]:
    """Check if hypothesis is entailed by premise using NLI cross-encoder.

    Args:
        premise: The source text (message).
        hypothesis: The fact statement to verify.
        threshold: Minimum entailment probability to accept.

    Returns:
        (is_entailed, entailment_score) tuple.
    """
    from models.nli_cross_encoder import get_nli_cross_encoder

    nli = get_nli_cross_encoder()
    scores = nli.predict_entailment(premise, hypothesis)
    entailment_score = scores["entailment"]
    return entailment_score > threshold, entailment_score


def verify_entailment_batch(
    pairs: list[tuple[str, str]],
    threshold: float = 0.6,
) -> list[tuple[bool, float]]:
    """Batch entailment verification for efficiency.

    Args:
        pairs: List of (premise, hypothesis) tuples.
        threshold: Minimum entailment probability.

    Returns:
        List of (is_entailed, score) tuples.
    """
    if not pairs:
        return []

    from models.nli_cross_encoder import get_nli_cross_encoder

    nli = get_nli_cross_encoder()
    all_scores = nli.predict_batch(pairs)

    return [(scores["entailment"] > threshold, scores["entailment"]) for scores in all_scores]


def fact_to_hypothesis(category: str, subject: str, predicate: str, value: str = "") -> str:
    """Convert a structured fact to a natural language hypothesis for NLI.

    Examples:
        ("relationship", "Sarah", "is_family_of", "sister") -> "Sarah is the user's sister"
        ("location", "Austin", "lives_in", "") -> "The person lives in Austin"
        ("work", "Google", "works_at", "") -> "The person works at Google"
    """
    if category == "relationship":
        if value:
            return f"{subject} is the user's {value}"
        return f"{subject} is related to the user"
    elif category == "location":
        verb = predicate.replace("_", " ")
        return f"The person {verb} {subject}"
    elif category == "work":
        return f"The person works at {subject}"
    elif category == "preference":
        verb = predicate.replace("_", " ")
        return f"The person {verb} {subject}"
    else:
        verb = predicate.replace("_", " ")
        return f"{subject} {verb}"
