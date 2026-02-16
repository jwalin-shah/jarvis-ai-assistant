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
    hypothesis: str | list[str],
    threshold: float = 0.6,
) -> tuple[bool, float]:
    """Check if hypothesis is entailed by premise using NLI cross-encoder.

    If multiple hypotheses are provided, returns True if ANY are entailed.

    Args:
        premise: The source text (message).
        hypothesis: The fact statement(s) to verify.
        threshold: Minimum entailment probability to accept.

    Returns:
        (is_entailed, entailment_score) tuple.
    """
    from models.nli_cross_encoder import get_nli_cross_encoder

    nli = get_nli_cross_encoder()

    hypotheses = [hypothesis] if isinstance(hypothesis, str) else hypothesis
    pairs = [(premise, h) for h in hypotheses]

    if not pairs:
        return False, 0.0

    all_scores = nli.predict_batch(pairs)
    max_score = max(s["entailment"] for s in all_scores)

    return max_score > threshold, max_score


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


def fact_to_hypothesis(category: str, subject: str, predicate: str, value: str = "") -> list[str]:
    """Convert a structured fact to a natural language hypothesis for NLI.

    Uses varied templates (casual/formal) to improve recall against chat premises.
    """
    p_clean = predicate.lower().replace("_", " ")

    if category == "relationship":
        if value:
            # Varied templates to match different chat styles
            return [
                f"{subject} is my {value}",
                f"{subject} is the user's {value}",
                f"{subject} is my {value}",
            ]
        return [f"{subject} and I are related", f"{subject} is related to the user"]

    elif category == "location":
        return [
            f"I {p_clean} in {subject}",
            f"The person {p_clean} in {subject}",
            f"I live in {subject}",
        ]

    elif category == "work":
        return [
            f"I work at {subject}",
            f"The person works at {subject}",
            f"My job is at {subject}",
        ]

    elif category == "preference":
        return [
            f"I {p_clean} {subject}",
            f"The person {p_clean} {subject}",
            f"I really {p_clean} {subject}",
        ]

    else:
        return [
            f"{subject} {p_clean}",
            f"{subject} {p_clean} {value}",
            f"I mentioned that {subject} {p_clean}",
        ]
