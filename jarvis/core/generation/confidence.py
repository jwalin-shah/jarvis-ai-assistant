from __future__ import annotations

from typing import Any

from jarvis.classifiers.response_mobilization import ResponsePressure

# Responses that signal the model is uncertain / lacks context
UNCERTAIN_SIGNALS = frozenset({"?", "??", "hm?", "what?", "huh?"})


def compute_confidence(
    pressure: ResponsePressure,
    rag_similarity: float,
    example_diversity: float,
    reply_length: int,
    reply_text: str,
    incoming_text: str = "",
    rerank_score: float | None = None,
) -> tuple[float, str]:
    """Compute confidence level based on multiple signals.

    Args:
        pressure: Response mobilization pressure level.
        rag_similarity: Top RAG result similarity score (0-1).
        example_diversity: Measure of example diversity (0-1).
        reply_length: Number of words in the reply.
        reply_text: The actual reply text for uncertain signal detection.
        incoming_text: Original incoming message for coherence check.
        rerank_score: Cross-encoder rerank score from top result (0-1).

    Returns:
        Tuple of (numeric_confidence 0-1, discrete label "high"/"medium"/"low").
    """
    # Base confidence by pressure
    base_confidence = {
        ResponsePressure.HIGH: 0.85,
        ResponsePressure.MEDIUM: 0.65,
        ResponsePressure.LOW: 0.45,
        ResponsePressure.NONE: 0.30,
    }[pressure]

    # Adjust based on RAG quality
    if rag_similarity < 0.5:
        base_confidence *= 0.8

    # Boost from cross-encoder reranking (more reliable than embedding sim)
    if rerank_score is not None and rerank_score > 0.7:
        base_confidence = min(base_confidence * 1.1, 0.95)

    # Adjust based on example diversity
    if example_diversity < 0.3:  # All from same contact
        base_confidence *= 0.9

    # Uncertain signals only matter if very short reply + high pressure
    if (
        reply_length < 3
        and pressure == ResponsePressure.HIGH
        and reply_text.lower() in UNCERTAIN_SIGNALS
    ):
        base_confidence *= 0.7

    # Coherence penalty: reply that parrots the input is low quality
    if incoming_text and reply_text:
        reply_lower = reply_text.lower().strip()
        incoming_lower = incoming_text.lower().strip()
        if reply_lower == incoming_lower or (
            len(reply_lower) > 5 and incoming_lower.startswith(reply_lower)
        ):
            base_confidence *= 0.5

    # Clamp to [0, 1]
    base_confidence = max(0.0, min(base_confidence, 1.0))

    # Map float to discrete level
    if base_confidence >= 0.7:
        label = "high"
    elif base_confidence >= 0.45:
        label = "medium"
    else:
        label = "low"

    return base_confidence, label


def compute_example_diversity(search_results: list[dict[str, Any]]) -> float:
    """Compute diversity of search results by unique contacts/contexts.

    Args:
        search_results: List of search result dicts.

    Returns:
        Diversity score from 0.0 (all same) to 1.0 (all unique).
    """
    if not search_results:
        return 0.0

    # Count unique trigger texts as proxy for diversity
    unique_triggers = len(set(r.get("trigger_text", "") for r in search_results))
    return min(unique_triggers / len(search_results), 1.0)
