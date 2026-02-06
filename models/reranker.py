"""Reranker service - thin layer over cross-encoder for retrieval reranking.

Scores candidate documents against a query using a cross-encoder model,
then returns the top-k candidates sorted by relevance.

Usage:
    from models.reranker import get_reranker

    reranker = get_reranker()
    reranked = reranker.rerank("what time is dinner?", candidates, top_k=3)
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks retrieval candidates using a cross-encoder.

    Lazy-loads the cross-encoder on first call to avoid startup cost
    when reranking is disabled.
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._cross_encoder = None

    def _get_cross_encoder(self):
        """Lazy-load the cross-encoder singleton."""
        if self._cross_encoder is None:
            from models.cross_encoder import get_cross_encoder

            self._cross_encoder = get_cross_encoder(self._model_name)
        return self._cross_encoder

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        text_key: str = "trigger_text",
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Rerank candidates by cross-encoder relevance to query.

        Args:
            query: The query text to score against.
            candidates: List of candidate dicts from vec_search.
            text_key: Key in candidate dicts containing the text to score.
            top_k: Number of top candidates to return.

        Returns:
            Top-k candidates sorted by rerank_score (descending),
            each augmented with a 'rerank_score' field.
        """
        if not candidates:
            return []

        if len(candidates) <= 1:
            for c in candidates:
                c["rerank_score"] = 1.0
            return candidates

        # Build (query, doc) pairs
        pairs = []
        valid_indices = []
        for i, cand in enumerate(candidates):
            text = cand.get(text_key, "")
            if text:
                pairs.append((query, text))
                valid_indices.append(i)

        if not pairs:
            return candidates[:top_k]

        ce = self._get_cross_encoder()
        scores = ce.predict(pairs)

        # Assign scores back to candidates
        scored = []
        score_idx = 0
        for i, cand in enumerate(candidates):
            if i in valid_indices:
                cand["rerank_score"] = float(scores[score_idx])
                score_idx += 1
            else:
                cand["rerank_score"] = 0.0
            scored.append(cand)

        # Sort by rerank_score descending
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        return scored[:top_k]


# =============================================================================
# Singleton
# =============================================================================

_reranker: CrossEncoderReranker | None = None
_reranker_lock = threading.Lock()


def get_reranker(model_name: str = "ms-marco-MiniLM-L-6-v2") -> CrossEncoderReranker:
    """Get or create the singleton CrossEncoderReranker."""
    global _reranker

    if _reranker is not None:
        return _reranker

    with _reranker_lock:
        if _reranker is None:
            _reranker = CrossEncoderReranker(model_name=model_name)
        return _reranker


def reset_reranker() -> None:
    """Reset the singleton for testing."""
    global _reranker

    with _reranker_lock:
        _reranker = None
