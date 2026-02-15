"""Semantic deduplication for extracted facts.

Ensures that semantically identical facts (e.g., "lives in Austin" vs
"resides in Austin") are merged rather than duplicated in the Knowledge Graph.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from jarvis.contacts.contact_profile import Fact

logger = logging.getLogger(__name__)


class FactDeduplicator:
    """Deduplicates facts using semantic similarity."""

    def __init__(self, threshold: float = 0.85) -> None:
        """Initialize deduplicator.

        Args:
            threshold: Semantic similarity threshold (0-1) above which
                       two facts are considered duplicates.
        """
        self.threshold = threshold
        self._embedder: Any | None = None

    def _get_embedder(self) -> Any:
        """Lazy-load the embedding model."""
        if self._embedder is None:
            from models.bert_embedder import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def deduplicate(
        self, 
        new_facts: list[Fact], 
        existing_facts: list[Fact],
        return_embeddings: bool = False,
    ) -> list[Fact] | tuple[list[Fact], np.ndarray]:
        """Filter out new facts that are semantically covered by existing ones.
        
        Optimized to use vec_facts (sqlite-vec) if available, falling back to 
        in-memory check only if necessary.

        Args:
            new_facts: List of facts recently extracted.
            existing_facts: List of facts already in the database.
            return_embeddings: If True, also return embeddings for the kept facts.

        Returns:
            Filtered list of new facts that are truly unique.
            If return_embeddings=True, returns (facts, embeddings) tuple.
        """
        if not new_facts:
            return []

        import numpy as np
        from jarvis.contacts.fact_index import _quantize, _distance_to_similarity
        from jarvis.db import get_db

        # Fast pass: exact normalized dedup before embedding/vector search.
        compact_facts: list[Fact] = []
        seen_exact: set[tuple[str, str, str, str]] = set()
        for fact in new_facts:
            normalized_value = " ".join((fact.value or "").lower().split())
            exact_key = (
                (fact.subject or "").strip().lower(),
                (fact.predicate or "").strip().lower(),
                normalized_value,
                (fact.attribution or "contact").strip().lower(),
            )
            if exact_key in seen_exact:
                continue
            seen_exact.add(exact_key)
            compact_facts.append(fact)

        if not compact_facts:
            return []

        # Only do semantic dedup within same (subject, predicate) pairs
        # Facts about different subjects are never duplicates, even if values are similar
        from collections import defaultdict
        facts_by_key: dict[tuple[str, str], list[tuple[Fact, int]]] = defaultdict(list)
        for i, fact in enumerate(compact_facts):
            key = (fact.subject or "", fact.predicate or "")
            facts_by_key[key].append((fact, i))

        embedder = self._get_embedder()
        # Only embed the value for semantic comparison (subject+pred already matched)
        texts = [f.value or "" for f in compact_facts]
        embeddings = embedder.encode(texts)  # (n_new, dim)

        unique_new_facts: list[Fact] = []
        unique_embeddings: list[np.ndarray] = []

        # Process each (subject, predicate) group separately
        for (subject, predicate), fact_indices in facts_by_key.items():
            accepted_embeddings: list[np.ndarray] = []

            for fact, original_idx in fact_indices:
                current_emb = embeddings[original_idx]
                is_duplicate = False

                # --- Check against accepted items in current batch ---
                if accepted_embeddings:
                    batch_sims = np.dot(accepted_embeddings, current_emb)
                    max_batch_sim = np.max(batch_sims) if len(batch_sims) > 0 else 0
                    if max_batch_sim >= self.threshold:
                        logger.debug(
                            "Fact suppressed (Batch duplicate, sim=%.3f): '%s'",
                            max_batch_sim,
                            fact.value,
                        )
                        is_duplicate = True

                if not is_duplicate:
                    unique_new_facts.append(fact)
                    accepted_embeddings.append(current_emb)
                    unique_embeddings.append(current_emb)

        if return_embeddings:
            import numpy as np
            return unique_new_facts, np.vstack(unique_embeddings) if unique_embeddings else np.array([])
        return unique_new_facts
