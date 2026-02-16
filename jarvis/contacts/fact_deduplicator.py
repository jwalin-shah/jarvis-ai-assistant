"""Semantic deduplication for extracted facts.

Ensures that semantically identical facts (e.g., "lives in Austin" vs
"resides in Austin") are merged rather than duplicated in the Knowledge Graph.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from jarvis.contacts.contact_profile import Fact

logger = logging.getLogger(__name__)


class FactDeduplicator:
    """Deduplicates facts using semantic similarity with persistent caching."""

    def __init__(self, threshold: float = 0.85) -> None:
        """Initialize deduplicator.

        Args:
            threshold: Semantic similarity threshold (0-1) above which
                       two facts are considered duplicates.
        """
        self.threshold = threshold
        self._embedder: Any | None = None
        self._cache_path = os.path.expanduser("~/.jarvis/embedding_cache.db")
        self._init_cache()

    def _init_cache(self) -> None:
        """Initialize the persistent embedding cache table."""
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
        with sqlite3.connect(self._cache_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    hash TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _get_cached_embeddings(self, texts: list[str]) -> dict[str, np.ndarray]:
        """Fetch multiple embeddings from the persistent cache."""
        hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
        results = {}
        with sqlite3.connect(self._cache_path) as conn:
            # Chunked fetch for large batches
            for i in range(0, len(hashes), 900):
                chunk = hashes[i : i + 900]
                placeholders = ",".join(["?"] * len(chunk))
                rows = conn.execute(
                    f"SELECT hash, embedding FROM embedding_cache WHERE hash IN ({placeholders})",  # nosec B608
                    chunk,
                ).fetchall()
                for h, blob in rows:
                    results[h] = np.frombuffer(blob, dtype=np.float32)
        return results

    def _save_to_cache(self, text_to_emb: dict[str, np.ndarray]) -> None:
        """Save new embeddings to the persistent cache."""
        if not text_to_emb:
            return
        data = [
            (hashlib.sha256(t.encode("utf-8")).hexdigest(), emb.tobytes())
            for t, emb in text_to_emb.items()
        ]
        with sqlite3.connect(self._cache_path) as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO embedding_cache (hash, embedding) VALUES (?, ?)", data
            )

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

        # --- CACHE-AWARE EMBEDDING ---
        texts = [f.to_searchable_text() for f in compact_facts]
        cached_map = self._get_cached_embeddings(texts)

        embeddings: list[Any] = [None] * len(texts)
        missing_indices = []
        missing_texts = []

        # 1. Fill from cache
        for i, text in enumerate(texts):
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if h in cached_map:
                embeddings[i] = cached_map[h]
            else:
                missing_indices.append(i)
                missing_texts.append(text)

        # 2. Encode missing only
        if missing_texts:
            embedder = self._get_embedder()
            new_embs = embedder.encode(missing_texts)

            new_cache_data = {}
            for idx, emb in zip(missing_indices, new_embs):
                embeddings[idx] = emb
                new_cache_data[texts[idx]] = emb

            # 3. Persist new embs
            self._save_to_cache(new_cache_data)
            logger.debug(
                "Encoded %d new facts, %d pulled from cache",
                len(missing_texts),
                len(texts) - len(missing_texts),
            )

        # Convert back to array for math
        embeddings_arr = np.array(embeddings)

        unique_new_facts: list[Fact] = []
        unique_embeddings: list[np.ndarray] = []

        # Process each (subject, predicate) group separately
        for (subject, predicate), fact_indices in facts_by_key.items():
            accepted_embeddings: list[np.ndarray] = []

            for fact, original_idx in fact_indices:
                current_emb = embeddings_arr[original_idx]
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

            return unique_new_facts, np.vstack(
                unique_embeddings
            ) if unique_embeddings else np.array([])
        return unique_new_facts
