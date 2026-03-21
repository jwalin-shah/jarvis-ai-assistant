"""BM25 Keyword Search - Exact match keyword retrieval for conversation chunks.

Complements semantic search by providing high precision for specific names,
technical terms, and exact phrases that might be blurred in vector space.

Uses the rank-bm25 library for scoring.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from jarvis.config import get_config

logger = logging.getLogger(__name__)

# Basic tokenizer for BM25 (reused from topic_segmenter pattern)
_WORD_RE = re.compile(r"\b[a-z0-9]{2,}\b")


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25 indexing and search."""
    return _WORD_RE.findall(text.lower())


@dataclass
class BM25Result:
    """Result from BM25 search."""

    rowid: int
    score: float
    text: str


class BM25Searcher:
    """Memory-resident BM25 index for conversation chunks."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._rowids: list[int] = []
        self._corpus: list[str] = []
        self._lock = threading.RLock()

    def is_indexed(self) -> bool:
        """Check if index is populated."""
        return self._bm25 is not None

    def index_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Build/rebuild the BM25 index from chunks.

        Args:
            chunks: List of dicts with 'rowid' and 'text' (concatenated context + reply)

        Returns:
            Number of chunks indexed.
        """
        with self._lock:
            self._rowids = []
            self._corpus = []
            tokenized_corpus = []

            for chunk in chunks:
                rowid = chunk.get("rowid")
                text = chunk.get("text", "")
                if rowid is not None and text:
                    self._rowids.append(rowid)
                    self._corpus.append(text)
                    tokenized_corpus.append(_tokenize(text))

            if tokenized_corpus:
                self._bm25 = BM25Okapi(tokenized_corpus)
                logger.info("Indexed %d chunks for BM25 search", len(tokenized_corpus))
                return len(tokenized_corpus)

            return 0

    def search(self, query: str, limit: int | None = None) -> list[BM25Result]:
        """Search the keyword index.

        Args:
            query: Search query string.
            limit: Max results. Uses config retrieval.bm25_limit if None.

        Returns:
            List of BM25Result objects.
        """
        if limit is None:
            limit = get_config().retrieval.bm25_limit

        if self._bm25 is None:
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        with self._lock:
            scores = self._bm25.get_scores(tokenized_query)
            # Get top indices
            top_n = np.argsort(scores)[::-1][:limit]

            results = []
            for i in top_n:
                if scores[i] > 0:
                    results.append(
                        BM25Result(
                            rowid=self._rowids[i],
                            score=float(scores[i]),
                            text=self._corpus[i],
                        )
                    )

            return results

    def to_dict(self) -> dict[str, Any]:
        """Serialize index to dictionary."""
        with self._lock:
            if self._bm25 is None:
                return {}

            return {
                "rowids": self._rowids,
                "corpus": self._corpus,
                "bm25": {
                    "corpus_size": self._bm25.corpus_size,
                    "avgdl": self._bm25.avgdl,
                    "doc_freqs": self._bm25.doc_freqs,
                    "idf": self._bm25.idf,
                    "doc_len": self._bm25.doc_len,
                    "average_idf": getattr(self._bm25, "average_idf", 0.0),
                },
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BM25Searcher:
        """Deserialize index from dictionary."""
        instance = cls()
        if not data or "bm25" not in data:
            return instance

        instance._rowids = data.get("rowids", [])
        instance._corpus = data.get("corpus", [])

        bm25_data = data["bm25"]
        # Create an empty BM25Okapi instance
        bm25 = BM25Okapi.__new__(BM25Okapi)

        # Restore state
        bm25.corpus_size = bm25_data.get("corpus_size", 0)
        bm25.avgdl = bm25_data.get("avgdl", 0.0)
        bm25.doc_freqs = bm25_data.get("doc_freqs", [])
        bm25.idf = bm25_data.get("idf", {})
        bm25.doc_len = bm25_data.get("doc_len", [])
        bm25.average_idf = bm25_data.get("average_idf", 0.0)

        # Constants from BM25Okapi default init
        bm25.k1 = 1.5
        bm25.b = 0.75
        bm25.epsilon = 0.25

        instance._bm25 = bm25

        return instance
