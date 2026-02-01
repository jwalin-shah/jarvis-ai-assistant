"""DA-Filtered Retrieval - Get examples by response type for multi-option generation.

Combines FAISS similarity search with DA type filtering to retrieve
examples of specific response types (AGREE, DECLINE, DEFER, etc.).

Used by multi-option generation to get diverse few-shot examples:
- For "Want to grab lunch?" → get AGREE examples, DECLINE examples, DEFER examples
- Each type gets its own few-shot examples for generation

Usage:
    from jarvis.retrieval import get_typed_retriever, TypedRetriever

    retriever = get_typed_retriever()

    # Get examples for a specific response type
    agree_examples = retriever.get_typed_examples(
        trigger="Want to grab lunch?",
        target_response_type="AGREE",
        k=5
    )

    # Get examples for all commitment response types at once
    all_examples = retriever.get_examples_for_commitment(
        trigger="Want to grab lunch?",
        k_per_type=3
    )
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from jarvis.db import JarvisDB, get_db
from jarvis.response_classifier import (
    COMMITMENT_RESPONSE_TYPES,
    TRIGGER_TO_VALID_RESPONSES,
    ResponseType,
)

if TYPE_CHECKING:
    from jarvis.embedding_adapter import Embedder
    from jarvis.index import TriggerIndexSearcher

logger = logging.getLogger(__name__)


@dataclass
class TypedExample:
    """An example with its response type."""

    trigger_text: str
    response_text: str
    response_type: ResponseType
    similarity: float  # Similarity to query trigger
    confidence: float  # DA classification confidence
    pair_id: int | None = None


@dataclass
class MultiTypeExamples:
    """Examples grouped by response type for multi-option generation."""

    query_trigger: str
    trigger_da: str | None  # Classified trigger type
    examples_by_type: dict[ResponseType, list[TypedExample]]

    def get_examples(self, response_type: ResponseType) -> list[TypedExample]:
        """Get examples for a specific response type."""
        return self.examples_by_type.get(response_type, [])

    def has_examples(self, response_type: ResponseType) -> bool:
        """Check if we have examples for a response type."""
        examples = self.examples_by_type.get(response_type, [])
        return len(examples) > 0

    @property
    def available_types(self) -> list[ResponseType]:
        """Get response types that have examples."""
        return [t for t, examples in self.examples_by_type.items() if examples]


class TypedRetriever:
    """Retrieves examples filtered by response DA type.

    Combines:
    1. FAISS similarity search (find similar triggers)
    2. DA type filtering (get specific response types)
    3. Quality filtering (prefer high-quality pairs)

    Thread Safety:
        This class is thread-safe. Index and classifiers loaded lazily with locking.
    """

    def __init__(
        self,
        db: JarvisDB | None = None,
        index_searcher: TriggerIndexSearcher | None = None,
    ) -> None:
        """Initialize the retriever.

        Args:
            db: Database instance. Uses default if None.
            index_searcher: FAISS index searcher. Created lazily if None.
        """
        self._db = db
        self._index_searcher = index_searcher
        self._trigger_classifier = None
        self._lock = threading.Lock()

    @property
    def db(self) -> JarvisDB:
        """Get or create the database instance."""
        if self._db is None:
            self._db = get_db()
            self._db.init_schema()
        return self._db

    @property
    def index_searcher(self) -> TriggerIndexSearcher:
        """Get or create the FAISS index searcher."""
        if self._index_searcher is None:
            with self._lock:
                if self._index_searcher is None:
                    from jarvis.index import TriggerIndexSearcher
                    self._index_searcher = TriggerIndexSearcher(self.db)
        return self._index_searcher

    @property
    def trigger_classifier(self):
        """Get or create the trigger DA classifier."""
        if self._trigger_classifier is None:
            with self._lock:
                if self._trigger_classifier is None:
                    try:
                        from scripts.build_da_classifier import DialogueActClassifier
                        self._trigger_classifier = DialogueActClassifier("trigger")
                    except Exception as e:
                        logger.warning("Failed to load trigger classifier: %s", e)
        return self._trigger_classifier

    def classify_trigger(self, trigger: str) -> tuple[str | None, float]:
        """Classify the trigger DA type.

        Args:
            trigger: Trigger text to classify.

        Returns:
            Tuple of (trigger_da_type, confidence).
        """
        if not self.trigger_classifier:
            return None, 0.0

        try:
            result = self.trigger_classifier.classify(trigger)
            return result.label, result.confidence
        except Exception as e:
            logger.warning("Trigger classification failed: %s", e)
            return None, 0.0

    def get_typed_examples(
        self,
        trigger: str,
        target_response_type: str | ResponseType,
        k: int = 5,
        min_similarity: float = 0.3,
        min_quality: float = 0.0,
        embedder: Embedder | None = None,
    ) -> list[TypedExample]:
        """Get examples filtered by response DA type.

        Strategy:
        1. Search FAISS for similar triggers (get more than k to filter)
        2. Filter to pairs with target response DA type
        3. Return top k by similarity

        Args:
            trigger: Query trigger text.
            target_response_type: Desired response DA type.
            k: Number of examples to return.
            min_similarity: Minimum similarity threshold.
            min_quality: Minimum pair quality score.
            embedder: Optional embedder for FAISS search.

        Returns:
            List of TypedExample objects.
        """
        if isinstance(target_response_type, str):
            try:
                target_response_type = ResponseType(target_response_type)
            except ValueError:
                logger.warning("Invalid response type: %s", target_response_type)
                return []

        # Search FAISS for similar triggers (get extra to account for filtering)
        try:
            search_results = self.index_searcher.search_with_pairs(
                query=trigger,
                k=k * 5,  # Get extra for filtering
                threshold=min_similarity,
                embedder=embedder,
            )
        except Exception as e:
            logger.warning("FAISS search failed: %s", e)
            search_results = []

        # Filter to target response type
        typed_examples = []
        for result in search_results:
            # search_with_pairs returns flattened dict with all fields
            pair_id = result.get("pair_id")
            if not pair_id:
                continue

            # Check response DA type
            if result.get("response_da_type") != target_response_type.value:
                continue

            # Check quality
            quality = result.get("quality_score", 0.0)
            if quality < min_quality:
                continue

            typed_examples.append(TypedExample(
                trigger_text=result["trigger_text"],
                response_text=result["response_text"],
                response_type=target_response_type,
                similarity=result["similarity"],
                confidence=result.get("response_da_conf") or 0.0,
                pair_id=pair_id,
            ))

            if len(typed_examples) >= k:
                break

        # If FAISS didn't return enough, fall back to DB query
        if len(typed_examples) < k:
            # Use set for O(1) duplicate checking instead of O(n) linear scan
            seen_pair_ids = {e.pair_id for e in typed_examples}

            db_pairs = self.db.get_pairs_by_response_da(
                response_da=target_response_type.value,
                min_quality=min_quality,
                limit=(k - len(typed_examples)) * 2,  # Get extra to account for duplicates
            )
            for pair in db_pairs:
                # Avoid duplicates - O(1) set lookup
                if pair.id in seen_pair_ids:
                    continue

                seen_pair_ids.add(pair.id)
                typed_examples.append(TypedExample(
                    trigger_text=pair.trigger_text,
                    response_text=pair.response_text,
                    response_type=target_response_type,
                    similarity=0.0,  # Not from FAISS
                    confidence=pair.response_da_conf or 0.0,
                    pair_id=pair.id,
                ))

                if len(typed_examples) >= k:
                    break

        return typed_examples

    def get_examples_for_commitment(
        self,
        trigger: str,
        k_per_type: int = 3,
        min_similarity: float = 0.3,
        min_quality: float = 0.0,
        embedder: Embedder | None = None,
    ) -> MultiTypeExamples:
        """Get examples for all commitment response types (AGREE, DECLINE, DEFER).

        Used for multi-option generation where we need examples of each type.

        Args:
            trigger: Query trigger text.
            k_per_type: Number of examples per response type.
            min_similarity: Minimum similarity threshold.
            min_quality: Minimum pair quality score.
            embedder: Optional embedder for FAISS search.

        Returns:
            MultiTypeExamples with examples grouped by type.
        """
        # Classify trigger to understand context
        trigger_da, trigger_conf = self.classify_trigger(trigger)

        # Get examples for each commitment type
        examples_by_type: dict[ResponseType, list[TypedExample]] = {}

        for response_type in COMMITMENT_RESPONSE_TYPES:
            examples = self.get_typed_examples(
                trigger=trigger,
                target_response_type=response_type,
                k=k_per_type,
                min_similarity=min_similarity,
                min_quality=min_quality,
                embedder=embedder,
            )
            if examples:
                examples_by_type[response_type] = examples

        return MultiTypeExamples(
            query_trigger=trigger,
            trigger_da=trigger_da,
            examples_by_type=examples_by_type,
        )

    def get_examples_for_trigger_type(
        self,
        trigger: str,
        trigger_da: str | None = None,
        k_per_type: int = 3,
        min_similarity: float = 0.3,
        min_quality: float = 0.0,
        embedder: Embedder | None = None,
    ) -> MultiTypeExamples:
        """Get examples for valid response types given a trigger DA type.

        Uses TRIGGER_TO_VALID_RESPONSES to determine which response types
        are appropriate for the trigger type.

        Args:
            trigger: Query trigger text.
            trigger_da: Trigger DA type (auto-classified if None).
            k_per_type: Number of examples per response type.
            min_similarity: Minimum similarity threshold.
            min_quality: Minimum pair quality score.
            embedder: Optional embedder for FAISS search.

        Returns:
            MultiTypeExamples with examples for valid response types.
        """
        # Auto-classify trigger if not provided
        if trigger_da is None:
            trigger_da, _ = self.classify_trigger(trigger)

        # Get valid response types for this trigger
        valid_types = TRIGGER_TO_VALID_RESPONSES.get(trigger_da, [])
        if not valid_types:
            # Default to commitment types if unknown trigger
            valid_types = list(COMMITMENT_RESPONSE_TYPES)

        # Get examples for each valid type
        examples_by_type: dict[ResponseType, list[TypedExample]] = {}

        for response_type in valid_types:
            examples = self.get_typed_examples(
                trigger=trigger,
                target_response_type=response_type,
                k=k_per_type,
                min_similarity=min_similarity,
                min_quality=min_quality,
                embedder=embedder,
            )
            if examples:
                examples_by_type[response_type] = examples

        return MultiTypeExamples(
            query_trigger=trigger,
            trigger_da=trigger_da,
            examples_by_type=examples_by_type,
        )


# =============================================================================
# Singleton Access
# =============================================================================

_retriever: TypedRetriever | None = None
_retriever_lock = threading.Lock()


def get_typed_retriever() -> TypedRetriever:
    """Get or create the singleton TypedRetriever instance."""
    global _retriever

    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = TypedRetriever()

    return _retriever


def reset_typed_retriever() -> None:
    """Reset the singleton retriever."""
    global _retriever

    with _retriever_lock:
        _retriever = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "TypedExample",
    "MultiTypeExamples",
    "TypedRetriever",
    "get_typed_retriever",
    "reset_typed_retriever",
]
