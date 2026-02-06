"""DA-Filtered Retrieval - Get examples by response type for multi-option generation.

Combines vector similarity search with DA type filtering to retrieve
examples of specific response types (AGREE, DECLINE, DEFER, etc.).

Used by multi-option generation to get diverse few-shot examples:
- For "Want to grab lunch?" → get AGREE examples, DECLINE examples, DEFER examples
- Each type gets its own few-shot examples for generation

Usage:
    from jarvis.search.retrieval import get_typed_retriever, TypedRetriever

    retriever = get_typed_retriever()

    # Get examples for a specific response type
    agree_examples = retriever.get_typed_examples(
        trigger="Want to grab lunch?",
        target_response_type="AGREE",
        k=5
    )
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from jarvis.classifiers.trigger_classifier import get_trigger_classifier
from jarvis.db import JarvisDB, get_db
from jarvis.response_types import (
    COMMITMENT_RESPONSE_TYPES,
    TRIGGER_TO_VALID_RESPONSES,
    ResponseType,
)

if TYPE_CHECKING:
    from jarvis.classifiers.trigger_classifier import HybridTriggerClassifier
    from jarvis.embedding_adapter import Embedder
    from jarvis.search.vec_search import VecSearcher

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
    1. Vector similarity search (find similar triggers)
    2. DA type filtering (get specific response types)
    3. Quality filtering (prefer high-quality pairs)

    Thread Safety:
        This class is thread-safe. Index and classifiers loaded lazily with locking.
    """

    def __init__(
        self,
        db: JarvisDB | None = None,
        vec_searcher: VecSearcher | None = None,
    ) -> None:
        """Initialize the retriever.

        Args:
            db: Database instance. Uses default if None.
            vec_searcher: sqlite-vec searcher. Created lazily if None.
        """
        self._db = db
        self._vec_searcher = vec_searcher
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
    def vec_searcher(self) -> VecSearcher:
        """Get or create the sqlite-vec searcher."""
        with self._lock:
            if self._vec_searcher is None:
                from jarvis.search.vec_search import get_vec_searcher

                self._vec_searcher = get_vec_searcher(self.db)
            return self._vec_searcher

    @property
    def trigger_classifier(self) -> HybridTriggerClassifier | None:
        """Get or create the trigger classifier (hybrid structural + SVM)."""
        with self._lock:
            if self._trigger_classifier is None:
                try:
                    self._trigger_classifier = get_trigger_classifier()
                except Exception as e:
                    logger.warning("Failed to load trigger classifier: %s", e)
            return self._trigger_classifier

    def classify_trigger(self, trigger: str) -> tuple[str | None, float]:
        """Classify the trigger type using hybrid structural + SVM classifier.

        Args:
            trigger: Trigger text to classify.

        Returns:
            Tuple of (trigger_type_value, confidence).
            trigger_type_value is one of: "commitment", "question", "reaction",
            "social", "statement", or None if classification fails.
        """
        if not self.trigger_classifier:
            return None, 0.0

        try:
            result = self.trigger_classifier.classify(trigger)
            # Return the enum value (e.g., "commitment") for compatibility
            return result.trigger_type.value, result.confidence
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
        1. Search vector DB for similar triggers (get more than k to filter)
        2. Filter to pairs with target response DA type
        3. Return top k by similarity

        Args:
            trigger: Query trigger text.
            target_response_type: Desired response DA type.
            k: Number of examples to return.
            min_similarity: Minimum similarity threshold.
            min_quality: Minimum pair quality score.
            embedder: Optional embedder for search.

        Returns:
            List of TypedExample objects.
        """
        if isinstance(target_response_type, str):
            try:
                target_response_type = ResponseType(target_response_type)
            except ValueError:
                logger.warning("Invalid response type: %s", target_response_type)
                return []

        # Search vector DB for similar triggers (get extra to account for filtering)
        # Use k*20 to improve recall for rare response types like DECLINE
        try:
            search_results = self.vec_searcher.search_with_pairs(
                query=trigger,
                k=k * 20,  # Increased oversampling for rare types
                threshold=min_similarity,
                embedder=embedder,
            )
        except Exception as e:
            logger.warning("Vec search failed: %s", e)
            search_results = []

        # Filter to target response type with diversity filtering
        typed_examples = []
        seen_responses: set[str] = set()  # Track seen responses for diversity
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

            # Diversity check: Skip duplicate/near-duplicate responses
            response_normalized = result["response_text"].strip().lower()
            if response_normalized in seen_responses:
                continue
            seen_responses.add(response_normalized)

            typed_examples.append(
                TypedExample(
                    trigger_text=result["trigger_text"],
                    response_text=result["response_text"],
                    response_type=target_response_type,
                    similarity=result["similarity"],
                    confidence=result.get("response_da_conf") or 0.0,
                    pair_id=pair_id,
                )
            )

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
        trigger_da: str | None = None,
    ) -> MultiTypeExamples:
        """Get examples for all commitment response types (AGREE, DECLINE, DEFER).

        Used for multi-option generation where we need examples of each type.

        Performance optimization: Does ONE vector search and filters results by type,
        instead of 3 separate searches. This reduces embedding computation from 3x to 1x.

        Args:
            trigger: Query trigger text.
            k_per_type: Number of examples per response type.
            min_similarity: Minimum similarity threshold.
            min_quality: Minimum pair quality score.
            embedder: Optional embedder for search (pass CachedEmbedder for reuse).
            trigger_da: Pre-classified trigger type (avoids re-classification).

        Returns:
            MultiTypeExamples with examples grouped by type.
        """
        # Use pre-classified trigger_da if provided, otherwise classify once
        if trigger_da is None:
            trigger_da, _ = self.classify_trigger(trigger)

        # Ensure we have a cached embedder for efficiency
        if embedder is None:
            from jarvis.embedding_adapter import CachedEmbedder, get_embedder

            embedder = CachedEmbedder(get_embedder())

        # OPTIMIZATION: Single vector search with larger k, then filter by type
        # This avoids 3 separate searches with 3 separate embeddings
        num_types = len(COMMITMENT_RESPONSE_TYPES)
        total_k = k_per_type * num_types * 3  # Extra buffer for filtering

        try:
            search_results = self.vec_searcher.search_with_pairs(
                query=trigger,
                k=total_k,
                threshold=min_similarity,
                embedder=embedder,
            )
        except Exception as e:
            logger.warning("Vec search failed: %s", e)
            search_results = []

        # Group results by response type
        examples_by_type: dict[ResponseType, list[TypedExample]] = {
            rt: [] for rt in COMMITMENT_RESPONSE_TYPES
        }

        for result in search_results:
            pair_id = result.get("pair_id")
            if not pair_id:
                continue

            # Check quality
            quality = result.get("quality_score", 0.0)
            if quality < min_quality:
                continue

            # Get response type and check if it's a commitment type
            response_da = result.get("response_da_type")
            if not response_da:
                continue

            try:
                response_type = ResponseType(response_da)
            except ValueError:
                continue

            if response_type not in COMMITMENT_RESPONSE_TYPES:
                continue

            # Add to appropriate bucket if not full
            if len(examples_by_type[response_type]) < k_per_type:
                examples_by_type[response_type].append(
                    TypedExample(
                        trigger_text=result["trigger_text"],
                        response_text=result["response_text"],
                        response_type=response_type,
                        similarity=result["similarity"],
                        confidence=result.get("response_da_conf") or 0.0,
                        pair_id=pair_id,
                    )
                )

        # Remove empty type entries
        examples_by_type = {k: v for k, v in examples_by_type.items() if v}

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

        Performance optimization: Does ONE vector search and filters results by type,
        instead of N separate searches for N valid types.

        Args:
            trigger: Query trigger text.
            trigger_da: Trigger DA type (auto-classified if None).
            k_per_type: Number of examples per response type.
            min_similarity: Minimum similarity threshold.
            min_quality: Minimum pair quality score.
            embedder: Optional embedder for search (pass CachedEmbedder for reuse).

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

        # Ensure we have a cached embedder for efficiency
        if embedder is None:
            from jarvis.embedding_adapter import CachedEmbedder, get_embedder

            embedder = CachedEmbedder(get_embedder())

        # OPTIMIZATION: Single vector search with larger k, then filter by type
        num_types = len(valid_types)
        total_k = k_per_type * num_types * 3  # Extra buffer for filtering

        try:
            search_results = self.vec_searcher.search_with_pairs(
                query=trigger,
                k=total_k,
                threshold=min_similarity,
                embedder=embedder,
            )
        except Exception as e:
            logger.warning("Vec search failed: %s", e)
            search_results = []

        # Initialize buckets for valid types only
        valid_types_set = set(valid_types)
        examples_by_type: dict[ResponseType, list[TypedExample]] = {rt: [] for rt in valid_types}

        for result in search_results:
            pair_id = result.get("pair_id")
            if not pair_id:
                continue

            quality = result.get("quality_score", 0.0)
            if quality < min_quality:
                continue

            response_da = result.get("response_da_type")
            if not response_da:
                continue

            try:
                response_type = ResponseType(response_da)
            except ValueError:
                continue

            if response_type not in valid_types_set:
                continue

            if len(examples_by_type[response_type]) < k_per_type:
                examples_by_type[response_type].append(
                    TypedExample(
                        trigger_text=result["trigger_text"],
                        response_text=result["response_text"],
                        response_type=response_type,
                        similarity=result["similarity"],
                        confidence=result.get("response_da_conf") or 0.0,
                        pair_id=pair_id,
                    )
                )

        # Remove empty type entries
        examples_by_type = {k: v for k, v in examples_by_type.items() if v}

        return MultiTypeExamples(
            query_trigger=trigger,
            trigger_da=trigger_da,
            examples_by_type=examples_by_type,
        )

    def get_typed_examples_weighted(
        self,
        trigger: str,
        target_response_type: str | ResponseType,
        chat_id: str | None = None,
        k: int = 5,
        min_similarity: float = 0.3,
        contact_boost: float = 1.5,
        embedder: Embedder | None = None,
    ) -> list[TypedExample]:
        """Get examples with contact-aware weighted scoring.

        Strategy:
        1. Single vector search for similar triggers (k=50)
        2. Apply contact boost (1.5x for same chat_id)
        3. Filter by target response DA type
        4. Return top k by weighted score

        Args:
            trigger: Query trigger text.
            target_response_type: Desired response DA type.
            chat_id: Target contact's chat_id for contact boosting.
            k: Number of examples to return.
            min_similarity: Minimum similarity threshold.
            contact_boost: Multiplier for same-contact results (default 1.5).
            embedder: Optional embedder for search.

        Returns:
            List of TypedExample objects, sorted by weighted score.
        """
        if isinstance(target_response_type, str):
            try:
                target_response_type = ResponseType(target_response_type)
            except ValueError:
                logger.warning("Invalid response type: %s", target_response_type)
                return []

        # Single vector search with large k (50) to have enough for filtering
        try:
            search_results = self.vec_searcher.search_with_pairs(
                query=trigger,
                k=50,
                threshold=min_similarity,
                embedder=embedder,
            )
        except Exception as e:
            logger.warning("Vec search failed: %s", e)
            search_results = []

        # Apply weighted scoring
        scored_results = []
        for result in search_results:
            pair_id = result.get("pair_id")
            if not pair_id:
                continue

            # Filter by DA type
            if result.get("response_da_type") != target_response_type.value:
                continue

            base_score = result.get("similarity", 0.0)

            # Contact boost: prefer results from same chat
            result_chat_id = result.get("chat_id")
            if chat_id and result_chat_id and result_chat_id == chat_id:
                score_multiplier = contact_boost
            else:
                score_multiplier = 1.0

            weighted_score = base_score * score_multiplier

            scored_results.append(
                {
                    **result,
                    "weighted_score": weighted_score,
                }
            )

        # Sort by weighted score descending
        scored_results.sort(key=lambda x: x["weighted_score"], reverse=True)

        # Convert to TypedExample
        typed_examples = []
        for result in scored_results[:k]:
            typed_examples.append(
                TypedExample(
                    trigger_text=result["trigger_text"],
                    response_text=result["response_text"],
                    response_type=target_response_type,
                    similarity=result["weighted_score"],  # Use weighted score
                    confidence=result.get("response_da_conf") or 0.0,
                    pair_id=result["pair_id"],
                )
            )

        return typed_examples

    def get_examples_for_commitment_weighted(
        self,
        trigger: str,
        chat_id: str | None = None,
        k_per_type: int = 3,
        min_similarity: float = 0.3,
        contact_boost: float = 1.5,
        embedder: Embedder | None = None,
    ) -> MultiTypeExamples:
        """Get commitment examples (AGREE/DECLINE/DEFER) with contact-aware weighting.

        Like get_examples_for_commitment but boosts results from the same contact.

        Performance optimization: Does ONE vector search and filters results by type,
        instead of N separate searches. This reduces embedding computation from Nx to 1x.

        Args:
            trigger: Query trigger text.
            chat_id: Target contact's chat_id for contact boosting.
            k_per_type: Number of examples per response type.
            min_similarity: Minimum similarity threshold.
            contact_boost: Multiplier for same-contact results.
            embedder: Optional embedder for search.

        Returns:
            MultiTypeExamples with contact-boosted examples grouped by type.
        """
        # Classify trigger to understand context
        trigger_da, _ = self.classify_trigger(trigger)

        # Ensure we have a cached embedder for efficiency
        if embedder is None:
            from jarvis.embedding_adapter import CachedEmbedder, get_embedder

            embedder = CachedEmbedder(get_embedder())

        # OPTIMIZATION: Single vector search with larger k, then filter by type and apply weighting
        num_types = len(COMMITMENT_RESPONSE_TYPES)
        total_k = k_per_type * num_types * 10  # Extra buffer for filtering

        try:
            search_results = self.vec_searcher.search_with_pairs(
                query=trigger,
                k=total_k,
                threshold=min_similarity,
                embedder=embedder,
            )
        except Exception as e:
            logger.warning("Vec search failed: %s", e)
            search_results = []

        # Apply weighted scoring and group by response type
        examples_by_type: dict[ResponseType, list[tuple[TypedExample, float]]] = {
            rt: [] for rt in COMMITMENT_RESPONSE_TYPES
        }

        for result in search_results:
            pair_id = result.get("pair_id")
            if not pair_id:
                continue

            # Filter by DA type
            response_da = result.get("response_da_type")
            if not response_da:
                continue

            try:
                response_type = ResponseType(response_da)
            except ValueError:
                continue

            if response_type not in COMMITMENT_RESPONSE_TYPES:
                continue

            # Skip if this type already has enough examples
            if len(examples_by_type[response_type]) >= k_per_type:
                continue

            base_score = result.get("similarity", 0.0)

            # Contact boost: prefer results from same chat
            result_chat_id = result.get("chat_id")
            if chat_id and result_chat_id and result_chat_id == chat_id:
                score_multiplier = contact_boost
            else:
                score_multiplier = 1.0

            weighted_score = base_score * score_multiplier

            typed_example = TypedExample(
                trigger_text=result["trigger_text"],
                response_text=result["response_text"],
                response_type=response_type,
                similarity=weighted_score,  # Use weighted score
                confidence=result.get("response_da_conf") or 0.0,
                pair_id=pair_id,
            )

            examples_by_type[response_type].append((typed_example, weighted_score))

        # Sort each type by weighted score and extract examples
        final_examples_by_type: dict[ResponseType, list[TypedExample]] = {}
        for response_type, examples_with_scores in examples_by_type.items():
            if examples_with_scores:
                # Sort by weighted score descending
                examples_with_scores.sort(key=lambda x: x[1], reverse=True)
                # Extract just the TypedExample objects
                final_examples_by_type[response_type] = [ex for ex, _score in examples_with_scores]

        return MultiTypeExamples(
            query_trigger=trigger,
            trigger_da=trigger_da,
            examples_by_type=final_examples_by_type,
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
