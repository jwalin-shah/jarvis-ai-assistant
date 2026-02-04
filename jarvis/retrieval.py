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
import math
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from jarvis.config import get_config
from jarvis.db import JarvisDB, get_db
from jarvis.response_classifier_v2 import (
    COMMITMENT_RESPONSE_TYPES,
    TRIGGER_TO_VALID_RESPONSES,
    ResponseType,
)
from jarvis.trigger_classifier import get_trigger_classifier

if TYPE_CHECKING:
    from jarvis.embedding_adapter import Embedder
    from jarvis.index import TriggerIndexSearcher
    from jarvis.trigger_classifier import HybridTriggerClassifier

logger = logging.getLogger(__name__)


# =============================================================================
# Scoring Utilities
# =============================================================================


def compute_temporal_decay(
    source_timestamp: datetime | None,
    half_life_days: float = 365.0,
    min_score: float = 0.1,
) -> float:
    """Compute exponential temporal decay factor for a message timestamp.

    Uses exponential decay: score = 0.5^(age_days / half_life_days)
    This means messages lose half their score after half_life_days.

    Args:
        source_timestamp: Timestamp of the message. If None, returns 1.0 (no decay).
        half_life_days: Number of days until score is halved. Default 365 (1 year).
        min_score: Minimum decay factor to prevent very old messages from being
            completely ignored. Default 0.1 (10% of original score).

    Returns:
        Decay factor between min_score and 1.0.

    Examples:
        >>> from datetime import datetime, timedelta
        >>> now = datetime.now()
        >>> compute_temporal_decay(now)  # Just now
        1.0
        >>> compute_temporal_decay(now - timedelta(days=365))  # 1 year ago
        0.5
        >>> compute_temporal_decay(now - timedelta(days=730))  # 2 years ago
        0.25
        >>> compute_temporal_decay(now - timedelta(days=3650), min_score=0.1)  # 10 years
        0.1  # Clamped to min_score
    """
    if source_timestamp is None:
        return 1.0

    age_days = (datetime.now() - source_timestamp).days
    if age_days <= 0:
        return 1.0

    # Exponential decay: 0.5^(age/half_life)
    decay = math.pow(0.5, age_days / half_life_days)

    # Clamp to minimum score
    return max(min_score, decay)


def reciprocal_rank_fusion(
    rankings: list[list[tuple[Any, float]]],
    k: int = 60,
) -> list[tuple[Any, float]]:
    """Combine multiple rankings using Reciprocal Rank Fusion (RRF).

    RRF is a simple and effective method for combining multiple ranked lists.
    It's robust to different score scales across rankings.

    Formula: score = sum(1 / (k + rank_i)) for each ranking containing the item

    Args:
        rankings: List of rankings, where each ranking is a list of (item_id, score) tuples
            sorted by score descending.
        k: RRF constant. Higher values give more weight to lower ranks. Default 60.

    Returns:
        Combined ranking as list of (item_id, fused_score) sorted by score descending.

    Example:
        >>> r1 = [(1, 0.9), (2, 0.8), (3, 0.7)]  # FAISS ranking
        >>> r2 = [(2, 0.95), (1, 0.85), (4, 0.6)]  # BM25 ranking
        >>> reciprocal_rank_fusion([r1, r2])
        [(2, 0.033), (1, 0.032), (3, 0.016), (4, 0.016)]  # Fused ranking
    """
    scores: dict[Any, float] = {}

    for ranking in rankings:
        for rank_idx, (item_id, _score) in enumerate(ranking):
            # RRF formula: 1 / (k + rank), where rank starts at 1
            rrf_score = 1.0 / (k + rank_idx + 1)
            scores[item_id] = scores.get(item_id, 0.0) + rrf_score

    # Sort by fused score descending
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


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
    def trigger_classifier(self) -> HybridTriggerClassifier | None:
        """Get or create the trigger classifier (hybrid structural + SVM)."""
        if self._trigger_classifier is None:
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
        # Use k*20 to improve recall for rare response types like DECLINE
        try:
            search_results = self.index_searcher.search_with_pairs(
                query=trigger,
                k=k * 20,  # Increased oversampling for rare types
                threshold=min_similarity,
                embedder=embedder,
            )
        except Exception as e:
            logger.warning("FAISS search failed: %s", e)
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

        # NOTE: DB fallback removed - use LLM generation instead when FAISS
        # doesn't find good semantic matches. This prevents returning
        # irrelevant examples that happen to have the same response type.

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

        Performance optimization: Does ONE FAISS search and filters results by type,
        instead of 3 separate searches. This reduces embedding computation from 3x to 1x.

        Args:
            trigger: Query trigger text.
            k_per_type: Number of examples per response type.
            min_similarity: Minimum similarity threshold.
            min_quality: Minimum pair quality score.
            embedder: Optional embedder for FAISS search (pass CachedEmbedder for reuse).
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

        # OPTIMIZATION: Single FAISS search with larger k, then filter by type
        # This avoids 3 separate searches with 3 separate embeddings
        num_types = len(COMMITMENT_RESPONSE_TYPES)
        total_k = k_per_type * num_types * 3  # Extra buffer for filtering

        try:
            search_results = self.index_searcher.search_with_pairs(
                query=trigger,
                k=total_k,
                threshold=min_similarity,
                embedder=embedder,
            )
        except Exception as e:
            logger.warning("FAISS search failed: %s", e)
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

        # NOTE: DB fallback removed - use LLM generation instead when FAISS
        # doesn't find good semantic matches. This prevents returning
        # irrelevant examples that happen to have the same response type.

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

        Performance optimization: Does ONE FAISS search and filters results by type,
        instead of N separate searches for N valid types.

        Args:
            trigger: Query trigger text.
            trigger_da: Trigger DA type (auto-classified if None).
            k_per_type: Number of examples per response type.
            min_similarity: Minimum similarity threshold.
            min_quality: Minimum pair quality score.
            embedder: Optional embedder for FAISS search (pass CachedEmbedder for reuse).

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

        # OPTIMIZATION: Single FAISS search with larger k, then filter by type
        num_types = len(valid_types)
        total_k = k_per_type * num_types * 3  # Extra buffer for filtering

        try:
            search_results = self.index_searcher.search_with_pairs(
                query=trigger,
                k=total_k,
                threshold=min_similarity,
                embedder=embedder,
            )
        except Exception as e:
            logger.warning("FAISS search failed: %s", e)
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

        # NOTE: DB fallback removed - use LLM generation instead when FAISS
        # doesn't find good semantic matches.

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
        1. Single FAISS search for similar triggers (k=50)
        2. Apply contact boost (1.5x for same chat_id)
        3. Apply topic boost (based on embedding similarity to contact profile)
        4. Filter by target response DA type
        5. Return top k by weighted score

        Args:
            trigger: Query trigger text.
            target_response_type: Desired response DA type.
            chat_id: Target contact's chat_id for contact boosting.
            k: Number of examples to return.
            min_similarity: Minimum similarity threshold.
            contact_boost: Multiplier for same-contact results (default 1.5).
            embedder: Optional embedder for FAISS search.

        Returns:
            List of TypedExample objects, sorted by weighted score.
        """
        if isinstance(target_response_type, str):
            try:
                target_response_type = ResponseType(target_response_type)
            except ValueError:
                logger.warning("Invalid response type: %s", target_response_type)
                return []

        # Single FAISS search with large k (50) to have enough for filtering
        try:
            search_results = self.index_searcher.search_with_pairs(
                query=trigger,
                k=50,
                threshold=min_similarity,
                embedder=embedder,
            )
        except Exception as e:
            logger.warning("FAISS search failed: %s", e)
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
                is_contact_match = True
            else:
                score_multiplier = 1.0
                is_contact_match = False

            weighted_score = base_score * score_multiplier

            scored_results.append(
                {
                    **result,
                    "weighted_score": weighted_score,
                    "is_contact_match": is_contact_match,
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

        Args:
            trigger: Query trigger text.
            chat_id: Target contact's chat_id for contact boosting.
            k_per_type: Number of examples per response type.
            min_similarity: Minimum similarity threshold.
            contact_boost: Multiplier for same-contact results.
            embedder: Optional embedder for FAISS search.

        Returns:
            MultiTypeExamples with contact-boosted examples grouped by type.
        """
        # Classify trigger to understand context
        trigger_da, _ = self.classify_trigger(trigger)

        # Get examples for each commitment type with weighting
        examples_by_type: dict[ResponseType, list[TypedExample]] = {}

        for response_type in COMMITMENT_RESPONSE_TYPES:
            examples = self.get_typed_examples_weighted(
                trigger=trigger,
                target_response_type=response_type,
                chat_id=chat_id,
                k=k_per_type,
                min_similarity=min_similarity,
                contact_boost=contact_boost,
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
# BM25 Index Manager
# =============================================================================


class BM25IndexManager:
    """Manages BM25 index for hybrid retrieval.

    Maintains an in-memory BM25 index of trigger texts for sparse retrieval.
    Can be combined with FAISS dense retrieval using reciprocal rank fusion.
    """

    def __init__(self) -> None:
        """Initialize the BM25 index manager."""
        self._index: Any | None = None
        self._pair_ids: list[int] = []
        self._triggers: list[str] = []
        self._lock = threading.Lock()
        self._built = False

    def build_index(self, pairs: list[dict[str, Any]]) -> None:
        """Build BM25 index from pairs.

        Args:
            pairs: List of pair dicts with 'pair_id' and 'trigger_text' keys.
        """
        from rank_bm25 import BM25Okapi

        with self._lock:
            if not pairs:
                logger.warning("No pairs provided for BM25 index")
                return

            # Extract and tokenize triggers
            self._pair_ids = [p["pair_id"] for p in pairs]
            self._triggers = [p["trigger_text"] for p in pairs]

            # Simple whitespace tokenization (can be enhanced later)
            tokenized = [self._tokenize(t) for t in self._triggers]

            self._index = BM25Okapi(tokenized)
            self._built = True
            logger.info("Built BM25 index with %d documents", len(pairs))

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenization with lowercasing.

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens.
        """
        return text.lower().split()

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """Search BM25 index.

        Args:
            query: Query text.
            k: Number of results to return.

        Returns:
            List of (pair_id, score) tuples sorted by score descending.
        """
        if not self._built or self._index is None:
            return []

        with self._lock:
            tokenized_query = self._tokenize(query)
            scores = self._index.get_scores(tokenized_query)

            # Get top-k indices
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

            return [(self._pair_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    @property
    def is_built(self) -> bool:
        """Check if index is built."""
        return self._built

    def clear(self) -> None:
        """Clear the index."""
        with self._lock:
            self._index = None
            self._pair_ids = []
            self._triggers = []
            self._built = False


# Singleton BM25 index
_bm25_index: BM25IndexManager | None = None
_bm25_lock = threading.Lock()


def get_bm25_index() -> BM25IndexManager:
    """Get or create the singleton BM25 index manager."""
    global _bm25_index
    if _bm25_index is None:
        with _bm25_lock:
            if _bm25_index is None:
                _bm25_index = BM25IndexManager()
    return _bm25_index


def reset_bm25_index() -> None:
    """Reset the BM25 index (for testing)."""
    global _bm25_index
    with _bm25_lock:
        if _bm25_index is not None:
            _bm25_index.clear()
        _bm25_index = None


# =============================================================================
# Cross-Encoder Reranker
# =============================================================================


class CrossEncoderReranker:
    """Cross-encoder reranker for improved retrieval accuracy.

    Uses a cross-encoder model to rerank candidates by computing query-document
    relevance scores. More accurate than bi-encoder (FAISS) but slower.
    """

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize the reranker.

        Args:
            model_name: Cross-encoder model name. If None, uses config default.
        """
        config = get_config()
        self._model_name = model_name or config.retrieval.rerank_model
        self._model: Any | None = None
        self._lock = threading.Lock()

    def _load_model(self) -> Any:
        """Lazy load the cross-encoder model."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import CrossEncoder

                        self._model = CrossEncoder(self._model_name)
                        logger.info("Loaded cross-encoder model: %s", self._model_name)
                    except Exception as e:
                        logger.warning("Failed to load cross-encoder: %s", e)
                        raise
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        text_key: str = "trigger_text",
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank candidates using cross-encoder.

        Args:
            query: Query text.
            candidates: List of candidate dicts.
            text_key: Key in candidate dict containing text to compare.
            top_k: Number of top results to return. If None, returns all.

        Returns:
            Reranked list of candidates with 'rerank_score' added.
        """
        if not candidates:
            return []

        try:
            model = self._load_model()
        except Exception:
            # Return candidates as-is if model fails to load
            return candidates[:top_k] if top_k else candidates

        # Create query-document pairs
        pairs = [(query, c.get(text_key, "")) for c in candidates]

        # Get cross-encoder scores
        scores = model.predict(pairs)

        # Add scores to candidates
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])

        # Sort by rerank score descending
        reranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)

        return reranked[:top_k] if top_k else reranked


# Singleton reranker
_reranker: CrossEncoderReranker | None = None
_reranker_lock = threading.Lock()


def get_reranker() -> CrossEncoderReranker:
    """Get or create the singleton reranker."""
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = CrossEncoderReranker()
    return _reranker


def reset_reranker() -> None:
    """Reset the reranker (for testing)."""
    global _reranker
    with _reranker_lock:
        _reranker = None


# =============================================================================
# Enhanced Retrieval Functions
# =============================================================================


def apply_temporal_decay_to_results(
    results: list[dict[str, Any]],
    half_life_days: float | None = None,
    min_score: float | None = None,
    score_key: str = "similarity",
    timestamp_key: str = "source_timestamp",
) -> list[dict[str, Any]]:
    """Apply temporal decay to search results.

    Modifies results in place, adding 'temporal_score' key.

    Args:
        results: List of result dicts.
        half_life_days: Days until score is halved. If None, uses config default.
        min_score: Minimum decay factor. If None, uses config default.
        score_key: Key containing the base score.
        timestamp_key: Key containing the timestamp.

    Returns:
        Results with 'temporal_score' added, sorted by temporal_score descending.
    """
    config = get_config()
    retrieval_config = config.retrieval

    if not retrieval_config.temporal_decay_enabled:
        # Return as-is if disabled
        for r in results:
            r["temporal_score"] = r.get(score_key, 0.0)
        return results

    half_life = half_life_days or retrieval_config.temporal_half_life_days
    min_s = min_score or retrieval_config.temporal_min_score

    for result in results:
        base_score = result.get(score_key, 0.0)
        timestamp = result.get(timestamp_key)

        decay = compute_temporal_decay(timestamp, half_life, min_s)
        result["temporal_score"] = base_score * decay
        result["temporal_decay"] = decay

    # Sort by temporal score
    results.sort(key=lambda x: x.get("temporal_score", 0), reverse=True)
    return results


def hybrid_search_with_bm25(
    query: str,
    faiss_results: list[dict[str, Any]],
    bm25_index: BM25IndexManager,
    k: int = 10,
    rrf_k: int | None = None,
) -> list[dict[str, Any]]:
    """Combine FAISS and BM25 results using reciprocal rank fusion.

    Args:
        query: Query text.
        faiss_results: Results from FAISS search (must have 'pair_id' key).
        bm25_index: BM25 index manager to search.
        k: Number of results to return.
        rrf_k: RRF constant. If None, uses config default.

    Returns:
        Fused results sorted by RRF score.
    """
    config = get_config()
    retrieval_config = config.retrieval

    if not retrieval_config.bm25_enabled or not bm25_index.is_built:
        return faiss_results[:k]

    rrf_constant = rrf_k or retrieval_config.rrf_k

    # Create FAISS ranking as (pair_id, score) tuples
    faiss_ranking = [
        (r["pair_id"], r.get("similarity", 0.0)) for r in faiss_results if r.get("pair_id")
    ]

    # Get BM25 ranking
    bm25_ranking = bm25_index.search(query, k=k * 2)

    # Apply RRF
    fused = reciprocal_rank_fusion([faiss_ranking, bm25_ranking], k=rrf_constant)

    # Build result dict for fused pairs
    faiss_by_pair = {r["pair_id"]: r for r in faiss_results if r.get("pair_id")}

    # Keep all FAISS results, update with RRF scores
    result_map: dict[int, dict[str, Any]] = {}
    for pair_id, rrf_score in fused:
        if pair_id in faiss_by_pair:
            result_map[pair_id] = {
                **faiss_by_pair[pair_id],
                "rrf_score": rrf_score,
                "in_bm25": pair_id in dict(bm25_ranking),
            }
        # Skip pairs only in BM25 (no FAISS data to include)

    # Sort by RRF score and return top k
    sorted_results = sorted(result_map.values(), key=lambda x: x.get("rrf_score", 0), reverse=True)
    return sorted_results[:k]


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
    # Core types
    "TypedExample",
    "MultiTypeExamples",
    "TypedRetriever",
    "get_typed_retriever",
    "reset_typed_retriever",
    # Scoring utilities
    "compute_temporal_decay",
    "reciprocal_rank_fusion",
    "apply_temporal_decay_to_results",
    # BM25 hybrid retrieval
    "BM25IndexManager",
    "get_bm25_index",
    "reset_bm25_index",
    "hybrid_search_with_bm25",
    # Cross-encoder reranking
    "CrossEncoderReranker",
    "get_reranker",
    "reset_reranker",
]
