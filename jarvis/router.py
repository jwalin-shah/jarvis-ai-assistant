"""Reply Router - Routes incoming messages to LLM generation with RAG context.

All non-empty messages go through LLM generation. Response mobilization
(Stivers & Rossano 2010) informs the prompt, not the routing decision.

Usage:
    from jarvis.router import get_reply_router, ReplyRouter

    router = get_reply_router()
    result = router.route(
        incoming="Want to grab lunch?",
        contact_id=1,
        thread=["Hey!", "What's up?"],
    )

    if result['type'] == 'generated':
        print(f"AI response: {result['response']}")
    else:  # clarify
        print(f"Need more info: {result['response']}")
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

from jarvis.classifiers.response_mobilization import (
    MobilizationResult,
    ResponsePressure,
    classify_response_pressure,
)
from jarvis.db import JarvisDB, get_db
from jarvis.embedding_adapter import CachedEmbedder, get_embedder
from jarvis.errors import ErrorCode, JarvisError
from jarvis.observability.metrics_router import (
    RoutingMetrics,
    get_routing_metrics_store,
    hash_query,
)
from jarvis.reply_service import ReplyService
from jarvis.services import ContextService

if TYPE_CHECKING:
    from integrations.imessage.reader import ChatDBReader
    from jarvis.search.semantic_search import SemanticSearcher
    from models import MLXGenerator

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class RouterError(JarvisError):
    """Raised when routing operations fail."""

    default_message = "Router operation failed"
    default_code = ErrorCode.UNKNOWN


class IndexNotAvailableError(RouterError):
    """Raised when vector index is not available."""

    default_message = "Vector index not available. Run 'jarvis db build-index' first."


# =============================================================================
# Data Classes & Type Definitions
# =============================================================================


class RoutingResponse(TypedDict, total=False):
    """Typed dictionary for routing response.

    Attributes:
        type: Response type ('generated' or 'clarify')
        response: The response text
        confidence: Confidence level ('high', 'medium', 'low')
        similarity_score: Best similarity score from vector search
        cluster_name: Name of matched cluster
        contact_style: Style notes for the contact
        similar_triggers: List of similar past triggers
        reason: Reason for clarification (if type='clarify')
    """

    type: str
    response: str
    confidence: str
    similarity_score: float
    cluster_name: str | None
    contact_style: str | None
    similar_triggers: list[str] | None
    reason: str


class StatsResponse(TypedDict, total=False):
    """Typed dictionary for routing statistics response."""

    db_stats: dict[str, Any]
    index_available: bool
    index_vectors: int
    index_type: str


@dataclass
class RouteResult:
    """Result of routing an incoming message.

    Attributes:
        response: The response text.
        type: Response type ('generated', 'clarify').
        confidence: Confidence level ('high', 'medium', 'low').
        similarity_score: Best similarity score from vector search.
        cluster_name: Name of matched cluster.
        contact_style: Style notes for the contact.
        similar_triggers: List of similar past triggers found.
    """

    response: str
    type: str  # 'generated', 'clarify'
    confidence: str  # 'high', 'medium', 'low'
    similarity_score: float = 0.0
    cluster_name: str | None = None
    contact_style: str | None = None
    similar_triggers: list[str] | None = None


# =============================================================================
# Reply Router
# =============================================================================


class ReplyRouter:
    """Routes incoming messages to LLM generation with RAG context.

    All non-empty messages go through generation. Response mobilization
    informs the prompt (tone/urgency), not the routing decision.

    Thread Safety:
        This class is thread-safe for routing operations.
        Index and generator initialization uses lazy loading.
    """

    def __init__(
        self,
        db: JarvisDB | None = None,
        generator: MLXGenerator | None = None,
        imessage_reader: ChatDBReader | None = None,
    ) -> None:
        """Initialize the router.

        Args:
            db: Database instance for contacts and pairs. Uses default if None.
            generator: MLX generator for LLM responses. Created lazily if None.
            imessage_reader: iMessage reader for fetching conversation history.
                Created lazily if None.
        """
        self._db = db
        self._semantic_searcher: SemanticSearcher | None = None
        self._generator = generator
        self._imessage_reader = imessage_reader
        self._lock = threading.RLock()

        # Services
        self._context_service: ContextService | None = None
        self._reply_service: ReplyService | None = None

    @property
    def db(self) -> JarvisDB:
        """Get or create the database instance."""
        if self._db is None:
            self._db = get_db()
            self._db.init_schema()
        return self._db

    @property
    def semantic_searcher(self) -> SemanticSearcher:
        """Get or create the semantic searcher."""
        with self._lock:
            if self._semantic_searcher is None and self.imessage_reader:
                from jarvis.search.semantic_search import get_semantic_searcher

                self._semantic_searcher = get_semantic_searcher(self.imessage_reader)
            return self._semantic_searcher

    @property
    def generator(self) -> MLXGenerator:
        """Get or create the MLX generator."""
        with self._lock:
            if self._generator is None:
                from models import get_generator

                self._generator = get_generator(skip_templates=True)
            return self._generator

    @property
    def imessage_reader(self) -> ChatDBReader | None:
        """Get or create the iMessage reader for fetching conversation history."""
        with self._lock:
            if self._imessage_reader is None:
                try:
                    from integrations.imessage.reader import ChatDBReader

                    self._imessage_reader = ChatDBReader()
                except Exception as e:
                    logger.warning("Could not initialize iMessage reader: %s", e)
                    return None
            return self._imessage_reader

    @property
    def context_service(self) -> ContextService:
        """Get or create the context service."""
        with self._lock:
            if self._context_service is None:
                self._context_service = ContextService(
                    db=self.db,
                    imessage_reader=self.imessage_reader,
                )
            return self._context_service

    @property
    def reply_service(self) -> ReplyService:
        """Get or create the reply service."""
        with self._lock:
            if self._reply_service is None:
                from jarvis.reply_service import ReplyService

                self._reply_service = ReplyService(
                    db=self.db, generator=self.generator, imessage_reader=self.imessage_reader
                )
            return self._reply_service

    def close(self) -> None:
        """Clean up resources.

        Closes the iMessage reader connection if it was created.
        Call this when done with the router, or use as a context manager.
        """
        if self._imessage_reader is not None:
            try:
                self._imessage_reader.close()
            except Exception as e:
                logger.debug("Error closing iMessage reader: %s", e)
            self._imessage_reader = None

    def __del__(self) -> None:
        """Clean up on deletion."""
        self.close()

    @staticmethod
    def _build_mobilization_hint(mobilization: MobilizationResult) -> str | None:
        """Translate mobilization analysis into a lightweight prompt hint.

        This is intentionally small and stable because mobilization no longer
        changes routing decisions; it only nudges generation style.
        """
        if mobilization.pressure == ResponsePressure.HIGH:
            return "Respond clearly and directly. Keep momentum and include a concrete next step."
        if mobilization.pressure == ResponsePressure.LOW:
            return "Keep the reply natural and concise."
        return None

    def _record_routing_metrics(
        self,
        incoming: str,
        decision: str,
        similarity_score: float,
        latency_ms: dict[str, float],
        cached_embedder: CachedEmbedder,
        vec_candidates: int,
        model_loaded: bool,
    ) -> None:
        try:
            metrics = RoutingMetrics(
                timestamp=time.time(),
                query_hash=hash_query(incoming),
                latency_ms=latency_ms,
                embedding_computations=cached_embedder.embedding_computations,
                vec_candidates=vec_candidates,
                routing_decision=decision,
                similarity_score=similarity_score,
                cache_hit=cached_embedder.cache_hit,
                model_loaded=model_loaded,
            )
            get_routing_metrics_store().record(metrics)
        except Exception as e:
            logger.debug("Routing metrics write failed: %s", e)

    def route(
        self,
        incoming: str,
        contact_id: int | None = None,
        thread: list[str] | None = None,
        chat_id: str | None = None,
    ) -> RoutingResponse:
        """Route an incoming message to LLM generation with RAG context.

        All non-empty messages go through generation. Response mobilization
        informs the prompt, not the routing decision.

        Args:
            incoming: The incoming message text to respond to.
            contact_id: Optional contact ID for personalization.
            thread: Optional list of recent messages for context.
            chat_id: Optional chat ID for context lookup.

        Returns:
            Dict with routing result containing:
            - type: 'generated' or 'clarify'
            - response: The response text
            - confidence: 'high', 'medium', or 'low'
            - Additional metadata (similarity_score, similar_triggers, etc.)
        """
        routing_start = time.perf_counter()
        latency_ms: dict[str, float] = {}
        cached_embedder = CachedEmbedder(get_embedder())

        # Normalize incoming message early
        incoming = incoming.strip() if incoming else ""

        # Precompute embedding (reused by vec search)
        if incoming:
            embed_start = time.perf_counter()
            cached_embedder.encode(incoming)
            latency_ms["embedding_precompute"] = (time.perf_counter() - embed_start) * 1000

        logger.info("=" * 60)
        logger.info("ROUTE START | input: %s", incoming[:80] if incoming else "(empty)")

        def record_and_return(
            result: dict[str, Any],
            similarity_score: float,
            vec_candidates: int = 0,
            model_loaded: bool = False,
            decision: str | None = None,
        ) -> dict[str, Any]:
            latency_ms["total"] = (time.perf_counter() - routing_start) * 1000
            routing_decision = decision or (
                "generate" if result.get("type") == "generated" else "clarify"
            )
            self._record_routing_metrics(
                incoming=incoming,
                decision=routing_decision,
                similarity_score=similarity_score,
                latency_ms=latency_ms,
                cached_embedder=cached_embedder,
                vec_candidates=vec_candidates,
                model_loaded=model_loaded,
            )
            logger.info(
                "ROUTE END | decision=%s sim=%.3f latency=%s",
                routing_decision,
                similarity_score,
                {k: f"{v:.1f}ms" for k, v in latency_ms.items()},
            )
            logger.info("ROUTE OUTPUT | %s", result.get("response", "")[:100])
            logger.info("=" * 60)
            return result

        # Empty message check (incoming already normalized/stripped above)
        if not incoming:
            # ReplyService.generate_reply handles empty/error cases
            result = self.reply_service.generate_reply(
                incoming="",
            )
            return record_and_return(result, similarity_score=0.0, decision="clarify")

        # Get contact
        contact = self.context_service.get_contact(contact_id, chat_id)

        # Classify response pressure
        mobilization_start = time.perf_counter()
        mobilization = classify_response_pressure(incoming)
        latency_ms["mobilization"] = (time.perf_counter() - mobilization_start) * 1000
        logger.debug(
            "Mobilization: pressure=%s type=%s conf=%.2f",
            mobilization.pressure.value,
            mobilization.response_type.value,
            mobilization.confidence,
        )

        # Search for similar examples (pass cached_embedder to avoid re-encoding)
        search_start = time.perf_counter()
        search_results = self.context_service.search_examples(
            incoming,
            chat_id=chat_id,
            contact_id=contact.id if contact else None,
            embedder=cached_embedder,
        )
        latency_ms["vec_search"] = (time.perf_counter() - search_start) * 1000

        # Generate response
        model_loaded = self.generator.is_loaded()
        generate_start = time.perf_counter()
        result = self.reply_service.generate_reply(
            incoming=incoming,
            contact=contact,
            search_results=search_results,
            thread=thread,
            chat_id=chat_id,
            mobilization=mobilization,
            cached_embedder=cached_embedder,
        )
        latency_ms["generate"] = (time.perf_counter() - generate_start) * 1000

        similarity = search_results[0]["similarity"] if search_results else 0.0

        # Map mobilization pressure to confidence
        if mobilization.pressure == ResponsePressure.HIGH:
            result["confidence"] = "high"
        else:
            result["confidence"] = "medium"

        return record_and_return(
            result,
            similarity_score=similarity,
            vec_candidates=len(search_results),
            model_loaded=model_loaded,
            decision="generate",
        )

    def get_routing_stats(self) -> StatsResponse:
        """Get statistics about the router's index and database.

        Returns:
            Dict with index and database statistics.
        """
        stats: StatsResponse = {
            "db_stats": self.db.get_stats(),
            "index_available": False,
        }

        try:
            # Check if vec tables have data
            with self.db.connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) as cnt FROM vec_chunks")
                row = cursor.fetchone()
                count = row["cnt"] if row else 0
                if count > 0:
                    stats["index_available"] = True
                    stats["index_vectors"] = count
                    stats["index_type"] = "sqlite-vec"
        except Exception as e:
            logger.debug("Failed to get index stats: %s", e)

        return stats


# =============================================================================
# Singleton Access
# =============================================================================

_router: ReplyRouter | None = None
_router_lock = threading.Lock()


def get_reply_router() -> ReplyRouter:
    """Get or create the singleton ReplyRouter instance.

    Returns:
        The shared ReplyRouter instance.
    """
    global _router

    if _router is None:
        with _router_lock:
            if _router is None:
                _router = ReplyRouter()

    return _router


def reset_reply_router() -> None:
    """Reset the singleton ReplyRouter.

    Useful for testing or when the index needs to be reloaded.
    """
    global _router

    with _router_lock:
        _router = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Exceptions
    "RouterError",
    "IndexNotAvailableError",
    # Classes
    "RouteResult",
    "ReplyRouter",
    # Singleton functions
    "get_reply_router",
    "reset_reply_router",
]
