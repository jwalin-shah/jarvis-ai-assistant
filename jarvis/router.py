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
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypedDict

from jarvis.classifiers.cascade import classify_with_cascade
from jarvis.classifiers.category_classifier import classify_category
from jarvis.classifiers.response_mobilization import (
    MobilizationResult,
    ResponsePressure,
)
from jarvis.contracts.pipeline import (
    CategoryType,
    ClassificationResult,
    GenerationResponse,
    IntentType,
    MessageContext,
    UrgencyLevel,
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
    def semantic_searcher(self) -> SemanticSearcher | None:
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

    @staticmethod
    def _to_intent_type(category: str) -> IntentType:
        mapping = {
            "question": IntentType.QUESTION,
            "request": IntentType.REQUEST,
            "statement": IntentType.STATEMENT,
            "emotion": IntentType.STATEMENT,
            "closing": IntentType.STATEMENT,
            "acknowledge": IntentType.STATEMENT,
        }
        return mapping.get(category, IntentType.UNKNOWN)

    @staticmethod
    def _to_confidence_label(confidence: float) -> str:
        if confidence >= 0.7:
            return "high"
        if confidence >= 0.45:
            return "medium"
        return "low"

    @staticmethod
    def _analyze_complexity(text: str) -> float:
        """Analyze the complexity of the input text (0.0 to 1.0).

        Factors: length, punctuation variety, word uniqueness.
        """
        if not text:
            return 0.0

        # Length score (logarithmic, caps at ~200 chars)
        length_score = min(1.0, math.log(len(text) + 1) / 5.3)

        # Punctuation complexity
        punctuation = set("?.!,:;")
        found_punc = [c for c in text if c in punctuation]
        punc_score = min(1.0, len(set(found_punc)) / 3.0)

        # Basic word variety
        words = text.split()
        if not words:
            variety_score = 0.0
        else:
            variety_score = len(set(words)) / len(words)

        # Weighted average
        complexity = (length_score * 0.5) + (punc_score * 0.3) + (variety_score * 0.2)
        return round(complexity, 2)

    def _build_classification_result(
        self,
        incoming: str,
        thread: list[str],
        mobilization: MobilizationResult,
    ) -> ClassificationResult:
        category_result = classify_category(
            incoming,
            context=thread,
            mobilization=mobilization,
        )

        if category_result.category == "closing":
            category = CategoryType.CLOSING
        elif category_result.category == "acknowledge":
            category = CategoryType.ACKNOWLEDGE
        else:
            category = CategoryType.FULL_RESPONSE

        if mobilization.pressure == ResponsePressure.HIGH:
            urgency = UrgencyLevel.HIGH
        elif mobilization.pressure == ResponsePressure.MEDIUM:
            urgency = UrgencyLevel.MEDIUM
        else:
            urgency = UrgencyLevel.LOW

        complexity = self._analyze_complexity(incoming)

        return ClassificationResult(
            intent=self._to_intent_type(category_result.category),
            category=category,
            urgency=urgency,
            confidence=min(1.0, (mobilization.confidence + category_result.confidence) / 2.0),
            requires_knowledge=category_result.category in {"question", "request"},
            metadata={
                "category_name": category_result.category,
                "category_confidence": category_result.confidence,
                "category_method": category_result.method,
                "mobilization_pressure": mobilization.pressure.value,
                "mobilization_response_type": mobilization.response_type.value,
                "mobilization_confidence": mobilization.confidence,
                "mobilization_method": mobilization.method,
                "complexity_score": complexity,
            },
        )

    @staticmethod
    def _to_legacy_response(response: GenerationResponse) -> dict[str, Any]:
        metadata = response.metadata
        result: dict[str, Any] = {
            "type": str(metadata.get("type", "generated")),
            "response": response.response,
            "confidence": ReplyRouter._to_confidence_label(response.confidence),
            "similarity_score": float(metadata.get("similarity_score", 0.0)),
            "similar_triggers": metadata.get("similar_triggers"),
            "reason": str(metadata.get("reason", "")),
        }
        category = metadata.get("category")
        if category:
            result["category"] = category
        return result

    def route_message(
        self,
        context: MessageContext,
        *,
        cached_embedder: CachedEmbedder | None = None,
    ) -> GenerationResponse:
        """Route a typed message context through classification and generation."""
        incoming = context.message_text.strip()

        if not incoming:
            return GenerationResponse(
                response="I received an empty message. Could you tell me what you need?",
                confidence=0.2,
                metadata={"type": "clarify", "reason": "empty_message", "similarity_score": 0.0},
            )

        thread = context.metadata.get("thread", [])
        if not isinstance(thread, list):
            thread = []
        thread = [msg for msg in thread if isinstance(msg, str)]

        contact_id_raw = context.metadata.get("contact_id")
        contact_id = contact_id_raw if isinstance(contact_id_raw, int) else None
        chat_id = context.chat_id or None

        if cached_embedder is None:
            cached_embedder = get_embedder()

        contact = self.context_service.get_contact(contact_id, chat_id)
        if contact and contact.display_name:
            context.metadata.setdefault("contact_name", contact.display_name)

        mobilization = classify_with_cascade(incoming)
        classification = self._build_classification_result(incoming, thread, mobilization)

        search_results = self.context_service.search_examples(
            incoming,
            chat_id=chat_id,
            contact_id=contact.id if contact else None,
            embedder=cached_embedder,
        )

        return self.reply_service.generate_reply(
            context=context,
            classification=classification,
            search_results=search_results,
            thread=thread,
            contact=contact,
            cached_embedder=cached_embedder,
        )

    def route(
        self,
        incoming: str,
        contact_id: int | None = None,
        thread: list[str] | None = None,
        chat_id: str | None = None,
    ) -> dict[str, Any]:
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
        cached_embedder = get_embedder()

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
        message_context = MessageContext(
            chat_id=chat_id or "",
            message_text=incoming,
            is_from_me=False,
            timestamp=datetime.utcnow(),
            metadata={
                "thread": thread or [],
                "contact_id": contact_id,
            },
        )

        model_loaded = self.generator.is_loaded()
        generate_start = time.perf_counter()
        response = self.route_message(
            message_context,
            cached_embedder=cached_embedder,
        )
        latency_ms["generate"] = (time.perf_counter() - generate_start) * 1000

        result = self._to_legacy_response(response)
        similarity = float(response.metadata.get("similarity_score", 0.0))
        response_type = str(response.metadata.get("type", result.get("type", "generated")))
        decision = (
            "clarify"
            if response_type in {"clarify", "uncertain", "skip", "fallback"}
            else "generate"
        )
        vec_candidates_raw = response.metadata.get("vec_candidates", 0)
        try:
            vec_candidates = int(vec_candidates_raw)
        except (TypeError, ValueError):
            vec_candidates = 0

        return record_and_return(
            result,
            similarity_score=similarity,
            vec_candidates=vec_candidates,
            model_loaded=model_loaded,
            decision=decision,
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


def route_message(context: MessageContext) -> GenerationResponse:
    """Route a typed message context using the shared router instance."""
    return get_reply_router().route_message(context)


def reset_reply_router() -> None:
    """Reset the singleton ReplyRouter.

    Useful for testing or when the index needs to be reloaded.
    """
    global _router

    with _router_lock:
        if _router is not None:
            _router.close()
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
    "route_message",
    "reset_reply_router",
]
