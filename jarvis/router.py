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
from typing import TYPE_CHECKING, Any

from jarvis.classifiers.response_mobilization import (
    MobilizationResult,
    ResponsePressure,
    ResponseType,
    classify_response_pressure,
)
from jarvis.contacts.contact_profile import MIN_MESSAGES_FOR_PROFILE, get_contact_profile
from jarvis.contacts.contact_profile_context import (
    ContactProfileContext,
    is_contact_profile_context_enabled,
)
from jarvis.db import Contact, JarvisDB, get_db
from jarvis.embedding_adapter import CachedEmbedder, get_embedder
from jarvis.errors import ErrorCode, JarvisError
from jarvis.metrics_router import RoutingMetrics, get_routing_metrics_store, hash_query

if TYPE_CHECKING:
    from integrations.imessage.reader import ChatDBReader
    from jarvis.index import TriggerIndexSearcher
    from models import MLXGenerator

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration (kept for backwards compatibility)
# =============================================================================

QUICK_REPLY_THRESHOLD = 0.95  # Legacy - no longer used for routing
CONTEXT_THRESHOLD = 0.65  # Legacy - no longer used for routing
GENERATE_THRESHOLD = 0.45  # Legacy - no longer used for routing

# FAISS search threshold - minimum similarity for retrieving examples
_SEARCH_THRESHOLD = 0.3


# =============================================================================
# Exceptions
# =============================================================================


class RouterError(JarvisError):
    """Raised when routing operations fail."""

    default_message = "Router operation failed"
    default_code = ErrorCode.UNKNOWN


class IndexNotAvailableError(RouterError):
    """Raised when FAISS index is not available."""

    default_message = "FAISS index not available. Run 'jarvis db build-index' first."


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RouteResult:
    """Result of routing an incoming message.

    Attributes:
        response: The response text.
        type: Response type ('generated', 'clarify').
        confidence: Confidence level ('high', 'medium', 'low').
        similarity_score: Best similarity score from FAISS search.
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
        index_searcher: TriggerIndexSearcher | None = None,
        generator: MLXGenerator | None = None,
        imessage_reader: ChatDBReader | None = None,
    ) -> None:
        """Initialize the router.

        Args:
            db: Database instance for contacts and pairs. Uses default if None.
            index_searcher: FAISS index searcher. Created lazily if None.
            generator: MLX generator for LLM responses. Created lazily if None.
            imessage_reader: iMessage reader for fetching conversation history.
                Created lazily if None.
        """
        self._db = db
        self._index_searcher = index_searcher
        self._generator = generator
        self._imessage_reader = imessage_reader
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
    def generator(self) -> MLXGenerator:
        """Get or create the MLX generator."""
        if self._generator is None:
            with self._lock:
                if self._generator is None:
                    from models import get_generator

                    self._generator = get_generator(skip_templates=True)
        return self._generator

    @property
    def imessage_reader(self) -> ChatDBReader | None:
        """Get or create the iMessage reader for fetching conversation history."""
        if self._imessage_reader is None:
            with self._lock:
                if self._imessage_reader is None:
                    try:
                        from integrations.imessage.reader import ChatDBReader

                        self._imessage_reader = ChatDBReader()
                    except Exception as e:
                        logger.warning("Could not initialize iMessage reader: %s", e)
                        return None
        return self._imessage_reader

    def _fetch_conversation_context(self, chat_id: str, limit: int = 10) -> list[str]:
        """Fetch recent conversation history from iMessage.

        Args:
            chat_id: The chat/conversation ID.
            limit: Maximum number of messages to fetch.

        Returns:
            List of formatted message strings for context.
        """
        if not self.imessage_reader:
            return []

        try:
            messages = self.imessage_reader.get_messages(chat_id, limit=limit)
            if not messages:
                return []

            # Format messages for context (newest first, so reverse for chronological)
            context_messages = []
            for msg in reversed(messages):
                sender = "You" if msg.is_from_me else (msg.sender_name or msg.sender or "Them")
                text = msg.text or ""
                if text:
                    context_messages.append(f"[{sender}]: {text}")

            return context_messages

        except Exception as e:
            logger.warning("Failed to fetch conversation context: %s", e)
            return []

    def _record_routing_metrics(
        self,
        incoming: str,
        decision: str,
        similarity_score: float,
        latency_ms: dict[str, float],
        cached_embedder: CachedEmbedder,
        faiss_candidates: int,
        model_loaded: bool,
    ) -> None:
        try:
            metrics = RoutingMetrics(
                timestamp=time.time(),
                query_hash=hash_query(incoming),
                latency_ms=latency_ms,
                embedding_computations=cached_embedder.embedding_computations,
                faiss_candidates=faiss_candidates,
                routing_decision=decision,
                similarity_score=similarity_score,
                cache_hit=cached_embedder.cache_hit,
                model_loaded=model_loaded,
            )
            get_routing_metrics_store().record(metrics)
        except Exception as e:
            logger.debug("Routing metrics write failed: %s", e)

    def _get_contact(self, contact_id: int | None, chat_id: str | None) -> Contact | None:
        """Look up contact by contact_id or chat_id."""
        if contact_id:
            return self.db.get_contact(contact_id)
        if chat_id:
            return self.db.get_contact_by_chat_id(chat_id)
        return None

    def _search_examples(
        self,
        incoming: str,
        cached_embedder: CachedEmbedder,
    ) -> list[dict[str, Any]]:
        """Search FAISS index for similar triggers with error handling."""
        try:
            return self.index_searcher.search_with_pairs(
                query=incoming,
                k=5,
                threshold=_SEARCH_THRESHOLD,
                prefer_recent=True,
                embedder=cached_embedder,
            )
        except FileNotFoundError:
            logger.warning("FAISS index not found, generating without examples")
            return []
        except Exception as e:
            logger.exception("Error searching index: %s", e)
            return []

    def _build_mobilization_hint(self, mobilization: MobilizationResult) -> str:
        """Build a prompt instruction hint from mobilization classification."""
        pressure = mobilization.pressure
        response_type = mobilization.response_type

        if pressure == ResponsePressure.HIGH:
            if response_type == ResponseType.COMMITMENT:
                return (
                    "This message is a request or invitation requiring a commitment "
                    "(accept, decline, or defer). Respond appropriately."
                )
            if response_type == ResponseType.ANSWER:
                return "This is a direct question. Provide a clear, specific answer."
            if response_type == ResponseType.CONFIRMATION:
                return "This seeks confirmation. Respond with a clear yes/no."
            return "This message requires a substantive response."

        if pressure == ResponsePressure.MEDIUM:
            if response_type == ResponseType.EMOTIONAL:
                return (
                    "This message shares emotional news or an experience. "
                    "Respond with appropriate empathy or enthusiasm."
                )
            if response_type == ResponseType.ALIGNMENT:
                return "This expresses an opinion or assessment. Respond naturally."
            return "This message warrants a thoughtful response."

        if pressure == ResponsePressure.NONE:
            return (
                "This is a backchannel or closing. Keep response very brief "
                "(1-3 words or an emoji)."
            )

        # LOW pressure
        return "This is a casual statement. Respond naturally and briefly."

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
        cached_embedder = CachedEmbedder(get_embedder())

        # Precompute embedding (reused by FAISS search)
        if incoming and incoming.strip():
            embed_start = time.perf_counter()
            cached_embedder.encode(incoming)
            latency_ms["embedding_precompute"] = (time.perf_counter() - embed_start) * 1000

        logger.info("=" * 60)
        logger.info("ROUTE START | input: %s", incoming[:80] if incoming else "(empty)")

        def record_and_return(
            result: dict[str, Any],
            similarity_score: float,
            faiss_candidates: int = 0,
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
                faiss_candidates=faiss_candidates,
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

        # Empty message check
        if not incoming or not incoming.strip():
            result = self._clarify_response(
                "I received an empty message. Could you tell me what you need?",
                reason="empty_message",
            )
            return record_and_return(result, similarity_score=0.0, decision="clarify")

        # Get contact
        contact = self._get_contact(contact_id, chat_id)

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

        # FAISS search for similar examples (always, low threshold)
        search_start = time.perf_counter()
        search_results = self._search_examples(incoming, cached_embedder)
        latency_ms["faiss_search"] = (time.perf_counter() - search_start) * 1000

        # Generate response
        model_loaded = self.generator.is_loaded()
        generate_start = time.perf_counter()
        result = self._generate_response(
            incoming,
            search_results,
            contact,
            thread,
            chat_id=chat_id,
            mobilization=mobilization,
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
            faiss_candidates=len(search_results),
            model_loaded=model_loaded,
            decision="generate",
        )

    def _generate_response(
        self,
        incoming: str,
        search_results: list[dict[str, Any]],
        contact: Contact | None,
        thread: list[str] | None,
        chat_id: str | None = None,
        mobilization: MobilizationResult | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Generate an LLM response with context from similar patterns.

        Args:
            incoming: The incoming message.
            search_results: Similar patterns from FAISS.
            contact: Contact for personalization.
            thread: Recent conversation context (if already available).
            chat_id: Chat ID for fetching conversation history from iMessage.
            mobilization: Response mobilization result for prompt guidance.
            reason: Optional reason why generation was chosen.

        Returns:
            Routing result dict with generated response.
        """
        from contracts.models import GenerationRequest
        from jarvis.prompts import build_rag_reply_prompt

        # Build few-shot examples from similar patterns
        similar_exchanges = []
        similar_triggers = []
        for result in search_results[:3]:
            similar_exchanges.append((result["trigger_text"], result["response_text"]))
            similar_triggers.append(result["trigger_text"])

        # Build conversation context - prefer passed thread, fetch from iMessage if not available
        context_messages = []
        if thread:
            context_messages = thread[-10:]
        elif chat_id:
            context_messages = self._fetch_conversation_context(chat_id, limit=10)
            if context_messages:
                logger.debug("Fetched %d messages from iMessage for context", len(context_messages))

        # Format context for prompt
        context = ""
        if context_messages:
            context = "\n".join(context_messages)
        context += f"\n[Incoming]: {incoming}"

        # Get relationship profile for the contact
        relationship_profile = None
        contact_context: ContactProfileContext | None = None
        if contact:
            profile = None
            if is_contact_profile_context_enabled():
                profile = get_contact_profile(chat_id) if chat_id else None
                if profile is None and contact.id:
                    profile = get_contact_profile(str(contact.id))
                if profile and profile.message_count >= MIN_MESSAGES_FOR_PROFILE:
                    contact_context = ContactProfileContext.from_contact_profile(profile)
                    relationship_profile = contact_context.to_prompt_payload()
                    logger.debug(
                        "Using contact profile context for %s (formality=%.2f, %d messages)",
                        contact.display_name,
                        profile.formality_score,
                        profile.message_count,
                    )
            if contact_context is None:
                relationship_profile = {
                    "tone": contact.style_notes or "casual",
                    "relationship": contact.relationship or "friend",
                }

        # Build mobilization hint for the prompt
        instruction = None
        if mobilization:
            instruction = self._build_mobilization_hint(mobilization)

        # Build the prompt
        prompt = build_rag_reply_prompt(
            context=context,
            last_message=incoming,
            contact_name=contact.display_name if contact else "them",
            similar_exchanges=similar_exchanges if similar_exchanges else None,
            relationship_profile=relationship_profile,
            contact_context=contact_context,
            instruction=instruction,
        )

        # Determine max_tokens based on mobilization
        max_tokens = 100
        if mobilization and mobilization.pressure == ResponsePressure.NONE:
            max_tokens = 20  # Backchannels should be very brief

        # Generate with the model
        try:
            request = GenerationRequest(
                prompt=prompt,
                context_documents=[context] if context else [],
                few_shot_examples=similar_exchanges,
                max_tokens=max_tokens,
            )

            response = self.generator.generate(request)
            generated_text = response.text.strip()

            # Remove common formal greetings that don't match texting style
            formal_greetings = (
                "hey!",
                "hi!",
                "hello!",
                "hey there!",
                "hi there!",
                "hello there!",
            )
            for greeting in formal_greetings:
                if generated_text.lower().startswith(greeting):
                    generated_text = generated_text[len(greeting) :].strip()
                    if generated_text:
                        generated_text = generated_text[0].upper() + generated_text[1:]
                    break

            # Trim overly long responses
            avg_msg_len = 50
            if contact_context:
                avg_msg_len = contact_context.avg_message_length
            elif relationship_profile:
                avg_msg_len = relationship_profile.get("avg_message_length", 50)
            expected_length = int(avg_msg_len) * 2
            if len(generated_text) > max(80, expected_length) and ". " in generated_text:
                sentences = generated_text.split(". ")
                trimmed = []
                current_len = 0
                for s in sentences:
                    if current_len + len(s) > expected_length:
                        break
                    trimmed.append(s)
                    current_len += len(s) + 2
                if trimmed:
                    generated_text = ". ".join(trimmed)
                    if not generated_text.endswith((".", "!", "?")):
                        generated_text += "."

            similarity = search_results[0]["similarity"] if search_results else 0.0

            result: dict[str, Any] = {
                "type": "generated",
                "response": generated_text,
                "confidence": "medium",
                "similarity_score": similarity,
                "contact_style": contact.style_notes if contact else None,
                "similar_triggers": similar_triggers if similar_triggers else None,
            }
            if reason:
                result["generation_reason"] = reason
            return result

        except Exception as e:
            logger.exception("Generation failed: %s", e)
            return self._clarify_response(
                "I'm having trouble generating a response. Could you give me more details?",
                reason="generation_error",
            )

    def _clarify_response(
        self,
        message: str,
        reason: str = "unknown",
    ) -> dict[str, Any]:
        """Create a clarification response.

        Args:
            message: The clarification message.
            reason: Why clarification was needed.

        Returns:
            Routing result dict.
        """
        return {
            "type": "clarify",
            "response": message,
            "confidence": "low",
            "similarity_score": 0.0,
            "reason": reason,
        }

    def route_multi_option(
        self,
        incoming: str,
        contact_id: int | None = None,
        chat_id: str | None = None,
        force_multi: bool = False,
    ) -> dict[str, Any]:
        """Route with multi-option generation for commitment questions.

        Simplified: delegates to route() and wraps result with multi-option keys.

        Args:
            incoming: The incoming message text.
            contact_id: Optional contact ID for personalization.
            chat_id: Optional chat ID for context lookup.
            force_multi: If True, force multi-option even for non-commitment.

        Returns:
            Dict with routing result including:
            - is_commitment: Whether this is a commitment question
            - options: List of {type, response, confidence} for commitment questions
            - suggestions: List of response texts (backward compatible)
            - trigger_da: Classified trigger type
        """
        result = self.route(
            incoming=incoming,
            contact_id=contact_id,
            chat_id=chat_id,
        )
        result["is_commitment"] = False
        result["options"] = []
        result["suggestions"] = [result["response"]]
        result["trigger_da"] = None
        return result

    def get_routing_stats(self) -> dict[str, Any]:
        """Get statistics about the router's index and database.

        Returns:
            Dict with index and database statistics.
        """
        stats = {
            "db_stats": self.db.get_stats(),
            "index_available": False,
        }

        try:
            active_index = self.db.get_active_index()
            if active_index:
                stats["index_available"] = True
                stats["index_version"] = active_index.version_id
                stats["index_vectors"] = active_index.num_vectors
                stats["index_model"] = active_index.model_name
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
    # Configuration (legacy, kept for backwards compatibility)
    "QUICK_REPLY_THRESHOLD",
    "CONTEXT_THRESHOLD",
    "GENERATE_THRESHOLD",
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
