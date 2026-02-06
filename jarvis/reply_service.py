"""Reply Service - Unified service for generating replies.

Consolidates logic from:
- jarvis/router.py (Main RAG generation)
- jarvis/generation.py (Health-aware utilities)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from jarvis.classifiers.response_mobilization import (
    MobilizationResult,
    ResponsePressure,
    classify_response_pressure,
)
from jarvis.db import Contact, JarvisDB, get_db
from jarvis.embedding_adapter import CachedEmbedder, get_embedder
from jarvis.errors import ErrorCode, JarvisError
from jarvis.fallbacks import (
    get_fallback_reply_suggestions,
)
from jarvis.observability.metrics_router import (
    RoutingMetrics,
    get_routing_metrics_store,
    hash_query,
)
from jarvis.services.context_service import ContextService

if TYPE_CHECKING:
    from contracts.models import GenerationRequest
    from integrations.imessage.reader import ChatDBReader
    from jarvis.search.semantic_search import SemanticSearcher
    from models import MLXGenerator

logger = logging.getLogger(__name__)


class ReplyServiceError(JarvisError):
    """Raised when reply service operations fail."""

    default_message = "Reply service operation failed"
    default_code = ErrorCode.UNKNOWN


class ReplyService:
    """Unified service for generating AI replies.

    Coordinates RAG generation for high-quality single replies.
    """

    def __init__(
        self,
        db: JarvisDB | None = None,
        generator: MLXGenerator | None = None,
        imessage_reader: ChatDBReader | None = None,
    ) -> None:
        self._db = db
        self._generator = generator
        self._imessage_reader = imessage_reader
        self._semantic_searcher: SemanticSearcher | None = None
        self._context_service: ContextService | None = None
        self._lock = threading.RLock()

    @property
    def db(self) -> JarvisDB:
        if self._db is None:
            self._db = get_db()
            self._db.init_schema()
        return self._db

    @property
    def generator(self) -> MLXGenerator:
        with self._lock:
            if self._generator is None:
                from models import get_generator

                self._generator = get_generator(skip_templates=True)
            return self._generator

    @property
    def imessage_reader(self) -> ChatDBReader | None:
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
    def semantic_searcher(self) -> SemanticSearcher:
        with self._lock:
            if self._semantic_searcher is None and self.imessage_reader:
                from jarvis.search.semantic_search import get_semantic_searcher

                self._semantic_searcher = get_semantic_searcher(self.imessage_reader)
            return self._semantic_searcher

    @property
    def context_service(self) -> ContextService:
        """Get or create the context service."""
        with self._lock:
            if self._context_service is None:
                self._context_service = ContextService(
                    db=self.db,
                    imessage_reader=self.imessage_reader,
                    semantic_searcher=self.semantic_searcher,
                )
            return self._context_service

    def can_use_llm(self) -> tuple[bool, str]:
        """Check if LLM can be used based on system health."""
        from jarvis.generation import can_use_llm as check_health

        return check_health()

    def prepare_streaming_context(
        self,
        incoming: str,
        thread: list[str] | None = None,
        chat_id: str | None = None,
        instruction: str | None = None,
    ) -> tuple[GenerationRequest, dict[str, Any]]:
        """Prepare a GenerationRequest through the full pipeline for streaming.

        Runs all the same steps as the non-streaming path (health check, contact
        lookup, mobilization classification, RAG search, prompt assembly) but
        returns the request instead of generating. Designed to be called via
        asyncio.to_thread() before streaming tokens.

        Args:
            incoming: The incoming message text to respond to.
            thread: Optional recent conversation messages for context.
            chat_id: Optional chat ID for context and contact lookup.
            instruction: Optional user-provided instruction.

        Returns:
            Tuple of (GenerationRequest, metadata_dict).

        Raises:
            ReplyServiceError: If LLM health check fails.
        """
        # 1. Health check
        can_generate, health_reason = self.can_use_llm()
        if not can_generate:
            raise ReplyServiceError(f"LLM unavailable: {health_reason}")

        # 2. Get contact from chat_id
        contact = self.context_service.get_contact(None, chat_id)

        # 3. Classify mobilization
        mobilization = classify_response_pressure(incoming)

        # 4. Search similar examples
        search_results = self.context_service.search_examples(incoming, chat_id=chat_id)

        # 5. Build request through full pipeline
        request = self.build_generation_request(
            incoming=incoming,
            search_results=search_results,
            contact=contact,
            thread=thread,
            chat_id=chat_id,
            mobilization=mobilization,
            instruction=instruction,
        )

        # 6. Build metadata
        similarity = search_results[0]["similarity"] if search_results else 0.0
        metadata = {
            "confidence": "high" if mobilization.pressure == ResponsePressure.HIGH else "medium",
            "similarity_score": similarity,
            "mobilization_pressure": mobilization.pressure.value,
        }

        return request, metadata

    def generate_reply(
        self,
        incoming: str,
        contact: Contact | None = None,
        search_results: list[dict[str, Any]] | None = None,
        thread: list[str] | None = None,
        chat_id: str | None = None,
        mobilization: MobilizationResult | None = None,
        cached_embedder: CachedEmbedder | None = None,
    ) -> dict[str, Any]:
        """Generate a single best reply using RAG and LLM.

        This is the primary method for high-quality generation.

        Args:
            cached_embedder: Optional pre-warmed embedder to reuse. Avoids
                recomputing embeddings that the caller already computed.
        """
        routing_start = time.perf_counter()
        latency_ms: dict[str, float] = {}
        if cached_embedder is None:
            cached_embedder = CachedEmbedder(get_embedder())

        if not incoming or not incoming.strip():
            return {
                "type": "clarify",
                "response": "I received an empty message. Could you tell me what you need?",
                "confidence": "low",
                "reason": "empty_message",
            }

        # 1. Context and classification
        if mobilization is None:
            mobilization = classify_response_pressure(incoming)

        # 2. Search for similar examples
        if search_results is None:
            search_results = self.context_service.search_examples(incoming, chat_id=chat_id)

        # 3. Generate response
        can_generate, health_reason = self.can_use_llm()
        if not can_generate:
            logger.warning("Using fallback due to health: %s", health_reason)
            return {
                "type": "generated",
                "response": get_fallback_reply_suggestions()[0],
                "confidence": "medium",
                "source": "fallback",
                "reason": health_reason,
            }

        result = self._generate_llm_reply(
            incoming,
            search_results,
            contact,
            thread,
            chat_id=chat_id,
            mobilization=mobilization,
        )

        # 4. Finalize result
        latency_ms["total"] = (time.perf_counter() - routing_start) * 1000
        similarity = search_results[0]["similarity"] if search_results else 0.0

        # Record metrics
        self._record_metrics(
            incoming=incoming,
            decision="generate",
            similarity_score=similarity,
            latency_ms=latency_ms,
            cached_embedder=cached_embedder,
            vec_candidates=len(search_results),
            model_loaded=self.generator.is_loaded(),
        )

        return result

    # --- Internal Helpers ---

    def build_generation_request(
        self,
        incoming: str,
        search_results: list[dict[str, Any]],
        contact: Contact | None,
        thread: list[str] | None,
        chat_id: str | None,
        mobilization: MobilizationResult,
        instruction: str | None = None,
    ) -> GenerationRequest:
        """Build a GenerationRequest through the full pipeline.

        Does context building, relationship profile lookup, mobilization hinting,
        RAG prompt assembly, and GenerationRequest construction.

        Args:
            incoming: The incoming message text.
            search_results: Similar examples from vector search.
            contact: Contact info for personalization.
            thread: Recent conversation messages.
            chat_id: Chat ID for context lookup.
            mobilization: Response mobilization classification.
            instruction: Optional user-provided instruction override.

        Returns:
            A GenerationRequest ready for generate() or generate_stream().
        """
        from contracts.models import GenerationRequest
        from jarvis.prompts import build_rag_reply_prompt

        # Build context
        context_messages = []
        if thread:
            context_messages = thread[-10:]
        elif chat_id:
            context_messages = self.context_service.fetch_conversation_context(chat_id, limit=10)

        context = "\n".join(context_messages) + f"\n[Incoming]: {incoming}"

        # Relationship profile
        relationship_profile, contact_context = self.context_service.get_relationship_profile(
            contact, chat_id
        )

        # Build mobilization hint, allow user instruction to override
        if instruction is None:
            instruction = self._build_mobilization_hint(mobilization)

        similar_exchanges = [(r["trigger_text"], r["response_text"]) for r in search_results[:3]]

        prompt = build_rag_reply_prompt(
            context=context,
            last_message=incoming,
            contact_name=contact.display_name if contact else "them",
            similar_exchanges=similar_exchanges if similar_exchanges else None,
            relationship_profile=relationship_profile,
            contact_context=contact_context,
            instruction=instruction,
        )

        max_tokens = 20 if mobilization.pressure == ResponsePressure.NONE else 100

        return GenerationRequest(
            prompt=prompt,
            context_documents=[],  # Already baked into RAG prompt
            few_shot_examples=[],  # Already baked into RAG prompt
            max_tokens=max_tokens,
        )

    # Responses that signal the model is uncertain / lacks context
    _UNCERTAIN_SIGNALS = frozenset({"?", "??", "hm?", "what?", "huh?"})

    def _generate_llm_reply(
        self,
        incoming: str,
        search_results: list[dict[str, Any]],
        contact: Contact | None,
        thread: list[str] | None,
        chat_id: str | None,
        mobilization: MobilizationResult,
    ) -> dict[str, Any]:
        # Pre-generation gate: skip when no response is needed and no examples found
        if mobilization.pressure == ResponsePressure.NONE and not search_results:
            return {
                "type": "skip",
                "response": "",
                "confidence": "none",
                "reason": "no_response_needed",
            }

        try:
            request = self.build_generation_request(
                incoming, search_results, contact, thread, chat_id, mobilization
            )
            response = self.generator.generate(request)
            text = response.text.strip()

            # Post-generation gate: detect model uncertainty signal
            if text.lower() in self._UNCERTAIN_SIGNALS:
                return {
                    "type": "uncertain",
                    "response": text,
                    "confidence": "low",
                    "reason": "model_uncertain",
                }

            return {
                "type": "generated",
                "response": text,
                "confidence": "high"
                if mobilization.pressure == ResponsePressure.HIGH
                else "medium",
                "similarity_score": search_results[0]["similarity"] if search_results else 0.0,
            }
        except Exception as e:
            logger.exception("LLM generation failed: %s", e)
            return {
                "type": "clarify",
                "response": "I'm having trouble generating a response.",
                "confidence": "low",
                "reason": "generation_error",
            }

    @staticmethod
    def _build_mobilization_hint(mobilization: MobilizationResult) -> str:
        """Build a generation instruction hint based on response mobilization."""
        if mobilization.pressure == ResponsePressure.HIGH:
            if mobilization.response_type.value == "commitment":
                return "Respond with a clear commitment (accept, decline, or defer)."
            elif mobilization.response_type.value == "answer":
                return "Answer the question directly and clearly."
            elif mobilization.response_type.value == "confirmation":
                return "Confirm or deny clearly."
            return "Respond directly to their question."
        elif mobilization.pressure == ResponsePressure.MEDIUM:
            return "Respond with appropriate emotion and empathy."
        elif mobilization.pressure == ResponsePressure.LOW:
            return "Keep the response brief and casual."
        return "A brief acknowledgment is fine."

    def _record_metrics(self, **kwargs):
        try:
            metrics = RoutingMetrics(
                timestamp=time.time(), query_hash=hash_query(kwargs["incoming"]), **kwargs
            )
            get_routing_metrics_store().record(metrics)
        except Exception:
            pass


_service: ReplyService | None = None
_service_lock = threading.Lock()


def get_reply_service() -> ReplyService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = ReplyService()
    return _service
