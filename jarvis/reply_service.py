"""Reply Service - Unified service for generating replies.

Consolidates logic from:
- jarvis/router.py (Main RAG generation)
- jarvis/generation.py (Health-aware utilities)
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from contracts.models import GenerationRequest as ModelGenerationRequest
from jarvis.classifiers.cascade import classify_with_cascade
from jarvis.classifiers.classification_result import build_classification_result
from jarvis.classifiers.response_mobilization import (
    MobilizationResult,
    ResponsePressure,
    ResponseType,
)
from jarvis.contracts.pipeline import (
    ClassificationResult,
    GenerationRequest,
    GenerationResponse,
    MessageContext,
)
from jarvis.core.exceptions import ErrorCode, JarvisError
from jarvis.core.generation.confidence import compute_example_diversity
from jarvis.core.generation.logging import log_custom_generation, persist_reply_log
from jarvis.core.generation.metrics import (
    record_routing_metrics,
)
from jarvis.db import Contact, JarvisDB, get_db
from jarvis.embedding_adapter import CachedEmbedder, get_embedder
from jarvis.observability.logging import log_event
from jarvis.prompts import (
    ACKNOWLEDGE_TEMPLATES,
    CLOSING_TEMPLATES,
    get_category_config,
)
from jarvis.reply_service_generation import (
    build_generation_request as build_generation_request_payload,
)
from jarvis.reply_service_generation import (
    dedupe_examples as dedupe_examples_payload,
)
from jarvis.reply_service_generation import (
    generate_llm_reply as generate_llm_reply_payload,
)
from jarvis.reply_service_generation import (
    get_cached_embeddings as get_cached_embeddings_payload,
)
from jarvis.reply_service_generation import (
    prepare_streaming_context as prepare_streaming_context_payload,
)
from jarvis.reply_service_generation import (
    to_model_generation_request as to_model_generation_request_payload,
)
from jarvis.reply_service_legacy import (
    get_routing_stats as get_routing_stats_payload,
)
from jarvis.reply_service_legacy import (
    route_legacy as route_legacy_payload,
)
from jarvis.reply_service_utils import (
    build_mobilization_hint,
    build_thread_context,
    max_tokens_for_pressure,
    pressure_from_classification,
    safe_float,
    to_legacy_response,
)
from jarvis.search.hybrid_search import get_hybrid_searcher
from jarvis.services.context_service import ContextService
from models.templates import TemplateMatcher

if TYPE_CHECKING:
    from integrations.imessage.reader import ChatDBReader
    from models import MLXGenerator

logger = logging.getLogger(__name__)


@dataclass
class PrecomputedContext:
    """Optional precomputed inputs to avoid redundant work in generate_reply /"""

    """prepare_streaming_context."""

    classification_result: ClassificationResult | None = None
    search_results: list[dict[str, Any]] | None = None
    contact: Contact | None = None
    cached_embedder: CachedEmbedder | None = None


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
        self._context_service: ContextService | None = None
        self._reranker: Any | None = None
        self._template_matcher: TemplateMatcher | None = None
        self._lock = threading.RLock()

    @property
    def db(self) -> JarvisDB:
        if self._db is None:
            self._db = get_db()
            self._db.init_schema()
        return self._db

    @property
    def generator(self) -> MLXGenerator:
        if self._generator is None:
            with self._lock:
                if self._generator is None:
                    from jarvis.model_warmer import get_warm_generator

                    self._generator = get_warm_generator(skip_templates=True)
        else:
            # Touch warmer if generator already exists
            from jarvis.model_warmer import get_model_warmer

            get_model_warmer().touch()
        return self._generator

    @property
    def imessage_reader(self) -> ChatDBReader | None:
        if self._imessage_reader is None:
            with self._lock:
                if self._imessage_reader is None:
                    try:
                        from integrations.imessage.reader import ChatDBReader

                        self._imessage_reader = ChatDBReader()
                    except (ImportError, OSError) as e:
                        logger.warning("Could not initialize iMessage reader: %s", e)
                        return None
        return self._imessage_reader

    @property
    def context_service(self) -> ContextService:
        """Get or create the context service."""
        if self._context_service is None:
            with self._lock:
                if self._context_service is None:
                    self._context_service = ContextService(
                        db=self.db,
                        imessage_reader=self.imessage_reader,
                    )
        return self._context_service

    @property
    def template_matcher(self) -> TemplateMatcher | None:
        """Get or create the semantic template matcher (lazy-loaded)."""
        if self._template_matcher is None:
            with self._lock:
                if self._template_matcher is None:
                    try:
                        self._template_matcher = TemplateMatcher()
                        logger.info(
                            "Semantic template matcher initialized with %d templates",
                            len(self._template_matcher.templates),
                        )
                    except Exception as e:
                        logger.warning("Could not initialize template matcher: %s", e)
                        return None
        return self._template_matcher

    @property
    def reranker(self) -> Any | None:
        """Get or create the cross-encoder reranker (lazy-loaded)."""
        if self._reranker is None:
            with self._lock:
                if self._reranker is None:
                    from models.cross_encoder import get_reranker

                    self._reranker = get_reranker()
        return self._reranker

    def can_use_llm(self) -> tuple[bool, str]:
        """Check if LLM can be used based on system health."""
        from jarvis.generation import can_use_llm as check_health

        return check_health()

    @staticmethod
    def _compute_example_diversity(search_results: list[dict[str, Any]]) -> float:
        """Compatibility shim for legacy callers/tests."""
        return compute_example_diversity(search_results)

    def _persist_reply_log(
        self,
        context: MessageContext,
        classification: ClassificationResult,
        search_results: list[dict[str, Any]] | None,
        result: GenerationResponse,
        latency_ms: dict[str, float],
    ) -> None:
        """Compatibility shim for legacy callers/tests."""
        persist_reply_log(self.db, context, classification, search_results, result, latency_ms)

    def log_custom_generation(
        self,
        chat_id: str | None,
        incoming_text: str,
        final_prompt: str,
        response_text: str,
        confidence: float = 0.5,
        category: str = "custom",
        rag_docs: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Compatibility shim for batch tasks and external integrations."""
        log_custom_generation(
            self.db,
            chat_id=chat_id,
            incoming_text=incoming_text,
            final_prompt=final_prompt,
            response_text=response_text,
            confidence=confidence,
            category=category,
            rag_docs=rag_docs,
            metadata=metadata,
        )

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        return safe_float(value, default=default)

    @staticmethod
    def _pressure_from_classification(classification: ClassificationResult) -> ResponsePressure:
        return pressure_from_classification(classification)

    @staticmethod
    def _max_tokens_for_pressure(pressure: ResponsePressure) -> int:
        return max_tokens_for_pressure(pressure)

    def _build_classification_result(
        self,
        incoming: str,
        thread: list[str],
        mobilization: MobilizationResult,
    ) -> ClassificationResult:
        return build_classification_result(incoming, thread, mobilization)

    @staticmethod
    def _build_thread_context(conversation_messages: list[Any]) -> list[str]:
        return build_thread_context(conversation_messages)

    @staticmethod
    def _to_legacy_response(response: GenerationResponse) -> dict[str, Any]:
        return to_legacy_response(
            response_text=response.response,
            confidence=response.confidence,
            metadata=response.metadata,
        )

    def route_legacy(
        self,
        incoming: str,
        contact_id: int | None = None,
        thread: list[str] | None = None,
        chat_id: str | None = None,
        conversation_messages: list[Any] | None = None,
        context: MessageContext | None = None,
    ) -> dict[str, Any]:
        return route_legacy_payload(
            self,
            incoming=incoming,
            contact_id=contact_id,
            thread=thread,
            chat_id=chat_id,
            conversation_messages=conversation_messages,
            context=context,
        )

    def get_routing_stats(self) -> dict[str, Any]:
        return get_routing_stats_payload(self)

    def prepare_streaming_context(
        self,
        incoming: str,
        thread: list[str] | None = None,
        chat_id: str | None = None,
        instruction: str | None = None,
        classification_result: ClassificationResult | None = None,
        contact: Contact | None = None,
        search_results: list[dict[str, Any]] | None = None,
        cached_embedder: CachedEmbedder | None = None,
        *,
        precomputed: PrecomputedContext | None = None,
    ) -> tuple[ModelGenerationRequest, dict[str, Any]]:
        """Prepare a model GenerationRequest through the typed pipeline for streaming.

        Runs all the same steps as the non-streaming path (health check, contact
        lookup, classification, RAG search, prompt assembly) but returns the
        model request instead of generating. Pass precomputed to reuse results.
        """
        if precomputed:
            classification_result = precomputed.classification_result or classification_result
            contact = precomputed.contact if precomputed.contact is not None else contact
            search_results = (
                precomputed.search_results
                if precomputed.search_results is not None
                else search_results
            )
            cached_embedder = precomputed.cached_embedder or cached_embedder
        return prepare_streaming_context_payload(
            self,
            incoming=incoming,
            thread=thread,
            chat_id=chat_id,
            instruction=instruction,
            classification_result=classification_result,
            contact=contact,
            search_results=search_results,
            cached_embedder=cached_embedder,
            reply_error_cls=ReplyServiceError,
        )

    def generate_reply(
        self,
        context: MessageContext,
        classification: ClassificationResult | None = None,
        search_results: list[dict[str, Any]] | None = None,
        thread: list[str] | None = None,
        contact: Contact | None = None,
        cached_embedder: CachedEmbedder | None = None,
        instruction: str | None = None,
        *,
        precomputed: PrecomputedContext | None = None,
    ) -> GenerationResponse:
        """Generate a reply from contract types.

        Orchestrates: validation -> template shortcut -> context search -> LLM gen -> metrics.
        Pass precomputed to reuse classification, search, contact, or embedder and
        skip redundant work.
        """
        if precomputed:
            classification = precomputed.classification_result or classification
            search_results = (
                precomputed.search_results
                if precomputed.search_results is not None
                else search_results
            )
            contact = precomputed.contact if precomputed.contact is not None else contact
            cached_embedder = precomputed.cached_embedder or cached_embedder
        routing_start = time.perf_counter()
        latency_ms: dict[str, float] = {}

        if cached_embedder is None:
            cached_embedder = get_embedder()

        incoming = context.message_text.strip()
        if not incoming or not incoming.strip():
            return self._empty_message_response()

        chat_id = context.chat_id or None
        thread_messages = self._resolve_thread(thread, context)

        if classification is None:
            mobilization = classify_with_cascade(incoming)
            classification = self._build_classification_result(
                incoming, thread_messages, mobilization
            )

        log_event(
            logger,
            "reply.generate.start",
            level=logging.DEBUG,
            chat_id=context.chat_id or "",
            category=str(classification.category.value),
        )

        category_name = str(
            classification.metadata.get("category_name", classification.category.value)
        )
        category_config = get_category_config(category_name)

        # Try semantic template matching first (context-aware, 74 templates)
        # This uses embeddings to match against universal texting patterns
        semantic_result = self._try_semantic_template(incoming, embedder=cached_embedder)
        if semantic_result:
            latency_ms["total"] = (time.perf_counter() - routing_start) * 1000
            self._persist_reply_log(context, classification, None, semantic_result, latency_ms)
            return semantic_result

        # Fall back to simple templates for acknowledge/closing categories
        if category_config.skip_slm:
            result = self._template_response(category_name, routing_start)
            latency_ms["total"] = (time.perf_counter() - routing_start) * 1000
            self._persist_reply_log(context, classification, None, result, latency_ms)
            return result

        logger.debug("   [debug] Starting RAG search...")
        search_results, latency_ms = self._search_context(
            search_results,
            incoming,
            chat_id,
            contact,
            cached_embedder,
            latency_ms,
        )
        logger.debug("   [debug] RAG search complete (%d candidates)", len(search_results))

        can_generate, health_reason = self.can_use_llm()
        if not can_generate:
            log_event(logger, "reply.fallback", level=logging.WARNING, reason=health_reason)
            result = GenerationResponse(
                response="",
                confidence=0.0,
                metadata={
                    "type": "fallback",
                    "reason": health_reason,
                    "similarity_score": 0.0,
                    "vec_candidates": len(search_results),
                },
            )
            latency_ms["total"] = (time.perf_counter() - routing_start) * 1000
            self._persist_reply_log(context, classification, search_results, result, latency_ms)
            return result

        logger.debug("   [debug] Starting LLM generation...")
        result, latency_ms = self._generate_response(
            context,
            classification,
            search_results,
            contact,
            thread_messages,
            cached_embedder,
            latency_ms,
            instruction=instruction,
        )
        logger.debug("   [debug] LLM generation complete")

        latency_ms["total"] = (time.perf_counter() - routing_start) * 1000
        self._log_and_record_metrics(
            result,
            category_name,
            incoming,
            search_results,
            latency_ms,
            cached_embedder,
        )

        # Persist full generation log for traceability
        persist_reply_log(self.db, context, classification, search_results, result, latency_ms)

        return result

    def _empty_message_response(self) -> GenerationResponse:
        """Return a clarification response for empty messages."""
        return GenerationResponse(
            response="I received an empty message. Could you tell me what you need?",
            confidence=0.2,
            metadata={"type": "clarify", "reason": "empty_message", "similarity_score": 0.0},
        )

    def _resolve_thread(
        self,
        thread: list[str] | None,
        context: MessageContext,
    ) -> list[str]:
        """Extract thread messages from explicit param or context metadata."""
        if thread is not None:
            return thread
        metadata_thread = context.metadata.get("thread", [])
        if isinstance(metadata_thread, list):
            return [msg for msg in metadata_thread if isinstance(msg, str)]
        return []

    def _try_semantic_template(
        self,
        incoming: str,
        embedder: CachedEmbedder | None = None,
    ) -> GenerationResponse | None:
        """Try to match incoming message with semantic templates.

        Uses sentence embeddings to find the best matching template from
        models/template_defaults.py. This provides context-aware responses
        based on 74 universal texting patterns.

        Args:
            incoming: The incoming message text
            embedder: Optional embedder for cache cohesion

        Returns:
            GenerationResponse if match found, None otherwise
        """
        if not self.template_matcher:
            return None

        try:
            match = self.template_matcher.match(incoming, embedder=embedder, track_analytics=True)

            if match and match.similarity >= 0.85:  # High confidence threshold
                log_event(
                    logger,
                    "reply.semantic_template",
                    template=match.template.name,
                    similarity=round(match.similarity, 3),
                    pattern=match.matched_pattern,
                )
                return GenerationResponse(
                    response=match.template.response,
                    confidence=min(0.95, match.similarity),
                    metadata={
                        "type": "semantic_template",
                        "template_name": match.template.name,
                        "matched_pattern": match.matched_pattern,
                        "similarity": match.similarity,
                        "reason": f"template_match:{match.template.name}",
                    },
                )
        except Exception as e:
            logger.debug("Semantic template matching failed: %s", e)

        return None

    def _template_response(
        self,
        category_name: str,
        routing_start: float,
    ) -> GenerationResponse:
        """Return a template response for categories that skip the SLM."""
        if category_name == "closing":
            template_response = random.choice(CLOSING_TEMPLATES)  # nosec B311
        else:
            template_response = random.choice(ACKNOWLEDGE_TEMPLATES)  # nosec B311

        log_event(
            logger,
            "reply.skip_slm",
            category=category_name,
            latency_ms=round((time.perf_counter() - routing_start) * 1000, 1),
        )
        return GenerationResponse(
            response=template_response,
            confidence=0.95,
            metadata={
                "type": category_name,
                "reason": f"category={category_name}",
                "category": category_name,
                "similarity_score": 0.0,
                "vec_candidates": 0,
            },
        )

    def _search_context(
        self,
        search_results: list[dict[str, Any]] | None,
        incoming: str,
        chat_id: str | None,
        contact: Contact | None,
        cached_embedder: CachedEmbedder,
        latency_ms: dict[str, float],
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """Run hybrid context search if results not already provided."""
        if search_results is None:
            search_start = time.perf_counter()
            hybrid_searcher = get_hybrid_searcher()
            # Hybrid search already uses vec_searcher internally
            search_results = hybrid_searcher.search(query=incoming, limit=5, rerank=True)
            latency_ms["context_search"] = (time.perf_counter() - search_start) * 1000
        return search_results, latency_ms

    def _generate_response(
        self,
        context: MessageContext,
        classification: ClassificationResult,
        search_results: list[dict[str, Any]],
        contact: Contact | None,
        thread_messages: list[str],
        cached_embedder: CachedEmbedder,
        latency_ms: dict[str, float],
        instruction: str | None = None,
    ) -> tuple[GenerationResponse, dict[str, float]]:
        """Build generation request and run LLM inference."""
        gen_start = time.perf_counter()
        request = self.build_generation_request(
            context=context,
            classification=classification,
            search_results=search_results,
            contact=contact,
            thread=thread_messages,
            instruction=instruction,
            cached_embedder=cached_embedder,
        )
        result = self._generate_llm_reply(request, search_results, thread_messages)
        latency_ms["generation"] = (time.perf_counter() - gen_start) * 1000
        return result, latency_ms

    def _log_and_record_metrics(
        self,
        result: GenerationResponse,
        category_name: str,
        incoming: str,
        search_results: list[dict[str, Any]],
        latency_ms: dict[str, float],
        cached_embedder: CachedEmbedder,
    ) -> None:
        """Log completion event and record routing metrics."""
        log_event(
            logger,
            "reply.generate.complete",
            category=category_name,
            confidence=result.confidence,
            total_ms=round(latency_ms["total"], 1),
            generation_ms=round(latency_ms["generation"], 1),
            vec_candidates=len(search_results),
        )
        similarity = self._safe_float(
            result.metadata.get(
                "similarity_score",
                search_results[0].get("similarity", 0.0) if search_results else 0.0,
            ),
            default=0.0,
        )

        record_routing_metrics(
            incoming=incoming,
            decision="generate",
            similarity_score=similarity,
            latency_ms=latency_ms,
            cached_embedder=cached_embedder,
            vec_candidates=len(search_results),
            model_loaded=self.generator.is_loaded(),
        )

    # --- Internal Helpers ---

    def _fetch_contact_facts(self, context: MessageContext, chat_id: str) -> None:
        try:
            from jarvis.contacts.fact_index import search_relevant_facts
            from jarvis.prompts import format_facts_for_prompt

            incoming_text = context.message_text or ""
            facts = search_relevant_facts(incoming_text, chat_id, limit=5)
            context.metadata["contact_facts"] = format_facts_for_prompt(facts)
        except Exception as e:
            logger.debug(f"Optional fact fetch failed: {e}")

    def _fetch_graph_context(self, context: MessageContext, chat_id: str) -> None:
        try:
            from jarvis.graph.context import get_graph_context

            graph_ctx = get_graph_context(contact_id=chat_id, chat_id=chat_id)
            if graph_ctx:
                context.metadata["relationship_graph"] = graph_ctx
        except Exception as e:
            logger.debug(f"Optional graph context fetch failed: {e}")

    def _resolve_instruction(
        self,
        instruction: str | None,
        category_name: str,
        category_config: Any,
        classification: ClassificationResult,
    ) -> str | None:
        from jarvis.prompts import get_optimized_instruction

        if instruction is None:
            optimized_instruction = get_optimized_instruction(category_name)
            if optimized_instruction:
                instruction = optimized_instruction
            elif category_config.system_prompt:
                instruction = category_config.system_prompt
            else:
                pressure = self._pressure_from_classification(classification)
                response_type_value = str(
                    classification.metadata.get(
                        "mobilization_response_type",
                        ResponseType.OPTIONAL.value,
                    )
                )
                try:
                    response_type = ResponseType(response_type_value)
                except ValueError:
                    response_type = ResponseType.OPTIONAL
                instruction = self._build_mobilization_hint(
                    MobilizationResult(
                        pressure=pressure,
                        response_type=response_type,
                        confidence=classification.confidence,
                        features={},
                        method="contract_bridge",
                    )
                )
        return instruction

    def build_generation_request(
        self,
        context: MessageContext,
        classification: ClassificationResult,
        search_results: list[dict[str, Any]],
        contact: Contact | None,
        thread: list[str] | None = None,
        instruction: str | None = None,
        cached_embedder: CachedEmbedder | None = None,
    ) -> GenerationRequest:
        """Build a typed GenerationRequest with context, classification, and RAG docs."""
        return build_generation_request_payload(
            self,
            context=context,
            classification=classification,
            search_results=search_results,
            contact=contact,
            thread=thread,
            instruction=instruction,
            cached_embedder=cached_embedder,
        )

    def _to_model_generation_request(self, request: GenerationRequest) -> ModelGenerationRequest:
        return to_model_generation_request_payload(self, request)

    def _dedupe_examples(
        self,
        examples: list[tuple[str, str]],
        embedder: CachedEmbedder,
        rerank_scores: list[float] | None = None,
    ) -> list[tuple[str, str]]:
        return dedupe_examples_payload(
            self,
            examples=examples,
            embedder=embedder,
            rerank_scores=rerank_scores,
        )

    def _get_cached_embeddings(
        self,
        texts: list[str],
        embedder: CachedEmbedder,
    ) -> Any:
        return get_cached_embeddings_payload(self, texts=texts, embedder=embedder)

    def _generate_llm_reply(
        self,
        request: GenerationRequest,
        search_results: list[dict[str, Any]],
        thread: list[str] | None,
    ) -> GenerationResponse:
        return generate_llm_reply_payload(
            self,
            request=request,
            search_results=search_results,
            thread=thread,
        )

    @staticmethod
    def _build_mobilization_hint(mobilization: MobilizationResult) -> str:
        return build_mobilization_hint(mobilization)


_service: ReplyService | None = None
_service_lock = threading.Lock()


def get_reply_service() -> ReplyService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = ReplyService()
    return _service
