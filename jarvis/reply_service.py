"""Reply Service - Unified service for generating replies.

Consolidates logic from:
- jarvis/router.py (Main RAG generation)
- jarvis/generation.py (Health-aware utilities)
"""

from __future__ import annotations

import hashlib
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from contracts.models import GenerationRequest as ModelGenerationRequest
from jarvis.classifiers.cascade import classify_with_cascade
from jarvis.classifiers.classification_result import build_classification_result
from jarvis.classifiers.response_mobilization import (
    MobilizationResult,
    ResponsePressure,
    ResponseType,
)
from jarvis.contracts.pipeline import (
    CategoryType,
    ClassificationResult,
    GenerationRequest,
    GenerationResponse,
    MessageContext,
    RAGDocument,
    UrgencyLevel,
)
from jarvis.core.generation.confidence import (
    UNCERTAIN_SIGNALS,
    compute_confidence,
    compute_example_diversity,
)
from jarvis.core.generation.logging import log_custom_generation, persist_reply_log
from jarvis.core.generation.metrics import (
    record_routing_metrics,
)
from jarvis.db import Contact, JarvisDB, get_db
from jarvis.embedding_adapter import CachedEmbedder, get_embedder
from jarvis.errors import ErrorCode, JarvisError
from jarvis.observability.logging import log_event
from jarvis.prompts import (
    ACKNOWLEDGE_TEMPLATES,
    CLOSING_TEMPLATES,
    build_prompt_from_request,
    get_category_config,
)
from jarvis.search.hybrid_search import get_hybrid_searcher
from jarvis.services.context_service import ContextService

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
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _pressure_from_classification(classification: ClassificationResult) -> ResponsePressure:
        pressure_raw = classification.metadata.get("mobilization_pressure")
        if isinstance(pressure_raw, str):
            try:
                return ResponsePressure(pressure_raw)
            except ValueError:
                pass
        if classification.category in {
            CategoryType.ACKNOWLEDGE,
            CategoryType.CLOSING,
            CategoryType.OFF_TOPIC,
        }:
            return ResponsePressure.NONE
        if classification.urgency == UrgencyLevel.HIGH:
            return ResponsePressure.HIGH
        if classification.urgency == UrgencyLevel.MEDIUM:
            return ResponsePressure.MEDIUM
        return ResponsePressure.LOW

    @staticmethod
    def _max_tokens_for_pressure(pressure: ResponsePressure) -> int:
        return 20 if pressure == ResponsePressure.NONE else 40

    def _build_classification_result(
        self,
        incoming: str,
        thread: list[str],
        mobilization: MobilizationResult,
    ) -> ClassificationResult:
        return build_classification_result(incoming, thread, mobilization)

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
        can_generate, health_reason = self.can_use_llm()
        if not can_generate:
            raise ReplyServiceError(f"LLM unavailable: {health_reason}")

        normalized_incoming = incoming.strip()
        thread_messages = [msg for msg in (thread or []) if isinstance(msg, str)]

        if cached_embedder is None:
            cached_embedder = get_embedder()

        if contact is None:
            contact = self.context_service.get_contact(None, chat_id)

        if classification_result is None:
            mobilization = classify_with_cascade(normalized_incoming)
            classification_result = self._build_classification_result(
                normalized_incoming,
                thread_messages,
                mobilization,
            )

        if search_results is None:
            # Hybrid search for better retrieval precision
            hybrid_searcher = get_hybrid_searcher()
            search_results = hybrid_searcher.search(query=normalized_incoming, limit=5, rerank=True)

        message_context = MessageContext(
            chat_id=chat_id or "",
            message_text=normalized_incoming,
            is_from_me=False,
            timestamp=datetime.now(UTC),
            metadata={
                "thread": thread_messages,
            },
        )

        pipeline_request = self.build_generation_request(
            context=message_context,
            classification=classification_result,
            search_results=search_results,
            contact=contact,
            thread=thread_messages,
            instruction=instruction,
        )
        model_request = self._to_model_generation_request(pipeline_request)

        similarity = search_results[0].get("similarity", 0.0) if search_results else 0.0
        example_diversity = self._compute_example_diversity(search_results)
        pressure = self._pressure_from_classification(classification_result)

        base_confidence = {
            ResponsePressure.HIGH: 0.85,
            ResponsePressure.MEDIUM: 0.65,
            ResponsePressure.LOW: 0.45,
            ResponsePressure.NONE: 0.30,
        }[pressure]

        if similarity < 0.5:
            base_confidence *= 0.8
        if example_diversity < 0.3:
            base_confidence *= 0.9

        if base_confidence >= 0.7:
            confidence = "high"
        elif base_confidence >= 0.45:
            confidence = "medium"
        else:
            confidence = "low"

        metadata = {
            "confidence": confidence,
            "confidence_score": base_confidence,
            "similarity_score": similarity,
            "example_diversity": example_diversity,
            "mobilization_pressure": pressure.value,
        }

        return model_request, metadata

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

        if category_config.skip_slm:
            result = self._template_response(category_name, routing_start)
            latency_ms["total"] = (time.perf_counter() - routing_start) * 1000
            self._persist_reply_log(context, classification, None, result, latency_ms)
            return result

        search_results, latency_ms = self._search_context(
            search_results,
            incoming,
            chat_id,
            contact,
            cached_embedder,
            latency_ms,
        )

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

    def _template_response(
        self,
        category_name: str,
        routing_start: float,
    ) -> GenerationResponse:
        """Return a template response for categories that skip the SLM."""
        if category_name == "closing":
            template_response = random.choice(CLOSING_TEMPLATES)
        else:
            template_response = random.choice(ACKNOWLEDGE_TEMPLATES)

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
        from jarvis.prompts import get_optimized_examples

        incoming = context.message_text.strip()
        chat_id = context.chat_id or None
        category_name = str(
            classification.metadata.get("category_name", classification.category.value)
        )
        category_config = get_category_config(category_name)
        context_depth = category_config.context_depth

        context_messages: list[str] = []
        if thread:
            context_messages = thread[-context_depth:] if context_depth > 0 else []
        elif chat_id and context_depth > 0:
            context_messages, _ = self.context_service.fetch_conversation_context(
                chat_id, limit=context_depth
            )

        relationship_profile, contact_context = self.context_service.get_relationship_profile(
            contact, chat_id
        )
        context.metadata["context_messages"] = context_messages
        context.metadata["relationship_profile"] = relationship_profile
        context.metadata["contact_context"] = contact_context
        if contact and contact.display_name:
            context.metadata.setdefault("contact_name", contact.display_name)

        if chat_id:
            with ThreadPoolExecutor(max_workers=2) as pool:
                facts_future = pool.submit(self._fetch_contact_facts, context, chat_id)
                graph_future = pool.submit(self._fetch_graph_context, context, chat_id)
                facts_future.result()
                graph_future.result()

        instruction = self._resolve_instruction(
            instruction, category_name, category_config, classification
        )
        context.metadata["instruction"] = instruction or ""

        if len(search_results) > 3 and self.reranker:
            search_results = self.reranker.rerank(
                query=incoming,
                candidates=search_results,
                text_key="context_text",
                top_k=5,
            )

        optimized_examples = get_optimized_examples(category_name)
        category_exchanges = [(ex.context, ex.output) for ex in optimized_examples]

        top_results = search_results[:3]
        similar_exchanges = [
            (str(r.get("context_text", "")), str(r.get("reply_text", ""))) for r in top_results
        ]
        rag_rerank_scores = [
            self._safe_float(r.get("rerank_score"), default=0.0) for r in top_results
        ]

        all_exchanges = similar_exchanges + [
            ex for ex in category_exchanges if ex not in similar_exchanges
        ]
        all_rerank_scores = rag_rerank_scores + [0.0] * (
            len(all_exchanges) - len(similar_exchanges)
        )

        if cached_embedder is None:
            cached_embedder = get_embedder()
        all_exchanges = self._dedupe_examples(
            all_exchanges, cached_embedder, rerank_scores=all_rerank_scores
        )
        all_exchanges = all_exchanges[:5]

        rag_documents: list[RAGDocument] = []
        for result in top_results:
            context_text = str(result.get("context_text", "")).strip()
            if not context_text:
                continue
            rag_documents.append(
                RAGDocument(
                    content=context_text,
                    source=str(result.get("topic") or chat_id or "rag"),
                    score=self._safe_float(result.get("similarity"), default=0.0),
                    metadata={
                        "response_text": str(result.get("response_text", "")),
                        "rerank_score": self._safe_float(result.get("rerank_score"), default=0.0),
                    },
                )
            )

        return GenerationRequest(
            context=context,
            classification=classification,
            extraction=None,
            retrieved_docs=rag_documents,
            few_shot_examples=[
                {
                    "input": ctx,
                    "output": response,
                }
                for ctx, response in all_exchanges
                if ctx and response
            ],
        )

    def _to_model_generation_request(self, request: GenerationRequest) -> ModelGenerationRequest:
        pressure = self._pressure_from_classification(request.classification)
        prompt = build_prompt_from_request(request)
        return ModelGenerationRequest(
            prompt=prompt,
            max_tokens=self._max_tokens_for_pressure(pressure),
        )

    def _dedupe_examples(
        self,
        examples: list[tuple[str, str]],
        embedder: CachedEmbedder,
        rerank_scores: list[float] | None = None,
    ) -> list[tuple[str, str]]:
        """Deduplicate examples using greedy semantic similarity filtering.

        Args:
            examples: List of (context, output) tuples.
            embedder: Embedder for computing similarity.
            rerank_scores: Optional per-example rerank scores (higher = better).

        Returns:
            Deduplicated list of examples (highest quality kept).
        """
        if len(examples) <= 1:
            return examples

        # Skip expensive embedding-based dedup for small example sets
        # Small sets are unlikely to have problematic duplicates
        if len(examples) <= 6:
            return examples

        texts = [f"{ctx} {out}" for ctx, out in examples]

        # Use content-addressable caching for dedupe embeddings
        # This avoids recomputing embeddings for the same examples across calls
        embeddings = self._get_cached_embeddings(texts, embedder)

        # Score for ordering: prefer rerank_score, fallback to context length
        scores = rerank_scores or [0.0] * len(examples)

        # Sort by score descending (best first) for greedy selection
        indexed = sorted(
            range(len(examples)),
            key=lambda i: (scores[i], len(examples[i][0])),
            reverse=True,
        )

        # Greedy dedup: keep candidate if not too similar to any already-kept item
        kept_indices: list[int] = []
        kept_embs: list[Any] = []
        for i in indexed:
            emb = embeddings[i]
            too_similar = any(float(np.dot(emb, k)) > 0.85 for k in kept_embs)
            if not too_similar:
                kept_indices.append(i)
                kept_embs.append(emb)

        # Return in original order
        kept_indices.sort()
        return [examples[i] for i in kept_indices]

    def _get_cached_embeddings(
        self,
        texts: list[str],
        embedder: CachedEmbedder,
    ) -> Any:
        """Get embeddings with content-addressable caching.

        Uses a simple hash-based cache to avoid recomputing embeddings
        for the same text across multiple deduplication calls.

        Args:
            texts: List of texts to embed.
            embedder: Embedder for computing embeddings.

        Returns:
            Array of embeddings corresponding to input texts.
        """
        # Use a class-level cache to persist across calls
        if not hasattr(self, '_embedding_cache'):
            # LRU cache with max 1000 entries (~4MB for 384-dim float32 embeddings)
            self._embedding_cache: dict[str, Any] = {}
            self._embedding_cache_hits = 0
            self._embedding_cache_misses = 0

        cache = self._embedding_cache
        results: list[Any] = []
        texts_to_encode: list[tuple[int, str]] = []

        # Check cache for each text
        for idx, text in enumerate(texts):
            # Use hash of text as cache key
            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:32]
            if text_hash in cache:
                results.append((idx, cache[text_hash]))
                self._embedding_cache_hits += 1
            else:
                texts_to_encode.append((idx, text))
                self._embedding_cache_misses += 1

        # Batch encode missing texts
        if texts_to_encode:
            missing_texts = [t for _, t in texts_to_encode]
            new_embeddings = embedder.encode(missing_texts, normalize=True)

            # Store in cache and results
            for (idx, text), emb in zip(texts_to_encode, new_embeddings):
                text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:32]
                # Simple LRU eviction: if cache too big, clear half
                if len(cache) >= 1000:
                    # Clear oldest half (arbitrary but simple)
                    keys_to_remove = list(cache.keys())[:500]
                    for k in keys_to_remove:
                        del cache[k]
                cache[text_hash] = emb
                results.append((idx, emb))

        # Sort by original index and return embeddings
        results.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in results])

    def _generate_llm_reply(
        self,
        request: GenerationRequest,
        search_results: list[dict[str, Any]],
        thread: list[str] | None,
    ) -> GenerationResponse:
        incoming = request.context.message_text.strip()
        pressure = self._pressure_from_classification(request.classification)

        if pressure == ResponsePressure.NONE and not search_results:
            return GenerationResponse(
                response="",
                confidence=0.2,
                metadata={
                    "type": "skip",
                    "reason": "no_response_needed",
                    "similarity_score": 0.0,
                    "vec_candidates": 0,
                },
            )

        cat_conf = self._safe_float(
            request.classification.metadata.get("category_confidence"),
            default=request.classification.confidence,
        )
        has_thin_context = not thread or len(thread) < 2
        is_short_msg = len(incoming.split()) <= 3
        if (
            cat_conf < 0.4
            and pressure == ResponsePressure.NONE
            and is_short_msg
            and has_thin_context
        ):
            return GenerationResponse(
                response="",
                confidence=max(0.1, cat_conf),
                metadata={
                    "type": "clarify",
                    "reason": "ambiguous_message",
                    "similarity_score": 0.0,
                    "vec_candidates": len(search_results),
                },
            )

        try:
            model_request = self._to_model_generation_request(request)
            final_prompt = model_request.prompt
            response = self.generator.generate(model_request)
            text = response.text.strip()

            if text.lower() in UNCERTAIN_SIGNALS:
                return GenerationResponse(
                    response=text,
                    confidence=0.25,
                    metadata={
                        "type": "uncertain",
                        "reason": "model_uncertain",
                        "similarity_score": 0.0,
                        "vec_candidates": len(search_results),
                        "final_prompt": final_prompt,
                    },
                )

            similarity = (
                self._safe_float(search_results[0].get("similarity"), default=0.0)
                if search_results
                else 0.0
            )
            example_diversity = compute_example_diversity(search_results)
            reply_length = len(text.split())
            rerank_score = (
                self._safe_float(search_results[0].get("rerank_score"), default=0.0)
                if search_results
                else None
            )

            confidence_score, confidence_label = compute_confidence(
                pressure,
                similarity,
                example_diversity,
                reply_length,
                text,
                incoming_text=incoming,
                rerank_score=rerank_score,
            )

            similar_triggers = [
                str(row.get("context_text", ""))
                for row in search_results[:3]
                if row.get("context_text")
            ]
            used_docs = [doc.content for doc in request.retrieved_docs[:3] if doc.content]

            return GenerationResponse(
                response=text,
                confidence=confidence_score,
                used_kg_facts=used_docs,
                metadata={
                    "type": "generated",
                    "reason": "generated",
                    "category": str(
                        request.classification.metadata.get(
                            "category_name",
                            request.classification.category.value,
                        )
                    ),
                    "similarity_score": similarity,
                    "example_diversity": example_diversity,
                    "confidence_label": confidence_label,
                    "vec_candidates": len(search_results),
                    "similar_triggers": similar_triggers,
                    "final_prompt": final_prompt,
                },
            )
        except Exception as e:
            logger.exception("LLM generation failed: %s", e)
            return GenerationResponse(
                response="I'm having trouble generating a response.",
                confidence=0.2,
                metadata={
                    "type": "clarify",
                    "reason": "generation_error",
                    "similarity_score": 0.0,
                    "vec_candidates": len(search_results),
                },
            )

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


_service: ReplyService | None = None
_service_lock = threading.Lock()


def get_reply_service() -> ReplyService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = ReplyService()
    return _service
