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
from jarvis.db import Contact, JarvisDB, get_db
from jarvis.embedding_adapter import CachedEmbedder, get_embedder
from jarvis.errors import ErrorCode, JarvisError
from jarvis.observability.logging import log_event
from jarvis.observability.metrics_router import (
    RoutingMetrics,
    get_routing_metrics_store,
    hash_query,
)
from jarvis.prompts import (
    ACKNOWLEDGE_TEMPLATES,
    CLOSING_TEMPLATES,
    build_prompt_from_request,
    get_category_config,
)
from jarvis.services.context_service import ContextService

if TYPE_CHECKING:
    from integrations.imessage.reader import ChatDBReader
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
        self._context_service: ContextService | None = None
        self._reranker = None
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
                    from models import get_generator

                    self._generator = get_generator(skip_templates=True)
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
    def reranker(self):
        """Get or create the cross-encoder reranker (lazy-loaded)."""
        if self._reranker is None:
            with self._lock:
                if self._reranker is None:
                    from models.reranker import get_reranker

                    self._reranker = get_reranker()
        return self._reranker

    def can_use_llm(self) -> tuple[bool, str]:
        """Check if LLM can be used based on system health."""
        from jarvis.generation import can_use_llm as check_health

        return check_health()

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
    ) -> tuple[ModelGenerationRequest, dict[str, Any]]:
        """Prepare a model GenerationRequest through the typed pipeline for streaming.

        Runs all the same steps as the non-streaming path (health check, contact
        lookup, classification, RAG search, prompt assembly) but returns the
        model request instead of generating. Designed to be called via
        asyncio.to_thread() before streaming tokens.

        Pre-computed results can be passed to skip redundant work:
            classification_result: Skip re-running classification cascade.
            contact: Skip re-running contact lookup.
            search_results: Skip re-running RAG search.
            cached_embedder: Reuse pre-computed embeddings to avoid
                re-encoding the query text.
        """
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
            # Pre-encode so search_examples gets a cache hit (skip if results pre-provided)
            if normalized_incoming:
                cached_embedder.encode(normalized_incoming)
            search_results = self.context_service.search_examples(
                normalized_incoming,
                chat_id=chat_id,
                contact_id=contact.id if contact else None,
                embedder=cached_embedder,
            )

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

        similarity = search_results[0]["similarity"] if search_results else 0.0
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
            "similarity_score": similarity,
            "example_diversity": example_diversity,
            "mobilization_pressure": pressure.value,
        }

        return model_request, metadata

    def generate_reply(
        self,
        context: MessageContext,
        classification: ClassificationResult,
        search_results: list[dict[str, Any]] | None = None,
        thread: list[str] | None = None,
        contact: Contact | None = None,
        cached_embedder: CachedEmbedder | None = None,
    ) -> GenerationResponse:
        """Generate a reply from contract types.

        Orchestrates: validation -> template shortcut -> context search -> LLM gen -> metrics.
        """
        routing_start = time.perf_counter()
        latency_ms: dict[str, float] = {}
        log_event(
            logger,
            "reply.generate.start",
            level=logging.DEBUG,
            chat_id=context.chat_id or "",
            category=str(classification.category.value),
        )
        if cached_embedder is None:
            cached_embedder = get_embedder()

        incoming = context.message_text.strip()
        if not incoming or not incoming.strip():
            return self._empty_message_response()

        chat_id = context.chat_id or None
        thread_messages = self._resolve_thread(thread, context)

        category_name = str(
            classification.metadata.get("category_name", classification.category.value)
        )
        category_config = get_category_config(category_name)

        if category_config.skip_slm:
            return self._template_response(category_name, routing_start)

        search_results, latency_ms = self._search_context(
            search_results, incoming, chat_id, contact, cached_embedder, latency_ms,
        )

        can_generate, health_reason = self.can_use_llm()
        if not can_generate:
            log_event(logger, "reply.fallback", level=logging.WARNING, reason=health_reason)
            return GenerationResponse(
                response="",
                confidence=0.0,
                metadata={
                    "type": "fallback",
                    "reason": health_reason,
                    "similarity_score": 0.0,
                    "vec_candidates": len(search_results),
                },
            )

        result, latency_ms = self._generate_response(
            context, classification, search_results, contact, thread_messages,
            cached_embedder, latency_ms,
        )

        latency_ms["total"] = (time.perf_counter() - routing_start) * 1000
        self._log_and_record_metrics(
            result, category_name, incoming, search_results,
            latency_ms, cached_embedder,
        )

        return result

    def _empty_message_response(self) -> GenerationResponse:
        """Return a clarification response for empty messages."""
        return GenerationResponse(
            response="I received an empty message. Could you tell me what you need?",
            confidence=0.2,
            metadata={"type": "clarify", "reason": "empty_message", "similarity_score": 0.0},
        )

    def _resolve_thread(
        self, thread: list[str] | None, context: MessageContext,
    ) -> list[str]:
        """Extract thread messages from explicit param or context metadata."""
        if thread is not None:
            return thread
        metadata_thread = context.metadata.get("thread", [])
        if isinstance(metadata_thread, list):
            return [msg for msg in metadata_thread if isinstance(msg, str)]
        return []

    def _template_response(
        self, category_name: str, routing_start: float,
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
        """Run context search if results not already provided."""
        if search_results is None:
            search_start = time.perf_counter()
            search_results = self.context_service.search_examples(
                incoming,
                chat_id=chat_id,
                contact_id=contact.id if contact else None,
                embedder=cached_embedder,
            )
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
    ) -> tuple[GenerationResponse, dict[str, float]]:
        """Build generation request and run LLM inference."""
        gen_start = time.perf_counter()
        request = self.build_generation_request(
            context=context,
            classification=classification,
            search_results=search_results,
            contact=contact,
            thread=thread_messages,
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
                search_results[0]["similarity"] if search_results else 0.0,
            ),
            default=0.0,
        )

        self._record_metrics(
            incoming=incoming,
            decision="generate",
            similarity_score=similarity,
            latency_ms=latency_ms,
            cached_embedder=cached_embedder,
            vec_candidates=len(search_results),
            model_loaded=self.generator.is_loaded(),
        )

    # --- Internal Helpers ---

    def _fetch_contact_facts(self, context: MessageContext, chat_id):
        try:
            from jarvis.contacts.fact_index import search_relevant_facts
            from jarvis.prompts import format_facts_for_prompt

            incoming_text = context.message_text or ""
            facts = search_relevant_facts(incoming_text, chat_id, limit=5)
            context.metadata["contact_facts"] = format_facts_for_prompt(facts)
        except Exception as e:
            logger.debug(f"Optional fact fetch failed: {e}")

    def _fetch_graph_context(self, context: MessageContext, chat_id):
        try:
            from jarvis.graph.context import get_graph_context

            graph_ctx = get_graph_context(contact_id=chat_id, chat_id=chat_id)
            if graph_ctx:
                context.metadata["relationship_graph"] = graph_ctx
        except Exception as e:
            logger.debug(f"Optional graph context fetch failed: {e}")

    def _resolve_instruction(self, instruction, category_name, category_config, classification):
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
            context_messages = self.context_service.fetch_conversation_context(
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
            self._fetch_contact_facts(context, chat_id)
            self._fetch_graph_context(context, chat_id)

        instruction = self._resolve_instruction(
            instruction, category_name, category_config, classification
        )
        context.metadata["instruction"] = instruction or ""

        if len(search_results) > 3:
            search_results = self.reranker.rerank(
                query=incoming,
                candidates=search_results,
                text_key="trigger_text",
                top_k=5,
            )

        optimized_examples = get_optimized_examples(category_name)
        category_exchanges = [(ex.context, ex.output) for ex in optimized_examples]

        top_results = search_results[:3]
        similar_exchanges = [
            (str(r.get("trigger_text", "")), str(r.get("response_text", ""))) for r in top_results
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
            trigger_text = str(result.get("trigger_text", "")).strip()
            if not trigger_text:
                continue
            rag_documents.append(
                RAGDocument(
                    content=trigger_text,
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

    # Responses that signal the model is uncertain / lacks context
    _UNCERTAIN_SIGNALS = frozenset({"?", "??", "hm?", "what?", "huh?"})

    @staticmethod
    def _compute_confidence(
        pressure: ResponsePressure,
        rag_similarity: float,
        example_diversity: float,
        reply_length: int,
        reply_text: str,
        incoming_text: str = "",
        rerank_score: float | None = None,
    ) -> tuple[float, str]:
        """Compute confidence level based on multiple signals.

        Args:
            pressure: Response mobilization pressure level.
            rag_similarity: Top RAG result similarity score (0-1).
            example_diversity: Measure of example diversity (0-1).
            reply_length: Number of words in the reply.
            reply_text: The actual reply text for uncertain signal detection.
            incoming_text: Original incoming message for coherence check.
            rerank_score: Cross-encoder rerank score from top result (0-1).

        Returns:
            Tuple of (numeric_confidence 0-1, discrete label "high"/"medium"/"low").
        """
        # Base confidence by pressure
        base_confidence = {
            ResponsePressure.HIGH: 0.85,
            ResponsePressure.MEDIUM: 0.65,
            ResponsePressure.LOW: 0.45,
            ResponsePressure.NONE: 0.30,
        }[pressure]

        # Adjust based on RAG quality
        if rag_similarity < 0.5:
            base_confidence *= 0.8

        # Boost from cross-encoder reranking (more reliable than embedding sim)
        if rerank_score is not None and rerank_score > 0.7:
            base_confidence = min(base_confidence * 1.1, 0.95)

        # Adjust based on example diversity
        if example_diversity < 0.3:  # All from same contact
            base_confidence *= 0.9

        # Uncertain signals only matter if very short reply + high pressure
        if (
            reply_length < 3
            and pressure == ResponsePressure.HIGH
            and reply_text.lower() in ReplyService._UNCERTAIN_SIGNALS
        ):
            base_confidence *= 0.7

        # Coherence penalty: reply that parrots the input is low quality
        if incoming_text and reply_text:
            reply_lower = reply_text.lower().strip()
            incoming_lower = incoming_text.lower().strip()
            if reply_lower == incoming_lower or (
                len(reply_lower) > 5 and incoming_lower.startswith(reply_lower)
            ):
                base_confidence *= 0.5

        # Clamp to [0, 1]
        base_confidence = max(0.0, min(base_confidence, 1.0))

        # Map float to discrete level
        if base_confidence >= 0.7:
            label = "high"
        elif base_confidence >= 0.45:
            label = "medium"
        else:
            label = "low"

        return base_confidence, label

    @staticmethod
    def _compute_example_diversity(search_results: list[dict[str, Any]]) -> float:
        """Compute diversity of search results by unique contacts/contexts.

        Args:
            search_results: List of search result dicts.

        Returns:
            Diversity score from 0.0 (all same) to 1.0 (all unique).
        """
        if not search_results:
            return 0.0

        # Count unique trigger texts as proxy for diversity
        unique_triggers = len(set(r.get("trigger_text", "") for r in search_results))
        return min(unique_triggers / len(search_results), 1.0)

    def _dedupe_examples(
        self,
        examples: list[tuple[str, str]],
        embedder: CachedEmbedder,
        rerank_scores: list[float] | None = None,
    ) -> list[tuple[str, str]]:
        """Deduplicate examples using semantic similarity with topic diversity.

        Args:
            examples: List of (context, output) tuples.
            embedder: Embedder for computing similarity.
            rerank_scores: Optional per-example rerank scores (higher = better).

        Returns:
            Deduplicated list of examples (highest quality kept).
        """
        if len(examples) <= 1:
            return examples

        # Compute embeddings for all examples
        texts = [f"{ctx} {out}" for ctx, out in examples]
        embeddings = embedder.encode(texts, normalize=True)

        # Score for tiebreaking: prefer rerank_score, fallback to context length
        scores = rerank_scores or [0.0] * len(examples)

        # Compute full similarity matrix once (replaces ~n^2 individual dot products)
        emb_array = np.array(embeddings)
        sim_matrix = np.dot(emb_array, emb_array.T)

        # Find near-duplicates (cosine sim > 0.85)
        keep = [True] * len(examples)
        for i in range(len(examples)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(examples)):
                if not keep[j]:
                    continue
                if sim_matrix[i, j] > 0.85:
                    # Keep the one with higher rerank score, break ties by context length
                    i_score = (scores[i], len(examples[i][0]))
                    j_score = (scores[j], len(examples[j][0]))
                    if i_score >= j_score:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break

        kept = [ex for ex, k in zip(examples, keep) if k]
        kept_indices = [i for i, k in enumerate(keep) if k]

        # Topic diversity: if all remaining examples are too similar to each other
        # (avg pairwise sim > 0.75), drop the least unique one to make room
        if len(kept) > 2:
            n = len(kept)
            # Extract sub-matrix for kept items
            kept_sim = sim_matrix[np.ix_(kept_indices, kept_indices)]
            # Average pairwise sim (exclude diagonal)
            mask = ~np.eye(n, dtype=bool)
            avg_sim = float(kept_sim[mask].mean()) if n > 1 else 0.0
            if avg_sim > 0.75:
                # Drop the example most similar to all others
                avg_per_ex = (kept_sim.sum(axis=1) - 1.0) / (n - 1)  # exclude self-sim
                drop_idx = int(np.argmax(avg_per_ex))
                kept.pop(drop_idx)

        return kept

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
            response = self.generator.generate(model_request)
            text = response.text.strip()

            if text.lower() in self._UNCERTAIN_SIGNALS:
                return GenerationResponse(
                    response=text,
                    confidence=0.25,
                    metadata={
                        "type": "uncertain",
                        "reason": "model_uncertain",
                        "similarity_score": 0.0,
                        "vec_candidates": len(search_results),
                    },
                )

            similarity = (
                self._safe_float(search_results[0].get("similarity"), default=0.0)
                if search_results
                else 0.0
            )
            example_diversity = self._compute_example_diversity(search_results)
            reply_length = len(text.split())
            rerank_score = (
                self._safe_float(search_results[0].get("rerank_score"), default=0.0)
                if search_results
                else None
            )

            confidence_score, confidence_label = self._compute_confidence(
                pressure,
                similarity,
                example_diversity,
                reply_length,
                text,
                incoming_text=incoming,
                rerank_score=rerank_score,
            )

            similar_triggers = [
                str(row.get("trigger_text", ""))
                for row in search_results[:3]
                if row.get("trigger_text")
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

    # Imported from jarvis.prompts - single source of truth for all prompts
    from jarvis.prompts import CHAT_SYSTEM_PROMPT as _CHAT_SYSTEM_PROMPT

    def _build_chat_prompt(
        self,
        incoming: str,
        instruction: str | None = None,
        exchanges: list[tuple[str, str]] | None = None,
    ) -> str:
        """Build a chat-template prompt with multi-turn few-shot examples.

        Uses the tokenizer's chat template for proper instruct formatting,
        which produces dramatically better results than raw XML prompts
        on small instruct models.
        """
        if not hasattr(self, "_tokenizer_cache"):
            loader = getattr(self.generator, "_loader", None)
            self._tokenizer_cache = getattr(loader, "_tokenizer", None) if loader else None
        tokenizer = self._tokenizer_cache
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            # Fallback to raw prompt if no chat template
            from jarvis.prompts import build_rag_reply_prompt

            return build_rag_reply_prompt(
                context="",
                last_message=incoming,
                contact_name="them",
                instruction=instruction,
            )

        system_msg = self._CHAT_SYSTEM_PROMPT
        if instruction:
            system_msg = f"{system_msg} {instruction}"

        messages: list[dict[str, str]] = [{"role": "system", "content": system_msg}]

        # Add few-shot examples as multi-turn conversation
        if exchanges:
            for trigger, response in exchanges[:4]:
                # Strip context prefixes, just use the core message
                trigger_clean = trigger.strip()
                if len(trigger_clean) > 150:
                    trigger_clean = trigger_clean[-150:]
                messages.append({"role": "user", "content": trigger_clean})
                messages.append({"role": "assistant", "content": response.strip()})

        # Add the actual message to reply to
        messages.append({"role": "user", "content": incoming})

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _record_metrics(
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
                generation_time_ms=latency_ms.get("generation", 0.0),
                tokens_per_second=0.0,
            )
            get_routing_metrics_store().record(metrics)
        except Exception as e:
            logger.debug("Metrics write failed: %s", e)


_service: ReplyService | None = None
_service_lock = threading.Lock()


def get_reply_service() -> ReplyService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = ReplyService()
    return _service
