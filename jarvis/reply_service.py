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
from typing import TYPE_CHECKING, Any

import numpy as np

from jarvis.classifiers.category_classifier import classify_category
from jarvis.classifiers.response_mobilization import (
    MobilizationResult,
    ResponsePressure,
    classify_response_pressure,
)
from jarvis.db import Contact, JarvisDB, get_db
from jarvis.embedding_adapter import CachedEmbedder, get_embedder
from jarvis.errors import ErrorCode, JarvisError
from jarvis.observability.metrics_router import (
    RoutingMetrics,
    get_routing_metrics_store,
    hash_query,
)
from jarvis.prompts import ACKNOWLEDGE_TEMPLATES, CLOSING_TEMPLATES, get_category_config
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

    @property
    def reranker(self):
        """Get or create the cross-encoder reranker (lazy-loaded)."""
        with self._lock:
            if self._reranker is None:
                from models.reranker import get_reranker

                self._reranker = get_reranker()
            return self._reranker

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

        # 6. Build metadata with improved confidence scoring
        similarity = search_results[0]["similarity"] if search_results else 0.0
        example_diversity = self._compute_example_diversity(search_results)

        # For streaming, we don't have the reply yet, so use pressure-based confidence
        base_confidence = {
            ResponsePressure.HIGH: 0.85,
            ResponsePressure.MEDIUM: 0.65,
            ResponsePressure.LOW: 0.45,
            ResponsePressure.NONE: 0.30,
        }[mobilization.pressure]

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

        # 1b. Category classification and routing

        category_result = classify_category(
            incoming, context=thread or [], mobilization=mobilization
        )
        category_config = get_category_config(category_result.category)

        # If closing/acknowledge category, skip SLM and return template
        if category_config.skip_slm:
            if category_result.category == "closing":
                template_response = random.choice(CLOSING_TEMPLATES)
            else:  # acknowledge
                template_response = random.choice(ACKNOWLEDGE_TEMPLATES)

            return {
                "type": category_result.category,
                "response": template_response,
                "confidence": "high",
                "reason": f"category={category_result.category}",
                "category": category_result.category,
            }

        # 2. Search for similar examples
        if search_results is None:
            search_start = time.perf_counter()
            search_results = self.context_service.search_examples(incoming, chat_id=chat_id)
            latency_ms["context_search"] = (time.perf_counter() - search_start) * 1000

        # 3. Generate response
        can_generate, health_reason = self.can_use_llm()
        if not can_generate:
            logger.warning("FALLBACK | reason=%s | returning empty response", health_reason)
            return {
                "type": "fallback",
                "response": "",
                "confidence": "none",
                "reason": health_reason,
            }

        gen_start = time.perf_counter()
        result = self._generate_llm_reply(
            incoming,
            search_results,
            contact,
            thread,
            chat_id=chat_id,
            mobilization=mobilization,
            category_result=category_result,
        )
        latency_ms["generation"] = (time.perf_counter() - gen_start) * 1000

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
        category_result=None,
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
        from jarvis.prompts import get_category_config

        # Build context with category-specific depth
        if category_result:
            category_config = get_category_config(category_result.category)
            context_depth = category_config.context_depth
        else:
            context_depth = 5  # default

        context_messages = []
        if thread:
            context_messages = thread[-context_depth:] if context_depth > 0 else []
        elif chat_id and context_depth > 0:
            context_messages = self.context_service.fetch_conversation_context(
                chat_id, limit=context_depth
            )

        context = "\n".join(context_messages) + f"\n[Incoming]: {incoming}"

        # Relationship profile
        relationship_profile, contact_context = self.context_service.get_relationship_profile(
            contact, chat_id
        )

        # Category routing: classify message and use MIPRO-optimized programs
        from jarvis.prompts import (
            get_optimized_examples,
            get_optimized_instruction,
            resolve_category,
        )

        category = resolve_category(
            incoming,
            context=context_messages,
            mobilization=mobilization,
        )

        # Use MIPRO-compiled instruction if available, else category hint
        if instruction is None:
            optimized_instruction = get_optimized_instruction(category)
            if optimized_instruction:
                instruction = optimized_instruction
            elif category_result and category_config.system_prompt:
                instruction = category_config.system_prompt
            else:
                instruction = self._build_mobilization_hint(mobilization)

        # Rerank search results for better relevance
        if len(search_results) > 1:
            search_results = self.reranker.rerank(
                query=incoming,
                candidates=search_results,
                text_key="trigger_text",
                top_k=5,
            )

        # Use category-specific few-shot examples if available
        optimized_examples = get_optimized_examples(category)
        category_exchanges = [(ex.context, ex.output) for ex in optimized_examples]

        top_results = search_results[:3]
        similar_exchanges = [(r["trigger_text"], r["response_text"]) for r in top_results]
        rag_rerank_scores = [r.get("rerank_score", 0.0) for r in top_results]

        # Merge: RAG-retrieved examples first, then category-specific (up to 5 total)
        all_exchanges = similar_exchanges + [
            ex for ex in category_exchanges if ex not in similar_exchanges
        ]
        # Extend rerank scores with 0.0 for category examples
        all_rerank_scores = rag_rerank_scores + [0.0] * (
            len(all_exchanges) - len(similar_exchanges)
        )

        # Deduplicate semantically similar examples
        cached_embedder = get_embedder()
        all_exchanges = self._dedupe_examples(
            all_exchanges, cached_embedder, rerank_scores=all_rerank_scores
        )

        # Limit to 5 total examples
        all_exchanges = all_exchanges[:5]

        # Build prompt using chat template (multi-turn few-shot)
        prompt = self._build_chat_prompt(
            incoming=incoming,
            instruction=instruction,
            exchanges=all_exchanges,
        )

        max_tokens = 20 if mobilization.pressure == ResponsePressure.NONE else 40

        return GenerationRequest(
            prompt=prompt,
            context_documents=[],  # Already baked into RAG prompt
            few_shot_examples=[],  # Already baked into RAG prompt
            max_tokens=max_tokens,
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

        # Find near-duplicates (cosine sim > 0.85)
        keep = [True] * len(examples)
        for i in range(len(examples)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(examples)):
                if not keep[j]:
                    continue
                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim > 0.85:
                    # Keep the one with higher rerank score, break ties by context length
                    i_score = (scores[i], len(examples[i][0]))
                    j_score = (scores[j], len(examples[j][0]))
                    if i_score >= j_score:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break

        kept = [ex for ex, k in zip(examples, keep) if k]
        kept_embs = [emb for emb, k in zip(embeddings, keep) if k]

        # Topic diversity: if all remaining examples are too similar to each other
        # (avg pairwise sim > 0.75), drop the least unique one to make room
        if len(kept) > 2:
            n = len(kept)
            pair_sims = []
            for i in range(n):
                for j in range(i + 1, n):
                    pair_sims.append(float(np.dot(kept_embs[i], kept_embs[j])))
            avg_sim = sum(pair_sims) / len(pair_sims) if pair_sims else 0.0
            if avg_sim > 0.75:
                # Drop the example most similar to all others
                avg_per_ex = []
                for i in range(n):
                    s = sum(
                        float(np.dot(kept_embs[i], kept_embs[j]))
                        for j in range(n) if j != i
                    ) / (n - 1)
                    avg_per_ex.append(s)
                drop_idx = int(np.argmax(avg_per_ex))
                kept.pop(drop_idx)

        return kept

    def _generate_llm_reply(
        self,
        incoming: str,
        search_results: list[dict[str, Any]],
        contact: Contact | None,
        thread: list[str] | None,
        chat_id: str | None,
        mobilization: MobilizationResult,
        category_result=None,
    ) -> dict[str, Any]:
        # Pre-generation gate: skip when no response is needed and no examples found
        if mobilization.pressure == ResponsePressure.NONE and not search_results:
            return {
                "type": "skip",
                "response": "",
                "confidence": "none",
                "reason": "no_response_needed",
            }

        # Ambiguity gate: clarify when signals are too weak to generate well
        cat_conf = getattr(category_result, "confidence", 1.0) if category_result else 1.0
        has_thin_context = not thread or len(thread) < 2
        is_short_msg = len(incoming.split()) <= 3
        if (
            cat_conf < 0.4
            and mobilization.pressure == ResponsePressure.NONE
            and is_short_msg
            and has_thin_context
        ):
            return {
                "type": "clarify",
                "response": "",
                "confidence": "low",
                "confidence_score": cat_conf,
                "reason": "ambiguous_message",
            }

        try:
            request = self.build_generation_request(
                incoming,
                search_results,
                contact,
                thread,
                chat_id,
                mobilization,
                category_result=category_result,
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

            # Compute confidence using multiple signals
            similarity = search_results[0]["similarity"] if search_results else 0.0
            example_diversity = self._compute_example_diversity(search_results)
            reply_length = len(text.split())
            rerank_score = search_results[0].get("rerank_score") if search_results else None

            confidence_score, confidence = self._compute_confidence(
                mobilization.pressure,
                similarity,
                example_diversity,
                reply_length,
                text,
                incoming_text=incoming,
                rerank_score=rerank_score,
            )

            return {
                "type": "generated",
                "response": text,
                "confidence": confidence,
                "confidence_score": confidence_score,
                "similarity_score": similarity,
                "example_diversity": example_diversity,
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
        tokenizer = getattr(self.generator, "_loader", None)
        tokenizer = getattr(tokenizer, "_tokenizer", None) if tokenizer else None
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
