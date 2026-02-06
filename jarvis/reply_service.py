"""Reply Service - Unified service for generating replies and suggestions.

Consolidates logic from:
- jarvis/router.py (Main RAG generation)
- jarvis/generation.py (Health-aware utilities)
- jarvis/reply_suggester.py (Fast pattern-based suggestions)
- jarvis/multi_option.py (Diverse commitment options)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from jarvis.classifiers.response_classifier import ResponseType
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
from jarvis.observability.metrics_router import RoutingMetrics, get_routing_metrics_store, hash_query
from jarvis.services.context_service import ContextService

if TYPE_CHECKING:
    from integrations.imessage.reader import ChatDBReader
    from jarvis.multi_option import MultiOptionResult, ResponseOption
    from jarvis.reply_suggester import ReplySuggestion
    from jarvis.search.retrieval import TypedRetriever
    from jarvis.search.semantic_search import SemanticSearcher
    from models import MLXGenerator

logger = logging.getLogger(__name__)

# Confidence threshold for retrieval
MIN_RETRIEVAL_CONFIDENCE = 0.5


class ReplyServiceError(JarvisError):
    """Raised when reply service operations fail."""

    default_message = "Reply service operation failed"
    default_code = ErrorCode.UNKNOWN


class ReplyService:
    """Unified service for generating AI replies and suggestions.

    This service coordinates between different generation strategies:
    1. Full RAG generation for high-quality single replies.
    2. Multi-option generation for commitment questions.
    3. Fast pattern-based suggestions for low-latency needs.
    """

    def __init__(
        self,
        db: JarvisDB | None = None,
        generator: MLXGenerator | None = None,
        imessage_reader: ChatDBReader | None = None,
        retriever: TypedRetriever | None = None,
    ) -> None:
        self._db = db
        self._generator = generator
        self._imessage_reader = imessage_reader
        self._retriever = retriever
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
    def retriever(self) -> TypedRetriever:
        with self._lock:
            if self._retriever is None:
                from jarvis.search.retrieval import get_typed_retriever

                self._retriever = get_typed_retriever()
            return self._retriever

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

    def generate_reply(
        self,
        incoming: str,
        contact: Contact | None = None,
        search_results: list[dict[str, Any]] | None = None,
        thread: list[str] | None = None,
        chat_id: str | None = None,
        mobilization: MobilizationResult | None = None,
    ) -> dict[str, Any]:
        """Generate a single best reply using RAG and LLM.

        This is the primary method for high-quality generation.
        """
        routing_start = time.perf_counter()
        latency_ms: dict[str, float] = {}
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

    def generate_options(
        self,
        incoming: str,
        contact_id: int | None = None,
        chat_id: str | None = None,
        force_commitment: bool = False,
    ) -> MultiOptionResult:
        """Generate diverse response options (AGREE, DECLINE, DEFER).

        Used primarily for commitment-style messages (invitations, requests).
        """
        from jarvis.multi_option import (
            FALLBACK_TEMPLATES,
            OPTION_PRIORITY,
            MultiOptionResult,
            ResponseOption,
        )

        # Check if commitment
        is_commitment, trigger_da = self._is_commitment_trigger(incoming)
        if force_commitment:
            is_commitment = True

        if not is_commitment:
            return MultiOptionResult(
                trigger=incoming,
                trigger_da=trigger_da,
                is_commitment=False,
                options=[],
            )

        # Implementation similar to MultiOptionGenerator.generate_options
        style_guide = self._get_style_guide(chat_id)
        cached_embedder = CachedEmbedder(get_embedder())

        multi_examples = self.retriever.get_examples_for_commitment(
            trigger=incoming,
            k_per_type=3,
            embedder=cached_embedder,
            trigger_da=trigger_da,
        )

        options: list[ResponseOption] = []
        for response_type in OPTION_PRIORITY:
            if len(options) >= 3:
                break

            examples = multi_examples.get_examples(response_type)

            # Strategy: Retrieval -> LLM -> Fallback
            if examples and examples[0].similarity >= MIN_RETRIEVAL_CONFIDENCE:
                options.append(
                    ResponseOption(
                        text=examples[0].response_text,
                        response_type=response_type,
                        confidence=examples[0].similarity,
                        source="template",
                    )
                )
                continue

            # LLM attempt
            llm_opt = self._generate_llm_option(incoming, response_type, style_guide, examples)
            if llm_opt:
                options.append(llm_opt)
                continue

            # Fallback
            import random

            templates = FALLBACK_TEMPLATES.get(response_type, ["Okay"])
            options.append(
                ResponseOption(
                    text=random.choice(templates),
                    response_type=response_type,
                    confidence=0.5,
                    source="fallback",
                )
            )

        # Deduplicate
        seen = set()
        unique_options = []
        for opt in options:
            if opt.text.lower().strip() not in seen:
                seen.add(opt.text.lower().strip())
                unique_options.append(opt)

        return MultiOptionResult(
            trigger=incoming,
            trigger_da=trigger_da,
            is_commitment=True,
            options=unique_options,
        )

    def generate_suggestions(
        self,
        incoming: str,
        contact_id: int | None = None,
        n_suggestions: int = 3,
    ) -> list[ReplySuggestion]:
        """Generate fast reply suggestions using patterns and retrieval.

        Does not use LLM, ensuring sub-200ms latency.
        """
        from jarvis.reply_suggester import (
            TEMPLATES,
            MessagePattern,
            ReplySuggestion,
            detect_pattern,
        )

        pattern = detect_pattern(incoming)
        suggestions: list[ReplySuggestion] = []

        try:
            from jarvis.search.vec_search import get_vec_searcher

            vec_searcher = get_vec_searcher(self.db)
            results = vec_searcher.search_with_pairs(
                query=incoming,
                limit=10,
            )

            seen = set()
            for r in results:
                resp = r.get("response_text", "").strip()
                if resp and resp.lower() not in seen:
                    seen.add(resp.lower())
                    suggestions.append(
                        ReplySuggestion(
                            text=resp,
                            source="retrieval",
                            confidence=r.get("similarity", 0.5),
                        )
                    )
                if len(suggestions) >= n_suggestions:
                    break
        except Exception as e:
            logger.warning("Suggestion retrieval failed: %s", e)

        # Fill with templates
        if len(suggestions) < n_suggestions:
            templates = TEMPLATES.get(pattern, TEMPLATES[MessagePattern.UNKNOWN])
            for t in templates:
                if len(suggestions) >= n_suggestions:
                    break
                if t.lower() not in [s.text.lower() for s in suggestions]:
                    suggestions.append(
                        ReplySuggestion(
                            text=t,
                            source="template",
                            confidence=0.7,
                        )
                    )

        return suggestions[:n_suggestions]

    # --- Internal Helpers ---

    def _generate_llm_reply(
        self,
        incoming: str,
        search_results: list[dict[str, Any]],
        contact: Contact | None,
        thread: list[str] | None,
        chat_id: str | None,
        mobilization: MobilizationResult,
    ) -> dict[str, Any]:
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

        # Build mobilization hint
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

        try:
            request = GenerationRequest(
                prompt=prompt,
                context_documents=[context],
                few_shot_examples=similar_exchanges,
                max_tokens=max_tokens,
            )
            response = self.generator.generate(request)
            text = response.text.strip()

            # Post-processing (simplified from router.py)
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

    def _generate_llm_option(
        self,
        trigger: str,
        response_type: ResponseType,
        style_guide: str,
        examples: list | None,
    ) -> ResponseOption | None:
        from jarvis.multi_option import ResponseOption

        can_generate, _ = self.can_use_llm()
        if not can_generate:
            return None

        from contracts.models import GenerationRequest
        from jarvis.prompts import COMMITMENT_PROMPT

        examples_section = ""
        if examples:
            ex_lines = []
            for ex in examples[:2]:
                ex_lines.append(f"Message: {ex.trigger_text}\nResponse: {ex.response_text}\n")
            examples_section = "\n### Examples:\n" + "\n".join(ex_lines)

        prompt = COMMITMENT_PROMPT.template.format(
            response_type=response_type.value.upper(),
            response_type_lower=response_type.value.lower(),
            style_guide=style_guide,
            trigger=trigger,
            examples_section=examples_section,
        )

        try:
            request = GenerationRequest(prompt=prompt, max_tokens=30, temperature=0.7)
            response = self.generator.generate(request)
            if response.text:
                return ResponseOption(
                    text=response.text.strip().split("\n")[0],
                    response_type=response_type,
                    confidence=0.7,
                    source="generated",
                )
        except Exception:
            pass
        return None

    def _is_commitment_trigger(self, trigger: str) -> tuple[bool, str | None]:
        from jarvis.multi_option import (
            COMMITMENT_TRIGGER_TYPES,
            _is_info_statement,
            _is_wh_question,
        )

        if _is_info_statement(trigger):
            return False, "statement"
        if _is_wh_question(trigger):
            return False, "question"
        trigger_da, _ = self.retriever.classify_trigger(trigger)
        return trigger_da in COMMITMENT_TRIGGER_TYPES, trigger_da

    def _get_style_guide(self, chat_id: str | None) -> str:
        if not chat_id:
            return "Use a casual, friendly tone."
        try:
            from jarvis.contacts.contact_profile import format_style_guide, get_contact_profile

            profile = get_contact_profile(chat_id)
            return format_style_guide(profile) if profile else "Use a casual, friendly tone."
        except Exception:
            return "Use a casual, friendly tone."

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
