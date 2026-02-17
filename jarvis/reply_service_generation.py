"""Generation/request-building helpers extracted from ReplyService."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np

from contracts.models import GenerationRequest as ModelGenerationRequest
from jarvis.config import get_config
from jarvis.contracts.pipeline import (
    GenerationRequest,
    GenerationResponse,
    MessageContext,
)
from jarvis.embedding_adapter import CachedEmbedder, get_embedder
from jarvis.prompts import build_prompt_from_request, get_category_config

logger = logging.getLogger(__name__)


def prepare_streaming_context(
    service: Any,
    *,
    incoming: str,
    thread: list[str] | None = None,
    chat_id: str | None = None,
    instruction: str | None = None,
    classification_result: Any = None,
    contact: Any = None,
    search_results: list[dict[str, Any]] | None = None,
    cached_embedder: CachedEmbedder | None = None,
    reply_error_cls: type[Exception] = RuntimeError,
) -> tuple[ModelGenerationRequest, dict[str, Any]]:
    """Prepare model request and metadata for streaming reply generation."""
    can_generate, health_reason = service.can_use_llm()
    if not can_generate:
        raise reply_error_cls(f"LLM unavailable: {health_reason}")

    normalized_incoming = incoming.strip()
    thread_messages = [msg for msg in (thread or []) if isinstance(msg, str)]

    if cached_embedder is None:
        cached_embedder = get_embedder()

    if contact is None:
        contact = service.context_service.get_contact(None, chat_id)

    # Direct-to-LLM: skip classification, use universal defaults
    # Classification runs in background (prefetch) for analytics, not at request time
    if classification_result is None:
        from dataclasses import dataclass

        from jarvis.contracts.pipeline import CategoryType, IntentType, UrgencyLevel

        @dataclass
        class DefaultClassification:
            category: CategoryType = CategoryType.FULL_RESPONSE
            intent: IntentType = IntentType.STATEMENT
            urgency: UrgencyLevel = UrgencyLevel.LOW
            requires_knowledge: bool = False
            confidence: float = 0.5
            metadata: dict = None

            def __post_init__(self):
                if self.metadata is None:
                    self.metadata = {"category_name": "statement"}

        classification_result = DefaultClassification()

    # Direct-to-LLM: no RAG search (causes hallucinations per research)
    # Just conversation context - same as chatting directly with Ollama

    message_context = MessageContext(
        chat_id=chat_id or "",
        message_text=normalized_incoming,
        is_from_me=False,
        timestamp=datetime.now(UTC),
        metadata={"thread": thread_messages},
    )

    pipeline_request = service.build_generation_request(
        context=message_context,
        classification=classification_result,
        search_results=[],
        contact=contact,
        thread=thread_messages,
        instruction=instruction,
    )
    model_request = service._to_model_generation_request(pipeline_request)

    # Direct-to-LLM: simple confidence (LLM handles routing)
    base_confidence = 0.65

    config = get_config()
    if base_confidence >= config.similarity_thresholds.high_confidence:
        confidence = "high"
    elif base_confidence >= config.similarity_thresholds.medium_confidence:
        confidence = "medium"
    else:
        confidence = "low"

    metadata = {
        "confidence": confidence,
        "confidence_score": base_confidence,
        "final_prompt": model_request.prompt,
    }
    return model_request, metadata


def build_generation_request(
    service: Any,
    *,
    context: MessageContext,
    classification: Any,
    search_results: list[dict[str, Any]],
    contact: Any,
    thread: list[str] | None = None,
    instruction: str | None = None,
    cached_embedder: CachedEmbedder | None = None,
) -> GenerationRequest:
    """Build typed generation request from routing/search context."""
    from jarvis.prompts import get_optimized_examples

    chat_id = context.chat_id or None
    category_name = classification.metadata.get("category_name")
    if category_name is None:
        category_name = getattr(classification.category, "value", classification.category)
    category_name = str(category_name)
    category_config = get_category_config(category_name)
    context_depth = category_config.context_depth

    context_messages: list[str] = []
    if thread:
        context_messages = thread[-context_depth:] if context_depth > 0 else []
    elif chat_id and context_depth > 0:
        context_messages, _ = service.context_service.fetch_conversation_context(
            chat_id,
            limit=context_depth,
        )

    relationship_profile, contact_context = service.context_service.get_relationship_profile(
        contact,
        chat_id,
    )
    context.metadata["context_messages"] = context_messages
    context.metadata["relationship_profile"] = relationship_profile
    context.metadata["contact_context"] = contact_context

    # Determine contact name for bot detection and style guide
    cname = "them"
    if contact and contact.display_name:
        cname = contact.display_name
    elif context.metadata.get("contact_name"):
        cname = str(context.metadata.get("contact_name"))

    # Bot/Service detection: skip irrelevant context for automated messages
    is_bot = service.context_service.is_bot_chat(chat_id, cname)

    if chat_id and not is_bot:
        # Background fact extraction disabled per user request to reduce latency/timeouts.
        # These operations should happen offline or in a non-blocking way.
        logger.debug("[build] Skipping blocking context fetch for %s", chat_id[:12])
    elif is_bot:
        logger.debug("[build] Skipping person-specific context for bot/service chat")

    instruction = service._resolve_instruction(
        instruction,
        category_name,
        category_config,
        classification,
    )

    # Personalization: explicitly mention the contact name in the instruction
    if cname and cname != "them":
        personalization = f" You are replying to {cname}."
        instruction = (instruction or "") + personalization

    # Add guidance for follow-ups if last message was from Me
    last_is_from_me = context.is_from_me
    if last_is_from_me:
        followup_guidance = (
            " Note: You spoke last. Add a follow-up or ask if they saw your message."
        )
        instruction = (instruction or "") + followup_guidance

    context.metadata["instruction"] = instruction or ""
    context.metadata.setdefault("contact_name", cname)

    # Direct-to-LLM: just use category examples, no RAG (causes hallucinations)
    optimized_examples = get_optimized_examples(category_name)
    all_exchanges = [(ex.context, ex.output) for ex in optimized_examples]

    logger.info(
        "[build] Context: %d messages, %d examples, contact=%s",
        len(context_messages),
        len(all_exchanges),
        context.metadata.get("contact_name", "unknown"),
    )

    return GenerationRequest(
        context=context,
        classification=classification,
        extraction=None,
        retrieved_docs=[],
        few_shot_examples=[],  # No examples - causes hallucinations per research
    )


def to_model_generation_request(service: Any, request: GenerationRequest) -> ModelGenerationRequest:
    """Convert pipeline request into model-native request."""
    prompt = build_prompt_from_request(request)
    config = get_config()
    return ModelGenerationRequest(
        prompt=prompt,
        max_tokens=config.model.max_tokens_reply,
        stop_sequences=["</reply>", "<system>", "<conversation>", "<examples>"],
    )


def dedupe_examples(
    service: Any,
    *,
    examples: list[tuple[str, str]],
    embedder: CachedEmbedder,
    rerank_scores: list[float] | None = None,
) -> list[tuple[str, str]]:
    """Deduplicate examples using semantic similarity filtering."""
    if len(examples) <= 1:
        return examples
    if len(examples) <= 6:
        return examples

    texts = [f"{ctx} {out}" for ctx, out in examples]
    embeddings = get_cached_embeddings(service, texts=texts, embedder=embedder)

    scores = rerank_scores or [0.0] * len(examples)
    indexed = sorted(
        range(len(examples)),
        key=lambda i: (scores[i], len(examples[i][0])),
        reverse=True,
    )

    kept_indices: list[int] = []
    kept_embs: list[Any] = []
    for i in indexed:
        emb = embeddings[i]
        too_similar = any(float(np.dot(emb, k)) > 0.85 for k in kept_embs)
        if not too_similar:
            kept_indices.append(i)
            kept_embs.append(emb)

    kept_indices.sort()
    return [examples[i] for i in kept_indices]


def get_cached_embeddings(
    service: Any,
    *,
    texts: list[str],
    embedder: CachedEmbedder,
) -> Any:
    """Get embeddings with content-addressable caching."""
    if not hasattr(service, "_embedding_cache"):
        service._embedding_cache = {}
        service._embedding_cache_hits = 0
        service._embedding_cache_misses = 0

    cache = service._embedding_cache
    results: list[Any] = []
    texts_to_encode: list[tuple[int, str]] = []

    for idx, text in enumerate(texts):
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
        if text_hash in cache:
            results.append((idx, cache[text_hash]))
            service._embedding_cache_hits += 1
        else:
            texts_to_encode.append((idx, text))
            service._embedding_cache_misses += 1

    if texts_to_encode:
        missing_texts = [t for _, t in texts_to_encode]
        new_embeddings = embedder.encode(missing_texts, normalize=True)

        for (idx, text), emb in zip(texts_to_encode, new_embeddings):
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
            if len(cache) >= 1000:
                keys_to_remove = list(cache.keys())[:500]
                for k in keys_to_remove:
                    del cache[k]
            cache[text_hash] = emb
            results.append((idx, emb))

    results.sort(key=lambda x: x[0])
    return np.array([emb for _, emb in results])


def generate_llm_reply(
    service: Any,
    *,
    request: GenerationRequest,
    search_results: list[dict[str, Any]],
    thread: list[str] | None,
) -> GenerationResponse:
    """Generate response from model and compute confidence metadata."""
    incoming = request.context.message_text.strip()
    # Direct-to-LLM: always let LLM decide whether to respond
    logger.info("[reply] Incoming: %r | direct-to-llm", incoming[:80])

    # Simple fast-path: skip obvious non-responses (reactions, acknowledgments)
    from jarvis.text_normalizer import is_acknowledgment_only, is_reaction

    if is_reaction(incoming) or is_acknowledgment_only(incoming):
        return GenerationResponse(
            response="",
            confidence=0.1,
            metadata={
                "type": "skip",
                "reason": "reaction_or_ack",
            },
        )

    has_thin_context = not thread or len(thread) < 2
    is_short_msg = len(incoming.split()) <= 3
    if is_short_msg and has_thin_context:
        # Direct-to-LLM: let model decide on short messages with thin context
        return GenerationResponse(
            response="",
            confidence=0.3,
            metadata={
                "type": "clarify",
                "reason": "ambiguous_message",
            },
        )

    try:
        model_request = to_model_generation_request(service, request)

        # Generate a single candidate at low temperature for stability
        import random
        jitter = random.uniform(-0.02, 0.02)
        model_request.temperature = max(0.01, 0.1 + jitter)

        # Ensure repetition penalty is set if not already
        if not model_request.repetition_penalty:
            from jarvis.prompts.generation_config import DEFAULT_REPETITION_PENALTY
            model_request.repetition_penalty = DEFAULT_REPETITION_PENALTY

        response = service.generator.generate(model_request)
        text = response.text.strip()

        # Basic cleanup
        if "</reply>" in text:
            text = text.split("</reply>")[0].strip()

        for tag in ["(Note:", "Note:", "<system>", "<style", "[lowercase]"]:
            if tag in text:
                text = text.split(tag)[0].strip()

        if not text:
            return GenerationResponse(
                response="...",
                confidence=0.1,
                metadata={"reason": "empty_candidate"},
            )

        logger.info("[reply] Generated response: %r", text)

        # Direct-to-LLM: simple confidence
        # Coherence check only
        base_conf = 0.65
        if text.lower().strip() == incoming.lower().strip():
            base_conf *= 0.5

        confidence_score = max(0.0, min(base_conf, 1.0))
        confidence_label = (
            "high" if confidence_score >= 0.7 else "medium" if confidence_score >= 0.45 else "low"
        )

        logger.info(
            "[reply] Confidence: %.2f (%s) | direct-to-llm",
            confidence_score,
            confidence_label,
        )

        return GenerationResponse(
            response=text,
            confidence=confidence_score,
            metadata={
                "type": "generated",
                "reason": "generated",
                "confidence_label": confidence_label,
                "final_prompt": model_request.prompt,
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
            },
        )
