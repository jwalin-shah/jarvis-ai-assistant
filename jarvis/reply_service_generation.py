"""Generation/request-building helpers extracted from ReplyService."""

from __future__ import annotations

import hashlib
import logging
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
            metadata: dict[str, Any] | None = None

            def __post_init__(self) -> None:
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

    config = get_config()
    use_rag = config.reply_pipeline.reply_enable_rag
    use_few_shot = config.reply_pipeline.reply_enable_few_shot

    return GenerationRequest(
        context=context,
        classification=classification,
        extraction=None,
        retrieved_docs=search_results if use_rag else [],
        few_shot_examples=all_exchanges if use_few_shot else [],
    )


def to_model_generation_request(service: Any, request: GenerationRequest) -> ModelGenerationRequest:
    """Convert pipeline request into model-native request."""
    prompt = build_prompt_from_request(request)
    config = get_config()

    # Debug: Log the full prompt to a separate file
    try:
        from pathlib import Path

        debug_log = Path("logs/last_prompt.txt")
        debug_log.parent.mkdir(parents=True, exist_ok=True)
        debug_log.write_text(prompt)
    except Exception:
        pass

    stop_sequences = [
        "<|im_end|>",
        "<|im_start|>",
        "</reply>",
        "<system>",
        "<conversation>",
        "<examples>",
        "Context:",
        "Last Message:",
        "[",
        "You:",
        "Me:",
        "Them:",
        "Assistant:",
        "System:",
    ]

    if config.reply_pipeline.reply_newline_stop_enabled:
        stop_sequences.append("\n")

    max_tokens = config.model.max_tokens_reply
    cap_mode = config.reply_pipeline.reply_word_cap_mode
    if cap_mode == "soft_25":
        max_tokens = min(max_tokens, 25)
    elif cap_mode == "soft_50":
        max_tokens = min(max_tokens, 50)

    return ModelGenerationRequest(
        prompt=prompt,
        max_tokens=max_tokens,
        stop_sequences=stop_sequences,
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
    """Get embeddings with content-addressable caching using LRU eviction."""
    from collections import OrderedDict

    if not hasattr(service, "_embedding_cache"):
        service._embedding_cache = OrderedDict()
        service._embedding_cache_hits = 0
        service._embedding_cache_misses = 0

    cache = service._embedding_cache
    cache_max_size = 1000
    results: list[Any] = []
    texts_to_encode: list[tuple[int, str, str]] = []

    for idx, text in enumerate(texts):
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
        if text_hash in cache:
            cache.move_to_end(text_hash)
            results.append((idx, cache[text_hash]))
            service._embedding_cache_hits += 1
        else:
            texts_to_encode.append((idx, text, text_hash))
            service._embedding_cache_misses += 1

    if texts_to_encode:
        missing_texts = [t for _, t, _ in texts_to_encode]
        new_embeddings = embedder.encode(missing_texts, normalize=True)

        for (idx, text, text_hash), emb in zip(texts_to_encode, new_embeddings):
            if len(cache) >= cache_max_size:
                cache.popitem(last=False)
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
    chat_id = request.context.chat_id
    cname = request.context.metadata.get("contact_name", "")

    # Skip obvious bot/automated chats
    if service.context_service.is_bot_chat(chat_id, cname):
        return GenerationResponse(
            response="",
            confidence=0.1,
            metadata={
                "type": "skip",
                "reason": "bot_chat",
            },
        )

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

    config = get_config()
    has_thin_context = not thread or len(thread) < 2
    is_short_msg = len(incoming.split()) <= 3
    if config.reply_pipeline.reply_short_msg_gate_enabled and is_short_msg and has_thin_context:
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
        # Use negative constraints from config (generalizable)
        model_request.negative_constraints = config.model.negative_constraints

        # Generate a single candidate at low temperature for stability
        import random

        jitter = random.uniform(-0.01, 0.01)
        model_request.temperature = max(0.0, config.model.temperature + jitter)

        # High repetition penalty to prevent "AI loops" and encourage variety
        model_request.repetition_penalty = 1.25

        response = service.generator.generate(model_request)
        generator_finish_reason = str(getattr(response, "finish_reason", "") or "")
        generator_error = str(getattr(response, "error", "") or "")
        generator_model = str(getattr(response, "model_name", "") or "")
        text = response.text.strip()

        # Advanced cleanup for 700M model leaks
        import re

        # 1. Strip ChatML and system tags
        text = re.sub(r"<\|im_.*?\|>", "", text)

        # 2. Aggressively strip transcript prefixes (e.g., "You: [10:00]", "Mihir Shah:", "Them:")
        # Matches common patterns at the start of the string or after a newline
        prefixes_to_strip = [
            r"^(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)*|You|Me|Them|Assistant|System):\s*(?:\[\d{2}:\d{2}\])?\s*",
            r"^Jwalin Shah:\s*",
            r"^Mihir Shah:\s*",
            r"^Sangati Shah:\s*",
            r"^Friend:\s*",
            r"^vibe check,\s*",
            r"^vibe check\s*",
            r"^vibe:\s*",
            r"^vp,\s*",
            r"^vb,\s*",
            r"^vb\s*",
            r"^ugh,\s*",
            r"^vibe\s*",
            r"^vbc\s*",
        ]
        for p_regex in prefixes_to_strip:
            text = re.sub(p_regex, "", text, flags=re.IGNORECASE | re.MULTILINE)

        # 3. Strip trailing AI notes or explanations
        for tag in [
            "(Note:",
            "Note:",
            "<system>",
            "<style",
            "[lowercase]",
            "tone",
            "Energy:",
            "Energy",
        ]:
            if tag in text:
                text = text.split(tag)[0].strip()

        # 4. Strip any leading/trailing quotes often added by small models
        text = text.strip("\"' ")

        # 5. Persona Normalization: Force lowercase and strip periods/formal punctuation
        text = text.lower().strip()
        # Remove trailing periods/exclamations
        text = text.rstrip(".!")

        # 6. Strict Language Filter: Strip any non-ASCII or non-English characters
        # This catches Chinese, robotic symbols, and junk
        text = re.sub(r"[^\x00-\x7F]+", "", text)
        # Also strip specific robotic markers often left by small models
        text = text.replace("  ", " ").strip()

        word_cap_mode = config.reply_pipeline.reply_word_cap_mode
        if word_cap_mode == "hard_10":
            words = text.split()
            if len(words) > 10:
                text = " ".join(words[:10])

        if not text or text == "..." or len(text) < 2:
            logger.warning("[reply] LLM generated empty/junk text, using fallback.")
            text = "copy"

        logger.info("[reply] Generated response: %r", text)

        if config.reply_pipeline.reply_confidence_mode == "scored":
            from jarvis.core.generation.confidence import (
                compute_confidence,
                compute_example_diversity,
            )
            from jarvis.reply_service_utils import pressure_from_classification

            rag_similarity = 0.0
            if search_results:
                rag_similarity = float(search_results[0].get("similarity", 0.0) or 0.0)
            diversity = compute_example_diversity(search_results)
            pressure = pressure_from_classification(request.classification)
            confidence_score, confidence_label = compute_confidence(
                pressure=pressure,
                rag_similarity=rag_similarity,
                example_diversity=diversity,
                reply_length=len(text.split()),
                reply_text=text,
                incoming_text=incoming,
            )
        else:
            # Direct-to-LLM: simple confidence
            # Coherence check only
            base_conf = 0.65
            if text.lower().strip() == incoming.lower().strip():
                base_conf *= 0.5

            confidence_score = max(0.0, min(base_conf, 1.0))
            confidence_label = (
                "high"
                if confidence_score >= 0.7
                else "medium"
                if confidence_score >= 0.45
                else "low"
            )

        route_type = "generated_llm"
        route_reason = "generated"
        if (
            generator_model == "fallback"
            or generator_finish_reason in {"fallback", "error"}
        ):
            route_type = "generated_fallback"
            route_reason = generator_error or (
                f"generator_{generator_finish_reason or 'fallback'}"
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
                "type": route_type,
                "reason": route_reason,
                "confidence_label": confidence_label,
                "final_prompt": model_request.prompt,
                "generator_finish_reason": generator_finish_reason,
                "generator_error": generator_error,
                "generator_model": generator_model,
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
