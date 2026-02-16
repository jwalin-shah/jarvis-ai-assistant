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
from jarvis.classifiers.cascade import classify_with_cascade
from jarvis.classifiers.response_mobilization import ResponsePressure
from jarvis.config import get_config
from jarvis.contracts.pipeline import (
    GenerationRequest,
    GenerationResponse,
    MessageContext,
    RAGDocument,
)
from jarvis.core.generation.confidence import (
    UNCERTAIN_SIGNALS,
    compute_confidence,
    compute_example_diversity,
)
from jarvis.embedding_adapter import CachedEmbedder, get_embedder
from jarvis.prompts import build_prompt_from_request, get_category_config
from jarvis.reply_service_utils import safe_float
from jarvis.search.hybrid_search import get_hybrid_searcher

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

    if classification_result is None:
        mobilization = classify_with_cascade(normalized_incoming)
        classification_result = service._build_classification_result(
            normalized_incoming,
            thread_messages,
            mobilization,
        )

    if search_results is None:
        hybrid_searcher = get_hybrid_searcher()
        search_results = hybrid_searcher.search(query=normalized_incoming, limit=5, rerank=True)

    logger.info(
        "[stream] RAG search: %d results, top_similarity=%.3f",
        len(search_results),
        search_results[0].get("similarity", 0.0) if search_results else 0.0,
    )

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
        search_results=search_results,
        contact=contact,
        thread=thread_messages,
        instruction=instruction,
    )
    model_request = service._to_model_generation_request(pipeline_request)

    similarity = search_results[0].get("similarity", 0.0) if search_results else 0.0
    example_diversity = service._compute_example_diversity(search_results)
    pressure = service._pressure_from_classification(classification_result)

    base_confidence = {
        ResponsePressure.HIGH: 0.85,
        ResponsePressure.MEDIUM: 0.65,
        ResponsePressure.LOW: 0.45,
        ResponsePressure.NONE: 0.30,
    }[pressure]

    config = get_config()
    if similarity < config.similarity_thresholds.rag_min_similarity:
        base_confidence *= config.similarity_thresholds.low_similarity_penalty
    if example_diversity < 0.3:  # Keep 0.3 as it's not in config yet
        base_confidence *= config.similarity_thresholds.medium_similarity_factor

    if base_confidence >= config.similarity_thresholds.high_confidence:
        confidence = "high"
    elif base_confidence >= config.similarity_thresholds.medium_confidence:
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
    category_name = str(classification.metadata.get("category_name", classification.category.value))
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
        logger.debug("[build] Fetching context for %s...", chat_id[:12])
        t_ctx_start = time.perf_counter()

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Launch both fetches in parallel
            f_facts = executor.submit(service._fetch_contact_facts, context, chat_id)
            f_graph = executor.submit(service._fetch_graph_context, context, chat_id)

            # Wait for both to complete (with safety timeout)
            try:
                f_facts.result(timeout=2.0)
            except Exception as e:
                logger.warning("[build] Parallel facts fetch failed: %s", e)

            try:
                f_graph.result(timeout=2.0)
            except Exception as e:
                logger.warning("[build] Parallel graph fetch failed: %s", e)

        logger.debug(
            "[build] Context fetching took %.1fms", (time.perf_counter() - t_ctx_start) * 1000
        )
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

    optimized_examples = get_optimized_examples(category_name)
    category_exchanges = [(ex.context, ex.output) for ex in optimized_examples]

    config = get_config()
    top_results = search_results[: config.retrieval.top_k_rag]
    similar_exchanges = [
        (str(r.get("context_text", "")), str(r.get("reply_text", ""))) for r in top_results
    ]
    rag_rerank_scores = [
        service._safe_float(r.get("rerank_score"), default=0.0) for r in top_results
    ]

    all_exchanges = similar_exchanges + [
        ex for ex in category_exchanges if ex not in similar_exchanges
    ]
    all_rerank_scores = rag_rerank_scores + [0.0] * (len(all_exchanges) - len(similar_exchanges))

    if cached_embedder is None:
        cached_embedder = get_embedder()

    logger.debug("[build] Deduplicating %d examples...", len(all_exchanges))
    t_dedupe_start = time.perf_counter()
    all_exchanges = dedupe_examples(
        service,
        examples=all_exchanges,
        embedder=cached_embedder,
        rerank_scores=all_rerank_scores,
    )
    all_exchanges = all_exchanges[: config.retrieval.top_k_examples]
    logger.debug("[build] Dedupe took %.1fms", (time.perf_counter() - t_dedupe_start) * 1000)

    logger.info(
        "[build] Context: %d messages, %d RAG docs, %d examples, contact=%s",
        len(context_messages),
        len(top_results),
        len(all_exchanges),
        context.metadata.get("contact_name", "unknown"),
    )

    rag_documents: list[RAGDocument] = []
    for result in top_results:
        context_text = str(result.get("context_text", "")).strip()
        if not context_text:
            continue
        rag_documents.append(
            RAGDocument(
                content=context_text,
                source=str(result.get("topic") or chat_id or "rag"),
                score=service._safe_float(result.get("similarity"), default=0.0),
                metadata={
                    "response_text": str(result.get("response_text", "")),
                    "rerank_score": service._safe_float(result.get("rerank_score"), default=0.0),
                },
            )
        )

    return GenerationRequest(
        context=context,
        classification=classification,
        extraction=None,
        retrieved_docs=rag_documents,
        few_shot_examples=[
            {"input": ctx, "output": response}
            for ctx, response in all_exchanges
            if ctx and response
        ],
    )


def to_model_generation_request(service: Any, request: GenerationRequest) -> ModelGenerationRequest:
    """Convert pipeline request into model-native request."""
    pressure = service._pressure_from_classification(request.classification)
    prompt = build_prompt_from_request(request)
    return ModelGenerationRequest(
        prompt=prompt,
        max_tokens=service._max_tokens_for_pressure(pressure),
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
    pressure = service._pressure_from_classification(request.classification)
    logger.info("[reply] Incoming: %r | pressure=%s", incoming[:80], pressure.value)

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

    cat_conf = safe_float(
        request.classification.metadata.get("category_confidence"),
        default=request.classification.confidence,
    )
    has_thin_context = not thread or len(thread) < 2
    is_short_msg = len(incoming.split()) <= 3
    if cat_conf < 0.4 and pressure == ResponsePressure.NONE and is_short_msg and has_thin_context:
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
        model_request = to_model_generation_request(service, request)
        final_prompt = model_request.prompt

        # --- Best of N Generation ---
        # Generate 2 candidates with different temperatures to explore style options
        # LFM models often prefer lower temperatures (0.1-0.3) for stability.
        candidates = []
        temperatures = [0.1, 0.3]

        for temp in temperatures:
            # Clone request with specific temp
            variant_req = model_request
            variant_req.temperature = temp
            # Ensure repetition penalty is set if not already
            # Use centralized default if not set
            if not variant_req.repetition_penalty:
                from jarvis.prompts.generation_config import DEFAULT_REPETITION_PENALTY

                variant_req.repetition_penalty = DEFAULT_REPETITION_PENALTY

            response = service.generator.generate(variant_req)
            text = response.text.strip()

            # Basic cleanup
            if "</reply>" in text:
                text = text.split("</reply>")[0].strip()

            for tag in ["(Note:", "Note:", "<system>", "<style", "[lowercase]"]:
                if tag in text:
                    text = text.split(tag)[0].strip()

            if text:
                candidates.append(text)

        if not candidates:
            return GenerationResponse(
                response="...",
                confidence=0.1,
                metadata={"reason": "empty_candidates"},
            )

        # --- Heuristic Reranking ---
        # Prefer: shorter, lowercase, fewer emojis (unless style dictates otherwise)
        # Penalize: hallucinated names/entities not in context
        best_candidate = candidates[0]
        best_score = -float("inf")

        # Gather context tokens for hallucination check
        cname = str(request.context.metadata.get("contact_name", "them"))
        context_text = (incoming + " " + " ".join(thread or []) + " " + cname).lower()
        context_tokens = set(re.findall(r"\w+", context_text))

        for cand in candidates:
            score = 0.0

            # Length penalty (prefer concise)
            words = cand.split()
            num_words = len(words)
            if num_words > 20:
                score -= 1.0
            config = get_config()
            if num_words < config.text_processing.short_reply_word_threshold:
                score += config.scoring_weights.short_reply_bonus

            # Lowercase bonus (casual)
            if cand.islower():
                score += 1.0

            # Emoji penalty (unless very short)
            emojis = len(re.findall(r"[^\w\s,.]", cand))
            if emojis > 1 and num_words > config.text_processing.emoji_penalty_word_threshold:
                score -= config.scoring_weights.emoji_penalty

            # Hallucination Guard: Check for capitalized words (potential names) not in context
            # We ignore common starting words or 'I', 'I'm' etc via simple length/stoplist check
            potential_names = [w for w in words if w[0].isupper() and len(w) > 1]
            hallucination_penalty = 0.0
            for name in potential_names:
                clean_name = re.sub(r"[^\w]", "", name).lower()
                common_words = ["hey", "yeah", "sure", "okay", "wow", "lol", "omg"]
                if clean_name not in context_tokens and clean_name not in common_words:
                    hallucination_penalty += 2.0  # Heavy penalty for invented names

            # Sentiment Mismatch Guard
            sad_emojis = ["😢", "😭", "😔", "💔", "☹️"]
            has_sad_response = any(e in cand for e in sad_emojis)
            has_sad_input = any(e in incoming for e in sad_emojis)
            if has_sad_response and not has_sad_input:
                hallucination_penalty += 1.5

            score -= hallucination_penalty

            if score > best_score:
                best_score = score
                best_candidate = cand

        text = best_candidate
        logger.info("[reply] Selected candidate: %r (score=%.1f)", text, best_score)

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
            safe_float(search_results[0].get("similarity"), default=0.0) if search_results else 0.0
        )
        example_diversity = compute_example_diversity(search_results)
        reply_length = len(text.split())
        rerank_score = (
            safe_float(search_results[0].get("rerank_score"), default=0.0)
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
        logger.info(
            "[reply] Confidence: %.2f (%s) | similarity=%.3f | diversity=%.3f",
            confidence_score,
            confidence_label,
            similarity,
            example_diversity,
        )

        config = get_config()
        similar_triggers = [
            str(row.get("context_text", ""))
            for row in search_results[: config.retrieval.top_k_rag]
            if row.get("context_text")
        ]
        used_docs = [
            doc.content
            for doc in request.retrieved_docs[: config.retrieval.top_k_rag]
            if doc.content
        ]

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
