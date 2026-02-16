"""Generation/request-building helpers extracted from ReplyService."""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import numpy as np

from contracts.models import GenerationRequest as ModelGenerationRequest
from jarvis.classifiers.cascade import classify_with_cascade
from jarvis.classifiers.response_mobilization import ResponsePressure
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

    incoming = context.message_text.strip()
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
    if contact and contact.display_name:
        context.metadata.setdefault("contact_name", contact.display_name)

    if chat_id:
        with ThreadPoolExecutor(max_workers=2) as pool:
            facts_future = pool.submit(service._fetch_contact_facts, context, chat_id)
            graph_future = pool.submit(service._fetch_graph_context, context, chat_id)
            facts_future.result()
            graph_future.result()

    instruction = service._resolve_instruction(
        instruction,
        category_name,
        category_config,
        classification,
    )
    context.metadata["instruction"] = instruction or ""

    if len(search_results) > 3 and service.reranker:
        search_results = service.reranker.rerank(
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
        service._safe_float(r.get("rerank_score"), default=0.0) for r in top_results
    ]

    all_exchanges = similar_exchanges + [
        ex for ex in category_exchanges if ex not in similar_exchanges
    ]
    all_rerank_scores = rag_rerank_scores + [0.0] * (len(all_exchanges) - len(similar_exchanges))

    if cached_embedder is None:
        cached_embedder = get_embedder()
    all_exchanges = dedupe_examples(
        service,
        examples=all_exchanges,
        embedder=cached_embedder,
        rerank_scores=all_rerank_scores,
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
        response = service.generator.generate(model_request)
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
