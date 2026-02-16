"""Pipeline helpers for draft endpoints."""

from __future__ import annotations

from typing import Any

from contracts.models import GenerationRequest
from jarvis.classifiers.cascade import classify_with_cascade
from jarvis.classifiers.classification_result import build_classification_result
from jarvis.search.hybrid_search import get_hybrid_searcher


def run_classification_and_search(
    last_message: str,
    thread: list[str],
) -> tuple[Any, list[dict[str, Any]]]:
    """Run mobilization/classification and hybrid retrieval."""
    mobilization = classify_with_cascade(last_message)
    classification = build_classification_result(last_message, thread, mobilization)
    searcher = get_hybrid_searcher()
    search_results = searcher.search(query=last_message, limit=5, rerank=True)
    return classification, search_results


def generate_summary(
    generator: object,
    prompt: str,
    summary_examples: list[dict[str, str]],
) -> str:
    """Generate a summary synchronously for threadpool execution."""
    gen_request = GenerationRequest(
        prompt=prompt,
        context_documents=[],
        few_shot_examples=summary_examples,
        max_tokens=500,
        temperature=0.5,
    )
    response = generator.generate(gen_request)  # type: ignore[attr-defined]
    return str(response.text) if response.text else ""


def route_reply_sync(
    chat_id: str,
    last_message: str,
    thread_context: list[str],
) -> dict[str, Any]:
    """Run reply router synchronously for threadpool execution."""
    from jarvis.router import get_reply_router

    router = get_reply_router()
    return router.route(
        incoming=last_message,
        chat_id=chat_id,
        thread=thread_context,
    )
