from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jarvis.contracts.pipeline import ClassificationResult, GenerationResponse, MessageContext
    from jarvis.db import JarvisDB

logger = logging.getLogger(__name__)


def persist_reply_log(
    db: JarvisDB,
    context: MessageContext,
    classification: ClassificationResult,
    search_results: list[dict[str, Any]] | None,
    result: GenerationResponse,
    latency_ms: dict[str, float],
) -> None:
    """Persist a detailed log of the generation process for traceability."""
    try:
        chat_id = context.chat_id
        contact_id = context.sender_id or chat_id
        incoming_text = context.message_text

        classification_dict = {
            "category": classification.category.value,
            "urgency": classification.urgency.value,
            "confidence": classification.confidence,
            "metadata": classification.metadata,
        }

        # RAG context: extract content and scores
        rag_docs = []
        if search_results:
            for res in search_results:
                rag_docs.append(
                    {
                        "content": res.get("context_text", ""),
                        "response": res.get("reply_text", ""),
                        "similarity": float(res.get("similarity", 0.0)),
                        "source": res.get("chat_id", ""),
                    }
                )

        final_prompt = result.metadata.get("final_prompt", "")

        db.save_reply_log(
            chat_id=chat_id,
            contact_id=contact_id,
            incoming_text=incoming_text,
            classification_json=json.dumps(classification_dict),
            rag_context_json=json.dumps(rag_docs),
            final_prompt=final_prompt,
            response_text=result.response,
            confidence=result.confidence,
            metadata_json=json.dumps({"latency_ms": latency_ms, "metadata": result.metadata}),
        )
    except Exception as e:
        logger.debug(f"Failed to persist reply log: {e}")


def log_custom_generation(
    db: JarvisDB,
    chat_id: str | None,
    incoming_text: str,
    final_prompt: str,
    response_text: str,
    confidence: float = 0.5,
    category: str = "custom",
    rag_docs: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Log a generation event from outside the standard reply pipeline."""
    try:
        classification_json = json.dumps(
            {
                "category": category,
                "urgency": "medium",
                "confidence": confidence,
                "metadata": {},
            }
        )

        rag_context_json = json.dumps(rag_docs or [])

        db.save_reply_log(
            chat_id=chat_id,
            contact_id=chat_id,
            incoming_text=incoming_text,
            classification_json=classification_json,
            rag_context_json=rag_context_json,
            final_prompt=final_prompt,
            response_text=response_text,
            confidence=confidence,
            metadata_json=json.dumps(metadata or {}),
        )
    except Exception as e:
        logger.debug(f"Failed to log custom generation: {e}")
