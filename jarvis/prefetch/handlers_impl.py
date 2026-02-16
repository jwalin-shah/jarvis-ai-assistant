"""Concrete implementations of prefetch handlers."""

from __future__ import annotations

import logging
from typing import Any

from .handlers import PrefetchHandler

logger = logging.getLogger(__name__)

class DraftReplyHandler(PrefetchHandler):
    @property
    def required_params(self) -> list[str]:
        return ["chat_id"]

    def execute(self, params: dict[str, Any]) -> dict[str, Any] | None:
        chat_id = params["chat_id"]

        # Import router lazily to avoid circular imports
        from jarvis.prefetch.executor import get_executor
        from jarvis.router import get_reply_router

        router = get_reply_router()
        executor = get_executor()

        # Get recent messages using shared reader
        reader = executor._get_reader()
        messages = reader.get_messages(chat_id, limit=10)

        if not messages:
            return None

        # Find last incoming message
        last_incoming = None
        for msg in messages:
            if not msg.is_from_me and msg.text:
                last_incoming = msg.text
                break

        if not last_incoming:
            return None

        result = router.route(
            incoming=last_incoming,
            chat_id=chat_id,
            conversation_messages=messages,
        )

        confidence = float(result.get("confidence_score", 0.6))

        return {
            "suggestions": [
                {
                    "text": result.get("response", ""),
                    "confidence": confidence,
                }
            ],
            "prefetched": True,
        }

class EmbeddingHandler(PrefetchHandler):
    @property
    def required_params(self) -> list[str]:
        return ["texts"]

    def execute(self, params: dict[str, Any]) -> dict[str, Any] | None:
        texts = params["texts"]
        from jarvis.embedding_adapter import get_embedder

        embedder = get_embedder()
        embeddings = embedder.encode(texts)

        return {
            "embeddings": embeddings,
            "texts": texts,
            "prefetched": True,
        }

class ContactProfileHandler(PrefetchHandler):
    @property
    def required_params(self) -> list[str]:
        return ["chat_id"]

    def execute(self, params: dict[str, Any]) -> dict[str, Any] | None:
        chat_id = params["chat_id"]
        from jarvis.db import get_db

        db = get_db()
        contact = db.get_contact_by_chat_id(chat_id)

        if contact:
            return {
                "contact": {
                    "id": contact.id,
                    "display_name": contact.display_name,
                    "relationship": contact.relationship,
                    "style_notes": contact.style_notes,
                },
                "prefetched": True,
            }
        return None

class ModelWarmHandler(PrefetchHandler):
    @property
    def required_params(self) -> list[str]:
        return ["model_type"]

    def execute(self, params: dict[str, Any]) -> dict[str, Any] | None:
        model_type = params["model_type"]
        if model_type == "llm":
            from models.loader import get_model
            model = get_model()
            if model and not model.is_loaded():
                model.load()
            return {"model": "llm", "warm": True}
        elif model_type == "embeddings":
            from jarvis.embedding_adapter import get_embedder
            embedder = get_embedder()
            embedder.encode(["warmup test"])
            return {"model": "embeddings", "warm": True}
        return None

class SearchResultsHandler(PrefetchHandler):
    @property
    def required_params(self) -> list[str]:
        return ["query"]

    def execute(self, params: dict[str, Any]) -> dict[str, Any] | None:
        query = params["query"]
        from jarvis.db import get_db
        from jarvis.search.vec_search import get_vec_searcher

        db = get_db()
        searcher = get_vec_searcher(db)
        results = searcher.search(query=query, k=10)

        return {
            "query": query,
            "results": [
                {"trigger": r.last_trigger, "response": r.last_response, "sim": r.similarity}
                for r in results
                if r.last_trigger
            ],
            "prefetched": True,
        }

class VecIndexHandler(PrefetchHandler):
    @property
    def required_params(self) -> list[str]:
        return []

    def execute(self, params: dict[str, Any]) -> dict[str, Any] | None:
        from jarvis.db import get_db
        from jarvis.search.vec_search import get_vec_searcher

        db = get_db()
        searcher = get_vec_searcher(db)
        if searcher._vec_tables_exist():
            return {"loaded": True}
        return None
