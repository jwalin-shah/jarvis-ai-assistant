"""Reply Router Facade - Delegates to ReplyService.

Note: This module is now a thin facade for jarvis.reply_service.
New code should use get_reply_service() directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypedDict

from jarvis.contracts.pipeline import GenerationResponse, MessageContext
from jarvis.errors import ErrorCode, JarvisError
from jarvis.reply_service import ReplyService

if TYPE_CHECKING:
    from integrations.imessage.reader import ChatDBReader
    from jarvis.db import JarvisDB
    from models import MLXGenerator

logger = logging.getLogger(__name__)


class RouterError(JarvisError):
    """Raised when routing operations fail."""
    default_message = "Router operation failed"
    default_code = ErrorCode.UNKNOWN


class IndexNotAvailableError(RouterError):
    """Raised when vector index is not available."""
    default_message = "Vector index not available. Run 'jarvis db build-index' first."


class StatsResponse(TypedDict, total=False):
    """Typed dictionary for routing statistics response."""
    db_stats: dict[str, Any]
    index_available: bool
    index_vectors: int
    index_type: str


class ReplyRouter:
    """Thin wrapper around ReplyService for backward compatibility."""

    def __init__(
        self,
        db: JarvisDB | None = None,
        generator: MLXGenerator | None = None,
        imessage_reader: ChatDBReader | None = None,
    ) -> None:
        self._service = ReplyService(db, generator, imessage_reader)

    def route(
        self,
        incoming: str,
        contact_id: int | None = None,
        thread: list[str] | None = None,
        chat_id: str | None = None,
        conversation_messages: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Route an incoming message (legacy entry point)."""
        # Convert to MessageContext
        from datetime import UTC, datetime
        context = MessageContext(
            chat_id=chat_id or "",
            message_text=incoming or "",
            is_from_me=False,
            timestamp=datetime.now(UTC),
            metadata={
                "thread": thread or [],
                "contact_id": contact_id,
            }
        )

        # Add thread from conversation_messages if provided
        if not thread and conversation_messages:
            context.metadata["thread"] = self._build_thread_context(conversation_messages)

        # generate_reply handles classification if None
        response = self._service.generate_reply(context, classification=None)
        return self._to_legacy_response(response)

    def route_message(self, context: MessageContext) -> GenerationResponse:
        """Route a typed message context."""
        return self._service.generate_reply(context, classification=None)

    def _build_thread_context(self, conversation_messages: list[Any]) -> list[str]:
        thread: list[str] = []
        for msg in reversed(conversation_messages):
            if isinstance(msg, dict):
                msg_text = msg.get("text", "")
            else:
                msg_text = getattr(msg, "text", None) or ""
            if msg_text:
                if isinstance(msg, dict):
                    is_from_me = msg.get("is_from_me", False)
                else:
                    is_from_me = getattr(msg, "is_from_me", False)
                if isinstance(msg, dict):
                    sender = msg.get("sender_name") or msg.get("sender", "")
                else:
                    sender = getattr(msg, "sender_name", None) or getattr(msg, "sender", "")
                prefix = "Me" if is_from_me else sender
                thread.append(f"{prefix}: {msg_text}")
        return thread[-10:]

    def _to_legacy_response(self, response: GenerationResponse) -> dict[str, Any]:
        metadata = response.metadata
        def to_label(conf: float) -> str:
            if conf >= 0.7:
                return "high"
            if conf >= 0.45:
                return "medium"
            return "low"

        return {
            "type": str(metadata.get("type", "generated")),
            "response": response.response,
            "confidence": to_label(response.confidence),
            "confidence_score": response.confidence,
            "similarity_score": float(metadata.get("similarity_score", 0.0)),
            "similar_triggers": metadata.get("similar_triggers"),
            "reason": str(metadata.get("reason", "")),
        }

    def get_routing_stats(self) -> StatsResponse:
        return {
            "db_stats": self._service.db.get_stats(),
            "index_available": True,  # Assume true if service is up
        }

    def close(self) -> None:
        if self._service.imessage_reader:
            self._service.imessage_reader.close()


_router: ReplyRouter | None = None

def get_reply_router() -> ReplyRouter:
    global _router
    if _router is None:
        _router = ReplyRouter()
    return _router

def reset_reply_router() -> None:
    global _router
    if _router:
        _router.close()
    _router = None

__all__ = [
    "RouterError",
    "IndexNotAvailableError",
    "ReplyRouter",
    "get_reply_router",
    "reset_reply_router",
]
