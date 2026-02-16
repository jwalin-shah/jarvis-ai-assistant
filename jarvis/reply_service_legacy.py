"""Legacy compatibility helpers for ReplyService."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from jarvis.contracts.pipeline import MessageContext


def route_legacy(
    service: Any,
    *,
    incoming: str,
    contact_id: int | None = None,
    thread: list[str] | None = None,
    chat_id: str | None = None,
    conversation_messages: list[Any] | None = None,
    context: MessageContext | None = None,
) -> dict[str, Any]:
    """Compatibility route API used by socket/api handlers and tests."""
    if context is None:
        context = MessageContext(
            chat_id=chat_id or "",
            message_text=incoming or "",
            is_from_me=False,
            timestamp=datetime.now(UTC),
            metadata={"thread": thread or [], "contact_id": contact_id},
        )
    if not thread and conversation_messages:
        context.metadata["thread"] = service._build_thread_context(conversation_messages)

    response = service.generate_reply(context, classification=None)
    return cast(dict[str, Any], service._to_legacy_response(response))


def get_routing_stats(service: Any) -> dict[str, Any]:
    """Return legacy routing stats payload."""
    return {"db_stats": service.db.get_stats(), "index_available": True}
