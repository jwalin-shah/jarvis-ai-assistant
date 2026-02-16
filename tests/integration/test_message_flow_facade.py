from __future__ import annotations

from types import SimpleNamespace

from jarvis.reply_service import ReplyService


class _ReplyServiceStub(ReplyService):
    def __init__(self, db=None, generator=None, imessage_reader=None) -> None:
        super().__init__(
            db=SimpleNamespace(get_stats=lambda: {"ok": True}),
            generator=generator,
            imessage_reader=imessage_reader,
        )
        self._last_context = None

    def generate_reply(self, context, classification=None):
        self._last_context = context
        return SimpleNamespace(
            response="Sounds good, let's do 1pm.",
            confidence=0.81,
            metadata={
                "type": "generated",
                "similarity_score": 0.77,
                "reason": "stubbed",
                "similar_triggers": ["lunch plans"],
            },
        )


def test_reply_router_route_maps_service_response_to_legacy_shape() -> None:
    service = _ReplyServiceStub()

    result = service.route_legacy(
        incoming="Want lunch tomorrow?",
        chat_id="chat-1",
        thread=["Me: sounds good", "Them: maybe tomorrow"],
    )

    assert result["type"] == "generated"
    assert result["response"]
    assert result["confidence"] == "high"
    assert result["confidence_score"] == 0.81
    assert result["similarity_score"] == 0.77


def test_reply_router_accepts_conversation_message_objects() -> None:
    service = _ReplyServiceStub()

    conversation_messages = [
        {"text": "ping", "is_from_me": False, "sender_name": "Alex"},
        {"text": "pong", "is_from_me": True, "sender_name": "Me"},
    ]
    result = service.route_legacy(
        incoming="next?",
        chat_id="chat-2",
        conversation_messages=conversation_messages,
    )

    assert result["type"] == "generated"
