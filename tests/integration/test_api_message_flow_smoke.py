from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from api.main import app
from contracts.imessage import Message
from tests.helpers_api import api_client_with_reader


def test_drafts_reply_smoke_with_stubbed_pipeline(monkeypatch) -> None:
    """Smoke-test API message flow through /drafts/reply with lightweight stubs."""
    import api.routers.drafts as drafts_router

    reader = MagicMock()
    reader.get_messages.return_value = [
        Message(
            id=1,
            chat_id="chat-smoke",
            sender="+15555550100",
            sender_name="Alex",
            text="Want to meet at 1?",
            date=datetime(2026, 2, 10, 10, 0),
            is_from_me=False,
            attachments=[],
            reactions=[],
        ),
        Message(
            id=2,
            chat_id="chat-smoke",
            sender="me",
            sender_name=None,
            text="Sure, works for me.",
            date=datetime(2026, 2, 10, 10, 1),
            is_from_me=True,
            attachments=[],
            reactions=[],
        ),
    ]

    class _ReplyServiceStub:
        def generate_reply(self, *args, **kwargs):
            return SimpleNamespace(
                response="Yes, 1pm works. See you there.",
                confidence=0.86,
                metadata={"final_prompt": "stub-prompt"},
            )

        def log_custom_generation(self, **kwargs) -> None:
            return None

    monkeypatch.setattr(
        drafts_router,
        "run_classification_and_search",
        lambda *_args, **_kwargs: (object(), [{"similarity": 0.91, "context_text": "ctx"}]),
    )
    monkeypatch.setattr("jarvis.reply_service.get_reply_service", lambda: _ReplyServiceStub())

    with api_client_with_reader(app, reader, raise_server_exceptions=False) as client:
        response = client.post(
            "/drafts/reply",
            json={"chat_id": "chat-smoke", "num_suggestions": 1, "context_messages": 10},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["suggestions"]
    assert payload["suggestions"][0]["text"]
    assert payload["context_used"]["last_message"] == "Want to meet at 1?"
