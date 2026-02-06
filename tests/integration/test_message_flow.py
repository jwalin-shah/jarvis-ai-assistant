"""Integration tests for end-to-end message flow.

Tests the complete pipeline from receiving a message through
classification to generating a response.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from contracts.imessage import Message
from jarvis.router import ReplyRouter, reset_reply_router

from .conftest import create_mock_message


@pytest.fixture(autouse=True)
def reset_router():
    """Reset the router singleton before and after each test."""
    reset_reply_router()
    yield
    reset_reply_router()


@pytest.fixture(autouse=True)
def mock_health_check():
    """Mock health check to allow generation in tests."""
    with patch("jarvis.generation.can_use_llm", return_value=(True, "ok")):
        yield


class TestRouterGeneratesAllMessages:
    """Tests that the router generates responses for all non-empty messages."""

    @pytest.fixture
    def router(self):
        """Create a router with mocked dependencies."""
        mock_db = MagicMock()
        mock_db.get_contact.return_value = None
        mock_db.get_contact_by_chat_id.return_value = None
        mock_db.init_schema.return_value = None

        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []

        mock_gen = MagicMock()
        mock_gen.is_loaded.return_value = False
        mock_response = MagicMock()
        mock_response.text = "Generated response"
        mock_gen.generate.return_value = mock_response

        r = ReplyRouter(db=mock_db, generator=mock_gen)
        r._vec_searcher = mock_searcher
        return r

    def test_acknowledgment_generates(self, router):
        """Acknowledgments go through generation instead of canned response."""
        result = router.route("thanks")
        assert result["type"] == "generated"

    def test_ok_generates(self, router):
        """'ok' goes through generation."""
        result = router.route("ok")
        assert result["type"] == "generated"

    def test_question_generates(self, router):
        """Questions go through generation."""
        result = router.route("what time is the meeting?")
        assert result["type"] == "generated"

    def test_context_dependent_generates(self, router):
        """Context-dependent messages go through generation."""
        result = router.route("what time?", thread=["Dinner tonight?", "Sure!"])
        assert result["type"] == "generated"


class TestRouterClarification:
    """Tests for clarification request handling."""

    @pytest.fixture
    def router(self):
        """Create a router with mocked dependencies."""
        mock_db = MagicMock()
        mock_db.get_contact.return_value = None
        mock_db.get_contact_by_chat_id.return_value = None
        mock_db.init_schema.return_value = None

        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []

        mock_gen = MagicMock()
        mock_gen.is_loaded.return_value = False
        mock_response = MagicMock()
        mock_response.text = "Generated response"
        mock_gen.generate.return_value = mock_response

        r = ReplyRouter(db=mock_db, generator=mock_gen)
        r._vec_searcher = mock_searcher
        return r

    def test_clarification_response_format(self, router):
        """Empty input returns clarification with correct format."""
        result = router.route("")

        assert result["type"] == "clarify"
        assert result["confidence"] == "low"
        assert "empty" in result["response"].lower()


class TestRouterEmptyInput:
    """Tests for empty input handling."""

    @pytest.fixture
    def router(self):
        """Create a router with mocked dependencies."""
        mock_db = MagicMock()
        mock_db.get_contact.return_value = None
        mock_db.get_contact_by_chat_id.return_value = None
        mock_db.init_schema.return_value = None

        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []

        mock_gen = MagicMock()
        mock_gen.is_loaded.return_value = False
        mock_response = MagicMock()
        mock_response.text = "Generated response"
        mock_gen.generate.return_value = mock_response

        r = ReplyRouter(db=mock_db, generator=mock_gen)
        r._vec_searcher = mock_searcher
        return r

    def test_empty_string_returns_clarify(self, router):
        """Empty string returns clarification request."""
        result = router.route("")

        assert result["type"] == "clarify"
        assert "empty message" in result["response"].lower()

    def test_whitespace_returns_clarify(self, router):
        """Whitespace-only input returns clarification request."""
        result = router.route("   ")

        assert result["type"] == "clarify"


class TestEndToEndMessagePipeline:
    """Tests for the complete message pipeline."""

    @pytest.fixture
    def mock_messages(self) -> list[Message]:
        """Sample conversation messages."""
        return [
            create_mock_message(
                "Hey, are you free for dinner?",
                is_from_me=False,
                msg_id=1,
                date=datetime(2024, 1, 15, 18, 0),
            ),
            create_mock_message(
                "Sure, what time?",
                is_from_me=True,
                msg_id=2,
                date=datetime(2024, 1, 15, 18, 5),
            ),
            create_mock_message(
                "7pm at the usual place?",
                is_from_me=False,
                msg_id=3,
                date=datetime(2024, 1, 15, 18, 10),
            ),
        ]

    @patch("jarvis.generation.can_use_llm", return_value=(True, "ok"))
    def test_message_with_context_generates_response(self, mock_health, mock_messages):
        """Message with conversation context generates appropriate response."""
        mock_db = MagicMock()
        mock_db.get_contact.return_value = None
        mock_db.get_contact_by_chat_id.return_value = None
        mock_db.init_schema.return_value = None

        mock_gen = MagicMock()
        mock_gen.is_loaded.return_value = False
        mock_response = MagicMock()
        mock_response.text = "Sounds great! See you there!"
        mock_gen.generate.return_value = mock_response

        router = ReplyRouter(db=mock_db, generator=mock_gen)

        thread = [
            "[Alice]: Hey, are you free for dinner?",
            "[You]: Sure, what time?",
            "[Alice]: 7pm at the usual place?",
        ]

        result = router.route(
            "7pm at the usual place?",
            thread=thread,
        )

        assert "response" in result
        assert result["type"] == "generated"


class TestRouterStats:
    """Tests for router statistics."""

    @pytest.fixture
    def router(self):
        """Create a router with mocked dependencies."""
        mock_db = MagicMock()
        mock_db.get_contact.return_value = None
        mock_db.get_contact_by_chat_id.return_value = None
        mock_db.init_schema.return_value = None
        mock_db.get_stats.return_value = {"pairs": 100}
        mock_db.get_active_index.return_value = None

        return ReplyRouter(db=mock_db)

    def test_get_routing_stats_returns_db_stats(self, router):
        """Get routing stats returns database statistics."""
        stats = router.get_routing_stats()

        assert "db_stats" in stats
        assert stats["db_stats"]["pairs"] == 100
        assert "index_available" in stats
