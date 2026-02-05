"""Unit tests for JARVIS Reply Router.

Tests cover routing logic, mobilization-based generation, singleton pattern,
and edge cases.

The ReplyRouter routes all non-empty messages through LLM generation with RAG context.
Response mobilization (Stivers & Rossano 2010) informs the prompt, not the routing decision.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jarvis.db import Contact
from jarvis.router import (
    CONTEXT_THRESHOLD,
    GENERATE_THRESHOLD,
    QUICK_REPLY_THRESHOLD,
    ReplyRouter,
    RouterError,
    RouteResult,
    get_reply_router,
    reset_reply_router,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_db() -> MagicMock:
    """Create a mock JarvisDB."""
    db = MagicMock()
    db.init_schema = MagicMock()
    db.get_contact = MagicMock(return_value=None)
    db.get_contact_by_chat_id = MagicMock(return_value=None)
    db.get_stats = MagicMock(return_value={"pairs": 100, "contacts": 10})
    db.get_active_index = MagicMock(return_value=None)
    return db


@pytest.fixture
def mock_index_searcher() -> MagicMock:
    """Create a mock TriggerIndexSearcher."""
    searcher = MagicMock()
    searcher.search_with_pairs = MagicMock(return_value=[])
    return searcher


@pytest.fixture
def mock_generator() -> MagicMock:
    """Create a mock MLXGenerator."""
    generator = MagicMock()
    generator.is_loaded = MagicMock(return_value=False)
    mock_response = MagicMock()
    mock_response.text = "Generated response"
    generator.generate = MagicMock(return_value=mock_response)
    return generator


@pytest.fixture
def router(
    mock_db: MagicMock,
    mock_index_searcher: MagicMock,
    mock_generator: MagicMock,
) -> ReplyRouter:
    """Create a ReplyRouter with all mocked dependencies."""
    return ReplyRouter(
        db=mock_db,
        index_searcher=mock_index_searcher,
        generator=mock_generator,
    )


@pytest.fixture
def sample_contact() -> Contact:
    """Create a sample contact for tests."""
    return Contact(
        id=1,
        chat_id="chat123",
        display_name="John Doe",
        phone_or_email="+1234567890",
        relationship="friend",
        style_notes="casual, uses emojis",
    )


# =============================================================================
# ReplyRouter Initialization Tests
# =============================================================================


class TestReplyRouterInit:
    """Tests for ReplyRouter initialization."""

    def test_init_with_all_dependencies(
        self,
        mock_db: MagicMock,
        mock_index_searcher: MagicMock,
        mock_generator: MagicMock,
    ) -> None:
        """Test initialization with all dependencies provided."""
        router = ReplyRouter(
            db=mock_db,
            index_searcher=mock_index_searcher,
            generator=mock_generator,
        )
        assert router._db is mock_db
        assert router._index_searcher is mock_index_searcher
        assert router._generator is mock_generator

    def test_init_with_no_dependencies(self) -> None:
        """Test initialization with no dependencies (lazy loading)."""
        router = ReplyRouter()
        assert router._db is None
        assert router._index_searcher is None
        assert router._generator is None

    def test_db_property_creates_default(self) -> None:
        """Test db property creates default instance when None."""
        with patch("jarvis.router.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db

            router = ReplyRouter()
            _ = router.db

            mock_get_db.assert_called_once()
            mock_db.init_schema.assert_called_once()

    def test_index_searcher_property_creates_default(self) -> None:
        """Test index_searcher property creates default instance when None."""
        mock_db = MagicMock()
        mock_db.init_schema = MagicMock()

        with patch("jarvis.router.get_db", return_value=mock_db):
            with patch("jarvis.index.TriggerIndexSearcher") as mock_searcher_cls:
                mock_searcher = MagicMock()
                mock_searcher_cls.return_value = mock_searcher

                router = ReplyRouter()
                router._index_searcher = None
                result = router.index_searcher

                assert result is mock_searcher

    def test_generator_property_creates_default(self) -> None:
        """Test generator property creates default instance when None."""
        mock_gen = MagicMock()
        router = ReplyRouter(generator=mock_gen)
        result = router.generator
        assert result is mock_gen


# =============================================================================
# Always Generates Tests
# =============================================================================


class TestAlwaysGenerates:
    """Tests that all non-empty messages go through LLM generation."""

    @pytest.mark.parametrize(
        "message",
        [
            "want to grab lunch?",
            "what time?",
            "ok",
            "thanks",
            "how are you?",
            "are you coming to the party?",
            "I'm doing well",
            "lol",
            "The weather is nice today",
            "sounds good to me",
        ],
    )
    def test_all_messages_generate(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
        mock_generator: MagicMock,
        message: str,
    ) -> None:
        """Test that all non-empty messages route to generation."""
        mock_index_searcher.search_with_pairs.return_value = []

        result = router.route(message)

        assert result["type"] == "generated"
        mock_generator.generate.assert_called()

    def test_empty_message_clarifies(self, router: ReplyRouter) -> None:
        """Test that empty messages return clarify."""
        result = router.route("")
        assert result["type"] == "clarify"
        assert "empty" in result["response"].lower()

    def test_whitespace_only_clarifies(self, router: ReplyRouter) -> None:
        """Test that whitespace-only messages return clarify."""
        result = router.route("   ")
        assert result["type"] == "clarify"


# =============================================================================
# Mobilization Integration Tests
# =============================================================================


class TestMobilizationIntegration:
    """Tests that mobilization pressure maps to confidence correctly."""

    def test_high_pressure_high_confidence(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
        mock_generator: MagicMock,
    ) -> None:
        """Test HIGH pressure messages produce high confidence."""
        mock_index_searcher.search_with_pairs.return_value = []

        result = router.route("Can you pick me up at 5?")

        assert result["type"] == "generated"
        assert result["confidence"] == "high"

    def test_low_pressure_medium_confidence(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
        mock_generator: MagicMock,
    ) -> None:
        """Test LOW pressure messages produce medium confidence."""
        mock_index_searcher.search_with_pairs.return_value = []

        result = router.route("I think the weather is nice")

        assert result["type"] == "generated"
        assert result["confidence"] == "medium"

    def test_medium_pressure_medium_confidence(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
        mock_generator: MagicMock,
    ) -> None:
        """Test MEDIUM pressure messages produce medium confidence."""
        mock_index_searcher.search_with_pairs.return_value = []

        result = router.route("I got the job!!")

        assert result["type"] == "generated"
        assert result["confidence"] == "medium"


# =============================================================================
# Generate Path Tests
# =============================================================================


class TestRouteGeneratePath:
    """Tests for generation with FAISS context."""

    def test_generate_with_similar_examples(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
        mock_generator: MagicMock,
    ) -> None:
        """Test generation uses similar patterns as context."""
        mock_index_searcher.search_with_pairs.return_value = [
            {
                "trigger_text": "coffee tomorrow?",
                "response_text": "Sounds great!",
                "similarity": 0.72,
            },
            {
                "trigger_text": "meet for coffee?",
                "response_text": "Sure, when?",
                "similarity": 0.68,
            },
        ]

        result = router.route("want to grab coffee?")

        assert result["type"] == "generated"
        assert result.get("similar_triggers") is not None
        assert len(result["similar_triggers"]) == 2

    def test_generate_fallback_on_no_index(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
        mock_generator: MagicMock,
    ) -> None:
        """Test generation works when FAISS index not found."""
        mock_index_searcher.search_with_pairs.side_effect = FileNotFoundError("Index not found")

        result = router.route("how are you doing?")

        assert result["type"] == "generated"

    def test_generate_fallback_on_index_error(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
        mock_generator: MagicMock,
    ) -> None:
        """Test generation works when FAISS search errors."""
        mock_index_searcher.search_with_pairs.side_effect = Exception("Index error")

        result = router.route("tell me about the project")

        assert result["type"] == "generated"


# =============================================================================
# Clarify Path Tests
# =============================================================================


class TestRouteClarifyPath:
    """Tests for routing to clarify path (empty messages only)."""

    def test_empty_message_returns_clarify(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test that empty message returns clarify response."""
        result = router.route("")

        assert result["type"] == "clarify"
        assert "empty" in result["response"].lower()

    def test_clarify_response_includes_reason(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test that clarify responses include a reason."""
        result = router.route("")

        assert result["type"] == "clarify"
        assert "reason" in result


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_reply_router()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_reply_router()

    def test_get_reply_router_returns_same_instance(self) -> None:
        """Test that get_reply_router returns the same instance."""
        with patch("jarvis.router.ReplyRouter") as mock_router_cls:
            mock_router = MagicMock()
            mock_router_cls.return_value = mock_router

            router1 = get_reply_router()
            router2 = get_reply_router()

            assert router1 is router2
            mock_router_cls.assert_called_once()

    def test_reset_reply_router(self) -> None:
        """Test that reset creates new instance on next access."""
        with patch("jarvis.router.ReplyRouter") as mock_router_cls:
            mock_router1 = MagicMock()
            mock_router2 = MagicMock()
            mock_router_cls.side_effect = [mock_router1, mock_router2]

            router1 = get_reply_router()
            reset_reply_router()
            router2 = get_reply_router()

            assert router1 is not router2


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_route_with_contact_id(
        self,
        router: ReplyRouter,
        mock_db: MagicMock,
        mock_index_searcher: MagicMock,
        sample_contact: Contact,
    ) -> None:
        """Test routing with contact_id for personalization."""
        mock_db.get_contact.return_value = sample_contact
        mock_index_searcher.search_with_pairs.return_value = []

        router.route("hello there!", contact_id=1)

        mock_db.get_contact.assert_called_with(1)

    def test_route_with_chat_id(
        self,
        router: ReplyRouter,
        mock_db: MagicMock,
        mock_index_searcher: MagicMock,
        sample_contact: Contact,
    ) -> None:
        """Test routing with chat_id for contact lookup."""
        mock_db.get_contact.return_value = None
        mock_db.get_contact_by_chat_id.return_value = sample_contact
        mock_index_searcher.search_with_pairs.return_value = []

        router.route("hello there!", chat_id="chat123")

        mock_db.get_contact_by_chat_id.assert_called_with("chat123")

    def test_generation_error_returns_clarify(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
        mock_generator: MagicMock,
    ) -> None:
        """Test that generation errors fall back to clarify response."""
        mock_index_searcher.search_with_pairs.return_value = [
            {"trigger_text": "test", "response_text": "test", "similarity": 0.75}
        ]
        mock_generator.generate.side_effect = RuntimeError("Generation failed")

        result = router.route("tell me something interesting")

        assert result["type"] == "clarify"
        assert "trouble" in result["response"].lower()


# =============================================================================
# Route Result Tests
# =============================================================================


class TestRouteResult:
    """Tests for RouteResult dataclass."""

    def test_route_result_creation(self) -> None:
        """Test basic RouteResult creation."""
        result = RouteResult(
            response="Hello!",
            type="generated",
            confidence="high",
            similarity_score=0.95,
        )
        assert result.response == "Hello!"
        assert result.type == "generated"
        assert result.confidence == "high"
        assert result.similarity_score == 0.95

    def test_route_result_defaults(self) -> None:
        """Test RouteResult default values."""
        result = RouteResult(
            response="Test",
            type="clarify",
            confidence="low",
        )
        assert result.similarity_score == 0.0
        assert result.cluster_name is None
        assert result.contact_style is None
        assert result.similar_triggers is None


# =============================================================================
# Legacy Thresholds Tests
# =============================================================================


class TestThresholds:
    """Tests for legacy routing thresholds (kept for backwards compatibility)."""

    def test_thresholds_exist(self) -> None:
        """Test that legacy threshold constants still exist."""
        assert QUICK_REPLY_THRESHOLD == 0.95
        assert CONTEXT_THRESHOLD == 0.65
        assert GENERATE_THRESHOLD == 0.45

    def test_threshold_ordering(self) -> None:
        """Test that thresholds are properly ordered."""
        assert QUICK_REPLY_THRESHOLD > CONTEXT_THRESHOLD > GENERATE_THRESHOLD


# =============================================================================
# RouterError Tests
# =============================================================================


class TestRouterError:
    """Tests for RouterError exception."""

    def test_router_error_creation(self) -> None:
        """Test RouterError can be created."""
        error = RouterError("Test error")
        assert str(error) == "Test error"

    def test_router_error_inherits_from_jarvis_error(self) -> None:
        """Test RouterError inherits from JarvisError."""
        from jarvis.errors import JarvisError

        assert issubclass(RouterError, JarvisError)


# =============================================================================
# IndexNotAvailableError Tests
# =============================================================================


class TestIndexNotAvailableError:
    """Tests for IndexNotAvailableError exception."""

    def test_index_not_available_error_creation(self) -> None:
        """Test IndexNotAvailableError can be created."""
        from jarvis.router import IndexNotAvailableError

        error = IndexNotAvailableError()
        assert "FAISS index not available" in str(error)

    def test_index_not_available_inherits_from_router_error(self) -> None:
        """Test IndexNotAvailableError inherits from RouterError."""
        from jarvis.router import IndexNotAvailableError

        assert issubclass(IndexNotAvailableError, RouterError)


# =============================================================================
# Get Routing Stats Tests
# =============================================================================


class TestGetRoutingStats:
    """Tests for get_routing_stats method."""

    def test_get_routing_stats_basic(
        self,
        router: ReplyRouter,
        mock_db: MagicMock,
    ) -> None:
        """Test basic routing stats retrieval."""
        mock_db.get_stats.return_value = {"pairs": 100, "contacts": 10}
        mock_db.get_active_index.return_value = None

        stats = router.get_routing_stats()

        assert "db_stats" in stats
        assert stats["index_available"] is False

    def test_get_routing_stats_with_index(
        self,
        router: ReplyRouter,
        mock_db: MagicMock,
    ) -> None:
        """Test routing stats with active index."""
        mock_index = MagicMock()
        mock_index.version_id = "v1.0"
        mock_index.num_vectors = 1000
        mock_index.model_name = "bge-small"
        mock_db.get_active_index.return_value = mock_index

        stats = router.get_routing_stats()

        assert stats["index_available"] is True
        assert stats["index_version"] == "v1.0"
        assert stats["index_vectors"] == 1000


# =============================================================================
# iMessage Reader Property Tests
# =============================================================================


class TestIMessageReaderProperty:
    """Tests for iMessage reader lazy initialization."""

    def test_imessage_reader_property_returns_none_on_error(
        self,
        mock_db: MagicMock,
        mock_index_searcher: MagicMock,
        mock_generator: MagicMock,
    ) -> None:
        """Test imessage_reader returns None when initialization fails."""
        router = ReplyRouter(
            db=mock_db,
            index_searcher=mock_index_searcher,
            generator=mock_generator,
        )
        router._imessage_reader = None

        with patch(
            "integrations.imessage.reader.ChatDBReader", side_effect=Exception("Cannot init")
        ):
            result = router.imessage_reader
            assert result is None

    def test_imessage_reader_property_caches_instance(self) -> None:
        """Test imessage_reader caches the instance after creation."""
        mock_reader = MagicMock()
        router = ReplyRouter(imessage_reader=mock_reader)

        reader1 = router.imessage_reader
        reader2 = router.imessage_reader

        assert reader1 is reader2
        assert reader1 is mock_reader


# =============================================================================
# Fetch Conversation Context Tests
# =============================================================================


class TestFetchConversationContext:
    """Tests for _fetch_conversation_context method."""

    def test_fetch_context_returns_empty_when_no_reader(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test context fetch returns empty list when reader is None."""
        router._imessage_reader = None
        result = router._fetch_conversation_context("chat123", limit=10)
        assert result == []

    def test_fetch_context_formats_messages(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test context fetch correctly formats messages."""
        mock_reader = MagicMock()
        mock_msg1 = MagicMock()
        mock_msg1.is_from_me = True
        mock_msg1.sender_name = None
        mock_msg1.sender = None
        mock_msg1.text = "Hello!"

        mock_msg2 = MagicMock()
        mock_msg2.is_from_me = False
        mock_msg2.sender_name = "John"
        mock_msg2.sender = "+1234567890"
        mock_msg2.text = "Hey there!"

        mock_reader.get_messages.return_value = [mock_msg1, mock_msg2]
        router._imessage_reader = mock_reader

        result = router._fetch_conversation_context("chat123", limit=10)

        assert len(result) == 2
        assert "[John]: Hey there!" in result[0]
        assert "[You]: Hello!" in result[1]

    def test_fetch_context_handles_empty_messages(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test context fetch handles empty message list."""
        mock_reader = MagicMock()
        mock_reader.get_messages.return_value = []
        router._imessage_reader = mock_reader

        result = router._fetch_conversation_context("chat123")
        assert result == []

    def test_fetch_context_handles_exception(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test context fetch handles exceptions gracefully."""
        mock_reader = MagicMock()
        mock_reader.get_messages.side_effect = Exception("Database error")
        router._imessage_reader = mock_reader

        result = router._fetch_conversation_context("chat123")
        assert result == []

    def test_fetch_context_skips_empty_text(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test context fetch skips messages with empty text."""
        mock_reader = MagicMock()
        mock_msg1 = MagicMock()
        mock_msg1.is_from_me = True
        mock_msg1.sender_name = None
        mock_msg1.sender = None
        mock_msg1.text = ""

        mock_msg2 = MagicMock()
        mock_msg2.is_from_me = False
        mock_msg2.sender_name = "John"
        mock_msg2.sender = None
        mock_msg2.text = "Hello"

        mock_reader.get_messages.return_value = [mock_msg1, mock_msg2]
        router._imessage_reader = mock_reader

        result = router._fetch_conversation_context("chat123")

        assert len(result) == 1
        assert "[John]: Hello" in result[0]


# =============================================================================
# Route Multi-Option Simplified Tests
# =============================================================================


class TestRouteMultiOptionSimplified:
    """Tests for simplified route_multi_option method."""

    def test_delegates_to_route(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
        mock_generator: MagicMock,
    ) -> None:
        """Test route_multi_option delegates to route()."""
        mock_index_searcher.search_with_pairs.return_value = []

        result = router.route_multi_option("want to grab lunch?")

        assert result["type"] == "generated"
        assert result["is_commitment"] is False
        assert result["options"] == []
        assert isinstance(result["suggestions"], list)
        assert len(result["suggestions"]) == 1
        assert result["trigger_da"] is None
        mock_generator.generate.assert_called()

    def test_preserves_response(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
        mock_generator: MagicMock,
    ) -> None:
        """Test route_multi_option preserves the generated response."""
        mock_index_searcher.search_with_pairs.return_value = []
        mock_response = MagicMock()
        mock_response.text = "Sounds good!"
        mock_generator.generate.return_value = mock_response

        result = router.route_multi_option("let's meet up")

        assert result["response"] == "Sounds good!"
        assert result["suggestions"] == ["Sounds good!"]
