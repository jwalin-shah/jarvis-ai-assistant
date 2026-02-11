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
    ReplyRouter,
    RouterError,
    RouteResult,
    get_reply_router,
    reset_reply_router,
)
from jarvis.services.context_service import ContextService

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def mock_health_check():
    """Mock health check to allow generation in tests."""
    with patch("jarvis.generation.can_use_llm", return_value=(True, "ok")):
        yield


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
    mock_generator: MagicMock,
) -> ReplyRouter:
    """Create a ReplyRouter with all mocked dependencies."""
    r = ReplyRouter(db=mock_db, generator=mock_generator)
    return r


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
        mock_generator: MagicMock,
    ) -> None:
        """Test initialization with all dependencies provided."""
        router = ReplyRouter(db=mock_db, generator=mock_generator)
        assert router._db is mock_db
        assert router._generator is mock_generator

    def test_init_with_no_dependencies(self) -> None:
        """Test initialization with no dependencies (lazy loading)."""
        router = ReplyRouter()
        assert router._db is None
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
    """Tests that non-empty messages go through LLM generation or skip for NONE pressure."""

    @pytest.mark.parametrize(
        "message",
        [
            "want to grab lunch?",
            "what time?",
            "how are you?",
            "are you coming to the party?",
            "I'm doing well",
            "The weather is nice today",
            "sounds good to me",
        ],
    )
    def test_non_backchannel_messages_generate(
        self,
        router: ReplyRouter,
        mock_generator: MagicMock,
        message: str,
    ) -> None:
        """Test that non-backchannel messages route to generation."""
        result = router.route(message)

        assert result["type"] == "generated"
        mock_generator.generate.assert_called()

    @pytest.mark.parametrize(
        "message,expected_type",
        [
            ("ok", "acknowledge"),
            ("thanks", "acknowledge"),
        ],
    )
    def test_acknowledgments_use_templates(
        self,
        router: ReplyRouter,
        mock_generator: MagicMock,
        message: str,
        expected_type: str,
    ) -> None:
        """Test that acknowledgments return template responses (no LLM generation)."""
        result = router.route(message)

        # Should return template response based on category
        assert result["type"] == expected_type
        # Should not call LLM generator
        mock_generator.generate.assert_not_called()

    def test_emotion_without_context_skips(
        self,
        router: ReplyRouter,
        mock_generator: MagicMock,
    ) -> None:
        """Test that emotion messages without context skip (NONE pressure, no examples)."""
        result = router.route("lol")

        # Emotion with no context -> NONE pressure -> skip
        assert result["type"] == "skip"
        assert result["reason"] == "no_response_needed"
        mock_generator.generate.assert_not_called()

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
    """Tests that mobilization pressure maps to confidence correctly.

    Without RAG context (no search results), _compute_confidence applies:
    - base_confidence[pressure] * 0.8 (sim < 0.5) * 0.9 (diversity < 0.3)
    Then maps: >= 0.7 -> "high", >= 0.45 -> "medium", else -> "low".
    """

    def test_high_pressure_yields_medium_confidence_without_rag(
        self,
        router: ReplyRouter,
        mock_generator: MagicMock,
    ) -> None:
        """HIGH pressure with no RAG: base 0.85 * 0.8 * 0.9 = 0.612 -> medium."""
        result = router.route("Can you pick me up at 5?")

        assert result["type"] == "generated"
        assert result["confidence"] == "medium"

    def test_low_pressure_yields_low_confidence_without_rag(
        self,
        router: ReplyRouter,
        mock_generator: MagicMock,
    ) -> None:
        """LOW pressure with no RAG: base 0.45 * 0.8 * 0.9 = 0.324 -> low."""
        result = router.route("I think the weather is nice")

        assert result["type"] == "generated"
        assert result["confidence"] == "low"

    def test_medium_pressure_yields_medium_confidence_without_rag(
        self,
        router: ReplyRouter,
        mock_generator: MagicMock,
    ) -> None:
        """MEDIUM pressure with no RAG: base 0.65 * 0.8 * 0.9 = 0.468 -> medium."""
        result = router.route("I got the job!!")

        assert result["type"] == "generated"
        assert result["confidence"] == "medium"


# =============================================================================
# Generate Path Tests
# =============================================================================


class TestRouteGeneratePath:
    """Tests for generation with search context."""

    def test_generate_with_similar_examples(
        self,
        router: ReplyRouter,
        mock_generator: MagicMock,
    ) -> None:
        """Test generation uses similar patterns as context."""
        search_results = [
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

        # Mock reranker to avoid requiring cross-encoder model in test environment
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = search_results

        with (
            patch.object(router.context_service, "search_examples", return_value=search_results),
            patch("models.reranker.get_reranker", return_value=mock_reranker),
        ):
            result = router.route("want to grab coffee?")

        assert result["type"] == "generated"
        mock_generator.generate.assert_called()

    def test_generate_fallback_on_search_error(
        self,
        router: ReplyRouter,
        mock_generator: MagicMock,
    ) -> None:
        """Test generation works when search returns empty (error handled internally)."""
        # search_examples catches exceptions and returns [], generation still proceeds
        result = router.route("how are you doing?")

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

    def test_route_with_contact_id_uses_contact_data(
        self,
        router: ReplyRouter,
        mock_db: MagicMock,
        mock_generator: MagicMock,
        sample_contact: Contact,
    ) -> None:
        """Test routing with contact_id fetches contact and uses it for generation."""
        mock_db.get_contact.return_value = sample_contact

        result = router.route("hello there!", contact_id=1)

        # Verify contact was fetched
        mock_db.get_contact.assert_called_with(1)
        # Verify generation was called (contact influences prompt building)
        mock_generator.generate.assert_called_once()
        # Verify the result includes a generated response (contact data influenced the pipeline)
        assert result["type"] == "generated"
        assert result["response"] == "Generated response"

    def test_route_with_chat_id_uses_contact_data(
        self,
        router: ReplyRouter,
        mock_db: MagicMock,
        mock_generator: MagicMock,
        sample_contact: Contact,
    ) -> None:
        """Test routing with chat_id falls back to chat_id lookup and uses contact for generation."""
        mock_db.get_contact.return_value = None
        mock_db.get_contact_by_chat_id.return_value = sample_contact

        result = router.route("hello there!", chat_id="chat123")

        # Verify chat_id-based contact lookup was used
        mock_db.get_contact_by_chat_id.assert_called_with("chat123")
        # Verify generation was called with the resolved contact
        mock_generator.generate.assert_called_once()
        assert result["type"] == "generated"
        assert result["response"] == "Generated response"

    def test_generation_error_returns_clarify(
        self,
        router: ReplyRouter,
        mock_generator: MagicMock,
    ) -> None:
        """Test that generation errors fall back to clarify response."""
        mock_generator.generate.side_effect = RuntimeError("Generation failed")

        result = router.route("tell me something interesting")

        assert result["type"] == "clarify"
        assert "trouble" in result["response"].lower()


# =============================================================================
# Route Result and Exception Tests
# =============================================================================


class TestRouteResultDefaults:
    """Tests for RouteResult dataclass default values."""

    def test_optional_fields_default_correctly(self) -> None:
        """Test that optional fields have meaningful defaults."""
        result = RouteResult(
            response="Test",
            type="clarify",
            confidence="low",
        )
        assert result.similarity_score == 0.0
        assert result.cluster_name is None
        assert result.contact_style is None
        assert result.similar_triggers is None


class TestExceptionHierarchy:
    """Tests for exception hierarchy (RouterError -> JarvisError)."""

    def test_router_error_inherits_from_jarvis_error(self) -> None:
        """RouterError should be catchable as JarvisError."""
        from jarvis.errors import JarvisError

        error = RouterError("Test error")
        assert isinstance(error, JarvisError)
        assert str(error) == "Test error"

    def test_index_not_available_inherits_from_router_error(self) -> None:
        """IndexNotAvailableError should be catchable as RouterError."""
        from jarvis.router import IndexNotAvailableError

        error = IndexNotAvailableError()
        assert isinstance(error, RouterError)
        assert "index not available" in str(error).lower()


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
        """Test routing stats with active vec_chunks data."""
        # Mock the connection context manager to return vec_chunks count
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"cnt": 1000}
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_db.connection.return_value = mock_conn

        stats = router.get_routing_stats()

        assert stats["index_available"] is True
        assert stats["index_vectors"] == 1000
        assert stats["index_type"] == "sqlite-vec"


# =============================================================================
# iMessage Reader Property Tests
# =============================================================================


class TestIMessageReaderProperty:
    """Tests for iMessage reader lazy initialization."""

    def test_imessage_reader_property_returns_none_on_error(
        self,
        mock_db: MagicMock,
        mock_generator: MagicMock,
    ) -> None:
        """Test imessage_reader returns None when initialization fails."""
        router = ReplyRouter(
            db=mock_db,
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
# Fetch Conversation Context Tests (via ContextService)
# =============================================================================


class TestFetchConversationContext:
    """Tests for ContextService.fetch_conversation_context."""

    def test_fetch_context_returns_empty_when_no_reader(
        self,
        mock_db: MagicMock,
    ) -> None:
        """Test context fetch returns empty list when reader is None."""
        ctx = ContextService(db=mock_db, imessage_reader=None)
        result = ctx.fetch_conversation_context("chat123", limit=10)
        assert result == []

    def test_fetch_context_formats_messages(
        self,
        mock_db: MagicMock,
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

        ctx = ContextService(db=mock_db, imessage_reader=mock_reader)
        result = ctx.fetch_conversation_context("chat123", limit=10)

        assert len(result) == 2
        assert "[John]: Hey there!" in result[0]
        assert "[You]: Hello!" in result[1]

    def test_fetch_context_handles_empty_messages(
        self,
        mock_db: MagicMock,
    ) -> None:
        """Test context fetch handles empty message list."""
        mock_reader = MagicMock()
        mock_reader.get_messages.return_value = []

        ctx = ContextService(db=mock_db, imessage_reader=mock_reader)
        result = ctx.fetch_conversation_context("chat123")
        assert result == []

    def test_fetch_context_handles_exception(
        self,
        mock_db: MagicMock,
    ) -> None:
        """Test context fetch handles exceptions gracefully."""
        mock_reader = MagicMock()
        mock_reader.get_messages.side_effect = Exception("Database error")

        ctx = ContextService(db=mock_db, imessage_reader=mock_reader)
        result = ctx.fetch_conversation_context("chat123")
        assert result == []

    def test_fetch_context_skips_empty_text(
        self,
        mock_db: MagicMock,
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

        ctx = ContextService(db=mock_db, imessage_reader=mock_reader)
        result = ctx.fetch_conversation_context("chat123")

        assert len(result) == 1
        assert "[John]: Hello" in result[0]


# =============================================================================
# _analyze_complexity Tests
# =============================================================================


class TestAnalyzeComplexity:
    """Tests for ReplyRouter._analyze_complexity static method."""

    def test_empty_string(self) -> None:
        assert ReplyRouter._analyze_complexity("") == 0.0

    def test_short_message(self) -> None:
        score = ReplyRouter._analyze_complexity("ok")
        assert 0.0 < score <= 0.3

    def test_long_complex_message(self) -> None:
        text = (
            "I was wondering if you could help me with something? "
            "I need to schedule a meeting with the team, "
            "coordinate with marketing, and prepare the presentation slides."
        )
        score = ReplyRouter._analyze_complexity(text)
        assert score > 0.5

    def test_punctuation_variety_increases_score(self) -> None:
        plain = "Hello there how are you"
        punctuated = "Hello! How are you? Great, thanks."
        assert ReplyRouter._analyze_complexity(punctuated) > ReplyRouter._analyze_complexity(plain)

    def test_word_variety_matters(self) -> None:
        repetitive = "the the the the the the the the"
        varied = "quick brown fox jumps over lazy dog today"
        # Same-ish length, but variety_score differs
        rep_score = ReplyRouter._analyze_complexity(repetitive)
        var_score = ReplyRouter._analyze_complexity(varied)
        assert var_score > rep_score

    def test_result_between_0_and_1(self) -> None:
        """All results should be in [0, 1]."""
        for text in ["", "hi", "a" * 500, "Hello? Yes! No, wait: maybe; perhaps."]:
            score = ReplyRouter._analyze_complexity(text)
            assert 0.0 <= score <= 1.0

    def test_single_word(self) -> None:
        score = ReplyRouter._analyze_complexity("hello")
        assert 0.0 < score < 0.4


# =============================================================================
# _to_intent_type Tests
# =============================================================================


class TestToIntentType:
    """Tests for to_intent_type (extracted to classification_result module)."""

    def test_question(self) -> None:
        from jarvis.classifiers.classification_result import to_intent_type
        from jarvis.contracts.pipeline import IntentType

        assert to_intent_type("question") == IntentType.QUESTION

    def test_request(self) -> None:
        from jarvis.classifiers.classification_result import to_intent_type
        from jarvis.contracts.pipeline import IntentType

        assert to_intent_type("request") == IntentType.REQUEST

    def test_statement(self) -> None:
        from jarvis.classifiers.classification_result import to_intent_type
        from jarvis.contracts.pipeline import IntentType

        assert to_intent_type("statement") == IntentType.STATEMENT

    def test_emotion(self) -> None:
        from jarvis.classifiers.classification_result import to_intent_type
        from jarvis.contracts.pipeline import IntentType

        assert to_intent_type("emotion") == IntentType.STATEMENT

    def test_closing(self) -> None:
        from jarvis.classifiers.classification_result import to_intent_type
        from jarvis.contracts.pipeline import IntentType

        assert to_intent_type("closing") == IntentType.STATEMENT

    def test_acknowledge(self) -> None:
        from jarvis.classifiers.classification_result import to_intent_type
        from jarvis.contracts.pipeline import IntentType

        assert to_intent_type("acknowledge") == IntentType.STATEMENT

    def test_unknown_category(self) -> None:
        from jarvis.classifiers.classification_result import to_intent_type
        from jarvis.contracts.pipeline import IntentType

        assert to_intent_type("gibberish") == IntentType.UNKNOWN

    def test_empty_string(self) -> None:
        from jarvis.classifiers.classification_result import to_intent_type
        from jarvis.contracts.pipeline import IntentType

        assert to_intent_type("") == IntentType.UNKNOWN


# =============================================================================
# _to_confidence_label Tests
# =============================================================================


class TestToConfidenceLabel:
    """Tests for ReplyRouter._to_confidence_label static method."""

    def test_high_confidence(self) -> None:
        assert ReplyRouter._to_confidence_label(0.9) == "high"

    def test_boundary_0_7(self) -> None:
        assert ReplyRouter._to_confidence_label(0.7) == "high"

    def test_medium_confidence(self) -> None:
        assert ReplyRouter._to_confidence_label(0.55) == "medium"

    def test_boundary_0_45(self) -> None:
        assert ReplyRouter._to_confidence_label(0.45) == "medium"

    def test_low_confidence(self) -> None:
        assert ReplyRouter._to_confidence_label(0.3) == "low"

    def test_zero(self) -> None:
        assert ReplyRouter._to_confidence_label(0.0) == "low"

    def test_just_below_0_7(self) -> None:
        assert ReplyRouter._to_confidence_label(0.699) == "medium"

    def test_just_below_0_45(self) -> None:
        assert ReplyRouter._to_confidence_label(0.449) == "low"


# =============================================================================
# _to_legacy_response Tests
# =============================================================================


class TestToLegacyResponse:
    """Tests for ReplyRouter._to_legacy_response static method."""

    def test_generated_response_format(self) -> None:
        from jarvis.contracts.pipeline import GenerationResponse

        resp = GenerationResponse(
            response="Hello there!",
            confidence=0.85,
            metadata={
                "type": "generated",
                "reason": "generated",
                "similarity_score": 0.72,
                "similar_triggers": ["hi", "hey"],
                "category": "question",
            },
        )
        result = ReplyRouter._to_legacy_response(resp)

        assert result["type"] == "generated"
        assert result["response"] == "Hello there!"
        assert result["confidence"] == "high"
        assert result["similarity_score"] == 0.72
        assert result["similar_triggers"] == ["hi", "hey"]
        assert result["category"] == "question"

    def test_clarify_response_format(self) -> None:
        from jarvis.contracts.pipeline import GenerationResponse

        resp = GenerationResponse(
            response="I'm having trouble generating a response.",
            confidence=0.2,
            metadata={
                "type": "clarify",
                "reason": "generation_error",
                "similarity_score": 0.0,
            },
        )
        result = ReplyRouter._to_legacy_response(resp)

        assert result["type"] == "clarify"
        assert result["confidence"] == "low"
        assert result["reason"] == "generation_error"

    def test_category_included_when_present(self) -> None:
        from jarvis.contracts.pipeline import GenerationResponse

        resp = GenerationResponse(
            response="Sure",
            confidence=0.5,
            metadata={"type": "generated", "category": "acknowledge"},
        )
        result = ReplyRouter._to_legacy_response(resp)
        assert result["category"] == "acknowledge"

    def test_category_omitted_when_absent(self) -> None:
        from jarvis.contracts.pipeline import GenerationResponse

        resp = GenerationResponse(
            response="Sure",
            confidence=0.5,
            metadata={"type": "generated"},
        )
        result = ReplyRouter._to_legacy_response(resp)
        assert "category" not in result

    def test_missing_metadata_fields_use_defaults(self) -> None:
        from jarvis.contracts.pipeline import GenerationResponse

        resp = GenerationResponse(
            response="text",
            confidence=0.5,
            metadata={},
        )
        result = ReplyRouter._to_legacy_response(resp)
        assert result["type"] == "generated"
        assert result["similarity_score"] == 0.0
        assert result["reason"] == ""
