"""Integration tests for the JARVIS message routing pipeline.

Tests the end-to-end flow: incoming message -> classify -> RAG search ->
prompt assembly -> generate -> response.

Mocks: MLX generator (GPU), BERT embedder (GPU), health checks, iMessage reader.
Real: mobilization cascade, classification result builder, ReplyService routing
logic, prompt assembly, ContextService wiring, JarvisDB (in-memory SQLite).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from contracts.models import GenerationResponse as ModelGenerationResponse
from jarvis.contracts.pipeline import (
    GenerationResponse,
    MessageContext,
)
from jarvis.db import JarvisDB
from jarvis.prompts import ACKNOWLEDGE_TEMPLATES, CLOSING_TEMPLATES
from jarvis.reply_service import ReplyService
from jarvis.router import ReplyRouter

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def mock_health_check():
    """Allow LLM generation in all tests."""
    with patch("jarvis.generation.can_use_llm", return_value=(True, "ok")):
        yield


@pytest.fixture
def in_memory_db(tmp_path: Path) -> JarvisDB:
    """Create a real JarvisDB backed by a temp file."""
    db_path = tmp_path / "test_jarvis.db"
    db = JarvisDB(db_path)
    db.init_schema()
    return db


@pytest.fixture
def mock_generator() -> MagicMock:
    """Mock MLX generator that returns controllable canned responses."""
    gen = MagicMock()
    gen.is_loaded.return_value = True
    gen.generate.return_value = ModelGenerationResponse(
        text="Sure, sounds great!",
        tokens_used=6,
        generation_time_ms=42.0,
        model_name="mock-model",
        used_template=False,
        template_name=None,
        finish_reason="stop",
    )
    return gen


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns deterministic 384-dim vectors."""
    embedder = MagicMock()
    embedder.model_name = "mock-bge-small"
    embedder.embedding_computations = 0
    embedder.cache_hit = False

    def _encode(texts, normalize=False):
        if isinstance(texts, str):
            texts = [texts]
        vecs = []
        for t in texts:
            rng = np.random.RandomState(hash(t) % (2**31))
            v = rng.randn(384).astype(np.float32)
            if normalize:
                v = v / (np.linalg.norm(v) + 1e-9)
            vecs.append(v)
        return np.array(vecs)

    embedder.encode = _encode
    return embedder


def _make_router(
    db: JarvisDB,
    generator: MagicMock,
    embedder: MagicMock,
) -> ReplyRouter:
    """Build a ReplyRouter wired to real DB but mocked GPU deps."""
    router = ReplyRouter(db=db, generator=generator)

    from jarvis.services.context_service import ContextService

    ctx_svc = ContextService(db=db, imessage_reader=None)
    router._context_service = ctx_svc

    reply_svc = ReplyService(db=db, generator=generator)
    reply_svc._context_service = ctx_svc
    router._reply_service = reply_svc

    return router


# =============================================================================
# Full Pipeline: Message In -> Response Out
# =============================================================================


class TestFullPipelineEndToEnd:
    """Tests that exercise the complete routing pipeline."""

    def test_question_produces_generated_response(
        self, in_memory_db, mock_generator, mock_embedder
    ):
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            result = router.route("What time should we meet tomorrow?")
        assert result["type"] == "generated"
        assert result["response"] == "Sure, sounds great!"
        assert result["confidence"] in ("high", "medium", "low")
        assert "similarity_score" in result

    def test_request_produces_generated_response(self, in_memory_db, mock_generator, mock_embedder):
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            result = router.route("Can you pick me up from the airport at 3pm?")
        assert result["type"] == "generated"
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_statement_produces_generated_response(
        self, in_memory_db, mock_generator, mock_embedder
    ):
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            result = router.route("I just got back from vacation, it was amazing")
        assert result["type"] == "generated"
        assert result["response"] == "Sure, sounds great!"

    def test_acknowledgment_returns_template_without_llm(
        self, in_memory_db, mock_generator, mock_embedder
    ):
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            result = router.route("ok")
        assert result["type"] == "acknowledge"
        assert result["response"] in ACKNOWLEDGE_TEMPLATES
        mock_generator.generate.assert_not_called()

    def test_closing_returns_template_without_llm(
        self, in_memory_db, mock_generator, mock_embedder
    ):
        from jarvis.classifiers.category_classifier import CategoryResult

        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        # Mock classifier to return "closing" - this test validates routing behavior,
        # not classifier accuracy (which depends on real BERT embeddings)
        closing_result = CategoryResult(category="closing", confidence=0.9, method="lightgbm")
        with (
            patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder),
            patch(
                "jarvis.classifiers.classification_result.classify_category",
                return_value=closing_result,
            ),
        ):
            result = router.route("goodbye")
        assert result["type"] in ("closing", "acknowledge")
        assert result["response"] in CLOSING_TEMPLATES + ACKNOWLEDGE_TEMPLATES
        mock_generator.generate.assert_not_called()

    def test_emoji_reaction_skips_llm(self, in_memory_db, mock_generator, mock_embedder):
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            result = router.route('Loved "great news!"')
        assert result["type"] in ("emotion", "acknowledge", "skip")
        mock_generator.generate.assert_not_called()


# =============================================================================
# Empty / Edge Case Messages
# =============================================================================


class TestEdgeCaseMessages:
    """Tests for empty, whitespace, and unusual message inputs."""

    def test_empty_message_returns_clarify(self, in_memory_db, mock_generator, mock_embedder):
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            result = router.route("")
        assert result["type"] == "clarify"
        assert "empty" in result["response"].lower()
        mock_generator.generate.assert_not_called()

    def test_whitespace_only_returns_clarify(self, in_memory_db, mock_generator, mock_embedder):
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            result = router.route("   \t\n  ")
        assert result["type"] == "clarify"

    def test_very_long_message_still_produces_response(
        self, in_memory_db, mock_generator, mock_embedder
    ):
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        long_message = "Hey " * 500 + "what do you think about this plan?"
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            result = router.route(long_message)
        assert result["type"] in ("generated", "clarify", "skip")
        assert isinstance(result["response"], str)

    def test_single_character_message(self, in_memory_db, mock_generator, mock_embedder):
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            result = router.route("?")
        assert "type" in result
        assert "response" in result

    def test_unicode_emoji_message(self, in_memory_db, mock_generator, mock_embedder):
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            result = router.route("\U0001f600\U0001f44d")
        assert "type" in result
        assert isinstance(result["response"], str)


# =============================================================================
# Thread Context Propagation
# =============================================================================


class TestThreadContextPropagation:
    """Tests that thread history is properly passed through the pipeline."""

    def test_thread_context_reaches_generation(self, in_memory_db, mock_generator, mock_embedder):
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        thread = [
            "Alice: Hey, want to grab dinner?",
            "Me: Sure, where?",
            "Alice: How about that new Italian place?",
        ]
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            result = router.route(
                "How about that new Italian place?",
                thread=thread,
            )
        assert result["type"] == "generated"
        assert result["response"] == "Sure, sounds great!"
        mock_generator.generate.assert_called_once()

    def test_conversation_messages_build_thread(self, in_memory_db, mock_generator, mock_embedder):
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        conversation_messages = [
            {"text": "Hey!", "is_from_me": False, "sender_name": "Bob"},
            {"text": "Hi Bob!", "is_from_me": True, "sender_name": ""},
            {"text": "Want to get coffee?", "is_from_me": False, "sender_name": "Bob"},
        ]
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            result = router.route(
                "Want to get coffee?",
                conversation_messages=conversation_messages,
            )
        assert result["type"] == "generated"


# =============================================================================
# Contact Integration
# =============================================================================


class TestContactIntegration:
    """Tests that contact data flows through the pipeline."""

    def test_contact_id_lookup_integrates_with_real_db(
        self, in_memory_db, mock_generator, mock_embedder
    ):
        in_memory_db.add_contact(
            chat_id="chat_sarah",
            display_name="Sarah",
            phone_or_email="+15551234567",
            relationship="sister",
            style_notes="casual, lots of emojis",
        )
        contact = in_memory_db.get_contact_by_chat_id("chat_sarah")
        assert contact is not None

        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            result = router.route("Want to grab lunch?", contact_id=contact.id)
        assert result["type"] == "generated"
        assert result["response"] == "Sure, sounds great!"

    def test_chat_id_lookup_finds_contact(self, in_memory_db, mock_generator, mock_embedder):
        in_memory_db.add_contact(
            chat_id="chat_john",
            display_name="John",
            phone_or_email="+15559876543",
            relationship="friend",
            style_notes="formal",
        )
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            result = router.route(
                "Are you coming to the party tonight?",
                chat_id="chat_john",
            )
        assert result["type"] == "generated"

    def test_missing_contact_does_not_crash(self, in_memory_db, mock_generator, mock_embedder):
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            result = router.route("Hello there!", contact_id=99999)
        assert result["type"] in ("generated", "clarify", "skip")
        assert isinstance(result["response"], str)


# =============================================================================
# Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests that pipeline errors produce graceful fallback responses."""

    def test_generator_exception_returns_clarify(self, in_memory_db, mock_generator, mock_embedder):
        mock_generator.generate.side_effect = RuntimeError("GPU out of memory")
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            result = router.route("Tell me something interesting")
        assert result["type"] == "clarify"
        assert "trouble" in result["response"].lower()

    def test_health_check_failure_returns_fallback(
        self, in_memory_db, mock_generator, mock_embedder
    ):
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        with (
            patch("jarvis.generation.can_use_llm", return_value=(False, "memory critical")),
            patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder),
        ):
            result = router.route("What is the weather like?")
        assert result["type"] == "fallback"
        mock_generator.generate.assert_not_called()

    def test_embedder_failure_does_not_crash_pipeline(
        self,
        in_memory_db,
        mock_generator,
    ):
        failing_embedder = MagicMock()
        failing_embedder.encode = MagicMock(side_effect=RuntimeError("MLX crash"))
        failing_embedder.embedding_computations = 0
        failing_embedder.cache_hit = False

        router = _make_router(in_memory_db, mock_generator, failing_embedder)
        with patch("jarvis.embedding_adapter.get_embedder", return_value=failing_embedder):
            result = router.route("How are you doing?")
        assert "type" in result
        assert isinstance(result["response"], str)


# =============================================================================
# Response Structure Validation
# =============================================================================


class TestResponseStructure:
    """Tests that pipeline responses have the expected shape and fields."""

    def test_generated_response_has_required_fields(
        self, in_memory_db, mock_generator, mock_embedder
    ):
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            result = router.route("What do you think about this idea?")
        assert "type" in result
        assert "response" in result
        assert "confidence" in result
        assert "similarity_score" in result
        assert result["confidence"] in ("high", "medium", "low")
        assert isinstance(result["similarity_score"], float)
        assert result["similarity_score"] >= 0.0

    def test_clarify_response_includes_reason(self, in_memory_db, mock_generator, mock_embedder):
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            result = router.route("")
        assert result["type"] == "clarify"
        assert "reason" in result

    def test_template_response_includes_category(self, in_memory_db, mock_generator, mock_embedder):
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            result = router.route("thanks")
        assert result["type"] in ("acknowledge", "closing")
        assert "category" in result


# =============================================================================
# route_message() Typed Contract
# =============================================================================


class TestRouteMessageTypedContract:
    """Tests for the typed route_message() path using MessageContext."""

    def test_route_message_returns_generation_response(
        self, in_memory_db, mock_generator, mock_embedder
    ):
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        context = MessageContext(
            chat_id="chat_test",
            message_text="What are we doing tonight?",
            is_from_me=False,
            timestamp=datetime(2025, 2, 10, 18, 0, 0),
            metadata={"thread": ["Alice: Let's hang out"]},
        )
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            response = router.route_message(context, cached_embedder=mock_embedder)
        assert isinstance(response, GenerationResponse)
        assert isinstance(response.response, str)
        assert isinstance(response.confidence, float)
        assert isinstance(response.metadata, dict)

    def test_route_message_empty_text_returns_clarify(
        self, in_memory_db, mock_generator, mock_embedder
    ):
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        context = MessageContext(
            chat_id="chat_test",
            message_text="",
            is_from_me=False,
            timestamp=datetime(2025, 2, 10, 18, 0, 0),
        )
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            response = router.route_message(context, cached_embedder=mock_embedder)
        assert response.metadata.get("type") == "clarify"
        assert "empty" in response.response.lower()


# =============================================================================
# Multiple Message Types in Sequence
# =============================================================================


class TestMessageTypeVariety:
    """Tests that different message types produce distinct routing behaviors."""

    @pytest.mark.parametrize(
        "message,expected_types",
        [
            ("Want to grab lunch?", {"generated"}),
            ("ok", {"acknowledge"}),
            ("thanks", {"acknowledge"}),
            ("", {"clarify"}),
            ('Liked "nice photo"', {"acknowledge"}),
            ('Laughed at "that joke was hilarious"', {"emotion", "skip"}),
        ],
    )
    def test_message_type_routing(
        self,
        in_memory_db,
        mock_generator,
        mock_embedder,
        message,
        expected_types,
    ):
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            result = router.route(message)
        assert result["type"] in expected_types, (
            f"Message '{message}' routed to '{result['type']}', expected one of {expected_types}"
        )

    def test_sequential_messages_do_not_leak_state(
        self, in_memory_db, mock_generator, mock_embedder
    ):
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            r1 = router.route("")
            r2 = router.route("What time is dinner?")
            r3 = router.route("ok")
        assert r1["type"] == "clarify"
        assert r2["type"] == "generated"
        assert r3["type"] == "acknowledge"


# =============================================================================
# Confidence Scoring
# =============================================================================


class TestConfidenceScoring:
    """Tests that confidence levels reflect routing context."""

    def test_high_pressure_message_has_reasonable_confidence(
        self, in_memory_db, mock_generator, mock_embedder
    ):
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            result = router.route("Can you send me the file by 5pm?")
        assert result["confidence"] in ("high", "medium")

    def test_template_response_has_high_confidence(
        self, in_memory_db, mock_generator, mock_embedder
    ):
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            result = router.route("ok")
        assert result["confidence"] == "high"

    def test_empty_message_has_low_confidence(self, in_memory_db, mock_generator, mock_embedder):
        router = _make_router(in_memory_db, mock_generator, mock_embedder)
        with patch("jarvis.embedding_adapter.get_embedder", return_value=mock_embedder):
            result = router.route("")
        assert result["confidence"] == "low"
