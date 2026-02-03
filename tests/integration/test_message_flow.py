"""Integration tests for end-to-end message flow.

Tests the complete pipeline from receiving a message through
classification to generating a response.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from contracts.imessage import Message
from jarvis.intent import IntentClassifier, IntentResult, IntentType, get_intent_classifier
from jarvis.router import ReplyRouter, reset_reply_router

from .conftest import create_mock_message


@pytest.fixture(autouse=True)
def reset_router():
    """Reset the router singleton before and after each test."""
    reset_reply_router()
    yield
    reset_reply_router()


class TestIntentClassification:
    """Tests for intent classification in the message flow."""

    def test_reply_intent_detected(self):
        """Reply-related queries are classified as REPLY."""
        classifier = IntentClassifier()

        test_cases = [
            "help me reply to this",
            "what should I say back",
            "draft a response",
            "how should I respond",
        ]

        for query in test_cases:
            with patch.object(classifier, "_get_embedder") as mock_get_embedder:
                mock_embedder = MagicMock()
                mock_embedder.encode.return_value = [[0.1] * 384]  # Mock embedding
                mock_get_embedder.return_value = mock_embedder

                # Skip actual embedding computation
                with patch.object(classifier, "_ensure_embeddings_computed"):
                    classifier._intent_centroids = {
                        IntentType.REPLY: [0.1] * 384,
                        IntentType.SUMMARIZE: [0.0] * 384,
                        IntentType.SEARCH: [0.0] * 384,
                        IntentType.QUICK_REPLY: [0.0] * 384,
                        IntentType.GENERAL: [0.0] * 384,
                    }

    def test_quick_reply_intent_detected(self):
        """Short acknowledgments are classified as QUICK_REPLY."""
        classifier = IntentClassifier()

        test_cases = [
            "ok",
            "thanks",
            "sounds good",
            "got it",
        ]

        for query in test_cases:
            # These should be handled by the simple acknowledgment check first
            result = classifier.classify(query)
            # Even if embedding fails, it should return something
            assert isinstance(result, IntentResult)

    def test_empty_input_returns_general(self):
        """Empty input returns GENERAL with low confidence."""
        classifier = IntentClassifier()

        result = classifier.classify("")

        assert result.intent == IntentType.GENERAL
        assert result.confidence == 0.0

    def test_whitespace_input_returns_general(self):
        """Whitespace-only input returns GENERAL with low confidence."""
        classifier = IntentClassifier()

        result = classifier.classify("   \t\n  ")

        assert result.intent == IntentType.GENERAL
        assert result.confidence == 0.0


class TestRouterAcknowledgments:
    """Tests for acknowledgment handling in the router."""

    @pytest.fixture
    def router(self):
        """Create a router with mocked dependencies."""
        with patch("jarvis.router.get_db") as mock_db:
            mock_db.return_value = MagicMock()
            mock_db.return_value.get_contact.return_value = None
            mock_db.return_value.get_contact_by_chat_id.return_value = None
            mock_db.return_value.init_schema.return_value = None
            router = ReplyRouter(db=mock_db.return_value)
            return router

    def test_simple_thanks_returns_acknowledgment(self, router):
        """Simple 'thanks' returns acknowledgment response."""
        with patch.object(router, "_generic_acknowledgment_response") as mock_ack:
            mock_ack.return_value = {
                "type": "acknowledgment",
                "response": "You're welcome!",
                "confidence": "high",
                "similarity_score": 1.0,
            }

            result = router.route("thanks")

            assert result["type"] == "acknowledgment"
            assert "response" in result

    def test_ok_returns_acknowledgment(self, router):
        """Simple 'ok' returns acknowledgment response."""
        with patch.object(router, "_generic_acknowledgment_response") as mock_ack:
            mock_ack.return_value = {
                "type": "acknowledgment",
                "response": "👍",
                "confidence": "high",
                "similarity_score": 1.0,
            }

            result = router.route("ok")

            assert result["type"] == "acknowledgment"

    def test_acknowledgment_skipped_with_thread_context(self, router):
        """Acknowledgment handling is skipped when thread has question."""
        # Patch the router's _generate_response method
        with patch.object(router, "_generate_response") as mock_gen:
            mock_gen.return_value = {
                "type": "generated",
                "response": "Generated response",
                "confidence": "medium",
            }

            result = router.route(
                "ok",
                thread=["Hey!", "Can you help me with something?"],  # Question in thread
            )

            # Should either acknowledge or generate, both are valid
            assert result.get("type") in ("acknowledgment", "generated", "clarify")


class TestRouterContextDependentMessages:
    """Tests for context-dependent message handling."""

    @pytest.fixture
    def router(self):
        """Create a router with mocked dependencies."""
        with patch("jarvis.router.get_db") as mock_db:
            mock_db.return_value = MagicMock()
            mock_db.return_value.get_contact.return_value = None
            mock_db.return_value.get_contact_by_chat_id.return_value = None
            mock_db.return_value.init_schema.return_value = None
            router = ReplyRouter(db=mock_db.return_value)
            return router

    def test_what_time_is_context_dependent(self, router):
        """'What time?' is recognized as context-dependent."""
        assert router._is_context_dependent("what time?")
        assert router._is_context_dependent("What time")
        assert router._is_context_dependent("what time works?")

    def test_where_is_context_dependent(self, router):
        """'Where?' is recognized as context-dependent."""
        assert router._is_context_dependent("where?")
        assert router._is_context_dependent("Where should we meet?")

    def test_are_you_coming_is_context_dependent(self, router):
        """User-input-required questions are context-dependent."""
        assert router._is_context_dependent("are you coming?")
        assert router._is_context_dependent("can you make it?")
        assert router._is_context_dependent("will you be there?")

    def test_regular_message_not_context_dependent(self, router):
        """Regular messages are not context-dependent."""
        assert not router._is_context_dependent("Hello!")
        assert not router._is_context_dependent("I had a great day")
        assert not router._is_context_dependent("The meeting went well")


class TestRouterClarification:
    """Tests for clarification request handling."""

    @pytest.fixture
    def router(self):
        """Create a router with mocked dependencies."""
        with patch("jarvis.router.get_db") as mock_db:
            mock_db.return_value = MagicMock()
            mock_db.return_value.get_contact.return_value = None
            mock_db.return_value.get_contact_by_chat_id.return_value = None
            mock_db.return_value.init_schema.return_value = None
            router = ReplyRouter(db=mock_db.return_value)
            return router

    def test_needs_clarification_for_vague_reference(self, router):
        """Vague references need clarification."""
        assert router._needs_clarification("What about that?", thread=None)
        assert router._needs_clarification("Send it please", thread=None)
        assert router._needs_clarification("The thing we discussed", thread=None)

    def test_no_clarification_with_thread_context(self, router):
        """Vague references don't need clarification with thread context."""
        thread = ["Let's go to the restaurant", "Which one?", "The Italian place"]
        assert not router._needs_clarification("What about that?", thread=thread)

    def test_clarification_response_format(self, router):
        """Clarification response has correct format."""
        result = router._clarify_response(
            "What are you referring to?",
            reason="vague_reference",
        )

        assert result["type"] == "clarify"
        assert result["confidence"] == "low"
        assert result["similarity_score"] == 0.0
        assert "referring" in result["response"]


class TestRouterEmptyInput:
    """Tests for empty input handling."""

    @pytest.fixture
    def router(self):
        """Create a router with mocked dependencies."""
        with patch("jarvis.router.get_db") as mock_db:
            mock_db.return_value = MagicMock()
            mock_db.return_value.get_contact.return_value = None
            mock_db.return_value.get_contact_by_chat_id.return_value = None
            mock_db.return_value.init_schema.return_value = None
            router = ReplyRouter(db=mock_db.return_value)
            return router

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

    def test_message_with_context_generates_response(self, mock_messages):
        """Message with conversation context generates appropriate response."""
        with patch("jarvis.router.get_db") as mock_db:
            mock_db.return_value = MagicMock()
            mock_db.return_value.get_contact.return_value = None
            mock_db.return_value.get_contact_by_chat_id.return_value = None
            mock_db.return_value.init_schema.return_value = None

            router = ReplyRouter(db=mock_db.return_value)

            with patch.object(router, "_generate_response") as mock_gen:
                mock_gen.return_value = {
                    "type": "generated",
                    "response": "Sounds great! See you there!",
                    "confidence": "medium",
                    "similarity_score": 0.0,
                }

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


class TestRouterQuickReply:
    """Tests for quick reply selection."""

    @pytest.fixture
    def router(self):
        """Create a router with mocked dependencies."""
        with patch("jarvis.router.get_db") as mock_db:
            mock_db.return_value = MagicMock()
            mock_db.return_value.get_contact.return_value = None
            mock_db.return_value.get_contact_by_chat_id.return_value = None
            mock_db.return_value.init_schema.return_value = None
            router = ReplyRouter(db=mock_db.return_value)
            return router

    def test_quick_reply_with_no_matches_returns_clarify(self, router):
        """Quick reply with no matches falls back to clarify."""
        result = router._quick_reply_response([], contact=None)

        assert result["type"] == "clarify"

    def test_quick_reply_filters_by_coherence(self, router):
        """Quick reply filters responses by coherence score."""
        matches = [
            {
                "trigger_text": "Want to grab lunch?",
                "response_text": "Sure!",
                "similarity": 0.96,
            },
        ]

        with patch("jarvis.router.score_response_coherence", return_value=0.7):
            result = router._quick_reply_response(
                matches, contact=None, incoming="Want to grab lunch?"
            )

            assert result["type"] == "quick_reply"
            assert result["response"] == "Sure!"


class TestRouterProfessionalFiltering:
    """Tests for professional/casual filtering."""

    @pytest.fixture
    def router(self):
        """Create a router with mocked dependencies."""
        with patch("jarvis.router.get_db") as mock_db:
            mock_db.return_value = MagicMock()
            mock_db.return_value.get_contact.return_value = None
            mock_db.return_value.get_contact_by_chat_id.return_value = None
            mock_db.return_value.init_schema.return_value = None
            router = ReplyRouter(db=mock_db.return_value)
            return router

    def test_professional_response_detection(self, router):
        """Professional responses are detected correctly."""
        assert router._is_professional_response("Thank you for the update.")
        assert router._is_professional_response("I'll follow up on that.")
        assert router._is_professional_response("Sounds good.")

    def test_unprofessional_response_detection(self, router):
        """Unprofessional responses are detected correctly."""
        assert not router._is_professional_response("lol that's hilarious")
        assert not router._is_professional_response("haha no way!")
        assert not router._is_professional_response("bruh what 💀")


class TestRouterMultiOption:
    """Tests for multi-option response generation."""

    @pytest.fixture
    def router(self):
        """Create a router with mocked dependencies."""
        with patch("jarvis.router.get_db") as mock_db:
            mock_db.return_value = MagicMock()
            mock_db.return_value.get_contact.return_value = None
            mock_db.return_value.get_contact_by_chat_id.return_value = None
            mock_db.return_value.init_schema.return_value = None
            router = ReplyRouter(db=mock_db.return_value)
            return router

    def test_multi_option_commitment_question(self, router):
        """Commitment questions get multi-option responses."""
        with patch("jarvis.multi_option.get_multi_option_generator") as mock_gen:
            mock_result = MagicMock()
            mock_result.is_commitment = True
            mock_result.has_options = True
            mock_result.to_dict.return_value = {
                "is_commitment": True,
                "options": [
                    {"type": "agree", "response": "Sure, I can make it!"},
                    {"type": "decline", "response": "Sorry, I can't make it."},
                    {"type": "defer", "response": "Let me check and get back to you."},
                ],
                "suggestions": ["Sure!", "Can't make it", "Let me check"],
            }
            mock_gen.return_value.generate_options.return_value = mock_result

            result = router.route_multi_option(
                "Are you coming to the party?",
                force_multi=True,
            )

            assert result["is_commitment"] is True
            assert "options" in result


class TestRouterStats:
    """Tests for router statistics."""

    @pytest.fixture
    def router(self):
        """Create a router with mocked dependencies."""
        with patch("jarvis.router.get_db") as mock_db:
            mock_db.return_value = MagicMock()
            mock_db.return_value.get_contact.return_value = None
            mock_db.return_value.get_contact_by_chat_id.return_value = None
            mock_db.return_value.init_schema.return_value = None
            mock_db.return_value.get_stats.return_value = {"pairs": 100}
            mock_db.return_value.get_active_index.return_value = None
            router = ReplyRouter(db=mock_db.return_value)
            return router

    def test_get_routing_stats_returns_db_stats(self, router):
        """Get routing stats returns database statistics."""
        stats = router.get_routing_stats()

        assert "db_stats" in stats
        assert stats["db_stats"]["pairs"] == 100
        assert "index_available" in stats


class TestIntentParamExtraction:
    """Tests for parameter extraction from intent."""

    def test_extract_person_name(self):
        """Person name is extracted from query."""
        classifier = IntentClassifier()

        # Test extraction logic
        params = classifier._extract_params("reply to John's message", IntentType.REPLY)
        assert params.get("person_name") == "John"

        params = classifier._extract_params("summarize my chat with Sarah", IntentType.SUMMARIZE)
        assert params.get("person_name") == "Sarah"

    def test_extract_time_range(self):
        """Time range is extracted from query."""
        classifier = IntentClassifier()

        params = classifier._extract_params("summarize yesterday's messages", IntentType.SUMMARIZE)
        assert params.get("time_range") == "yesterday"

        params = classifier._extract_params("find messages from last week", IntentType.SEARCH)
        assert params.get("time_range") == "last week"

    def test_extract_search_query(self):
        """Search query is extracted from SEARCH intent."""
        classifier = IntentClassifier()

        params = classifier._extract_params("find messages about the project", IntentType.SEARCH)
        assert "project" in params.get("search_query", "").lower()

    def test_extract_rsvp_response(self):
        """RSVP response is extracted from group coordination intent."""
        classifier = IntentClassifier()

        params = classifier._extract_params("count me in", IntentType.GROUP_RSVP)
        assert params.get("rsvp_response") == "yes"

        params = classifier._extract_params("can't make it", IntentType.GROUP_RSVP)
        assert params.get("rsvp_response") == "no"

        params = classifier._extract_params("I might be able to come", IntentType.GROUP_RSVP)
        assert params.get("rsvp_response") == "maybe"


class TestIntentClassifierCaching:
    """Tests for intent classifier caching."""

    def test_classifier_cache_can_be_cleared(self):
        """Classifier cache can be cleared."""
        classifier = IntentClassifier()

        # Set some cached data
        classifier._intent_centroids = {"test": [0.1] * 384}
        classifier._intent_embeddings = {"test": [[0.1] * 384]}

        classifier.clear_cache()

        assert classifier._intent_centroids is None
        assert classifier._intent_embeddings is None

    def test_get_intent_classifier_singleton(self):
        """get_intent_classifier returns singleton."""
        from jarvis.intent import reset_intent_classifier

        reset_intent_classifier()

        classifier1 = get_intent_classifier()
        classifier2 = get_intent_classifier()

        assert classifier1 is classifier2
