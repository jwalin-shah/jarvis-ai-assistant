"""Integration tests for JARVIS pipeline data flow.

Tests real pipeline behavior: MessageContext flows through classification,
mobilization, and generation stages. Verifies routing decisions, template
short-circuits, error handling, and downstream effects of classification output.

Mocks: MLX generator, iMessage DB, health checks, embedder.
Real: mobilization classifier, category classifier (fast path + fallback),
      ReplyService routing logic, classification result builder.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from contracts.models import GenerationResponse as ModelGenerationResponse
from jarvis.classifiers.cascade import classify_with_cascade
from jarvis.classifiers.classification_result import build_classification_result
from jarvis.classifiers.response_mobilization import ResponsePressure
from jarvis.contracts.pipeline import (
    CategoryType,
    ClassificationResult,
    GenerationResponse,
    IntentType,
    MessageContext,
    UrgencyLevel,
)
from jarvis.prompts import ACKNOWLEDGE_TEMPLATES, CLOSING_TEMPLATES
from jarvis.reply_service import ReplyService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(
    text: str,
    chat_id: str = "chat_test",
) -> MessageContext:
    """Build a MessageContext with sensible defaults."""
    return MessageContext(
        chat_id=chat_id,
        message_text=text,
        is_from_me=False,
        timestamp=datetime(2025, 2, 10, 12, 0, 0),
    )


def _make_classification(
    category: CategoryType,
    urgency: UrgencyLevel = UrgencyLevel.LOW,
    confidence: float = 0.85,
    category_name: str | None = None,
    pressure: str = "low",
) -> ClassificationResult:
    """Build a ClassificationResult with metadata the reply service expects."""
    cat_name = category_name or category.value
    return ClassificationResult(
        intent=IntentType.STATEMENT,
        category=category,
        urgency=urgency,
        confidence=confidence,
        requires_knowledge=False,
        metadata={
            "category_name": cat_name,
            "category_confidence": confidence,
            "category_method": "test",
            "mobilization_pressure": pressure,
            "mobilization_response_type": "optional",
            "mobilization_confidence": confidence,
            "mobilization_method": "test",
        },
    )


def _mock_generator() -> MagicMock:
    """Create a mock MLX generator that returns canned responses."""
    gen = MagicMock()
    gen.is_loaded.return_value = True
    gen.generate.return_value = ModelGenerationResponse(
        text="Sounds good, let's do it!",
        tokens_used=8,
        generation_time_ms=50.0,
        model_name="mock-model",
        used_template=False,
        template_name=None,
        finish_reason="stop",
    )
    return gen


def _make_service(generator: MagicMock | None = None) -> ReplyService:
    """Create a ReplyService with mocked external deps."""
    gen = generator or _mock_generator()
    svc = ReplyService(generator=gen)

    # Mock context service to avoid real DB/iMessage access
    mock_ctx_svc = MagicMock()
    mock_ctx_svc.get_contact.return_value = None
    mock_ctx_svc.search_examples.return_value = []
    mock_ctx_svc.get_relationship_profile.return_value = (None, None)
    mock_ctx_svc.fetch_conversation_context.return_value = []
    svc._context_service = mock_ctx_svc

    return svc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClassificationFlowsDownstream:
    """Classification output (category, pressure) drives routing."""

    def test_acknowledge_category_returns_template_no_llm(self):
        """Acknowledge short-circuits to template; LLM never called."""
        svc = _make_service()
        ctx = _make_context("ok")
        classification = _make_classification(
            CategoryType.ACKNOWLEDGE, category_name="acknowledge"
        )

        result = svc.generate_reply(ctx, classification)

        assert result.response in ACKNOWLEDGE_TEMPLATES
        assert result.confidence == 0.95
        assert result.metadata.get("category") == "acknowledge"
        svc._generator.generate.assert_not_called()

    def test_closing_category_returns_template_no_llm(self):
        """Closing short-circuits to closing template; no LLM call."""
        svc = _make_service()
        ctx = _make_context("bye")
        classification = _make_classification(
            CategoryType.CLOSING, category_name="closing"
        )

        result = svc.generate_reply(ctx, classification)

        assert result.response in CLOSING_TEMPLATES
        assert result.confidence == 0.95
        assert result.metadata.get("category") == "closing"
        svc._generator.generate.assert_not_called()

    @patch("jarvis.generation.can_use_llm", return_value=(True, "ok"))
    def test_question_category_triggers_llm_generation(self, _health):
        """Question category routes through LLM generation."""
        gen = _mock_generator()
        svc = _make_service(generator=gen)
        ctx = _make_context("What time is the meeting?")
        classification = _make_classification(
            CategoryType.FULL_RESPONSE,
            urgency=UrgencyLevel.HIGH,
            category_name="question",
            pressure="high",
        )

        result = svc.generate_reply(ctx, classification)

        assert result.response == "Sounds good, let's do it!"
        assert result.metadata.get("type") == "generated"
        gen.generate.assert_called_once()


class TestMobilizationAffectsClassification:
    """Mobilization pressure propagates into ClassificationResult."""

    def test_high_pressure_question_maps_to_high_urgency(self):
        """Direct request -> HIGH pressure -> HIGH urgency."""
        mobilization = classify_with_cascade("Can you pick me up at 5?")
        assert mobilization.pressure == ResponsePressure.HIGH

        classification = build_classification_result(
            "Can you pick me up at 5?", thread=[], mobilization=mobilization
        )
        assert classification.urgency == UrgencyLevel.HIGH
        assert classification.metadata["mobilization_pressure"] == "high"

    def test_backchannel_maps_to_none_pressure(self):
        """Backchannel 'ok' -> NONE pressure -> LOW urgency."""
        mobilization = classify_with_cascade("ok")
        assert mobilization.pressure == ResponsePressure.NONE

        classification = build_classification_result(
            "ok", thread=[], mobilization=mobilization
        )
        assert classification.urgency == UrgencyLevel.LOW

    def test_emotional_message_maps_to_medium(self):
        """Emotional content -> MEDIUM pressure -> MEDIUM urgency."""
        mobilization = classify_with_cascade("I got the job!!")
        assert mobilization.pressure == ResponsePressure.MEDIUM

        classification = build_classification_result(
            "I got the job!!", thread=[], mobilization=mobilization
        )
        assert classification.urgency == UrgencyLevel.MEDIUM


class TestSearchResultsPassthrough:
    """Search results from RAG pass through to generation."""

    @patch("jarvis.generation.can_use_llm", return_value=(True, "ok"))
    def test_search_results_flow_to_generator(self, _health):
        """Pre-computed search results are wired into the generation request."""
        gen = _mock_generator()
        svc = _make_service(generator=gen)
        ctx = _make_context("Want to grab lunch?")
        classification = _make_classification(
            CategoryType.FULL_RESPONSE,
            urgency=UrgencyLevel.HIGH,
            category_name="request",
            pressure="high",
        )
        search_results = [
            {
                "trigger_text": "Want to get dinner?",
                "response_text": "Sure, where?",
                "similarity": 0.88,
                "topic": "food",
            },
        ]

        result = svc.generate_reply(
            ctx, classification, search_results=search_results
        )

        assert result.metadata.get("type") == "generated"
        gen.generate.assert_called_once()
        # Search similarity should be tracked in metadata
        assert result.metadata.get("similarity_score") == 0.88


class TestErrorHandling:
    """Errors in one pipeline stage are handled gracefully."""

    @patch("jarvis.generation.can_use_llm", return_value=(True, "ok"))
    def test_llm_failure_returns_graceful_fallback(self, _health):
        """LLM exception -> fallback response, not crash."""
        gen = _mock_generator()
        gen.generate.side_effect = RuntimeError("GPU out of memory")
        svc = _make_service(generator=gen)
        ctx = _make_context("Tell me about your day")
        classification = _make_classification(
            CategoryType.FULL_RESPONSE, category_name="statement"
        )

        result = svc.generate_reply(ctx, classification)

        assert result.metadata.get("type") == "clarify"
        assert "generation_error" in result.metadata.get("reason", "")
        assert result.confidence < 0.5

    @patch("jarvis.generation.can_use_llm", return_value=(False, "Memory critical"))
    def test_health_check_failure_returns_fallback(self, _health):
        """Bad system health -> fallback, no LLM call."""
        svc = _make_service()
        ctx = _make_context("What's up?")
        classification = _make_classification(
            CategoryType.FULL_RESPONSE, category_name="question"
        )

        result = svc.generate_reply(ctx, classification)

        assert result.metadata.get("type") == "fallback"
        assert result.confidence == 0.0
        assert "Memory critical" in result.metadata.get("reason", "")
        svc._generator.generate.assert_not_called()

    def test_empty_message_returns_clarify(self):
        """Empty message text short-circuits to clarify."""
        svc = _make_service()
        ctx = _make_context("   ")
        classification = _make_classification(CategoryType.FULL_RESPONSE)

        result = svc.generate_reply(ctx, classification)

        assert result.metadata.get("type") == "clarify"
        assert result.metadata.get("reason") == "empty_message"
        svc._generator.generate.assert_not_called()


class TestEndToEndPipelineFlow:
    """Full flow: mobilization -> classification -> reply service."""

    @patch("jarvis.generation.can_use_llm", return_value=(True, "ok"))
    def test_question_end_to_end(self, _health):
        """Question flows through all stages -> generated reply."""
        incoming = "Where should we eat tonight?"

        # Stage 1: Mobilization (real)
        mobilization = classify_with_cascade(incoming)
        assert mobilization.pressure in {
            ResponsePressure.HIGH,
            ResponsePressure.MEDIUM,
        }

        # Stage 2: Classification (real fast-path + fallback)
        classification = build_classification_result(
            incoming, thread=[], mobilization=mobilization
        )
        assert classification.category in {
            CategoryType.FULL_RESPONSE,
            CategoryType.DEFER,
        }

        # Stage 3: Generation (mocked LLM)
        gen = _mock_generator()
        svc = _make_service(generator=gen)
        ctx = _make_context(incoming)

        result = svc.generate_reply(ctx, classification)

        assert result.response  # Non-empty
        assert result.metadata.get("type") == "generated"
        gen.generate.assert_called_once()

    def test_acknowledgment_end_to_end(self):
        """Acknowledgment flows through all stages and skips LLM."""
        incoming = "ok"

        # Stage 1: Mobilization (real)
        mobilization = classify_with_cascade(incoming)
        assert mobilization.pressure == ResponsePressure.NONE

        # Stage 2: Classification (real - fast path detects ack)
        classification = build_classification_result(
            incoming, thread=[], mobilization=mobilization
        )
        assert classification.category == CategoryType.ACKNOWLEDGE

        # Stage 3: Generation (template, no LLM)
        svc = _make_service()
        ctx = _make_context(incoming)
        result = svc.generate_reply(ctx, classification)

        assert result.response in ACKNOWLEDGE_TEMPLATES
        assert result.confidence == 0.95
        svc._generator.generate.assert_not_called()
