"""Unit tests for ReplyService - static/pure methods and integration paths.

Tests cover:
- _compute_confidence(): multi-factor confidence scoring
- _compute_example_diversity(): diversity metric
- _safe_float(): type coercion
- _max_tokens_for_pressure(): token budget mapping
- _pressure_from_classification(): enum mapping with fallback
- _build_mobilization_hint(): prompt hint generation
- generate_reply(): template, skip, fallback, and error paths
- generate_reply(): happy-path generation (RAG -> LLM -> response)
"""

from __future__ import annotations

import random
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from jarvis.classifiers.response_mobilization import (
    MobilizationResult,
    ResponsePressure,
    ResponseType,
)
from jarvis.contracts.pipeline import (
    CategoryType,
    ClassificationResult,
    GenerationRequest,
    IntentType,
    MessageContext,
    UrgencyLevel,
)
from jarvis.reply_service import ReplyService


# =============================================================================
# _compute_confidence Tests
# =============================================================================


class TestComputeConfidence:
    """Tests for ReplyService._compute_confidence static method."""

    def test_high_pressure_baseline(self) -> None:
        """HIGH pressure with good signals → 0.85 baseline."""
        score, label = ReplyService._compute_confidence(
            pressure=ResponsePressure.HIGH,
            rag_similarity=0.8,
            example_diversity=0.8,
            reply_length=10,
            reply_text="Sure, I can pick you up at 5",
        )
        assert score == pytest.approx(0.85, abs=0.01)
        assert label == "high"

    def test_none_pressure_baseline(self) -> None:
        """NONE pressure → 0.30 baseline."""
        score, label = ReplyService._compute_confidence(
            pressure=ResponsePressure.NONE,
            rag_similarity=0.8,
            example_diversity=0.8,
            reply_length=5,
            reply_text="got it",
        )
        assert score == pytest.approx(0.30, abs=0.01)
        assert label == "low"

    def test_medium_pressure_baseline(self) -> None:
        """MEDIUM pressure → 0.65 baseline."""
        score, label = ReplyService._compute_confidence(
            pressure=ResponsePressure.MEDIUM,
            rag_similarity=0.8,
            example_diversity=0.8,
            reply_length=10,
            reply_text="That's amazing, congrats!",
        )
        assert score == pytest.approx(0.65, abs=0.01)
        assert label == "medium"

    def test_low_pressure_baseline(self) -> None:
        """LOW pressure → 0.45 baseline."""
        score, label = ReplyService._compute_confidence(
            pressure=ResponsePressure.LOW,
            rag_similarity=0.8,
            example_diversity=0.8,
            reply_length=10,
            reply_text="Yeah the weather is nice",
        )
        assert score == pytest.approx(0.45, abs=0.01)
        assert label == "medium"

    def test_low_rag_similarity_penalty(self) -> None:
        """RAG similarity < 0.5 applies 0.8x penalty."""
        good_score, _ = ReplyService._compute_confidence(
            pressure=ResponsePressure.HIGH,
            rag_similarity=0.8,
            example_diversity=0.8,
            reply_length=10,
            reply_text="reply text",
        )
        bad_score, _ = ReplyService._compute_confidence(
            pressure=ResponsePressure.HIGH,
            rag_similarity=0.3,
            example_diversity=0.8,
            reply_length=10,
            reply_text="reply text",
        )
        assert bad_score < good_score
        assert bad_score == pytest.approx(0.85 * 0.8, abs=0.01)

    def test_rerank_boost(self) -> None:
        """Rerank score > 0.7 boosts confidence by 1.1x (capped at 0.95)."""
        base_score, _ = ReplyService._compute_confidence(
            pressure=ResponsePressure.HIGH,
            rag_similarity=0.8,
            example_diversity=0.8,
            reply_length=10,
            reply_text="reply text",
            rerank_score=None,
        )
        boosted_score, _ = ReplyService._compute_confidence(
            pressure=ResponsePressure.HIGH,
            rag_similarity=0.8,
            example_diversity=0.8,
            reply_length=10,
            reply_text="reply text",
            rerank_score=0.9,
        )
        assert boosted_score > base_score
        # 0.85 * 1.1 = 0.935, capped at 0.95
        assert boosted_score == pytest.approx(min(0.85 * 1.1, 0.95), abs=0.01)

    def test_rerank_below_threshold_no_boost(self) -> None:
        """Rerank score <= 0.7 does not boost."""
        base_score, _ = ReplyService._compute_confidence(
            pressure=ResponsePressure.HIGH,
            rag_similarity=0.8,
            example_diversity=0.8,
            reply_length=10,
            reply_text="reply text",
            rerank_score=None,
        )
        same_score, _ = ReplyService._compute_confidence(
            pressure=ResponsePressure.HIGH,
            rag_similarity=0.8,
            example_diversity=0.8,
            reply_length=10,
            reply_text="reply text",
            rerank_score=0.5,
        )
        assert same_score == base_score

    def test_low_diversity_penalty(self) -> None:
        """Example diversity < 0.3 applies 0.9x penalty."""
        good_score, _ = ReplyService._compute_confidence(
            pressure=ResponsePressure.HIGH,
            rag_similarity=0.8,
            example_diversity=0.8,
            reply_length=10,
            reply_text="reply text",
        )
        low_div_score, _ = ReplyService._compute_confidence(
            pressure=ResponsePressure.HIGH,
            rag_similarity=0.8,
            example_diversity=0.1,
            reply_length=10,
            reply_text="reply text",
        )
        assert low_div_score < good_score
        assert low_div_score == pytest.approx(0.85 * 0.9, abs=0.01)

    def test_uncertain_signal_short_reply_high_pressure(self) -> None:
        """Short uncertain reply ('?') with HIGH pressure gets 0.7x penalty."""
        score, _ = ReplyService._compute_confidence(
            pressure=ResponsePressure.HIGH,
            rag_similarity=0.8,
            example_diversity=0.8,
            reply_length=1,
            reply_text="?",
        )
        assert score == pytest.approx(0.85 * 0.7, abs=0.01)

    def test_uncertain_signal_not_high_pressure_no_penalty(self) -> None:
        """Uncertain signal with LOW pressure → no uncertain penalty."""
        score, _ = ReplyService._compute_confidence(
            pressure=ResponsePressure.LOW,
            rag_similarity=0.8,
            example_diversity=0.8,
            reply_length=1,
            reply_text="?",
        )
        # LOW baseline = 0.45, no uncertain penalty (only applies to HIGH)
        assert score == pytest.approx(0.45, abs=0.01)

    def test_uncertain_signal_long_reply_no_penalty(self) -> None:
        """Long reply with '?' is not treated as uncertain (reply_length >= 3)."""
        score, _ = ReplyService._compute_confidence(
            pressure=ResponsePressure.HIGH,
            rag_similarity=0.8,
            example_diversity=0.8,
            reply_length=5,
            reply_text="?",
        )
        assert score == pytest.approx(0.85, abs=0.01)

    def test_coherence_penalty_parrot(self) -> None:
        """Reply that exactly matches incoming gets 0.5x penalty."""
        score, _ = ReplyService._compute_confidence(
            pressure=ResponsePressure.HIGH,
            rag_similarity=0.8,
            example_diversity=0.8,
            reply_length=5,
            reply_text="How are you doing?",
            incoming_text="How are you doing?",
        )
        assert score == pytest.approx(0.85 * 0.5, abs=0.01)

    def test_coherence_penalty_prefix_match(self) -> None:
        """Incoming starts with reply (>5 chars) gets 0.5x penalty."""
        score, _ = ReplyService._compute_confidence(
            pressure=ResponsePressure.HIGH,
            rag_similarity=0.8,
            example_diversity=0.8,
            reply_length=3,
            reply_text="How are you",
            incoming_text="How are you doing today?",
        )
        assert score == pytest.approx(0.85 * 0.5, abs=0.01)

    def test_confidence_clamped_0_1(self) -> None:
        """Confidence is always in [0, 1] regardless of combined penalties."""
        score, _ = ReplyService._compute_confidence(
            pressure=ResponsePressure.HIGH,
            rag_similarity=0.1,
            example_diversity=0.1,
            reply_length=1,
            reply_text="?",
            incoming_text="?",
        )
        assert 0.0 <= score <= 1.0

    def test_label_high_threshold(self) -> None:
        """Score >= 0.7 → 'high' label."""
        _, label = ReplyService._compute_confidence(
            pressure=ResponsePressure.HIGH,
            rag_similarity=0.8,
            example_diversity=0.8,
            reply_length=10,
            reply_text="Great response here",
            rerank_score=0.9,
        )
        assert label == "high"

    def test_label_medium_threshold(self) -> None:
        """Score in [0.45, 0.7) → 'medium' label."""
        score, label = ReplyService._compute_confidence(
            pressure=ResponsePressure.MEDIUM,
            rag_similarity=0.8,
            example_diversity=0.8,
            reply_length=10,
            reply_text="That's nice",
        )
        assert 0.45 <= score < 0.7
        assert label == "medium"

    def test_label_low_threshold(self) -> None:
        """Score < 0.45 → 'low' label."""
        score, label = ReplyService._compute_confidence(
            pressure=ResponsePressure.NONE,
            rag_similarity=0.8,
            example_diversity=0.8,
            reply_length=5,
            reply_text="ok",
        )
        assert score < 0.45
        assert label == "low"

    def test_multiple_penalties_stack(self) -> None:
        """Low RAG + low diversity penalties stack multiplicatively."""
        score, _ = ReplyService._compute_confidence(
            pressure=ResponsePressure.HIGH,
            rag_similarity=0.3,  # 0.8x
            example_diversity=0.1,  # 0.9x
            reply_length=10,
            reply_text="some reply",
        )
        assert score == pytest.approx(0.85 * 0.8 * 0.9, abs=0.01)


# =============================================================================
# _compute_example_diversity Tests
# =============================================================================


class TestComputeExampleDiversity:
    """Tests for ReplyService._compute_example_diversity static method."""

    def test_empty_results(self) -> None:
        """Empty list → 0.0."""
        assert ReplyService._compute_example_diversity([]) == 0.0

    def test_all_unique(self) -> None:
        """All unique trigger texts → 1.0."""
        results = [
            {"trigger_text": "hello"},
            {"trigger_text": "goodbye"},
            {"trigger_text": "how are you"},
        ]
        assert ReplyService._compute_example_diversity(results) == 1.0

    def test_all_duplicate(self) -> None:
        """All same trigger text → 1/n."""
        results = [
            {"trigger_text": "hello"},
            {"trigger_text": "hello"},
            {"trigger_text": "hello"},
        ]
        score = ReplyService._compute_example_diversity(results)
        assert score == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_partial_duplicates(self) -> None:
        """Mix of unique and duplicate → between 0 and 1."""
        results = [
            {"trigger_text": "hello"},
            {"trigger_text": "hello"},
            {"trigger_text": "goodbye"},
            {"trigger_text": "how are you"},
        ]
        score = ReplyService._compute_example_diversity(results)
        # 3 unique out of 4
        assert score == pytest.approx(3.0 / 4.0, abs=0.01)

    def test_missing_trigger_text(self) -> None:
        """Results without trigger_text key use empty string."""
        results = [
            {"trigger_text": "hello"},
            {},
            {"trigger_text": "goodbye"},
        ]
        score = ReplyService._compute_example_diversity(results)
        # "hello", "", "goodbye" = 3 unique out of 3
        assert score == 1.0

    def test_single_result(self) -> None:
        """Single result → 1.0 (1 unique / 1 total)."""
        assert ReplyService._compute_example_diversity([{"trigger_text": "hi"}]) == 1.0


# =============================================================================
# _safe_float Tests
# =============================================================================


class TestSafeFloat:
    """Tests for ReplyService._safe_float static method."""

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            (3.14, 3.14),
            (42, 42.0),
            ("0.75", 0.75),
            (None, 0.0),
            ("not_a_number", 0.0),
            ("", 0.0),
            ([1, 2, 3], 0.0),
        ],
        ids=[
            "valid_float",
            "valid_int",
            "string_number",
            "none_returns_default",
            "invalid_string_returns_default",
            "empty_string_returns_default",
            "list_returns_default",
        ],
    )
    def test_safe_float_conversion(self, input_val, expected: float) -> None:
        assert ReplyService._safe_float(input_val) == expected

    def test_none_returns_custom_default(self) -> None:
        assert ReplyService._safe_float(None, default=-1.0) == -1.0


# =============================================================================
# _max_tokens_for_pressure Tests
# =============================================================================


class TestMaxTokensForPressure:
    """Tests for ReplyService._max_tokens_for_pressure static method."""

    @pytest.mark.parametrize(
        "pressure,expected_tokens",
        [
            (ResponsePressure.NONE, 20),
            (ResponsePressure.HIGH, 40),
            (ResponsePressure.MEDIUM, 40),
            (ResponsePressure.LOW, 40),
        ],
        ids=["none", "high", "medium", "low"],
    )
    def test_max_tokens_for_pressure(self, pressure: ResponsePressure, expected_tokens: int) -> None:
        assert ReplyService._max_tokens_for_pressure(pressure) == expected_tokens


# =============================================================================
# _pressure_from_classification Tests
# =============================================================================


class TestPressureFromClassification:
    """Tests for ReplyService._pressure_from_classification static method."""

    def _make_classification(
        self,
        category: CategoryType = CategoryType.FULL_RESPONSE,
        urgency: UrgencyLevel = UrgencyLevel.LOW,
        metadata: dict | None = None,
    ) -> ClassificationResult:
        return ClassificationResult(
            intent=IntentType.STATEMENT,
            category=category,
            urgency=urgency,
            confidence=0.8,
            requires_knowledge=False,
            metadata=metadata or {},
        )

    def test_metadata_pressure_string_used(self) -> None:
        """When metadata has valid pressure string, it's used directly."""
        c = self._make_classification(metadata={"mobilization_pressure": "high"})
        assert ReplyService._pressure_from_classification(c) == ResponsePressure.HIGH

    def test_invalid_pressure_string_fallback(self) -> None:
        """Invalid pressure string falls through to category/urgency logic."""
        c = self._make_classification(
            metadata={"mobilization_pressure": "invalid_value"},
            urgency=UrgencyLevel.MEDIUM,
        )
        assert ReplyService._pressure_from_classification(c) == ResponsePressure.MEDIUM

    def test_acknowledge_category(self) -> None:
        """ACKNOWLEDGE category → NONE regardless of urgency."""
        c = self._make_classification(
            category=CategoryType.ACKNOWLEDGE,
            urgency=UrgencyLevel.HIGH,
        )
        assert ReplyService._pressure_from_classification(c) == ResponsePressure.NONE

    def test_closing_category(self) -> None:
        """CLOSING category → NONE regardless of urgency."""
        c = self._make_classification(
            category=CategoryType.CLOSING,
            urgency=UrgencyLevel.HIGH,
        )
        assert ReplyService._pressure_from_classification(c) == ResponsePressure.NONE

    def test_off_topic_category(self) -> None:
        """OFF_TOPIC category → NONE."""
        c = self._make_classification(
            category=CategoryType.OFF_TOPIC,
            urgency=UrgencyLevel.HIGH,
        )
        assert ReplyService._pressure_from_classification(c) == ResponsePressure.NONE

    def test_high_urgency(self) -> None:
        """FULL_RESPONSE with HIGH urgency → HIGH pressure."""
        c = self._make_classification(urgency=UrgencyLevel.HIGH)
        assert ReplyService._pressure_from_classification(c) == ResponsePressure.HIGH

    def test_medium_urgency(self) -> None:
        """FULL_RESPONSE with MEDIUM urgency → MEDIUM pressure."""
        c = self._make_classification(urgency=UrgencyLevel.MEDIUM)
        assert ReplyService._pressure_from_classification(c) == ResponsePressure.MEDIUM

    def test_low_urgency_fallback(self) -> None:
        """FULL_RESPONSE with LOW urgency → LOW pressure."""
        c = self._make_classification(urgency=UrgencyLevel.LOW)
        assert ReplyService._pressure_from_classification(c) == ResponsePressure.LOW

    def test_non_string_pressure_ignored(self) -> None:
        """Non-string pressure value in metadata is ignored (falls through)."""
        c = self._make_classification(
            metadata={"mobilization_pressure": 42},
            urgency=UrgencyLevel.HIGH,
        )
        assert ReplyService._pressure_from_classification(c) == ResponsePressure.HIGH


# =============================================================================
# _build_mobilization_hint Tests
# =============================================================================


class TestBuildMobilizationHint:
    """Tests for ReplyService._build_mobilization_hint static method."""

    def _make_mobilization(
        self,
        pressure: ResponsePressure,
        response_type: ResponseType = ResponseType.OPTIONAL,
    ) -> MobilizationResult:
        return MobilizationResult(
            pressure=pressure,
            response_type=response_type,
            confidence=0.8,
            features={},
            method="test",
        )

    def test_high_pressure_commitment(self) -> None:
        m = self._make_mobilization(ResponsePressure.HIGH, ResponseType.COMMITMENT)
        hint = ReplyService._build_mobilization_hint(m)
        assert "commitment" in hint.lower() or "accept" in hint.lower()

    def test_high_pressure_answer(self) -> None:
        m = self._make_mobilization(ResponsePressure.HIGH, ResponseType.ANSWER)
        hint = ReplyService._build_mobilization_hint(m)
        assert "answer" in hint.lower()

    def test_high_pressure_confirmation(self) -> None:
        m = self._make_mobilization(ResponsePressure.HIGH, ResponseType.CONFIRMATION)
        hint = ReplyService._build_mobilization_hint(m)
        assert "confirm" in hint.lower()

    def test_high_pressure_default(self) -> None:
        """HIGH pressure with non-specific type → generic direct response."""
        m = self._make_mobilization(ResponsePressure.HIGH, ResponseType.OPTIONAL)
        hint = ReplyService._build_mobilization_hint(m)
        assert "respond" in hint.lower() or "question" in hint.lower()

    def test_medium_pressure(self) -> None:
        m = self._make_mobilization(ResponsePressure.MEDIUM)
        hint = ReplyService._build_mobilization_hint(m)
        assert "emotion" in hint.lower() or "empathy" in hint.lower()

    def test_low_pressure(self) -> None:
        m = self._make_mobilization(ResponsePressure.LOW)
        hint = ReplyService._build_mobilization_hint(m)
        assert "brief" in hint.lower() or "casual" in hint.lower()

    def test_none_pressure(self) -> None:
        m = self._make_mobilization(ResponsePressure.NONE)
        hint = ReplyService._build_mobilization_hint(m)
        assert "acknowledgment" in hint.lower() or "brief" in hint.lower()


# =============================================================================
# generate_reply Integration Tests (with mocks)
# =============================================================================


class TestGenerateReply:
    """Tests for ReplyService.generate_reply with mocked dependencies."""

    @pytest.fixture(autouse=True)
    def mock_health_check(self):
        with patch("jarvis.generation.can_use_llm", return_value=(True, "ok")):
            yield

    def _make_context(self, text: str = "hello") -> MessageContext:
        return MessageContext(
            chat_id="chat123",
            message_text=text,
            is_from_me=False,
            timestamp=datetime.utcnow(),
            metadata={"thread": []},
        )

    def _make_classification(
        self,
        category: CategoryType = CategoryType.FULL_RESPONSE,
        category_name: str = "statement",
        urgency: UrgencyLevel = UrgencyLevel.MEDIUM,
    ) -> ClassificationResult:
        return ClassificationResult(
            intent=IntentType.STATEMENT,
            category=category,
            urgency=urgency,
            confidence=0.8,
            requires_knowledge=False,
            metadata={
                "category_name": category_name,
                "mobilization_pressure": "medium",
                "mobilization_response_type": "optional",
            },
        )

    def _make_service(self, mock_generator: MagicMock | None = None) -> ReplyService:
        mock_gen = mock_generator or MagicMock()
        if mock_generator is None:
            mock_response = MagicMock()
            mock_response.text = "Generated response"
            mock_gen.generate.return_value = mock_response
            mock_gen.is_loaded.return_value = True
        svc = ReplyService(generator=mock_gen)
        return svc

    def test_empty_message_returns_clarify(self) -> None:
        svc = self._make_service()
        ctx = self._make_context("")
        result = svc.generate_reply(ctx, self._make_classification())
        assert result.metadata["type"] == "clarify"
        assert result.metadata["reason"] == "empty_message"
        assert result.confidence == pytest.approx(0.2)

    def test_whitespace_message_returns_clarify(self) -> None:
        svc = self._make_service()
        ctx = self._make_context("   ")
        result = svc.generate_reply(ctx, self._make_classification())
        assert result.metadata["type"] == "clarify"

    def test_acknowledge_category_returns_template(self) -> None:
        """Acknowledge category skips SLM and returns a template."""
        from jarvis.prompts import ACKNOWLEDGE_TEMPLATES

        svc = self._make_service()
        mock_gen = svc._generator
        ctx = self._make_context("ok")
        classification = self._make_classification(
            category=CategoryType.ACKNOWLEDGE,
            category_name="acknowledge",
        )
        result = svc.generate_reply(ctx, classification)

        assert result.metadata["type"] == "acknowledge"
        assert result.response in ACKNOWLEDGE_TEMPLATES
        assert result.confidence == 0.95
        mock_gen.generate.assert_not_called()

    def test_closing_category_returns_template(self) -> None:
        """Closing category skips SLM and returns a template."""
        from jarvis.prompts import CLOSING_TEMPLATES

        svc = self._make_service()
        mock_gen = svc._generator
        ctx = self._make_context("bye")
        classification = self._make_classification(
            category=CategoryType.CLOSING,
            category_name="closing",
        )
        result = svc.generate_reply(ctx, classification)

        assert result.metadata["type"] == "closing"
        assert result.response in CLOSING_TEMPLATES
        assert result.confidence == 0.95
        mock_gen.generate.assert_not_called()

    def test_health_check_failure_returns_fallback(self) -> None:
        """When LLM health check fails, returns fallback response."""
        svc = self._make_service()
        ctx = self._make_context("tell me something")
        classification = self._make_classification()

        with patch("jarvis.generation.can_use_llm", return_value=(False, "memory_pressure")):
            result = svc.generate_reply(ctx, classification, search_results=[])

        assert result.metadata["type"] == "fallback"
        assert result.metadata["reason"] == "memory_pressure"
        assert result.confidence == 0.0
        assert result.response == ""

    def test_generation_error_returns_clarify(self) -> None:
        """When LLM generation throws, returns clarify with generation_error reason."""
        mock_gen = MagicMock()
        mock_gen.generate.side_effect = RuntimeError("GPU OOM")
        mock_gen.is_loaded.return_value = True
        svc = self._make_service(mock_gen)
        ctx = self._make_context("tell me something")
        classification = self._make_classification()

        # Mock the reranker and context service to avoid real model loading
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = []
        svc._reranker = mock_reranker

        mock_ctx_svc = MagicMock()
        mock_ctx_svc.get_relationship_profile.return_value = (None, "")
        mock_ctx_svc.fetch_conversation_context.return_value = []
        svc._context_service = mock_ctx_svc

        result = svc.generate_reply(ctx, classification, search_results=[])

        assert result.metadata["type"] == "clarify"
        assert result.metadata["reason"] == "generation_error"
        assert "trouble" in result.response.lower()

    def test_none_pressure_no_examples_returns_skip(self) -> None:
        """NONE pressure + no search results → skip (no response needed)."""
        svc = self._make_service()
        ctx = self._make_context("ok cool")
        classification = ClassificationResult(
            intent=IntentType.STATEMENT,
            category=CategoryType.FULL_RESPONSE,
            urgency=UrgencyLevel.LOW,
            confidence=0.8,
            requires_knowledge=False,
            metadata={
                "category_name": "emotion",
                "mobilization_pressure": "none",
                "mobilization_response_type": "optional",
            },
        )

        mock_ctx_svc = MagicMock()
        mock_ctx_svc.get_relationship_profile.return_value = (None, "")
        mock_ctx_svc.fetch_conversation_context.return_value = []
        svc._context_service = mock_ctx_svc

        result = svc.generate_reply(ctx, classification, search_results=[])

        assert result.metadata["type"] == "skip"
        assert result.metadata["reason"] == "no_response_needed"


# =============================================================================
# Happy-Path Generation Tests
# =============================================================================


class TestGenerateReplyHappyPath:
    """Tests for the main generation codepath: message + search + RAG -> response.

    These tests verify that data flows correctly through the pipeline:
    search results -> reranker -> prompt builder -> LLM generator -> response.
    Only the generator (MLX model), context service, and reranker are mocked.
    """

    @pytest.fixture(autouse=True)
    def mock_health_check(self):
        with patch("jarvis.generation.can_use_llm", return_value=(True, "ok")):
            yield

    def _make_context(
        self,
        text: str = "Want to grab dinner tonight?",
        chat_id: str = "chat123",
        thread: list[str] | None = None,
    ) -> MessageContext:
        return MessageContext(
            chat_id=chat_id,
            message_text=text,
            is_from_me=False,
            timestamp=datetime.utcnow(),
            metadata={"thread": thread or []},
        )

    def _make_classification(
        self,
        category_name: str = "question",
        urgency: UrgencyLevel = UrgencyLevel.MEDIUM,
        pressure: str = "medium",
    ) -> ClassificationResult:
        return ClassificationResult(
            intent=IntentType.STATEMENT,
            category=CategoryType.FULL_RESPONSE,
            urgency=urgency,
            confidence=0.85,
            requires_knowledge=False,
            metadata={
                "category_name": category_name,
                "mobilization_pressure": pressure,
                "mobilization_response_type": "answer",
            },
        )

    def _make_search_results(
        self,
        count: int = 3,
        similarity: float = 0.82,
    ) -> list[dict]:
        """Create realistic search results with distinct trigger/response pairs."""
        pairs = [
            ("Want to get food later?", "Sure, how about Thai?"),
            ("Are you free for lunch?", "Yeah I can do noon"),
            ("Dinner tonight?", "Sounds good, where?"),
            ("Wanna grab coffee?", "Definitely, I'll meet you there"),
            ("Should we order pizza?", "Yes please, pepperoni"),
        ]
        results = []
        for i in range(min(count, len(pairs))):
            trigger, response = pairs[i]
            results.append({
                "trigger_text": trigger,
                "response_text": response,
                "similarity": similarity - (i * 0.05),
                "rerank_score": 0.75 - (i * 0.05),
                "topic": "food",
            })
        return results

    def _make_contact(
        self,
        display_name: str = "Alex",
        style_notes: str | None = None,
    ):
        from jarvis.db import Contact

        return Contact(
            id=1,
            chat_id="chat123",
            display_name=display_name,
            phone_or_email="+15551234567",
            relationship="friend",
            style_notes=style_notes,
        )

    def _make_service(
        self, generated_text: str = "Sounds great, where do you want to go?"
    ) -> ReplyService:
        """Create a ReplyService with controlled mock dependencies."""
        mock_gen = MagicMock()
        mock_response = MagicMock()
        mock_response.text = generated_text
        mock_gen.generate.return_value = mock_response
        mock_gen.is_loaded.return_value = True

        svc = ReplyService(generator=mock_gen)

        # Mock context service to avoid real DB/iMessage access
        mock_ctx_svc = MagicMock()
        mock_ctx_svc.get_relationship_profile.return_value = (None, None)
        mock_ctx_svc.fetch_conversation_context.return_value = []
        svc._context_service = mock_ctx_svc

        # Mock reranker to pass through candidates (preserving order)
        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = (
            lambda query, candidates, **kw: candidates[: kw.get("top_k", 5)]
        )
        svc._reranker = mock_reranker

        return svc

    def test_happy_path_generation(self) -> None:
        """MEDIUM pressure + search results -> generated response with metadata."""
        generated_text = "Sounds great, where do you want to go?"
        svc = self._make_service(generated_text)
        ctx = self._make_context("Want to grab dinner tonight?")
        classification = self._make_classification(pressure="medium")
        search_results = self._make_search_results(count=3, similarity=0.82)

        result = svc.generate_reply(ctx, classification, search_results=search_results)

        # Verify the generated text flows through to the response
        assert result.response == generated_text
        assert result.metadata["type"] == "generated"
        # MEDIUM pressure baseline = 0.65; similarity 0.82 >= 0.5 -> no penalty
        assert 0.0 < result.confidence <= 1.0
        assert result.metadata["similarity_score"] == pytest.approx(0.82, abs=0.01)
        assert result.metadata["vec_candidates"] == 3
        # All 3 search results have unique triggers -> diversity = 1.0
        assert result.metadata["example_diversity"] == pytest.approx(1.0, abs=0.01)
        # Generator was actually called
        svc._generator.generate.assert_called_once()

    def test_contact_name_in_prompt(self) -> None:
        """Contact display_name appears in the prompt passed to the generator."""
        svc = self._make_service("yeah for sure!")
        contact = self._make_contact(display_name="Jordan")
        # Return a relationship profile dict so it's injected into context metadata
        svc._context_service.get_relationship_profile.return_value = (
            {"tone": "casual", "avg_message_length": 25},
            None,
        )

        ctx = self._make_context("Hey are you coming to the party?")
        classification = self._make_classification(pressure="medium")
        search_results = self._make_search_results(count=2, similarity=0.7)

        result = svc.generate_reply(
            ctx, classification, search_results=search_results, contact=contact
        )

        assert result.response == "yeah for sure!"
        assert result.metadata["type"] == "generated"

        # Verify the prompt contains the contact name
        model_request = svc._generator.generate.call_args[0][0]
        prompt_text = model_request.prompt
        assert "Jordan" in prompt_text

    def test_thread_context_included_in_prompt(self) -> None:
        """Thread messages are included in the generation prompt."""
        svc = self._make_service("I'll be there at 7!")
        thread = [
            "Hey are we still on for tonight?",
            "Yeah definitely, what time?",
            "How about 7pm?",
        ]
        ctx = self._make_context(text="How about 7pm?", thread=thread)
        classification = self._make_classification(
            category_name="question", pressure="high"
        )
        search_results = self._make_search_results(count=2, similarity=0.75)

        result = svc.generate_reply(
            ctx, classification, search_results=search_results, thread=thread
        )

        assert result.response == "I'll be there at 7!"
        assert result.metadata["type"] == "generated"

        # Verify thread messages appear in the prompt
        model_request = svc._generator.generate.call_args[0][0]
        prompt_text = model_request.prompt
        # Thread messages should be in the prompt context section
        assert "still on for tonight" in prompt_text or "what time" in prompt_text

    def test_search_results_flow_to_prompt(self) -> None:
        """Search results' trigger/response pairs appear in the prompt as examples."""
        svc = self._make_service("How about Thai food?")
        ctx = self._make_context("Want to grab dinner?")
        classification = self._make_classification(pressure="medium")
        search_results = self._make_search_results(count=3, similarity=0.85)

        result = svc.generate_reply(ctx, classification, search_results=search_results)

        assert result.metadata["type"] == "generated"

        # Verify search result trigger/response pairs appear in the prompt
        model_request = svc._generator.generate.call_args[0][0]
        prompt_text = model_request.prompt
        # Top search result's trigger text should be in the prompt
        assert "Want to get food later?" in prompt_text
        # Its response text should also be present as the example reply
        assert "Sure, how about Thai?" in prompt_text

        # Verify similar_triggers metadata is populated from search results
        assert result.metadata["similar_triggers"]
        assert "Want to get food later?" in result.metadata["similar_triggers"]

    def test_confidence_high_pressure_high_similarity(self) -> None:
        """HIGH pressure + high similarity + rerank boost -> high confidence."""
        svc = self._make_service("I'll pick you up at 5")
        ctx = self._make_context("Can you pick me up at 5?")
        classification = self._make_classification(pressure="high")
        search_results = self._make_search_results(count=3, similarity=0.9)

        result = svc.generate_reply(ctx, classification, search_results=search_results)

        # HIGH baseline = 0.85; sim 0.9 >= 0.5 no penalty;
        # rerank_score 0.75 > 0.7 gives 1.1x boost -> 0.85 * 1.1 = 0.935
        assert result.confidence >= 0.7
        assert result.metadata["confidence_label"] == "high"

    def test_confidence_penalized_by_low_similarity(self) -> None:
        """MEDIUM pressure + low similarity -> 0.8x RAG penalty applied."""
        svc = self._make_service("Not sure what you mean")
        ctx = self._make_context("Have you seen the new exhibit at the museum?")
        classification = self._make_classification(pressure="medium")
        # Low similarity triggers the < 0.5 penalty
        search_results = self._make_search_results(count=2, similarity=0.35)

        result = svc.generate_reply(ctx, classification, search_results=search_results)

        # MEDIUM baseline = 0.65; similarity 0.35 < 0.5 -> RAG penalty applied
        # Other factors (diversity boost, etc.) may shift slightly
        assert result.confidence < 0.65, (
            f"Expected penalty to lower confidence below 0.65, got {result.confidence}"
        )
        assert result.confidence > 0.3, "Confidence shouldn't drop below 0.3"
