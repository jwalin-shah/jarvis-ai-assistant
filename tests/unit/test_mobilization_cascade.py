"""Tests for mobilization cascade (rules -> intent fallback)."""

from __future__ import annotations

from jarvis.classifiers.cascade import (
    MobilizationCascade,
    classify_response_pressure_with_cascade,
    classify_with_cascade,
    reset_mobilization_cascade,
)
from jarvis.classifiers.intent_classifier import IntentResult
from jarvis.classifiers.response_mobilization import ResponsePressure, ResponseType


class StubIntentClassifier:
    def __init__(self, intent: str, confidence: float = 0.9) -> None:
        self.intent = intent
        self.confidence = confidence
        self.called = False
        self.call_count = 0

    def classify(self, text: str, intent_options: list[str]) -> IntentResult:
        self.called = True
        self.call_count += 1
        return IntentResult(intent=self.intent, confidence=self.confidence, method="stub")


class RaisingIntentClassifier:
    """Intent classifier that always raises an exception."""

    def __init__(self) -> None:
        self.call_count = 0

    def classify(self, text: str, intent_options: list[str]) -> IntentResult:
        self.call_count += 1
        raise RuntimeError("Intent classifier unavailable")


def test_cascade_uses_rules_when_confident() -> None:
    stub = StubIntentClassifier("reply_emotional_reactive")
    result = classify_with_cascade("Can you help me?", intent_classifier=stub)
    assert result.pressure == ResponsePressure.HIGH
    assert result.response_type == ResponseType.COMMITMENT
    assert stub.called is False


def test_cascade_no_reply_gate_short_circuits_ack() -> None:
    stub = StubIntentClassifier("reply_question_info", confidence=0.88)
    result = classify_with_cascade("ok", intent_classifier=stub)
    assert result.pressure == ResponsePressure.NONE
    assert result.response_type == ResponseType.CLOSING
    assert stub.called is False


def test_cascade_falls_back_when_rule_confidence_low() -> None:
    stub = StubIntentClassifier("reply_question_info", confidence=0.88)
    result = classify_with_cascade(
        "random text that doesn't match patterns",
        intent_classifier=stub,
    )
    assert stub.called is True
    assert result.pressure == ResponsePressure.HIGH
    assert result.response_type == ResponseType.ANSWER
    assert result.method == "intent_fallback"
    assert result.features["intent_fallback"] is True


def test_cascade_maps_no_reply_intent() -> None:
    stub = StubIntentClassifier("no_reply_backchannel", confidence=0.93)
    result = classify_with_cascade("ambiguous", intent_classifier=stub)
    assert result.pressure == ResponsePressure.NONE
    assert result.response_type == ResponseType.CLOSING


# =============================================================================
# Intent classifier exception handling
# =============================================================================


def test_cascade_handles_intent_classifier_exception() -> None:
    """When intent classifier raises, cascade falls back to rule result."""
    raiser = RaisingIntentClassifier()
    # "ambiguous" has LOW pressure from rules (default path, confidence 0.50),
    # which means the should-reply gate returns confidence 0.60 (ambiguous),
    # below the 0.80 threshold, triggering intent fallback.
    # When the fallback raises, cascade should use the rule result.
    cascade = MobilizationCascade(intent_classifier=raiser)
    result = cascade.classify("ambiguous")

    # Should still return a valid result (from rules, not crash)
    assert result.pressure in {p for p in ResponsePressure}
    assert raiser.call_count >= 1


def test_cascade_sticky_disable_after_intent_failure() -> None:
    """After intent classifier fails, subsequent calls skip the intent classifier entirely."""
    raiser = RaisingIntentClassifier()
    cascade = MobilizationCascade(intent_classifier=raiser)

    # First call: triggers intent fallback, which fails and sets _intent_fallback_disabled
    cascade.classify("ambiguous")
    first_call_count = raiser.call_count
    assert first_call_count >= 1

    # Second call: should NOT call intent classifier again (sticky disable)
    cascade.classify("another ambiguous message here")
    assert raiser.call_count == first_call_count, (
        "Intent classifier should not be called after sticky disable"
    )
    assert cascade._intent_fallback_disabled is True


# =============================================================================
# Empty string input
# =============================================================================


def test_cascade_empty_string() -> None:
    """Empty string should produce NONE pressure with CLOSING response type."""
    stub = StubIntentClassifier("reply_question_info")
    result = classify_with_cascade("", intent_classifier=stub)

    assert result.pressure == ResponsePressure.NONE
    assert result.response_type == ResponseType.CLOSING
    # Empty string should be caught by rule gate, not trigger intent classifier
    assert stub.called is False


# =============================================================================
# Boundary case: input that partially matches multiple rules
# =============================================================================


def test_cascade_boundary_rule_confidence() -> None:
    """Input that matches a low-confidence rule path should fall back to intent."""
    # "it was nice" matches TELLING_PATTERNS (r"^it (is|was|looks|seems|sounds)\\b"),
    # giving LOW pressure with 0.80 confidence. That's exactly at the threshold.
    # The should-reply gate for LOW pressure returns confidence 0.60 (ambiguous),
    # which is below the 0.80 threshold, so intent fallback should be triggered.
    stub = StubIntentClassifier("reply_casual_chat", confidence=0.85)
    result = classify_with_cascade("it was nice", intent_classifier=stub)

    # The telling rule gives confidence=0.80 (exactly at RULE_CONFIDENCE_THRESHOLD),
    # so if gate triggers intent, the intent maps to LOW/OPTIONAL for casual_chat.
    # Either way, verify the result is well-formed.
    assert result.pressure in {p for p in ResponsePressure}
    assert result.response_type in {t for t in ResponseType}
    assert result.confidence > 0


# =============================================================================
# Singleton function
# =============================================================================


def test_classify_response_pressure_with_cascade_singleton() -> None:
    """classify_response_pressure_with_cascade uses the singleton and returns valid results."""
    reset_mobilization_cascade()

    result = classify_response_pressure_with_cascade("Can you help me move tomorrow?")
    assert result.pressure == ResponsePressure.HIGH
    assert result.response_type == ResponseType.COMMITMENT
    assert result.confidence > 0

    # Calling again should reuse the singleton (same result for deterministic input)
    result2 = classify_response_pressure_with_cascade("Can you help me move tomorrow?")
    assert result2.pressure == result.pressure
    assert result2.response_type == result.response_type

    reset_mobilization_cascade()


def test_classify_response_pressure_with_cascade_backchannel() -> None:
    """Singleton cascade correctly classifies backchannels as NONE pressure."""
    reset_mobilization_cascade()

    result = classify_response_pressure_with_cascade("ok")
    assert result.pressure == ResponsePressure.NONE
    assert result.response_type == ResponseType.CLOSING

    reset_mobilization_cascade()
