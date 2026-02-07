"""Tests for jarvis/classifiers/category_classifier.py - Category classification."""

import pytest

from jarvis.classifiers.category_classifier import (
    CategoryClassifier,
    CategoryResult,
    classify_category,
    reset_category_classifier,
)
from jarvis.classifiers.response_mobilization import (
    MobilizationResult,
    ResponsePressure,
    ResponseType,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset classifier singleton between tests."""
    reset_category_classifier()
    yield
    reset_category_classifier()


# =============================================================================
# CategoryResult
# =============================================================================


class TestCategoryResult:
    def test_repr(self) -> None:
        result = CategoryResult("brief", 0.87, "svm")
        assert "brief" in repr(result)
        assert "0.87" in repr(result)
        assert "svm" in repr(result)

    def test_fields(self) -> None:
        result = CategoryResult("warm", 0.90, "structural")
        assert result.category == "warm"
        assert result.confidence == 0.90
        assert result.method == "structural"


# =============================================================================
# Structural patterns (Layer 1)
# =============================================================================


class TestStructuralPatterns:
    def test_bare_question_mark(self) -> None:
        result = classify_category("?")
        assert result.category == "clarify"
        assert result.method == "structural"

    def test_bare_exclamation(self) -> None:
        result = classify_category("!")
        assert result.category == "clarify"
        assert result.method == "structural"

    def test_bare_ellipsis(self) -> None:
        result = classify_category("...")
        assert result.category == "clarify"
        assert result.method == "structural"

    def test_empty_string(self) -> None:
        result = classify_category("")
        assert result.category == "clarify"
        assert result.method == "structural"

    def test_whitespace_only(self) -> None:
        result = classify_category("   ")
        assert result.category == "clarify"
        assert result.method == "structural"

    def test_professional_keywords_no_structural(self) -> None:
        """Professional keywords no longer match structural patterns (handled by detect_tone)."""
        result = classify_category("Regarding the quarterly report")
        assert result.method != "structural"

    def test_emotional_pattern(self) -> None:
        result = classify_category("I'm so depressed right now")
        assert result.category == "warm"
        assert result.method == "structural"

    def test_emotional_breakup(self) -> None:
        result = classify_category("I just broke up with my girlfriend")
        assert result.category == "warm"
        assert result.method == "structural"

    def test_emotional_grief(self) -> None:
        result = classify_category("My grandma passed away last night")
        assert result.category == "warm"
        assert result.method == "structural"

    def test_normal_message_no_structural(self) -> None:
        """Normal messages should not match structural patterns."""
        result = classify_category("Want to grab lunch tomorrow?")
        assert result.method != "structural"


# =============================================================================
# Mobilization mapping (Layer 2)
# =============================================================================


class TestMobilizationMapping:
    def test_high_commitment_maps_to_brief(self) -> None:
        mob = MobilizationResult(
            pressure=ResponsePressure.HIGH,
            response_type=ResponseType.COMMITMENT,
            confidence=0.90,
            features={},
        )
        result = classify_category("Can you pick me up?", mobilization=mob)
        assert result.category == "brief"
        assert result.method == "mobilization"

    def test_high_answer_maps_to_brief(self) -> None:
        mob = MobilizationResult(
            pressure=ResponsePressure.HIGH,
            response_type=ResponseType.ANSWER,
            confidence=0.90,
            features={},
        )
        result = classify_category("What time is the meeting?", mobilization=mob)
        assert result.category == "brief"
        assert result.method == "mobilization"

    def test_medium_emotional_maps_to_warm(self) -> None:
        mob = MobilizationResult(
            pressure=ResponsePressure.MEDIUM,
            response_type=ResponseType.EMOTIONAL,
            confidence=0.85,
            features={},
        )
        result = classify_category("I got the promotion!!", mobilization=mob)
        assert result.category == "warm"
        assert result.method == "mobilization"

    def test_none_closing_maps_to_clarify(self) -> None:
        mob = MobilizationResult(
            pressure=ResponsePressure.NONE,
            response_type=ResponseType.CLOSING,
            confidence=0.95,
            features={},
        )
        result = classify_category("ok", mobilization=mob)
        assert result.category == "clarify"
        assert result.method == "mobilization"

    def test_low_optional_falls_through(self) -> None:
        """LOW/OPTIONAL mobilization should NOT route via mobilization."""
        mob = MobilizationResult(
            pressure=ResponsePressure.LOW,
            response_type=ResponseType.OPTIONAL,
            confidence=0.80,
            features={},
        )
        result = classify_category("I went to the store", mobilization=mob)
        # Should fall through to SVM or default, not mobilization
        assert result.method != "mobilization"


# =============================================================================
# Fallback without trained model (Layer 5)
# =============================================================================


class TestFallbackNoModel:
    def test_default_social(self) -> None:
        """Without SVM model, non-matching messages should default to social."""
        result = classify_category("Hey how have you been?")
        assert result.category == "social"
        assert result.method == "default"

    def test_default_with_context(self) -> None:
        result = classify_category(
            "Not much, just hanging out",
            context=["Hey what's up"],
        )
        assert result.category == "social"
        assert result.method == "default"


# =============================================================================
# API contract
# =============================================================================


class TestAPIContract:
    def test_classify_category_returns_category_result(self) -> None:
        result = classify_category("hello")
        assert isinstance(result, CategoryResult)

    def test_result_has_valid_category(self) -> None:
        from jarvis.classifiers.category_classifier import VALID_CATEGORIES

        result = classify_category("test message")
        assert result.category in VALID_CATEGORIES

    def test_confidence_in_range(self) -> None:
        result = classify_category("test")
        assert 0.0 <= result.confidence <= 1.0

    def test_method_is_string(self) -> None:
        result = classify_category("test")
        assert isinstance(result.method, str)

    def test_context_param_optional(self) -> None:
        # Should work without context
        result = classify_category("hello there")
        assert result is not None

    def test_mobilization_param_optional(self) -> None:
        # Should work without mobilization
        result = classify_category("hello there", context=["hi"])
        assert result is not None

    def test_all_params(self) -> None:
        mob = MobilizationResult(
            pressure=ResponsePressure.HIGH,
            response_type=ResponseType.ANSWER,
            confidence=0.90,
            features={},
        )
        result = classify_category(
            "What time?", context=["Let's meet up"], mobilization=mob
        )
        assert result is not None
        assert result.category in {"brief", "clarify", "social", "warm"}


# =============================================================================
# Classifier class internals
# =============================================================================


class TestCategoryClassifier:
    def test_classifier_init(self) -> None:
        clf = CategoryClassifier()
        assert clf._svm_model is None
        assert clf._svm_loaded is False

    def test_load_svm_returns_false_when_no_model(self) -> None:
        clf = CategoryClassifier()
        assert clf._load_svm() is False
        assert clf._svm_loaded is True  # Attempted

    def test_structural_takes_priority_over_mobilization(self) -> None:
        """Structural patterns should fire before mobilization."""
        mob = MobilizationResult(
            pressure=ResponsePressure.HIGH,
            response_type=ResponseType.ANSWER,
            confidence=0.90,
            features={},
        )
        # "?" matches structural clarify, even though mob says brief
        result = classify_category("?", mobilization=mob)
        assert result.category == "clarify"
        assert result.method == "structural"
