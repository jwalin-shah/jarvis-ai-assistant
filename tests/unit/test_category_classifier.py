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
        result = CategoryResult("request", 0.87, "lightgbm")
        assert "request" in repr(result)
        assert "0.87" in repr(result)
        assert "lightgbm" in repr(result)

    def test_fields(self) -> None:
        result = CategoryResult("emotion", 0.90, "fast_path")
        assert result.category == "emotion"
        assert result.confidence == 0.90
        assert result.method == "fast_path"


# =============================================================================
# Fast path (Layer 0)
# =============================================================================


class TestFastPath:
    def test_reaction_tapback(self) -> None:
        """iMessage reactions categorized by intent: Loved = emotion."""
        result = classify_category('Loved "Hey there"')
        assert result.category == "emotion"
        assert result.method == "fast_path"
        assert result.confidence == 1.0

    def test_acknowledgment_ok(self) -> None:
        result = classify_category("ok")
        assert result.category == "acknowledge"
        assert result.method == "fast_path"

    def test_acknowledgment_got_it(self) -> None:
        result = classify_category("got it")
        assert result.category == "acknowledge"
        assert result.method == "fast_path"

    def test_acknowledgment_sounds_good(self) -> None:
        result = classify_category("sounds good")
        assert result.category == "acknowledge"
        assert result.method == "fast_path"

    def test_normal_message_no_fast_path(self) -> None:
        """Normal messages should not match fast path."""
        result = classify_category("Want to grab lunch tomorrow?")
        assert result.method != "fast_path"


# =============================================================================
# Fallback without trained model (default)
# =============================================================================


class TestWithLightGBMModel:
    """Tests with LightGBM model (previously TestFallbackNoModel)."""

    def test_classifies_question(self) -> None:
        """With LightGBM model, questions are classified correctly."""
        result = classify_category("Hey how have you been?")
        assert result.category == "question"
        assert result.method == "lightgbm"
        assert result.confidence > 0.5

    def test_classifies_with_context(self) -> None:
        """Model classifies with context (context embedding zeroed internally)."""
        result = classify_category(
            "Not much, just hanging out",
            context=["Hey what's up"],
        )
        # Should classify as statement
        assert result.method == "lightgbm"
        assert result.confidence > 0.3


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
        # Should be one of valid categories
        assert result.category in {"closing", "acknowledge", "question", "request", "emotion", "statement"}


# =============================================================================
# Classifier class internals
# =============================================================================


class TestCategoryClassifier:
    def test_classifier_init(self) -> None:
        clf = CategoryClassifier()
        assert clf._pipeline is None
        assert clf._pipeline_loaded is False

    def test_load_pipeline_attempts_load(self) -> None:
        """Loading pipeline should update the loaded flag."""
        clf = CategoryClassifier()
        result = clf._load_pipeline()
        # Either loads successfully (True) or fails gracefully (False)
        assert isinstance(result, bool)
        assert clf._pipeline_loaded is True  # Attempted

    def test_fast_path_takes_priority(self) -> None:
        """Fast path should fire before SVM."""
        mob = MobilizationResult(
            pressure=ResponsePressure.HIGH,
            response_type=ResponseType.ANSWER,
            confidence=0.90,
            features={},
        )
        # "ok" matches fast path acknowledge, even with mobilization
        result = classify_category("ok", mobilization=mob)
        assert result.category == "acknowledge"
        assert result.method == "fast_path"


# =============================================================================
# Feature extraction
# =============================================================================


class TestFeatureExtraction:
    def test_extract_hand_crafted_features(self) -> None:
        """Hand-crafted features should return 26 values."""
        from jarvis.features import CategoryFeatureExtractor

        extractor = CategoryFeatureExtractor()
        features = extractor.extract_hand_crafted(
            text="Hey what's up?",
            context=["Hello"],
            mob_pressure="none",
            mob_type="answer",
        )
        assert len(features) == 26
        assert features.dtype.name == "float32"

    def test_extract_spacy_features(self) -> None:
        """SpaCy features should return 94 values (14 original + 80 new)."""
        from jarvis.features import CategoryFeatureExtractor

        extractor = CategoryFeatureExtractor()
        features = extractor.extract_spacy_features("Can you help me?")
        assert len(features) == 94
        assert features.dtype.name == "float32"

    def test_spacy_imperative_detection(self) -> None:
        from jarvis.features import CategoryFeatureExtractor

        extractor = CategoryFeatureExtractor()
        # "Send" is VB at start -> imperative
        features = extractor.extract_spacy_features("Send me the file")
        assert features[0] == 1.0  # has_imperative

    def test_spacy_you_modal_detection(self) -> None:
        from jarvis.features import CategoryFeatureExtractor

        extractor = CategoryFeatureExtractor()
        features = extractor.extract_spacy_features("Can you help?")
        assert features[1] == 1.0  # you_modal

    def test_spacy_agreement_detection(self) -> None:
        from jarvis.features import CategoryFeatureExtractor

        extractor = CategoryFeatureExtractor()
        features = extractor.extract_spacy_features("sure thing")
        assert features[8] == 1.0  # has_agreement
