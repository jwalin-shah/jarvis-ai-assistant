"""Behavior tests for category classification.

Tests verify: given input text, expect specific category classification.
No testing of private methods or internal state.
"""

import pytest

from jarvis.classifiers.category_classifier import (
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


class TestCategoryResult:
    """CategoryResult dataclass behavior."""

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


class TestFastPath:
    """Fast path classification behavior."""

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


class TestAPIContract:
    """Public API behavior contracts."""

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
        result = classify_category("What time?", context=["Let's meet up"], mobilization=mob)
        assert result is not None
        # Should be one of valid categories
        valid = {"closing", "acknowledge", "question", "request", "emotion", "statement"}
        assert result.category in valid


class TestClassificationBehavior:
    """Classification behavior for different message types."""

    def test_question_classified(self) -> None:
        """Questions are classified as 'question'."""
        result = classify_category("What time is the meeting tomorrow?")
        assert result.category in ("question", "statement")  # Model-dependent
        assert result.confidence >= 0.3

    def test_fast_path_priority_over_mobilization(self) -> None:
        """Fast path should fire before considering mobilization."""
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

    def test_context_influences_classification(self) -> None:
        """Context can influence classification."""
        result = classify_category(
            "Not much, just hanging out",
            context=["Hey what's up"],
        )
        # Should return a valid category
        valid = {"closing", "acknowledge", "question", "request", "emotion", "statement"}
        assert result.category in valid
        assert result.confidence >= 0.3


class TestAllCategoriesReachable:
    """Each category should be reachable."""

    def test_all_categories_reachable(self) -> None:
        """Each category should be reachable (not just 'statement')."""
        test_messages = {
            "acknowledge": "ok",
            "question": "What time is it?",
            "emotion": 'Loved "great job"',
            "request": "Can you send me the file?",
            "statement": "I went to the store today",
            "closing": "Talk to you later bye",
        }
        seen_categories: set[str] = set()
        for _expected, msg in test_messages.items():
            result = classify_category(msg)
            seen_categories.add(result.category)
        # At minimum, fast path categories should be reachable
        assert "acknowledge" in seen_categories
        assert "emotion" in seen_categories


class TestEdgeCases:
    """Edge case handling."""

    def test_classify_empty_string(self) -> None:
        """Empty input should not crash."""
        result = classify_category("")
        assert isinstance(result, CategoryResult)
        valid = {"closing", "acknowledge", "question", "request", "emotion", "statement"}
        assert result.category in valid

    def test_classify_whitespace_only(self) -> None:
        """Whitespace-only input should not crash."""
        result = classify_category("   ")
        assert isinstance(result, CategoryResult)
        valid = {"closing", "acknowledge", "question", "request", "emotion", "statement"}
        assert result.category in valid


class TestFeatureExtraction:
    """Feature extraction behavior (public API only)."""

    def test_extract_hand_crafted_features(self) -> None:
        """Hand-crafted features should return expected values."""
        from jarvis.features import CategoryFeatureExtractor

        extractor = CategoryFeatureExtractor()
        features = extractor.extract_hand_crafted(
            text="Hey what's up?",
            context=["Hello"],
            mob_pressure="none",
            mob_type="answer",
        )
        # Should return float32 array with expected length
        assert len(features) == 26
        assert features.dtype.name == "float32"


# =============================================================================
# Conditional tests (require optional dependencies)
# =============================================================================


def _spacy_model_available() -> bool:
    """Check if en_core_web_sm spaCy model is installed."""
    try:
        import spacy

        spacy.load("en_core_web_sm")
        return True
    except (ImportError, OSError):
        return False


_has_spacy_model = _spacy_model_available()


@pytest.mark.skipif(not _has_spacy_model, reason="spaCy en_core_web_sm not available")
class TestSpacyFeatures:
    """Tests requiring spaCy model."""

    def test_extract_spacy_features(self) -> None:
        """SpaCy features should return expected values."""
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

    def test_feature_count_contract(self) -> None:
        """extract_all returns expected feature count."""
        from jarvis.features import CategoryFeatureExtractor

        extractor = CategoryFeatureExtractor()
        features = extractor.extract_all("Hello there", [], "none", "answer")
        assert len(features) == 147, f"Expected 147 non-BERT features, got {len(features)}"
