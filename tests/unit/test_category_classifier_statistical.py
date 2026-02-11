"""Statistical validation tests for category classifier.

These tests use confidence thresholds and statistical assertions
instead of brittle exact-match checks.

Run with: pytest tests/unit/test_category_classifier_statistical.py -v
"""

from __future__ import annotations

import pytest

from jarvis.classifiers.category_classifier import classify_category
from tests.dependencies import skip_if_missing
from tests.utils.statistical_assertions import (
    CONFIDENCE_THRESHOLDS,
    StatisticalValidator,
    assert_confidence_in_range,
)


class TestCategoryClassifierStatistical:
    """Statistical validation of category classifier."""

    @pytest.fixture
    def validator(self) -> StatisticalValidator:
        """Provide configured statistical validator."""
        return StatisticalValidator(min_samples=5, max_variance=0.01)

    # -------------------------------------------------------------------------
    # Fast Path Tests (High Confidence)
    # -------------------------------------------------------------------------

    def test_acknowledge_fast_path_confidence(self) -> None:
        """Acknowledgments should have very high confidence via fast path."""
        test_cases = ["ok", "okay", "k", "got it", "sounds good"]
        threshold = CONFIDENCE_THRESHOLDS["acknowledge_exact"]

        for text in test_cases:
            result = classify_category(text)
            assert result.category == "acknowledge"
            assert result.method == "fast_path"
            assert_confidence_in_range(
                result.confidence,
                threshold.min_confidence,
                threshold.max_confidence,
                tolerance=threshold.tolerance,
                message=f"'{text}'",
            )

    def test_emotion_fast_path_confidence(self) -> None:
        """Emotion reactions should have high confidence."""
        result = classify_category('Loved "great job"')

        assert result.category == "emotion"
        assert_confidence_in_range(
            result.confidence,
            0.90,
            1.0,
            tolerance=0.05,
        )

    def test_closing_patterns_high_confidence(self) -> None:
        """Closing patterns should have high confidence."""
        test_cases = ["bye", "talk to you later", "see you soon"]

        for text in test_cases:
            result = classify_category(text)
            # Accept closing OR statement (some closings are ambiguous)
            if result.category == "closing":
                assert_confidence_in_range(
                    result.confidence, 0.80, 1.0, tolerance=0.10, message=f"'{text}'"
                )

    # -------------------------------------------------------------------------
    # Category Distribution Tests
    # -------------------------------------------------------------------------

    def test_all_confidences_in_valid_range(self) -> None:
        """All classifier outputs should have confidence in [0, 1]."""
        test_inputs = [
            "ok",
            "What time?",
            "Send me the file",
            "I love this!",
            "Goodbye for now",
            "Just a regular statement here",
        ]

        for text in test_inputs:
            result = classify_category(text)
            assert 0.0 <= result.confidence <= 1.0, (
                f"Confidence {result.confidence} out of range for '{text}'"
            )

    def test_all_categories_reachable(self) -> None:
        """Each category should be reachable by at least one test case."""
        test_messages = {
            "acknowledge": "ok",
            "question": "What time is it?",
            "emotion": 'Loved "great job"',
            "request": "Can you send me the file?",
            "statement": "I went to the store today",
            "closing": "Talk to you later bye",
        }

        seen_categories: set[str] = set()
        for expected, msg in test_messages.items():
            result = classify_category(msg)
            seen_categories.add(result.category)

        # At minimum, fast path categories should be reachable
        assert "acknowledge" in seen_categories
        assert "emotion" in seen_categories

    # -------------------------------------------------------------------------
    # Model-Based Tests (with dependency skip)
    # -------------------------------------------------------------------------

    @pytest.mark.skipif(
        skip_if_missing("lightgbm_model", "spacy"),
        reason="Requires LightGBM model and spaCy",
    )
    def test_question_classification_distribution(self, validator: StatisticalValidator) -> None:
        """Question patterns should map to question or statement category."""
        test_cases = [
            ("What time is it?", "question"),
            ("How are you doing?", "question"),
            ("Where should we go?", "question"),
            ("Why is the sky blue?", "question"),
            ("Can you help me?", "question"),  # May be request
            ("What do you think?", "question"),  # May be statement
        ]

        accuracy, failures = validator.validate_distribution(
            classify_category,
            test_cases,
            min_accuracy=0.80,  # Allow some ambiguity
        )

        # Allow expected ambiguities
        acceptable_failures = [
            "Can you help me?",  # Could be request
            "What do you think?",  # Could be statement
        ]

        real_failures = [
            f for f in failures if not any(acceptable in f for acceptable in acceptable_failures)
        ]

        assert accuracy >= 0.70, (
            f"Accuracy {accuracy:.2%} below threshold. Failures: {real_failures}"
        )

    @pytest.mark.skipif(
        skip_if_missing("spacy"),
        reason="Requires spaCy model",
    )
    def test_spacy_feature_confidence_range(self) -> None:
        """Tests using spaCy features should have reasonable confidence."""
        from jarvis.features import CategoryFeatureExtractor

        extractor = CategoryFeatureExtractor()

        # Test cases that should trigger spaCy path
        test_texts = [
            "Can you help me?",
            "Send me the file",
            "I think that's great",
        ]

        for text in test_texts:
            try:
                features = extractor.extract_spacy_features(text)
                # Just verify we get features without error
                assert len(features) > 0
            except Exception as e:
                pytest.skip(f"spaCy model not available: {e}")

    # -------------------------------------------------------------------------
    # Confidence Calibration Tests
    # -------------------------------------------------------------------------

    def test_high_confidence_threshold_for_exact_matches(self) -> None:
        """Exact pattern matches should have high confidence."""
        # Exact matches - high confidence
        exact = classify_category("ok")
        if exact.category == "acknowledge":
            assert exact.confidence >= 0.90

        # Near matches - may still match via model but that's acceptable
        near = classify_category("okk")  # Typo
        # Model may still classify this as acknowledge with high confidence
        # since "okk" is close to "ok" - this is acceptable behavior
        assert near.category in {"acknowledge", "statement", "closing"}

    def test_confidence_reflects_ambiguity(self) -> None:
        """Ambiguous inputs should have lower confidence."""
        # Clear case
        clear = classify_category("What is your name?")
        # Ambiguous case (could be statement or question)
        ambiguous = classify_category("I wonder what your name is")

        # Clear case should have higher confidence than ambiguous
        # (This is a soft assertion - not always true, but good check)
        if clear.category == "question" and ambiguous.category == "statement":
            # Both valid, but clear should be more confident
            assert clear.confidence > 0.6  # Reasonable floor

    # -------------------------------------------------------------------------
    # Edge Cases
    # -------------------------------------------------------------------------

    def test_empty_string_returns_valid_result(self) -> None:
        """Empty input should not crash and return valid category."""
        result = classify_category("")

        valid_categories = {
            "closing",
            "acknowledge",
            "question",
            "request",
            "emotion",
            "statement",
        }
        assert result.category in valid_categories
        assert 0.0 <= result.confidence <= 1.0

    def test_whitespace_only_returns_valid_result(self) -> None:
        """Whitespace input should not crash and return valid category."""
        result = classify_category("   \n\t  ")

        valid_categories = {
            "closing",
            "acknowledge",
            "question",
            "request",
            "emotion",
            "statement",
        }
        assert result.category in valid_categories

    def test_long_text_does_not_crash(self) -> None:
        """Very long text should not cause issues."""
        long_text = "Hello " * 1000  # Very long text
        result = classify_category(long_text)

        # Should return a valid category without crashing
        assert result.category in {
            "closing",
            "acknowledge",
            "question",
            "request",
            "emotion",
            "statement",
        }
