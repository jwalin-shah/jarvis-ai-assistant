"""Tests for jarvis.response_classifier module."""

from __future__ import annotations

import pytest

from jarvis.response_classifier import (
    COMMITMENT_RESPONSE_TYPES,
    STRUCTURAL_PATTERNS,
    TRIGGER_TO_VALID_RESPONSES,
    ClassificationResult,
    HybridResponseClassifier,
    ResponseType,
    get_response_classifier,
    reset_response_classifier,
)


class TestResponseType:
    """Tests for ResponseType enum."""

    def test_enum_values(self):
        """ResponseType has expected values."""
        assert ResponseType.AGREE.value == "AGREE"
        assert ResponseType.DECLINE.value == "DECLINE"
        assert ResponseType.DEFER.value == "DEFER"
        assert ResponseType.ACKNOWLEDGE.value == "ACKNOWLEDGE"
        assert ResponseType.ANSWER.value == "ANSWER"
        assert ResponseType.QUESTION.value == "QUESTION"
        assert ResponseType.REACT_POSITIVE.value == "REACT_POSITIVE"
        assert ResponseType.REACT_SYMPATHY.value == "REACT_SYMPATHY"
        assert ResponseType.STATEMENT.value == "STATEMENT"
        assert ResponseType.GREETING.value == "GREETING"

    def test_string_enum(self):
        """ResponseType is a string enum."""
        assert isinstance(ResponseType.AGREE, str)
        assert ResponseType.AGREE == "AGREE"


class TestCommitmentResponseTypes:
    """Tests for COMMITMENT_RESPONSE_TYPES constant."""

    def test_contains_agree_decline_defer(self):
        """Commitment types include AGREE, DECLINE, DEFER."""
        assert ResponseType.AGREE in COMMITMENT_RESPONSE_TYPES
        assert ResponseType.DECLINE in COMMITMENT_RESPONSE_TYPES
        assert ResponseType.DEFER in COMMITMENT_RESPONSE_TYPES

    def test_is_frozenset(self):
        """COMMITMENT_RESPONSE_TYPES is immutable."""
        assert isinstance(COMMITMENT_RESPONSE_TYPES, frozenset)

    def test_exactly_three_types(self):
        """Only three commitment response types."""
        assert len(COMMITMENT_RESPONSE_TYPES) == 3


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_basic_creation(self):
        """Can create ClassificationResult with required fields."""
        result = ClassificationResult(
            label=ResponseType.AGREE,
            confidence=0.95,
            method="structural",
        )
        assert result.label == ResponseType.AGREE
        assert result.confidence == 0.95
        assert result.method == "structural"
        assert result.structural_match is False  # default
        assert result.da_label is None
        assert result.da_confidence is None

    def test_with_optional_fields(self):
        """Can create with all optional fields."""
        result = ClassificationResult(
            label=ResponseType.DECLINE,
            confidence=0.85,
            method="svm",
            structural_match=True,
            da_label="DECLINE",
            da_confidence=0.80,
        )
        assert result.structural_match is True
        assert result.da_label == "DECLINE"
        assert result.da_confidence == 0.80


class TestStructuralPatterns:
    """Tests for structural pattern matching."""

    @pytest.mark.parametrize(
        "text,expected_type",
        [
            # AGREE patterns
            ("yes", ResponseType.AGREE),
            ("yeah", ResponseType.AGREE),
            ("yep", ResponseType.AGREE),
            ("sure", ResponseType.AGREE),
            ("definitely", ResponseType.AGREE),
            ("i'm down", ResponseType.AGREE),
            ("sounds good", ResponseType.AGREE),
            ("count me in", ResponseType.AGREE),
            ("for sure", ResponseType.AGREE),
            ("bet", ResponseType.AGREE),
            # DECLINE patterns
            ("no", ResponseType.DECLINE),
            ("nope", ResponseType.DECLINE),
            ("nah", ResponseType.DECLINE),
            ("can't", ResponseType.DECLINE),
            ("i can't", ResponseType.DECLINE),
            ("sorry i can't", ResponseType.DECLINE),
            ("i'll pass", ResponseType.DECLINE),
            ("not today", ResponseType.DECLINE),
            # DEFER patterns
            ("maybe", ResponseType.DEFER),
            ("let me check", ResponseType.DEFER),
            ("i'll see", ResponseType.DEFER),
            ("not sure", ResponseType.DEFER),
            ("depends", ResponseType.DEFER),
            ("we'll see", ResponseType.DEFER),
            # ACKNOWLEDGE patterns
            ("ok", ResponseType.ACKNOWLEDGE),
            ("okay", ResponseType.ACKNOWLEDGE),
            ("got it", ResponseType.ACKNOWLEDGE),
            ("alright", ResponseType.ACKNOWLEDGE),
            ("cool", ResponseType.ACKNOWLEDGE),
            ("noted", ResponseType.ACKNOWLEDGE),
            # QUESTION patterns
            ("what?", ResponseType.QUESTION),
            ("when is it?", ResponseType.QUESTION),
            ("where?", ResponseType.QUESTION),
            ("huh?", ResponseType.QUESTION),
            # REACT_POSITIVE patterns
            ("congrats!", ResponseType.REACT_POSITIVE),
            ("that's awesome", ResponseType.REACT_POSITIVE),
            ("lol", ResponseType.REACT_POSITIVE),
            ("haha", ResponseType.REACT_POSITIVE),
            ("yay!", ResponseType.REACT_POSITIVE),
            # REACT_SYMPATHY patterns
            ("i'm sorry", ResponseType.REACT_SYMPATHY),
            ("that sucks", ResponseType.REACT_SYMPATHY),
            ("oh no", ResponseType.REACT_SYMPATHY),
            # GREETING patterns
            ("hey", ResponseType.GREETING),
            ("hi", ResponseType.GREETING),
            ("hello", ResponseType.GREETING),
            ("good morning", ResponseType.GREETING),
        ],
    )
    def test_structural_patterns_match(self, text: str, expected_type: ResponseType):
        """Structural patterns match expected response types."""
        classifier = HybridResponseClassifier(
            use_centroid_verification=False,
            use_svm=False,
        )
        result = classifier.classify(text)

        assert result.label == expected_type, (
            f"Expected {expected_type.value} for '{text}', got {result.label.value}"
        )


class TestTapbackDetection:
    """Tests for iMessage tapback detection."""

    @pytest.mark.parametrize(
        "text,expected_type",
        [
            ('Liked "great job"', ResponseType.REACT_POSITIVE),
            ('Loved "thanks!"', ResponseType.REACT_POSITIVE),
            ("Laughed at an image", ResponseType.REACT_POSITIVE),
        ],
    )
    def test_positive_tapbacks(self, text: str, expected_type: ResponseType):
        """Positive tapbacks classified as REACT_POSITIVE."""
        classifier = HybridResponseClassifier(
            use_centroid_verification=False,
            use_svm=False,
        )
        result = classifier.classify(text)

        assert result.label == expected_type
        assert result.method == "tapback_positive"

    @pytest.mark.parametrize(
        "text",
        [
            'Disliked "that message"',
            'Emphasized "important"',
            'Questioned "really?"',
        ],
    )
    def test_filtered_tapbacks(self, text: str):
        """Filtered tapbacks return ANSWER with low confidence."""
        classifier = HybridResponseClassifier(
            use_centroid_verification=False,
            use_svm=False,
        )
        result = classifier.classify(text)

        assert result.label == ResponseType.ANSWER
        assert result.method == "tapback_filtered"
        assert result.confidence == 0.3


class TestHybridResponseClassifier:
    """Tests for HybridResponseClassifier."""

    def test_empty_input(self):
        """Empty input returns STATEMENT with zero confidence."""
        classifier = HybridResponseClassifier(
            use_centroid_verification=False,
            use_svm=False,
        )

        result = classifier.classify("")
        assert result.label == ResponseType.STATEMENT
        assert result.confidence == 0.0
        assert result.method == "empty"

    def test_whitespace_only(self):
        """Whitespace-only input treated as empty."""
        classifier = HybridResponseClassifier(
            use_centroid_verification=False,
            use_svm=False,
        )

        result = classifier.classify("   \t\n  ")
        assert result.label == ResponseType.STATEMENT
        assert result.method == "empty"

    def test_structural_match_high_confidence(self):
        """Structural matches have high confidence."""
        classifier = HybridResponseClassifier(
            use_centroid_verification=False,
            use_svm=False,
        )

        result = classifier.classify("yes!")
        assert result.label == ResponseType.AGREE
        assert result.confidence >= 0.9
        assert result.structural_match is True

    def test_is_commitment_response_method(self):
        """is_commitment_response correctly identifies commitment types."""
        classifier = HybridResponseClassifier(
            use_centroid_verification=False,
            use_svm=False,
        )

        agree_result = classifier.classify("yes")
        decline_result = classifier.classify("no")
        defer_result = classifier.classify("maybe")
        question_result = classifier.classify("what?")

        assert classifier.is_commitment_response(agree_result) is True
        assert classifier.is_commitment_response(decline_result) is True
        assert classifier.is_commitment_response(defer_result) is True
        assert classifier.is_commitment_response(question_result) is False


class TestClassifyBatch:
    """Tests for batch classification.

    Note: Full batch tests require embeddings. We test the fast-path cases
    that don't require embeddings (empty list, tapbacks).
    """

    def test_empty_list(self):
        """Empty list returns empty results."""
        classifier = HybridResponseClassifier(
            use_centroid_verification=False,
            use_svm=False,
        )
        results = classifier.classify_batch([])
        assert results == []

    def test_all_empty_strings(self):
        """Batch of empty strings uses fast path (no embedding needed)."""
        classifier = HybridResponseClassifier(
            use_centroid_verification=False,
            use_svm=False,
        )
        texts = ["", "   ", "\t\n"]
        results = classifier.classify_batch(texts)

        assert len(results) == 3
        # All empty -> STATEMENT
        for r in results:
            assert r.label == ResponseType.STATEMENT

    def test_all_tapbacks(self):
        """Batch of tapbacks uses fast path (no embedding needed)."""
        classifier = HybridResponseClassifier(
            use_centroid_verification=False,
            use_svm=False,
        )
        texts = ['Liked "message"', 'Loved "great"', 'Disliked "bad"']
        results = classifier.classify_batch(texts)

        assert len(results) == 3
        assert results[0].label == ResponseType.REACT_POSITIVE
        assert results[1].label == ResponseType.REACT_POSITIVE
        assert results[2].label == ResponseType.ANSWER  # filtered tapback


class TestSingletonFactory:
    """Tests for singleton factory functions."""

    def test_get_response_classifier_returns_singleton(self):
        """get_response_classifier returns same instance."""
        reset_response_classifier()

        classifier1 = get_response_classifier()
        classifier2 = get_response_classifier()

        assert classifier1 is classifier2

    def test_reset_response_classifier_clears_singleton(self):
        """reset_response_classifier clears the singleton."""
        classifier1 = get_response_classifier()
        reset_response_classifier()
        classifier2 = get_response_classifier()

        assert classifier1 is not classifier2


class TestTriggerToValidResponses:
    """Tests for trigger to valid response mappings."""

    def test_commitment_trigger_allows_commitment_responses(self):
        """Commitment triggers allow AGREE/DECLINE/DEFER."""
        commitment_responses = TRIGGER_TO_VALID_RESPONSES.get("commitment", [])
        assert ResponseType.AGREE in commitment_responses
        assert ResponseType.DECLINE in commitment_responses
        assert ResponseType.DEFER in commitment_responses

    def test_question_trigger_allows_answer(self):
        """Question triggers allow ANSWER responses."""
        question_responses = TRIGGER_TO_VALID_RESPONSES.get("question", [])
        assert ResponseType.ANSWER in question_responses

    def test_reaction_trigger_allows_react_types(self):
        """Reaction triggers allow REACT_* responses."""
        reaction_responses = TRIGGER_TO_VALID_RESPONSES.get("reaction", [])
        assert (
            ResponseType.REACT_POSITIVE in reaction_responses
            or ResponseType.REACT_SYMPATHY in reaction_responses
        )

    def test_social_trigger_allows_greeting(self):
        """Social triggers allow GREETING responses."""
        social_responses = TRIGGER_TO_VALID_RESPONSES.get("social", [])
        assert (
            ResponseType.GREETING in social_responses
            or ResponseType.ACKNOWLEDGE in social_responses
        )


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_case_insensitive_matching(self):
        """Patterns match case-insensitively."""
        classifier = HybridResponseClassifier(
            use_centroid_verification=False,
            use_svm=False,
        )

        lower = classifier.classify("yes")
        upper = classifier.classify("YES")
        mixed = classifier.classify("YeS")

        assert lower.label == upper.label == mixed.label == ResponseType.AGREE

    def test_unicode_handling(self):
        """Unicode characters don't cause errors."""
        classifier = HybridResponseClassifier(
            use_centroid_verification=False,
            use_svm=False,
        )

        # Should not raise
        result = classifier.classify("yes 👍")
        assert result is not None

        result = classifier.classify("café sounds good")
        assert result is not None

    def test_very_long_input(self):
        """Very long input doesn't cause issues."""
        classifier = HybridResponseClassifier(
            use_centroid_verification=False,
            use_svm=False,
        )
        long_text = "this is a test " * 100

        result = classifier.classify(long_text)
        assert result is not None

    def test_special_characters(self):
        """Special characters are handled."""
        classifier = HybridResponseClassifier(
            use_centroid_verification=False,
            use_svm=False,
        )

        result = classifier.classify("yes!!!")
        assert result.label == ResponseType.AGREE

        result = classifier.classify("no...")
        assert result.label == ResponseType.DECLINE


class TestStructuralPatternsCompiled:
    """Tests for compiled structural patterns."""

    def test_patterns_defined_for_main_types(self):
        """Main response types have structural patterns."""
        types_with_patterns = [
            ResponseType.AGREE,
            ResponseType.DECLINE,
            ResponseType.DEFER,
            ResponseType.ACKNOWLEDGE,
            ResponseType.QUESTION,
            ResponseType.REACT_POSITIVE,
            ResponseType.REACT_SYMPATHY,
            ResponseType.GREETING,
        ]

        for response_type in types_with_patterns:
            assert response_type in STRUCTURAL_PATTERNS
            assert len(STRUCTURAL_PATTERNS[response_type]) > 0

    def test_answer_statement_have_no_patterns(self):
        """ANSWER and STATEMENT don't have structural patterns (catch-all types)."""
        # These are handled by DA classifier fallback
        assert ResponseType.ANSWER not in STRUCTURAL_PATTERNS
        assert ResponseType.STATEMENT not in STRUCTURAL_PATTERNS
