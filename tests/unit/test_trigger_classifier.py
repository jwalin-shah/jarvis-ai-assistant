"""Tests for jarvis.trigger_classifier module."""

from __future__ import annotations

import pytest

from jarvis.classifiers.trigger_classifier import (
    TRIGGER_TO_RESPONSE_TYPES,
    HybridTriggerClassifier,
    TriggerClassification,
    TriggerType,
    classify_trigger,
    get_trigger_classifier,
    reset_trigger_classifier,
)
from jarvis.config import get_config


class TestTriggerType:
    """Tests for TriggerType enum."""

    def test_enum_values(self):
        """TriggerType has expected values."""
        assert TriggerType.COMMITMENT.value == "commitment"
        assert TriggerType.QUESTION.value == "question"
        assert TriggerType.REACTION.value == "reaction"
        assert TriggerType.SOCIAL.value == "social"
        assert TriggerType.STATEMENT.value == "statement"
        assert TriggerType.UNKNOWN.value == "unknown"

    def test_string_enum(self):
        """TriggerType is a string enum."""
        assert isinstance(TriggerType.COMMITMENT, str)
        assert TriggerType.COMMITMENT == "commitment"

    def test_all_types_have_response_mappings(self):
        """Every TriggerType has valid response types mapped."""
        for trigger_type in TriggerType:
            assert trigger_type in TRIGGER_TO_RESPONSE_TYPES


class TestTriggerClassification:
    """Tests for TriggerClassification dataclass."""

    def test_basic_creation(self):
        """Can create TriggerClassification with required fields."""
        result = TriggerClassification(
            trigger_type=TriggerType.COMMITMENT,
            confidence=0.95,
            method="structural",
            valid_response_types=["AGREE", "DECLINE"],
        )
        assert result.trigger_type == TriggerType.COMMITMENT
        assert result.confidence == 0.95
        assert result.method == "structural"
        assert result.valid_response_types == ["AGREE", "DECLINE"]

    def test_is_commitment_property(self):
        """is_commitment property works correctly."""
        commitment = TriggerClassification(TriggerType.COMMITMENT, 0.9, "structural", ["AGREE"])
        question = TriggerClassification(TriggerType.QUESTION, 0.9, "structural", ["ANSWER"])

        assert commitment.is_commitment is True
        assert question.is_commitment is False

    def test_is_question_property(self):
        """is_question property works correctly."""
        question = TriggerClassification(TriggerType.QUESTION, 0.9, "structural", ["ANSWER"])
        statement = TriggerClassification(TriggerType.STATEMENT, 0.9, "structural", ["ACK"])

        assert question.is_question is True
        assert statement.is_question is False

    def test_is_reaction_property(self):
        """is_reaction property works correctly."""
        reaction = TriggerClassification(TriggerType.REACTION, 0.9, "structural", ["REACT"])
        social = TriggerClassification(TriggerType.SOCIAL, 0.9, "structural", ["ACK"])

        assert reaction.is_reaction is True
        assert social.is_reaction is False

    def test_is_social_property(self):
        """is_social property works correctly."""
        social = TriggerClassification(TriggerType.SOCIAL, 0.9, "structural", ["ACK"])
        statement = TriggerClassification(TriggerType.STATEMENT, 0.9, "structural", ["ACK"])

        assert social.is_social is True
        assert statement.is_social is False


class TestStructuralPatterns:
    """Tests for structural pattern matching."""

    @pytest.mark.parametrize(
        "text,expected_type",
        [
            # Tapbacks: normalized to empty by text_normalizer (iMessage metadata),
            # so classifier returns UNKNOWN for empty input
            ('Liked "great job"', TriggerType.UNKNOWN),
            ('Loved "thanks!"', TriggerType.UNKNOWN),
            ('Laughed at "funny message"', TriggerType.UNKNOWN),
            # SOCIAL: Greetings
            ("hey", TriggerType.SOCIAL),
            ("hi!", TriggerType.SOCIAL),
            ("hello", TriggerType.SOCIAL),
            ("yo", TriggerType.SOCIAL),
            ("what's up", TriggerType.SOCIAL),
            # SOCIAL: Acknowledgments
            ("ok", TriggerType.SOCIAL),
            ("sure", TriggerType.SOCIAL),
            ("got it", TriggerType.SOCIAL),
            ("sounds good", TriggerType.SOCIAL),
            ("thanks", TriggerType.SOCIAL),
            # "lol" gets expanded by slang normalization before pattern matching,
            # so the structural pattern no longer matches. Falls through to STATEMENT.
            ("lol", TriggerType.STATEMENT),
            # COMMITMENT: Invitations
            ("wanna grab lunch?", TriggerType.COMMITMENT),
            ("want to hang out?", TriggerType.COMMITMENT),
            ("you free tonight?", TriggerType.COMMITMENT),
            ("let's go get coffee", TriggerType.COMMITMENT),
            # COMMITMENT: Requests
            ("can you send me the file?", TriggerType.COMMITMENT),
            ("could you please check?", TriggerType.COMMITMENT),
            ("please call me back", TriggerType.COMMITMENT),
            # QUESTION: Info questions
            ("what time is it?", TriggerType.QUESTION),
            ("when are you coming?", TriggerType.QUESTION),
            ("where should we meet?", TriggerType.QUESTION),
            ("how much does it cost?", TriggerType.QUESTION),
            # REACTION: Reaction prompts (exclamations, not questions)
            ("that's crazy!", TriggerType.REACTION),
            ("omg that's insane!", TriggerType.REACTION),
            ("holy shit!", TriggerType.REACTION),
            # REACTION: Good/bad news
            ("i got the job!", TriggerType.REACTION),
            ("i passed!", TriggerType.REACTION),
            ("great news!", TriggerType.REACTION),
        ],
    )
    def test_structural_patterns_match(self, text: str, expected_type: TriggerType):
        """Structural patterns match expected trigger types."""
        classifier = HybridTriggerClassifier()
        result = classifier.classify(text)

        assert result.trigger_type == expected_type, (
            f"Expected {expected_type.value} for '{text}', got {result.trigger_type.value}"
        )

    def test_question_mark_fallback(self):
        """Question mark at end triggers QUESTION fallback."""
        classifier = HybridTriggerClassifier()
        result = classifier.classify("something random?")

        assert result.trigger_type == TriggerType.QUESTION


class TestHybridTriggerClassifier:
    """Tests for HybridTriggerClassifier."""

    def test_empty_input(self):
        """Empty input returns UNKNOWN with zero confidence."""
        classifier = HybridTriggerClassifier()

        result = classifier.classify("")
        assert result.trigger_type == TriggerType.UNKNOWN
        assert result.confidence == 0.0
        assert result.method == "empty"

        result = classifier.classify("   ")
        assert result.trigger_type == TriggerType.UNKNOWN
        assert result.confidence == 0.0

    def test_whitespace_only(self):
        """Whitespace-only input treated as empty."""
        classifier = HybridTriggerClassifier()
        result = classifier.classify("   \t\n  ")
        assert result.trigger_type == TriggerType.UNKNOWN
        assert result.method == "empty"

    def test_fallback_to_statement(self):
        """Unknown patterns fall back to STATEMENT."""
        classifier = HybridTriggerClassifier()
        result = classifier.classify("some random text here")

        assert result.trigger_type == TriggerType.STATEMENT
        assert result.method == "fallback"
        assert result.confidence == 0.3

    def test_high_confidence_structural_match(self):
        """High confidence structural matches skip SVM."""
        classifier = HybridTriggerClassifier()
        result = classifier.classify("hey!")

        assert result.trigger_type == TriggerType.SOCIAL
        assert result.confidence >= 0.85
        assert result.method == "structural"

    def test_valid_response_types_populated(self):
        """Valid response types are populated from mapping."""
        classifier = HybridTriggerClassifier()
        result = classifier.classify("wanna grab lunch?")

        assert result.trigger_type == TriggerType.COMMITMENT
        assert "AGREE" in result.valid_response_types
        assert "DECLINE" in result.valid_response_types
        assert "DEFER" in result.valid_response_types


class TestPerClassThresholds:
    """Tests for per-class SVM thresholds from config."""

    def test_thresholds_defined_in_config(self):
        """Main trigger types have thresholds defined in config."""
        cfg = get_config().classifier_thresholds
        # All threshold fields should exist and be valid floats
        assert 0 < cfg.trigger_svm_commitment <= 1.0
        assert 0 < cfg.trigger_svm_question <= 1.0
        assert 0 < cfg.trigger_svm_reaction <= 1.0
        assert 0 < cfg.trigger_svm_social <= 1.0
        assert 0 < cfg.trigger_svm_statement <= 1.0

    def test_commitment_has_highest_threshold(self):
        """COMMITMENT has highest threshold (most important)."""
        cfg = get_config().classifier_thresholds
        thresholds = {
            "commitment": cfg.trigger_svm_commitment,
            "question": cfg.trigger_svm_question,
            "reaction": cfg.trigger_svm_reaction,
            "social": cfg.trigger_svm_social,
            "statement": cfg.trigger_svm_statement,
        }
        commitment_threshold = thresholds["commitment"]
        for name, threshold in thresholds.items():
            if name != "commitment":
                assert commitment_threshold >= threshold

    def test_social_has_lowest_threshold(self):
        """SOCIAL has lowest threshold (strong structural patterns)."""
        cfg = get_config().classifier_thresholds
        thresholds = {
            "commitment": cfg.trigger_svm_commitment,
            "question": cfg.trigger_svm_question,
            "reaction": cfg.trigger_svm_reaction,
            "social": cfg.trigger_svm_social,
            "statement": cfg.trigger_svm_statement,
        }
        social_threshold = thresholds["social"]
        for name, threshold in thresholds.items():
            if name != "social":
                assert social_threshold <= threshold

    def test_default_threshold_exists(self):
        """Default SVM threshold is defined in config."""
        cfg = get_config().classifier_thresholds
        assert cfg.trigger_svm_default > 0
        assert cfg.trigger_svm_default < 1.0


class TestSingletonFactory:
    """Tests for singleton factory functions."""

    def test_get_trigger_classifier_returns_singleton(self):
        """get_trigger_classifier returns same instance."""
        reset_trigger_classifier()

        classifier1 = get_trigger_classifier()
        classifier2 = get_trigger_classifier()

        assert classifier1 is classifier2

    def test_reset_trigger_classifier_clears_singleton(self):
        """reset_trigger_classifier clears the singleton."""
        classifier1 = get_trigger_classifier()
        reset_trigger_classifier()
        classifier2 = get_trigger_classifier()

        assert classifier1 is not classifier2

    def test_classify_trigger_convenience_function(self):
        """classify_trigger uses singleton classifier."""
        reset_trigger_classifier()

        result = classify_trigger("hey!")
        assert result.trigger_type == TriggerType.SOCIAL


class TestTriggerToResponseMapping:
    """Tests for trigger to response type mapping."""

    def test_commitment_allows_agree_decline_defer(self):
        """COMMITMENT triggers allow AGREE, DECLINE, DEFER responses."""
        responses = TRIGGER_TO_RESPONSE_TYPES[TriggerType.COMMITMENT]
        assert "AGREE" in responses
        assert "DECLINE" in responses
        assert "DEFER" in responses

    def test_question_allows_answer(self):
        """QUESTION triggers allow ANSWER responses."""
        responses = TRIGGER_TO_RESPONSE_TYPES[TriggerType.QUESTION]
        assert "ANSWER" in responses or "YES" in responses or "NO" in responses

    def test_reaction_allows_react_types(self):
        """REACTION triggers allow reactive response types."""
        responses = TRIGGER_TO_RESPONSE_TYPES[TriggerType.REACTION]
        has_react = any("REACT" in r for r in responses)
        assert has_react or "QUESTION" in responses

    def test_social_allows_greeting_acknowledge(self):
        """SOCIAL triggers allow GREETING and ACKNOWLEDGE."""
        responses = TRIGGER_TO_RESPONSE_TYPES[TriggerType.SOCIAL]
        assert "GREETING" in responses or "ACKNOWLEDGE" in responses


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_case_insensitive_matching(self):
        """Patterns match case-insensitively."""
        classifier = HybridTriggerClassifier()

        lower = classifier.classify("hey")
        upper = classifier.classify("HEY")
        mixed = classifier.classify("HeY")

        assert lower.trigger_type == upper.trigger_type == mixed.trigger_type

    def test_unicode_handling(self):
        """Unicode characters don't cause errors."""
        classifier = HybridTriggerClassifier()

        # Should not raise
        result = classifier.classify("Hello 👋")
        assert result is not None

        result = classifier.classify("café meeting?")
        assert result is not None

    def test_very_long_input(self):
        """Very long input doesn't cause issues.

        Text exceeding max_length (1000 chars for classification profile)
        is normalized to empty, so classifier returns UNKNOWN.
        """
        classifier = HybridTriggerClassifier()
        long_text = "this is a test " * 100 + "?"

        result = classifier.classify(long_text)
        assert result.trigger_type == TriggerType.UNKNOWN

    def test_special_characters(self):
        """Special characters are handled."""
        classifier = HybridTriggerClassifier()

        result = classifier.classify("hey!!!")
        assert result.trigger_type == TriggerType.SOCIAL

        result = classifier.classify("what???")
        assert result.trigger_type == TriggerType.QUESTION
