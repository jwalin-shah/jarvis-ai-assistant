"""Tests for jarvis/classifiers/response_mobilization.py - Response pressure classification."""


from jarvis.classifiers.cascade import MobilizationCascade, reset_mobilization_cascade
from jarvis.classifiers.intent_classifier import IntentResult, KeywordIntentClassifier
from jarvis.classifiers.response_mobilization import (
    COMMITMENT_RESPONSE_OPTIONS,
    EMOTIONAL_RESPONSE_OPTIONS,
    QUESTION_RESPONSE_OPTIONS,
    ResponseOptionType,
    ResponsePressure,
    ResponseType,
    classify_legacy,
    classify_response_pressure,
    get_response_pressure,
    get_valid_response_options,
    requires_response,
    response_optional,
    to_legacy_category,
)

# =============================================================================
# Test ResponsePressure Enum
# =============================================================================


class TestResponsePressure:
    """Tests for ResponsePressure enum."""

    def test_enum_values(self) -> None:
        """Test that enum has correct values."""
        assert ResponsePressure.HIGH == "high"
        assert ResponsePressure.MEDIUM == "medium"
        assert ResponsePressure.LOW == "low"
        assert ResponsePressure.NONE == "none"


# =============================================================================
# Test ResponseType Enum
# =============================================================================


class TestResponseType:
    """Tests for ResponseType enum."""

    def test_enum_values(self) -> None:
        """Test that enum has correct values."""
        assert ResponseType.COMMITMENT == "commitment"
        assert ResponseType.ANSWER == "answer"
        assert ResponseType.EMOTIONAL == "emotional"
        assert ResponseType.OPTIONAL == "optional"
        assert ResponseType.CLOSING == "closing"


# =============================================================================
# Test HIGH Pressure - Requests and Invitations
# =============================================================================


class TestHighPressureRequests:
    """Tests for HIGH pressure requests (COMMITMENT)."""

    def test_can_you_patterns(self) -> None:
        """Test 'can you' request patterns."""
        result = classify_response_pressure("Can you help me?")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.COMMITMENT
        assert result.confidence > 0.8

        result = classify_response_pressure("can you pick me up")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.COMMITMENT

    def test_wanna_patterns(self) -> None:
        """Test 'wanna' invitation patterns."""
        result = classify_response_pressure("wanna grab lunch?")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.COMMITMENT

        result = classify_response_pressure("want to hang out?")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.COMMITMENT

    def test_short_invitation_patterns(self) -> None:
        """Test one-word invitation prompts."""
        result = classify_response_pressure("wanna?")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.COMMITMENT

        result = classify_response_pressure("down?")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.COMMITMENT

    def test_let_me_know_patterns(self) -> None:
        """Test 'let me know' patterns."""
        result = classify_response_pressure("let me know when you're free")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.COMMITMENT

        result = classify_response_pressure("lmk if you can make it")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.COMMITMENT

    def test_imperative_verbs(self) -> None:
        """Test imperative verb patterns."""
        result = classify_response_pressure("send me the file")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.COMMITMENT

        result = classify_response_pressure("pick me up at 5")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.COMMITMENT

        result = classify_response_pressure("call me later")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.COMMITMENT

    def test_offers_requiring_decision(self) -> None:
        """Test offers that require a decision."""
        result = classify_response_pressure("want me to pick you up?")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.COMMITMENT

        result = classify_response_pressure("should i bring anything?")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.COMMITMENT


# =============================================================================
# Test HIGH Pressure - Questions
# =============================================================================


class TestHighPressureQuestions:
    """Tests for HIGH pressure questions (ANSWER)."""

    def test_wh_questions_with_question_mark(self) -> None:
        """Test WH-questions with question mark."""
        result = classify_response_pressure("What time is it?")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.ANSWER

        result = classify_response_pressure("Where are you?")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.ANSWER

        result = classify_response_pressure("When does it start?")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.ANSWER

    def test_aux_inversion_questions(self) -> None:
        """Test auxiliary inversion questions."""
        result = classify_response_pressure("Are you coming?")
        assert result.pressure == ResponsePressure.HIGH
        # "Are you coming?" is a commitment question
        assert result.response_type == ResponseType.COMMITMENT

        result = classify_response_pressure("Do you know the answer?")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.ANSWER

        result = classify_response_pressure("Can you help?")
        assert result.pressure == ResponsePressure.HIGH
        # "Can you" is a request pattern, so COMMITMENT
        assert result.response_type == ResponseType.COMMITMENT

    def test_recipient_oriented_questions(self) -> None:
        """Test recipient-oriented questions."""
        result = classify_response_pressure("Do you know where it is?")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.ANSWER

        result = classify_response_pressure("Are you free?")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.COMMITMENT

    def test_info_question_patterns(self) -> None:
        """Test information-seeking question patterns."""
        result = classify_response_pressure("what time is the game")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.ANSWER

        result = classify_response_pressure("where are you at")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.ANSWER

        result = classify_response_pressure("when does it start")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.ANSWER

    def test_slang_question_patterns(self) -> None:
        """Slang and short forms should be treated as direct questions."""
        for text in ("wya", "wyd?", "hbu", "u free", "what about you?", "thoughts?"):
            result = classify_response_pressure(text)
            assert result.pressure == ResponsePressure.HIGH
            assert result.response_type == ResponseType.ANSWER
            assert result.confidence >= 0.80

    def test_declarative_questions(self) -> None:
        """Test declarative questions (B-event statements)."""
        result = classify_response_pressure("You're coming?")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.CONFIRMATION

        result = classify_response_pressure("So we're meeting at 5?")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.CONFIRMATION


# =============================================================================
# Test MEDIUM Pressure - Emotional/Reactive
# =============================================================================


class TestMediumPressureReactive:
    """Tests for MEDIUM pressure reactive content (EMOTIONAL)."""

    def test_multiple_exclamation_marks(self) -> None:
        """Test multiple exclamation marks."""
        result = classify_response_pressure("I got the job!!")
        assert result.pressure == ResponsePressure.MEDIUM
        assert result.response_type == ResponseType.EMOTIONAL

        result = classify_response_pressure("We won!!!")
        assert result.pressure == ResponsePressure.MEDIUM
        assert result.response_type == ResponseType.EMOTIONAL

    def test_emotional_interjections(self) -> None:
        """Test emotional interjections."""
        result = classify_response_pressure("omg that's amazing!")
        assert result.pressure == ResponsePressure.MEDIUM
        assert result.response_type == ResponseType.EMOTIONAL

        result = classify_response_pressure("wow congrats!")
        assert result.pressure == ResponsePressure.MEDIUM
        assert result.response_type == ResponseType.EMOTIONAL

    def test_news_announcements(self) -> None:
        """Test news announcements."""
        result = classify_response_pressure("I got the job!")
        assert result.pressure == ResponsePressure.MEDIUM
        assert result.response_type == ResponseType.EMOTIONAL

        result = classify_response_pressure("we won the game!")
        assert result.pressure == ResponsePressure.MEDIUM
        assert result.response_type == ResponseType.EMOTIONAL

    def test_strong_emotion_patterns(self) -> None:
        """Test strong emotion patterns."""
        result = classify_response_pressure("I'm so happy!")
        assert result.pressure == ResponsePressure.MEDIUM
        assert result.response_type == ResponseType.EMOTIONAL

        result = classify_response_pressure("can't believe it!")
        assert result.pressure == ResponsePressure.MEDIUM
        assert result.response_type == ResponseType.EMOTIONAL

    def test_congratulations(self) -> None:
        """Test congratulations patterns."""
        result = classify_response_pressure("congrats on the promotion!")
        assert result.pressure == ResponsePressure.MEDIUM
        assert result.response_type == ResponseType.EMOTIONAL


# =============================================================================
# Test LOW Pressure - Optional
# =============================================================================


class TestLowPressureOptional:
    """Tests for LOW pressure optional responses."""

    def test_speaker_oriented_musings(self) -> None:
        """Test speaker-oriented musings."""
        result = classify_response_pressure("I wonder if they'll win")
        assert result.pressure == ResponsePressure.LOW
        assert result.response_type == ResponseType.OPTIONAL

        result = classify_response_pressure("I'm curious about that")
        assert result.pressure == ResponsePressure.LOW
        assert result.response_type == ResponseType.OPTIONAL

    def test_opinions(self) -> None:
        """Test opinion statements."""
        result = classify_response_pressure("I think that's a good idea")
        assert result.pressure == ResponsePressure.LOW
        assert result.response_type == ResponseType.OPTIONAL

        result = classify_response_pressure("probably not gonna happen")
        assert result.pressure == ResponsePressure.LOW
        assert result.response_type == ResponseType.OPTIONAL

    def test_tellings(self) -> None:
        """Test telling/informing patterns."""
        result = classify_response_pressure("I went to the store")
        assert result.pressure == ResponsePressure.LOW
        assert result.response_type == ResponseType.OPTIONAL

        result = classify_response_pressure("fyi the meeting is at 3")
        assert result.pressure == ResponsePressure.LOW
        assert result.response_type == ResponseType.OPTIONAL

        result = classify_response_pressure("btw I'll be there")
        assert result.pressure == ResponsePressure.LOW
        assert result.response_type == ResponseType.OPTIONAL

    def test_rhetorical_questions(self) -> None:
        """Test rhetorical questions."""
        result = classify_response_pressure("why do dads even do that")
        assert result.pressure == ResponsePressure.LOW
        assert result.response_type == ResponseType.OPTIONAL

        result = classify_response_pressure("how does that even work")
        assert result.pressure == ResponsePressure.LOW
        assert result.response_type == ResponseType.OPTIONAL

    def test_rhetorical_with_question_mark(self) -> None:
        """Test rhetorical questions with question mark."""
        result = classify_response_pressure("Why do dads even do that?")
        # Should still be LOW (rhetorical pattern overrides)
        assert result.pressure == ResponsePressure.LOW
        assert result.response_type == ResponseType.OPTIONAL


# =============================================================================
# Test NONE Pressure - Backchannels and Closings
# =============================================================================


class TestNonePressureBackchannels:
    """Tests for NONE pressure backchannels and closings."""

    def test_acknowledgments(self) -> None:
        """Test acknowledgment words."""
        result = classify_response_pressure("ok")
        assert result.pressure == ResponsePressure.NONE
        assert result.response_type == ResponseType.CLOSING

        result = classify_response_pressure("sure")
        assert result.pressure == ResponsePressure.NONE
        assert result.response_type == ResponseType.CLOSING

        result = classify_response_pressure("gotcha")
        assert result.pressure == ResponsePressure.NONE
        assert result.response_type == ResponseType.CLOSING

    def test_reaction_words(self) -> None:
        """Test reaction words."""
        result = classify_response_pressure("lol")
        assert result.pressure == ResponsePressure.NONE
        assert result.response_type == ResponseType.CLOSING

        result = classify_response_pressure("nice")
        assert result.pressure == ResponsePressure.NONE
        assert result.response_type == ResponseType.CLOSING

    def test_gratitude(self) -> None:
        """Test gratitude expressions."""
        result = classify_response_pressure("thanks")
        assert result.pressure == ResponsePressure.NONE
        assert result.response_type == ResponseType.CLOSING

        result = classify_response_pressure("thank you")
        assert result.pressure == ResponsePressure.NONE
        assert result.response_type == ResponseType.CLOSING

    def test_closings(self) -> None:
        """Test closing expressions."""
        result = classify_response_pressure("bye")
        assert result.pressure == ResponsePressure.NONE
        assert result.response_type == ResponseType.CLOSING

        result = classify_response_pressure("see ya")
        assert result.pressure == ResponsePressure.NONE
        assert result.response_type == ResponseType.CLOSING

    def test_negated_request_is_not_actionable(self) -> None:
        result = classify_response_pressure("Don't text me")
        assert result.pressure == ResponsePressure.NONE
        assert result.response_type == ResponseType.CLOSING
        assert result.confidence >= 0.95

# =============================================================================
# Test Greetings
# =============================================================================


class TestGreetings:
    """Tests for greeting patterns."""

    def test_greetings_low_pressure(self) -> None:
        """Test that greetings are LOW pressure."""
        result = classify_response_pressure("hey")
        assert result.pressure == ResponsePressure.LOW
        assert result.response_type == ResponseType.OPTIONAL

        result = classify_response_pressure("hi")
        assert result.pressure == ResponsePressure.LOW
        assert result.response_type == ResponseType.OPTIONAL

        result = classify_response_pressure("what's up")
        assert result.pressure == ResponsePressure.LOW
        assert result.response_type == ResponseType.OPTIONAL


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_string(self) -> None:
        """Test empty string input."""
        result = classify_response_pressure("")
        assert result.pressure == ResponsePressure.NONE
        assert result.response_type == ResponseType.CLOSING
        assert result.confidence == 0.0
        assert result.method == "empty"

    def test_whitespace_only(self) -> None:
        """Test whitespace-only input."""
        result = classify_response_pressure("   ")
        assert result.pressure == ResponsePressure.NONE
        assert result.response_type == ResponseType.CLOSING

    def test_default_fallback(self) -> None:
        """Test default fallback for unrecognized patterns."""
        result = classify_response_pressure("random text that doesn't match patterns")
        # Should default to LOW
        assert result.pressure == ResponsePressure.LOW
        assert result.response_type == ResponseType.OPTIONAL
        assert result.method == "default"

    def test_feature_detection(self) -> None:
        """Test that features are detected correctly."""
        result = classify_response_pressure("Can you help?")
        assert result.features["is_request"] is True
        assert result.features["has_question_mark"] is True

        result = classify_response_pressure("I wonder if...")
        assert result.features["is_speaker_oriented"] is True

        result = classify_response_pressure("omg!!")
        assert result.features["has_multiple_exclamation"] is True
        assert result.features["is_reactive"] is True


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_response_pressure(self) -> None:
        """Test get_response_pressure convenience function."""
        pressure = get_response_pressure("Can you help?")
        assert pressure == ResponsePressure.HIGH

        pressure = get_response_pressure("ok")
        assert pressure == ResponsePressure.NONE

    def test_requires_response(self) -> None:
        """Test requires_response convenience function."""
        assert requires_response("Can you help?") is True
        assert requires_response("What time is it?") is True
        assert requires_response("ok") is False
        assert requires_response("I wonder if...") is False

    def test_response_optional(self) -> None:
        """Test response_optional convenience function."""
        assert response_optional("ok") is True
        assert response_optional("I think so") is True
        assert response_optional("Can you help?") is False
        assert response_optional("What time?") is False


# =============================================================================
# Test Legacy Compatibility
# =============================================================================


class TestLegacyCompatibility:
    """Tests for legacy category mapping."""

    def test_to_legacy_category(self) -> None:
        """Test to_legacy_category mapping."""
        result = classify_response_pressure("Can you help?")
        legacy = to_legacy_category(result)
        assert legacy == "ACTIONABLE"

        result = classify_response_pressure("What time is it?")
        legacy = to_legacy_category(result)
        assert legacy == "ANSWERABLE"

        result = classify_response_pressure("I got the job!!")
        legacy = to_legacy_category(result)
        assert legacy == "REACTIVE"

        result = classify_response_pressure("ok")
        legacy = to_legacy_category(result)
        assert legacy == "ACKNOWLEDGEABLE"

    def test_classify_legacy(self) -> None:
        """Test classify_legacy convenience function."""
        assert classify_legacy("Can you help?") == "ACTIONABLE"
        assert classify_legacy("What time is it?") == "ANSWERABLE"
        assert classify_legacy("I got the job!!") == "REACTIVE"
        assert classify_legacy("ok") == "ACKNOWLEDGEABLE"


# =============================================================================
# Test Response Options
# =============================================================================


class TestResponseOptions:
    """Tests for response option types."""

    def test_commitment_response_options(self) -> None:
        """Test commitment response options."""
        result = classify_response_pressure("Can you help?")
        options = get_valid_response_options(result)
        assert ResponseOptionType.AGREE in options
        assert ResponseOptionType.DECLINE in options
        assert ResponseOptionType.DEFER in options

    def test_question_response_options(self) -> None:
        """Test question response options."""
        result = classify_response_pressure("What time is it?")
        options = get_valid_response_options(result)
        assert ResponseOptionType.YES in options
        assert ResponseOptionType.NO in options
        assert ResponseOptionType.ANSWER in options

    def test_emotional_response_options(self) -> None:
        """Test emotional response options."""
        result = classify_response_pressure("I got the job!!")
        options = get_valid_response_options(result)
        assert ResponseOptionType.REACT_POSITIVE in options
        assert ResponseOptionType.REACT_SYMPATHY in options

    def test_optional_response_options(self) -> None:
        """Test optional response options."""
        result = classify_response_pressure("I think so")
        options = get_valid_response_options(result)
        assert ResponseOptionType.ACKNOWLEDGE in options

    def test_confirmation_response_options(self) -> None:
        """Test confirmation response options."""
        result = classify_response_pressure("You're coming?")
        options = get_valid_response_options(result)
        assert ResponseOptionType.YES in options
        assert ResponseOptionType.NO in options

    def test_response_option_constants(self) -> None:
        """Test that response option constants are defined."""
        assert len(COMMITMENT_RESPONSE_OPTIONS) == 3
        assert len(QUESTION_RESPONSE_OPTIONS) == 3
        assert len(EMOTIONAL_RESPONSE_OPTIONS) == 2


# =============================================================================
# Test Confidence Scores
# =============================================================================


class TestConfidenceScores:
    """Tests for confidence score calculation."""

    def test_high_confidence_patterns(self) -> None:
        """Test patterns that should have high confidence."""
        result = classify_response_pressure("Can you help?")
        assert result.confidence > 0.8

        result = classify_response_pressure("ok")
        assert result.confidence > 0.8

        result = classify_response_pressure("I got the job!!")
        assert result.confidence > 0.8

    def test_confidence_range(self) -> None:
        """Test that confidence is in valid range."""
        test_cases = [
            "Can you help?",
            "What time is it?",
            "I wonder if...",
            "ok",
            "random text",
        ]

        for text in test_cases:
            result = classify_response_pressure(text)
            assert 0.0 <= result.confidence <= 1.0


# =============================================================================
# Test Normalization
# =============================================================================


class TestNormalization:
    """Tests for text normalization."""

    def test_case_insensitive(self) -> None:
        """Test that classification is case-insensitive."""
        result1 = classify_response_pressure("Can you help?")
        result2 = classify_response_pressure("CAN YOU HELP?")
        assert result1.pressure == result2.pressure
        assert result1.response_type == result2.response_type

    def test_whitespace_handling(self) -> None:
        """Test that extra whitespace is handled."""
        result1 = classify_response_pressure("Can you help?")
        result2 = classify_response_pressure("  Can you help?  ")
        assert result1.pressure == result2.pressure

    def test_punctuation_handling(self) -> None:
        """Test that punctuation is handled correctly."""
        result1 = classify_response_pressure("Can you help?")
        _ = classify_response_pressure("Can you help")
        # May differ slightly but should be similar
        assert result1.pressure in (ResponsePressure.HIGH, ResponsePressure.LOW)


# =============================================================================
# Test Cascade + Intent Fallback
# =============================================================================


class TestIntentFallbackClassifier:
    """Tests for keyword-based intent fallback classifier."""

    def test_keyword_classifier_no_reply(self) -> None:
        clf = KeywordIntentClassifier()
        result = clf.classify("ok", ["no_reply_ack", "reply_casual_chat"])
        assert result.intent == "no_reply_ack"
        assert result.confidence >= 0.8

    def test_keyword_classifier_question_slang(self) -> None:
        clf = KeywordIntentClassifier()
        result = clf.classify("wya", ["reply_question_info", "reply_casual_chat"])
        assert result.intent == "reply_question_info"
        assert result.confidence >= 0.8


class _StubIntentClassifier:
    def __init__(self, intent: str, confidence: float = 0.9) -> None:
        self.intent = intent
        self.confidence = confidence
        self.called = False

    def classify(self, text: str, intent_options: list[str]) -> IntentResult:
        self.called = True
        return IntentResult(intent=self.intent, confidence=self.confidence, method="stub")


class TestMobilizationCascade:
    """Tests for rules -> intent cascade behavior."""

    def test_cascade_keeps_high_conf_rule_result(self) -> None:
        stub = _StubIntentClassifier("reply_emotional_support", 0.95)
        cascade = MobilizationCascade(intent_classifier=stub, confidence_threshold=0.80)
        result = cascade.classify("Can you help me?")
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.COMMITMENT
        assert stub.called is False

    def test_cascade_uses_fallback_for_low_conf_rule_result(self) -> None:
        stub = _StubIntentClassifier("reply_question_info", 0.92)
        cascade = MobilizationCascade(intent_classifier=stub, confidence_threshold=0.80)
        result = cascade.classify("random words maybe")
        assert stub.called is True
        assert result.pressure == ResponsePressure.HIGH
        assert result.response_type == ResponseType.ANSWER
        assert result.method == "intent_fallback"

    def test_cascade_singleton_reset(self) -> None:
        reset_mobilization_cascade()
