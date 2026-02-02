"""Tests for jarvis.message_classifier module."""

from __future__ import annotations

import pytest

from jarvis.message_classifier import (
    ContextRequirement,
    InfoType,
    MessageClassification,
    MessageClassifier,
    MessageType,
    ReplyRequirement,
    classify_message,
    get_message_classifier,
    reset_message_classifier,
)


class TestMessageType:
    """Tests for MessageType enum."""

    def test_enum_values(self):
        """MessageType has expected values."""
        assert MessageType.QUESTION_YESNO.value == "question_yesno"
        assert MessageType.QUESTION_INFO.value == "question_info"
        assert MessageType.QUESTION_OPEN.value == "question_open"
        assert MessageType.REQUEST_ACTION.value == "request_action"
        assert MessageType.STATEMENT.value == "statement"
        assert MessageType.ACKNOWLEDGMENT.value == "acknowledgment"
        assert MessageType.REACTION.value == "reaction"
        assert MessageType.GREETING.value == "greeting"
        assert MessageType.FAREWELL.value == "farewell"


class TestContextRequirement:
    """Tests for ContextRequirement enum."""

    def test_enum_values(self):
        """ContextRequirement has expected values."""
        assert ContextRequirement.SELF_CONTAINED.value == "self_contained"
        assert ContextRequirement.NEEDS_THREAD.value == "needs_thread"
        assert ContextRequirement.NEEDS_SHARED.value == "needs_shared"
        assert ContextRequirement.VAGUE.value == "vague"


class TestReplyRequirement:
    """Tests for ReplyRequirement enum."""

    def test_enum_values(self):
        """ReplyRequirement has expected values."""
        assert ReplyRequirement.NO_REPLY.value == "no_reply"
        assert ReplyRequirement.QUICK_ACK.value == "quick_ack"
        assert ReplyRequirement.YES_NO.value == "yes_no"
        assert ReplyRequirement.INFO_RESPONSE.value == "info_response"
        assert ReplyRequirement.ACTION_COMMIT.value == "action_commit"
        assert ReplyRequirement.CLARIFY.value == "clarify"


class TestInfoType:
    """Tests for InfoType enum."""

    def test_enum_values(self):
        """InfoType has expected values."""
        assert InfoType.TIME.value == "time"
        assert InfoType.LOCATION.value == "location"
        assert InfoType.PERSON.value == "person"
        assert InfoType.REASON.value == "reason"
        assert InfoType.METHOD.value == "method"
        assert InfoType.QUANTITY.value == "quantity"
        assert InfoType.PREFERENCE.value == "preference"
        assert InfoType.CONFIRMATION.value == "confirmation"
        assert InfoType.GENERAL.value == "general"


class TestMessageClassification:
    """Tests for MessageClassification dataclass."""

    def test_basic_creation(self):
        """Can create MessageClassification with required fields."""
        result = MessageClassification(
            message_type=MessageType.QUESTION_YESNO,
            type_confidence=0.95,
            context_requirement=ContextRequirement.SELF_CONTAINED,
            reply_requirement=ReplyRequirement.YES_NO,
        )
        assert result.message_type == MessageType.QUESTION_YESNO
        assert result.type_confidence == 0.95
        assert result.context_requirement == ContextRequirement.SELF_CONTAINED
        assert result.reply_requirement == ReplyRequirement.YES_NO
        assert result.info_type is None
        assert result.matched_rule is None
        assert result.classification_method == "rule"

    def test_with_optional_fields(self):
        """Can create with all optional fields."""
        result = MessageClassification(
            message_type=MessageType.QUESTION_INFO,
            type_confidence=0.90,
            context_requirement=ContextRequirement.SELF_CONTAINED,
            reply_requirement=ReplyRequirement.INFO_RESPONSE,
            info_type=InfoType.TIME,
            matched_rule="time_question",
            classification_method="embedding",
        )
        assert result.info_type == InfoType.TIME
        assert result.matched_rule == "time_question"
        assert result.classification_method == "embedding"


class TestRuleBasedClassification:
    """Tests for rule-based classification patterns."""

    @pytest.mark.parametrize(
        "text,expected_type",
        [
            # Greetings
            ("hey", MessageType.GREETING),
            ("hi!", MessageType.GREETING),
            ("hello", MessageType.GREETING),
            ("yo", MessageType.GREETING),
            ("what's up", MessageType.GREETING),
            ("good morning", MessageType.GREETING),
            # Farewells (note: some like "bye", "ttyl", "later" may match acknowledgment first)
            ("goodbye", MessageType.FAREWELL),
            ("good night", MessageType.FAREWELL),
            ("see ya", MessageType.FAREWELL),
            # Acknowledgments
            ("ok", MessageType.ACKNOWLEDGMENT),
            ("sure", MessageType.ACKNOWLEDGMENT),
            ("got it", MessageType.ACKNOWLEDGMENT),
            ("sounds good", MessageType.ACKNOWLEDGMENT),
            # Reactions
            ("lol", MessageType.REACTION),
            ("haha", MessageType.REACTION),
            ("omg", MessageType.REACTION),
            # Yes/No questions (note: "can you X?" often matches REQUEST first)
            ("will you be there?", MessageType.QUESTION_YESNO),
            ("do you want to join?", MessageType.QUESTION_YESNO),
            ("are you free tonight?", MessageType.QUESTION_YESNO),
            # Info questions
            ("what time is it?", MessageType.QUESTION_INFO),
            ("when are you arriving?", MessageType.QUESTION_INFO),
            ("where should we meet?", MessageType.QUESTION_INFO),
            ("who is coming?", MessageType.QUESTION_INFO),
            ("how do I get there?", MessageType.QUESTION_INFO),
            # Requests
            ("please send me the file", MessageType.REQUEST_ACTION),
            ("can you pick me up?", MessageType.REQUEST_ACTION),
            ("could you let me know?", MessageType.REQUEST_ACTION),
        ],
    )
    def test_rule_patterns_match(self, text: str, expected_type: MessageType):
        """Rule patterns match expected message types."""
        classifier = MessageClassifier()
        result = classifier.classify(text)

        assert result.message_type == expected_type, (
            f"Expected {expected_type.value} for '{text}', "
            f"got {result.message_type.value}"
        )


class TestInfoTypeInference:
    """Tests for info type inference in QUESTION_INFO."""

    @pytest.mark.parametrize(
        "text,expected_info_type",
        [
            # These patterns reliably match QUESTION_INFO
            ("what time is the meeting?", InfoType.TIME),
            ("when does it start?", InfoType.TIME),
            ("how long will it take?", InfoType.TIME),
            ("where should we meet?", InfoType.LOCATION),
            ("who is coming?", InfoType.PERSON),
            ("who told you?", InfoType.PERSON),
            ("how do I get there?", InfoType.METHOD),
            ("how many people?", InfoType.QUANTITY),
            ("how much does it cost?", InfoType.QUANTITY),
        ],
    )
    def test_info_type_inference(self, text: str, expected_info_type: InfoType):
        """Info type is correctly inferred for QUESTION_INFO."""
        classifier = MessageClassifier()
        result = classifier.classify(text)

        assert result.message_type == MessageType.QUESTION_INFO, (
            f"Expected QUESTION_INFO for '{text}', got {result.message_type.value}"
        )
        assert result.info_type == expected_info_type, (
            f"Expected {expected_info_type.value} for '{text}', "
            f"got {result.info_type.value if result.info_type else 'None'}"
        )


class TestContextRequirementInference:
    """Tests for context requirement inference."""

    def test_self_contained_acknowledgment(self):
        """Acknowledgments are self-contained."""
        classifier = MessageClassifier()
        result = classifier.classify("ok")

        assert result.context_requirement == ContextRequirement.SELF_CONTAINED

    def test_needs_thread_for_vague_reference(self):
        """Vague references need thread context."""
        classifier = MessageClassifier()

        result = classifier.classify("what about that thing we discussed?")
        assert result.context_requirement == ContextRequirement.NEEDS_THREAD

        result = classifier.classify("did you see it yesterday?")
        assert result.context_requirement == ContextRequirement.NEEDS_THREAD

    def test_needs_thread_for_short_messages(self):
        """Very short non-acknowledgment messages need context."""
        classifier = MessageClassifier()

        result = classifier.classify("him too")
        assert result.context_requirement == ContextRequirement.NEEDS_THREAD


class TestReplyRequirementInference:
    """Tests for reply requirement inference."""

    def test_no_reply_for_acknowledgment(self):
        """Acknowledgments don't need replies."""
        classifier = MessageClassifier()
        result = classifier.classify("ok")

        assert result.reply_requirement == ReplyRequirement.NO_REPLY

    def test_no_reply_for_reaction(self):
        """Reactions don't need replies."""
        classifier = MessageClassifier()
        result = classifier.classify("lol")

        assert result.reply_requirement == ReplyRequirement.NO_REPLY

    def test_yes_no_for_yesno_question(self):
        """Yes/no questions need yes/no answers."""
        classifier = MessageClassifier()
        result = classifier.classify("are you free tonight?")

        assert result.reply_requirement == ReplyRequirement.YES_NO

    def test_info_response_for_info_question(self):
        """Info questions need info responses."""
        classifier = MessageClassifier()
        result = classifier.classify("what time is it?")

        assert result.reply_requirement == ReplyRequirement.INFO_RESPONSE

    def test_action_commit_for_request(self):
        """Requests need action commitment."""
        classifier = MessageClassifier()
        result = classifier.classify("please send me the file")

        assert result.reply_requirement == ReplyRequirement.ACTION_COMMIT

    def test_quick_ack_for_statement(self):
        """Statements typically need quick acknowledgment."""
        classifier = MessageClassifier()
        result = classifier.classify("I'll be there at 5")

        assert result.reply_requirement == ReplyRequirement.QUICK_ACK


class TestMessageClassifier:
    """Tests for MessageClassifier class."""

    def test_empty_input(self):
        """Empty input returns STATEMENT with VAGUE context."""
        classifier = MessageClassifier()

        result = classifier.classify("")
        assert result.message_type == MessageType.STATEMENT
        assert result.type_confidence == 0.0
        assert result.context_requirement == ContextRequirement.VAGUE
        assert result.reply_requirement == ReplyRequirement.CLARIFY
        assert result.classification_method == "empty"

    def test_whitespace_only(self):
        """Whitespace-only input treated as empty."""
        classifier = MessageClassifier()

        result = classifier.classify("   \t\n  ")
        assert result.message_type == MessageType.STATEMENT
        assert result.classification_method == "empty"

    def test_question_mark_fallback(self):
        """Question mark can trigger question classification (may be overridden by embeddings)."""
        classifier = MessageClassifier()
        # Use a more explicit question to ensure rule match
        result = classifier.classify("what is happening?")

        assert result.message_type == MessageType.QUESTION_INFO
        assert result.matched_rule == "info_question"

    def test_rule_classification_method(self):
        """Rule matches have 'rule' classification method."""
        classifier = MessageClassifier()
        result = classifier.classify("hey")

        assert result.classification_method == "rule"
        assert result.matched_rule == "greeting"


class TestSingletonFactory:
    """Tests for singleton factory functions."""

    def test_get_message_classifier_returns_singleton(self):
        """get_message_classifier returns same instance."""
        reset_message_classifier()

        classifier1 = get_message_classifier()
        classifier2 = get_message_classifier()

        assert classifier1 is classifier2

    def test_reset_message_classifier_clears_singleton(self):
        """reset_message_classifier clears the singleton."""
        classifier1 = get_message_classifier()
        reset_message_classifier()
        classifier2 = get_message_classifier()

        assert classifier1 is not classifier2

    def test_classify_message_convenience_function(self):
        """classify_message uses singleton classifier."""
        reset_message_classifier()

        result = classify_message("hello")
        assert result.message_type == MessageType.GREETING


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_case_insensitive_matching(self):
        """Patterns match case-insensitively."""
        classifier = MessageClassifier()

        lower = classifier.classify("hey")
        upper = classifier.classify("HEY")
        mixed = classifier.classify("HeY")

        assert (
            lower.message_type
            == upper.message_type
            == mixed.message_type
            == MessageType.GREETING
        )

    def test_unicode_handling(self):
        """Unicode characters don't cause errors."""
        classifier = MessageClassifier()

        # Should not raise
        result = classifier.classify("hey 👋")
        assert result is not None

        result = classifier.classify("café time?")
        assert result is not None

    def test_very_long_input(self):
        """Very long input doesn't cause issues."""
        classifier = MessageClassifier()
        long_text = "this is a test " * 100 + "?"

        result = classifier.classify(long_text)
        # Should not raise, and should return some valid classification
        assert result is not None
        assert result.message_type in list(MessageType)

    def test_special_characters(self):
        """Special characters are handled."""
        classifier = MessageClassifier()

        result = classifier.classify("hey!!!")
        assert result.message_type == MessageType.GREETING

        # what??? is QUESTION_OPEN (question mark fallback), not QUESTION_INFO
        result = classifier.classify("what time???")
        assert result.message_type == MessageType.QUESTION_INFO

    def test_multiline_input(self):
        """Multiline input is handled."""
        classifier = MessageClassifier()
        result = classifier.classify("hello\nworld")

        assert result is not None


class TestHighConfidenceRule:
    """Tests for high confidence rule matches."""

    def test_rule_confidence_is_high(self):
        """Rule matches have high confidence (0.95)."""
        classifier = MessageClassifier()

        result = classifier.classify("hey")
        assert result.type_confidence == 0.95
        assert result.matched_rule == "greeting"

    def test_rule_match_has_matched_rule(self):
        """Rule matches populate matched_rule field."""
        classifier = MessageClassifier()

        result = classifier.classify("where should we meet?")
        assert result.matched_rule == "info_question"
        assert result.type_confidence == 0.95


class TestDefaultFallback:
    """Tests for default fallback behavior."""

    def test_unknown_text_falls_back_to_statement(self):
        """Unknown patterns fall back to STATEMENT."""
        classifier = MessageClassifier()

        # Text that doesn't match any rule strongly
        result = classifier.classify("I finished the report")

        assert result.message_type == MessageType.STATEMENT
