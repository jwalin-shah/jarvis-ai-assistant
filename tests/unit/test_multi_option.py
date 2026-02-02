"""Tests for jarvis.multi_option module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jarvis.multi_option import (
    COMMITMENT_TRIGGER_TYPES,
    FALLBACK_TEMPLATES,
    OPTION_PRIORITY,
    MultiOptionGenerator,
    MultiOptionResult,
    ResponseOption,
    generate_response_options,
    get_multi_option_generator,
    reset_multi_option_generator,
)
from jarvis.response_classifier import ResponseType


class TestResponseOption:
    """Tests for ResponseOption dataclass."""

    def test_basic_creation(self):
        """Can create ResponseOption with required fields."""
        option = ResponseOption(
            text="Yeah I'm down!",
            response_type=ResponseType.AGREE,
            confidence=0.95,
            source="template",
        )
        assert option.text == "Yeah I'm down!"
        assert option.response_type == ResponseType.AGREE
        assert option.confidence == 0.95
        assert option.source == "template"

    def test_to_dict(self):
        """to_dict returns correct dictionary."""
        option = ResponseOption(
            text="Sorry, can't make it",
            response_type=ResponseType.DECLINE,
            confidence=0.8,
            source="generated",
        )
        result = option.to_dict()

        assert result["type"] == "DECLINE"
        assert result["response"] == "Sorry, can't make it"
        assert result["confidence"] == 0.8
        assert result["source"] == "generated"


class TestMultiOptionResult:
    """Tests for MultiOptionResult dataclass."""

    def test_basic_creation(self):
        """Can create MultiOptionResult with required fields."""
        result = MultiOptionResult(
            trigger="Want to grab lunch?",
            trigger_da="commitment",
            is_commitment=True,
        )
        assert result.trigger == "Want to grab lunch?"
        assert result.trigger_da == "commitment"
        assert result.is_commitment is True
        assert result.options == []

    def test_with_options(self):
        """Can create with options list."""
        options = [
            ResponseOption("Yes!", ResponseType.AGREE, 0.9, "template"),
            ResponseOption("No", ResponseType.DECLINE, 0.9, "template"),
        ]
        result = MultiOptionResult(
            trigger="Test?",
            trigger_da="commitment",
            is_commitment=True,
            options=options,
        )
        assert len(result.options) == 2

    def test_to_dict(self):
        """to_dict returns correct dictionary."""
        options = [
            ResponseOption("Yes!", ResponseType.AGREE, 0.9, "template"),
        ]
        result = MultiOptionResult(
            trigger="Test?",
            trigger_da="commitment",
            is_commitment=True,
            options=options,
        )
        d = result.to_dict()

        assert d["trigger"] == "Test?"
        assert d["trigger_da"] == "commitment"
        assert d["is_commitment"] is True
        assert len(d["options"]) == 1
        assert d["suggestions"] == ["Yes!"]

    def test_has_options_property(self):
        """has_options property works correctly."""
        empty = MultiOptionResult("Test?", "commitment", True, [])
        assert empty.has_options is False

        with_options = MultiOptionResult(
            "Test?",
            "commitment",
            True,
            [ResponseOption("Yes", ResponseType.AGREE, 0.9, "t")],
        )
        assert with_options.has_options is True

    def test_get_option_by_type(self):
        """get_option returns correct option by type."""
        options = [
            ResponseOption("Yes!", ResponseType.AGREE, 0.9, "template"),
            ResponseOption("No", ResponseType.DECLINE, 0.8, "template"),
        ]
        result = MultiOptionResult("Test?", "commitment", True, options)

        agree = result.get_option(ResponseType.AGREE)
        assert agree is not None
        assert agree.text == "Yes!"

        decline = result.get_option(ResponseType.DECLINE)
        assert decline is not None
        assert decline.text == "No"

        defer = result.get_option(ResponseType.DEFER)
        assert defer is None


class TestFallbackTemplates:
    """Tests for fallback template constants."""

    def test_main_types_have_templates(self):
        """Main response types have fallback templates."""
        assert ResponseType.AGREE in FALLBACK_TEMPLATES
        assert ResponseType.DECLINE in FALLBACK_TEMPLATES
        assert ResponseType.DEFER in FALLBACK_TEMPLATES

    def test_templates_are_non_empty(self):
        """Each template list is non-empty."""
        for response_type, templates in FALLBACK_TEMPLATES.items():
            assert len(templates) > 0, f"{response_type} has no templates"

    def test_templates_are_strings(self):
        """All templates are strings."""
        for response_type, templates in FALLBACK_TEMPLATES.items():
            for template in templates:
                assert isinstance(template, str)


class TestOptionPriority:
    """Tests for option priority ordering."""

    def test_agree_first(self):
        """AGREE is first in priority (most common response)."""
        assert OPTION_PRIORITY[0] == ResponseType.AGREE

    def test_contains_commitment_types(self):
        """Priority contains all commitment types."""
        assert ResponseType.AGREE in OPTION_PRIORITY
        assert ResponseType.DECLINE in OPTION_PRIORITY
        assert ResponseType.DEFER in OPTION_PRIORITY


class TestCommitmentTriggerTypes:
    """Tests for commitment trigger types constant."""

    def test_contains_commitment(self):
        """Contains 'commitment' (new classifier label)."""
        assert "commitment" in COMMITMENT_TRIGGER_TYPES

    def test_contains_legacy_labels(self):
        """Contains legacy labels for backwards compatibility."""
        assert "INVITATION" in COMMITMENT_TRIGGER_TYPES
        assert "REQUEST" in COMMITMENT_TRIGGER_TYPES

    def test_is_frozenset(self):
        """COMMITMENT_TRIGGER_TYPES is immutable."""
        assert isinstance(COMMITMENT_TRIGGER_TYPES, frozenset)


class TestInfoStatementPatterns:
    """Tests for INFO_STATEMENT pattern detection."""

    @pytest.mark.parametrize(
        "text",
        [
            # Location/transit status
            "on my way",
            "omw",
            "I'm heading there now",
            "just left",
            "almost there",
            "be there in 5",
            "5 minutes away",
            "eta 10",
            # Running late
            "running late",
            "gonna be late",
            "sorry stuck in traffic",
            "got held up",
            # Simple status
            "I'm here",
            "I'm home",
            "just woke up",
            "already done",
        ],
    )
    def test_info_statement_patterns_match(self, text: str):
        """INFO_STATEMENT patterns correctly identified."""
        from jarvis.multi_option import _is_info_statement

        assert _is_info_statement(text) is True, f"'{text}' should match INFO_STATEMENT"

    @pytest.mark.parametrize(
        "text",
        [
            "want to grab lunch?",
            "can you help me?",
            "are you free tonight?",
            "what time works?",
        ],
    )
    def test_non_info_statements_dont_match(self, text: str):
        """Non-INFO_STATEMENT patterns don't match."""
        from jarvis.multi_option import _is_info_statement

        assert (
            _is_info_statement(text) is False
        ), f"'{text}' should NOT match INFO_STATEMENT"


class TestWhQuestionPatterns:
    """Tests for WH_QUESTION pattern detection."""

    @pytest.mark.parametrize(
        "text",
        [
            "who's coming?",
            "who else is going?",
            "who did you invite?",
            "what time?",
            "what's the plan?",
            "when is it?",
            "where are we meeting?",
            "how long will it take?",
            "how much does it cost?",
        ],
    )
    def test_wh_question_patterns_match(self, text: str):
        """WH_QUESTION patterns correctly identified."""
        from jarvis.multi_option import _is_wh_question

        assert _is_wh_question(text) is True, f"'{text}' should match WH_QUESTION"

    @pytest.mark.parametrize(
        "text",
        [
            "want to come?",
            "can you make it?",
            "you free?",
            "let's go!",
        ],
    )
    def test_non_wh_questions_dont_match(self, text: str):
        """Non-WH_QUESTION patterns don't match."""
        from jarvis.multi_option import _is_wh_question

        assert _is_wh_question(text) is False, f"'{text}' should NOT match WH_QUESTION"


class TestMultiOptionGenerator:
    """Tests for MultiOptionGenerator class."""

    def test_init_default(self):
        """Can create with default settings."""
        generator = MultiOptionGenerator()
        assert generator._max_options == 3
        assert generator._retriever is None  # Lazy loaded

    def test_init_custom_max_options(self):
        """Can create with custom max_options."""
        generator = MultiOptionGenerator(max_options=5)
        assert generator._max_options == 5

    def test_get_fallback_option(self):
        """_get_fallback_option returns valid option."""
        generator = MultiOptionGenerator()

        option = generator._get_fallback_option(ResponseType.AGREE)
        assert option.response_type == ResponseType.AGREE
        assert option.source == "fallback"
        assert option.confidence == 0.5
        assert option.text in FALLBACK_TEMPLATES[ResponseType.AGREE]

    @patch("jarvis.multi_option.get_typed_retriever")
    def test_is_commitment_trigger_info_statement(self, mock_retriever):
        """INFO_STATEMENT patterns are not commitment triggers."""
        generator = MultiOptionGenerator()

        is_commit, trigger_da = generator.is_commitment_trigger("on my way")
        assert is_commit is False
        assert trigger_da == "statement"

    @patch("jarvis.multi_option.get_typed_retriever")
    def test_is_commitment_trigger_wh_question(self, mock_retriever):
        """WH_QUESTION patterns are not commitment triggers."""
        generator = MultiOptionGenerator()

        is_commit, trigger_da = generator.is_commitment_trigger("who's coming?")
        assert is_commit is False
        assert trigger_da == "question"


class TestGenerateOptions:
    """Tests for generate_options method."""

    @patch("jarvis.multi_option.get_typed_retriever")
    def test_non_commitment_returns_empty_options(self, mock_retriever):
        """Non-commitment triggers return empty options."""
        mock_ret = MagicMock()
        mock_ret.classify_trigger.return_value = ("statement", 0.9)
        mock_retriever.return_value = mock_ret

        generator = MultiOptionGenerator(retriever=mock_ret)
        result = generator.generate_options("I'm on my way")

        assert result.is_commitment is False
        assert result.options == []

    @patch.object(MultiOptionGenerator, "_generate_llm_option", return_value=None)
    @patch("jarvis.embedding_adapter.get_embedder")
    @patch("jarvis.multi_option.get_typed_retriever")
    def test_commitment_returns_options(
        self, mock_retriever, mock_embedder, mock_llm_option
    ):
        """Commitment triggers return multiple options (fallback templates)."""
        # Mock retriever
        mock_ret = MagicMock()
        mock_ret.classify_trigger.return_value = ("commitment", 0.9)

        # Mock examples (empty to trigger fallback)
        mock_examples = MagicMock()
        mock_examples.get_examples.return_value = []
        mock_ret.get_examples_for_commitment.return_value = mock_examples

        mock_retriever.return_value = mock_ret

        # Mock embedder
        mock_emb = MagicMock()
        mock_embedder.return_value = mock_emb

        generator = MultiOptionGenerator(retriever=mock_ret)
        result = generator.generate_options("want to grab lunch?")

        assert result.is_commitment is True
        assert len(result.options) == 3  # AGREE, DECLINE, DEFER (fallback templates)

    @patch.object(MultiOptionGenerator, "_generate_llm_option", return_value=None)
    @patch("jarvis.embedding_adapter.get_embedder")
    @patch("jarvis.multi_option.get_typed_retriever")
    def test_force_commitment_flag(
        self, mock_retriever, mock_embedder, mock_llm_option
    ):
        """force_commitment=True treats any trigger as commitment."""
        mock_ret = MagicMock()
        mock_ret.classify_trigger.return_value = ("statement", 0.9)

        mock_examples = MagicMock()
        mock_examples.get_examples.return_value = []
        mock_ret.get_examples_for_commitment.return_value = mock_examples

        mock_retriever.return_value = mock_ret

        mock_emb = MagicMock()
        mock_embedder.return_value = mock_emb

        generator = MultiOptionGenerator(retriever=mock_ret)
        result = generator.generate_options("random text", force_commitment=True)

        assert result.is_commitment is True
        assert len(result.options) > 0


class TestSingletonFactory:
    """Tests for singleton factory functions."""

    def test_get_multi_option_generator_returns_singleton(self):
        """get_multi_option_generator returns same instance."""
        reset_multi_option_generator()

        gen1 = get_multi_option_generator()
        gen2 = get_multi_option_generator()

        assert gen1 is gen2

    def test_reset_multi_option_generator_clears_singleton(self):
        """reset_multi_option_generator clears the singleton."""
        gen1 = get_multi_option_generator()
        reset_multi_option_generator()
        gen2 = get_multi_option_generator()

        assert gen1 is not gen2


class TestConvenienceFunction:
    """Tests for generate_response_options convenience function."""

    @patch("jarvis.multi_option.get_multi_option_generator")
    def test_uses_singleton(self, mock_get_generator):
        """generate_response_options uses singleton generator."""
        mock_gen = MagicMock()
        mock_result = MultiOptionResult("test", "commitment", True, [])
        mock_gen.generate_options.return_value = mock_result
        mock_get_generator.return_value = mock_gen

        result = generate_response_options("test trigger")

        mock_gen.generate_options.assert_called_once()
        assert result == mock_result


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_trigger(self):
        """Empty trigger handled gracefully."""
        # INFO_STATEMENT check handles empty strings
        from jarvis.multi_option import _is_info_statement, _is_wh_question

        assert _is_info_statement("") is False
        assert _is_wh_question("") is False

    def test_whitespace_trigger(self):
        """Whitespace-only trigger handled gracefully."""
        from jarvis.multi_option import _is_info_statement, _is_wh_question

        assert _is_info_statement("   ") is False
        assert _is_wh_question("   ") is False

    def test_unicode_in_trigger(self):
        """Unicode in trigger handled gracefully."""
        from jarvis.multi_option import _is_info_statement, _is_wh_question

        # Should not raise
        assert _is_info_statement("on my way 🚗") is True
        assert _is_wh_question("who's coming? 🤔") is True
