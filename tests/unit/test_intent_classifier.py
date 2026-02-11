"""Tests for intent classifier backends.

Tests keyword classifier with real inputs, MLX prompt label extraction with
controlled stubs, zero-shot candidate building, factory function routing,
and edge cases.
"""

from __future__ import annotations

import pytest

from jarvis.classifiers.intent_classifier import (
    INTENT_MODEL_ALIASES,
    ZERO_SHOT_LABEL_DESCRIPTIONS,
    HFTransformersIntentClassifier,
    IntentResult,
    KeywordIntentClassifier,
    MLXPromptIntentClassifier,
    _closest_intent_option,
    create_intent_classifier,
)

ALL_INTENTS = list(ZERO_SHOT_LABEL_DESCRIPTIONS.keys())


# ---------------------------------------------------------------------------
# KeywordIntentClassifier - real inputs, real outputs
# ---------------------------------------------------------------------------


class TestKeywordClassifier:
    """Verify keyword classifier picks correct intents from real text."""

    @pytest.fixture()
    def clf(self) -> KeywordIntentClassifier:
        return KeywordIntentClassifier()

    @pytest.mark.parametrize(
        "text,expected_intent",
        [
            # Questions
            ("wya", "reply_question_info"),
            ("wyd", "reply_question_info"),
            ("where are you", "reply_question_info"),
            ("how are you doing", "reply_question_info"),
            # No-reply acknowledgments
            ("ok", "no_reply_ack"),
            ("thanks", "no_reply_ack"),
            ("lol", "no_reply_ack"),
            ("bye", "no_reply_ack"),
            ("k", "no_reply_ack"),
            # Requests
            ("please pick me up", "reply_request_action"),
            ("lmk if you're free", "reply_request_action"),
            # Emotional
            ("omg I can't believe it!!", "reply_emotional_support"),
            ("i love you so much", "reply_emotional_support"),
        ],
    )
    def test_keyword_picks_correct_intent(
        self, clf: KeywordIntentClassifier, text: str, expected_intent: str
    ) -> None:
        candidates = [
            "no_reply_ack",
            "reply_question_info",
            "reply_request_action",
            "reply_emotional_support",
            "reply_casual_chat",
        ]
        result = clf.classify(text, candidates)
        assert result.intent == expected_intent
        assert result.method == "keyword_fallback"
        assert result.confidence > 0.0

    def test_keyword_returns_all_scores(self, clf: KeywordIntentClassifier) -> None:
        candidates = ["no_reply_ack", "reply_question_info", "reply_casual_chat"]
        result = clf.classify("what time is it", candidates)
        assert set(result.all_scores.keys()) == set(candidates)
        # Question should score highest
        assert result.all_scores["reply_question_info"] > result.all_scores["no_reply_ack"]

    def test_keyword_empty_candidates_returns_casual_fallback(
        self, clf: KeywordIntentClassifier
    ) -> None:
        result = clf.classify("hello there", [])
        assert result.intent == "reply_casual_chat"
        assert result.confidence == 0.0

    def test_keyword_casual_chat_is_moderate_baseline(
        self, clf: KeywordIntentClassifier
    ) -> None:
        """Casual chat gets a moderate score when no strong pattern matches."""
        candidates = ["no_reply_ack", "reply_casual_chat"]
        result = clf.classify("hey what's going on with that new show", candidates)
        assert result.all_scores["reply_casual_chat"] == 0.55


# ---------------------------------------------------------------------------
# IntentResult dataclass
# ---------------------------------------------------------------------------


class TestIntentResult:
    def test_fields_populated(self) -> None:
        r = IntentResult(
            intent="reply_question_info",
            confidence=0.88,
            method="keyword_fallback",
            all_scores={"reply_question_info": 0.88},
        )
        assert r.intent == "reply_question_info"
        assert r.confidence == 0.88
        assert r.method == "keyword_fallback"
        assert r.all_scores == {"reply_question_info": 0.88}

    def test_default_all_scores_is_empty_dict(self) -> None:
        r = IntentResult(intent="x", confidence=0.5, method="test")
        assert r.all_scores == {}


# ---------------------------------------------------------------------------
# _closest_intent_option (shared label-matching logic)
# ---------------------------------------------------------------------------


class TestClosestIntentOption:
    def test_exact_match(self) -> None:
        label, score = _closest_intent_option("reply_question_info", ALL_INTENTS)
        assert label == "reply_question_info"
        assert score == 0.88

    def test_exact_match_case_insensitive(self) -> None:
        label, score = _closest_intent_option("REPLY_QUESTION_INFO", ALL_INTENTS)
        assert label == "reply_question_info"
        assert score == 0.88

    def test_partial_match_substring(self) -> None:
        label, score = _closest_intent_option(
            "the intent is reply_question_info here", ALL_INTENTS
        )
        assert label == "reply_question_info"
        assert score == 0.72

    def test_no_match_falls_back_to_first(self) -> None:
        options = ["reply_casual_chat", "no_reply_ack"]
        label, score = _closest_intent_option("complete_garbage_xyz", options)
        assert label == "reply_casual_chat"  # first option
        assert score == 0.50

    def test_empty_options_falls_back_to_casual(self) -> None:
        label, score = _closest_intent_option("anything", [])
        assert label == "reply_casual_chat"
        assert score == 0.50


# ---------------------------------------------------------------------------
# MLXPromptIntentClassifier._extract_label (static, no model needed)
# ---------------------------------------------------------------------------


class TestMLXExtractLabel:
    """Test the label extraction/parsing logic in isolation."""

    def test_exact_label_in_output(self) -> None:
        label, conf = MLXPromptIntentClassifier._extract_label(
            "reply_question_info", ALL_INTENTS
        )
        assert label == "reply_question_info"
        assert conf == 0.88

    def test_label_with_surrounding_whitespace(self) -> None:
        label, conf = MLXPromptIntentClassifier._extract_label(
            "  reply_urgent_action  \n", ALL_INTENTS
        )
        assert label == "reply_urgent_action"
        assert conf == 0.88

    def test_label_wrapped_in_quotes(self) -> None:
        label, conf = MLXPromptIntentClassifier._extract_label(
            '"no_reply_closing"', ALL_INTENTS
        )
        assert label == "no_reply_closing"
        assert conf == 0.88

    def test_label_embedded_in_sentence(self) -> None:
        label, conf = MLXPromptIntentClassifier._extract_label(
            "The label is reply_casual_chat because the message is casual.",
            ALL_INTENTS,
        )
        assert label == "reply_casual_chat"
        assert conf == 0.72

    def test_garbage_output_falls_back_to_first_option(self) -> None:
        label, conf = MLXPromptIntentClassifier._extract_label(
            "I don't understand the question", ALL_INTENTS
        )
        assert label == ALL_INTENTS[0]
        assert conf == 0.50

    def test_empty_output(self) -> None:
        label, conf = MLXPromptIntentClassifier._extract_label("", ALL_INTENTS)
        assert label == ALL_INTENTS[0]
        assert conf == 0.50

    def test_multiline_uses_first_line(self) -> None:
        raw = "reply_question_info\nThis is some explanation on the next line."
        label, conf = MLXPromptIntentClassifier._extract_label(raw, ALL_INTENTS)
        assert label == "reply_question_info"
        assert conf == 0.88

    def test_empty_candidates_returns_casual_default(self) -> None:
        label, conf = MLXPromptIntentClassifier._extract_label("anything", [])
        assert label == "reply_casual_chat"
        assert conf == 0.50


# ---------------------------------------------------------------------------
# _build_zero_shot_candidates
# ---------------------------------------------------------------------------


class TestBuildZeroShotCandidates:
    def test_all_known_intents_get_descriptions(self) -> None:
        candidates, label_to_intent = (
            HFTransformersIntentClassifier._build_zero_shot_candidates(ALL_INTENTS)
        )
        assert len(candidates) == len(ALL_INTENTS)
        # Every candidate is a natural language description, not a raw label
        for c in candidates:
            assert "_" not in c, (
                f"Candidate '{c}' looks like a raw label, not a description"
            )
        # Reverse map covers all
        assert set(label_to_intent.values()) == set(ALL_INTENTS)

    def test_unknown_intent_uses_underscore_replacement(self) -> None:
        candidates, label_to_intent = (
            HFTransformersIntentClassifier._build_zero_shot_candidates(
                ["custom_new_intent"]
            )
        )
        assert candidates == ["custom new intent"]
        assert label_to_intent["custom new intent"] == "custom_new_intent"

    def test_round_trip_mapping(self) -> None:
        candidates, label_to_intent = (
            HFTransformersIntentClassifier._build_zero_shot_candidates(ALL_INTENTS)
        )
        for candidate in candidates:
            intent = label_to_intent[candidate]
            assert intent in ALL_INTENTS


# ---------------------------------------------------------------------------
# create_intent_classifier factory
# ---------------------------------------------------------------------------


class TestCreateIntentClassifier:
    def test_keyword_backend_returns_working_classifier(self) -> None:
        clf = create_intent_classifier("keyword")
        assert isinstance(clf, KeywordIntentClassifier)
        result = clf.classify("wya", ["reply_question_info", "no_reply_ack"])
        assert result.intent == "reply_question_info"

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown intent backend"):
            create_intent_classifier("nope")

    def test_alias_backend_requires_model_alias(self) -> None:
        with pytest.raises(ValueError, match="model_alias is required"):
            create_intent_classifier("alias")

    def test_alias_backend_unknown_alias_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model_alias"):
            create_intent_classifier("alias", model_alias="nonexistent_model")

    def test_keyword_backend_case_insensitive(self) -> None:
        clf = create_intent_classifier("  Keyword  ")
        assert isinstance(clf, KeywordIntentClassifier)


# ---------------------------------------------------------------------------
# INTENT_MODEL_ALIASES coverage
# ---------------------------------------------------------------------------


class TestIntentModelAliases:
    def test_all_aliases_have_required_fields(self) -> None:
        for name, alias in INTENT_MODEL_ALIASES.items():
            assert alias.name == name
            assert alias.env_var.startswith("JARVIS_INTENT_MODEL_")
            assert alias.default_path, f"Alias '{name}' has empty default_path"
            assert alias.preferred_backend in {"hf", "mlx", "mlx_distilbert"}
            assert alias.task in {"sequence", "zero_shot", "seq2seq", "prompt"}

    def test_falconsai_alias_uses_mlx_distilbert(self) -> None:
        alias = INTENT_MODEL_ALIASES["falconsai"]
        assert alias.preferred_backend == "mlx_distilbert"
        assert alias.task == "sequence"

    def test_mlx_aliases_use_mlx_backend(self) -> None:
        for name in ("mlx-qwen-0.5b", "mlx-qwen-1.5b"):
            alias = INTENT_MODEL_ALIASES[name]
            assert alias.preferred_backend == "mlx"
            assert alias.task == "prompt"
