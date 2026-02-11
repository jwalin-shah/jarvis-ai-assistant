"""Parametrized edge case tests across JARVIS modules.

Tests all key modules with adversarial and boundary inputs:
- Empty strings, None, whitespace-only
- Very long input (10,000+ chars)
- Unicode/emoji/CJK/Arabic
- Single character, numeric-only
- Special characters (null bytes, tabs, newlines)
- HTML/script injection
"""

import pytest

from jarvis.contacts.fact_extractor import FactExtractor
from jarvis.nlp.slang import expand_slang
from jarvis.text_normalizer import (
    extract_temporal_refs,
    extract_text_features,
    get_attachment_token,
    is_acknowledgment_only,
    is_emoji_only,
    is_question,
    is_reaction,
    is_spam_message,
    normalize_text,
    starts_new_topic,
    trigger_expects_content,
)

# ---------------------------------------------------------------------------
# Shared edge case inputs
# ---------------------------------------------------------------------------

EMPTY_INPUTS = ["", "   ", "\t", "\n", "\r\n", "\t\n  \r"]
VERY_LONG_TEXT = "a" * 10_000
VERY_LONG_WORDS = " ".join(["word"] * 5_000)
UNICODE_INPUTS = [
    "\u4f60\u597d\u4e16\u754c",  # Chinese
    "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645",  # Arabic
    "\u3053\u3093\u306b\u3061\u306f\u4e16\u754c",  # Japanese
    "\uc548\ub155\ud558\uc138\uc694",  # Korean
    "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440",  # Russian
    "\U0001f389\U0001f525\U0001f4af\U0001f64c",  # Emoji-only
    "Hello \u4f60\u597d \u0645\u0631\u062d\u0628\u0627",  # Mixed
    "caf\u00e9 r\u00e9sum\u00e9 na\u00efve",  # Accented
]
SPECIAL_CHAR_INPUTS = [
    "\x00",
    "hello\x00world",
    "line1\nline2\nline3",
    "tab\there",
    "back\\slash",
    "\u200b\u200c\u200d",
    "\ufeff",
]
INJECTION_INPUTS = [
    "<script>alert('xss')</script>",
    "'; DROP TABLE pairs; --",
    "<img onerror=alert(1) src=x>",
    "{{template_injection}}",
    "${command_injection}",
    "$(rm -rf /)",
]
SINGLE_CHARS = ["a", "1", "?", "!", ".", " "]
NUMERIC_INPUTS = ["123", "0", "-1", "3.14", "1e10", "999999999"]


# ===========================================================================
# normalize_text edge cases
# ===========================================================================


class TestNormalizeTextEdgeCases:
    """Edge case tests for normalize_text."""

    @pytest.mark.parametrize("text", EMPTY_INPUTS)
    def test_empty_and_whitespace(self, text: str) -> None:
        result = normalize_text(text)
        assert result == "" or result.strip() == ""

    def test_none_returns_empty(self) -> None:
        assert normalize_text(None) == ""  # type: ignore[arg-type]

    def test_very_long_input_no_crash(self) -> None:
        result = normalize_text(VERY_LONG_TEXT)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_very_long_words_no_crash(self) -> None:
        result = normalize_text(VERY_LONG_WORDS)
        assert isinstance(result, str)

    @pytest.mark.parametrize("text", UNICODE_INPUTS)
    def test_unicode_inputs(self, text: str) -> None:
        result = normalize_text(text)
        assert isinstance(result, str)

    @pytest.mark.parametrize("text", SPECIAL_CHAR_INPUTS)
    def test_special_characters_stripped(self, text: str) -> None:
        result = normalize_text(text)
        assert isinstance(result, str)
        assert "\x00" not in result

    @pytest.mark.parametrize("text", INJECTION_INPUTS)
    def test_injection_inputs_no_crash(self, text: str) -> None:
        result = normalize_text(text)
        assert isinstance(result, str)

    @pytest.mark.parametrize("text", SINGLE_CHARS)
    def test_single_character(self, text: str) -> None:
        result = normalize_text(text)
        assert isinstance(result, str)

    @pytest.mark.parametrize("text", NUMERIC_INPUTS)
    def test_numeric_only(self, text: str) -> None:
        result = normalize_text(text)
        assert isinstance(result, str)


# ===========================================================================
# is_reaction edge cases
# ===========================================================================


class TestIsReactionEdgeCases:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("", False),
            ("   ", False),
            ("Liked", False),
            ('Liked "hello"', True),
            ("Liked \u201chello\u201d", True),
            ("a" * 10_000, False),
            ("\u4f60\u597d\u4e16\u754c", False),
            ("<script>alert(1)</script>", False),
            ("Liked \x00", False),
        ],
    )
    def test_parametrized(self, text: str, expected: bool) -> None:
        assert is_reaction(text) is expected

    def test_none_input(self) -> None:
        assert is_reaction(None) is False  # type: ignore[arg-type]


# ===========================================================================
# is_acknowledgment_only edge cases
# ===========================================================================


class TestIsAcknowledgmentEdgeCases:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("", False),
            ("ok", True),
            ("OK", True),
            ("ok.", True),
            ("ok!!!", True),
            ("OKAY", True),
            ("   ok   ", True),
            ("a" * 10_000, False),
            ("ok ok ok", False),
            ("\u4f60\u597d", False),
            ("\x00", False),
        ],
    )
    def test_parametrized(self, text: str, expected: bool) -> None:
        assert is_acknowledgment_only(text) is expected

    def test_none_input(self) -> None:
        assert is_acknowledgment_only(None) is False  # type: ignore[arg-type]


# ===========================================================================
# is_question edge cases
# ===========================================================================


class TestIsQuestionEdgeCases:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("", False),
            ("?", True),
            ("???", True),
            ("What", True),
            ("what time", True),
            ("a" * 10_000 + "?", True),
            ("\u4f60\u597d?", True),
            ("How are you", True),
            ("statement", False),
            (" ", False),
            ("\t?", True),
        ],
    )
    def test_parametrized(self, text: str, expected: bool) -> None:
        assert is_question(text) is expected

    def test_none_input(self) -> None:
        assert is_question(None) is False  # type: ignore[arg-type]


# ===========================================================================
# is_emoji_only edge cases
# ===========================================================================


class TestIsEmojiOnlyEdgeCases:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("", False),
            ("\U0001f389", True),
            ("\U0001f389\U0001f525\U0001f4af", True),
            ("\U0001f389 \U0001f525 \U0001f4af", True),
            ("hello \U0001f389", False),
            ("a", False),
            ("123", False),
            (" ", False),
        ],
    )
    def test_parametrized(self, text: str, expected: bool) -> None:
        assert is_emoji_only(text) is expected

    def test_none_input(self) -> None:
        assert is_emoji_only(None) is False  # type: ignore[arg-type]


# ===========================================================================
# starts_new_topic edge cases
# ===========================================================================


class TestStartsNewTopicEdgeCases:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("", False),
            ("btw something", True),
            ("BTW something", True),
            ("anyway", True),
            ("hello", False),
            ("a" * 10_000, False),
            ("   btw", True),
            ("\tbtw", True),
        ],
    )
    def test_parametrized(self, text: str, expected: bool) -> None:
        assert starts_new_topic(text) is expected

    def test_none_input(self) -> None:
        assert starts_new_topic(None) is False  # type: ignore[arg-type]


# ===========================================================================
# trigger_expects_content edge cases
# ===========================================================================


class TestTriggerExpectsContentEdgeCases:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("", False),
            ("Can you help?", True),
            ("What do you think?", True),
            ("Let me know", True),
            ("ok", False),
            ("a" * 10_000, False),
            ("\u4f60\u80fd\u5e2e\u6211\u5417?", True),
            ("<script>can you</script>", True),
        ],
    )
    def test_parametrized(self, text: str, expected: bool) -> None:
        assert trigger_expects_content(text) is expected

    def test_none_input(self) -> None:
        assert trigger_expects_content(None) is False  # type: ignore[arg-type]


# ===========================================================================
# extract_text_features edge cases
# ===========================================================================


class TestExtractTextFeaturesEdgeCases:
    @pytest.mark.parametrize("text", ["", None])
    def test_empty_or_none(self, text: str) -> None:
        features = extract_text_features(text)  # type: ignore[arg-type]
        assert features.word_count == 0
        assert features.char_count == 0
        assert features.is_reaction is False

    def test_very_long_input(self) -> None:
        features = extract_text_features(VERY_LONG_TEXT)
        assert features.char_count == 10_000
        assert features.word_count == 1

    @pytest.mark.parametrize("text", UNICODE_INPUTS)
    def test_unicode(self, text: str) -> None:
        features = extract_text_features(text)
        assert features.char_count > 0

    @pytest.mark.parametrize("text", INJECTION_INPUTS)
    def test_injection_safety(self, text: str) -> None:
        features = extract_text_features(text)
        assert isinstance(features.word_count, int)


# ===========================================================================
# extract_temporal_refs edge cases
# ===========================================================================


class TestExtractTemporalRefsEdgeCases:
    @pytest.mark.parametrize(
        "text,has_refs",
        [
            ("", False),
            ("today", True),
            ("tomorrow at 5pm", True),
            ("a" * 10_000, False),
            ("\u4f60\u597d\u4e16\u754c", False),
            ("next week we should meet", True),
            ("<script>today</script>", True),
        ],
    )
    def test_parametrized(self, text: str, has_refs: bool) -> None:
        refs = extract_temporal_refs(text)
        assert isinstance(refs, list)
        if has_refs:
            assert len(refs) > 0
        else:
            assert len(refs) == 0

    def test_none_input(self) -> None:
        assert extract_temporal_refs(None) == []  # type: ignore[arg-type]


# ===========================================================================
# is_spam_message edge cases
# ===========================================================================


class TestIsSpamMessageEdgeCases:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("", False),
            ("Your order has shipped", True),
            ("normal chat message", False),
            ("a" * 10_000, False),
            ("text STOP to cancel", True),
            ("50% off limited time offer", True),
        ],
    )
    def test_parametrized(self, text: str, expected: bool) -> None:
        assert is_spam_message(text) is expected

    def test_none_input(self) -> None:
        assert is_spam_message(None) is False  # type: ignore[arg-type]


# ===========================================================================
# get_attachment_token edge cases
# ===========================================================================


class TestGetAttachmentTokenEdgeCases:
    @pytest.mark.parametrize(
        "mime,expected",
        [
            (None, "<ATTACHMENT:file>"),
            ("", "<ATTACHMENT:file>"),
            ("image/jpeg", "<ATTACHMENT:image>"),
            ("video/mp4", "<ATTACHMENT:video>"),
            ("audio/mp3", "<ATTACHMENT:audio>"),
            ("application/pdf", "<ATTACHMENT:pdf>"),
            ("application/octet-stream", "<ATTACHMENT:file>"),
            ("IMAGE/PNG", "<ATTACHMENT:image>"),
            ("totally-invalid", "<ATTACHMENT:file>"),
        ],
    )
    def test_parametrized(self, mime: str | None, expected: str) -> None:
        assert get_attachment_token(mime) == expected


# ===========================================================================
# expand_slang edge cases
# ===========================================================================


class TestExpandSlangEdgeCases:
    @pytest.mark.parametrize(
        "text,check",
        [
            ("", ""),
            ("u coming rn?", "you coming right now?"),
            ("HELLO", "HELLO"),
            ("a" * 10_000, None),
            ("\u4f60\u597d u \u4e16\u754c", None),
            ("u\x00u", None),
            ("\U0001f389 btw \U0001f525", None),
        ],
    )
    def test_parametrized(self, text: str, check: str | None) -> None:
        result = expand_slang(text)
        assert isinstance(result, str)
        if check is not None:
            assert result == check

    def test_none_input(self) -> None:
        result = expand_slang(None)  # type: ignore[arg-type]
        assert result is None

    @pytest.mark.parametrize("text", INJECTION_INPUTS)
    def test_injection_passthrough(self, text: str) -> None:
        result = expand_slang(text)
        assert isinstance(result, str)


# ===========================================================================
# FactExtractor edge cases
# ===========================================================================


class TestFactExtractorEdgeCases:
    """Edge cases for fact extraction pipeline."""

    @pytest.fixture
    def extractor(self) -> FactExtractor:
        return FactExtractor()

    @pytest.mark.parametrize("text", EMPTY_INPUTS)
    def test_empty_messages(self, extractor: FactExtractor, text: str) -> None:
        msgs = [{"text": text, "id": 1}]
        facts = extractor.extract_facts(msgs)
        assert facts == []

    def test_none_text_in_message(self, extractor: FactExtractor) -> None:
        msgs = [{"text": None, "id": 1}]
        facts = extractor.extract_facts(msgs)
        assert facts == []

    def test_very_long_message(self, extractor: FactExtractor) -> None:
        msgs = [{"text": "My sister Sarah " + "x" * 10_000, "id": 1}]
        facts = extractor.extract_facts(msgs)
        assert isinstance(facts, list)

    @pytest.mark.parametrize("text", UNICODE_INPUTS)
    def test_unicode_messages(self, extractor: FactExtractor, text: str) -> None:
        msgs = [{"text": text, "id": 1}]
        facts = extractor.extract_facts(msgs)
        assert isinstance(facts, list)

    @pytest.mark.parametrize("text", INJECTION_INPUTS)
    def test_injection_safety(self, extractor: FactExtractor, text: str) -> None:
        msgs = [{"text": text, "id": 1}]
        facts = extractor.extract_facts(msgs)
        assert isinstance(facts, list)

    def test_empty_message_list(self, extractor: FactExtractor) -> None:
        assert extractor.extract_facts([]) == []

    @pytest.mark.parametrize(
        "text,should_extract",
        [
            ("My sister Sarah lives in Austin", True),
            ("I work at Google", True),
            ("I love eating sushi rolls and hate fresh cilantro leaves", True),
            ("moved to Portland last month", True),
            ("ok", False),
            ("yeah", False),
            ("123", False),
            ("!!!", False),
        ],
    )
    def test_known_patterns(
        self, extractor: FactExtractor, text: str, should_extract: bool
    ) -> None:
        msgs = [{"text": text, "id": 1}]
        facts = extractor.extract_facts(msgs)
        if should_extract:
            assert len(facts) > 0, f"Expected facts from: {text}"
        else:
            assert len(facts) == 0, f"Expected no facts from: {text}"

    def test_numeric_only_message(self, extractor: FactExtractor) -> None:
        msgs = [{"text": "999999", "id": 1}]
        facts = extractor.extract_facts(msgs)
        assert facts == []

    def test_special_chars_message(self, extractor: FactExtractor) -> None:
        msgs = [{"text": "\t\n\r\x00", "id": 1}]
        facts = extractor.extract_facts(msgs)
        assert facts == []

    def test_duplicate_facts_deduped(self, extractor: FactExtractor) -> None:
        """Same fact in two messages is deduplicated."""
        msgs = [
            {"text": "My sister Sarah is great", "id": 1},
            {"text": "My sister Sarah came over", "id": 2},
        ]
        facts = extractor.extract_facts(msgs)
        sarah_facts = [f for f in facts if f.subject == "Sarah"]
        assert len(sarah_facts) <= 1


# ===========================================================================
# FactExtractor quality filter edge cases
# ===========================================================================


class TestFactExtractorQualityFilters:
    """Edge cases for fact quality filters."""

    @pytest.fixture
    def extractor(self) -> FactExtractor:
        return FactExtractor()

    @pytest.mark.parametrize(
        "subject,expected",
        [
            ("it", True),
            ("that", True),
            ("this", True),
            ("them", True),
            ("he", True),
            ("she", True),
            ("me", True),
            ("you", True),
            ("Sarah", False),
            ("Austin", False),
            ("Google", False),
        ],
    )
    def test_vague_subject_detection(
        self, extractor: FactExtractor, subject: str, expected: bool
    ) -> None:
        assert extractor._is_vague_subject(subject) is expected

    @pytest.mark.parametrize(
        "subject,coherent",
        [
            ("Sarah", True),
            ("Google", True),
            ("it", False),
            ("that", False),
            ("ab", True),
            ("1", False),
            ("", False),
            ("it in August", False),
            ("cilantro in my food", True),
        ],
    )
    def test_coherent_subject(self, extractor: FactExtractor, subject: str, coherent: bool) -> None:
        assert extractor._is_coherent_subject(subject) is coherent

    @pytest.mark.parametrize(
        "category,subject,too_short",
        [
            ("preference", "a b", True),
            ("preference", "a b c", False),
            ("relationship", "Sarah", False),
            ("work", "Google", False),
            ("work", "google", True),
            ("location", "Austin", False),
            ("location", "austin", True),
        ],
    )
    def test_too_short_detection(
        self, extractor: FactExtractor, category: str, subject: str, too_short: bool
    ) -> None:
        assert extractor._is_too_short(category, subject) is too_short

    @pytest.mark.parametrize(
        "base_conf,is_vague,is_short,word_count,expected_direction",
        [
            (0.8, True, False, 2, "lower"),
            (0.8, False, True, 2, "lower"),
            (0.8, False, False, 5, "higher_or_equal"),
            (0.5, False, False, 1, "same"),
        ],
    )
    def test_confidence_recalibration(
        self,
        extractor: FactExtractor,
        base_conf: float,
        is_vague: bool,
        is_short: bool,
        word_count: int,
        expected_direction: str,
    ) -> None:
        subject = " ".join(["word"] * word_count)
        result = extractor._calculate_confidence(
            base_conf, "preference", subject, is_vague, is_short
        )
        if expected_direction == "lower":
            assert result < base_conf
        elif expected_direction == "higher_or_equal":
            assert result >= base_conf
        else:
            assert result == base_conf


# ===========================================================================
# ReplyRouter._analyze_complexity static method edge cases
# ===========================================================================


class TestRouterAnalyzeComplexity:
    """Edge cases for the ReplyRouter._analyze_complexity static method."""

    @pytest.mark.parametrize(
        "text,expected_range",
        [
            ("", (0.0, 0.0)),
            ("hi", (0.0, 0.5)),
            ("What time should we meet for dinner tomorrow?", (0.3, 1.0)),
            ("a" * 10_000, (0.5, 1.0)),
            ("\u4f60\u597d\u4e16\u754c test", (0.1, 1.0)),
            ("? ! . , : ;", (0.1, 1.0)),
        ],
    )
    def test_complexity_range(self, text: str, expected_range: tuple) -> None:
        from jarvis.router import ReplyRouter

        score = ReplyRouter._analyze_complexity(text)
        lo, hi = expected_range
        assert lo <= score <= hi, f"Complexity {score} not in [{lo}, {hi}] for '{text[:30]}'"

    def test_returns_float(self) -> None:
        from jarvis.router import ReplyRouter

        assert isinstance(ReplyRouter._analyze_complexity("hello world"), float)

    def test_monotonic_with_length(self) -> None:
        """Longer text should generally produce higher complexity."""
        from jarvis.router import ReplyRouter

        short = ReplyRouter._analyze_complexity("hi")
        long = ReplyRouter._analyze_complexity(
            "What time should we meet for dinner at the restaurant tomorrow evening?"
        )
        assert long > short


# ===========================================================================
# ReplyRouter._build_thread_context edge cases
# ===========================================================================


class TestRouterBuildThreadContext:
    """Edge cases for ReplyRouter._build_thread_context."""

    @pytest.mark.parametrize(
        "messages,expected_none",
        [
            ([], True),
            ([{"text": "", "is_from_me": False}], True),
            ([{"text": "hi", "is_from_me": True}], False),
        ],
    )
    def test_edge_cases(self, messages: list, expected_none: bool) -> None:
        from jarvis.router import ReplyRouter

        result = ReplyRouter._build_thread_context(messages)
        if expected_none:
            assert result is None
        else:
            assert isinstance(result, list)
            assert len(result) > 0

    def test_max_10_messages(self) -> None:
        from jarvis.router import ReplyRouter

        messages = [{"text": f"msg {i}", "is_from_me": False} for i in range(20)]
        result = ReplyRouter._build_thread_context(messages)
        assert result is not None
        assert len(result) <= 10

    def test_unicode_in_thread(self) -> None:
        from jarvis.router import ReplyRouter

        messages = [
            {
                "text": "\u4f60\u597d\u4e16\u754c",
                "is_from_me": False,
                "sender_name": "Alice",
            },
            {"text": "\U0001f389\U0001f525", "is_from_me": True},
        ]
        result = ReplyRouter._build_thread_context(messages)
        assert result is not None
        assert isinstance(result, list)

    def test_is_from_me_prefix(self) -> None:
        from jarvis.router import ReplyRouter

        messages = [
            {"text": "hello", "is_from_me": True},
            {"text": "hey", "is_from_me": False, "sender_name": "Bob"},
        ]
        result = ReplyRouter._build_thread_context(messages)
        assert result is not None
        assert any("Me:" in m for m in result)
        assert any("Bob:" in m for m in result)


# ===========================================================================
# ReplyRouter._to_confidence_label edge cases
# ===========================================================================


class TestRouterConfidenceLabel:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (1.0, "high"),
            (0.7, "high"),
            (0.69, "medium"),
            (0.45, "medium"),
            (0.44, "low"),
            (0.0, "low"),
            (-0.1, "low"),
        ],
    )
    def test_thresholds(self, score: float, expected: str) -> None:
        from jarvis.router import ReplyRouter

        assert ReplyRouter._to_confidence_label(score) == expected
