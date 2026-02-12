"""Unit tests for LLMFactVerifier.

All tests mock the generator - no model loading required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from jarvis.contacts.candidate_extractor import FactCandidate
from jarvis.contacts.llm_fact_verifier import _SKIP_CATEGORIES, LLMFactVerifier


def _make_candidate(
    span_text: str = "Austin",
    span_label: str = "place",
    fact_type: str = "location.current",
    message_id: int = 1,
    source_text: str = "I live in Austin",
    gliner_score: float = 0.7,
) -> FactCandidate:
    return FactCandidate(
        message_id=message_id,
        span_text=span_text,
        span_label=span_label,
        gliner_score=gliner_score,
        fact_type=fact_type,
        start_char=0,
        end_char=len(span_text),
        source_text=source_text,
    )


def _mock_generator(response_text: str) -> MagicMock:
    gen = MagicMock()
    resp = MagicMock()
    resp.text = response_text
    gen.generate.return_value = resp
    return gen


# --- Prompt building ---


class TestBuildPrompt:
    def test_single_candidate(self):
        c = _make_candidate()
        prompt = LLMFactVerifier._build_prompt("I live in Austin", [c])
        assert 'Message: "I live in Austin"' in prompt
        assert "1. Austin (place)" in prompt
        assert prompt.endswith("Answers:")

    def test_multiple_candidates(self):
        c1 = _make_candidate(span_text="Austin", span_label="place")
        c2 = _make_candidate(span_text="Google", span_label="org", fact_type="work.employer")
        prompt = LLMFactVerifier._build_prompt("I work at Google in Austin", [c1, c2])
        assert "1. Austin (place)" in prompt
        assert "2. Google (org)" in prompt
        # Candidates should be on single line
        assert "Candidates: 1. Austin (place) 2. Google (org)" in prompt

    def test_special_chars_in_message(self):
        text = 'She said "hello {world}"'
        c = _make_candidate(source_text=text)
        prompt = LLMFactVerifier._build_prompt(text, [c])
        # Braces should be escaped so format() doesn't break
        assert "hello" in prompt

    def test_few_shot_examples_present(self):
        c = _make_candidate()
        prompt = LLMFactVerifier._build_prompt("test", [c])
        assert "my dad has someone who developed Intel chips" in prompt
        assert "Colorado" in prompt
        assert "I work at Google in San Francisco" in prompt
        assert "Answers: 1. YES 2. NO" in prompt
        assert "Answers: 1. YES 2. YES" in prompt

    def test_unsure_keep_instruction(self):
        c = _make_candidate()
        prompt = LLMFactVerifier._build_prompt("test", [c])
        assert "When unsure, answer YES (keep the candidate)." in prompt


# --- Response parsing ---


class TestParseResponse:
    def test_clean_format(self):
        result = LLMFactVerifier._parse_response("1. YES 2. NO 3. YES", 3)
        assert result == [True, False, True]

    def test_newline_format(self):
        result = LLMFactVerifier._parse_response("1. YES\n2. NO", 2)
        assert result == [True, False]

    def test_messy_format(self):
        result = LLMFactVerifier._parse_response("1 -> YES\n2->NO", 2)
        assert result == [True, False]

    def test_partial_output(self):
        # Only first candidate answered - rest default to True (keep)
        result = LLMFactVerifier._parse_response("1. NO", 3)
        assert result == [False, True, True]

    def test_garbage_input(self):
        # Unparseable -> keep all (favor recall)
        result = LLMFactVerifier._parse_response("I don't understand", 2)
        assert result == [True, True]

    def test_empty_response(self):
        result = LLMFactVerifier._parse_response("", 2)
        assert result == [True, True]

    def test_out_of_range_index_ignored(self):
        result = LLMFactVerifier._parse_response("1. YES\n5. NO", 2)
        assert result == [True, True]  # index 5 out of range, ignored

    def test_case_insensitive(self):
        result = LLMFactVerifier._parse_response("1. yes 2. No", 2)
        assert result == [True, False]

    def test_single_candidate_plain_yes(self):
        result = LLMFactVerifier._parse_response("YES", 1)
        assert result == [True]

    def test_single_candidate_plain_no(self):
        result = LLMFactVerifier._parse_response("NO", 1)
        assert result == [False]

    def test_single_candidate_plain_no_case(self):
        result = LLMFactVerifier._parse_response("no", 1)
        assert result == [False]

    def test_plain_no_ignored_for_multiple(self):
        # Plain "NO" without number should NOT affect multi-candidate parsing
        result = LLMFactVerifier._parse_response("NO", 2)
        assert result == [True, True]

    def test_fallback_response_keeps_all(self):
        # Simulate the memory-pressure fallback response
        result = LLMFactVerifier._parse_response(
            "I'm a bit overloaded right now. Give me a moment and try again.", 2
        )
        assert result == [True, True]


# --- Integration with mock generator ---


class TestVerifyCandidates:
    def test_all_yes(self):
        gen = _mock_generator("1. YES")
        verifier = LLMFactVerifier(generator=gen)
        c = _make_candidate()
        result = verifier.verify_candidates([c])
        assert len(result) == 1
        assert result[0].span_text == "Austin"
        gen.generate.assert_called_once()

    def test_all_no(self):
        gen = _mock_generator("1. NO")
        verifier = LLMFactVerifier(generator=gen)
        c = _make_candidate(
            span_text="Intel",
            span_label="org",
            fact_type="work.employer",
            source_text="my dad has someone who developed Intel chips",
        )
        result = verifier.verify_candidates([c])
        assert len(result) == 0

    def test_plain_no_single_candidate(self):
        """Single candidate with plain NO (no number) should be rejected."""
        gen = _mock_generator("NO")
        verifier = LLMFactVerifier(generator=gen)
        c = _make_candidate(
            span_text="Netflix",
            span_label="org",
            fact_type="work.employer",
            source_text="Did you see that new Netflix show?",
        )
        result = verifier.verify_candidates([c])
        assert len(result) == 0

    def test_mixed_verdicts(self):
        gen = _mock_generator("1. YES 2. NO")
        verifier = LLMFactVerifier(generator=gen)
        c1 = _make_candidate(
            span_text="dad", span_label="family_member", fact_type="relationship.family"
        )
        c2 = _make_candidate(span_text="Intel", span_label="org", fact_type="work.employer")
        result = verifier.verify_candidates([c1, c2])
        assert len(result) == 1
        assert result[0].span_text == "dad"

    def test_skip_categories_pass_through(self):
        gen = _mock_generator("")
        verifier = LLMFactVerifier(generator=gen)
        c = _make_candidate(fact_type="preference.activity")
        result = verifier.verify_candidates([c])
        assert len(result) == 1
        # Generator should not be called for skip categories
        gen.generate.assert_not_called()

    def test_empty_input(self):
        gen = _mock_generator("")
        verifier = LLMFactVerifier(generator=gen)
        result = verifier.verify_candidates([])
        assert result == []
        gen.generate.assert_not_called()

    def test_message_grouping(self):
        """Candidates from different messages should trigger separate LLM calls."""
        gen = _mock_generator("1. YES")
        verifier = LLMFactVerifier(generator=gen)
        c1 = _make_candidate(message_id=1, source_text="I live in Austin")
        c2 = _make_candidate(
            message_id=2,
            span_text="Google",
            span_label="org",
            fact_type="work.employer",
            source_text="I work at Google",
        )
        result = verifier.verify_candidates([c1, c2])
        # Two messages -> two LLM calls
        assert gen.generate.call_count == 2
        assert len(result) == 2

    def test_all_skip_categories(self):
        """All candidates in skip categories -> no LLM calls."""
        gen = _mock_generator("")
        verifier = LLMFactVerifier(generator=gen)
        candidates = [_make_candidate(fact_type=ft) for ft in list(_SKIP_CATEGORIES)[:3]]
        result = verifier.verify_candidates(candidates)
        assert len(result) == 3
        gen.generate.assert_not_called()

    def test_mix_skip_and_verify(self):
        """Mix of skip and non-skip categories."""
        gen = _mock_generator("1. YES")
        verifier = LLMFactVerifier(generator=gen)
        skip_c = _make_candidate(fact_type="preference.activity")
        verify_c = _make_candidate(fact_type="location.current")
        result = verifier.verify_candidates([skip_c, verify_c])
        assert len(result) == 2
        gen.generate.assert_called_once()

    def test_generation_request_params(self):
        """Verify correct GenerationRequest parameters."""
        gen = _mock_generator("1. YES 2. NO")
        verifier = LLMFactVerifier(generator=gen)
        c1 = _make_candidate()
        c2 = _make_candidate(span_text="Google", span_label="org", fact_type="work.employer")
        verifier.verify_candidates([c1, c2])

        call_args = gen.generate.call_args
        request = call_args[0][0]
        assert request.max_tokens == 15  # max(15, 5 * 2)
        assert request.temperature == 0.1
        assert request.context_documents == []
        assert request.few_shot_examples == []
        assert "Message:" in request.stop_sequences
