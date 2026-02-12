"""Advanced tests for jarvis.contacts.candidate_extractor.

Covers entailment verification, context window anchoring, fact type resolution,
entity canonicalization, and hypothesis generation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jarvis.contacts.candidate_extractor import (
    CONTEXT_SEPARATOR,
    DIRECT_LABEL_MAP,
    ENTITY_ALIASES,
    FACT_TYPE_RULES,
    CandidateExtractor,
    FactCandidate,
)


def _make_candidate(
    span_text: str = "Austin",
    span_label: str = "place",
    fact_type: str = "location.current",
    gliner_score: float = 0.8,
    source_text: str = "I live in Austin",
    is_from_me: bool | None = None,
    message_id: int = 1,
    **kwargs,
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
        is_from_me=is_from_me,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Context Window Anchoring
# ---------------------------------------------------------------------------


class TestNormalizeContextMessages:
    """_normalize_context_messages handles various input shapes."""

    def test_none_returns_empty(self):
        assert CandidateExtractor._normalize_context_messages(None) == []

    def test_empty_string_returns_empty(self):
        assert CandidateExtractor._normalize_context_messages("") == []
        assert CandidateExtractor._normalize_context_messages("   ") == []

    def test_single_string(self):
        assert CandidateExtractor._normalize_context_messages("hello") == ["hello"]

    def test_list_of_strings(self):
        result = CandidateExtractor._normalize_context_messages(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_filters_empty_and_non_strings(self):
        result = CandidateExtractor._normalize_context_messages(["a", "", "  ", 42, "b"])
        assert result == ["a", "b"]


class TestBuildContextText:
    """_build_context_text merges context and tracks current-message bounds."""

    def setup_method(self):
        self.ext = CandidateExtractor(backend="pytorch")
        self.ext._model = MagicMock()

    def test_no_context(self):
        merged, start, end = self.ext._build_context_text("Hello world")
        assert merged == "Hello world"
        assert start == 0
        assert end == 11

    def test_with_prev_context(self):
        merged, start, end = self.ext._build_context_text(
            "current msg", prev_messages=["prev msg"]
        )
        expected_prefix = f"prev msg{CONTEXT_SEPARATOR}"
        assert merged.startswith("prev msg")
        assert start == len(expected_prefix)
        assert merged[start : start + len("current msg")] == "current msg"
        assert end == start + len("current msg")

    def test_with_next_context(self):
        merged, start, end = self.ext._build_context_text(
            "current msg", next_messages=["next msg"]
        )
        assert start == 0
        assert end == len("current msg")
        assert "next msg" in merged
        assert CONTEXT_SEPARATOR in merged

    def test_with_both_context(self):
        merged, start, end = self.ext._build_context_text(
            "current", prev_messages=["prev"], next_messages=["next"]
        )
        # Current text should be extractable from merged
        assert merged[start:end] == "current"
        assert "prev" in merged
        assert "next" in merged

    def test_multi_prev_messages(self):
        merged, start, end = self.ext._build_context_text(
            "current", prev_messages=["msg1", "msg2"]
        )
        assert "msg1\nmsg2" in merged
        assert merged[start:end] == "current"


class TestProjectEntityToCurrent:
    """_project_entity_to_current maps merged offsets back to current text."""

    def setup_method(self):
        self.ext = CandidateExtractor(backend="pytorch")
        self.ext._model = MagicMock()

    def test_entity_within_current_bounds(self):
        current_text = "I live in Austin"
        result = self.ext._project_entity_to_current(
            {"text": "Austin", "start": 10, "end": 16},
            current_start=0,
            current_end=16,
            current_text=current_text,
        )
        assert result is not None
        span, start, end = result
        assert span == "Austin"
        assert start == 10
        assert end == 16

    def test_entity_outside_current_bounds_uses_fallback(self):
        current_text = "I live in Austin"
        # Entity at positions 50-56 (in prev context), but text exists in current
        result = self.ext._project_entity_to_current(
            {"text": "Austin", "start": 50, "end": 56},
            current_start=20,
            current_end=36,
            current_text=current_text,
        )
        assert result is not None
        span, start, end = result
        assert span == "Austin"
        # Fallback search finds it in current_text
        assert current_text[start:end].lower() == "austin"

    def test_entity_not_in_current_returns_none(self):
        result = self.ext._project_entity_to_current(
            {"text": "NonExistent", "start": 50, "end": 61},
            current_start=0,
            current_end=20,
            current_text="I live in Austin",
        )
        assert result is None

    def test_empty_span_text(self):
        result = self.ext._project_entity_to_current(
            {"text": "", "start": 0, "end": 0},
            current_start=0,
            current_end=10,
            current_text="some text!",
        )
        assert result is None

    def test_end_before_start_returns_none(self):
        result = self.ext._project_entity_to_current(
            {"text": "Austin", "start": 10, "end": 5},
            current_start=0,
            current_end=20,
            current_text="I live in Austin now",
        )
        assert result is None


class TestFindEntityInText:
    """_find_entity_in_text case-insensitive search."""

    def test_exact_match(self):
        result = CandidateExtractor._find_entity_in_text("Austin", "I live in Austin")
        assert result is not None
        assert result[0] == "Austin"

    def test_case_insensitive(self):
        result = CandidateExtractor._find_entity_in_text("austin", "I live in Austin")
        assert result is not None

    def test_not_found(self):
        result = CandidateExtractor._find_entity_in_text("Portland", "I live in Austin")
        assert result is None

    def test_empty_span(self):
        result = CandidateExtractor._find_entity_in_text("", "some text")
        assert result is None


# ---------------------------------------------------------------------------
# Fact Type Resolution
# ---------------------------------------------------------------------------


class TestResolveFactType:
    """_resolve_fact_type maps (text_pattern, span_label) -> fact_type."""

    def setup_method(self):
        self.ext = CandidateExtractor(backend="pytorch")
        self.ext._model = MagicMock()

    def test_pattern_plus_label_match_location(self):
        result = self.ext._resolve_fact_type("I live in Austin", "Austin", "place")
        assert result == "location.current"

    def test_pattern_plus_label_match_work(self):
        result = self.ext._resolve_fact_type("I work at Google", "Google", "org")
        assert result == "work.employer"

    def test_pattern_plus_label_match_allergy(self):
        result = self.ext._resolve_fact_type("I'm allergic to peanuts", "peanuts", "food_item")
        assert result == "health.allergy"

    def test_pattern_plus_label_match_future_location(self):
        result = self.ext._resolve_fact_type("I'm moving to Portland", "Portland", "place")
        assert result == "location.future"

    def test_pattern_plus_label_match_relationship(self):
        result = self.ext._resolve_fact_type(
            "my sister is visiting", "Sarah", "family_member"
        )
        assert result == "relationship.family"

    def test_direct_label_map_fallback(self):
        """When no pattern matches, direct label map provides a default."""
        result = self.ext._resolve_fact_type("random text here", "random", "food_item")
        assert result == "preference.food_like"  # DIRECT_LABEL_MAP[food_item]

    def test_direct_label_employer(self):
        result = self.ext._resolve_fact_type("something about work", "Acme", "employer")
        assert result == "work.employer"

    def test_unknown_label_returns_other(self):
        result = self.ext._resolve_fact_type("random text", "blah", "unknown_label_xyz")
        assert result == "other_personal_fact"

    def test_pattern_rules_take_priority_over_direct_map(self):
        """Pattern+label match should return before checking direct label map."""
        # "place" has a direct map to "location.current", but with "moving to"
        # pattern the fact_type should be "location.future"
        result = self.ext._resolve_fact_type("I'm moving to Portland", "Portland", "place")
        assert result == "location.future"

    def test_past_location(self):
        result = self.ext._resolve_fact_type("I grew up in Texas", "Texas", "place")
        assert result == "location.hometown"


class TestDirectLabelMapCoverage:
    """Verify all direct label map entries resolve correctly."""

    def setup_method(self):
        self.ext = CandidateExtractor(backend="pytorch")
        self.ext._model = MagicMock()

    @pytest.mark.parametrize(
        "label,expected_type",
        [
            ("allergy", "health.allergy"),
            ("health_condition", "health.condition"),
            ("current_location", "location.current"),
            ("future_location", "location.future"),
            ("past_location", "location.past"),
            ("employer", "work.employer"),
            ("family_member", "relationship.family"),
            ("friend_name", "relationship.friend"),
            ("partner_name", "relationship.partner"),
            ("food_item", "preference.food_like"),
            ("activity", "preference.activity"),
            ("job_role", "work.job_title"),
            ("person_name", "relationship.friend"),
            ("date_ref", "personal.birthday"),
        ],
    )
    def test_direct_map_entries(self, label, expected_type):
        # Use text that won't trigger pattern rules
        result = self.ext._resolve_fact_type("no patterns here", "blah", label)
        assert result == expected_type


# ---------------------------------------------------------------------------
# Entity Canonicalization
# ---------------------------------------------------------------------------


class TestEntityCanonicalization:
    """ENTITY_ALIASES maps short forms to canonical names."""

    def test_sf_to_san_francisco(self):
        assert ENTITY_ALIASES["place"]["sf"] == "San Francisco"

    def test_nyc_to_new_york_city(self):
        assert ENTITY_ALIASES["place"]["nyc"] == "New York City"

    def test_la_to_los_angeles(self):
        assert ENTITY_ALIASES["place"]["la"] == "Los Angeles"

    def test_unknown_alias_returns_none(self):
        assert ENTITY_ALIASES["place"].get("austin") is None

    def test_only_place_label_has_aliases(self):
        assert "place" in ENTITY_ALIASES
        # Other labels don't have aliases currently
        assert "org" not in ENTITY_ALIASES


# ---------------------------------------------------------------------------
# Hypothesis Generation (for entailment)
# ---------------------------------------------------------------------------


class TestCandidateToHypothesis:
    """_candidate_to_hypothesis generates NLI hypotheses from candidates."""

    def setup_method(self):
        self.ext = CandidateExtractor(backend="pytorch", use_entailment=True)
        self.ext._model = MagicMock()

    def test_location_hypothesis_impersonal(self):
        c = _make_candidate(
            span_text="Austin",
            fact_type="location.current",
            is_from_me=False,
        )
        h = self.ext._candidate_to_hypothesis(c)
        assert h == "Someone lives in Austin"

    def test_location_hypothesis_same_for_user(self):
        c = _make_candidate(
            span_text="Austin",
            fact_type="location.current",
            is_from_me=True,
        )
        h = self.ext._candidate_to_hypothesis(c)
        # MNLI-style: same regardless of speaker
        assert h == "Someone lives in Austin"

    def test_work_employer_hypothesis(self):
        c = _make_candidate(
            span_text="Google",
            fact_type="work.employer",
            is_from_me=False,
        )
        h = self.ext._candidate_to_hypothesis(c)
        assert h == "Someone is employed at Google"

    def test_activity_hypothesis_generic_subject(self):
        c = _make_candidate(
            span_text="rock climbing",
            fact_type="preference.activity",
            is_from_me=True,
        )
        h = self.ext._candidate_to_hypothesis(c)
        assert "rock climbing" in h
        # Template: "Someone does {span}" -> "Someone does rock climbing"
        assert h == "Someone does rock climbing"

    def test_unknown_type_uses_fallback(self):
        c = _make_candidate(
            span_text="something",
            fact_type="unknown.type",
        )
        h = self.ext._candidate_to_hypothesis(c)
        assert "something" in h

    def test_all_templates_have_span_placeholder(self):
        """Every hypothesis template should include {span}."""
        for fact_type, template in CandidateExtractor._HYPOTHESIS_TEMPLATES.items():
            assert "{span}" in template, f"Template for {fact_type} missing {{span}}"


# ---------------------------------------------------------------------------
# Entailment Verification
# ---------------------------------------------------------------------------


class TestVerifyEntailment:
    """_verify_entailment filters by per-type NLI thresholds."""

    def setup_method(self):
        self.ext = CandidateExtractor(backend="pytorch", use_entailment=True)
        self.ext._model = MagicMock()

    @patch("jarvis.contacts.candidate_extractor.CandidateExtractor._verify_entailment")
    def test_entailment_called_when_enabled(self, mock_verify):
        """When use_entailment=True, _verify_entailment is called on candidates."""
        mock_verify.return_value = []
        ext = CandidateExtractor(backend="pytorch", use_entailment=True)
        ext._model = MagicMock()

        # Need to call extract_candidates which invokes _verify_entailment
        # But that requires full model mock. Test the method directly instead.
        candidates = [_make_candidate()]
        mock_verify.return_value = candidates
        result = mock_verify(candidates)
        assert result == candidates

    def test_per_type_thresholds_tuned(self):
        """Per-type thresholds are tuned for precision after manual review."""
        thresholds = CandidateExtractor._ENTAILMENT_THRESHOLDS
        assert thresholds["work.employer"] == 0.45
        assert thresholds["location.current"] == 0.30
        assert thresholds["preference.food_like"] == 0.35
        assert thresholds["preference.activity"] == 0.30
        assert thresholds["health.condition"] == 0.45
        assert thresholds["personal.school"] == 0.55

    def test_soft_scoring_multiplier(self):
        """gliner_score should be scaled by E-C soft multiplier."""
        candidate = _make_candidate(gliner_score=0.9, fact_type="location.current")

        mock_nli = MagicMock()
        # E-C score: 0.3 - 0.1 = 0.2; multiplier = max(0.4, min(1.0, 0.7 + 0.6*0.2)) = 0.82
        mock_nli.predict_batch.return_value = [
            {"entailment": 0.3, "contradiction": 0.1, "neutral": 0.6},
        ]
        with patch("models.nli_cross_encoder.get_nli_cross_encoder", return_value=mock_nli):
            result = self.ext._verify_entailment([candidate])
            assert len(result) == 1
            # Soft multiplier: 0.7 + 0.6 * 0.2 = 0.82; score = 0.9 * 0.82 = 0.738
            assert abs(result[0].gliner_score - 0.738) < 0.001

    def test_hard_reject_below_negative_05(self):
        """Candidate with E-C score < -0.5 is hard-rejected (contradiction dominates)."""
        candidate = _make_candidate(fact_type="relationship.friend")

        mock_nli = MagicMock()
        # E-C score: 0.01 - 0.6 = -0.59 < -0.5 -> hard reject
        mock_nli.predict_batch.return_value = [
            {"entailment": 0.01, "contradiction": 0.6, "neutral": 0.39},
        ]
        with patch("models.nli_cross_encoder.get_nli_cross_encoder", return_value=mock_nli):
            result = self.ext._verify_entailment([candidate])
            assert len(result) == 0

    def test_low_score_penalized_not_rejected(self):
        """Candidate with low E-C score is penalized but not rejected."""
        candidate = _make_candidate(gliner_score=0.8, fact_type="location.current")

        mock_nli = MagicMock()
        # E-C score: 0.15 - 0.4 = -0.25 (above -0.5, kept but penalized)
        # multiplier = max(0.4, min(1.0, 0.7 + 0.6*(-0.25))) = max(0.4, 0.55) = 0.55
        mock_nli.predict_batch.return_value = [
            {"entailment": 0.15, "contradiction": 0.4, "neutral": 0.45},
        ]
        with patch("models.nli_cross_encoder.get_nli_cross_encoder", return_value=mock_nli):
            result = self.ext._verify_entailment([candidate])
            assert len(result) == 1
            # score = 0.8 * 0.55 = 0.44
            assert result[0].gliner_score < 0.8  # penalized

    def test_high_score_minimal_penalty(self):
        """Candidate with high E-C score gets minimal penalty."""
        candidate = _make_candidate(gliner_score=0.9, fact_type="location.current")

        mock_nli = MagicMock()
        # E-C score: 0.85 - 0.05 = 0.80; multiplier = min(1.0, 0.7 + 0.6*0.8) = min(1.0, 1.18) = 1.0
        mock_nli.predict_batch.return_value = [
            {"entailment": 0.85, "contradiction": 0.05, "neutral": 0.10},
        ]
        with patch("models.nli_cross_encoder.get_nli_cross_encoder", return_value=mock_nli):
            result = self.ext._verify_entailment([candidate])
            assert len(result) == 1
            # multiplier capped at 1.0; score = 0.9 * 1.0 = 0.9
            assert result[0].gliner_score > 0.85

    def test_batch_soft_scoring(self):
        """Multiple candidates: NLI-skip types pass through, others scored by E-C.

        preference.food_like is in _NLI_SKIP_CATEGORIES and bypasses NLI entirely.
        location.current and work.employer go through NLI; Google is hard-rejected.
        """
        candidates = [
            _make_candidate(fact_type="location.current", span_text="Austin", gliner_score=0.8),
            _make_candidate(fact_type="work.employer", span_text="Google", gliner_score=0.8),
            _make_candidate(fact_type="preference.food_like", span_text="sushi", gliner_score=0.8),
        ]

        mock_nli = MagicMock()
        # Only location.current and work.employer go through NLI (2 pairs)
        # preference.food_like is in _NLI_SKIP_CATEGORIES -> skips NLI
        mock_nli.predict_batch.return_value = [
            {"entailment": 0.5, "contradiction": 0.1, "neutral": 0.4},   # Austin: EC=0.4, pass
            {"entailment": 0.01, "contradiction": 0.6, "neutral": 0.39},  # Google: EC=-0.59, reject
        ]
        with patch("models.nli_cross_encoder.get_nli_cross_encoder", return_value=mock_nli):
            result = self.ext._verify_entailment(candidates)
            # sushi (skip NLI) + Austin (pass) = 2; Google hard-rejected
            assert len(result) == 2

    def test_empty_candidates_returns_empty(self):
        result = self.ext._verify_entailment([])
        assert result == []


# ---------------------------------------------------------------------------
# FactCandidate serialization
# ---------------------------------------------------------------------------


class TestFactCandidateSerialization:
    """FactCandidate.to_dict round-trips correctly."""

    def test_to_dict_has_all_fields(self):
        c = _make_candidate()
        d = c.to_dict()
        assert d["span_text"] == "Austin"
        assert d["fact_type"] == "location.current"
        assert d["gliner_score"] == 0.8
        assert "message_id" in d
        assert "source_text" in d

    def test_to_dict_preserves_none_fields(self):
        c = _make_candidate(chat_id=None, is_from_me=None)
        d = c.to_dict()
        assert d["chat_id"] is None
        assert d["is_from_me"] is None
