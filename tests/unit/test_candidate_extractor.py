"""Tests for CandidateExtractor with mocked GLiNER model."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jarvis.contacts.candidate_extractor import (
    VAGUE,
    CandidateExtractor,
    FactCandidate,
    labels_for_profile,
    list_label_profiles,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entity(text, label, score=0.8, start=0, end=None):
    """Build a GLiNER entity dict."""
    return {
        "text": text,
        "label": label,
        "score": score,
        "start": start,
        "end": end if end is not None else start + len(text),
    }


def _mock_extractor(entities: list[dict]) -> CandidateExtractor:
    """Return a CandidateExtractor with a mocked GLiNER model."""
    ext = CandidateExtractor(backend="pytorch", use_entailment=False)
    ext._model = MagicMock()
    ext._model.predict_entities.return_value = entities
    return ext


# ---------------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------------


class TestLabelProfiles:
    def test_list_label_profiles(self):
        profiles = list_label_profiles()
        assert "high_recall" in profiles
        assert "balanced" in profiles
        assert "high_precision" in profiles

    def test_labels_for_profile_balanced_prunes_noisy(self):
        labels = labels_for_profile("balanced")
        assert "person_name" not in labels
        assert "date_ref" not in labels
        assert "family_member" in labels
        assert "org" in labels

    def test_labels_for_profile_unknown_raises(self):
        with pytest.raises(ValueError):
            labels_for_profile("does_not_exist")

    def test_extractor_label_profile_applied(self):
        ext = CandidateExtractor(label_profile="high_precision")
        assert "person_name" not in ext._labels_canonical
        assert "date_ref" not in ext._labels_canonical
        assert "place" in ext._labels_canonical  # re-enabled with high threshold
        assert "org" in ext._labels_canonical


class TestExtractCandidates:
    def test_basic_extraction(self):
        ents = [_make_entity("Austin", "place", 0.85, start=11, end=17)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates(
            "I live in Austin", message_id=1, chat_id=42,
            is_from_me=True, sender_handle_id=5, message_date=700000000,
        )

        assert len(candidates) == 1
        c = candidates[0]
        assert c.span_text == "Austin"
        assert c.span_label == "place"
        assert c.gliner_score == 0.85
        assert c.message_id == 1
        assert c.chat_id == 42
        assert c.is_from_me is True
        assert c.sender_handle_id == 5
        assert c.message_date == 700000000
        assert c.start_char == 10
        assert c.end_char == 16
        assert c.status == "pending"

    def test_multiple_entities(self):
        ents = [
            _make_entity("Google", "org", 0.9, start=10, end=16),
            _make_entity("software engineer", "job_role", 0.7, start=22, end=39),
        ]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates(
            "I work at Google as a software engineer", message_id=2, use_gate=False,
        )
        assert len(candidates) == 2
        assert candidates[0].span_text == "Google"
        assert candidates[1].span_text == "software engineer"


# ---------------------------------------------------------------------------
# Threshold filtering
# ---------------------------------------------------------------------------

class TestThresholdFiltering:
    def test_per_label_threshold_rejects_low_score(self):
        # org requires 0.60, give it 0.40
        ents = [_make_entity("Acme", "org", 0.40)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates("works at Acme", message_id=1, use_gate=False)
        assert len(candidates) == 0

    def test_per_label_threshold_accepts_high_score(self):
        ents = [_make_entity("Acme", "org", 0.65)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates("I work at Acme", message_id=1, use_gate=False)
        assert len(candidates) == 1

    def test_default_threshold_used_for_unknown_label(self):
        # DEFAULT_MIN is 0.55 - use food_item (threshold 0.50) but with score below default
        # Actually test with a label not in PER_LABEL_MIN to hit the default
        ents = [_make_entity("something", "unknown_label", 0.50)]
        ext = _mock_extractor(ents)

        # 0.50 < DEFAULT_MIN (0.55) -> rejected by threshold (before fact_type check)
        candidates = ext.extract_candidates("I enjoy something", message_id=1, use_gate=False)
        assert len(candidates) == 0

    def test_default_threshold_passes(self):
        # Use a label that maps to a fact type (food_item -> preference.food_like)
        ents = [_make_entity("pizza", "food_item", 0.60)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates("I enjoy pizza a lot", message_id=1, use_gate=False)
        assert len(candidates) == 1


# ---------------------------------------------------------------------------
# Vague word rejection
# ---------------------------------------------------------------------------

class TestVagueWordRejection:
    @pytest.mark.parametrize("word", sorted(VAGUE))
    def test_vague_words_rejected(self, word):
        ents = [_make_entity(word, "person_name", 0.9)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates(f"I love {word}", message_id=1, use_gate=False)
        assert len(candidates) == 0

    def test_vague_case_insensitive(self):
        ents = [_make_entity("IT", "org", 0.9)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates("love IT", message_id=1, use_gate=False)
        assert len(candidates) == 0

    def test_single_char_rejected(self):
        ents = [_make_entity("x", "person_name", 0.9)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates("I like x", message_id=1, use_gate=False)
        assert len(candidates) == 0

    def test_non_vague_word_accepted(self):
        ents = [_make_entity("Sarah", "person_name", 0.8)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates("my friend Sarah", message_id=1, use_gate=False)
        assert len(candidates) == 1


# ---------------------------------------------------------------------------
# Junk message filtering
# ---------------------------------------------------------------------------

class TestJunkFiltering:
    def test_bot_message_skipped(self):
        ext = _mock_extractor([_make_entity("CVS", "org", 0.9)])

        candidates = ext.extract_candidates(
            "Your CVS Pharmacy prescription is ready", message_id=1
        )
        assert len(candidates) == 0
        # Model should not even be called
        ext._model.predict_entities.assert_not_called()

    def test_professional_message_skipped(self):
        ext = _mock_extractor([_make_entity("John", "person_name", 0.9)])

        candidates = ext.extract_candidates(
            "Dear John, I hope this finds you well. Best regards, HR", message_id=1
        )
        assert len(candidates) == 0
        ext._model.predict_entities.assert_not_called()

    def test_short_message_skipped(self):
        ext = _mock_extractor([])

        candidates = ext.extract_candidates("hi", message_id=1)
        assert len(candidates) == 0

    def test_empty_message_skipped(self):
        ext = _mock_extractor([])

        candidates = ext.extract_candidates("", message_id=1)
        assert len(candidates) == 0


# ---------------------------------------------------------------------------
# Fact type mapping
# ---------------------------------------------------------------------------

class TestFactTypeMapping:
    def test_location_current(self):
        ents = [_make_entity("Austin", "place", 0.8)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates("I live in Austin now", message_id=1, use_gate=False)
        assert candidates[0].fact_type == "location.current"

    def test_location_future(self):
        ents = [_make_entity("LA", "future_location", 0.7)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates(
            "moving to LA next month", message_id=1, use_gate=False,
        )
        assert candidates[0].fact_type == "location.future"

    def test_work_employer(self):
        ents = [_make_entity("Google", "org", 0.9)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates("I work at Google", message_id=1, use_gate=False)
        assert candidates[0].fact_type == "work.employer"

    def test_relationship_family(self):
        ents = [_make_entity("Sarah", "person_name", 0.8)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates(
            "my sister Sarah is visiting", message_id=1, use_gate=False,
        )
        assert candidates[0].fact_type == "relationship.family"

    def test_relationship_partner(self):
        ents = [_make_entity("Alex", "person_name", 0.8)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates(
            "my girlfriend Alex", message_id=1, use_gate=False,
        )
        assert candidates[0].fact_type == "relationship.partner"

    def test_preference_food_like(self):
        ents = [_make_entity("Thai food", "food_item", 0.7)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates(
            "I love Thai food", message_id=1, use_gate=False,
        )
        assert candidates[0].fact_type == "preference.food_like"

    def test_preference_food_dislike(self):
        ents = [_make_entity("cilantro", "food_item", 0.7)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates(
            "I hate cilantro", message_id=1, use_gate=False,
        )
        assert candidates[0].fact_type == "preference.food_dislike"

    def test_health_allergy(self):
        ents = [_make_entity("peanuts", "allergy", 0.7)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates(
            "I'm allergic to peanuts", message_id=1, use_gate=False,
        )
        assert candidates[0].fact_type == "health.allergy"

    def test_personal_school(self):
        ents = [_make_entity("MIT", "org", 0.8)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates(
            "I graduated from MIT", message_id=1, use_gate=False,
        )
        assert candidates[0].fact_type == "personal.school"

    def test_personal_pet(self):
        ents = [_make_entity("Buddy", "person_name", 0.7)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates(
            "my dog Buddy is adorable", message_id=1, use_gate=False,
        )
        assert candidates[0].fact_type == "personal.pet"

    def test_other_personal_fact_filtered_out(self):
        """Spans with unknown labels resolve to other_personal_fact and are dropped."""
        ents = [_make_entity("Fluffy", "animal", 0.8)]
        ext = _mock_extractor(ents)

        # Unknown label "animal" -> other_personal_fact -> filtered
        candidates = ext.extract_candidates(
            "Fluffy ran around the yard", message_id=1, use_gate=False,
        )
        assert len(candidates) == 0

    def test_person_name_defaults_to_friend(self):
        """person_name without pattern match falls back to relationship.friend."""
        ents = [_make_entity("Sarah", "person_name", 0.8)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates(
            "Sarah told me about it", message_id=1, use_gate=False,
        )
        assert len(candidates) == 1
        assert candidates[0].fact_type == "relationship.friend"

    def test_direct_label_map_no_pattern_needed(self):
        """Fact-like labels map directly even without pattern match."""
        ents = [_make_entity("Acme Corp", "employer", 0.7)]
        ext = _mock_extractor(ents)

        # No "work at" pattern, but label is "employer"
        candidates = ext.extract_candidates(
            "Acme Corp laid off 200 people", message_id=1, use_gate=False,
        )
        assert candidates[0].fact_type == "work.employer"

    def test_direct_label_map_health_condition(self):
        """health_condition maps directly to health.condition."""
        ents = [_make_entity("migraines", "health_condition", 0.7)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates(
            "I deal with migraines often", message_id=1, use_gate=False,
        )
        assert len(candidates) == 1
        assert candidates[0].fact_type == "health.condition"

    def test_direct_label_map_food_item(self):
        """food_item maps directly to preference.food_like."""
        ents = [_make_entity("sushi", "food_item", 0.7)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates(
            "Had sushi for dinner", message_id=1, use_gate=False,
        )
        assert len(candidates) == 1
        assert candidates[0].fact_type == "preference.food_like"

    def test_direct_label_map_activity(self):
        """activity maps directly to preference.activity."""
        ents = [_make_entity("surfing", "activity", 0.8)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates(
            "Went surfing this weekend", message_id=1, use_gate=False,
        )
        assert len(candidates) == 1
        assert candidates[0].fact_type == "preference.activity"

    def test_direct_label_map_job_role(self):
        """job_role maps directly to work.job_title."""
        ents = [_make_entity("engineer", "job_role", 0.8)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates(
            "She is an engineer at the company", message_id=1, use_gate=False,
        )
        assert len(candidates) == 1
        assert candidates[0].fact_type == "work.job_title"

    def test_org_without_pattern_defaults_to_employer(self):
        """org spans without a matching pattern fall back to work.employer."""
        ents = [_make_entity("Netflix", "org", 0.9)]
        ext = _mock_extractor(ents)

        # No work/school pattern in text -> DIRECT_LABEL_MAP fallback
        candidates = ext.extract_candidates(
            "Netflix released a new show", message_id=1, use_gate=False,
        )
        assert len(candidates) == 1
        assert candidates[0].fact_type == "work.employer"

    def test_former_employer(self):
        ents = [_make_entity("Meta", "org", 0.8)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates(
            "I quit Meta last year", message_id=1, use_gate=False,
        )
        assert candidates[0].fact_type == "work.former_employer"


# ---------------------------------------------------------------------------
# Strict dedup
# ---------------------------------------------------------------------------

class TestStrictDedup:
    def test_duplicate_span_label_deduplicated(self):
        """Same span_text + label on same message should emit once."""
        ents = [
            _make_entity("Austin", "place", 0.8, start=0, end=6),
            _make_entity("Austin", "place", 0.7, start=30, end=36),
        ]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates(
            "Austin is great, I live in Austin", message_id=1, use_gate=False,
        )
        assert len(candidates) == 1

    def test_same_span_different_label_not_deduplicated(self):
        """Same text but different label should both be kept when both resolve."""
        ents = [
            _make_entity("Google", "org", 0.8),
            _make_entity("Google", "employer", 0.7),
        ]
        ext = _mock_extractor(ents)

        # "work at" matches org -> work.employer, "employer" direct maps -> work.employer
        # But same span+label dedup catches "Google"+"org" and "Google"+"employer" separately
        candidates = ext.extract_candidates(
            "I work at Google and Google is great", message_id=1, use_gate=False,
        )
        assert len(candidates) == 2

    def test_case_insensitive_dedup(self):
        """Dedup should be case-insensitive."""
        ents = [
            _make_entity("sushi", "food_item", 0.8),
            _make_entity("Sushi", "food_item", 0.7),
        ]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates(
            "sushi vs Sushi", message_id=1, use_gate=False,
        )
        assert len(candidates) == 1

    def test_whitespace_trimmed_dedup(self):
        """Leading/trailing whitespace should not prevent dedup."""
        ents = [
            _make_entity(" sushi ", "food_item", 0.8),
            _make_entity("sushi", "food_item", 0.7),
        ]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates(
            "I love sushi so much", message_id=1, use_gate=False,
        )
        assert len(candidates) == 1


# ---------------------------------------------------------------------------
# Character offsets
# ---------------------------------------------------------------------------

class TestCharOffsets:
    def test_offsets_from_gliner(self):
        ents = [_make_entity("Google", "org", 0.9, start=10, end=16)]
        ext = _mock_extractor(ents)

        candidates = ext.extract_candidates(
            "I work at Google", message_id=1, use_gate=False,
        )
        assert candidates[0].start_char == 10
        assert candidates[0].end_char == 16

    def test_offsets_default_when_missing(self):
        """If GLiNER omits offsets, recover offsets by matching span text."""
        ent = {"text": "Google", "label": "org", "score": 0.9}
        ext = _mock_extractor([ent])

        candidates = ext.extract_candidates(
            "I work at Google", message_id=1, use_gate=False,
        )
        assert candidates[0].start_char == 10
        assert candidates[0].end_char == 16


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

class TestBatchExtraction:
    def test_batch_basic(self):
        ext = CandidateExtractor()
        ext._model = MagicMock()
        ext._model.batch_predict_entities.return_value = [
            [_make_entity("Austin", "place", 0.8)],
            [_make_entity("Google", "org", 0.9)],
        ]

        messages = [
            {"text": "I live in Austin", "message_id": 1, "chat_id": 10},
            {"text": "I work at Google", "message_id": 2, "chat_id": 10},
        ]

        candidates = ext.extract_batch(messages, batch_size=32)
        assert len(candidates) == 2
        assert candidates[0].span_text == "Austin"
        assert candidates[0].chat_id == 10
        assert candidates[1].span_text == "Google"
        assert candidates[1].message_id == 2

    def test_batch_skips_junk(self):
        ext = CandidateExtractor()
        ext._model = MagicMock()
        # Only one valid message passes junk filter
        ext._model.batch_predict_entities.return_value = [
            [_make_entity("Austin", "place", 0.8)],
        ]

        messages = [
            {"text": "Your CVS Pharmacy prescription is ready", "message_id": 1},
            {"text": "I live in Austin", "message_id": 2},
        ]

        candidates = ext.extract_batch(messages, batch_size=32)
        assert len(candidates) == 1
        assert candidates[0].message_id == 2

    def test_batch_preserves_metadata(self):
        ext = CandidateExtractor()
        ext._model = MagicMock()
        ext._model.batch_predict_entities.return_value = [
            [_make_entity("Austin", "place", 0.8)],
        ]

        messages = [
            {
                "text": "I live in Austin",
                "message_id": 42,
                "chat_id": 7,
                "is_from_me": True,
                "sender_handle_id": 99,
                "message_date": 700000000,
            },
        ]

        candidates = ext.extract_batch(messages, batch_size=32)
        assert len(candidates) == 1
        c = candidates[0]
        assert c.message_id == 42
        assert c.chat_id == 7
        assert c.is_from_me is True
        assert c.sender_handle_id == 99
        assert c.message_date == 700000000


# ---------------------------------------------------------------------------
# Context window anchoring
# ---------------------------------------------------------------------------

class TestContextWindowAnchoring:
    def test_extract_candidates_context_keeps_only_current_spans(self):
        ext = CandidateExtractor(backend="pytorch")
        ext._model = MagicMock()

        current = "moving to Austin next month"
        prev = ["I live in Dallas now"]
        nxt = ["that sounds great"]
        _, current_start, _ = ext._build_context_text(
            current, prev_messages=prev, next_messages=nxt
        )

        austin_start = current.index("Austin")
        ext._model.predict_entities.return_value = [
            _make_entity("Dallas", "place", 0.95, start=10, end=16),  # previous context
            _make_entity(
                "Austin",
                "place",
                0.95,
                start=current_start + austin_start,
                end=current_start + austin_start + len("Austin"),
            ),
        ]

        candidates = ext.extract_candidates(
            current,
            message_id=1,
            threshold=0.1,
            apply_label_thresholds=False,
            prev_messages=prev,
            next_messages=nxt,
            use_gate=False,
        )

        assert len(candidates) == 1
        assert candidates[0].span_text == "Austin"
        assert candidates[0].start_char == austin_start
        assert candidates[0].end_char == austin_start + len("Austin")

    def test_predict_raw_entities_context_projects_offsets(self):
        ext = CandidateExtractor(backend="pytorch")
        ext._model = MagicMock()

        current = "I moved to Seattle"
        prev = ["I used to live in Chicago"]
        _, current_start, _ = ext._build_context_text(
            current,
            prev_messages=prev,
            next_messages=None,
        )
        local_start = current.index("Seattle")
        ext._model.predict_entities.return_value = [
            _make_entity("Chicago", "place", 0.9, start=15, end=22),
            _make_entity(
                "Seattle",
                "place",
                0.9,
                start=current_start + local_start,
                end=current_start + local_start + len("Seattle"),
            ),
        ]

        entities = ext.predict_raw_entities(current, threshold=0.1, prev_messages=prev)
        assert len(entities) == 1
        assert entities[0]["text"] == "Seattle"
        assert entities[0]["start"] == local_start
        assert entities[0]["end"] == local_start + len("Seattle")

    def test_extract_batch_respects_context_prev_next(self):
        ext = CandidateExtractor(backend="pytorch")
        ext._model = MagicMock()

        # Messages must be same length so length-sort preserves original order
        m1_text = "I work at Google now"  # 20 chars
        m1_prev = ["My cousin moved to Austin"]
        _, m1_start, _ = ext._build_context_text(m1_text, prev_messages=m1_prev, next_messages=[])
        google_local = m1_text.index("Google")

        m2_text = "I really enjoy sushi"  # 20 chars (same length as m1)
        m2_next = ["my brother hates fish"]
        _, m2_start, _ = ext._build_context_text(m2_text, prev_messages=[], next_messages=m2_next)
        sushi_local = m2_text.index("sushi")

        # After length-sort, both texts are 20 chars so order is stable
        ext._model.batch_predict_entities.return_value = [
            [
                _make_entity("Austin", "place", 0.9, start=8, end=14),  # previous context
                _make_entity(
                    "Google",
                    "org",
                    0.95,
                    start=m1_start + google_local,
                    end=m1_start + google_local + len("Google"),
                ),
            ],
            [
                _make_entity(
                    "sushi",
                    "food_item",
                    0.95,
                    start=m2_start + sushi_local,
                    end=m2_start + sushi_local + len("sushi"),
                ),
            ],
        ]

        messages = [
            {
                "text": m1_text,
                "message_id": 1,
                "context_prev": m1_prev,
                "context_next": [],
            },
            {
                "text": m2_text,
                "message_id": 2,
                "context_prev": [],
                "context_next": m2_next,
            },
        ]

        candidates = ext.extract_batch(messages, batch_size=8)
        assert len(candidates) == 2
        assert {c.span_text for c in candidates} == {"Google", "sushi"}


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_roundtrip(self):
        c = FactCandidate(
            message_id=1, span_text="Austin", span_label="place",
            gliner_score=0.85, fact_type="location.current",
            start_char=10, end_char=16,
            source_text="I live in Austin", chat_id=42,
            is_from_me=True, sender_handle_id=5, message_date=700000000,
        )
        d = c.to_dict()

        assert d["message_id"] == 1
        assert d["span_text"] == "Austin"
        assert d["start_char"] == 10
        assert d["end_char"] == 16
        assert d["is_from_me"] is True
        assert d["sender_handle_id"] == 5
        assert d["message_date"] == 700000000
        assert d["status"] == "pending"

    def test_to_dict_none_fields(self):
        c = FactCandidate(
            message_id=1, span_text="X", span_label="org",
            gliner_score=0.5, fact_type="other_personal_fact",
            start_char=0, end_char=1,
        )
        d = c.to_dict()
        assert d["chat_id"] is None
        assert d["is_from_me"] is None
        assert d["sender_handle_id"] is None
        assert d["message_date"] is None
