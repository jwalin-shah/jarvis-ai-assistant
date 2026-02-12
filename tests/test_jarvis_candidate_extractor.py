from unittest.mock import MagicMock, patch

import pytest

from jarvis.contacts.candidate_extractor import (
    ENTITY_ALIASES,
    VAGUE,
    CandidateExtractor,
    FactCandidate,
)


@pytest.fixture
def extractor():
    mock_model = MagicMock()
    ext = CandidateExtractor(use_entailment=False, backend="pytorch")
    # Inject mock model directly (skips real GLiNER load)
    ext._model = mock_model
    return ext


def test_fact_candidate_to_dict():
    candidate = FactCandidate(
        message_id=123,
        span_text="Austin",
        span_label="place",
        gliner_score=0.9,
        fact_type="location.current",
        start_char=10,
        end_char=16,
        source_text="I live in Austin",
    )
    data = candidate.to_dict()
    assert data["message_id"] == 123
    assert data["span_text"] == "Austin"
    assert data["fact_type"] == "location.current"


def test_label_conversion(extractor):
    assert extractor._to_model_label("person_name") == "first name or nickname of a person"
    assert extractor._to_model_label("unknown_label") == "unknown label"

    assert extractor._canonicalize_label("first name or nickname of a person") == "person_name"
    assert extractor._canonicalize_label("place") == "place"


def test_resolve_fact_type(extractor):
    # Pattern-based
    assert (
        extractor._resolve_fact_type("I am allergic to peanuts", "peanuts", "food_item")
        == "health.allergy"
    )
    assert (
        extractor._resolve_fact_type("I live in New York", "New York", "place")
        == "location.current"
    )
    assert extractor._resolve_fact_type("I work at Google", "Google", "org") == "work.employer"
    assert (
        extractor._resolve_fact_type("my sister Sarah", "Sarah", "person_name")
        == "relationship.family"
    )

    # Direct label map
    assert extractor._resolve_fact_type("random text", "milk", "allergy") == "health.allergy"

    # Fallback
    assert extractor._resolve_fact_type("I saw a dog", "dog", "animal") == "other_personal_fact"


@patch("jarvis.contacts.candidate_extractor.is_junk_message")
def test_extract_candidates(mock_junk, extractor):
    mock_junk.return_value = False

    extractor._model.predict_entities.return_value = [
        {"text": "Austin", "label": "place", "score": 0.9, "start": 10, "end": 16},
        {"text": "Google", "label": "org", "score": 0.8, "start": 25, "end": 31},
    ]

    candidates = extractor.extract_candidates(
        "I live in Austin and work at Google",
        message_id=1,
        use_gate=False,
    )

    assert len(candidates) == 2
    assert candidates[0].span_text == "Austin"
    assert candidates[0].fact_type == "location.current"
    assert candidates[1].span_text == "Google"
    assert candidates[1].fact_type == "work.employer"


def test_extract_candidates_filtering(extractor):
    extractor._model.predict_entities.return_value = [
        {"text": "it", "label": "place", "score": 0.9},  # Vague
        {"text": "Small", "label": "place", "score": 0.1},  # Low score
        {"text": "Austin", "label": "place", "score": 0.9},  # OK
        {"text": "Austin", "label": "place", "score": 0.95},  # Duplicate
    ]

    # Need "live in" pattern so place resolves to location.current (not other_personal_fact)
    candidates = extractor.extract_candidates("I live in Austin", message_id=1, use_gate=False)
    assert len(candidates) == 1
    assert candidates[0].span_text == "Austin"


def test_extract_batch(extractor):
    extractor._model.batch_predict_entities.return_value = [
        [{"text": "Austin", "label": "place", "score": 0.9}],
        [{"text": "Google", "label": "org", "score": 0.8}],
    ]

    messages = [
        {"text": "I live in Austin", "message_id": 1},
        {"text": "I work at Google", "message_id": 2},
    ]

    candidates = extractor.extract_batch(messages)
    assert len(candidates) == 2
    assert candidates[0].span_text == "Austin"
    assert candidates[1].span_text == "Google"


def test_resolve_fact_type_defaults(extractor):
    """place/org/person_name/date_ref fall through to DIRECT_LABEL_MAP defaults."""
    # No pattern match -> falls through to DIRECT_LABEL_MAP
    assert extractor._resolve_fact_type("random text", "Charlotte", "place") == "location.current"
    assert extractor._resolve_fact_type("random text", "Stripe", "org") == "work.employer"
    assert (
        extractor._resolve_fact_type("random text", "Rohith", "person_name")
        == "relationship.friend"
    )
    assert (
        extractor._resolve_fact_type("random text", "March 5th", "date_ref") == "personal.birthday"
    )


def test_resolve_fact_type_patterns_override_defaults(extractor):
    """Regex rules take priority over DIRECT_LABEL_MAP defaults."""
    # "grew up in" -> hometown, NOT the default location.current
    assert (
        extractor._resolve_fact_type("I grew up in Charlotte", "Charlotte", "place")
        == "location.hometown"
    )
    # "go to" -> personal.school, NOT the default work.employer
    assert extractor._resolve_fact_type("I go to Stanford", "Stanford", "org") == "personal.school"


def test_resolve_fact_type_new_patterns(extractor):
    """New regex patterns fire for common chat phrasing."""
    # "already in" -> location.current
    assert extractor._resolve_fact_type("I'm already in sf", "sf", "place") == "location.current"
    # "here in" -> location.current (via "here...in" pattern)
    assert (
        extractor._resolve_fact_type("here in charlotte", "charlotte", "place")
        == "location.current"
    )
    # "born on" -> personal.birthday
    assert (
        extractor._resolve_fact_type("I was born on March 5", "March 5", "date_ref")
        == "personal.birthday"
    )
    # "interning at" -> work.employer
    assert (
        extractor._resolve_fact_type("I'm interning at Google", "Google", "org") == "work.employer"
    )
    # "applied to" -> personal.school
    assert (
        extractor._resolve_fact_type("I applied to Stanford", "Stanford", "org")
        == "personal.school"
    )
    # "flying to" -> location.future
    assert (
        extractor._resolve_fact_type("flying to NYC tomorrow", "NYC", "place") == "location.future"
    )


def test_candidate_to_hypothesis_mnli_style(extractor):
    """Hypotheses use MNLI-style impersonal 'Someone' patterns."""
    base = FactCandidate(
        message_id=1,
        span_text="Austin",
        span_label="place",
        gliner_score=0.9,
        fact_type="location.current",
        start_char=0,
        end_char=6,
        source_text="I live in Austin",
    )

    # Templates are speaker-agnostic (MNLI style)
    base.is_from_me = True
    h = extractor._candidate_to_hypothesis(base)
    assert h == "Someone lives in Austin"

    base.is_from_me = False
    h = extractor._candidate_to_hypothesis(base)
    assert h == "Someone lives in Austin"

    base.is_from_me = None
    h = extractor._candidate_to_hypothesis(base)
    assert h == "Someone lives in Austin"


def test_vague_filter_expanded():
    """Expanded VAGUE set includes pronouns/contractions."""
    for word in ["i", "i'm", "i'll", "i've", "i'd", "we", "we're"]:
        assert word in VAGUE, f"'{word}' should be in VAGUE set"
    # Original words still present
    for word in ["it", "this", "that", "me", "you"]:
        assert word in VAGUE
    # Curly apostrophe variants (iMessage uses U+2019)
    for word in ["i\u2019m", "i\u2019ll", "i\u2019ve", "i\u2019d", "we\u2019re"]:
        assert word in VAGUE, f"curly apostrophe '{word}' should be in VAGUE set"
    # Chat abbreviations
    for word in ["ik", "ai", "boi"]:
        assert word in VAGUE


def test_entity_canonicalization():
    """Common abbreviations are expanded to canonical names."""
    assert ENTITY_ALIASES["place"]["sf"] == "San Francisco"
    assert ENTITY_ALIASES["place"]["nyc"] == "New York City"
    assert ENTITY_ALIASES["place"]["la"] == "Los Angeles"
    # Labels without aliases return nothing
    assert ENTITY_ALIASES.get("org", {}).get("goog") is None


def test_tapback_reactions_filtered():
    """iMessage tapback reactions are junk-filtered."""
    from jarvis.contacts.junk_filters import is_tapback_reaction

    assert is_tapback_reaction("Loved \u201csome message text\u201d")
    assert is_tapback_reaction("Liked \u201cOk sounds good\u201d")
    assert is_tapback_reaction("Emphasized \u201clong term here is a nightmare\u201d")
    assert is_tapback_reaction("Laughed at \u201cDw I disssapoint me too\u201d")
    assert is_tapback_reaction("Disliked \u201csomething\u201d")
    assert is_tapback_reaction("Questioned \u201cwhat?\u201d")
    # Normal messages should not match
    assert not is_tapback_reaction("I loved that movie")
    assert not is_tapback_reaction("Loved it!")
    assert not is_tapback_reaction("She liked the food")


def test_extract_candidates_junk_filter(extractor):
    with patch("jarvis.contacts.candidate_extractor.is_junk_message") as mock_junk:
        mock_junk.return_value = True
        candidates = extractor.extract_candidates("some junk", message_id=1)
        assert len(candidates) == 0
