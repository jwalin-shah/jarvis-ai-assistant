from unittest.mock import MagicMock, patch

import pytest
from jarvis.contacts.candidate_extractor import CandidateExtractor, FactCandidate


@pytest.fixture
def extractor():
    with patch("gliner.GLiNER.from_pretrained") as mock_from:
        mock_model = MagicMock()
        mock_from.return_value = mock_model
        ext = CandidateExtractor()
        # Trigger lazy load
        ext._load_model()
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
    assert extractor._to_model_label("person_name") == "person name"
    assert extractor._to_model_label("unknown_label") == "unknown label"

    assert extractor._canonicalize_label("person name") == "person_name"
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
        "I live in Austin and work at Google", message_id=1, use_gate=False,
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


def test_extract_candidates_junk_filter(extractor):
    with patch("jarvis.contacts.candidate_extractor.is_junk_message") as mock_junk:
        mock_junk.return_value = True
        candidates = extractor.extract_candidates("some junk", message_id=1)
        assert len(candidates) == 0
