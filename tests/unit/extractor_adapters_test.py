"""Tests for extractor adapters (spaCy + ensemble)."""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.contacts.extractors import list_extractors
from jarvis.contacts.extractors.base import ExtractedCandidate
from jarvis.contacts.extractors.ensemble_adapter import EnsembleAdapter
from jarvis.contacts.extractors.gliner_adapter import GLiNERAdapter
from jarvis.contacts.extractors.spacy_adapter import SpaCyAdapter


@dataclass
class _FakeEnt:
    text: str
    label_: str
    start_char: int
    end_char: int


class _FakeDoc:
    def __init__(self, ents: list[_FakeEnt]) -> None:
        self.ents = ents


class _FakeNLP:
    def __init__(self, docs_by_text: dict[str, _FakeDoc]) -> None:
        self._docs_by_text = docs_by_text

    def __call__(self, text: str) -> _FakeDoc:
        return self._docs_by_text[text]

    def pipe(self, texts: list[str], batch_size: int = 1):
        del batch_size
        for text in texts:
            yield self._docs_by_text[text]


class _StubAdapter:
    def __init__(self, candidates: list[ExtractedCandidate]) -> None:
        self._candidates = candidates

    def _load_model(self):
        return self

    def extract_from_text(self, *args, **kwargs):
        del args, kwargs
        return list(self._candidates)

    def extract_batch(self, messages, batch_size=32, threshold=None):
        del batch_size, threshold
        return [
            type(
                "Result",
                (),
                {
                    "message_id": msg["message_id"],
                    "candidates": list(self._candidates),
                    "processing_time_ms": 1.0,
                    "error": None,
                },
            )
            for msg in messages
        ]


def _make_candidate(span: str, label: str, score: float, source: str) -> ExtractedCandidate:
    return ExtractedCandidate(
        span_text=span,
        span_label=label,
        score=score,
        start_char=0,
        end_char=len(span),
        fact_type="other_personal_fact",
        extractor_metadata={"source": source},
    )


def test_list_extractors_includes_spacy_and_ensemble() -> None:
    extractors = set(list_extractors())
    assert "spacy" in extractors
    assert "ensemble" in extractors


def test_gliner_adapter_respects_explicit_labels() -> None:
    adapter = GLiNERAdapter(config={"labels": ["family_member", "health_condition"]})
    assert adapter.supported_labels == ["family_member", "health_condition"]


def test_spacy_adapter_maps_entities_and_deduplicates() -> None:
    text = "Sarah works at Google and moved to Austin next week."
    sarah_start = text.index("Sarah")
    google_start = text.index("Google")
    austin_start = text.index("Austin")
    week_start = text.index("next week")

    doc = _FakeDoc(
        [
            _FakeEnt("Sarah", "PERSON", sarah_start, sarah_start + len("Sarah")),
            _FakeEnt("Google", "ORG", google_start, google_start + len("Google")),
            _FakeEnt("Google", "ORG", google_start, google_start + len("Google")),
            _FakeEnt("Austin", "GPE", austin_start, austin_start + len("Austin")),
            _FakeEnt("next week", "DATE", week_start, week_start + len("next week")),
        ]
    )

    adapter = SpaCyAdapter(config={"apply_thresholds": True})
    adapter._nlp = _FakeNLP({text: doc})

    candidates = adapter.extract_from_text(text=text, message_id=1)

    labels = {c.span_label for c in candidates}
    assert labels == {"person_name", "org", "place", "date_ref"}
    assert len(candidates) == 4
    assert all(c.extractor_metadata.get("source") == "spacy" for c in candidates)


def test_spacy_adapter_context_projection_keeps_only_current_message() -> None:
    current = "I live in Austin"
    prev = ["Sarah called yesterday"]
    merged, current_start, _ = SpaCyAdapter._build_context_text(current, context_prev=prev)

    sarah = "Sarah"
    austin = "Austin"
    sarah_start = merged.index(sarah)
    austin_start = merged.index(austin)

    doc = _FakeDoc(
        [
            _FakeEnt(sarah, "PERSON", sarah_start, sarah_start + len(sarah)),
            _FakeEnt(austin, "GPE", austin_start, austin_start + len(austin)),
        ]
    )

    adapter = SpaCyAdapter(config={"apply_thresholds": True})
    adapter._nlp = _FakeNLP({merged: doc})

    candidates = adapter.extract_from_text(
        text=current,
        message_id=11,
        context_prev=prev,
        context_next=None,
    )

    assert len(candidates) == 1
    assert candidates[0].span_text == "Austin"
    assert candidates[0].start_char == current.index("Austin")
    assert candidates[0].fact_type == "location.current"
    assert current_start > 0


def test_ensemble_prefers_higher_score_and_tracks_sources() -> None:
    spacy_candidates = [
        _make_candidate("Sarah", "person_name", 0.82, "spacy"),
        _make_candidate("Google", "org", 0.78, "spacy"),
    ]
    gliner_candidates = [
        _make_candidate("Sarah", "person_name", 0.61, "gliner"),
        _make_candidate("cousin", "family_member", 0.68, "gliner"),
    ]

    adapter = EnsembleAdapter(
        config={
            "spacy_labels": ["person_name", "org"],
            "gliner_labels": ["person_name", "family_member"],
            "prefer_source": "spacy",
        }
    )
    adapter._spacy_adapter = _StubAdapter(spacy_candidates)
    adapter._gliner_adapter = _StubAdapter(gliner_candidates)

    merged = adapter.extract_from_text(text="my cousin Sarah works at Google", message_id=3)

    assert {c.span_label for c in merged} == {"person_name", "org", "family_member"}

    sarah = next(c for c in merged if c.span_label == "person_name")
    assert sarah.score == 0.82
    assert sarah.extractor_metadata["ensemble_sources"] == ["gliner", "spacy"]
