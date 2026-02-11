"""spaCy adapter for extractor bakeoff.

Uses spaCy NER for high-precision standard entities:
- PERSON -> person_name
- ORG -> org
- GPE/LOC/FAC -> place
- DATE/TIME -> date_ref
"""

from __future__ import annotations

import logging
import re
from typing import Any

from jarvis.contacts.extractors.base import (
    ExtractedCandidate,
    ExtractionResult,
    ExtractorAdapter,
    register_extractor,
)
from jarvis.contacts.extractors.gliner_adapter import DIRECT_LABEL_MAP, FACT_TYPE_RULES, VAGUE

logger = logging.getLogger(__name__)

SPACY_TO_CANONICAL: dict[str, str] = {
    "PERSON": "person_name",
    "PER": "person_name",
    "ORG": "org",
    "GPE": "place",
    "LOC": "place",
    "FAC": "place",
    "DATE": "date_ref",
    "TIME": "date_ref",
}

PER_LABEL_MIN: dict[str, float] = {
    "person_name": 0.40,
    "org": 0.50,
    "place": 0.45,
    "date_ref": 0.40,
}

DEFAULT_LABEL_SCORE: dict[str, float] = {
    "person_name": 0.82,
    "org": 0.78,
    "place": 0.78,
    "date_ref": 0.72,
}

CONTEXT_SEPARATOR = "\n[CTX]\n"


class SpaCyAdapter(ExtractorAdapter):
    """Adapter for spaCy NER extraction."""

    SPAN_LABELS = ["person_name", "org", "place", "date_ref"]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("spacy", config)
        self._nlp: Any = None
        self._model_name = self.config.get("model_name", "en_core_web_sm")
        self._global_threshold = float(self.config.get("threshold", 0.0))
        self._apply_thresholds = self.config.get("apply_thresholds", True)
        raw_labels = self.config.get("labels")
        if isinstance(raw_labels, list) and raw_labels:
            self._enabled_labels = {str(label) for label in raw_labels if str(label).strip()}
        else:
            self._enabled_labels = set(self.SPAN_LABELS)
        label_scores = self.config.get("label_scores", {})
        self._label_scores = {
            label: float(label_scores.get(label, DEFAULT_LABEL_SCORE[label]))
            for label in self.SPAN_LABELS
        }

    @property
    def supported_labels(self) -> list[str]:
        return sorted(self._enabled_labels)

    @property
    def default_threshold(self) -> float:
        return self._global_threshold

    def _load_model(self) -> Any:
        """Lazy-load spaCy model."""
        if self._nlp is not None:
            return self._nlp

        try:
            import spacy
        except ImportError as exc:
            raise ImportError(
                "spaCy is not installed. Install it with: uv pip install spacy"
            ) from exc

        try:
            self._nlp = spacy.load(self._model_name)
        except OSError as exc:
            raise ImportError(
                f"spaCy model '{self._model_name}' not found. "
                f"Install it with: python -m spacy download {self._model_name}"
            ) from exc

        logger.info("spaCy adapter loaded: %s", self._model_name)
        return self._nlp

    @staticmethod
    def _build_context_text(
        current_text: str,
        context_prev: list[str] | None = None,
        context_next: list[str] | None = None,
    ) -> tuple[str, int, int]:
        """Build model input text and return current-message character bounds."""
        prev = [p.strip() for p in (context_prev or []) if p.strip()]
        nxt = [n.strip() for n in (context_next or []) if n.strip()]

        if not prev and not nxt:
            return current_text, 0, len(current_text)

        prev_block = "\n".join(prev)
        next_block = "\n".join(nxt)

        if prev_block:
            merged = f"{prev_block}{CONTEXT_SEPARATOR}{current_text}"
            current_start = len(prev_block) + len(CONTEXT_SEPARATOR)
        else:
            merged = current_text
            current_start = 0

        current_end = current_start + len(current_text)
        if next_block:
            merged = f"{merged}{CONTEXT_SEPARATOR}{next_block}"

        return merged, current_start, current_end

    @staticmethod
    def _project_entity_to_current(
        *,
        raw_span: str,
        raw_start: int,
        raw_end: int,
        current_start: int,
        current_end: int,
        current_text: str,
    ) -> tuple[str, int, int] | None:
        """Project an entity span from merged context text to current message offsets."""
        if raw_end <= raw_start:
            return None

        if raw_start < current_start or raw_end > current_end:
            if raw_span:
                idx = current_text.casefold().find(raw_span.casefold())
                if idx >= 0:
                    return raw_span, idx, idx + len(raw_span)
            return None

        start_char = raw_start - current_start
        end_char = raw_end - current_start
        if start_char < 0 or end_char > len(current_text) or end_char <= start_char:
            return None

        span_text = current_text[start_char:end_char].strip()
        if not span_text and raw_span:
            idx = current_text.casefold().find(raw_span.casefold())
            if idx >= 0:
                return raw_span, idx, idx + len(raw_span)
            return None

        if raw_span and span_text.casefold() != raw_span.casefold():
            idx = current_text.casefold().find(raw_span.casefold())
            if idx >= 0:
                return raw_span, idx, idx + len(raw_span)

        return span_text, start_char, end_char

    @staticmethod
    def _resolve_fact_type(text: str, span_label: str) -> str:
        """Map (text_pattern, span_label) -> fact_type."""
        for pattern, label_set, fact_type in FACT_TYPE_RULES:
            if span_label in label_set and re.search(pattern, text, re.IGNORECASE):
                return fact_type

        if span_label in DIRECT_LABEL_MAP:
            return DIRECT_LABEL_MAP[span_label]

        return "other_personal_fact"

    def _extract_from_doc(
        self,
        *,
        text: str,
        doc: Any,
        current_start: int,
        current_end: int,
        apply_thresholds: bool,
    ) -> list[ExtractedCandidate]:
        candidates: list[ExtractedCandidate] = []
        seen: set[tuple[str, str]] = set()

        for ent in doc.ents:
            canonical_label = SPACY_TO_CANONICAL.get(str(ent.label_))
            if canonical_label is None:
                continue
            if canonical_label not in self._enabled_labels:
                continue

            raw_span = str(ent.text).strip()
            projected = self._project_entity_to_current(
                raw_span=raw_span,
                raw_start=int(ent.start_char),
                raw_end=int(ent.end_char),
                current_start=current_start,
                current_end=current_end,
                current_text=text,
            )
            if projected is None:
                continue

            span, start_char, end_char = projected
            if span.casefold() in VAGUE or len(span) < 2:
                continue

            score = self._label_scores.get(canonical_label, 0.7)
            if score < self._global_threshold:
                continue
            if apply_thresholds and score < PER_LABEL_MIN.get(canonical_label, 0.5):
                continue

            dedup_key = (span.casefold(), canonical_label)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            candidates.append(
                ExtractedCandidate(
                    span_text=span,
                    span_label=canonical_label,
                    score=score,
                    start_char=start_char,
                    end_char=end_char,
                    fact_type=self._resolve_fact_type(text, canonical_label),
                    extractor_metadata={
                        "source": "spacy",
                        "spacy_label": str(ent.label_),
                        "model": self._model_name,
                    },
                )
            )

        return candidates

    def extract_from_text(
        self,
        text: str,
        message_id: int,
        *,
        chat_id: int | None = None,
        is_from_me: bool | None = None,
        sender_handle_id: int | None = None,
        message_date: int | None = None,
        threshold: float | None = None,
        context_prev: list[str] | None = None,
        context_next: list[str] | None = None,
    ) -> list[ExtractedCandidate]:
        nlp = self._load_model()
        call_threshold = threshold if threshold is not None else self._global_threshold
        original_threshold = self._global_threshold

        try:
            self._global_threshold = call_threshold
            merged_text, current_start, current_end = self._build_context_text(
                text,
                context_prev=context_prev,
                context_next=context_next,
            )
            doc = nlp(merged_text)
            return self._extract_from_doc(
                text=text,
                doc=doc,
                current_start=current_start,
                current_end=current_end,
                apply_thresholds=self._apply_thresholds,
            )
        finally:
            self._global_threshold = original_threshold

    def extract_batch(
        self,
        messages: list[dict[str, Any]],
        batch_size: int = 64,
        threshold: float | None = None,
    ) -> list[ExtractionResult]:
        import time

        nlp = self._load_model()
        call_threshold = threshold if threshold is not None else self._global_threshold
        original_threshold = self._global_threshold

        merged_texts: list[str] = []
        bounds: list[tuple[int, int]] = []
        source_texts: list[str] = []

        for msg in messages:
            text = msg.get("text", "")
            merged_text, start, end = self._build_context_text(
                text,
                context_prev=msg.get("context_prev"),
                context_next=msg.get("context_next"),
            )
            merged_texts.append(merged_text)
            bounds.append((start, end))
            source_texts.append(text)

        results: list[ExtractionResult] = []
        self._global_threshold = call_threshold
        try:
            start = time.perf_counter()
            docs = list(nlp.pipe(merged_texts, batch_size=max(batch_size, 1)))
            elapsed_ms = (time.perf_counter() - start) * 1000
            per_msg_ms = elapsed_ms / max(len(messages), 1)

            for msg, doc, (current_start, current_end), source_text in zip(
                messages,
                docs,
                bounds,
                source_texts,
            ):
                candidates = self._extract_from_doc(
                    text=source_text,
                    doc=doc,
                    current_start=current_start,
                    current_end=current_end,
                    apply_thresholds=self._apply_thresholds,
                )
                results.append(
                    ExtractionResult(
                        message_id=msg["message_id"],
                        candidates=candidates,
                        extractor_name=self.name,
                        processing_time_ms=per_msg_ms,
                    )
                )
        finally:
            self._global_threshold = original_threshold

        return results


register_extractor("spacy", SpaCyAdapter)
