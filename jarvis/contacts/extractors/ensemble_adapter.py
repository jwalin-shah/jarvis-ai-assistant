"""Hybrid ensemble adapter combining spaCy and GLiNER.

Design:
- GLiNER is the primary extractor (all stage-1 labels by default)
- spaCy is optional, conservative augmentation (place by default)
- Results are merged with strict dedup by (span_text, span_label)
"""

from __future__ import annotations

import logging
from typing import Any

from jarvis.contacts.extractors.base import (
    ExtractedCandidate,
    ExtractionResult,
    ExtractorAdapter,
    register_extractor,
)
from jarvis.contacts.extractors.gliner_adapter import GLiNERAdapter
from jarvis.contacts.extractors.regex_adapter import RegexAdapter
from jarvis.contacts.extractors.spacy_adapter import SpaCyAdapter

logger = logging.getLogger(__name__)

DEFAULT_SPACY_LABELS = {"place"}
DEFAULT_REGEX_LABELS = {"email", "phone_number"}
DEFAULT_GLINER_LABELS = {
    "person_name",
    "family_member",
    "place",
    "org",
    "date_ref",
    "food_item",
    "job_role",
    "health_condition",
    "activity",
}


class EnsembleAdapter(ExtractorAdapter):
    """Adapter that ensembles spaCy, GLiNER, and Regex outputs."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("ensemble", config)
        self._spacy_adapter: SpaCyAdapter | None = None
        self._gliner_adapter: GLiNERAdapter | None = None
        self._regex_adapter: RegexAdapter | None = None

        self._global_threshold = float(self.config.get("threshold", 0.15))
        self._spacy_labels = set(self.config.get("spacy_labels", sorted(DEFAULT_SPACY_LABELS)))
        self._gliner_labels = set(self.config.get("gliner_labels", sorted(DEFAULT_GLINER_LABELS)))
        self._regex_labels = set(self.config.get("regex_labels", sorted(DEFAULT_REGEX_LABELS)))
        self._prefer_source = str(self.config.get("prefer_source", "gliner"))

    @property
    def supported_labels(self) -> list[str]:
        return sorted(self._spacy_labels | self._gliner_labels | self._regex_labels)

    @property
    def default_threshold(self) -> float:
        return self._global_threshold

    def _build_spacy_config(self) -> dict[str, Any]:
        cfg = dict(self.config.get("spacy_config", {}))
        cfg.setdefault("threshold", self._global_threshold)
        cfg.setdefault("apply_thresholds", self.config.get("apply_thresholds", True))
        cfg.setdefault("labels", sorted(self._spacy_labels))
        return cfg

    def _build_gliner_config(self) -> dict[str, Any]:
        cfg = dict(self.config.get("gliner_config", {}))
        cfg.setdefault("threshold", self._global_threshold)
        cfg.setdefault("apply_thresholds", self.config.get("apply_thresholds", True))
        cfg.setdefault("label_profile", self.config.get("label_profile", "high_recall"))
        cfg.setdefault("labels", sorted(self._gliner_labels))
        return cfg

    def _build_regex_config(self) -> dict[str, Any]:
        cfg = dict(self.config.get("regex_config", {}))
        cfg.setdefault("labels", sorted(self._regex_labels))
        return cfg

    def _load_model(self) -> tuple[SpaCyAdapter, GLiNERAdapter, RegexAdapter]:
        """Load underlying adapters."""
        if self._spacy_adapter is None:
            self._spacy_adapter = SpaCyAdapter(config=self._build_spacy_config())
        if self._gliner_adapter is None:
            self._gliner_adapter = GLiNERAdapter(config=self._build_gliner_config())
        if self._regex_adapter is None:
            self._regex_adapter = RegexAdapter(config=self._build_regex_config())

        self._spacy_adapter._load_model()
        self._gliner_adapter._load_model()
        self._regex_adapter._load_model()
        return self._spacy_adapter, self._gliner_adapter, self._regex_adapter

    @staticmethod
    def _dedup_key(candidate: ExtractedCandidate) -> tuple[str, str]:
        return candidate.span_text.casefold(), candidate.span_label

    def _merge_candidates(
        self,
        spacy_candidates: list[ExtractedCandidate],
        gliner_candidates: list[ExtractedCandidate],
        regex_candidates: list[ExtractedCandidate] = None,
    ) -> list[ExtractedCandidate]:
        merged: dict[tuple[str, str], ExtractedCandidate] = {}
        regex_candidates = regex_candidates or []

        # Ordering of sources determines priority for the base candidate data
        # Regex always has highest priority if present
        ordered = [("regex", regex_candidates)]
        if self._prefer_source == "gliner":
            ordered.extend([("gliner", gliner_candidates), ("spacy", spacy_candidates)])
        else:
            ordered.extend([("spacy", spacy_candidates), ("gliner", gliner_candidates)])

        for source_name, source_candidates in ordered:
            for candidate in source_candidates:
                if candidate.span_label not in self.supported_labels:
                    continue

                key = self._dedup_key(candidate)
                if key not in merged:
                    candidate.extractor_metadata = dict(candidate.extractor_metadata)
                    candidate.extractor_metadata["ensemble_sources"] = [source_name]
                    merged[key] = candidate
                    continue

                existing = merged[key]
                sources = set(existing.extractor_metadata.get("ensemble_sources", []))
                sources.add(source_name)
                existing.extractor_metadata = dict(existing.extractor_metadata)
                existing.extractor_metadata["ensemble_sources"] = sorted(sources)

        return list(merged.values())

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
        spacy_adapter, gliner_adapter, regex_adapter = self._load_model()

        spacy_candidates = spacy_adapter.extract_from_text(
            text=text,
            message_id=message_id,
            chat_id=chat_id,
            is_from_me=is_from_me,
            sender_handle_id=sender_handle_id,
            message_date=message_date,
            threshold=threshold,
            context_prev=context_prev,
            context_next=context_next,
        )
        spacy_candidates = [c for c in spacy_candidates if c.span_label in self._spacy_labels]

        gliner_candidates = gliner_adapter.extract_from_text(
            text=text,
            message_id=message_id,
            chat_id=chat_id,
            is_from_me=is_from_me,
            sender_handle_id=sender_handle_id,
            message_date=message_date,
            threshold=threshold,
            context_prev=context_prev,
            context_next=context_next,
        )
        gliner_candidates = [c for c in gliner_candidates if c.span_label in self._gliner_labels]

        regex_candidates = regex_adapter.extract_from_text(
            text=text,
            message_id=message_id,
        )
        regex_candidates = [c for c in regex_candidates if c.span_label in self._regex_labels]

        return self._merge_candidates(spacy_candidates, gliner_candidates, regex_candidates)

    def extract_batch(
        self,
        messages: list[dict[str, Any]],
        batch_size: int = 32,
        threshold: float | None = None,
    ) -> list[ExtractionResult]:
        spacy_adapter, gliner_adapter, regex_adapter = self._load_model()

        spacy_results = spacy_adapter.extract_batch(
            messages=messages,
            batch_size=batch_size,
            threshold=threshold,
        )
        gliner_results = gliner_adapter.extract_batch(
            messages=messages,
            batch_size=batch_size,
            threshold=threshold,
        )
        regex_results = regex_adapter.extract_batch(
            messages=messages,
            batch_size=batch_size,
        )

        spacy_by_id = {r.message_id: r for r in spacy_results}
        gliner_by_id = {r.message_id: r for r in gliner_results}
        regex_by_id = {r.message_id: r for r in regex_results}

        combined: list[ExtractionResult] = []
        for msg in messages:
            msg_id = msg["message_id"]
            spacy_result = spacy_by_id.get(msg_id)
            gliner_result = gliner_by_id.get(msg_id)
            regex_result = regex_by_id.get(msg_id)

            spacy_candidates = (
                [c for c in spacy_result.candidates if c.span_label in self._spacy_labels]
                if spacy_result
                else []
            )
            gliner_candidates = (
                [c for c in gliner_result.candidates if c.span_label in self._gliner_labels]
                if gliner_result
                else []
            )
            regex_candidates = (
                [c for c in regex_result.candidates if c.span_label in self._regex_labels]
                if regex_result
                else []
            )

            merged_candidates = self._merge_candidates(
                spacy_candidates, gliner_candidates, regex_candidates
            )
            total_ms = (
                (spacy_result.processing_time_ms if spacy_result else 0.0)
                + (gliner_result.processing_time_ms if gliner_result else 0.0)
                + (regex_result.processing_time_ms if regex_result else 0.0)
            )

            errors = []
            if spacy_result and spacy_result.error:
                errors.append(f"spacy: {spacy_result.error}")
            if gliner_result and gliner_result.error:
                errors.append(f"gliner: {gliner_result.error}")
            if regex_result and regex_result.error:
                errors.append(f"regex: {regex_result.error}")

            combined.append(
                ExtractionResult(
                    message_id=msg_id,
                    candidates=merged_candidates,
                    extractor_name=self.name,
                    processing_time_ms=total_ms,
                    error="; ".join(errors) if errors else None,
                )
            )

        return combined


register_extractor("ensemble", EnsembleAdapter)
