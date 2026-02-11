"""GLiNER adapter for the extractor bakeoff.

Wraps the existing CandidateExtractor with the new adapter interface.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from jarvis.contacts.candidate_extractor import (
    DEFAULT_MIN,
    DIRECT_LABEL_MAP,
    FACT_TYPE_RULES,
    VAGUE,
    per_label_min_for_profile,
)
from jarvis.contacts.extractors.base import (
    ExtractedCandidate,
    ExtractionResult,
    ExtractorAdapter,
    register_extractor,
)

logger = logging.getLogger(__name__)

# Quality guardrails for spans
STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "my",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "and",
    "or",
    "but",
    "if",
    "so",
    "that",
    "this",
    "just",
    "not",
    "no",
    "our",
    "your",
    "their",
    "have",
    "has",
    "had",
    "me",
    "you",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "hers",
    "its",
    "ours",
    "theirs",
}

# Patterns that indicate a span is likely junk or a partial sentence
BAD_SPAN_PATTERNS = [
    r"^[.,!?;:]",  # Starts with punctuation
    r"[.,!?;:]$",  # Ends with punctuation (often GLiNER includes trailing periods)
    r"^(and|but|or|so|if|then|when|because|while|since)\b",  # Starts with conjunction
    r"\b(i|you|he|she|it|we|they)\b",  # Contains standalone pronouns
]


def clean_span(span: str) -> str:
    """Strip trailing/leading punctuation often mis-included by GLiNER."""
    return span.strip(".,!?;:()[]\"' ")


class GLiNERAdapter(ExtractorAdapter):
    """Adapter for GLiNER (baseline) extractor.

    Uses the existing CandidateExtractor internally but exposes the
    standardized adapter interface.
    """

    # Canonical labels used for extraction
    SPAN_LABELS = [
        "person_name",
        "family_member",
        "place",
        "org",
        "date_ref",
        "food_item",
        "job_role",
        "health_condition",
        "activity",
    ]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("gliner", config)
        self._extractor: Any = None
        self._label_profile = self.config.get("label_profile", "balanced")
        self._global_threshold = self.config.get("threshold", 0.15)
        self._model_name = self.config.get("model_name", "urchade/gliner_medium-v2.1")
        self._apply_thresholds = self.config.get("apply_thresholds", True)
        self._per_label_min = per_label_min_for_profile(self._label_profile)
        raw_labels = self.config.get("labels")
        self._labels: list[str] | None = None
        if isinstance(raw_labels, list):
            self._labels = [str(label) for label in raw_labels if str(label).strip()]

    @property
    def supported_labels(self) -> list[str]:
        """Return the list of labels this extractor supports."""
        if self._labels is not None:
            return list(self._labels)
        return list(self.SPAN_LABELS)

    @property
    def default_threshold(self) -> float:
        """Return the default confidence threshold."""
        return self._global_threshold

    def _load_model(self) -> Any:
        """Lazy-load the GLiNER extractor."""
        if self._extractor is not None:
            return self._extractor

        from jarvis.contacts.candidate_extractor import CandidateExtractor

        logger.info("Loading GLiNER adapter with model: %s", self._model_name)
        self._extractor = CandidateExtractor(
            model_name=self._model_name,
            labels=self._labels,
            label_profile=self._label_profile,
            global_threshold=self._global_threshold,
        )
        # Force model load
        self._extractor._load_model()
        logger.info("GLiNER adapter loaded")
        return self._extractor

    def _resolve_fact_type(self, text: str, span: str, span_label: str) -> str:
        """Map (text_pattern, span_label) -> fact_type."""
        # Pattern-based rules
        for pattern, label_set, fact_type in FACT_TYPE_RULES:
            if span_label in label_set and re.search(pattern, text, re.IGNORECASE):
                return fact_type

        # Direct label map for fact-like labels
        if span_label in DIRECT_LABEL_MAP:
            return DIRECT_LABEL_MAP[span_label]

        return "other_personal_fact"

    def _filter_candidates(
        self,
        entities: list[dict[str, Any]],
        text: str,
        apply_thresholds: bool = True,
    ) -> list[ExtractedCandidate]:
        """Filter and normalize raw GLiNER entities."""
        candidates: list[ExtractedCandidate] = []
        seen: set[tuple[str, str]] = set()

        for e in entities:
            raw_span = str(e.get("text", ""))
            span = clean_span(raw_span)
            label = str(e.get("label", ""))
            score = float(e.get("score", 0.0))

            # 1. Per-label threshold (profile-aware)
            if apply_thresholds and score < self._per_label_min.get(label, DEFAULT_MIN):
                continue

            # 2. Vague word rejection
            if span.casefold() in VAGUE or len(span) < 2:
                continue

            # 3. Stopword span rejection (entire span is a stopword)
            if span.casefold() in STOPWORDS:
                continue

            # 4. Token length guardrails (e.g. max 6 tokens for most entities)
            tokens = span.split()
            if len(tokens) > 6 and label not in ("activity", "health_condition"):
                continue

            # 5. Bad pattern rejection
            if any(re.search(p, span, re.IGNORECASE) for p in BAD_SPAN_PATTERNS):
                continue

            # Strict dedup
            dedup_key = (span.casefold(), label)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # Resolve fact type
            fact_type = self._resolve_fact_type(text, span, label)

            # Drop unresolvable fallback spans (highest FP source)
            if fact_type == "other_personal_fact":
                continue

            candidates.append(
                ExtractedCandidate(
                    span_text=span,
                    span_label=label,
                    score=score,
                    start_char=int(e.get("start", 0)),
                    end_char=int(e.get("end", 0)),
                    fact_type=fact_type,
                    extractor_metadata={"source": "gliner", "raw_score": score},
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
        """Extract candidates from a single message using GLiNER."""
        extractor = self._load_model()

        call_threshold = threshold if threshold is not None else self._global_threshold

        # Get raw entities using the underlying extractor
        raw_entities = extractor.predict_raw_entities(
            text=text,
            threshold=call_threshold,
            prev_messages=context_prev,
            next_messages=context_next,
        )

        # Filter and normalize
        return self._filter_candidates(
            raw_entities,
            text,
            apply_thresholds=self._apply_thresholds,
        )

    def extract_batch(
        self,
        messages: list[dict[str, Any]],
        batch_size: int = 32,
        threshold: float | None = None,
    ) -> list[ExtractionResult]:
        """Batch extraction with true batching support."""
        import time

        extractor = self._load_model()
        call_threshold = threshold if threshold is not None else self._global_threshold

        results: list[ExtractionResult] = []

        # Process in batches
        for i in range(0, len(messages), batch_size):
            batch = messages[i : i + batch_size]

            # Build merged texts with context
            merged_texts: list[str] = []
            current_bounds: list[tuple[int, int]] = []

            for msg in batch:
                merged_text, current_start, current_end = extractor._build_context_text(
                    msg.get("text", ""),
                    prev_messages=msg.get("context_prev"),
                    next_messages=msg.get("context_next"),
                )
                merged_texts.append(merged_text)
                current_bounds.append((current_start, current_end))

            # Batch prediction
            batch_entities = extractor._model.batch_predict_entities(
                merged_texts,
                extractor._model_labels,
                threshold=call_threshold,
                flat_ner=True,
            )

            # Process each message's results
            for msg, bounds, entities in zip(batch, current_bounds, batch_entities):
                msg_start = time.perf_counter()

                # Project and canonicalize entities before filtering
                projected_entities: list[dict[str, Any]] = []
                current_text = msg.get("text", "")
                current_start, current_end = bounds

                for e in entities:
                    projected = extractor._project_entity_to_current(
                        e,
                        current_start=current_start,
                        current_end=current_end,
                        current_text=current_text,
                    )
                    if projected is None:
                        continue

                    span, start_char, end_char = projected
                    p_ent = dict(e)
                    p_ent["text"] = span
                    p_ent["start"] = start_char
                    p_ent["end"] = end_char
                    p_ent["label"] = extractor._canonicalize_label(str(e.get("label", "")))
                    projected_entities.append(p_ent)

                # Filter and normalize using shared logic
                candidates = self._filter_candidates(
                    projected_entities,
                    current_text,
                    apply_thresholds=self._apply_thresholds,
                )

                elapsed = (time.perf_counter() - msg_start) * 1000
                results.append(
                    ExtractionResult(
                        message_id=msg["message_id"],
                        candidates=candidates,
                        extractor_name=self.name,
                        processing_time_ms=elapsed,
                    )
                )

        return results


# Register the adapter
register_extractor("gliner", GLiNERAdapter)
