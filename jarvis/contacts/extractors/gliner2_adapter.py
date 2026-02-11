"""GLiNER2 adapter for the extractor bakeoff.

GLiNER2 is a newer version with architectural improvements.
Reference: https://github.com/urchade/GLiNER

Note: GLiNER2 uses a different model architecture and API.
Model: "urchade/gliner_multi-v2.1" or similar v2.1+ models.
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

logger = logging.getLogger(__name__)

# Same fact type resolution as GLiNER baseline
FACT_TYPE_RULES: list[tuple[str, set[str], str]] = [
    (r"allergic to", {"food_item", "health_condition", "allergy"}, "health.allergy"),
    (r"(live|living|based) in", {"place", "current_location"}, "location.current"),
    (r"(moving|headed|relocating) to", {"place", "future_location"}, "location.future"),
    (r"(grew up|from|raised) in", {"place", "past_location"}, "location.hometown"),
    (r"(moved from|used to live|lived in)", {"place", "past_location"}, "location.past"),
    (r"(work|working|started|joined) at", {"org", "employer"}, "work.employer"),
    (r"(left|quit|fired from|used to work)", {"org", "employer"}, "work.former_employer"),
    (r"(i'm a|work as|job as|position as)", {"job_role"}, "work.job_title"),
    (
        r"my (sister|brother|mom|dad|mother|father|aunt|uncle|cousin|grandma|grandpa)",
        {"person_name", "family_member"},
        "relationship.family",
    ),
    (r"my (friend|buddy|pal|homie)", {"person_name", "friend_name"}, "relationship.friend"),
    (
        r"my (wife|husband|girlfriend|boyfriend|partner|fiancee?)",
        {"person_name", "partner_name"},
        "relationship.partner",
    ),
    (
        r"(love|obsessed with|addicted to|favorite)",
        {"food_item"},
        "preference.food_like",
    ),
    (r"(hate|can't stand|despise|gross)", {"food_item"}, "preference.food_dislike"),
    (
        r"(love|enjoy|into) +(running|hiking|swimming|yoga|cooking|gaming|reading)",
        {"activity"},
        "preference.activity",
    ),
    (r"birthday is", {"date_ref"}, "personal.birthday"),
    (r"(go to|attend|graduated from|studying at)", {"org"}, "personal.school"),
    (r"my (dog|cat|pet|puppy|kitten)", {"person_name"}, "personal.pet"),
]

DIRECT_LABEL_MAP: dict[str, str] = {
    "allergy": "health.allergy",
    "current_location": "location.current",
    "future_location": "location.future",
    "past_location": "location.past",
    "employer": "work.employer",
    "family_member": "relationship.family",
    "friend_name": "relationship.friend",
    "partner_name": "relationship.partner",
}

PER_LABEL_MIN: dict[str, float] = {
    "person_name": 0.50,
    "family_member": 0.45,
    "place": 0.45,
    "org": 0.55,
    "date_ref": 0.55,
    "food_item": 0.45,
    "job_role": 0.50,
    "health_condition": 0.45,
    "activity": 0.50,
    "allergy": 0.40,
    "employer": 0.45,
    "current_location": 0.40,
    "future_location": 0.40,
    "past_location": 0.40,
    "friend_name": 0.45,
    "partner_name": 0.45,
}

VAGUE = {"it", "this", "that", "thing", "stuff", "them", "there", "here", "me", "you"}


class GLiNER2Adapter(ExtractorAdapter):
    """Adapter for GLiNER2 extractor.

    GLiNER2 uses a newer architecture with improved multi-task capabilities.
    It supports the same interface but may have different performance characteristics.

    Key differences from GLiNER v1:
    - Better multi-label support
    - Improved context handling
    - Different default thresholds (tuned lower for recall)
    """

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

    # Natural language prompts for GLiNER2
    NL_LABELS: dict[str, str] = {
        "person_name": "person name or individual",
        "family_member": "family member or relative",
        "place": "place or location",
        "org": "organization or company",
        "date_ref": "date or time reference",
        "food_item": "food or drink item",
        "job_role": "job title or profession",
        "health_condition": "medical condition or health issue",
        "activity": "activity or hobby",
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("gliner2", config)
        self._model: Any = None
        self._global_threshold = self.config.get("threshold", 0.15)
        self._model_name = self.config.get("model_name", "urchade/gliner_medium-v2.1")
        self._apply_thresholds = self.config.get("apply_thresholds", True)

    @property
    def supported_labels(self) -> list[str]:
        return list(self.SPAN_LABELS)

    @property
    def default_threshold(self) -> float:
        return self._global_threshold

    def _load_model(self) -> Any:
        """Lazy-load the GLiNER2 model."""
        if self._model is not None:
            return self._model

        try:
            from gliner import GLiNER
        except ImportError:
            raise ImportError("GLiNER not installed. Install with: uv pip install gliner")

        logger.info("Loading GLiNER2 model: %s", self._model_name)
        self._model = GLiNER.from_pretrained(self._model_name)
        logger.info("GLiNER2 model loaded")
        return self._model

    def _to_model_label(self, canonical_label: str) -> str:
        """Convert canonical label to natural language prompt."""
        return self.NL_LABELS.get(canonical_label, canonical_label.replace("_", " "))

    def _resolve_fact_type(self, text: str, span: str, span_label: str) -> str:
        """Map (text_pattern, span_label) -> fact_type."""
        for pattern, label_set, fact_type in FACT_TYPE_RULES:
            if span_label in label_set and re.search(pattern, text, re.IGNORECASE):
                return fact_type

        if span_label in DIRECT_LABEL_MAP:
            return DIRECT_LABEL_MAP[span_label]

        return "other_personal_fact"

    def _build_context_text(
        self,
        current_text: str,
        context_prev: list[str] | None = None,
        context_next: list[str] | None = None,
    ) -> tuple[str, int, int]:
        """Build model input with context."""
        prev = [p.strip() for p in (context_prev or []) if p.strip()]
        nxt = [n.strip() for n in (context_next or []) if n.strip()]

        if not prev and not nxt:
            return current_text, 0, len(current_text)

        separator = "\n[CTX]\n"

        if prev:
            merged = f"{' '.join(prev)}{separator}{current_text}"
            current_start = len(f"{' '.join(prev)}{separator}")
        else:
            merged = current_text
            current_start = 0

        current_end = current_start + len(current_text)

        if nxt:
            merged = f"{merged}{separator}{' '.join(nxt)}"

        return merged, current_start, current_end

    def _project_entity_to_current(
        self,
        entity: dict[str, Any],
        current_start: int,
        current_end: int,
        current_text: str,
    ) -> tuple[str, int, int] | None:
        """Project entity from merged context to current message offsets."""
        raw_start = int(entity.get("start", 0))
        raw_end = int(entity.get("end", raw_start + len(str(entity.get("text", "")))))
        raw_span = str(entity.get("text", "")).strip()

        if raw_end <= raw_start:
            return None

        # Check if entity is within current message bounds
        if raw_start < current_start or raw_end > current_end:
            # Try to find in current text
            if raw_span:
                idx = current_text.casefold().find(raw_span.casefold())
                if idx >= 0:
                    return raw_span, idx, idx + len(raw_span)
            return None

        start_char = raw_start - current_start
        end_char = raw_end - current_start

        if start_char < 0 or end_char > len(current_text) or end_char <= start_char:
            if raw_span:
                idx = current_text.casefold().find(raw_span.casefold())
                if idx >= 0:
                    return raw_span, idx, idx + len(raw_span)
            return None

        span_text = current_text[start_char:end_char].strip()

        # Validate match
        if raw_span and span_text.casefold() != raw_span.casefold():
            idx = current_text.casefold().find(raw_span.casefold())
            if idx >= 0:
                return raw_span, idx, idx + len(raw_span)

        if not span_text and raw_span:
            idx = current_text.casefold().find(raw_span.casefold())
            if idx >= 0:
                return raw_span, idx, idx + len(raw_span)
            return None

        return span_text or raw_span, start_char, end_char

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
        """Extract candidates using GLiNER2."""
        model = self._load_model()
        call_threshold = threshold if threshold is not None else self._global_threshold

        # Build context
        merged_text, current_start, current_end = self._build_context_text(
            text,
            context_prev=context_prev,
            context_next=context_next,
        )

        # Convert labels to natural language
        model_labels = [self._to_model_label(lbl) for lbl in self.SPAN_LABELS]
        label_to_canonical = {self._to_model_label(lbl): lbl for lbl in self.SPAN_LABELS}
        label_to_canonical.update({lbl: lbl for lbl in self.SPAN_LABELS})

        # Predict
        entities = model.predict_entities(
            merged_text,
            model_labels,
            threshold=call_threshold,
            flat_ner=True,
        )

        # Filter and normalize
        candidates: list[ExtractedCandidate] = []
        seen: set[tuple[str, str]] = set()

        for entity in entities:
            projected = self._project_entity_to_current(
                entity,
                current_start=current_start,
                current_end=current_end,
                current_text=text,
            )
            if projected is None:
                continue

            span, start_char, end_char = projected
            raw_label = str(entity.get("label", ""))
            label = label_to_canonical.get(raw_label, raw_label)
            score = float(entity.get("score", 0.0))

            # Apply thresholds
            if self._apply_thresholds and score < PER_LABEL_MIN.get(label, 0.55):
                continue

            if span.casefold() in VAGUE or len(span) < 2:
                continue

            dedup_key = (span.casefold(), label)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            fact_type = self._resolve_fact_type(text, span, label)

            candidates.append(
                ExtractedCandidate(
                    span_text=span,
                    span_label=label,
                    score=score,
                    start_char=start_char,
                    end_char=end_char,
                    fact_type=fact_type,
                    extractor_metadata={"source": "gliner2", "raw_score": score},
                )
            )

        return candidates

    def extract_batch(
        self,
        messages: list[dict[str, Any]],
        batch_size: int = 32,
        threshold: float | None = None,
    ) -> list[ExtractionResult]:
        """Batch extraction with GLiNER2."""
        import time

        model = self._load_model()
        call_threshold = threshold if threshold is not None else self._global_threshold

        # Convert labels
        model_labels = [self._to_model_label(lbl) for lbl in self.SPAN_LABELS]
        label_to_canonical = {self._to_model_label(lbl): lbl for lbl in self.SPAN_LABELS}
        label_to_canonical.update({lbl: lbl for lbl in self.SPAN_LABELS})

        results: list[ExtractionResult] = []

        for i in range(0, len(messages), batch_size):
            batch = messages[i : i + batch_size]

            # Build merged texts
            merged_texts: list[str] = []
            current_bounds: list[tuple[int, int]] = []

            for msg in batch:
                merged, start, end = self._build_context_text(
                    msg.get("text", ""),
                    context_prev=msg.get("context_prev"),
                    context_next=msg.get("context_next"),
                )
                merged_texts.append(merged)
                current_bounds.append((start, end))

            # Batch prediction
            batch_start = time.perf_counter()
            batch_entities = model.batch_predict_entities(
                merged_texts,
                model_labels,
                threshold=call_threshold,
                flat_ner=True,
            )

            # Process results
            for msg, bounds, entities in zip(batch, current_bounds, batch_entities):
                msg_start = time.perf_counter()

                candidates: list[ExtractedCandidate] = []
                seen: set[tuple[str, str]] = set()
                current_text = msg.get("text", "")
                current_start, current_end = bounds

                for e in entities:
                    projected = self._project_entity_to_current(
                        e,
                        current_start=current_start,
                        current_end=current_end,
                        current_text=current_text,
                    )
                    if projected is None:
                        continue

                    span, start_char, end_char = projected
                    raw_label = str(e.get("label", ""))
                    label = label_to_canonical.get(raw_label, raw_label)
                    score = float(e.get("score", 0.0))

                    if score < PER_LABEL_MIN.get(label, 0.55):
                        continue
                    if span.casefold() in VAGUE or len(span) < 2:
                        continue

                    dedup_key = (span.casefold(), label)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    fact_type = self._resolve_fact_type(current_text, span, label)

                    candidates.append(
                        ExtractedCandidate(
                            span_text=span,
                            span_label=label,
                            score=score,
                            start_char=start_char,
                            end_char=end_char,
                            fact_type=fact_type,
                            extractor_metadata={"source": "gliner2", "raw_score": score},
                        )
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
register_extractor("gliner2", GLiNER2Adapter)
