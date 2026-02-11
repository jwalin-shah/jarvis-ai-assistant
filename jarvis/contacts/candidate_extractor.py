"""GLiNER-based candidate extraction for two-stage fact pipeline.

Stage 1: High-recall candidate extraction using GLiNER NER.
Outputs FactCandidate objects (or JSONL) for downstream filtering,
attribution, and threading (stage 2).

Two-layer label architecture:
- span_label: what the text chunk *is* (GLiNER output)
- fact_type: what claim the span represents in context (our ontology)
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

from jarvis.contacts.junk_filters import is_junk_message

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GLiNER config
# ---------------------------------------------------------------------------

MODEL = "urchade/gliner_medium-v2.1"
GLOBAL_THRESHOLD = 0.35  # high recall

# Canonical labels used internally by our pipeline and evaluation.
SPAN_LABELS = [
    # Generic
    "person_name",
    "place",
    "org",
    "date_ref",
    # Fact-like
    "food_item",
    "job_role",
    "health_condition",
    "activity",
    "allergy",
    "employer",
    "current_location",
    "future_location",
    "past_location",
    "family_member",
    "friend_name",
    "partner_name",
]

# Natural-language prompts sent to GLiNER.
# Keep these semantically explicit and close to everyday wording.
NATURAL_LANGUAGE_LABELS: dict[str, str] = {
    "person_name": "person name",
    "place": "place",
    "org": "organization",
    "date_ref": "date reference",
    "food_item": "food item",
    "job_role": "job role",
    "health_condition": "medical condition",
    "activity": "activity",
    "allergy": "allergy",
    "employer": "employer",
    "current_location": "current location",
    "future_location": "future location",
    "past_location": "past location",
    "family_member": "family member",
    "friend_name": "friend name",
    "partner_name": "romantic partner",
}

# Per-label minimum thresholds (after global threshold)
PER_LABEL_MIN: dict[str, float] = {
    "person_name": 0.55,
    "place": 0.50,
    "org": 0.60,
    "food_item": 0.50,
    "job_role": 0.55,
    "health_condition": 0.50,
    "activity": 0.55,
    "date_ref": 0.60,
    "allergy": 0.45,
    "employer": 0.50,
    "current_location": 0.45,
    "future_location": 0.45,
    "past_location": 0.45,
    "family_member": 0.50,
    "friend_name": 0.50,
    "partner_name": 0.50,
}
DEFAULT_MIN = 0.55

# Vague words to reject as span_text
VAGUE = {"it", "this", "that", "thing", "stuff", "them", "there", "here", "me", "you"}

# ---------------------------------------------------------------------------
# Fact type ontology (~20 high-value memory types)
# ---------------------------------------------------------------------------

FACT_TYPES = {
    # location
    "location.current",
    "location.past",
    "location.future",
    "location.hometown",
    # work
    "work.employer",
    "work.job_title",
    "work.former_employer",
    # relationship
    "relationship.family",
    "relationship.friend",
    "relationship.partner",
    # preference
    "preference.food_like",
    "preference.food_dislike",
    "preference.activity",
    # health
    "health.allergy",
    "health.dietary",
    "health.condition",
    # personal
    "personal.birthday",
    "personal.school",
    "personal.pet",
    # fallback
    "other_personal_fact",
}

# Mapping: (text_pattern_regex, span_label_set, fact_type)
# span_label_set supports both generic and fact-like labels
FACT_TYPE_RULES: list[tuple[str, set[str], str]] = [
    # health / allergy
    (r"allergic to", {"food_item", "health_condition", "allergy"}, "health.allergy"),
    # location
    (r"(live|living|based) in", {"place", "current_location"}, "location.current"),
    (r"(moving|headed|relocating) to", {"place", "future_location"}, "location.future"),
    (r"(grew up|from|raised) in", {"place", "past_location"}, "location.hometown"),
    (r"(moved from|used to live|lived in)", {"place", "past_location"}, "location.past"),
    # work
    (r"(work|working|started|joined) at", {"org", "employer"}, "work.employer"),
    (r"(left|quit|fired from|used to work)", {"org", "employer"}, "work.former_employer"),
    (r"(i'm a|work as|job as|position as)", {"job_role"}, "work.job_title"),
    # relationship
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
    # preference
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
    # personal
    (r"birthday is", {"date_ref"}, "personal.birthday"),
    (r"(go to|attend|graduated from|studying at)", {"org"}, "personal.school"),
    (r"my (dog|cat|pet|puppy|kitten)", {"person_name"}, "personal.pet"),
]

# Direct label → fact_type for fact-like labels (no pattern needed)
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

# ---------------------------------------------------------------------------
# FactCandidate dataclass
# ---------------------------------------------------------------------------


@dataclass
class FactCandidate:
    """A candidate fact extracted by GLiNER, pending downstream filtering."""

    message_id: int  # iMessage ROWID
    span_text: str  # extracted entity text ("Austin", "Google", "Sarah")
    span_label: str  # GLiNER label (place, org, person_name, allergy, etc.)
    gliner_score: float  # raw GLiNER confidence
    fact_type: str  # mapped type (location.future, work.employer, etc.)
    start_char: int  # character offset start in source_text
    end_char: int  # character offset end in source_text
    source_text: str = ""  # full message text for context
    chat_id: int | None = None
    is_from_me: bool | None = None  # True if sent by user
    sender_handle_id: int | None = None  # handle ROWID of sender
    message_date: int | None = None  # iMessage date (Core Data timestamp)
    status: str = "pending"  # pending | accepted | rejected

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSONL output."""
        return asdict(self)


# ---------------------------------------------------------------------------
# CandidateExtractor
# ---------------------------------------------------------------------------


class CandidateExtractor:
    """Extract fact candidates from messages using GLiNER.

    Lazy-loads the model (~500MB) on first call.
    Uses hybrid label set (generic + fact-like) for high recall.
    """

    def __init__(
        self,
        model_name: str = MODEL,
        labels: list[str] | None = None,
        global_threshold: float = GLOBAL_THRESHOLD,
        per_label_min: dict[str, float] | None = None,
        default_min: float = DEFAULT_MIN,
    ):
        self._model: Any = None
        self._model_name = model_name
        # `labels` are canonical label ids. GLiNER sees natural-language prompts.
        self._labels_canonical = labels or SPAN_LABELS
        self._model_labels = [self._to_model_label(lbl) for lbl in self._labels_canonical]
        self._label_to_canonical = {
            self._to_model_label(lbl): lbl for lbl in self._labels_canonical
        }
        # Also accept canonical labels directly if the model returns them.
        self._label_to_canonical.update({lbl: lbl for lbl in self._labels_canonical})
        self._global_threshold = global_threshold
        self._per_label_min = (
            dict(per_label_min) if per_label_min is not None else dict(PER_LABEL_MIN)
        )
        self._default_min = default_min

    @staticmethod
    def _to_model_label(canonical_label: str) -> str:
        """Convert canonical label id to a natural-language GLiNER prompt label."""
        return NATURAL_LANGUAGE_LABELS.get(canonical_label, canonical_label.replace("_", " "))

    def _canonicalize_label(self, predicted_label: str) -> str:
        """Map GLiNER label output back to canonical label ids."""
        return self._label_to_canonical.get(predicted_label, predicted_label)

    def _load_model(self) -> Any:
        """Lazy-load GLiNER model (500MB, load once)."""
        if self._model is not None:
            return self._model
        from gliner import GLiNER

        logger.info("Loading GLiNER model: %s", self._model_name)
        self._model = GLiNER.from_pretrained(self._model_name)
        logger.info("GLiNER model loaded")
        return self._model

    def predict_raw_entities(self, text: str, threshold: float = 0.01) -> list[dict[str, Any]]:
        """Run raw GLiNER prediction without local filtering.

        Args:
            text: Input message text.
            threshold: GLiNER score threshold used at model call time.
        """
        model = self._load_model()
        entities = model.predict_entities(
            text,
            self._model_labels,
            threshold=threshold,
            flat_ner=True,
        )
        normalized: list[dict[str, Any]] = []
        for entity in entities:
            item = dict(entity)
            raw_label = str(item.get("label", ""))
            item["raw_label"] = raw_label
            item["label"] = self._canonicalize_label(raw_label)
            normalized.append(item)
        return normalized

    def extract_candidates(
        self,
        text: str,
        message_id: int,
        *,
        chat_id: int | None = None,
        is_from_me: bool | None = None,
        sender_handle_id: int | None = None,
        message_date: int | None = None,
        threshold: float | None = None,
        apply_label_thresholds: bool = True,
        apply_vague_filter: bool = True,
    ) -> list[FactCandidate]:
        """Single message -> list of FactCandidates.

        Applies junk filter, per-label thresholds, vague word rejection,
        and strict dedup before returning.
        """
        if is_junk_message(text):
            return []

        self._load_model()
        call_threshold = self._global_threshold if threshold is None else threshold
        ents = self._model.predict_entities(
            text,
            self._model_labels,
            threshold=call_threshold,
            flat_ner=True,
        )

        out: list[FactCandidate] = []
        seen: set[tuple[str, str]] = set()  # (span_casefolded, label) for dedup

        for e in ents:
            span = e["text"].strip()
            label = self._canonicalize_label(e["label"])
            score = float(e.get("score", 0.0))

            # Per-label threshold
            if apply_label_thresholds and score < self._per_label_min.get(label, self._default_min):
                continue

            # Vague word rejection
            if apply_vague_filter and (span.casefold() in VAGUE or len(span) < 2):
                continue

            # Strict dedup: same span + label + message_id emitted once
            dedup_key = (span.casefold(), label)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # Resolve character offsets
            start_char = e.get("start", 0)
            end_char = e.get("end", start_char + len(span))

            fact_type = self._resolve_fact_type(text, span, label)

            out.append(
                FactCandidate(
                    message_id=message_id,
                    span_text=span,
                    span_label=label,
                    gliner_score=score,
                    fact_type=fact_type,
                    start_char=start_char,
                    end_char=end_char,
                    source_text=text,
                    chat_id=chat_id,
                    is_from_me=is_from_me,
                    sender_handle_id=sender_handle_id,
                    message_date=message_date,
                )
            )

        return out

    def extract_batch(
        self,
        messages: list[dict[str, Any]],
        batch_size: int = 32,
    ) -> list[FactCandidate]:
        """Batch extraction with progress reporting.

        Each message dict should have at minimum:
            - text: str
            - message_id: int
        Optional keys: chat_id, is_from_me, sender_handle_id, message_date
        """
        self._load_model()

        # Pre-filter junk messages
        valid_msgs = [m for m in messages if not is_junk_message(m.get("text", ""))]

        all_candidates: list[FactCandidate] = []
        total = len(valid_msgs)

        for i in range(0, total, batch_size):
            batch = valid_msgs[i : i + batch_size]
            texts = [m["text"] for m in batch]

            # GLiNER batch prediction
            batch_entities = self._model.batch_predict_entities(
                texts, self._model_labels, threshold=self._global_threshold, flat_ner=True
            )

            for msg, ents in zip(batch, batch_entities):
                seen: set[tuple[str, str]] = set()
                msg_id = msg["message_id"]

                for e in ents:
                    span = e["text"].strip()
                    label = self._canonicalize_label(e["label"])
                    score = float(e.get("score", 0.0))

                    if score < self._per_label_min.get(label, self._default_min):
                        continue
                    if span.casefold() in VAGUE or len(span) < 2:
                        continue

                    dedup_key = (span.casefold(), label)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    start_char = e.get("start", 0)
                    end_char = e.get("end", start_char + len(span))
                    fact_type = self._resolve_fact_type(msg["text"], span, label)

                    all_candidates.append(
                        FactCandidate(
                            message_id=msg_id,
                            span_text=span,
                            span_label=label,
                            gliner_score=score,
                            fact_type=fact_type,
                            start_char=start_char,
                            end_char=end_char,
                            source_text=msg["text"],
                            chat_id=msg.get("chat_id"),
                            is_from_me=msg.get("is_from_me"),
                            sender_handle_id=msg.get("sender_handle_id"),
                            message_date=msg.get("message_date"),
                        )
                    )

            processed = min(i + batch_size, total)
            logger.info(
                "Batch progress: %d/%d messages (%d candidates so far)",
                processed,
                total,
                len(all_candidates),
            )

        return all_candidates

    def _resolve_fact_type(self, text: str, span: str, span_label: str) -> str:
        """Map (text_pattern, span_label) -> fact_type.

        Priority:
        1. Pattern + label match from FACT_TYPE_RULES
        2. Direct label map for fact-like labels (allergy, employer, etc.)
        3. Fallback: other_personal_fact
        """
        # Pattern-based rules
        for pattern, label_set, fact_type in FACT_TYPE_RULES:
            if span_label in label_set and re.search(pattern, text, re.IGNORECASE):
                return fact_type

        # Direct label map for fact-like labels
        if span_label in DIRECT_LABEL_MAP:
            return DIRECT_LABEL_MAP[span_label]

        return "other_personal_fact"
