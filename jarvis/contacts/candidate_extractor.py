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

from jarvis.contacts.fact_filter import is_fact_likely
from jarvis.contacts.junk_filters import is_junk_message

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GLiNER config
# ---------------------------------------------------------------------------

MODEL = "urchade/gliner_medium-v2.1"
GLOBAL_THRESHOLD = 0.35  # high recall

# Canonical extraction labels used by default for stage-1 candidate generation.
# These are intentionally coarse; downstream filtering and attribution resolve
# finer distinctions (e.g., current vs future location, employer vs school).
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

# Label profiles:
# - high_recall: broad mining (default behavior / research mode)
# - balanced: drops labels with highest nuisance rate in chat data
# - high_precision: stricter shortlist for production candidate quality
LABEL_PROFILES: dict[str, list[str]] = {
    "high_recall": list(SPAN_LABELS),
    "balanced": [
        "family_member",
        "place",
        "org",
        "food_item",
        "job_role",
        "health_condition",
        "activity",
    ],
    "high_precision": [
        "family_member",
        "place",
        "org",
        "food_item",
        "job_role",
        "health_condition",
        "activity",
    ],
}

# Natural-language prompts sent to GLiNER.
# Keep these semantically explicit and close to everyday wording.
# Tuned via scripts/sweep_gliner_thresholds.py --label-variants
NATURAL_LANGUAGE_LABELS: dict[str, str] = {
    "person_name": "first name or nickname of a person",
    "family_member": "family member (mom, sister, etc)",
    "place": "city, town, country, or geographic location",
    "org": "company, school, university, or employer",
    "date_ref": "date or time reference",
    "food_item": "food or drink",
    "job_role": "job title or profession",
    "health_condition": "medical condition or symptom",
    "activity": "hobby, sport, game, or pastime",
    # Optional aliases retained for custom label sets.
    "allergy": "allergy",
    "employer": "employer",
    "current_location": "current location",
    "future_location": "future location",
    "past_location": "past location",
    "friend_name": "friend name",
    "partner_name": "romantic partner",
}

# Per-label minimum thresholds (after global threshold)
PER_LABEL_MIN: dict[str, float] = {
    "person_name": 0.45,  # Was 0.55; lowered for recall with improved prompt
    "family_member": 0.45,  # Was 0.50; high-signal label, entailment filters FPs
    "place": 0.45,  # Was 0.50; entailment gate handles precision
    "org": 0.55,  # Was 0.60; lowered slightly for recall
    "date_ref": 0.60,
    "food_item": 0.50,
    "job_role": 0.60,  # Was 0.65; entailment gate handles FP reduction
    "health_condition": 0.45,
    "activity": 0.40,  # Was 0.65; lowered for recall, entailment gate filters FPs
    # Optional aliases retained for custom label sets.
    "allergy": 0.45,
    "employer": 0.50,
    "current_location": 0.45,
    "future_location": 0.45,
    "past_location": 0.45,
    "friend_name": 0.50,
    "partner_name": 0.50,
}
DEFAULT_MIN = 0.50
CONTEXT_SEPARATOR = "\n[CTX]\n"

# Profile-specific per-label minimum score overrides.
# These are merged on top of PER_LABEL_MIN.
PER_LABEL_MIN_PROFILE_OVERRIDES: dict[str, dict[str, float]] = {
    # Keep broad recall in research mode.
    "high_recall": {},
    # Slightly tighter on historically noisy labels.
    "balanced": {
        "place": 0.55,
        "org": 0.60,
        "activity": 0.50,
        "job_role": 0.65,
    },
    # Stronger precision-oriented thresholds.
    "high_precision": {
        "family_member": 0.55,
        "place": 0.60,
        "org": 0.65,
        "food_item": 0.55,
        "job_role": 0.68,
        "health_condition": 0.50,
        "activity": 0.55,
    },
}

# Common abbreviation -> canonical name for entity normalization
ENTITY_ALIASES: dict[str, dict[str, str]] = {
    "place": {
        "sf": "San Francisco",
        "nyc": "New York City",
        "ny": "New York",
        "la": "Los Angeles",
        "dc": "Washington DC",
        "philly": "Philadelphia",
        "chi": "Chicago",
        "atl": "Atlanta",
        "bos": "Boston",
    },
}

# Vague words to reject as span_text.
# Includes both straight (') and curly (\u2019) apostrophe variants
# since iMessage uses curly quotes.
VAGUE = {
    "it",
    "this",
    "that",
    "thing",
    "stuff",
    "them",
    "there",
    "here",
    "me",
    "you",
    "i",
    "i'm",
    "i\u2019m",
    "i'll",
    "i\u2019ll",
    "i've",
    "i\u2019ve",
    "i'd",
    "i\u2019d",
    "we",
    "we're",
    "we\u2019re",
    # Common chat abbreviations GLiNER falsely tags as person_name
    "ik",  # "I know"
    "ai",  # the word "AI"
    "boi",  # slang
}

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
_FACT_TYPE_RULES_RAW: list[tuple[str, set[str], str]] = [
    # health / allergy
    (r"allergic to", {"food_item", "health_condition", "allergy"}, "health.allergy"),
    # location
    (r"(live|living|based) in", {"place", "current_location"}, "location.current"),
    (r"(stay|staying|reside) +(in|at|near)", {"place", "current_location"}, "location.current"),
    (
        r"(already|currently|right now|here|back) +in",
        {"place", "current_location"},
        "location.current",
    ),
    (r"(came|arrived|just got) +(to|in|at)", {"place", "current_location"}, "location.current"),
    (r"(moving|headed|relocating) to", {"place", "future_location"}, "location.future"),
    (r"(going|headed|flying|driving|drove) +to", {"place", "future_location"}, "location.future"),
    (r"(visiting|trip to|vacation in)", {"place"}, "location.current"),
    (r"(grew up|from|raised) in", {"place", "past_location"}, "location.hometown"),
    (r"(born|hometown|originally) +(in|from)", {"place", "past_location"}, "location.hometown"),
    (r"(moved from|used to live|lived in)", {"place", "past_location"}, "location.past"),
    # work
    (r"(work|working|started|joined) at", {"org", "employer"}, "work.employer"),
    (r"(hired|promoted|intern) at", {"org", "employer"}, "work.employer"),
    (r"(interning|internship) +(at|with)", {"org", "employer"}, "work.employer"),
    (r"(applied|applying|app) +(to|at|for)", {"org"}, "personal.school"),
    (r"(left|quit|fired from|used to work)", {"org", "employer"}, "work.former_employer"),
    (r"(i'm a|i am a|i am an|work as|job as|position as)", {"job_role"}, "work.job_title"),
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
        r"(love|obsessed with|addicted to|favorite|crave)",
        {"food_item"},
        "preference.food_like",
    ),
    (r"(hate|can't stand|despise|gross)", {"food_item"}, "preference.food_dislike"),
    (
        r"(love|enjoy|into|started|play|playing) +\w+",
        {"activity"},
        "preference.activity",
    ),
    # personal
    (r"birthday is", {"date_ref"}, "personal.birthday"),
    (r"(born|bday|b-day) +(on|is)?", {"date_ref"}, "personal.birthday"),
    (r"(birthday|bday).{0,10}(on|is)", {"date_ref"}, "personal.birthday"),
    (r"(go to|attend|graduated from|studying at)", {"org"}, "personal.school"),
    (r"my (dog|cat|pet|puppy|kitten)", {"person_name"}, "personal.pet"),
]

# Pre-compile regex patterns at module load for O(1) reuse (not per-call re.search)
FACT_TYPE_RULES: list[tuple[re.Pattern, set[str], str]] = [
    (re.compile(pattern, re.IGNORECASE), label_set, fact_type)
    for pattern, label_set, fact_type in _FACT_TYPE_RULES_RAW
]

# Direct label -> fact_type for fact-like labels (no pattern needed)
DIRECT_LABEL_MAP: dict[str, str] = {
    "allergy": "health.allergy",
    "health_condition": "health.condition",
    "current_location": "location.current",
    "future_location": "location.future",
    "past_location": "location.past",
    "employer": "work.employer",
    "family_member": "relationship.family",
    "friend_name": "relationship.friend",
    "partner_name": "relationship.partner",
    "food_item": "preference.food_like",
    "activity": "preference.activity",
    "job_role": "work.job_title",
    # Fallback defaults for generic GLiNER labels (pattern rules take priority)
    "place": "location.current",
    "org": "work.employer",
    "person_name": "relationship.friend",
    "date_ref": "personal.birthday",
}


def list_label_profiles() -> list[str]:
    """Return valid label profile names."""
    return sorted(LABEL_PROFILES.keys())


def labels_for_profile(profile: str) -> list[str]:
    """Return canonical labels for a known profile."""
    if profile not in LABEL_PROFILES:
        valid = ", ".join(list_label_profiles())
        raise ValueError(f"Unknown label profile '{profile}'. Valid profiles: {valid}")
    return list(LABEL_PROFILES[profile])


def per_label_min_for_profile(profile: str) -> dict[str, float]:
    """Return per-label min thresholds for a profile."""
    merged = dict(PER_LABEL_MIN)
    merged.update(PER_LABEL_MIN_PROFILE_OVERRIDES.get(profile, {}))
    return merged


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
    Uses a coarse, natural-language label set for high-recall candidate mining.
    """

    def __init__(
        self,
        model_name: str = MODEL,
        labels: list[str] | None = None,
        label_profile: str | None = None,
        global_threshold: float = GLOBAL_THRESHOLD,
        per_label_min: dict[str, float] | None = None,
        default_min: float = DEFAULT_MIN,
        backend: str = "auto",
        use_entailment: bool = False,  # disabled: NLI scores ~0 for all non-preference types
        entailment_threshold: float = 0.12,
    ):
        self._model: Any = None
        self._mlx_model: Any = None
        self._model_name = model_name
        self._backend = backend  # "auto", "mlx", or "pytorch"
        self._label_profile = label_profile or "high_recall"

        if labels is not None:
            # Explicit labels override profile selection.
            resolved_labels = list(labels)
        elif label_profile is not None:
            resolved_labels = labels_for_profile(label_profile)
        else:
            resolved_labels = list(SPAN_LABELS)

        # `labels` are canonical label ids. GLiNER sees natural-language prompts.
        self._labels_canonical = resolved_labels
        self._model_labels = [self._to_model_label(lbl) for lbl in self._labels_canonical]
        self._label_to_canonical = {
            self._to_model_label(lbl): lbl for lbl in self._labels_canonical
        }
        # Also accept canonical labels directly if the model returns them.
        self._label_to_canonical.update({lbl: lbl for lbl in self._labels_canonical})
        self._global_threshold = global_threshold
        if per_label_min is not None:
            self._per_label_min = dict(per_label_min)
        else:
            self._per_label_min = (
                per_label_min_for_profile(label_profile)
                if label_profile is not None
                else dict(PER_LABEL_MIN)
            )
        self._default_min = default_min
        self._use_entailment = use_entailment
        self._entailment_threshold = entailment_threshold

    @staticmethod
    def _to_model_label(canonical_label: str) -> str:
        """Convert canonical label id to a natural-language GLiNER prompt label."""
        return NATURAL_LANGUAGE_LABELS.get(canonical_label, canonical_label.replace("_", " "))

    def _canonicalize_label(self, predicted_label: str) -> str:
        """Map GLiNER label output back to canonical label ids."""
        return self._label_to_canonical.get(predicted_label, predicted_label)

    @staticmethod
    def _normalize_context_messages(messages: Any) -> list[str]:
        """Normalize optional context payload into a clean list of message strings."""
        if messages is None:
            return []
        if isinstance(messages, str):
            msg = messages.strip()
            return [msg] if msg else []
        normalized: list[str] = []
        for item in messages:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if text:
                normalized.append(text)
        return normalized

    def _build_context_text(
        self,
        current_text: str,
        prev_messages: Any = None,
        next_messages: Any = None,
    ) -> tuple[str, int, int]:
        """Build model input text and return current-message character bounds."""
        prev = self._normalize_context_messages(prev_messages)
        nxt = self._normalize_context_messages(next_messages)
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
    def _find_entity_in_text(
        raw_span: str, current_text: str
    ) -> tuple[str, int, int] | None:
        """Case-insensitive search for *raw_span* inside *current_text*.

        Returns (raw_span, start, end) on match, or None.
        """
        if not raw_span:
            return None
        idx = current_text.casefold().find(raw_span.casefold())
        if idx < 0:
            return None
        return raw_span, idx, idx + len(raw_span)

    def _project_entity_to_current(
        self,
        entity: dict[str, Any],
        *,
        current_start: int,
        current_end: int,
        current_text: str,
    ) -> tuple[str, int, int] | None:
        """Project a model entity span from merged context text to current message offsets."""
        raw_start = int(entity.get("start", 0))
        raw_end = int(entity.get("end", raw_start + len(str(entity.get("text", "")))))
        raw_span = str(entity.get("text", "")).strip()
        if raw_end <= raw_start:
            return None

        # Only keep entities fully contained in the current message segment.
        if raw_start < current_start or raw_end > current_end:
            return self._find_entity_in_text(raw_span, current_text)

        start_char = raw_start - current_start
        end_char = raw_end - current_start
        if start_char < 0 or end_char > len(current_text) or end_char <= start_char:
            return self._find_entity_in_text(raw_span, current_text)

        span_text = current_text[start_char:end_char].strip()
        if raw_span and span_text.casefold() != raw_span.casefold():
            found = self._find_entity_in_text(raw_span, current_text)
            if found is not None:
                return found
        if not span_text:
            return self._find_entity_in_text(raw_span, current_text)

        return span_text, start_char, end_char

    def _use_mlx(self) -> bool:
        """Check whether to use the MLX backend."""
        if self._backend == "pytorch":
            return False
        if self._backend == "mlx":
            return True
        # auto: try MLX, fall back to PyTorch
        try:
            import mlx.core  # noqa: F401

            return True
        except ImportError:
            return False

    def _load_mlx_model(self) -> Any:
        """Lazy-load MLX GLiNER model."""
        if self._mlx_model is not None:
            return self._mlx_model
        from models.gliner_mlx import get_mlx_gliner

        self._mlx_model = get_mlx_gliner()
        self._mlx_model.load_model()
        return self._mlx_model

    def _load_model(self) -> Any:
        """Lazy-load PyTorch GLiNER model (500MB, load once)."""
        if self._model is not None:
            return self._model
        from gliner import GLiNER

        logger.info("Loading GLiNER model: %s", self._model_name)
        self._model = GLiNER.from_pretrained(self._model_name)
        logger.info("GLiNER model loaded")
        return self._model

    def predict_raw_entities(
        self,
        text: str,
        threshold: float = 0.01,
        *,
        prev_messages: Any = None,
        next_messages: Any = None,
    ) -> list[dict[str, Any]]:
        """Run raw GLiNER prediction without local filtering.

        Args:
            text: Current input message text.
            threshold: GLiNER score threshold used at model call time.
            prev_messages: Optional previous messages for context.
            next_messages: Optional next messages for context.
        """
        merged_text, current_start, current_end = self._build_context_text(
            text,
            prev_messages=prev_messages,
            next_messages=next_messages,
        )

        if self._use_mlx():
            mlx_model = self._load_mlx_model()
            entities = mlx_model.predict_entities(
                merged_text,
                self._model_labels,
                threshold=threshold,
                flat_ner=True,
            )
        else:
            model = self._load_model()
            entities = model.predict_entities(
                merged_text,
                self._model_labels,
                threshold=threshold,
                flat_ner=True,
            )

        normalized: list[dict[str, Any]] = []
        for entity in entities:
            projected = self._project_entity_to_current(
                entity,
                current_start=current_start,
                current_end=current_end,
                current_text=text,
            )
            if projected is None:
                continue
            span_text, start_char, end_char = projected
            item = dict(entity)
            raw_label = str(item.get("label", ""))
            item["raw_label"] = raw_label
            item["label"] = self._canonicalize_label(raw_label)
            item["text"] = span_text
            item["start"] = start_char
            item["end"] = end_char
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
        prev_messages: Any = None,
        next_messages: Any = None,
        use_gate: bool = True,
    ) -> list[FactCandidate]:
        """Single message -> list of FactCandidates.

        If prev/next context is provided, GLiNER runs on merged context while
        candidates are strictly anchored to the current message span.

        Applies junk filter, message gate, per-label thresholds, vague word rejection,
        and strict dedup before returning.
        """
        if is_junk_message(text):
            return []

        if use_gate and not is_fact_likely(text, is_from_me=is_from_me or False):
            logger.debug("Gate closed for message: %s", text[:50])
            return []

        call_threshold = self._global_threshold if threshold is None else threshold
        ents = self.predict_raw_entities(
            text,
            threshold=call_threshold,
            prev_messages=prev_messages,
            next_messages=next_messages,
        )

        out: list[FactCandidate] = []
        seen: set[tuple[str, str]] = set()  # (span_casefolded, label) for dedup

        for e in ents:
            span = str(e.get("text", "")).strip()
            label = str(e.get("label", ""))
            score = float(e.get("score", 0.0))

            # Per-label threshold
            if apply_label_thresholds and score < self._per_label_min.get(label, self._default_min):
                continue

            # Vague word rejection
            if apply_vague_filter and (span.casefold() in VAGUE or len(span) < 2):
                continue

            # Entity canonicalization (e.g. "sf" -> "San Francisco")
            canonical = ENTITY_ALIASES.get(label, {}).get(span.casefold())
            if canonical:
                span = canonical

            # Strict dedup: same span + label + message_id emitted once
            dedup_key = (span.casefold(), label)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # Resolve character offsets
            start_char = int(e.get("start", 0))
            end_char = int(e.get("end", start_char + len(span)))

            fact_type = self._resolve_fact_type(text, span, label)

            # Drop unresolvable fallback spans (highest FP source)
            if fact_type == "other_personal_fact":
                continue

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

        if self._use_entailment and out:
            out = self._verify_entailment(out)

        return out

    def _process_batch_entity(
        self,
        entity: dict[str, Any],
        *,
        current_start: int,
        current_end: int,
        current_text: str,
        msg: dict[str, Any],
        seen: set[tuple[str, str]],
    ) -> FactCandidate | None:
        """Process a single GLiNER entity from a batch prediction into a FactCandidate.

        Applies projection, threshold filtering, vague rejection, canonicalization,
        dedup, and fact-type resolution. Returns None if the entity is filtered out.
        Updates *seen* in-place for dedup tracking.
        """
        projected = self._project_entity_to_current(
            entity,
            current_start=current_start,
            current_end=current_end,
            current_text=current_text,
        )
        if projected is None:
            return None
        span, start_char, end_char = projected

        label = self._canonicalize_label(str(entity.get("label", "")))
        score = float(entity.get("score", 0.0))

        if score < self._per_label_min.get(label, self._default_min):
            return None
        if span.casefold() in VAGUE or len(span) < 2:
            return None

        # Entity canonicalization (e.g. "sf" -> "San Francisco")
        canonical = ENTITY_ALIASES.get(label, {}).get(span.casefold())
        if canonical:
            span = canonical

        dedup_key = (span.casefold(), label)
        if dedup_key in seen:
            return None
        seen.add(dedup_key)

        fact_type = self._resolve_fact_type(current_text, span, label)

        # Drop unresolvable fallback spans (highest FP source)
        if fact_type == "other_personal_fact":
            return None

        return FactCandidate(
            message_id=msg["message_id"],
            span_text=span,
            span_label=label,
            gliner_score=score,
            fact_type=fact_type,
            start_char=start_char,
            end_char=end_char,
            source_text=current_text,
            chat_id=msg.get("chat_id"),
            is_from_me=msg.get("is_from_me"),
            sender_handle_id=msg.get("sender_handle_id"),
            message_date=msg.get("message_date"),
        )

    def extract_batch(
        self,
        messages: list[dict[str, Any]],
        batch_size: int = 16,
    ) -> list[FactCandidate]:
        """Batch extraction with length-sorted batching for minimal padding waste.

        Sorts messages by text length so each batch has similar-length sequences,
        reducing padding overhead on the GPU. Results are collected regardless of
        processing order.

        Each message dict should have at minimum:
            - text: str
            - message_id: int
        Optional keys: chat_id, is_from_me, sender_handle_id, message_date,
        context_prev (list[str]), context_next (list[str]).
        """
        use_mlx = self._use_mlx()
        if use_mlx:
            self._load_mlx_model()
        else:
            self._load_model()

        # Pre-filter junk messages
        valid_msgs = [m for m in messages if not is_junk_message(m.get("text", ""))]

        # Sort by text length to minimize padding waste within each batch.
        # Messages of similar length get batched together, so the GPU doesn't
        # waste compute on padding short messages to the longest one's length.
        valid_msgs.sort(key=lambda m: len(m.get("text", "")))

        all_candidates: list[FactCandidate] = []
        total = len(valid_msgs)

        for i in range(0, total, batch_size):
            batch = valid_msgs[i : i + batch_size]
            merged_texts: list[str] = []
            current_bounds: list[tuple[int, int]] = []

            for msg in batch:
                merged_text, current_start, current_end = self._build_context_text(
                    msg["text"],
                    prev_messages=msg.get("context_prev"),
                    next_messages=msg.get("context_next"),
                )
                merged_texts.append(merged_text)
                current_bounds.append((current_start, current_end))

            # GLiNER batch prediction
            if use_mlx:
                batch_entities = self._mlx_model.predict_batch(
                    merged_texts,
                    self._model_labels,
                    batch_size=len(merged_texts),
                    threshold=self._global_threshold,
                    flat_ner=True,
                )
            else:
                batch_entities = self._model.batch_predict_entities(
                    merged_texts,
                    self._model_labels,
                    threshold=self._global_threshold,
                    flat_ner=True,
                )

            for msg, bounds, ents in zip(batch, current_bounds, batch_entities):
                seen: set[tuple[str, str]] = set()
                current_start, current_end = bounds
                for e in ents:
                    candidate = self._process_batch_entity(
                        e,
                        current_start=current_start,
                        current_end=current_end,
                        current_text=msg["text"],
                        msg=msg,
                        seen=seen,
                    )
                    if candidate is not None:
                        all_candidates.append(candidate)

            processed = min(i + batch_size, total)
            logger.info(
                "Batch progress: %d/%d messages (%d candidates so far)",
                processed,
                total,
                len(all_candidates),
            )

        if self._use_entailment and all_candidates:
            all_candidates = self._verify_entailment(all_candidates)

        return all_candidates

    # ------------------------------------------------------------------
    # Entailment gate
    # ------------------------------------------------------------------

    # Per-type entailment thresholds (replaces global self._entailment_threshold).
    # Tuned via scripts/sweep_gliner_thresholds.py entailment sweep.
    # Types not listed here fall back to self._entailment_threshold (default 0.12).
    _ENTAILMENT_THRESHOLDS: dict[str, float] = {
        "preference.activity": 0.03,  # Activity hypotheses score low on NLI; very permissive
        "relationship.family": 0.05,  # Family mentions are high-signal, low NLI scores
        "preference.food_like": 0.08,
        "health.condition": 0.08,
        "health.allergy": 0.08,
        "location.current": 0.12,
        "work.employer": 0.12,
        "work.job_title": 0.12,
        "personal.school": 0.10,
        "relationship.friend": 0.25,  # person_name fallback is noisy, strict gate
        "personal.birthday": 0.20,  # date_ref fallback catches irrelevant dates
    }

    _HYPOTHESIS_TEMPLATES: dict[str, str] = {
        "relationship.family": "The message mentions a family member named {span}",
        "relationship.friend": "{span} is a friend of {subject}",
        "relationship.partner": "{span} is {poss} romantic partner",
        "location.current": "{subject} lives in {span}",
        "location.past": "{subject} used to live in {span}",
        "location.future": "{subject} is moving to {span}",
        "location.hometown": "{subject} is from {span}",
        "work.employer": "{span} is {poss} employer",
        "work.former_employer": "{subject} previously worked at {span}",
        "work.job_title": "{poss} job title is {span}",
        "preference.food_like": "{subject} likes {span}",
        "preference.food_dislike": "{subject} dislikes {span}",
        "preference.activity": "Someone does {span} as an activity or hobby",
        "health.allergy": "{subject} is allergic to {span}",
        "health.dietary": "{subject} has a dietary restriction related to {span}",
        "health.condition": "{subject} has a health condition related to {span}",
        "personal.birthday": "{poss} birthday is {span}",
        "personal.school": "{subject} attends or attended {span}",
        "personal.pet": "{subject} has a pet named {span}",
    }

    def _candidate_to_hypothesis(self, candidate: FactCandidate) -> str:
        """Convert a FactCandidate to a natural language hypothesis for NLI."""
        template = self._HYPOTHESIS_TEMPLATES.get(candidate.fact_type)
        if template:
            if candidate.is_from_me is False:
                subject, poss = "the contact", "the contact's"
            else:
                subject, poss = "the user", "the user's"
            return template.format(
                span=candidate.span_text,
                label=candidate.span_label,
                subject=subject,
                poss=poss,
            )
        return f"The message contains a fact about {candidate.span_text}"

    def _verify_entailment(
        self,
        candidates: list[FactCandidate],
    ) -> list[FactCandidate]:
        """Filter candidates by NLI entailment score with per-type thresholds."""
        if not candidates:
            return candidates

        from jarvis.nlp.entailment import verify_entailment_batch

        pairs = []
        for c in candidates:
            hypothesis = self._candidate_to_hypothesis(c)
            pairs.append((c.source_text, hypothesis))

        # Score all pairs at threshold=0 so we get raw scores back
        results = verify_entailment_batch(pairs, threshold=0.0)

        verified = []
        for candidate, (_, score) in zip(candidates, results):
            # Use per-type threshold, falling back to the global threshold
            type_threshold = self._ENTAILMENT_THRESHOLDS.get(
                candidate.fact_type, self._entailment_threshold
            )
            if score > type_threshold:
                candidate.gliner_score = min(candidate.gliner_score, score)
                verified.append(candidate)
            else:
                logger.debug(
                    "Entailment rejected: '%s' (%s) score=%.3f < %.2f",
                    candidate.span_text,
                    candidate.fact_type,
                    score,
                    type_threshold,
                )

        logger.debug(
            "Entailment gate: %d -> %d candidates",
            len(candidates),
            len(verified),
        )
        return verified

    def _resolve_fact_type(self, text: str, span: str, span_label: str) -> str:
        """Map (text_pattern, span_label) -> fact_type.

        Priority:
        1. Pattern + label match from FACT_TYPE_RULES
        2. Direct label map for fact-like labels (allergy, employer, etc.)
        3. Fallback: other_personal_fact
        """
        # Pattern-based rules (pre-compiled regexes)
        for compiled_pat, label_set, fact_type in FACT_TYPE_RULES:
            if span_label in label_set and compiled_pat.search(text):
                return fact_type

        # Direct label map for fact-like labels
        if span_label in DIRECT_LABEL_MAP:
            return DIRECT_LABEL_MAP[span_label]

        return "other_personal_fact"
