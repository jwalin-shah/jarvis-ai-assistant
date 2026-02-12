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
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

from jarvis.contacts.fact_filter import is_fact_likely
from jarvis.contacts.junk_filters import is_junk_message

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cached model singletons (shared across all CandidateExtractor
# instances). Mirrors the _get_shared_nlp() pattern in fact_extractor.py.
# ---------------------------------------------------------------------------
_gliner_mlx_model: Any = None
_gliner_pytorch_model: Any = None
_gliner_model_lock = threading.Lock()
_spacy_nlp_model: Any = None

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
    "skill",
    "cultural_event",
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
    "skill": "programming language, technical skill, or tool",
    "cultural_event": "holiday, festival, or cultural celebration",
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
    "family_member": 0.35,  # Was 0.45; "brother" at 0.391 was rejected
    "place": 0.40,  # Was 0.45; soft entailment handles precision
    "org": 0.45,  # Was 0.55; soft entailment handles precision
    "date_ref": 0.60,
    "food_item": 0.40,  # Was 0.50; soft entailment handles precision
    "job_role": 0.35,  # Was 0.40; 83% miss rate needs more recall
    "health_condition": 0.35,  # Was 0.40; 81% miss rate needs more recall
    "activity": 0.30,  # Was 0.35; 98% miss rate needs more recall
    "skill": 0.40,
    "cultural_event": 0.45,
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
# Words that GLiNER falsely tags as "activity" in sports/household context
ACTIVITY_NEGATIVE_WORDS = {
    "offense", "defense", "game", "score", "season", "league", "match",
    "cleaning", "round", "draft", "trade", "roster", "bench", "punt",
    "tackle", "block", "play", "win", "loss",
}

# Location context patterns: if a place candidate appears near these verbs,
# it's more likely to be a real location mention (not just a noun).
_LOCATION_CONTEXT_RE = re.compile(
    r"\b(?:live[sd]?|living|based|from|moving|moved|relocat(?:e|ing|ed)|"
    r"born|raised|grew up|headed|going|visit(?:ing|ed)?|trip|vacation|"
    r"fly(?:ing)?|drove|driving|stay(?:ing|ed)?|reside[sd]?|arrived|"
    r"currently|already|back|here)\s+(?:in|to|at|near)\b",
    re.IGNORECASE,
)

CONTEXT_SEPARATOR = "\n[CTX]\n"

# Profile-specific per-label minimum score overrides.
# These are merged on top of PER_LABEL_MIN.
PER_LABEL_MIN_PROFILE_OVERRIDES: dict[str, dict[str, float]] = {
    # Keep broad recall in research mode.
    "high_recall": {},
    # Slightly tighter on historically noisy labels.
    "balanced": {
        "place": 0.45,
        "org": 0.50,
        "activity": 0.40,
        "job_role": 0.50,
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
    # skill
    "preference.skill",
    # cultural
    "personal.cultural_event",
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
    "skill": "preference.skill",
    "cultural_event": "personal.cultural_event",
    # Fallback defaults for generic GLiNER labels (pattern rules take priority)
    "place": "location.current",
    "org": "work.employer",
    "person_name": "relationship.friend",
    "date_ref": "personal.birthday",
}


# ---------------------------------------------------------------------------
# High-precision regex patterns for candidate extraction
# Ported from fact_extractor.py patterns. These run alongside GLiNER
# and produce candidates with confidence >= 0.85 (skip entailment).
# ---------------------------------------------------------------------------

_REGEX_FAMILY = re.compile(
    r"\b[Mm]y\s+(sister|brother|mom|mother|dad|father|daughter|son|cousin|"
    r"aunt|uncle|[Gg]randma|[Gg]randmother|[Gg]randpa|[Gg]randfather|"
    r"roommate|boss|coworker|colleague|neighbor|friend|best friend)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
)

_REGEX_PARTNER = re.compile(
    r"\b[Mm]y\s+(wife|husband|girlfriend|boyfriend|partner|fiancée?)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
)

_REGEX_WORK = re.compile(
    r"\b(?:work(?:s|ing|ed)?\s+(?:at|for)|started\s+(?:at|with)|"
    r"joined|hired\s+(?:at|by|with))\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
)

# Family member without a trailing capitalized name:
# "my brother is coming" / "my mom said" / "talked to my sister"
_REGEX_FAMILY_NONAME = re.compile(
    r"\b[Mm]y\s+(sister|brother|mom|mother|dad|father|daughter|son|cousin|"
    r"aunt|uncle|[Gg]randma|[Gg]randmother|[Gg]randpa|[Gg]randfather)\b"
    r"(?!\s+[A-Z])",  # negative lookahead: not followed by a capitalized name
)

_REGEX_LOCATION_CURRENT = re.compile(
    r"\b(?:[Ll]ive[sd]?\s+in|[Ll]iving\s+in|[Bb]ased\s+in|[Cc]urrently\s+in)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
)

_REGEX_LOCATION_FUTURE = re.compile(
    r"\b(?:[Mm]oving\s+to|[Rr]elocating\s+to)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
)


# ---------------------------------------------------------------------------
# Profile boosting: known entities get a score boost
# ---------------------------------------------------------------------------

_known_entities_cache: dict[str, tuple[float, set[str]]] = {}
_KNOWN_ENTITIES_TTL = 300.0  # 5 minutes


def _get_known_entities(contact_id: str | None) -> set[str]:
    """Return lowercased entity texts for a contact's high-confidence facts.

    Uses a TTL cache (5 minutes) to avoid repeated DB queries.
    """
    if not contact_id:
        return set()

    now = time.monotonic()
    cached = _known_entities_cache.get(contact_id)
    if cached is not None:
        ts, entities = cached
        if now - ts < _KNOWN_ENTITIES_TTL:
            return entities

    try:
        from jarvis.contacts.fact_storage import get_facts_for_contact

        facts = get_facts_for_contact(contact_id)
        entities = {
            f.subject.lower()
            for f in facts
            if f.confidence is not None and f.confidence >= 0.7
        }
        _known_entities_cache[contact_id] = (now, entities)
        return entities
    except Exception:
        logger.debug("Failed to load known entities for %s", contact_id)
        return set()


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
        use_entailment: bool = False,
        entailment_threshold: float = 0.12,
        use_llm_verifier: bool = False,
    ):
        self._model_name = model_name
        self._backend = backend  # "auto", "mlx", or "pytorch"
        self._label_profile = label_profile or "high_recall"
        self._use_llm_verifier = use_llm_verifier

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
    def _find_entity_in_text(raw_span: str, current_text: str) -> tuple[str, int, int] | None:
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
        """Lazy-load MLX GLiNER model (module-level singleton, thread-safe).

        Also stores the model as ``self._mlx_model`` so callers (and tests) can
        reference it via the instance attribute.
        """
        # Allow tests to inject a mock via self._mlx_model
        if getattr(self, "_mlx_model", None) is not None:
            return self._mlx_model
        global _gliner_mlx_model
        if _gliner_mlx_model is not None:
            self._mlx_model = _gliner_mlx_model
            return _gliner_mlx_model
        with _gliner_model_lock:
            if _gliner_mlx_model is not None:
                self._mlx_model = _gliner_mlx_model
                return _gliner_mlx_model
            from models.gliner_mlx import get_mlx_gliner

            _gliner_mlx_model = get_mlx_gliner()
            _gliner_mlx_model.load_model()
        self._mlx_model = _gliner_mlx_model
        return _gliner_mlx_model

    def _load_model(self) -> Any:
        """Lazy-load PyTorch GLiNER model (module-level singleton, thread-safe).

        Also stores the model as ``self._model`` so callers (and tests) can
        reference it via the instance attribute.
        """
        # Allow tests to inject a mock via self._model
        if getattr(self, "_model", None) is not None:
            return self._model
        global _gliner_pytorch_model
        if _gliner_pytorch_model is not None:
            self._model = _gliner_pytorch_model
            return _gliner_pytorch_model
        with _gliner_model_lock:
            if _gliner_pytorch_model is not None:
                self._model = _gliner_pytorch_model
                return _gliner_pytorch_model
            from gliner import GLiNER

            logger.info("Loading GLiNER model: %s", self._model_name)
            _gliner_pytorch_model = GLiNER.from_pretrained(self._model_name)
            logger.info("GLiNER model loaded")
        self._model = _gliner_pytorch_model
        return _gliner_pytorch_model

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
            self._load_mlx_model()
            entities = self._mlx_model.predict_entities(
                merged_text,
                self._model_labels,
                threshold=threshold,
                flat_ner=True,
            )
        else:
            self._load_model()
            entities = self._model.predict_entities(
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

    @staticmethod
    def _regex_extract(text: str, message_id: int, **kwargs: Any) -> list[FactCandidate]:
        """Extract high-precision candidates using regex patterns.

        Returns candidates with gliner_score=0.85 that skip entailment verification.
        """
        candidates: list[FactCandidate] = []
        seen: set[tuple[str, str]] = set()

        def _add(span: str, label: str, fact_type: str, start: int, end: int) -> None:
            key = (span.casefold(), label)
            if key in seen:
                return
            seen.add(key)
            candidates.append(
                FactCandidate(
                    message_id=message_id,
                    span_text=span,
                    span_label=label,
                    gliner_score=0.85,
                    fact_type=fact_type,
                    start_char=start,
                    end_char=end,
                    source_text=text,
                    **kwargs,
                )
            )

        # Family with name: "my sister Sarah"
        for m in _REGEX_FAMILY.finditer(text):
            name = m.group(2)
            _add(name, "family_member", "relationship.family", m.start(2), m.end(2))

        # Family without name: "my brother is coming"
        for m in _REGEX_FAMILY_NONAME.finditer(text):
            relation = m.group(1)
            key = (relation.casefold(), "family_member")
            if key not in seen:
                seen.add(key)
                candidates.append(
                    FactCandidate(
                        message_id=message_id,
                        span_text=relation,
                        span_label="family_member",
                        gliner_score=0.80,
                        fact_type="relationship.family",
                        start_char=m.start(1),
                        end_char=m.end(1),
                        source_text=text,
                        **kwargs,
                    )
                )

        # Partner: "my girlfriend Alex"
        for m in _REGEX_PARTNER.finditer(text):
            name = m.group(2)
            _add(name, "partner_name", "relationship.partner", m.start(2), m.end(2))

        # Location current: "live in Austin"
        for m in _REGEX_LOCATION_CURRENT.finditer(text):
            place = m.group(1)
            _add(place, "place", "location.current", m.start(1), m.end(1))

        # Location future: "moving to LA"
        for m in _REGEX_LOCATION_FUTURE.finditer(text):
            place = m.group(1)
            _add(place, "place", "location.future", m.start(1), m.end(1))

        # Work: "work at Google"
        for m in _REGEX_WORK.finditer(text):
            org = m.group(1)
            _add(org, "org", "work.employer", m.start(1), m.end(1))

        return candidates

    def extract_candidates(
        self,
        text: str,
        message_id: int,
        *,
        chat_id: int | None = None,
        is_from_me: bool | None = None,
        sender_handle_id: int | None = None,
        message_date: int | None = None,
        contact_id: str | None = None,
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
        if is_junk_message(text, chat_id=str(chat_id) if chat_id else ""):
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

            # Activity negative word filter (sports/household terms)
            if label == "activity" and span.casefold() in ACTIVITY_NEGATIVE_WORDS:
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

            # Place context scoring: penalize place candidates without
            # location-indicating verbs nearby (soft penalty, not hard reject)
            if label == "place" and not _LOCATION_CONTEXT_RE.search(text):
                score *= 0.6

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

        # Profile boosting: boost score for entities already known for this contact
        known = _get_known_entities(contact_id)
        if known:
            for c in out:
                if c.span_text.lower() in known:
                    c.gliner_score = min(1.0, c.gliner_score + 0.08)

        # Merge regex candidates (high-precision, skip entailment)
        extra_kwargs: dict[str, Any] = {}
        if chat_id is not None:
            extra_kwargs["chat_id"] = chat_id
        if is_from_me is not None:
            extra_kwargs["is_from_me"] = is_from_me
        if sender_handle_id is not None:
            extra_kwargs["sender_handle_id"] = sender_handle_id
        if message_date is not None:
            extra_kwargs["message_date"] = message_date
        regex_candidates = self._regex_extract(text, message_id, **extra_kwargs)

        # Regex takes priority on overlap (higher precision)
        regex_keys = {(c.span_text.casefold(), c.span_label) for c in regex_candidates}
        gliner_only = [c for c in out if (c.span_text.casefold(), c.span_label) not in regex_keys]

        # Only GLiNER candidates go through verification
        if self._use_llm_verifier and gliner_only:
            from jarvis.contacts.llm_fact_verifier import LLMFactVerifier

            gliner_only = LLMFactVerifier().verify_candidates(gliner_only)
        elif self._use_entailment and gliner_only:
            gliner_only = self._verify_entailment(gliner_only)

        return regex_candidates + gliner_only

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

        # Activity negative word filter (sports/household terms)
        if label == "activity" and span.casefold() in ACTIVITY_NEGATIVE_WORDS:
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

        # Place context scoring: penalize place candidates without
        # location-indicating verbs nearby
        if label == "place" and not _LOCATION_CONTEXT_RE.search(current_text):
            score *= 0.6

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

        # Pre-filter junk messages (pass chat_id for short-code detection)
        valid_msgs = [
            m for m in messages
            if not is_junk_message(m.get("text", ""), chat_id=str(m.get("chat_id", "")))
        ]

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

        # Run regex extraction on all messages (high-precision, skip entailment)
        all_regex: list[FactCandidate] = []
        for msg in valid_msgs:
            extra_kwargs: dict[str, Any] = {}
            if msg.get("chat_id") is not None:
                extra_kwargs["chat_id"] = msg["chat_id"]
            if msg.get("is_from_me") is not None:
                extra_kwargs["is_from_me"] = msg["is_from_me"]
            if msg.get("sender_handle_id") is not None:
                extra_kwargs["sender_handle_id"] = msg["sender_handle_id"]
            if msg.get("message_date") is not None:
                extra_kwargs["message_date"] = msg["message_date"]
            all_regex.extend(
                self._regex_extract(msg["text"], msg["message_id"], **extra_kwargs)
            )

        # Regex takes priority on overlap
        regex_keys = {
            (c.message_id, c.span_text.casefold(), c.span_label)
            for c in all_regex
        }
        gliner_only = [
            c for c in all_candidates
            if (c.message_id, c.span_text.casefold(), c.span_label) not in regex_keys
        ]

        if self._use_llm_verifier and gliner_only:
            from jarvis.contacts.llm_fact_verifier import LLMFactVerifier

            gliner_only = LLMFactVerifier().verify_candidates(gliner_only)
        elif self._use_entailment and gliner_only:
            gliner_only = self._verify_entailment(gliner_only)

        return all_regex + gliner_only

    # ------------------------------------------------------------------
    # Entailment gate
    # ------------------------------------------------------------------

    # Per-type entailment thresholds (replaces global self._entailment_threshold).
    # Tuned via scripts/sweep_gliner_thresholds.py entailment sweep.
    # Types not listed here fall back to self._entailment_threshold (default 0.12).
    # Uniform low threshold (0.1) after fixing hypothesis templates to MNLI-style.
    # Previous per-type thresholds were tuned against broken formal templates that
    # scored ~0 for most types. Revisit after observing real score distributions.
    _ENTAILMENT_THRESHOLDS: dict[str, float] = {
        # Work: high bar - filter recruiters, spam, medical facilities
        "work.employer": 0.45,
        "work.former_employer": 0.45,
        "work.job_title": 0.55,
        # Location: moderate - valid at 0.38+ but noise at 0.10-0.18
        "location.current": 0.30,
        "location.past": 0.30,
        "location.future": 0.20,
        "location.hometown": 0.30,
        # Preferences: filter list mentions (0.10-0.14), keep clear prefs (0.50+)
        "preference.food_like": 0.35,
        "preference.food_dislike": 0.35,
        "preference.activity": 0.30,
        # Relationships: filter pronouns/generic terms, keep named people
        "relationship.family": 0.30,
        "relationship.friend": 0.30,
        "relationship.partner": 0.30,
        # Health/personal: conservative, high-value facts
        "health.condition": 0.45,
        "health.allergy": 0.45,
        "health.dietary": 0.45,
        "personal.school": 0.55,
        "personal.birthday": 0.45,
        "personal.pet": 0.45,
    }

    # MNLI-style hypotheses: simple "Someone [verb] [entity]" patterns.
    # DeBERTa-v3-xsmall was trained on MNLI which uses direct, impersonal
    # hypotheses. Formal/possessive templates (e.g. "{poss} employer") scored
    # ~0 because they don't match the MNLI training distribution.
    # MNLI-style hypotheses tuned for casual chat premises.
    # Key principles:
    # - Use weak predicates the premise can actually entail
    # - Avoid editorial injection ("as a hobby", "suffers from")
    # - Match the verb patterns commonly found in messages
    _HYPOTHESIS_TEMPLATES: dict[str, str] = {
        "relationship.family": "{span} is a family member",
        "relationship.friend": "{span} is a friend",
        "relationship.partner": "{span} is a romantic partner",
        "location.current": "Someone lives in {span}",
        "location.past": "Someone used to live in {span}",
        "location.future": "Someone plans to move to {span}",
        "location.hometown": "Someone grew up in {span}",
        "work.employer": "Someone is employed at {span}",
        "work.former_employer": "Someone used to work at {span}",
        "work.job_title": "Someone works in {span}",  # was "works as a" (grammar)
        "preference.food_like": "Someone likes {span}",  # was "enjoys eating" (too strong)
        "preference.food_dislike": "Someone dislikes {span}",
        "preference.activity": "Someone does {span}",  # was "enjoys as a hobby" (editorial)
        "health.allergy": "Someone is allergic to {span}",
        "health.dietary": "Someone avoids eating {span}",
        "health.condition": "Someone has {span}",  # was "suffers from" (too clinical)
        "personal.birthday": "Someone's birthday is {span}",
        "personal.school": "Someone is a student at {span}",
        "personal.pet": "Someone has a pet named {span}",
    }

    # Categories where NLI consistently destroys recall with no precision benefit.
    # These skip entailment entirely and rely on GLiNER score + regex patterns.
    _NLI_SKIP_CATEGORIES: set[str] = {
        "preference.activity",    # 98% rejection rate, 0% recall after NLI
        "preference.food_like",   # kills recall for modest precision gain
        "preference.food_dislike",
        "health.condition",       # 84% rejection rate
        "health.allergy",         # same template problem as health.condition
        "preference.skill",       # new label, no NLI training signal
        "personal.cultural_event",  # new label, no NLI training signal
    }

    def _candidate_to_hypothesis(self, candidate: FactCandidate) -> str:
        """Convert a FactCandidate to a natural language hypothesis for NLI."""
        template = self._HYPOTHESIS_TEMPLATES.get(candidate.fact_type)
        if template:
            return template.format(span=candidate.span_text)
        return f"The message mentions {candidate.span_text}"

    def _verify_entailment(
        self,
        candidates: list[FactCandidate],
    ) -> list[FactCandidate]:
        """Entailment verification with category skipping and E-C scoring.

        Uses entailment - contradiction (E-C) score instead of raw entailment.
        This accepts neutral mentions (not contradicted) instead of rejecting them.
        Categories in _NLI_SKIP_CATEGORIES bypass NLI entirely.
        """
        if not candidates:
            return candidates

        # Split candidates: some skip NLI entirely
        needs_nli = []
        skip_nli = []
        for c in candidates:
            if c.fact_type in self._NLI_SKIP_CATEGORIES:
                skip_nli.append(c)
            else:
                needs_nli.append(c)

        if not needs_nli:
            return skip_nli + needs_nli

        from models.nli_cross_encoder import get_nli_cross_encoder

        nli = get_nli_cross_encoder()
        pairs = [
            (c.source_text, self._candidate_to_hypothesis(c))
            for c in needs_nli
        ]
        # Get full score dicts (contradiction, entailment, neutral)
        all_scores = nli.predict_batch(pairs)

        verified = []
        for candidate, scores in zip(needs_nli, all_scores):
            # E-C score: positive means entailment > contradiction
            # Neutral mentions (high neutral, low both) get near-zero scores
            ec_score = scores["entailment"] - scores["contradiction"]

            if ec_score < -0.5:
                # Hard reject: clear contradiction dominates
                logger.debug(
                    "Entailment hard-reject: '%s' (%s) E=%.3f C=%.3f EC=%.3f",
                    candidate.span_text,
                    candidate.fact_type,
                    scores["entailment"],
                    scores["contradiction"],
                    ec_score,
                )
                continue

            # Soft multiplier based on EC score mapped to [0.4, 1.0]
            # EC range is roughly [-1, 1], map to multiplier:
            # EC=-0.5 -> 0.4, EC=0.0 -> 0.7, EC=0.5 -> 1.0
            multiplier = max(0.4, min(1.0, 0.7 + 0.6 * ec_score))
            candidate.gliner_score *= multiplier
            verified.append(candidate)

        logger.debug(
            "Entailment: %d skip, %d scored (%d kept, %d rejected)",
            len(skip_nli),
            len(needs_nli),
            len(verified),
            len(needs_nli) - len(verified),
        )
        return skip_nli + verified

    # ------------------------------------------------------------------
    # spaCy NER extraction (high-recall for common entity types)
    # ------------------------------------------------------------------

    _SPACY_LABEL_MAP: dict[str, str] = {
        "PERSON": "person_name",
        "ORG": "org",
        "GPE": "place",
        "LOC": "place",
        "FAC": "place",
    }

    def extract_spacy_candidates(
        self,
        text: str,
        message_id: int,
        *,
        chat_id: int | None = None,
        is_from_me: bool | None = None,
        sender_handle_id: int | None = None,
        message_date: int | None = None,
    ) -> list[FactCandidate]:
        """Extract candidates using spaCy NER only.

        Maps spaCy entity labels (PERSON, ORG, GPE, LOC, FAC) to our taxonomy.
        Returns FactCandidate objects with gliner_score=0.50 (moderate confidence).
        """
        nlp = self._get_spacy_nlp()
        if nlp is None:
            return []

        doc = nlp(text)
        candidates: list[FactCandidate] = []
        seen: set[tuple[str, str]] = set()

        for ent in doc.ents:
            label = self._SPACY_LABEL_MAP.get(ent.label_)
            if label is None:
                continue
            span_text = ent.text.strip().strip(".,!?;:'\"()[]{}").strip()
            if len(span_text) < 2 or span_text.casefold() in VAGUE:
                continue

            key = (span_text.casefold(), label)
            if key in seen:
                continue
            seen.add(key)

            fact_type = self._resolve_fact_type(text, span_text, label)
            if fact_type == "other_personal_fact":
                continue

            candidates.append(
                FactCandidate(
                    message_id=message_id,
                    span_text=span_text,
                    span_label=label,
                    gliner_score=0.50,
                    fact_type=fact_type,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    source_text=text,
                    chat_id=chat_id,
                    is_from_me=is_from_me,
                    sender_handle_id=sender_handle_id,
                    message_date=message_date,
                )
            )

        return candidates

    @staticmethod
    def _get_spacy_nlp() -> Any:
        """Lazy-load spaCy en_core_web_sm model (singleton)."""
        global _spacy_nlp_model
        if _spacy_nlp_model is not None:
            return _spacy_nlp_model
        try:
            import spacy
            _spacy_nlp_model = spacy.load("en_core_web_sm")
            return _spacy_nlp_model
        except (ImportError, OSError):
            logger.warning("spaCy en_core_web_sm not available; spaCy extraction disabled")
            return None

    # ------------------------------------------------------------------
    # LLM extraction (nuanced, context-aware for complex entity types)
    # ------------------------------------------------------------------

    def extract_llm_candidates(
        self,
        text: str,
        message_id: int,
        *,
        chat_id: int | None = None,
        is_from_me: bool | None = None,
        sender_handle_id: int | None = None,
        message_date: int | None = None,
        context_prev: str = "",
        context_next: str = "",
        few_shot_examples: list[dict[str, Any]] | None = None,
    ) -> list[FactCandidate]:
        """Extract candidates using LLM with structured prompt.

        Handles entity types that spaCy struggles with: family_member, activity,
        health_condition, food_item, job_role.
        """

        system_prompt = self._build_llm_extraction_prompt(few_shot_examples or [])
        user_prompt = self._build_llm_user_prompt(text, context_prev, context_next)

        try:
            from models.generator import get_generator
            gen = get_generator()
            response = gen.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=512,
                temperature=0.1,
            )
            spans = self._parse_llm_spans(response)
        except Exception as e:
            logger.warning("LLM extraction failed: %s", e)
            return []

        candidates: list[FactCandidate] = []
        seen: set[tuple[str, str]] = set()
        for span in spans:
            span_text = span.get("span_text", "").strip()
            span_label = span.get("span_label", "").strip()
            if not span_text or not span_label or len(span_text) < 2:
                continue
            if span_text.casefold() in VAGUE:
                continue

            key = (span_text.casefold(), span_label)
            if key in seen:
                continue
            seen.add(key)

            fact_type = self._resolve_fact_type(text, span_text, span_label)

            # Find character offsets in source text
            idx = text.casefold().find(span_text.casefold())
            start_char = idx if idx >= 0 else 0
            end_char = start_char + len(span_text) if idx >= 0 else len(span_text)

            candidates.append(
                FactCandidate(
                    message_id=message_id,
                    span_text=span_text,
                    span_label=span_label,
                    gliner_score=0.65,
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

        return candidates

    @staticmethod
    def _build_llm_extraction_prompt(few_shot_examples: list[dict[str, Any]]) -> str:
        """Build system prompt for LLM-based extraction."""
        import json as _json

        prompt = (
            "You are a personal fact extractor. Given an iMessage, extract lasting "
            "personal facts as structured spans.\n\n"
            "Labels: family_member, person_name, place, org, job_role, food_item, "
            "activity, health_condition\n\n"
            "Rules:\n"
            "- Only extract LASTING personal facts (not transient events)\n"
            "- Extract minimal spans (just the entity, not surrounding words)\n"
            "- Skip vague references (it, that, stuff)\n"
            "- Output JSON array of {span_text, span_label} or empty array []\n"
        )
        if few_shot_examples:
            prompt += "\nExamples:\n"
            for ex in few_shot_examples[:5]:
                msg = ex.get("message_text", "")
                cands = ex.get("expected_candidates", [])
                prompt += f'Message: "{msg}"\nOutput: {_json.dumps(cands)}\n\n'
        return prompt

    @staticmethod
    def _build_llm_user_prompt(
        text: str, context_prev: str = "", context_next: str = "",
    ) -> str:
        """Build user prompt for LLM extraction."""
        parts = []
        if context_prev:
            parts.append(f"Previous: {context_prev}")
        parts.append(f'Message: "{text}"')
        if context_next:
            parts.append(f"Next: {context_next}")
        parts.append(
            "\nExtract lasting personal facts as JSON array of "
            "{span_text, span_label}. Output [] if none."
        )
        return "\n".join(parts)

    @staticmethod
    def _parse_llm_spans(response: str) -> list[dict[str, str]]:
        """Parse LLM response into span dicts with JSON fallbacks."""
        import json as _json

        response = response.strip()

        # Try direct parse
        try:
            result = _json.loads(response)
            if isinstance(result, list):
                return [s for s in result if isinstance(s, dict) and "span_text" in s]
        except _json.JSONDecodeError:
            pass

        # Try extracting from code block
        code_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
        if code_match:
            try:
                result = _json.loads(code_match.group(1))
                if isinstance(result, list):
                    return [s for s in result if isinstance(s, dict) and "span_text" in s]
            except _json.JSONDecodeError:
                pass

        # Try finding array
        bracket_match = re.search(r"\[.*\]", response, re.DOTALL)
        if bracket_match:
            try:
                result = _json.loads(bracket_match.group(0))
                if isinstance(result, list):
                    return [s for s in result if isinstance(s, dict) and "span_text" in s]
            except _json.JSONDecodeError:
                pass

        logger.warning("Could not parse LLM response: %s", response[:100])
        return []

    # ------------------------------------------------------------------
    # Hybrid extraction (spaCy + LLM merged)
    # ------------------------------------------------------------------

    def extract_hybrid(
        self,
        text: str,
        message_id: int,
        *,
        chat_id: int | None = None,
        is_from_me: bool | None = None,
        sender_handle_id: int | None = None,
        message_date: int | None = None,
        context_prev: str = "",
        context_next: str = "",
        few_shot_examples: list[dict[str, Any]] | None = None,
    ) -> list[FactCandidate]:
        """Hybrid extraction: union of spaCy + LLM candidates, deduplicated.

        spaCy handles PERSON, ORG, GPE/LOC entities with high recall.
        LLM handles nuanced types: family_member, activity, health_condition, food_item.
        Regex patterns provide high-precision anchors.

        Dedup: on span overlap (Jaccard > 0.5 = same entity), keep higher confidence.
        """
        common_kwargs: dict[str, Any] = {
            "chat_id": chat_id,
            "is_from_me": is_from_me,
            "sender_handle_id": sender_handle_id,
            "message_date": message_date,
        }

        # Layer 1: Regex (highest precision, ~0.85 confidence)
        extra_kwargs: dict[str, Any] = {}
        if chat_id is not None:
            extra_kwargs["chat_id"] = chat_id
        if is_from_me is not None:
            extra_kwargs["is_from_me"] = is_from_me
        if sender_handle_id is not None:
            extra_kwargs["sender_handle_id"] = sender_handle_id
        if message_date is not None:
            extra_kwargs["message_date"] = message_date
        regex_cands = self._regex_extract(text, message_id, **extra_kwargs)

        # Layer 2: spaCy NER
        spacy_cands = self.extract_spacy_candidates(
            text, message_id, **common_kwargs,
        )

        # Layer 3: LLM
        llm_cands = self.extract_llm_candidates(
            text,
            message_id,
            context_prev=context_prev,
            context_next=context_next,
            few_shot_examples=few_shot_examples,
            **common_kwargs,
        )

        # Merge with priority: regex > LLM > spaCy
        merged = list(regex_cands)
        merged_keys = {(c.span_text.casefold(), c.span_label) for c in merged}

        for cand in llm_cands:
            key = (cand.span_text.casefold(), cand.span_label)
            if key not in merged_keys and not self._overlaps_existing(cand, merged):
                merged.append(cand)
                merged_keys.add(key)

        for cand in spacy_cands:
            key = (cand.span_text.casefold(), cand.span_label)
            if key not in merged_keys and not self._overlaps_existing(cand, merged):
                merged.append(cand)
                merged_keys.add(key)

        return merged

    @staticmethod
    def _overlaps_existing(
        candidate: FactCandidate,
        existing: list[FactCandidate],
        jaccard_threshold: float = 0.5,
    ) -> bool:
        """Check if a candidate overlaps with any existing candidate."""
        for ex in existing:
            if candidate.span_label != ex.span_label:
                continue
            # Token-level Jaccard
            tokens_a = set(candidate.span_text.lower().split())
            tokens_b = set(ex.span_text.lower().split())
            if not tokens_a or not tokens_b:
                continue
            jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
            if jaccard >= jaccard_threshold:
                return True
            # Substring containment
            if (
                candidate.span_text.lower() in ex.span_text.lower()
                or ex.span_text.lower() in candidate.span_text.lower()
            ):
                return True
        return False

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
