"""NuExtract adapter for the extractor bakeoff.

NuExtract uses a schema-driven approach with an LLM for structured extraction.
Runs locally on Apple Silicon via MLX (no API calls needed).

Model: numind/NuExtract-1.5-smol (SmolLM2-1.7B fine-tune, ~960MB quantized)
Prompt format: <|input|> / <|output|> markers with JSON template.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

from jarvis.contacts.extractors.base import (
    ExtractedCandidate,
    ExtractionResult,
    ExtractorAdapter,
    register_extractor,
)

logger = logging.getLogger(__name__)

# Fact type resolution (same as other adapters)
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

VAGUE = {"it", "this", "that", "thing", "stuff", "them", "there", "here", "me", "you"}

# NuExtract JSON template - numbered slots allow multiple entities per field.
# Unused slots are returned as empty strings (no hallucination).
# 3 slots per high-cardinality field, 2 for the rest.
NUEXTRACT_TEMPLATE = {
    "person_name_1": "",
    "person_name_2": "",
    "person_name_3": "",
    "family_member_1": "",
    "family_member_2": "",
    "place_1": "",
    "place_2": "",
    "org_1": "",
    "org_2": "",
    "date_ref": "",
    "food_item_1": "",
    "food_item_2": "",
    "job_role": "",
    "health_condition": "",
    "activity_1": "",
    "activity_2": "",
    "activity_3": "",
}

# Map numbered slot keys back to canonical label names
_SLOT_TO_LABEL: dict[str, str] = {}
for _label in [
    "person_name",
    "family_member",
    "place",
    "org",
    "date_ref",
    "food_item",
    "job_role",
    "health_condition",
    "activity",
]:
    _SLOT_TO_LABEL[_label] = _label
    for _i in range(1, 4):
        _SLOT_TO_LABEL[f"{_label}_{_i}"] = _label

# Default MLX model path (4-bit quantized, ~960MB)
DEFAULT_MLX_MODEL_PATH = "models/nuextract-1.5-smol-mlx-4bit"


class NuExtractAdapter(ExtractorAdapter):
    """Adapter for NuExtract schema-driven extractor via MLX.

    Runs NuExtract-1.5-smol locally on Apple Silicon using mlx_lm.
    Uses the NuExtract prompt format: <|input|> / <|output|> markers.

    Configuration options:
        - model: MLX model path (default: models/nuextract-1.5-smol-mlx-4bit)
        - max_tokens: Max generation tokens (default: 512)
        - threshold: Confidence threshold (default: 0.7, unused for LLM)
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

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("nuextract", config)
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_path = self.config.get("model", DEFAULT_MLX_MODEL_PATH)
        self._max_tokens = self.config.get("max_tokens", 512)
        self._global_threshold = self.config.get("threshold", 0.7)

    @property
    def supported_labels(self) -> list[str]:
        return list(self.SPAN_LABELS)

    @property
    def default_threshold(self) -> float:
        return self._global_threshold

    def _load_model(self) -> Any:
        """Load the NuExtract model via mlx_lm."""
        if self._model is not None:
            return self._model

        model_path = Path(self._model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"NuExtract MLX model not found at {model_path}. "
                'Convert it with: uv run python -c "'
                "from mlx_lm.convert import convert; "
                "convert('numind/NuExtract-1.5-smol', "
                f"mlx_path='{model_path}', quantize=True, q_bits=4)\""
            )

        import mlx.core as mx
        from mlx_lm import load

        logger.info("Loading NuExtract MLX model from %s", model_path)

        # Use shared GPU lock from MLXModelLoader if available
        gpu_lock = self._get_gpu_lock()
        with gpu_lock:
            mx.set_memory_limit(1 * 1024 * 1024 * 1024)  # 1GB
            mx.set_cache_limit(512 * 1024 * 1024)  # 512MB
            self._model, self._tokenizer = load(str(model_path))

        logger.info("NuExtract MLX model loaded")
        return self._model

    @staticmethod
    def _get_gpu_lock() -> threading.Lock:
        """Get the shared MLX GPU lock."""
        try:
            from models.loader import MLXModelLoader

            return MLXModelLoader.gpu_lock()
        except ImportError:
            # Fallback: standalone lock (bakeoff script without full jarvis)
            if not hasattr(NuExtractAdapter, "_fallback_lock"):
                NuExtractAdapter._fallback_lock = threading.Lock()
            return NuExtractAdapter._fallback_lock

    def _build_prompt(self, text: str) -> str:
        """Build the NuExtract prompt with the official format."""
        template_json = json.dumps(NUEXTRACT_TEMPLATE, indent=2)
        return f"<|input|>\n### Template:\n{template_json}\n### Text:\n{text}\n\n<|output|>\n"

    def _generate(self, prompt: str) -> str:
        """Run MLX generation with GPU lock."""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        # temp=0.0 for deterministic extraction
        sampler = make_sampler(temp=0.0)

        gpu_lock = self._get_gpu_lock()
        with gpu_lock:
            response = generate(
                model=self._model,
                tokenizer=self._tokenizer,
                prompt=prompt,
                max_tokens=self._max_tokens,
                sampler=sampler,
                verbose=False,
            )
        return response

    def _parse_response(self, response: str, text: str) -> list[ExtractedCandidate]:
        """Parse NuExtract JSON response into candidates."""
        candidates: list[ExtractedCandidate] = []

        # NuExtract outputs JSON directly after <|output|>
        json_str = response.strip()

        # If wrapped in code fences, extract
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"(\{[\s\S]*\})", json_str)
            if json_match:
                json_str = json_match.group(1)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.debug("Failed to parse NuExtract JSON: %s", response[:200])
            return candidates

        seen: set[tuple[str, str]] = set()

        for slot_key, raw_value in data.items():
            label = _SLOT_TO_LABEL.get(slot_key)
            if label is None or label not in self.SPAN_LABELS:
                continue
            if not raw_value:
                continue

            # Value can be a string (possibly comma-separated) or a list
            if isinstance(raw_value, list):
                span_parts = [str(v) for v in raw_value if v]
            elif isinstance(raw_value, str):
                span_parts = [raw_value]
            else:
                continue

            for span_raw in span_parts:
                # NuExtract may return comma-separated values
                for span in span_raw.split(","):
                    span = span.strip()
                    if not span or span.casefold() in VAGUE or len(span) < 2:
                        continue

                    dedup_key = (span.casefold(), label)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    idx = text.casefold().find(span.casefold())
                    if idx < 0:
                        continue

                    fact_type = self._resolve_fact_type(text, span, label)

                    candidates.append(
                        ExtractedCandidate(
                            span_text=span,
                            span_label=label,
                            score=0.85,
                            start_char=idx,
                            end_char=idx + len(span),
                            fact_type=fact_type,
                            extractor_metadata={"source": "nuextract"},
                        )
                    )

        return candidates

    def _resolve_fact_type(self, text: str, span: str, span_label: str) -> str:
        """Map (text_pattern, span_label) -> fact_type."""
        for pattern, label_set, fact_type in FACT_TYPE_RULES:
            if span_label in label_set and re.search(pattern, text, re.IGNORECASE):
                return fact_type

        if span_label in DIRECT_LABEL_MAP:
            return DIRECT_LABEL_MAP[span_label]

        return "other_personal_fact"

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
        """Extract candidates using NuExtract via MLX."""
        self._load_model()

        prompt = self._build_prompt(text)

        try:
            response = self._generate(prompt)
            return self._parse_response(response, text)
        except Exception as e:
            logger.error("NuExtract extraction failed: %s", e)
            return []

    def extract_batch(
        self,
        messages: list[dict[str, Any]],
        batch_size: int = 1,
        threshold: float | None = None,
    ) -> list[ExtractionResult]:
        """Batch extraction - sequential since LLM generates one at a time."""
        return super().extract_batch(messages, batch_size=1, threshold=threshold)


# Register the adapter
register_extractor("nuextract", NuExtractAdapter)
