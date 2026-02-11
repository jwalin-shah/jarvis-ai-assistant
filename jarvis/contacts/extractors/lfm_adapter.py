"""LFM adapter for the extractor bakeoff.

Uses LiquidAI's LFM models for JSON entity extraction via direct prompting.
Runs locally on Apple Silicon via MLX (no API calls needed).

Models:
- LFM 1.2B: LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit (~700MB)
- LFM 350M: mlx-community/LFM2-350M-4bit (~200MB)
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any

from jarvis.contacts.extractors.base import (
    ExtractedCandidate,
    ExtractionResult,
    ExtractorAdapter,
    register_extractor,
)
from jarvis.contacts.fact_filter import is_fact_likely

logger = logging.getLogger(__name__)

# Model registry
LFM_MODELS: dict[str, str] = {
    "lfm-1.2b": "LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit",
    "lfm-0.3b": "mlx-community/LFM2-350M-4bit",
}

SYSTEM_PROMPT = (
    "You are a high-precision personal fact extractor. "
    "Extract structured personal facts (locations, work, preferences, relationships) from chat messages. "
    "For each fact, identify the subject, relationship, and value. "
    "Output ONLY a JSON array of objects, nothing else."
)

USER_PROMPT_TEMPLATE = (
    "Extract personal facts from this message as a JSON array.\n\n"
    "Schema:\n"
    "- subject: Who the fact is about (usually 'I' or a family member/friend)\n"
    "- predicate: The relationship (lives_in, works_at, likes, is_family_of, allergic_to, etc.)\n"
    "- object: The value/entity (Austin, Google, Pizza, etc.)\n"
    "- span: The exact text from the message supporting this fact\n\n"
    "Example:\n"
    'Message: "My brother works at Google and loves hiking"\n'
    '[\n'
    '  {{"subject": "brother", "predicate": "works_at", "object": "Google", "span": "works at Google"}},\n'
    '  {{"subject": "brother", "predicate": "likes", "object": "hiking", "span": "loves hiking"}}\n'
    ']\n\n'
    "Example:\n"
    'Message: "I live in Austin now"\n'
    '[\n'
    '  {{"subject": "I", "predicate": "lives_in", "object": "Austin", "span": "live in Austin"}}\n'
    ']\n\n'
    "Example:\n"
    'Message: "sounds good see you at 5"\n'
    '[]\n\n'
    "Now extract from this message. Return ONLY the JSON array.\n"
    'Message: "{text}"'
)

# Mapping from LLM predicates to our canonical fact_types
PREDICATE_MAP: dict[str, str] = {
    "lives_in": "location.current",
    "lived_in": "location.past",
    "from": "location.hometown",
    "moving_to": "location.future",
    "works_at": "work.employer",
    "worked_at": "work.former_employer",
    "job_title": "work.job_title",
    "likes": "preference.activity",
    "dislikes": "preference.food_dislike",
    "allergic_to": "health.allergy",
    "has_condition": "health.condition",
    "is_family_of": "relationship.family",
    "is_friend_of": "relationship.friend",
    "is_partner_of": "relationship.partner",
    "school": "personal.school",
    "attends": "personal.school",
    "graduated_from": "personal.school",
    "birthday": "personal.birthday",
    "pet": "personal.pet",
    "hobby": "preference.activity",
    "plays": "preference.activity",
}

# Mapping from LLM predicates to bakeoff-compatible entity-type labels.
# The bakeoff's spans_match() uses LABEL_ALIASES keyed on these types.
PREDICATE_TO_LABEL: dict[str, str] = {
    "lives_in": "place",
    "lived_in": "place",
    "from": "place",
    "moving_to": "place",
    "works_at": "org",
    "worked_at": "org",
    "job_title": "job_role",
    "likes": "activity",
    "dislikes": "food_item",
    "allergic_to": "health_condition",
    "has_condition": "health_condition",
    "is_family_of": "family_member",
    "is_friend_of": "person_name",
    "is_partner_of": "person_name",
    "attends": "org",
    "graduated_from": "org",
    "school": "org",
    "hobby": "activity",
    "plays": "activity",
    "birthday": "date_ref",
    "pet": "person_name",
}

VAGUE = {"it", "this", "that", "thing", "stuff", "them", "there", "here", "me", "you"}

class LFMAdapter(ExtractorAdapter):
    """Adapter for LFM instruct models for high-precision triple extraction.

    Uses structured JSON prompting with LFM's chat template format.
    Includes a message-level pre-filter to avoid unnecessary LLM calls.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("lfm", config)
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_id = self.config.get("model_id") or self.config.get("model_name", "lfm-1.2b")
        self._max_tokens = self.config.get("max_tokens", 256)
        self._global_threshold = self.config.get("threshold", 0.7)
        self._use_gate = self.config.get("use_gate", True)

    @property
    def supported_labels(self) -> list[str]:
        return sorted(set(PREDICATE_TO_LABEL.values()))

    @property
    def default_threshold(self) -> float:
        return self._global_threshold

    def _load_model(self) -> Any:
        """Load the LFM model via mlx_lm."""
        if self._model is not None:
            return self._model

        model_path = LFM_MODELS.get(self._model_id)
        if model_path is None:
            available = ", ".join(LFM_MODELS.keys())
            raise ValueError(f"Unknown LFM model_id '{self._model_id}'. Available: {available}")

        import mlx.core as mx
        from mlx_lm import load

        logger.info("Loading LFM model: %s (%s)", self._model_id, model_path)

        gpu_lock = self._get_gpu_lock()
        with gpu_lock:
            mx.set_memory_limit(1 * 1024 * 1024 * 1024)  # 1GB
            mx.set_cache_limit(512 * 1024 * 1024)  # 512MB
            self._model, self._tokenizer = load(model_path)

        logger.info("LFM model loaded: %s", self._model_id)
        return self._model

    @staticmethod
    def _get_gpu_lock() -> threading.Lock:
        """Get the shared MLX GPU lock."""
        try:
            from models.loader import MLXModelLoader

            return MLXModelLoader.gpu_lock()
        except ImportError:
            if not hasattr(LFMAdapter, "_fallback_lock"):
                LFMAdapter._fallback_lock = threading.Lock()
            return LFMAdapter._fallback_lock

    def _build_prompt(self, text: str) -> str:
        """Build the chat-template prompt for LFM."""
        user_msg = USER_PROMPT_TEMPLATE.format(text=text)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _generate(self, prompt: str) -> str:
        """Run MLX generation with GPU lock."""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

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

        # Periodic cache clear to prevent OOM on long runs (8GB system)
        self._gen_count = getattr(self, "_gen_count", 0) + 1
        if self._gen_count % 10 == 0:
            import mlx.core as mx
            mx.clear_cache()

        return response

    def _parse_response(self, response: str, text: str) -> list[ExtractedCandidate]:
        """Parse LFM JSON triple response into candidates."""
        candidates: list[ExtractedCandidate] = []

        json_str = response.strip()

        # Extract JSON array from markdown fences
        fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", json_str, re.DOTALL)
        if fence_match:
            json_str = fence_match.group(1)
        else:
            # Find first JSON array in response
            arr_match = re.search(r"(\[[\s\S]*?\])", json_str)
            if arr_match:
                json_str = arr_match.group(1)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.debug("Failed to parse LFM JSON: %s", response[:200])
            return candidates

        if not isinstance(data, list):
            logger.debug("LFM response is not a list: %s", type(data))
            return candidates

        seen: set[tuple[str, str, str]] = set()

        for item in data:
            if not isinstance(item, dict):
                continue
            
            # Triple format: subject, predicate, object, span
            subject = str(item.get("subject", "")).strip()
            predicate = str(item.get("predicate", "")).strip().lower()
            obj = str(item.get("object", "")).strip()
            span = str(item.get("span", "")).strip()

            if not span or not predicate:
                continue

            # Dedup
            dedup_key = (subject.casefold(), predicate, obj.casefold())
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # Find span in text for offsets
            idx = text.casefold().find(span.casefold())
            if idx < 0:
                # Fallback: if span not found, try object text
                idx = text.casefold().find(obj.casefold())
                if idx >= 0:
                    span = obj
                else:
                    continue

            # Resolve fact_type from predicate
            fact_type = PREDICATE_MAP.get(predicate, "other_personal_fact")

            # Skip unrecognized predicates
            if fact_type == "other_personal_fact":
                continue

            # Map predicate to bakeoff-compatible entity-type label
            span_label = PREDICATE_TO_LABEL.get(predicate, predicate)

            candidates.append(
                ExtractedCandidate(
                    span_text=span,
                    span_label=span_label,
                    score=0.90,
                    start_char=idx,
                    end_char=idx + len(span),
                    fact_type=fact_type,
                    extractor_metadata={
                        "source": "lfm-triples",
                        "model_id": self._model_id,
                        "subject": subject,
                        "object": obj,
                        "predicate": predicate,
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
        """Extract candidates using LFM with pre-filter gate."""
        # Stage 1: Message-level gate (Approach B)
        if self._use_gate:
            if not is_fact_likely(text, is_from_me=is_from_me or False):
                logger.debug("Gate closed for message: %s", text[:50])
                return []

        # Stage 2: LLM Extraction (Approach C)
        self._load_model()
        prompt = self._build_prompt(text)

        try:
            response = self._generate(prompt)
            return self._parse_response(response, text)
        except Exception as e:
            logger.error("LFM extraction failed: %s", e)
            return []

    def extract_batch(
        self,
        messages: list[dict[str, Any]],
        batch_size: int = 1,
        threshold: float | None = None,
    ) -> list[ExtractionResult]:
        """Batch extraction - sequential since LLM generates one at a time."""
        return super().extract_batch(messages, batch_size=1, threshold=threshold)


# Register both variants
register_extractor("lfm", LFMAdapter)
