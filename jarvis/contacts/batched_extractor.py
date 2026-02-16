"""Batched fact extraction from multiple segments in single LLM call.

This module provides batched extraction that processes 5-10 segments per LLM call
instead of 1 segment per call, giving 5-10x speedup on backfill operations.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from jarvis.contacts.attribution import AttributionResolver
from jarvis.contacts.contact_profile import Fact
from jarvis.prompts.extraction import EXTRACTION_MODELS
from models.loader import MLXModelLoader, ModelConfig

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns for better performance
_CLEAN_OUTPUT_PATTERNS = [
    (re.compile(r'["\']?\w+["\']?:\s*'), ""),
    (re.compile(r"[\[\]{}]"), ""),
    (re.compile(r'(^["\']|["\']$)', re.MULTILINE), ""),
]
_SEGMENT_PATTERN = re.compile(r"\[Segment\s*(\d+)\]\s*-?\s*(.*)", re.IGNORECASE)
_CLEAN_FACT_PATTERNS = [
    (re.compile(r'["\']+'), ""),
    (re.compile(r"[,\.\-:]+$"), ""),
    (re.compile(r"^[,\.\-:]+"), ""),
]

# --- BATCHED EXTRACTION PROMPTS (Aligned with instruction_extractor.py) ---

_BATCH_EXTRACTION_SYSTEM_PROMPT = """You extract durable personal facts from chat turns.

Task:
- Return ONLY stable personal claims that are useful for a long-lived profile.

Allowed claim types:
- identity/relationship (family, partner, close friend)
- work/school/role/company
- home/current location (not temporary travel)
- durable preferences (likes/dislikes/habits)
- stable schedule preference (e.g., "usually free after 6")

Do NOT extract:
- one-off logistics (meetups, ETAs, "on my way", "can we call", orders, deliveries)
- sports/news chatter, jokes, reactions, tapbacks
- meta speech ("X said", "X asked", "X mentioned")
- facts about group chats/platforms/companies unless clearly about a person
- speculative/uncertain claims

Output format:
- Output JSONL only: one JSON object per line, no markdown.
- Schema:
  {"segment_id": <int>, "subject": "<person>", "speaker": "<speaker>",
   "claim": "<durable fact>", "evidence_quote": "<exact quote from conversation>"}
- Max 3 claims per segment.
- If no claims at all, output exactly: NONE

Hard rules:
- Only explicit claims from text (no inference).
- Use 3rd-person wording.
- Subject must be a person, never a group/chat.
- `evidence_quote` must be verbatim text copied from the conversation.
- Do not use placeholders, brackets, or variables.
- No headings, markdown, commentary, or extra labels."""

_BATCH_EXTRACTION_USER_PROMPT = """Conversation Segments:
{segments_text}

Return JSONL durable claims now (or NONE):
"""

_BATCH_VERIFY_SYSTEM_PROMPT = """You are a precise fact-checker and structurer.
Review these extracted claims against the conversation segments and convert them into triples.

RULES:
1. Output format: [Segment N] - [Subject] | [Predicate] | [Object]
2. Subject must be a SPECIFIC PERSON'S NAME or "{user_name}".
3. Predicate: Short verb phrase (1-3 words).
4. Object: Capture the essential detail (1-4 words).
5. Ensure the [Segment N] prefix matches the source segment exactly.
6. Remove conversational filler, duplicates, or ungrounded claims.
7. DO NOT output any other text or commentary."""

_BATCH_VERIFY_USER_PROMPT = """Conversation Segments for Context:
{segments_text}

Claims to Verify and Structure:
{facts}

Verified Structured Facts ([Segment N] - Subject | Predicate | Object):
- """


class BatchedInstructionFactExtractor:
    """Fact extractor that processes multiple segments in single LLM call."""

    def __init__(self, model_tier: str = "1.2b", batch_size: int = 5) -> None:
        model_path = EXTRACTION_MODELS.get(
            model_tier, EXTRACTION_MODELS.get(model_tier, EXTRACTION_MODELS["1.2b"])
        )
        self._config = ModelConfig(
            model_path=model_path,
            default_temperature=0.1,
        )
        self._loader = MLXModelLoader(self._config)
        self._tier = model_tier
        self._attribution_resolver = AttributionResolver()
        self._batch_size = batch_size

    def load(self) -> bool:
        try:
            # Ensure memory is ready for LLM
            from jarvis.model_manager import get_model_manager

            get_model_manager().prepare_for("llm")

            return self._loader.load()
        except Exception as e:
            logger.error(f"Failed to load {self._tier} extract model: {e}")
            return False

    def unload(self) -> None:
        self._loader.unload()

    def is_loaded(self) -> bool:
        return self._loader.is_loaded()

    def extract_facts_from_segments_batch(
        self,
        segments: list[Any],
        contact_id: str = "",
        contact_name: str = "Contact",
        user_name: str = "Me",
    ) -> tuple[list[tuple[int, list[Fact]]], int]:
        """Extract facts from multiple segments in batched LLM calls.

        Args:
            segments: List of segments, each with .messages attribute
            contact_id: Contact identifier
            contact_name: Display name for contact
            user_name: Display name for user

        Returns:
            Tuple of (List of (segment_index, facts), total_rejections)
        """
        if not self._loader.is_loaded():
            if not self.load():
                return [], 0

        if not segments:
            return [], 0

        results = []
        total_rejections = 0

        # Process in batches
        for batch_start in range(0, len(segments), self._batch_size):
            batch_segments = segments[batch_start : batch_start + self._batch_size]
            batch_results, batch_rejections = self._extract_batch(
                batch_segments, batch_start, contact_id, contact_name, user_name
            )
            results.extend(batch_results)
            total_rejections += batch_rejections

        return results, total_rejections

    def _extract_batch(
        self,
        segments: list[Any],
        segment_offset: int,
        contact_id: str,
        contact_name: str,
        user_name: str,
    ) -> tuple[list[tuple[int, list[Fact]]], int]:
        """Extract facts from a single batch of segments."""
        # Format all segments with index for prompt
        segment_texts = []
        for i, segment in enumerate(segments):
            messages = getattr(segment, "messages", [])
            if not messages:
                continue

            prompt_lines = []
            for m in messages:
                sender = user_name if m.is_from_me else contact_name
                clean_msg = " ".join((m.text or "").splitlines())
                prompt_lines.append(f"{sender}: {clean_msg}")

            chat_text = "\n".join(prompt_lines)
            segment_texts.append(f"[Segment {i}]\n{chat_text}")

        if not segment_texts:
            return [], 0

        segments_text = "\n\n".join(segment_texts)

        try:
            # PASS 1: JSONL Extraction
            messages = [
                {"role": "system", "content": _BATCH_EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _BATCH_EXTRACTION_USER_PROMPT.format(segments_text=segments_text),
                },
            ]
            formatted1 = self._loader._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            res1 = self._loader.generate_sync(
                prompt=formatted1,
                max_tokens=1000,
                temperature=0.0,
                pre_formatted=True,
            )
            raw_jsonl = res1.text.strip()

            if not raw_jsonl or raw_jsonl.upper() == "NONE":
                return [], 0

            # PASS 2: Verification and Triple Structuring
            messages2 = [
                {
                    "role": "system",
                    "content": _BATCH_VERIFY_SYSTEM_PROMPT.format(user_name=user_name),
                },
                {
                    "role": "user",
                    "content": _BATCH_VERIFY_USER_PROMPT.format(
                        segments_text=segments_text, facts=raw_jsonl
                    ),
                },
            ]
            formatted2 = self._loader._tokenizer.apply_chat_template(
                messages2, tokenize=False, add_generation_prompt=True
            )
            res2 = self._loader.generate_sync(
                prompt=formatted2,
                max_tokens=800,
                temperature=0.0,
                pre_formatted=True,
            )

            verified_output = "- " + res2.text.strip()

            # Parse facts with segment attribution
            verified_batch = self._parse_batched_facts(
                verified_output, segments, segment_offset, contact_id, user_name, contact_name
            )

            return verified_batch, 0

        except Exception as e:
            logger.warning(f"Batch extraction failed: {e}")
            return [], 0

    def _parse_batched_facts(
        self,
        output: str,
        segments: list[Any],
        segment_offset: int,
        contact_id: str,
        user_name: str,
        contact_name: str,
    ) -> list[tuple[int, list[Fact]]]:
        """Parse verified triple facts and attribute them to correct segments.

        Format: [Segment N] - Subject | Predicate | Object
        """
        # Initialize results for each segment
        results: list[tuple[int, list[Fact]]] = [
            (segment_offset + i, []) for i in range(len(segments))
        ]

        # Triple pattern: [Segment N] - Subject | Predicate | Object
        triple_re = re.compile(
            r"\[Segment\s*(\d+)\]\s*-\s*([^|]+)\|\s*([^|]+)\|\s*(.*)", re.IGNORECASE
        )

        for line in output.split("\n"):
            line = line.strip().lstrip("- ")
            if not line:
                continue

            match = triple_re.match(line)
            if not match:
                continue

            seg_idx = int(match.group(1))
            subject = match.group(2).strip()
            predicate = match.group(3).strip()
            obj = match.group(4).strip()

            # Validate segment index
            if seg_idx < 0 or seg_idx >= len(segments):
                continue

            # Basic hallucination check: subject/object should not be placeholders
            if any(p in obj.lower() for p in ["[subject", "[name", "[phone", "placeholder"]):
                continue

            # Determine attribution (is it about the user or the contact?)
            is_about_user = False
            if user_name.lower() in subject.lower() or subject.lower() == "me":
                is_about_user = True
            elif contact_name.lower() in subject.lower():
                is_about_user = False
            else:
                # Default to contact if not user
                is_about_user = False

            # Categorize based on predicate and object
            category = "other"
            combined = (predicate + " " + obj).lower()

            if any(w in combined for w in ["job", "work", "employ", "role", "title", "office"]):
                category = "work"
            elif any(w in combined for w in ["live", "home", "city", "state", "from", "stay"]):
                category = "location"
            elif any(w in combined for w in ["mom", "dad", "wife", "husband", "son", "sister"]):
                category = "relationship"
            elif any(w in combined for w in ["like", "love", "prefer", "hate", "favorite"]):
                category = "preference"
            elif any(w in combined for w in ["born", "study", "degree", "college", "school"]):
                category = "background"

            # Create fact
            segment = segments[seg_idx]
            fact = Fact(
                category=category,
                subject=subject,
                predicate=predicate,
                value=obj,
                source_text=getattr(segment, "text", "")[:500],
                confidence=0.85,  # Higher confidence for verified triples
                contact_id=contact_id,
                attribution="user" if is_about_user else "contact",
            )

            results[seg_idx][1].append(fact)

        return results


def get_batched_instruction_extractor(
    model_tier: str = "350m", batch_size: int = 5
) -> BatchedInstructionFactExtractor:
    """Get a batched instruction extractor instance."""
    return BatchedInstructionFactExtractor(model_tier=model_tier, batch_size=batch_size)
