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

# Model options
MODELS = {
    "1.2b": "models/lfm2-1.2b-extract-mlx-4bit",
    "0.7b": "models/lfm-0.7b-4bit",  # NEW: 0.7B model (default for extraction)
    "350m": "models/lfm2-350m-extract-mlx-4bit",
}

# --- BATCHED EXTRACTION PROMPT ---

_BATCH_USER_PROMPT_TEMPLATE = """Extract personal facts from these conversation segments between {user_name} and {contact_name}.
Format: [Segment N] - [Name] Fact

{segments_text}

Bullet Points:"""

_BATCH_VERIFY_PROMPT_TEMPLATE = """You are a precise fact-checker. Review these extracted facts against the conversation segments.
Rules:
1. Correct nuances (e.g. change "is moving" to "is open to moving" if that's what was said).
2. Ensure labels [Name] are correct.
3. Remove conversational filler or duplicates.
4. Keep the [Segment N] prefix to identify which segment each fact came from.

Conversation Segments:
{segments_text}

Original Facts:
{facts}

Nuanced & Verified Facts:
-"""


class BatchedInstructionFactExtractor:
    """Fact extractor that processes multiple segments in single LLM call."""

    def __init__(self, model_tier: str = "1.2b", batch_size: int = 5) -> None:
        model_path = MODELS.get(model_tier, MODELS.get(model_tier, MODELS["1.2b"]))
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
        # Format all segments
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
            # PASS 1: Extraction
            p1_prompt = _BATCH_USER_PROMPT_TEMPLATE.format(
                segments_text=segments_text, user_name=user_name, contact_name=contact_name
            )
            res1 = self._loader.generate_sync(
                prompt=p1_prompt,
                max_tokens=800,  # More tokens for batch
                temperature=0.0,
                stop_sequences=["###"],
            )
            raw_facts = "- " + res1.text.strip()

            # PASS 2: Self-Correction (LLM-based verification for better casual chat handling)
            p2_prompt = _BATCH_VERIFY_PROMPT_TEMPLATE.format(
                segments_text=segments_text, facts=raw_facts
            )
            res2 = self._loader.generate_sync(
                prompt=p2_prompt, max_tokens=800, temperature=0.0, stop_sequences=["###"]
            )

            verified_output = "- " + res2.text.strip()

            # Parse facts with segment attribution
            verified_batch = self._parse_batched_facts(
                verified_output, segments, segment_offset, contact_id, user_name, contact_name
            )

            # Rejection count is implicit in the LLM's removal of facts between Pass 1 and 2
            # For monitoring purposes, we can estimate this by comparing lengths, 
            # but for now we'll report 0 as it's handled internally by the LLM.
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
        """Parse facts and attribute them to correct segments."""
        # Initialize results for each segment
        results = [(segment_offset + i, []) for i in range(len(segments))]

        # Clean output using pre-compiled patterns
        clean_output = output
        for pattern, replacement in _CLEAN_OUTPUT_PATTERNS:
            clean_output = pattern.sub(replacement, clean_output)

        # Parse each line
        for line in clean_output.split("\n"):
            line = line.strip()
            if not line or len(line) < 10:
                continue

            # Extract segment indicator
            segment_match = _SEGMENT_PATTERN.match(line)
            if not segment_match:
                continue

            segment_idx = int(segment_match.group(1))
            fact_text = segment_match.group(2).strip()

            # Validate segment index
            if segment_idx < 0 or segment_idx >= len(segments):
                continue

            # Apply filters
            junk_indicators = {
                "loved",
                "liked",
                "emphasized",
                "laughed at",
                "questioned",
                "reacted",
                "lol",
                "lmfao",
                "omg",
                "thanks",
                "yessir",
            }

            fact_lower = fact_text.lower()
            if any(w in fact_lower for w in junk_indicators) and len(fact_lower) < 60:
                continue
            if fact_text.endswith("?"):
                continue

            # Clean fact text using pre-compiled patterns
            fact_claim = fact_text
            for pattern, replacement in _CLEAN_FACT_PATTERNS:
                fact_claim = pattern.sub(replacement, fact_claim)

            for n in [user_name, contact_name, "[Me]", "[Contact]", "Me", "Contact"]:
                # Compile patterns once per name and cache
                _name_patterns = getattr(_parse_batched_facts, "_name_patterns", {})
                if n not in _name_patterns:
                    _name_patterns[n] = (
                        re.compile(rf"\[{re.escape(n)}\]", re.IGNORECASE),
                        re.compile(rf"\b{re.escape(n)}\b", re.IGNORECASE),
                    )
                    _parse_batched_facts._name_patterns = _name_patterns
                bracket_pat, word_pat = _name_patterns[n]
                fact_claim = bracket_pat.sub("", fact_claim)
                fact_claim = word_pat.sub("", fact_claim)
            fact_claim = fact_claim.strip(": ").strip()

            if not fact_claim:
                continue

            # Determine attribution
            segment = segments[segment_idx]
            messages = getattr(segment, "messages", [])

            claim_words = set(w for w in fact_claim.lower().split() if len(w) > 3)
            actual_is_from_me = None
            best_match_count = 0

            if claim_words and messages:
                for msg in messages:
                    msg_text = (msg.text or "").lower()
                    if not msg_text:
                        continue
                    match_count = sum(1 for w in claim_words if w in msg_text)
                    if match_count > best_match_count:
                        best_match_count = match_count
                        actual_is_from_me = msg.is_from_me

                if best_match_count == 0:
                    continue  # Hallucination

            # Fallback attribution
            if actual_is_from_me is None and messages:
                from collections import Counter

                sender_counts = Counter()
                for msg in messages:
                    sender_counts[msg.is_from_me] += 1
                if sender_counts:
                    actual_is_from_me = sender_counts.most_common(1)[0][0]

            is_about_user = actual_is_from_me if actual_is_from_me is not None else False

            # Categorize
            category = "other"
            val_lower = fact_claim.lower()
            if any(
                w in val_lower
                for w in ["job", "work", "employed", "agent", "admin", "manage", "hiring"]
            ):
                category = "work"
            elif any(
                w in val_lower
                for w in [
                    "live",
                    "location",
                    "moving",
                    "from",
                    "staying",
                    "dallas",
                    "austin",
                    "city",
                    "apartment",
                    "house",
                ]
            ):
                category = "location"
            elif any(
                w in val_lower
                for w in [
                    "brother",
                    "sister",
                    "mom",
                    "dad",
                    "father",
                    "mother",
                    "family",
                    "wife",
                    "husband",
                    "married",
                    "girlfriend",
                    "boyfriend",
                    "dating",
                    "sibling",
                    "parent",
                ]
            ):
                category = "relationship"
            elif any(
                w in val_lower
                for w in [
                    "like",
                    "love",
                    "enjoy",
                    "hate",
                    "prefer",
                    "favorite",
                    "want",
                    "interested",
                    "care about",
                    "dislike",
                ]
            ):
                category = "preference"
            elif any(
                w in val_lower
                for w in [
                    "birthday",
                    "born",
                    "graduated",
                    "school",
                    "college",
                    "degree",
                    "education",
                    "studied",
                    "university",
                ]
            ):
                category = "background"
            elif any(
                w in val_lower
                for w in [
                    "meet",
                    "available",
                    "schedule",
                    "weekend",
                    "friday",
                    "saturday",
                    "sunday",
                    "going",
                    "plan",
                    "event",
                    "party",
                    "dinner",
                    "lunch",
                    "trip",
                    "travel",
                    "vacation",
                    "holiday",
                ]
            ):
                category = "event"

            # Create fact
            fact = Fact(
                category=category,
                subject=contact_name if not is_about_user else user_name,
                predicate="has_fact",
                value=fact_claim,
                source_text=getattr(segment, "text", "")[:500],
                confidence=0.8,
                contact_id=contact_id,
                attribution="user" if is_about_user else "contact",
            )

            results[segment_idx][1].append(fact)

        return results


def get_batched_instruction_extractor(
    model_tier: str = "350m", batch_size: int = 5
) -> BatchedInstructionFactExtractor:
    """Get a batched instruction extractor instance."""
    return BatchedInstructionFactExtractor(model_tier=model_tier, batch_size=batch_size)
