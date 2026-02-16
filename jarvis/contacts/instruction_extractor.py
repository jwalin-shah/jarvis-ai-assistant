"""Instruction-based Fact Extraction - Using fine-tuned LFM-350M/1.2B.

Optimized for 8GB RAM: phase-based batch processing with NL extraction and NLI verification.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from jarvis.contacts.attribution import AttributionResolver
from jarvis.contacts.contact_profile import Fact
from jarvis.contacts.junk_filters import is_junk_message
from jarvis.prompts.extraction import (
    EXTRACTION_MODELS,
    EXTRACTION_SYSTEM_PROMPT,
    NEGATIVE_CONSTRAINTS,
)
from jarvis.text_normalizer import normalize_text
from models.loader import MLXModelLoader, ModelConfig

logger = logging.getLogger(__name__)

# Post-LLM hard gates for low-information and reaction artifacts.
_LOW_INFO_VALUES = {
    "me",
    "you",
    "that",
    "this",
    "it",
    "them",
    "him",
    "her",
    "someone",
    "something",
    "anything",
    "nothing",
    "to see it",
    "to see those",
    "to see that",
    "to see this",
    "not specified",
}

_REACTION_PREFIXES = (
    "liked ",
    "loved ",
    "laughed at ",
    "emphasized ",
    "questioned ",
    "disliked ",
    "reacted ",
)

_REACTION_MARKERS = (
    " an attachment",
    " tapback",
    " reacted to ",
    "liked “",
    "loved “",
    "laughed at “",
)

_HAS_FACT_META_MARKERS = (
    " mentions ",
    " said ",
    " says ",
    " asks ",
    " asked ",
    " talks about ",
    " discussed ",
    " discussing ",
    " planning ",
    " is planning ",
)
_SPEAKER_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 _.'-]{0,30}:\s+")

_PASS1_META_LINE_RE = re.compile(
    r"(?:"
    r"\[segment\s*n\]|"
    r"\*\*|"
    r"\bidentity\s*/\s*relationship\b|"
    r"\bwork\s*/\s*school\s*/\s*location\b|"
    r"\bstable\s+planning\b|"
    r"\bkeep\s*\(|\bdrop\s*\(|"
    r"\btelecommunications\s+company\b"
    r")",
    re.IGNORECASE,
)
_PASS1_PLACEHOLDER_RE = re.compile(
    r"(?:\[[^\]]{1,40}\]|<[^>]{1,40}>|\b(?:unknown|not specified|n/?a)\b)",
    re.IGNORECASE,
)

_EPHEMERAL_TIME_RE = re.compile(
    r"(?:"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b|"
    r"\b\d{1,2}[:]\d{2}\s*(?:am|pm)?\b|"
    r"\b(?:am|pm)\b|"
    r"\b(?:tomorrow|tmrw|tonight|today|next week|this weekend)\b|"
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    r")",
    re.IGNORECASE,
)

_STABLE_SCHEDULE_HINT_RE = re.compile(
    r"(?:"
    r"\b(?:usually|typically|normally|always|often)\b|"
    r"\b(?:free after|works? (?:nights|mornings|weekends)|schedule is)\b|"
    r"\b(?:prefers? to plan|likes? to plan|plans ahead)\b"
    r")",
    re.IGNORECASE,
)

_TRANSACTIONAL_MSG_RE = re.compile(
    r"(?:"
    r"\bone[- ]?time\s+passcode\b|"
    r"\bverification\s+code\b|"
    r"\botp\b|"
    r"\bpasscode\s+is\s+\d{4,8}\b|"
    r"\bcode\s+is\s+\d{4,8}\b|"
    r"\byour\s+order\b|"
    r"\border\s+(?:is\s+)?(?:ready|confirmed|shipped|delivered)\b|"
    r"\bpickup\s+instructions\b|"
    r"\btrack(?:ing)?\s+(?:number|id|link)\b|"
    r"\bdoordash\b|\bubereats\b|\bgrubhub\b|\binstacart\b|"
    r"\bmessage/data\s+rates\s+may\s+apply\b"
    r")",
    re.IGNORECASE,
)


def _is_transactional_message(text: str) -> bool:
    """Detect transactional alerts that are not durable personal facts."""
    return bool(_TRANSACTIONAL_MSG_RE.search(text))


def _build_extraction_system_prompt(user_name: str, contact_name: str, **kwargs: Any) -> str:
    """Build pass-1 prompt for single-segment extraction."""
    return EXTRACTION_SYSTEM_PROMPT


def _parse_pass1_json_lines(
    raw_output: str, segment_count: int | None = None
) -> tuple[list[list[str]], str]:
    """Parse pass-1 output for single-segment extraction."""
    # Always returning 1 segment list (segment_count accepted for API compatibility)
    claims: list[str] = []
    canonical_lines: list[str] = []

    # --- Step 1: Preprocess — strip markdown fences, join multi-line JSON ---
    cleaned = re.sub(r"```(?:jsonl?|)\s*\n?", "", raw_output)  # strip fences
    # Accumulate multi-line JSON objects by tracking brace depth
    json_objects: list[str] = []
    buf: list[str] = []
    depth = 0
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for ch in stripped:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
        buf.append(stripped)
        if depth <= 0 and buf:
            joined = " ".join(buf)
            # Only try if it looks like JSON
            if "{" in joined:
                json_objects.append(joined)
            else:
                json_objects.append(joined)  # keep for pipe-delimited fallback
            buf = []
            depth = 0

    # --- Step 2: Parse JSON objects ---
    def _accept_item(item: dict[str, Any]) -> bool:
        subject = str(item.get("subject", "")).strip()
        speaker = str(item.get("speaker", "")).strip()
        claim = str(item.get("claim", "")).strip()
        evidence_quote = str(item.get("evidence_quote", "")).strip()

        if not subject or not speaker or not claim or not evidence_quote:
            return False
        if _PASS1_META_LINE_RE.search(claim) or _PASS1_PLACEHOLDER_RE.search(claim):
            return False
        if _PASS1_META_LINE_RE.search(evidence_quote) or _PASS1_PLACEHOLDER_RE.search(
            evidence_quote
        ):
            return False

        claims.append(f"{subject} | {claim}")
        canonical_lines.append(f"- {subject} | {claim}")
        return True

    for obj_str in json_objects:
        # Try JSON parse
        clean = re.sub(r"^[\s\-\*\d\.]+\s*", "", obj_str).strip()
        if not clean or clean.lower() == "none":
            continue
        try:
            item = json.loads(clean)
            if isinstance(item, dict):
                _accept_item(item)
                continue
        except json.JSONDecodeError:
            pass

        # Fallback: pipe-delimited "Subject | claim"
        # Since single segment, ignore [Segment N] prefixes if present
        clean = re.sub(r"\[Segment\s*\d+\]\s*", "", clean, flags=re.IGNORECASE).strip()
        parts = [p.strip() for p in clean.split("|")]
        if len(parts) >= 2:
            subject = parts[0].strip()
            claim = parts[1].strip()
            if (
                subject
                and claim
                and not _PASS1_META_LINE_RE.search(claim)
                and not _PASS1_PLACEHOLDER_RE.search(claim)
            ):
                claims.append(f"{subject} | {claim}")
                canonical_lines.append(f"- {subject} | {claim}")

    canonical_facts = "\n".join(canonical_lines) if canonical_lines else "NONE"
    # Format matches expected return: list of segments (size 1) each containing list of strings
    return [claims], canonical_facts


def _is_low_signal_block(text: str) -> bool:
    """Fast check to skip LLM extraction for windows with no meaningful content."""
    # 1. Very short (less than 15 chars)
    clean = text.strip()
    if len(clean) < 15:
        return True

    # 2. Only emojis and punctuation
    if not any(c.isalnum() for c in clean):
        return True

    # 3. Common low-signal reactions (case-insensitive)
    low_signal_patterns = [
        (
            r"^(?:lol|haha|ok|okay|yep|yup|nope|no|yes|thanks|thx|cool|nice|wow|"
            r"sounds good|bet|got it|will do)\W*$"
        )
    ]
    low_signal_re = re.compile("|".join(low_signal_patterns), re.IGNORECASE)
    if low_signal_re.match(clean):
        return True

    return False


class InstructionFactExtractor:
    """Fact extractor using fine-tuned LFM models with Two-Pass Self-Correction."""

    def __init__(self, model_tier: str = "1.2b") -> None:
        model_path = EXTRACTION_MODELS.get(
            model_tier, EXTRACTION_MODELS.get(model_tier, EXTRACTION_MODELS["1.2b"])
        )
        # Use LFM-optimal defaults
        self._config = ModelConfig(
            model_path=model_path,
            default_temperature=0.1,
        )
        self._loader = MLXModelLoader(self._config)
        self._tier = model_tier
        self._attribution_resolver = AttributionResolver()
        self._last_batch_stats: dict[str, int] = {
            "raw_triples": 0,
            "prefilter_rejected": 0,
            "accepted": 0,
            "prefilter_messages_skipped": 0,
        }
        self._last_batch_pass1_claims: list[list[str]] = []

    def load(self) -> bool:
        try:
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

    def get_last_batch_stats(self) -> dict[str, int]:
        """Return counters from the most recent batch extraction."""
        return dict(self._last_batch_stats)

    def get_last_batch_pass1_claims(self) -> list[list[str]]:
        """Return pass-1 natural-language claims grouped by extraction-window index."""
        return [list(claims) for claims in self._last_batch_pass1_claims]

    def extract_facts_from_segment(
        self,
        segment: Any,
        contact_id: str = "",
        contact_name: str = "Contact",
        user_name: str = "Me",
    ) -> list[Fact]:
        """Process a single extraction window using two-pass extraction.

        Note: `segment` here is an extraction window payload, not a topic
        segmentation result.
        """
        results = self.extract_facts_from_batch(
            [segment], contact_id=contact_id, contact_name=contact_name, user_name=user_name
        )
        return results[0] if results else []

    def extract_pass1_claims_from_batch(
        self,
        segments: list[Any],
        contact_id: str = "",
        contact_name: str = "Contact",
        user_name: str = "Me",
    ) -> list[list[str]]:
        """Run only pass-1 claim extraction and return claims by segment."""
        if not self._loader.is_loaded():
            if not self.load():
                return [[] for _ in segments]

        if not segments:
            return []

        self._last_batch_stats = {
            "raw_triples": 0,
            "prefilter_rejected": 0,
            "accepted": 0,
            "prefilter_messages_skipped": 0,
        }
        self._last_batch_pass1_claims = [[] for _ in segments]

        segment_texts = []
        for i, segment in enumerate(segments):
            messages = getattr(segment, "messages", [])
            prompt_lines = []
            if messages:
                current_label = None
                current_block = []
                for m in messages:
                    if m.is_from_me:
                        label = user_name
                    else:
                        label = getattr(m, "sender_name", None) or contact_name

                    raw_msg = " ".join((m.text or "").splitlines()).strip()
                    if not raw_msg:
                        continue
                    clean_msg = normalize_text(
                        raw_msg,
                        filter_garbage=True,
                        filter_attributed_artifacts=True,
                        strip_signatures=True,
                    )
                    if not clean_msg:
                        self._last_batch_stats["prefilter_messages_skipped"] += 1
                        continue
                    if is_junk_message(clean_msg, contact_id):
                        self._last_batch_stats["prefilter_messages_skipped"] += 1
                        continue
                    if _is_transactional_message(clean_msg):
                        self._last_batch_stats["prefilter_messages_skipped"] += 1
                        continue

                    if label == current_label:
                        current_block.append(clean_msg)
                    else:
                        if current_block and current_label:
                            prompt_lines.append(f"{current_label}: {' '.join(current_block)}")
                        current_label = label
                        current_block = [clean_msg]

                if current_block and current_label:
                    prompt_lines.append(f"{current_label}: {' '.join(current_block)}")

            seg_text = "\n".join(prompt_lines)
            if len(segments) == 1:
                segment_texts.append(seg_text)
            else:
                segment_texts.append(f"[Segment {i}]\n{seg_text}")

        batch_text = "\n\n".join(segment_texts)

        # OPTIMIZATION: Skip LLM call if window has no meaningful content after filtering
        # This saves 1-3 seconds per junk-only window (20-40% of windows in some SMS contacts)
        if not batch_text or len(batch_text.strip()) < 20:
            return [[] for _ in segments]

        try:
            p1_system = _build_extraction_system_prompt(
                user_name=user_name,
                contact_name=contact_name,
                segment_count=len(segments),
            )
            p1_user = f"Conversation:\n{batch_text}\n\nReturn JSONL durable claims now (or NONE):\n"
            messages_p1 = [
                {"role": "system", "content": p1_system},
                {"role": "user", "content": p1_user},
            ]
            formatted_p1 = self._loader._tokenizer.apply_chat_template(
                messages_p1, tokenize=False, add_generation_prompt=True
            )
            if not formatted_p1.endswith("\n"):
                formatted_p1 += "\n"

            res1 = self._loader.generate_sync(
                prompt=formatted_p1,
                max_tokens=240,
                temperature=0.0,
                stop_sequences=["<|im_end|>", "###"],
                pre_formatted=True,
                negative_constraints=NEGATIVE_CONSTRAINTS,
            )
            self._last_batch_pass1_claims, _ = _parse_pass1_json_lines(
                res1.text.strip(), len(segments)
            )
            return [list(claims) for claims in self._last_batch_pass1_claims]
        except Exception as e:
            logger.error("Pass-1 extraction failed: %s", e)
            return [[] for _ in segments]

    def extract_facts_from_batch(
        self,
        segments: list[Any],
        contact_id: str = "",
        contact_name: str = "Contact",
        user_name: str = "Me",
    ) -> list[list[Fact]]:
        """Two-pass extraction on a batch of extraction windows.

        Concatenates windows into a single prompt using [Segment N] markers
        for high-throughput processing without losing attribution.
        """
        if not self._loader.is_loaded():
            if not self.load():
                return [[] for _ in segments]

        if not segments:
            return []

        if len(segments) > 1:
            # Process one extraction window per model call to keep memory usage stable,
            # while still covering every requested window.
            all_results: list[list[Fact]] = []
            for window in segments:
                single_result = self.extract_facts_from_batch(
                    [window],
                    contact_id=contact_id,
                    contact_name=contact_name,
                    user_name=user_name,
                )
                all_results.append(single_result[0] if single_result else [])
            return all_results

        self._last_batch_stats = {
            "raw_triples": 0,
            "prefilter_rejected": 0,
            "accepted": 0,
            "prefilter_messages_skipped": 0,
        }

        # 1. Format Batch Text
        segment_texts = []
        for i, segment in enumerate(segments):
            messages = getattr(segment, "messages", [])
            prompt_lines = []
            if messages:
                # Use first message for flow start; dynamic labels for group chats
                current_label = None
                current_block = []

                for m in messages:
                    # Resolve the label for this specific message
                    if m.is_from_me:
                        label = user_name
                    else:
                        # Use sender_name if available (group chat member), else contact_name
                        label = getattr(m, "sender_name", None) or contact_name

                    raw_msg = " ".join((m.text or "").splitlines()).strip()
                    if not raw_msg:
                        continue
                    clean_msg = normalize_text(
                        raw_msg,
                        filter_garbage=True,
                        filter_attributed_artifacts=True,
                        strip_signatures=True,
                    )
                    if not clean_msg:
                        self._last_batch_stats["prefilter_messages_skipped"] += 1
                        continue
                    if is_junk_message(clean_msg, contact_id):
                        self._last_batch_stats["prefilter_messages_skipped"] += 1
                        continue
                    if _is_transactional_message(clean_msg):
                        self._last_batch_stats["prefilter_messages_skipped"] += 1
                        continue

                    if label == current_label:
                        current_block.append(clean_msg)
                    else:
                        if current_block and current_label:
                            prompt_lines.append(f"{current_label}: {' '.join(current_block)}")
                        current_label = label
                        current_block = [clean_msg]

                # Flush final block
                if current_block and current_label:
                    prompt_lines.append(f"{current_label}: {' '.join(current_block)}")

            seg_text = "\n".join(prompt_lines)
            if len(segments) == 1:
                segment_texts.append(seg_text)
            else:
                segment_texts.append(f"[Segment {i}]\n{seg_text}")

        batch_text = "\n\n".join(segment_texts)

        # EMPTY-INPUT GUARD: skip LLM calls entirely when text is empty or low-signal
        if not batch_text.strip() or _is_low_signal_block(batch_text):
            logger.debug("Skipping LLM: Empty or low-signal text after filtering")
            return [[] for _ in segments]

        try:
            self._last_batch_pass1_claims = [[] for _ in segments]
            # SINGLE PASS: Triple Extraction
            system_prompt = _build_extraction_system_prompt(
                user_name=user_name,
                contact_name=contact_name,
            )
            user_prompt = f"Conversation:\n{batch_text}\n\nTriples:\n- "

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            formatted_prompt = self._loader._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Nudge model into generating triples immediately
            if not formatted_prompt.endswith("- "):
                formatted_prompt += "- "

            res = self._loader.generate_sync(
                prompt=formatted_prompt,
                max_tokens=400,
                temperature=0.0,
                stop_sequences=["<|im_end|>", "###"],
                pre_formatted=True,
                negative_constraints=NEGATIVE_CONSTRAINTS,
            )

            raw_output = "- " + res.text.strip()
            logger.debug(f"Batch Extract Output: {raw_output}")

            if "NONE" in raw_output[:10]:
                return [[] for _ in segments]

            # 3. ROBUST PARSING WITH SEGMENT ATTRIBUTION
            batch_facts: list[list[Fact]] = [[] for _ in segments]
            commentary_markers = ["removing", "keeping", "verified", "here are", "revised"]

            for line in raw_output.split("\n"):
                line = line.strip()
                if not line or len(line) < 5:
                    continue

                # Identify segment
                seg_idx = 0
                seg_match = re.search(r"\[Segment\s*(\d+)\]", line, re.IGNORECASE)
                if seg_match:
                    seg_idx = int(seg_match.group(1))
                    line = line.replace(seg_match.group(0), "").strip()

                if seg_idx >= len(segments):
                    continue

                # Triple parsing
                clean_line = re.sub(r"^[\s\-\*\d\.]+\s*", "", line).strip()
                if any(m in clean_line.lower() for m in commentary_markers):
                    continue

                parts = [p.strip() for p in clean_line.split("|")]
                subject_name, predicate, fact_value = "", "has_fact", ""

                if len(parts) >= 3:
                    subject_name, predicate, fact_value = parts[0], parts[1], parts[2]
                elif len(parts) == 1:
                    subject_name, fact_value = contact_name, parts[0]
                elif len(parts) == 2:
                    subject_name, fact_value = parts[0], parts[1]
                else:
                    continue
                self._last_batch_stats["raw_triples"] += 1

                # CLEANUP PREFIXES (Model hallucination fix)
                for prefix in ["Subject:", "Predicate:", "Object:", "Value:", "Factor:", "Claim:"]:
                    if subject_name.lower().startswith(prefix.lower()):
                        subject_name = subject_name[len(prefix) :].strip()
                    if predicate.lower().startswith(prefix.lower()):
                        predicate = predicate[len(prefix) :].strip()
                    if fact_value.lower().startswith(prefix.lower()):
                        fact_value = fact_value[len(prefix) :].strip()

                # --- VALIDATION GATES ---

                # 1. Subject Validation
                # Reject if subject is the group/chat name itself
                if contact_name.lower() in subject_name.lower() and len(contact_name) > 15:
                    # Likely a group chat name being used as subject
                    self._last_batch_stats["prefilter_rejected"] += 1
                    continue
                if any(
                    x in subject_name.lower()
                    for x in ["chat", "group", "conversation", "participants"]
                ):
                    self._last_batch_stats["prefilter_rejected"] += 1
                    continue
                if any(c in subject_name for c in ["*", "_", "[", "]"]):
                    self._last_batch_stats["prefilter_rejected"] += 1
                    continue

                # 2. Predicate Validation
                # Normalize predicate
                predicate = predicate.lower().replace(" ", "_")
                # Reject if too long (likely a sentence)
                if len(predicate.split("_")) > 5:
                    self._last_batch_stats["prefilter_rejected"] += 1
                    continue

                # 3. Value Validation
                if any(c in fact_value for c in ["*", "_", "[", "]"]):
                    self._last_batch_stats["prefilter_rejected"] += 1
                    continue
                if not fact_value or len(fact_value) < 2:
                    self._last_batch_stats["prefilter_rejected"] += 1
                    continue

                # 4. Hard gate for low-information/reaction artifacts.
                norm_value = " ".join(fact_value.lower().split()).strip(" .,!?:;\"'")
                if norm_value in _LOW_INFO_VALUES:
                    self._last_batch_stats["prefilter_rejected"] += 1
                    continue
                if norm_value.startswith(_REACTION_PREFIXES) or any(
                    m in norm_value for m in _REACTION_MARKERS
                ):
                    self._last_batch_stats["prefilter_rejected"] += 1
                    continue
                # 5. Keep stable schedule/preference signals, drop one-off timestamp chatter.
                if _EPHEMERAL_TIME_RE.search(norm_value) and not _STABLE_SCHEDULE_HINT_RE.search(
                    norm_value
                ):
                    self._last_batch_stats["prefilter_rejected"] += 1
                    continue
                # 6. Generic has_fact should be specific; drop meta/logistics paraphrases.
                if predicate == "has_fact":
                    norm_with_spaces = f" {norm_value} "
                    if (
                        len(norm_value.split()) > 14
                        or _SPEAKER_PREFIX_RE.match(fact_value)
                        or any(marker in norm_with_spaces for marker in _HAS_FACT_META_MARKERS)
                    ):
                        self._last_batch_stats["prefilter_rejected"] += 1
                        continue

                # Subject resolution (Me/User/Contact)
                if len(subject_name.split()) > 3 or any(
                    v in subject_name.lower() for v in [" is ", " has ", " works "]
                ):
                    old_sub = subject_name
                    subject_name = (
                        user_name
                        if (user_name.lower() in old_sub.lower() or "jwalin" in old_sub.lower())
                        else contact_name
                    )
                    fact_value, predicate = old_sub, "has_fact"

                is_about_user = any(
                    u in subject_name.lower() for u in [user_name.lower(), "jwalin", "me", "i "]
                )

                # Traceability (find best msg in specific segment)
                segment = segments[seg_idx]
                msgs = getattr(segment, "messages", [])
                claim_words = set(w for w in fact_value.lower().split() if len(w) > 3)
                best_msg = msgs[-1] if msgs else None
                if claim_words and msgs:
                    best_match = 0
                    for m in msgs:
                        matches = sum(1 for w in claim_words if w in (m.text or "").lower())
                        if matches > best_match:
                            best_match, best_msg = matches, m

                # Heuristic Confidence Scoring
                # Map predicate types to confidence based on "information value"
                p_lower = predicate.lower()
                base_confidence = 0.7  # Default for generic facts

                if any(
                    k in p_lower
                    for k in [
                        "is",
                        "works",
                        "lives",
                        "family",
                        "partner",
                        "spouse",
                        "child",
                        "school",
                        "degree",
                    ]
                ):
                    base_confidence = 1.0  # Strong identity/demographic facts
                elif any(
                    k in p_lower
                    for k in ["likes", "loves", "hates", "prefers", "hobby", "skill", "speaks"]
                ):
                    base_confidence = 0.8  # Preferences and skills
                elif any(k in p_lower for k in ["schedule", "availability", "free", "busy"]):
                    base_confidence = 0.6  # Scheduling is often ephemeral
                elif p_lower == "has_fact":
                    base_confidence = 0.7  # Generic bucket

                batch_facts[seg_idx].append(
                    Fact(
                        category="other",  # Storage layer re-categorizes
                        subject=user_name
                        if is_about_user
                        else contact_name
                        if not is_about_user and subject_name == contact_name
                        else subject_name,
                        predicate=predicate,
                        value=fact_value,
                        source_text=(best_msg.text or "")[:300] if best_msg else "",
                        source_message_id=getattr(best_msg, "id", None) if best_msg else None,
                        contact_id=contact_id,
                        confidence=base_confidence,
                        attribution=self._attribution_resolver.resolve(
                            source_text=getattr(best_msg, "text", "") or "",
                            subject=subject_name,
                            is_from_me=best_msg.is_from_me if best_msg else False,
                            category="other",  # category is determined later
                        ),
                    )
                )
                self._last_batch_stats["accepted"] += 1

            # POST-PROCESSING GROUNDING FILTER
            # Reject facts whose value is not grounded in source text
            for seg_idx, seg_facts in enumerate(batch_facts):
                if not seg_facts:
                    continue
                seg_source = "\n".join(
                    (m.text or "") for m in getattr(segments[seg_idx], "messages", [])
                ).lower()
                if not seg_source.strip():
                    continue
                grounded_facts = []
                for f in seg_facts:
                    v_lower = f.value.lower().strip()
                    if not v_lower:
                        self._last_batch_stats["prefilter_rejected"] += 1
                        continue
                    # Exact match
                    if v_lower in seg_source:
                        grounded_facts.append(f)
                        continue
                    # Fuzzy: 60%+ of words present
                    words = [w for w in v_lower.split() if len(w) > 2]
                    if words:
                        found = sum(1 for w in words if w in seg_source)
                        if found / len(words) >= 0.6:
                            grounded_facts.append(f)
                            continue
                    self._last_batch_stats["prefilter_rejected"] += 1
                    self._last_batch_stats["accepted"] -= 1
                batch_facts[seg_idx] = grounded_facts

            return batch_facts

        except Exception as e:
            import traceback

            logger.error(f"Batch extraction failed: {e}\n{traceback.format_exc()}")
            return [[] for _ in segments]


_extractor: InstructionFactExtractor | None = None


def get_instruction_extractor(tier: str = "1.2b") -> InstructionFactExtractor:
    global _extractor
    if _extractor is None or _extractor._tier != tier:
        if _extractor:
            _extractor.unload()
        _extractor = InstructionFactExtractor(model_tier=tier)
    return _extractor


def reset_instruction_extractor() -> None:
    """Unload the singleton extractor so reply LLM or others can use memory."""
    global _extractor
    if _extractor is not None:
        _extractor.unload()
        _extractor = None
