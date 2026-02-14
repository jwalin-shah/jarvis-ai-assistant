"""Instruction-based Fact Extraction - Using fine-tuned LFM-350M/1.2B.

Optimized for 8GB RAM: Phase-based batch processing with Natural Language extraction and NLI verification.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from jarvis.contacts.contact_profile import Fact
from jarvis.contacts.attribution import AttributionResolver
from contracts.models import GenerationRequest
from models.loader import MLXModelLoader, ModelConfig

logger = logging.getLogger(__name__)

# Model options
MODELS = {
    "1.2b": "models/lfm2-1.2b-extract-mlx-4bit",
    "0.7b": "models/lfm-0.7b-4bit",
    "350m": "models/lfm2-350m-extract-mlx-4bit",
}

# --- V4 IMPROVED PROMPTS (ChatML optimized) ---

_EXTRACTION_SYSTEM_PROMPT = """You are a chat analyzer. 
Extract 3-5 PERSONAL facts learned about {user_name} or {contact_name} from the chat below.

STRICT RULES:
1. ONLY extract facts about the PEOPLE (e.g. where they live, work, their health, preferences).
2. DO NOT extract general news, TikTok stories, memes, or group chat banter.
3. USE FORMAT: - [Name]: [fact]
4. NO commentary. If no personal facts exist, output "NONE"."""

_EXTRACTION_USER_PROMPT = """Chat Turns:
{text}

Personal Facts:
- """

_VERIFY_SYSTEM_PROMPT = """You are a precise fact-checker. 
Review extracted facts against the chat history. 
Remove any that are memes, general news, or conversational filler.

STRICT RULES:
1. Output ONLY the facts in format: - [Name]: [fact]
2. DO NOT output any other text."""

_VERIFY_USER_PROMPT = """Chat:
{text}

Original Facts:
{facts}

Verified Personal Facts:
- """


class InstructionFactExtractor:
    """Fact extractor using fine-tuned LFM models with Two-Pass Self-Correction."""

    def __init__(self, model_tier: str = "1.2b") -> None:
        model_path = MODELS.get(model_tier, MODELS.get(model_tier, MODELS["1.2b"]))
        # Use LFM-optimal defaults
        self._config = ModelConfig(
            model_path=model_path,
            default_temperature=0.1,
        )
        self._loader = MLXModelLoader(self._config)
        self._tier = model_tier
        self._attribution_resolver = AttributionResolver()

    def load(self) -> bool:
        try:
            return self._loader.load()
        except Exception as e:
            logger.error(f"Failed to load {self._tier} extract model: {e}")
            return False

    def unload(self) -> None:
        self._loader.unload()

    def is_loaded(self) -> bool:
        return self._loader.is_loaded()

    def extract_facts_from_segment(
        self,
        segment: Any,
        contact_id: str = "",
        contact_name: str = "Contact",
        user_name: str = "Me",
    ) -> list[Fact]:
        """Two-pass extraction: 1. Raw extraction, 2. Self-Correction & NUANCE check."""
        if not self._loader.is_loaded():
            if not self.load():
                return []

        messages = getattr(segment, "messages", [])
        if not messages:
            return []

        # Format chat - combine consecutive messages from same sender
        prompt_lines = []
        if messages:
            current_sender = user_name if messages[0].is_from_me else contact_name
            current_messages = []
            
            for m in messages:
                sender = user_name if m.is_from_me else contact_name
                clean_msg = " ".join((m.text or "").splitlines()).strip()
                if not clean_msg:
                    continue
                    
                if sender == current_sender:
                    current_messages.append(clean_msg)
                else:
                    # Flush previous sender
                    if current_messages:
                        prompt_lines.append(f"{current_sender}: {' '.join(current_messages)}")
                    current_sender = sender
                    current_messages = [clean_msg]
            
            # Final flush
            if current_messages:
                prompt_lines.append(f"{current_sender}: {' '.join(current_messages)}")
                
        chat_text = "\n".join(prompt_lines)

        try:
            # PASS 1: Extraction with ChatML (System + User) + NUDGE
            p1_system = _EXTRACTION_SYSTEM_PROMPT.format(
                user_name=user_name, contact_name=contact_name
            )
            p1_user = _EXTRACTION_USER_PROMPT.format(text=chat_text)
            
            messages_p1 = [
                {"role": "system", "content": p1_system},
                {"role": "user", "content": p1_user}
            ]
            formatted_p1 = self._loader._tokenizer.apply_chat_template(
                messages_p1, tokenize=False, add_generation_prompt=True
            )
            if not formatted_p1.endswith("- "):
                formatted_p1 += "- "
                
            res1 = self._loader.generate_sync(
                prompt=formatted_p1,
                max_tokens=250,
                temperature=0.0,
                stop_sequences=["<|im_end|>", "###"],
                pre_formatted=True
            )
            raw_facts = "- " + res1.text.strip()

            # PASS 2: Self-Correction with ChatML (System + User) + NUDGE
            p2_system = _VERIFY_SYSTEM_PROMPT
            p2_user = _VERIFY_USER_PROMPT.format(text=chat_text, facts=raw_facts)
            
            messages_p2 = [
                {"role": "system", "content": p2_system},
                {"role": "user", "content": p2_user}
            ]
            formatted_p2 = self._loader._tokenizer.apply_chat_template(
                messages_p2, tokenize=False, add_generation_prompt=True
            )
            if not formatted_p2.endswith("- "):
                formatted_p2 += "- "
                
            res2 = self._loader.generate_sync(
                prompt=formatted_p2,
                max_tokens=250,
                temperature=0.0,
                stop_sequences=["<|im_end|>"],
                pre_formatted=True
            )

            verified_output = "- " + res2.text.strip()
            
            # 3. STRICT PARSING: Handle bullets, numbers, markdown, and plain lines
            candidates = []
            commentary_markers = ["removing", "keeping", "verified facts", "here are", "revised", "adheres", "original text", "analyzing"]
            for line in verified_output.split("\n"):
                line = line.strip()
                if not line: continue
                if ":" not in line: continue
                
                clean_line = re.sub(r"^[\s\-\*\d\.]+\s*", "", line)
                clean_line = clean_line.replace("**", "")
                
                lower_line = clean_line.lower()
                if any(m in lower_line for m in commentary_markers):
                    continue
                
                if len(clean_line) > 8:
                    candidates.append(clean_line)

            if not candidates:
                return []

            # 5. Targeted NLI + Hallucination Check
            from jarvis.nlp.entailment import verify_entailment_batch

            facts = []
            for cand in candidates:
                if ":" not in cand: continue
                parts = cand.split(":", 1)
                subject_name = parts[0].strip()
                fact_claim = parts[1].strip()

                subject_name = re.sub(r"^[\s\-\*\d\.]+\s*", "", subject_name).rstrip(":-. ").strip()
                if not subject_name or len(subject_name) < 2:
                    subject_name = contact_name

                # --- FIND BEST SOURCE MESSAGE ---
                claim_words = set(w for w in fact_claim.lower().split() if len(w) > 3)
                best_msg = None
                best_match_count = 0
                
                if claim_words:
                    for msg in messages:
                        msg_text = (msg.text or "").lower()
                        match_count = sum(1 for w in claim_words if w in msg_text)
                        if match_count > best_match_count:
                            best_match_count = match_count
                            best_msg = msg

                if not best_msg or best_match_count == 0:
                    continue

                # --- TARGETED NLI ---
                hypothesis = f"{subject_name} {fact_claim}"
                premise = best_msg.text or ""
                
                nli_results = verify_entailment_batch([(premise, hypothesis)], threshold=0.10)
                is_verified, nli_score = nli_results[0]

                if not is_verified:
                    # Fallback to full Turn-Based context with HIGHER threshold for news/memes
                    nli_results_fb = verify_entailment_batch([(chat_text, hypothesis)], threshold=0.30)
                    is_verified, nli_score = nli_results_fb[0]

                if not is_verified:
                    continue

                # --- RESOLVE ATTRIBUTION ---
                # Determine who the fact is about
                attr_res = self._attribution_resolver.resolve(
                    source_text=premise,
                    subject=subject_name,
                    is_from_me=best_msg.is_from_me
                )
                
                # If subject_name is explicitly Jwalin, it's user
                if subject_name.lower() in [user_name.lower(), "jwalin"]:
                    is_about_user = True
                elif subject_name.lower() in [contact_name.lower(), contact_id.lower()]:
                    is_about_user = False
                else:
                    is_about_user = (attr_res == "user")

                clean_value = fact_claim
                for n in [user_name, contact_name, "[Me]", "[Contact]", "Me", "Contact"]:
                    clean_value = re.compile(rf"\b{n}\b", re.IGNORECASE).sub("", clean_value)
                clean_value = re.sub(r'["\']+', "", clean_value).strip()
                clean_value = re.sub(r"[,.\-:]+$", "", clean_value).strip()

                # Categorization
                category = "other"
                val_lower = clean_value.lower()
                if any(w in val_lower for w in ["job", "work", "employed", "hiring", "office", "company"]):
                    category = "work"
                elif any(w in val_lower for w in ["live", "location", "moving", "from", "staying"]):
                    category = "location"
                elif any(w in val_lower for w in ["pain", "health", "diagnosed", "neuropathy", "doctor"]):
                    category = "health"
                elif any(w in val_lower for w in ["hobby", "interested", "loves", "enjoys", "likes"]):
                    category = "preference"
                elif any(w in val_lower for w in ["sister", "brother", "mom", "dad", "friend", "partner"]):
                    category = "relationship"

                facts.append(
                    Fact(
                        category=category,
                        subject=user_name if is_about_user else contact_name,
                        predicate="", # Empty for natural reading
                        value=clean_value,
                        source_text=premise[:300],
                        contact_id=contact_id,
                        confidence=nli_score,
                        attribution="user" if is_about_user else "contact",
                    )
                )

            return facts

        except Exception as e:
            logger.error(f"Two-pass extraction failed: {e}")
            return []


_extractor: InstructionFactExtractor | None = None


def get_instruction_extractor(tier: str = "1.2b") -> InstructionFactExtractor:
    global _extractor
    if _extractor is None or _extractor._tier != tier:
        if _extractor:
            _extractor.unload()
        _extractor = InstructionFactExtractor(model_tier=tier)
    return _extractor
