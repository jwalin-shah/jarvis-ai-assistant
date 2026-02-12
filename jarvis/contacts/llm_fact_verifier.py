"""LLM-based fact candidate verifier.

Replaces NLI entailment with LLM context verification.
ONE LLM call per message (batches all candidates from same message).
Reuses existing MLXGenerator singleton (zero additional memory).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from jarvis.contacts.candidate_extractor import FactCandidate

logger = logging.getLogger(__name__)

# Categories that skip verification entirely (same as NLI skip list).
_SKIP_CATEGORIES: set[str] = {
    "preference.activity",
    "preference.food_like",
    "preference.food_dislike",
    "health.condition",
    "health.allergy",
    "preference.skill",
    "personal.cultural_event",
}

_PROMPT_TEMPLATE = """\
Decide if each candidate is a PERSONAL FACT about the message sender.
Reply with the candidate number and YES or NO.
YES = sender personally has this trait/relationship/location
NO = CLEARLY about a third party (not the sender), or hypothetical/generic (not a real claim)
When unsure, answer YES (keep the candidate).

Message: "my dad has someone who developed Intel chips"
Candidates: 1. dad (family_member) 2. Intel (employer)
Answers: 1. YES 2. NO

Message: "I just got back from Colorado today"
Candidates: 1. Colorado (location)
Answers: 1. YES

Message: "I work at Google in San Francisco"
Candidates: 1. Google (employer) 2. San Francisco (location)
Answers: 1. YES 2. YES

Message: "{message}"
Candidates: {candidates}
Answers:"""

_RESPONSE_RE = re.compile(r"(\d+).*?(YES|NO)", re.IGNORECASE)


class LLMFactVerifier:
    """Verify GLiNER fact candidates using the LLM."""

    def __init__(self, generator: object | None = None) -> None:
        self._generator = generator

    def _get_generator(self) -> object:
        if self._generator is not None:
            return self._generator
        from models import get_generator

        self._generator = get_generator()
        return self._generator

    def verify_candidates(self, candidates: list[FactCandidate]) -> list[FactCandidate]:
        """Filter candidates using LLM verification.

        Groups candidates by message, makes one LLM call per message,
        and returns only candidates the LLM confirms as personal facts.
        Candidates in skip categories are kept without verification.
        """
        if not candidates:
            return []

        # Split: skip categories pass through, rest need verification
        needs_verify: list[FactCandidate] = []
        skip_verify: list[FactCandidate] = []
        for c in candidates:
            if c.fact_type in _SKIP_CATEGORIES:
                skip_verify.append(c)
            else:
                needs_verify.append(c)

        if not needs_verify:
            return skip_verify

        # Group by message for batched LLM calls
        by_message: dict[int, list[FactCandidate]] = defaultdict(list)
        for c in needs_verify:
            by_message[c.message_id].append(c)

        verified: list[FactCandidate] = []
        for msg_id, msg_candidates in by_message.items():
            kept = self._verify_message_candidates(msg_candidates)
            verified.extend(kept)

        logger.debug(
            "LLM verifier: %d skip, %d checked (%d kept, %d rejected)",
            len(skip_verify),
            len(needs_verify),
            len(verified),
            len(needs_verify) - len(verified),
        )
        return skip_verify + verified

    def _verify_message_candidates(
        self, candidates: list[FactCandidate]
    ) -> list[FactCandidate]:
        """Verify candidates from a single message with one LLM call."""
        if not candidates:
            return []

        source_text = candidates[0].source_text
        prompt = self._build_prompt(source_text, candidates)

        from contracts.models import GenerationRequest

        gen = self._get_generator()
        request = GenerationRequest(
            prompt=prompt,
            context_documents=[],
            few_shot_examples=[],
            max_tokens=max(15, 5 * len(candidates)),
            temperature=0.1,
            stop_sequences=["Message:", "\n\n", "Candidates:"],
        )
        response = gen.generate(request)
        verdicts = self._parse_response(response.text, len(candidates))

        kept = []
        for candidate, keep in zip(candidates, verdicts):
            if keep:
                kept.append(candidate)
            else:
                logger.debug(
                    "LLM rejected: '%s' (%s) in '%s'",
                    candidate.span_text,
                    candidate.fact_type,
                    source_text[:60],
                )
        return kept

    @staticmethod
    def _build_prompt(message_text: str, candidates: list[FactCandidate]) -> str:
        """Build verification prompt for a set of candidates from one message."""
        # Escape braces in message text to avoid format string issues
        safe_message = message_text.replace("{", "{{").replace("}", "}}")
        parts = []
        for i, c in enumerate(candidates, 1):
            parts.append(f"{i}. {c.span_text} ({c.span_label})")
        candidate_block = " ".join(parts)
        return _PROMPT_TEMPLATE.format(message=safe_message, candidates=candidate_block)

    @staticmethod
    def _parse_response(response_text: str, num_candidates: int) -> list[bool]:
        """Parse LLM response into per-candidate verdicts.

        Returns a list of booleans (True=keep, False=reject).
        Defaults to keep (True) for any unparseable or missing entries.
        """
        # Default: keep all (favor recall on parse failure)
        verdicts = [True] * num_candidates
        text = response_text.strip()

        # Try numbered format first: "1. YES 2. NO"
        found_numbered = False
        for match in _RESPONSE_RE.finditer(text):
            idx = int(match.group(1)) - 1  # 1-indexed to 0-indexed
            answer = match.group(2).upper()
            if 0 <= idx < num_candidates:
                verdicts[idx] = answer == "YES"
                found_numbered = True

        # Fallback for single candidate: plain "YES" or "NO" without number
        if not found_numbered and num_candidates == 1 and text:
            first_word = text.split()[0].upper().rstrip(".,;:")
            if first_word in ("YES", "NO"):
                verdicts[0] = first_word == "YES"

        return verdicts
