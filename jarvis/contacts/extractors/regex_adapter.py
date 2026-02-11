"""Lightweight regex-based extractor for high-confidence entities.

Handles emails, phone numbers, and other pattern-based entities
that don't require a transformer model.
"""

from __future__ import annotations

import re
from typing import Any

from jarvis.contacts.extractors.base import (
    ExtractedCandidate,
    ExtractorAdapter,
    register_extractor,
)

# Common regex patterns
EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
# Simple phone pattern: (123) 456-7890, 123-456-7890, 1234567890
PHONE_PATTERN = r"\b(?:\+?1[-. ]?)?\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})\b"


class RegexAdapter(ExtractorAdapter):
    """Adapter for regex-based extraction."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("regex", config)
        self._labels = self.config.get("labels", ["email", "phone_number"])

    @property
    def supported_labels(self) -> list[str]:
        return self._labels

    @property
    def default_threshold(self) -> float:
        return 1.0

    def _load_model(self) -> Any:
        return "regex"

    def extract_from_text(
        self,
        text: str,
        message_id: int,
        **kwargs: Any,
    ) -> list[ExtractedCandidate]:
        candidates: list[ExtractedCandidate] = []

        if "email" in self._labels:
            for match in re.finditer(EMAIL_PATTERN, text):
                candidates.append(
                    ExtractedCandidate(
                        span_text=match.group(),
                        span_label="email",
                        score=1.0,
                        start_char=match.start(),
                        end_char=match.end(),
                        fact_type="contact.email",
                        extractor_metadata={"source": "regex"},
                    )
                )

        if "phone_number" in self._labels:
            for match in re.finditer(PHONE_PATTERN, text):
                candidates.append(
                    ExtractedCandidate(
                        span_text=match.group(),
                        span_label="phone_number",
                        score=1.0,
                        start_char=match.start(),
                        end_char=match.end(),
                        fact_type="contact.phone",
                        extractor_metadata={"source": "regex"},
                    )
                )

        return candidates


register_extractor("regex", RegexAdapter)
