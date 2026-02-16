"""Base classes for contact extractors.

Provides the ExtractorAdapter interface and related types for the extractor bakeoff system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExtractedCandidate:
    """A candidate extraction from text.

    Attributes:
        span_text: The extracted entity text
        span_label: Normalized label (place, org, etc.)
        score: Confidence score (0.0-1.0)
        start_char: Start offset in source text (-1 if unknown)
        end_char: End offset in source text (-1 if unknown)
        fact_type: Optional fact type identifier (e.g., "relationship.lives_in")
        extractor_metadata: Optional metadata from the extractor
    """

    span_text: str
    span_label: str
    score: float = 1.0
    start_char: int = -1
    end_char: int = -1
    fact_type: str = ""
    extractor_metadata: dict[str, Any] | None = None


class ExtractorAdapter:
    """Base class for extractor adapters.

    All extractors must subclass this and implement extract_from_text.
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        """Initialize the extractor adapter.

        Args:
            name: Unique name for this extractor
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}

    def extract_from_text(
        self, text: str, message_id: int, **kwargs: Any
    ) -> list[ExtractedCandidate]:
        """Extract candidates from a single text.

        Args:
            text: The text to extract from
            message_id: Unique identifier for the message
            **kwargs: Additional extractor-specific arguments

        Returns:
            List of extracted candidates
        """
        raise NotImplementedError

    def extract_batch(self, messages: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
        """Extract candidates from a batch of messages.

        Args:
            messages: List of message dictionaries with 'text', 'id', etc.
            **kwargs: Additional extractor-specific arguments

        Returns:
            List of extraction results (one per message)
        """
        results = []
        for msg in messages:
            candidates = self.extract_from_text(
                text=msg.get("text", ""),
                message_id=msg.get("id", 0),
                is_from_me=msg.get("is_from_me", False),
                **kwargs,
            )
            results.append(
                {
                    "message_id": msg.get("id"),
                    "candidates": candidates,
                }
            )
        return results


# Registry of extractors
_extractor_registry: dict[str, type[ExtractorAdapter]] = {}


def register_extractor(
    name: str, extractor_class: type[ExtractorAdapter]
) -> type[ExtractorAdapter]:
    """Register an extractor class.

    Args:
        name: Unique name for the extractor
        extractor_class: The extractor adapter class to register

    Returns:
        The registered class (for use as a decorator)
    """
    _extractor_registry[name] = extractor_class
    return extractor_class


def get_extractor(name: str) -> type[ExtractorAdapter] | None:
    """Get a registered extractor class by name.

    Args:
        name: Name of the registered extractor

    Returns:
        The extractor class if found, None otherwise
    """
    return _extractor_registry.get(name)


def list_extractors() -> list[str]:
    """List all registered extractor names.

    Returns:
        List of registered extractor names
    """
    return list(_extractor_registry.keys())
