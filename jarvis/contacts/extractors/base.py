"""Base adapter interface for fact candidate extractors.

Defines the common schema and interface for all extractors in the bakeoff.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractedCandidate:
    """Common output schema for all extractors.

    This is the normalized representation that all adapters must produce,
    regardless of their internal extraction mechanism.
    """

    span_text: str  # The extracted entity text
    span_label: str  # Normalized label (place, org, person_name, etc.)
    score: float  # Extractor confidence (0.0-1.0)
    start_char: int  # Character offset start in source text
    end_char: int  # Character offset end in source text
    fact_type: str = "other_personal_fact"  # Mapped fact type
    extractor_metadata: dict[str, Any] = field(default_factory=dict)  # Tool-specific extras


@dataclass
class ExtractionResult:
    """Result of extracting candidates from a single message."""

    message_id: int
    candidates: list[ExtractedCandidate]
    extractor_name: str  # Which extractor produced this
    processing_time_ms: float = 0.0
    error: str | None = None  # If extraction failed


class ExtractorAdapter(ABC):
    """Base class for all extractor adapters.

    Each adapter wraps a specific extraction tool (GLiNER, GLiNER2, NuExtract, etc.)
    and normalizes its output to the common ExtractedCandidate schema.
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        self.name = name
        self.config = config or {}
        self._model: Any = None

    @property
    @abstractmethod
    def supported_labels(self) -> list[str]:
        """Return the list of labels this extractor supports."""
        ...

    @property
    @abstractmethod
    def default_threshold(self) -> float:
        """Return the default confidence threshold."""
        ...

    @abstractmethod
    def _load_model(self) -> Any:
        """Lazy-load the underlying model. Called internally."""
        ...

    @abstractmethod
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
        """Extract candidates from a single message.

        Args:
            text: The message text to extract from
            message_id: iMessage ROWID
            chat_id: Chat ROWID
            is_from_me: True if sent by user
            sender_handle_id: Handle ROWID of sender
            message_date: iMessage date (Core Data timestamp)
            threshold: Override confidence threshold
            context_prev: Previous messages for context
            context_next: Next messages for context

        Returns:
            List of extracted candidates
        """
        ...

    def extract_batch(
        self,
        messages: list[dict[str, Any]],
        batch_size: int = 32,
        threshold: float | None = None,
    ) -> list[ExtractionResult]:
        """Extract candidates from a batch of messages.

        Default implementation processes sequentially. Adapters may override
        for true batch processing if the underlying tool supports it.

        Args:
            messages: List of message dicts with at least 'text' and 'message_id'
            batch_size: Batch size (if supported by extractor)
            threshold: Confidence threshold

        Returns:
            List of ExtractionResult objects
        """
        import time

        results: list[ExtractionResult] = []

        for msg in messages:
            start = time.perf_counter()
            try:
                candidates = self.extract_from_text(
                    text=msg.get("text", ""),
                    message_id=msg["message_id"],
                    chat_id=msg.get("chat_id"),
                    is_from_me=msg.get("is_from_me"),
                    sender_handle_id=msg.get("sender_handle_id"),
                    message_date=msg.get("message_date"),
                    threshold=threshold,
                    context_prev=msg.get("context_prev"),
                    context_next=msg.get("context_next"),
                )
                elapsed = (time.perf_counter() - start) * 1000
                results.append(
                    ExtractionResult(
                        message_id=msg["message_id"],
                        candidates=candidates,
                        extractor_name=self.name,
                        processing_time_ms=elapsed,
                    )
                )
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                results.append(
                    ExtractionResult(
                        message_id=msg["message_id"],
                        candidates=[],
                        extractor_name=self.name,
                        processing_time_ms=elapsed,
                        error=str(e),
                    )
                )

        return results

    def to_dict(self) -> dict[str, Any]:
        """Serialize adapter configuration."""
        return {
            "name": self.name,
            "supported_labels": self.supported_labels,
            "default_threshold": self.default_threshold,
            "config": self.config,
        }


# Registry of available extractors
_EXTRACTOR_REGISTRY: dict[str, type[ExtractorAdapter]] = {}


def register_extractor(name: str, cls: type[ExtractorAdapter]) -> None:
    """Register an extractor adapter class."""
    _EXTRACTOR_REGISTRY[name] = cls


def get_extractor_class(name: str) -> type[ExtractorAdapter] | None:
    """Get an extractor adapter class by name."""
    return _EXTRACTOR_REGISTRY.get(name)


def list_extractors() -> list[str]:
    """List all registered extractor names."""
    return sorted(_EXTRACTOR_REGISTRY.keys())


def create_extractor(name: str, config: dict[str, Any] | None = None) -> ExtractorAdapter:
    """Factory function to create an extractor by name."""
    cls = get_extractor_class(name)
    if cls is None:
        available = ", ".join(list_extractors())
        raise ValueError(f"Unknown extractor '{name}'. Available: {available}")
    return cls(config=config)
