"""Model loading and generation interfaces.

Workstream 8 implements against these contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class GenerationRequest:
    """Request for text generation.

    Default parameters are optimized for LFM2.5-1.2B-Instruct.
    Source: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct

    Attributes:
        prompt: Input prompt for generation.
        context_documents: RAG context documents to inject.
        few_shot_examples: Few-shot examples as (input, output) tuples.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (lower = more focused).
        top_p: Nucleus sampling threshold.
        top_k: Limit vocabulary to top-k tokens.
        repetition_penalty: Penalty to reduce repetition (1.0 = no penalty).
        stop_sequences: Optional list of sequences that stop generation.
    """

    prompt: str
    context_documents: list[str] = field(default_factory=list)
    few_shot_examples: list[tuple[str, str]] = field(default_factory=list)
    max_tokens: int = 100
    temperature: float = 0.1
    top_p: float = 0.1
    top_k: int = 50
    repetition_penalty: float = 1.05
    stop_sequences: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate field constraints."""
        if not self.prompt.strip():
            msg = "Prompt cannot be empty"
            raise ValueError(msg)
        if self.max_tokens < 1:
            msg = f"max_tokens must be >= 1, got {self.max_tokens}"
            raise ValueError(msg)
        if not 0.0 <= self.temperature <= 2.0:
            msg = f"temperature must be 0.0-2.0, got {self.temperature}"
            raise ValueError(msg)
        if not 0.0 <= self.top_p <= 1.0:
            msg = f"top_p must be 0.0-1.0, got {self.top_p}"
            raise ValueError(msg)
        if self.top_k < 1:
            msg = f"top_k must be >= 1, got {self.top_k}"
            raise ValueError(msg)
        if self.repetition_penalty < 1.0:
            msg = f"repetition_penalty must be >= 1.0, got {self.repetition_penalty}"
            raise ValueError(msg)


@dataclass
class GenerationResponse:
    """Response from text generation.

    Attributes:
        text: Generated text output.
        tokens_used: Number of tokens used in generation.
        generation_time_ms: Generation time in milliseconds.
        model_name: Name of the model used.
        used_template: Whether a template was used instead of generation.
        template_name: Name of template if used_template is True.
        finish_reason: Reason generation finished (stop/length/template/fallback/error).
        error: Error message if finish_reason is "error" or "fallback".
    """

    text: str
    tokens_used: int
    generation_time_ms: float
    model_name: str
    used_template: bool
    template_name: str | None
    finish_reason: str
    error: str | None = None

    def __post_init__(self) -> None:
        """Validate field constraints."""
        valid_finish_reasons = {"stop", "length", "template", "fallback", "error"}
        if self.finish_reason not in valid_finish_reasons:
            msg = f"finish_reason must be one of {valid_finish_reasons}, got {self.finish_reason}"
            raise ValueError(msg)
        if self.tokens_used < 0:
            msg = f"tokens_used must be >= 0, got {self.tokens_used}"
            raise ValueError(msg)
        if self.generation_time_ms < 0:
            msg = f"generation_time_ms must be >= 0, got {self.generation_time_ms}"
            raise ValueError(msg)
        if self.finish_reason in {"error", "fallback"} and not self.error:
            msg = f'finish_reason "{self.finish_reason}" requires error message'
            raise ValueError(msg)
        if self.used_template and not self.template_name:
            msg = "used_template=True requires template_name"
            raise ValueError(msg)


class Generator(Protocol):
    """Interface for text generation (Workstream 8)."""

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate response. May use template or model.

        Args:
            request: Generation request with prompt, context, and parameters.

        Returns:
            Generation response with text and metadata.
        """
        ...

    def is_loaded(self) -> bool:
        """Check if model is loaded in memory.

        Returns:
            True if model is currently loaded in memory.
        """
        ...

    def load(self) -> bool:
        """Load model into memory. Returns success.

        Returns:
            True if model was successfully loaded.
        """
        ...

    def unload(self) -> None:
        """Unload model to free memory.

        Note:
            After calling this method, is_loaded() should return False.
        """
        ...

    def get_memory_usage_mb(self) -> float:
        """Return current memory usage of the model.

        Returns:
            Memory usage in megabytes, or 0.0 if model is not loaded.
        """
        ...
