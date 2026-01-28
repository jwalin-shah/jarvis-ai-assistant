"""Model registry for JARVIS v2.

Defines available MLX models for generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelSpec:
    """Specification for an MLX model."""

    id: str
    path: str  # HuggingFace path
    display_name: str
    size_gb: float
    quality: Literal["basic", "good", "excellent"]
    description: str


# Available models
MODELS: dict[str, ModelSpec] = {
    # LFM2.5 - Best for text replies (fast + natural)
    "lfm2.5-1.2b": ModelSpec(
        id="lfm2.5-1.2b",
        path="LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit",  # Official source
        display_name="LFM2.5 1.2B",
        size_gb=0.5,
        quality="excellent",
        description="Fastest + most natural replies",
    ),
    "lfm2.5-1.2b-8bit": ModelSpec(
        id="lfm2.5-1.2b-8bit",
        path="LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit",
        display_name="LFM2.5 1.2B (8-bit)",
        size_gb=0.7,
        quality="excellent",
        description="Slightly better quality than 4-bit",
    ),
    # Llama 3.2 - Fast alternative
    "llama-3.2-1b": ModelSpec(
        id="llama-3.2-1b",
        path="mlx-community/Llama-3.2-1B-Instruct-4bit",
        display_name="Llama 3.2 1B",
        size_gb=0.7,
        quality="good",
        description="Very fast, simple replies",
    ),
    # Qwen3 - Good quality, slower
    "qwen3-4b": ModelSpec(
        id="qwen3-4b",
        path="Qwen/Qwen3-4B-MLX-4bit",
        display_name="Qwen3 4B",
        size_gb=2.1,
        quality="excellent",
        description="Best quality, slower",
    ),
    "qwen3-1.7b": ModelSpec(
        id="qwen3-1.7b",
        path="mlx-community/Qwen3-1.7B-4bit",
        display_name="Qwen3 1.7B",
        size_gb=1.2,
        quality="excellent",
        description="Good balance of speed/quality",
    ),
}

# Default model - LFM2.5 for fast, natural text replies
DEFAULT_MODEL = "lfm2.5-1.2b"


def get_model_spec(model_id: str) -> ModelSpec:
    """Get model specification by ID.

    Args:
        model_id: Model identifier

    Returns:
        ModelSpec for the model

    Raises:
        KeyError: If model not found
    """
    if model_id not in MODELS:
        raise KeyError(f"Unknown model: {model_id}. Available: {list(MODELS.keys())}")
    return MODELS[model_id]


def get_recommended_model(available_ram_gb: float = 8.0) -> ModelSpec:
    """Get recommended model based on available RAM.

    Args:
        available_ram_gb: Available system RAM in GB

    Returns:
        Best ModelSpec for the RAM constraint
    """
    # Leave ~4GB for OS and other apps
    usable_ram = available_ram_gb - 4.0

    # Find best quality model that fits
    candidates = [
        spec for spec in MODELS.values()
        if spec.size_gb <= usable_ram
    ]

    if not candidates:
        # Fall back to smallest model
        return MODELS["lfm2.5-1.2b"]

    # Sort by quality (excellent > good > basic), then by size (larger is better)
    quality_order = {"excellent": 3, "good": 2, "basic": 1}
    candidates.sort(key=lambda s: (quality_order[s.quality], s.size_gb), reverse=True)

    return candidates[0]
