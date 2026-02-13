"""Model Registry for JARVIS.

Provides model specifications and functions for selecting the best model
based on system capabilities. Supports multiple model tiers for different
RAM configurations.

Usage:
    from models.registry import get_recommended_model, get_model_spec, MODEL_REGISTRY

    # Get the best model for the user's system
    spec = get_recommended_model(available_ram_gb=16)

    # Get a specific model by ID
    spec = get_model_spec("qwen-1.5b")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# Download configuration defaults
DEFAULT_DOWNLOAD_TIMEOUT = 60  # seconds
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 1.0  # seconds (exponential backoff base)

# Session cache for model availability checks
# Maps model_id -> bool (is_available)
_availability_cache: dict[str, bool] = {}


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a supported MLX model.

    Attributes:
        id: Unique identifier for the model (e.g., "qwen-0.5b").
        path: HuggingFace model path for MLX.
        display_name: Human-readable name for display in UI.
        size_gb: Approximate GPU memory usage in GB.
        min_ram_gb: Minimum system RAM required.
        quality_tier: Quality classification ("basic", "good", "excellent").
        description: User-facing description of the model.
        recommended_for: List of use cases this model excels at.
    """

    id: str
    path: str
    display_name: str
    size_gb: float
    min_ram_gb: int
    quality_tier: Literal["basic", "good", "excellent"]
    description: str
    recommended_for: list[str] = field(default_factory=list)

    @property
    def estimated_memory_mb(self) -> float:
        """Return estimated memory usage in MB."""
        return self.size_gb * 1024


# Registry of supported models - LFM only
MODEL_REGISTRY: dict[str, ModelSpec] = {
    # Base models
    "lfm-350m": ModelSpec(
        id="lfm-350m",
        path="mlx-community/LFM2-350M-4bit",
        display_name="LFM 2.5 350M (Base)",
        size_gb=0.35,
        min_ram_gb=4,
        quality_tier="basic",
        description="LFM 2.5 350M base model for fast fact extraction.",
        recommended_for=["fact_extraction"],
    ),
    "lfm-1.2b-base": ModelSpec(
        id="lfm-1.2b-base",
        path="mlx-community/LFM2.5-1.2B-Base-4bit",
        display_name="LFM 2.5 1.2B (Base)",
        size_gb=1.2,
        min_ram_gb=8,
        quality_tier="good",
        description="LFM 2.5 1.2B base model. No instruct tuning - raw completion.",
        recommended_for=["completion", "few_shot", "style_matching"],
    ),
    # Fine-tuned variants
    "lfm-1.2b-ft": ModelSpec(
        id="lfm-1.2b-ft",
        path="models/lfm-1.2b-final",
        display_name="LFM 2.5 1.2B Fine-Tuned",
        size_gb=1.2,
        min_ram_gb=8,
        quality_tier="excellent",
        description="LFM 2.5 1.2B fine-tuned on SOC + ORPO aligned. Best for texting.",
        recommended_for=["quick_replies", "natural_conversation", "drafting", "iMessage"],
    ),
    "lfm-1.2b-sft": ModelSpec(
        id="lfm-1.2b-sft",
        path="models/lfm-1.2b-soc-fused",
        display_name="LFM 2.5 1.2B SFT Only",
        size_gb=1.2,
        min_ram_gb=8,
        quality_tier="excellent",
        description="LFM 2.5 1.2B SFT fine-tuned on SOC conversations (no ORPO).",
        recommended_for=["quick_replies", "natural_conversation", "iMessage"],
    ),
    "lfm-0.3b-ft": ModelSpec(
        id="lfm-0.3b-ft",
        path="models/lfm-0.3b-soc-fused",
        display_name="LFM 2.5 0.3B Fine-Tuned",
        size_gb=0.3,
        min_ram_gb=4,
        quality_tier="basic",
        description="LFM 2.5 0.3B fine-tuned draft model for speculative decoding.",
        recommended_for=["speculative_decoding", "testing"],
    ),
}

# Default model ID when none specified
# LFM 2.5 fine-tuned is the default for conversational use cases
DEFAULT_MODEL_ID = "lfm-1.2b-ft"


def get_model_spec(model_id: str) -> ModelSpec | None:
    """Get model specification by ID.

    Args:
        model_id: The model identifier (e.g., "qwen-1.5b").

    Returns:
        ModelSpec if found, None otherwise.
    """
    return MODEL_REGISTRY.get(model_id)


def get_model_spec_by_path(model_path: str) -> ModelSpec | None:
    """Get model specification by HuggingFace path.

    Args:
        model_path: The HuggingFace model path (e.g., "mlx-community/Qwen2.5-1.5B-Instruct-4bit").

    Returns:
        ModelSpec if found, None otherwise.
    """
    for spec in MODEL_REGISTRY.values():
        if spec.path == model_path:
            return spec
    return None


def get_recommended_model(available_ram_gb: float) -> ModelSpec:
    """Return the best model for the user's available system RAM.

    Selects the highest quality model that fits within available RAM,
    with a safety buffer of 2GB for OS and other applications.

    Args:
        available_ram_gb: Available system RAM in GB.

    Returns:
        The recommended ModelSpec for the system.
    """
    # Sort models by quality tier (descending: excellent -> good -> basic)
    tier_order = {"excellent": 3, "good": 2, "basic": 1}
    sorted_models = sorted(
        MODEL_REGISTRY.values(),
        key=lambda m: tier_order[m.quality_tier],
        reverse=True,
    )

    # Find the best model that fits
    for model in sorted_models:
        # Add 2GB buffer for OS and other apps
        if available_ram_gb >= model.min_ram_gb:
            logger.debug(
                "Recommended model %s for %.1fGB RAM (min: %dGB)",
                model.id,
                available_ram_gb,
                model.min_ram_gb,
            )
            return model

    # Fallback to smallest model if somehow none fit
    fallback = MODEL_REGISTRY[DEFAULT_MODEL_ID]
    logger.warning(
        "No model fits %.1fGB RAM, falling back to %s",
        available_ram_gb,
        fallback.id,
    )
    return fallback


def get_all_models() -> list[ModelSpec]:
    """Return all available models sorted by quality tier.

    Returns:
        List of ModelSpec objects, sorted from basic to excellent.
    """
    tier_order = {"basic": 1, "good": 2, "excellent": 3}
    return sorted(
        MODEL_REGISTRY.values(),
        key=lambda m: tier_order[m.quality_tier],
    )


def is_model_available(model_id: str, use_cache: bool = True) -> bool:
    """Check if a model is downloaded and cached locally.

    Checks the HuggingFace cache directory for the model files.
    Results are cached for the session to avoid repeated filesystem checks.

    Args:
        model_id: The model identifier to check.
        use_cache: If True, uses cached result if available (default: True).

    Returns:
        True if model is cached locally, False otherwise.
    """
    global _availability_cache

    # Check session cache first
    if use_cache and model_id in _availability_cache:
        return _availability_cache[model_id]

    spec = get_model_spec(model_id)
    if spec is None:
        return False

    # Check HuggingFace cache
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_dir.exists():
        _availability_cache[model_id] = False
        return False

    # HuggingFace cache uses a specific naming pattern
    # e.g., models--mlx-community--Qwen2.5-0.5B-Instruct-4bit
    model_cache_name = f"models--{spec.path.replace('/', '--')}"
    model_cache_path = cache_dir / model_cache_name

    is_available = False
    if model_cache_path.exists():
        # Check if there are actual model files (snapshots directory with content)
        snapshots_dir = model_cache_path / "snapshots"
        if snapshots_dir.exists():
            # Check for at least one snapshot with model files
            for snapshot in snapshots_dir.iterdir():
                if snapshot.is_dir() and any(snapshot.iterdir()):
                    is_available = True
                    break

    # Cache the result
    _availability_cache[model_id] = is_available
    return is_available


def _invalidate_availability_cache(model_id: str | None = None) -> None:
    """Invalidate the model availability cache.

    Args:
        model_id: Specific model to invalidate, or None to clear entire cache.
    """
    global _availability_cache
    if model_id is None:
        _availability_cache.clear()
        logger.debug("Cleared entire model availability cache")
    elif model_id in _availability_cache:
        del _availability_cache[model_id]
        logger.debug("Invalidated availability cache for %s", model_id)


def clear_availability_cache() -> None:
    """Clear the model availability cache.

    Call this if models may have been downloaded/deleted outside of this module.
    """
    _invalidate_availability_cache(None)


def ensure_model_available(
    model_id: str,
    timeout: int = DEFAULT_DOWNLOAD_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
) -> bool:
    """Download model if not available.

    Uses huggingface_hub to download the model to the local cache.
    Implements retry logic with exponential backoff for network failures.

    Args:
        model_id: The model identifier to ensure is available.
        timeout: Timeout in seconds for the download (default: 60).
        max_retries: Maximum number of retry attempts (default: 3).
        retry_base_delay: Base delay in seconds for exponential backoff (default: 1.0).

    Returns:
        True if model is available (was cached or downloaded successfully),
        False if model_id is invalid or download failed after all retries.
    """
    spec = get_model_spec(model_id)
    if spec is None:
        logger.error("Unknown model ID: %s", model_id)
        return False

    # Check if already available (uses cache)
    if is_model_available(model_id):
        logger.debug("Model %s already available in cache", model_id)
        return True

    # Try to download with retries
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import (  # type: ignore[attr-defined]
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )
    except ImportError:
        logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Downloading model %s (%s)... [attempt %d/%d]",
                model_id,
                spec.path,
                attempt,
                max_retries,
            )

            # Configure etag_timeout and other network timeouts
            snapshot_download(
                repo_id=spec.path,
                etag_timeout=timeout,
                # Force online mode for download
                local_files_only=False,
            )

            logger.info("Model %s downloaded successfully", model_id)
            # Invalidate cache since model is now available
            _invalidate_availability_cache(model_id)
            return True

        except RepositoryNotFoundError:
            logger.error(
                "Model not found: '%s' does not exist on HuggingFace Hub. Check the model path: %s",
                model_id,
                spec.path,
            )
            return False  # Don't retry - model doesn't exist

        except GatedRepoError:
            logger.error(
                "Access denied: Model '%s' requires authentication. "
                "Run `huggingface-cli login` and accept the model terms at: "
                "https://huggingface.co/%s",
                model_id,
                spec.path,
            )
            return False  # Don't retry - auth issue

        except HfHubHTTPError as e:
            last_error = e
            status_code = getattr(e, "response", None)
            status_code = getattr(status_code, "status_code", None) if status_code else None

            if status_code == 429:
                logger.warning(
                    "Rate limited by HuggingFace Hub. Retrying in %ds...",
                    retry_base_delay * (2 ** (attempt - 1)),
                )
            elif status_code and 400 <= status_code < 500:
                logger.error("Client error downloading model %s: %s", model_id, e)
                return False  # Don't retry client errors
            else:
                logger.warning(
                    "Network error downloading model %s (attempt %d/%d): %s",
                    model_id,
                    attempt,
                    max_retries,
                    e,
                )

        except TimeoutError as e:
            last_error = e
            logger.warning(
                "Download timed out after %ds (attempt %d/%d). "
                "Try increasing timeout or check your network connection.",
                timeout,
                attempt,
                max_retries,
            )

        except OSError as e:
            last_error = e
            # Check for common network-related OSError
            error_str = str(e).lower()
            if "connection" in error_str or "network" in error_str or "timeout" in error_str:
                logger.warning(
                    "Network error downloading model %s (attempt %d/%d): %s",
                    model_id,
                    attempt,
                    max_retries,
                    e,
                )
            else:
                # Disk/permission errors - don't retry
                logger.error(
                    "Filesystem error downloading model %s: %s. Check disk space and permissions.",
                    model_id,
                    e,
                )
                return False

        except Exception as e:
            last_error = e
            logger.warning(
                "Unexpected error downloading model %s (attempt %d/%d): %s",
                model_id,
                attempt,
                max_retries,
                e,
            )

        # Exponential backoff before retry
        if attempt < max_retries:
            delay = retry_base_delay * (2 ** (attempt - 1))
            logger.debug("Waiting %.1fs before retry...", delay)
            time.sleep(delay)

    # All retries exhausted
    logger.error(
        "Failed to download model %s after %d attempts. Last error: %s",
        model_id,
        max_retries,
        last_error,
    )
    return False
