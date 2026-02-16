"""Resilience layer for model loading and generation.

Provides memory-pressure-aware fallback tiers so the generator
degrades gracefully instead of crashing on 8GB RAM systems.

Fallback tiers (highest to lowest quality):
1. Full model generation (normal path)
2. Template-only response (no model needed)
3. Static fallback response (zero cost)

Usage:
    from models.resilience import should_skip_model_load, get_fallback_response

    if should_skip_model_load():
        return get_fallback_response(request.prompt)
"""

from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger(__name__)

# Memory pressure threshold above which we skip model loads.
# macOS pressure_level: 0 = good, >1000 = warning, >2500 = critical.
# macOS vm.memory_pressure is a relative score, not a percentage.
# 1000 is a safe threshold for 8GB systems with compression.
PRESSURE_SKIP_THRESHOLD = 2500

# Swap percentage above which we skip model loads (non-macOS fallback).
SWAP_SKIP_PERCENT = 60.0

# Static fallback when no model or template is available.
FALLBACK_RESPONSE = "I'm a bit overloaded right now. Give me a moment and try again."


class FallbackTier(Enum):
    """Generation quality tiers, from best to worst."""

    MODEL = "model"  # Full model generation
    TEMPLATE = "template"  # Template-matched response
    STATIC = "static"  # Hardcoded fallback string


def check_memory_pressure() -> tuple[bool, str]:
    """Check if memory pressure is too high for model loading.

    Returns:
        Tuple of (is_high_pressure, reason).
        is_high_pressure=True means we should skip model loads.
    """
    try:
        from jarvis.utils.memory import IS_MACOS, get_macos_memory_pressure, get_swap_info

        if IS_MACOS:
            pressure = get_macos_memory_pressure()
            if pressure is not None and pressure.pressure_level >= PRESSURE_SKIP_THRESHOLD:
                reason = (
                    f"macOS memory pressure {pressure.pressure_level} "
                    f">= {PRESSURE_SKIP_THRESHOLD} "
                    f"(compressed: {pressure.compressed_mb:.0f}MB, "
                    f"free: {pressure.free_mb:.0f}MB)"
                )
                logger.warning("High memory pressure: %s", reason)
                return True, reason
        else:
            swap = get_swap_info()
            if swap["percent"] >= SWAP_SKIP_PERCENT:
                reason = (
                    f"Swap usage {swap['percent']:.1f}% >= {SWAP_SKIP_PERCENT}% "
                    f"({swap['used_mb']:.0f}MB used)"
                )
                logger.warning("High swap usage: %s", reason)
                return True, reason

    except Exception as e:
        logger.debug("Memory pressure check failed (proceeding with load): %s", e)

    return False, ""


def should_skip_model_load() -> bool:
    """Check if model loading should be skipped due to memory pressure.

    Returns:
        True if memory pressure is too high for safe model loading.
    """
    is_high, _reason = check_memory_pressure()
    return is_high


def get_fallback_response() -> str:
    """Return a static fallback response for when generation is unavailable.

    Returns:
        A safe, generic response string.
    """
    return FALLBACK_RESPONSE
