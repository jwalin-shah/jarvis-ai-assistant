from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.classifiers.response_mobilization import MobilizationResult
    from jarvis.prompts.constants import CategoryConfig


def resolve_category(
    last_message: str,
    context: list[str] | None = None,
    tone: str = "casual",
    mobilization: MobilizationResult | None = None,
) -> str:
    """Classify a message into an optimization category."""
    from jarvis.prompts.constants import CATEGORY_MAP

    try:
        from jarvis.classifiers.category_classifier import classify_category

        result = classify_category(last_message, context=context, mobilization=mobilization)
        return result.category
    except Exception:
        return CATEGORY_MAP.get(tone, "statement")


def get_category_config(category: str) -> CategoryConfig:
    """Get routing configuration for a category."""
    from jarvis.prompts.constants import CATEGORY_CONFIGS

    return CATEGORY_CONFIGS.get(category, CATEGORY_CONFIGS["statement"])
