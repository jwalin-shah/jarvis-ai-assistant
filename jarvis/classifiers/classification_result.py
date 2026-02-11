"""Shared classification result builder used by router and reply_service."""

from __future__ import annotations

from jarvis.classifiers.category_classifier import classify_category
from jarvis.classifiers.response_mobilization import (
    MobilizationResult,
    ResponsePressure,
)
from jarvis.contracts.pipeline import (
    CategoryType,
    ClassificationResult,
    IntentType,
    UrgencyLevel,
)

_CATEGORY_TO_INTENT: dict[str, IntentType] = {
    "question": IntentType.QUESTION,
    "request": IntentType.REQUEST,
    "statement": IntentType.STATEMENT,
    "emotion": IntentType.STATEMENT,
    "closing": IntentType.STATEMENT,
    "acknowledge": IntentType.STATEMENT,
}


def to_intent_type(category: str) -> IntentType:
    """Map a category string to an IntentType enum value."""
    return _CATEGORY_TO_INTENT.get(category, IntentType.UNKNOWN)


def build_classification_result(
    incoming: str,
    thread: list[str],
    mobilization: MobilizationResult,
    *,
    extra_metadata: dict[str, object] | None = None,
) -> ClassificationResult:
    """Classify incoming text and build a ClassificationResult.

    Args:
        incoming: The incoming message text.
        thread: Conversation thread for context.
        mobilization: Pre-computed mobilization result.
        extra_metadata: Additional metadata keys to merge (e.g. complexity_score).
    """
    category_result = classify_category(
        incoming,
        context=thread,
        mobilization=mobilization,
    )

    if category_result.category == "closing":
        category = CategoryType.CLOSING
    elif category_result.category == "acknowledge":
        category = CategoryType.ACKNOWLEDGE
    else:
        category = CategoryType.FULL_RESPONSE

    if mobilization.pressure == ResponsePressure.HIGH:
        urgency = UrgencyLevel.HIGH
    elif mobilization.pressure == ResponsePressure.MEDIUM:
        urgency = UrgencyLevel.MEDIUM
    else:
        urgency = UrgencyLevel.LOW

    metadata: dict[str, object] = {
        "category_name": category_result.category,
        "category_confidence": category_result.confidence,
        "category_method": category_result.method,
        "mobilization_pressure": mobilization.pressure.value,
        "mobilization_response_type": mobilization.response_type.value,
        "mobilization_confidence": mobilization.confidence,
        "mobilization_method": mobilization.method,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return ClassificationResult(
        intent=to_intent_type(category_result.category),
        category=category,
        urgency=urgency,
        confidence=min(1.0, (mobilization.confidence + category_result.confidence) / 2.0),
        requires_knowledge=category_result.category in {"question", "request"},
        metadata=metadata,
    )
