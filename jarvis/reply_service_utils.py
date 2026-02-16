"""Utility helpers for reply service orchestration."""

from __future__ import annotations

from typing import Any

from jarvis.classifiers.response_mobilization import MobilizationResult, ResponsePressure
from jarvis.contracts.pipeline import CategoryType, ClassificationResult, UrgencyLevel


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely cast value to float with fallback default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pressure_from_classification(classification: ClassificationResult) -> ResponsePressure:
    """Infer response pressure from classification metadata/category/urgency."""
    pressure_raw = classification.metadata.get("mobilization_pressure")
    if isinstance(pressure_raw, str):
        try:
            return ResponsePressure(pressure_raw)
        except ValueError:
            pass

    if classification.category in {
        CategoryType.ACKNOWLEDGE,
        CategoryType.CLOSING,
        CategoryType.OFF_TOPIC,
    }:
        return ResponsePressure.NONE

    if classification.urgency == UrgencyLevel.HIGH:
        return ResponsePressure.HIGH
    if classification.urgency == UrgencyLevel.MEDIUM:
        return ResponsePressure.MEDIUM
    return ResponsePressure.LOW


def max_tokens_for_pressure(pressure: ResponsePressure) -> int:
    """Return max generation token budget for pressure level."""
    return 20 if pressure == ResponsePressure.NONE else 40


def build_thread_context(conversation_messages: list[Any]) -> list[str]:
    """Build bounded thread context list from message-like objects/dicts."""
    thread: list[str] = []
    for msg in reversed(conversation_messages):
        if isinstance(msg, dict):
            msg_text = msg.get("text", "")
            is_from_me = msg.get("is_from_me", False)
            sender = msg.get("sender_name") or msg.get("sender", "")
        else:
            msg_text = getattr(msg, "text", None) or ""
            is_from_me = getattr(msg, "is_from_me", False)
            sender = getattr(msg, "sender_name", None) or getattr(msg, "sender", "")

        if msg_text:
            prefix = "Me" if is_from_me else sender
            thread.append(f"{prefix}: {msg_text}")

    return thread[-10:]


def to_legacy_response(
    response_text: str,
    confidence: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Convert generation output into legacy route response shape."""

    def confidence_label(conf: float) -> str:
        if conf >= 0.7:
            return "high"
        if conf >= 0.45:
            return "medium"
        return "low"

    return {
        "type": str(metadata.get("type", "generated")),
        "response": response_text,
        "confidence": confidence_label(confidence),
        "confidence_score": confidence,
        "similarity_score": float(metadata.get("similarity_score", 0.0)),
        "similar_triggers": metadata.get("similar_triggers"),
        "reason": str(metadata.get("reason", "")),
    }


def build_mobilization_hint(mobilization: MobilizationResult) -> str:
    """Build a generation instruction hint based on response mobilization."""
    if mobilization.pressure == ResponsePressure.HIGH:
        if mobilization.response_type.value == "commitment":
            return "Respond with a clear commitment (accept, decline, or defer)."
        if mobilization.response_type.value == "answer":
            return "Answer the question directly and clearly."
        if mobilization.response_type.value == "confirmation":
            return "Confirm or deny clearly."
        return "Respond directly to their question."
    if mobilization.pressure == ResponsePressure.MEDIUM:
        return "Respond with appropriate emotion and empathy."
    if mobilization.pressure == ResponsePressure.LOW:
        return "Keep the response brief and casual."
    return "A brief acknowledgment is fine."
