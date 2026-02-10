"""Cascade mobilization classifier.

Flow:
1. Decide if a reply is needed (rules first, intent fallback)
2. If no reply needed -> NONE/CLOSING
3. If reply needed -> map to mobilization via rules or intent
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from jarvis.classifiers.factory import SingletonFactory
from jarvis.classifiers.intent_classifier import IntentClassifier, KeywordIntentClassifier
from jarvis.classifiers.response_mobilization import (
    MobilizationResult,
    ResponsePressure,
    ResponseType,
    classify_response_pressure,
)
from jarvis.text_normalizer import is_acknowledgment_only, is_reaction

logger = logging.getLogger(__name__)

RULE_CONFIDENCE_THRESHOLD = 0.80
SHOULD_REPLY_THRESHOLD = 0.80

UNIFIED_INTENTS = [
    "no_reply_ack",
    "no_reply_closing",
    "reply_casual_chat",
    "reply_question_info",
    "reply_request_action",
    "reply_urgent_action",
    "reply_emotional_support",
]


@dataclass
class ShouldReplyDecision:
    """Decision output for replyability gating."""

    should_reply: bool
    confidence: float
    method: str
    intent: str | None = None


def _rule_should_reply(text: str, rule_result: MobilizationResult) -> ShouldReplyDecision:
    stripped = text.strip()
    if not stripped:
        return ShouldReplyDecision(False, 1.0, "should_reply_rule_empty")

    if is_reaction(stripped) or is_acknowledgment_only(stripped):
        return ShouldReplyDecision(False, 0.98, "should_reply_rule_fastpath")

    if rule_result.features.get("is_negated_request"):
        return ShouldReplyDecision(False, 0.98, "should_reply_rule_negated")

    if rule_result.pressure in {ResponsePressure.HIGH, ResponsePressure.MEDIUM}:
        return ShouldReplyDecision(True, 0.92, "should_reply_rule_pressure")
    if rule_result.pressure == ResponsePressure.NONE:
        return ShouldReplyDecision(False, 0.92, "should_reply_rule_pressure")

    # LOW is often ambiguous; keep confidence intentionally below threshold.
    return ShouldReplyDecision(True, 0.60, "should_reply_rule_ambiguous")


def _intent_implies_reply(intent: str) -> bool:
    return not intent.startswith("no_reply")


def _intent_to_mobilization(
    intent: str,
    confidence: float,
    *,
    features: dict[str, bool],
) -> MobilizationResult:
    """Map intent labels to mobilization classes."""
    if intent.startswith("no_reply"):
        pressure = ResponsePressure.NONE
        response_type = ResponseType.CLOSING
    elif intent.startswith("reply_question"):
        pressure = ResponsePressure.HIGH
        response_type = ResponseType.ANSWER
    elif intent.startswith("reply_request") or intent.startswith("reply_urgent"):
        pressure = ResponsePressure.HIGH
        response_type = ResponseType.COMMITMENT
    elif intent.startswith("reply_emotional"):
        pressure = ResponsePressure.MEDIUM
        response_type = ResponseType.EMOTIONAL
    else:
        pressure = ResponsePressure.LOW
        response_type = ResponseType.OPTIONAL

    return MobilizationResult(
        pressure=pressure,
        response_type=response_type,
        confidence=confidence,
        features=features,
        method="intent_fallback",
    )


class MobilizationCascade:
    """Rules-first mobilization classifier with intent fallback."""

    def __init__(
        self,
        intent_classifier: IntentClassifier | None = None,
        confidence_threshold: float = RULE_CONFIDENCE_THRESHOLD,
        should_reply_threshold: float = SHOULD_REPLY_THRESHOLD,
    ) -> None:
        self._intent_classifier = intent_classifier or KeywordIntentClassifier()
        self._confidence_threshold = confidence_threshold
        self._should_reply_threshold = should_reply_threshold
        self._intent_fallback_disabled = False

    def classify(self, text: str) -> MobilizationResult:
        """Classify mobilization using two-step gate -> intent cascade."""
        rule_result = classify_response_pressure(text)

        # Step 1: should we reply?
        gate = _rule_should_reply(text, rule_result)
        intent_result = None
        if gate.confidence < self._should_reply_threshold and not self._intent_fallback_disabled:
            try:
                intent_result = self._intent_classifier.classify(text, UNIFIED_INTENTS)
                gate = ShouldReplyDecision(
                    should_reply=_intent_implies_reply(intent_result.intent),
                    confidence=max(gate.confidence, intent_result.confidence),
                    method="should_reply_intent",
                    intent=intent_result.intent,
                )
            except Exception as exc:  # pragma: no cover - defensive fallback
                self._intent_fallback_disabled = True
                logger.warning("Should-reply fallback failed, using rule gate: %s", exc)

        if not gate.should_reply:
            used_intent_fallback = gate.method == "should_reply_intent"
            return MobilizationResult(
                pressure=ResponsePressure.NONE,
                response_type=ResponseType.CLOSING,
                confidence=gate.confidence,
                features={**rule_result.features, "intent_fallback": used_intent_fallback},
                method=f"{gate.method}_no_reply",
            )

        # Step 2: if rules are strong and reply is needed, keep rule mobilization.
        if rule_result.confidence >= self._confidence_threshold:
            return rule_result

        if self._intent_fallback_disabled:
            return rule_result

        try:
            if intent_result is None:
                intent_result = self._intent_classifier.classify(text, UNIFIED_INTENTS)
            return _intent_to_mobilization(
                intent_result.intent,
                max(intent_result.confidence, rule_result.confidence),
                features={**rule_result.features, "intent_fallback": True},
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._intent_fallback_disabled = True
            logger.warning("Intent fallback failed, using rule result: %s", exc)
            return rule_result


_factory: SingletonFactory[MobilizationCascade] = SingletonFactory(MobilizationCascade)


def get_mobilization_cascade() -> MobilizationCascade:
    """Get singleton cascade classifier."""
    return _factory.get()


def reset_mobilization_cascade() -> None:
    """Reset singleton cascade classifier (for tests)."""
    _factory.reset()


def classify_response_pressure_with_cascade(text: str) -> MobilizationResult:
    """Convenience wrapper for singleton cascade classification."""
    return get_mobilization_cascade().classify(text)


def classify_with_cascade(
    text: str,
    *,
    intent_classifier: IntentClassifier | None = None,
    threshold: float = RULE_CONFIDENCE_THRESHOLD,
    should_reply_threshold: float = SHOULD_REPLY_THRESHOLD,
) -> MobilizationResult:
    """Classify with cascade and allow dependency injection for tests/customization."""
    if (
        intent_classifier is None
        and threshold == RULE_CONFIDENCE_THRESHOLD
        and should_reply_threshold == SHOULD_REPLY_THRESHOLD
    ):
        return classify_response_pressure_with_cascade(text)
    return MobilizationCascade(
        intent_classifier=intent_classifier,
        confidence_threshold=threshold,
        should_reply_threshold=should_reply_threshold,
    ).classify(text)


__all__ = [
    "RULE_CONFIDENCE_THRESHOLD",
    "SHOULD_REPLY_THRESHOLD",
    "ShouldReplyDecision",
    "UNIFIED_INTENTS",
    "MobilizationCascade",
    "classify_with_cascade",
    "classify_response_pressure_with_cascade",
    "get_mobilization_cascade",
    "reset_mobilization_cascade",
]
