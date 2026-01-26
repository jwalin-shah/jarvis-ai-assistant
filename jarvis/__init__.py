"""JARVIS - Local-first AI assistant for macOS.

Provides CLI commands and setup wizard for iMessage management.
"""

from jarvis.intent import (
    IntentClassifier,
    IntentResult,
    IntentType,
    get_intent_classifier,
    reset_intent_classifier,
)

__version__ = "1.0.0"

__all__ = [
    "IntentClassifier",
    "IntentResult",
    "IntentType",
    "get_intent_classifier",
    "reset_intent_classifier",
]
