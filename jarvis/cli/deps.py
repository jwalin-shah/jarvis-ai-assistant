"""CLI dependencies - centralized imports for testability.

This module provides a single import point for external dependencies used by the CLI.
Tests can patch `jarvis.cli.deps.*` to mock these dependencies consistently.

Usage in CLI code:
    from jarvis.cli.deps import console, get_degradation_controller

Usage in tests:
    @patch("jarvis.cli.deps.console")
    @patch("jarvis.cli.deps.get_degradation_controller")
    def test_something(self, mock_ctrl, mock_console):
        ...
"""

from rich.console import Console

from core.health import get_degradation_controller, reset_degradation_controller
from core.memory import get_memory_controller, reset_memory_controller
from jarvis.context import ContextFetcher
from jarvis.intent import IntentClassifier, IntentType
from jarvis.system import (
    FEATURE_CHAT,
    FEATURE_IMESSAGE,
    _check_imessage_access,
    initialize_system,
)

# Rich console - single instance for CLI output
console = Console()

# Models (lazy import in actual usage, but exposed here for patching)
# Note: MLX models should be imported lazily in actual code to avoid loading
# until needed, but we expose the import path here for test patching.

def cleanup() -> None:
    """Cleanup function to reset singletons and release resources.

    Called at the end of CLI execution to ensure clean state.
    """
    from models import reset_generator

    try:
        reset_generator()
    except Exception:
        pass  # Ignore errors during cleanup

    try:
        reset_degradation_controller()
    except Exception:
        pass

    try:
        reset_memory_controller()
    except Exception:
        pass


__all__ = [
    # Console
    "console",
    # Controllers
    "get_degradation_controller",
    "reset_degradation_controller",
    "get_memory_controller",
    "reset_memory_controller",
    # Context and intent
    "ContextFetcher",
    "IntentClassifier",
    "IntentType",
    # System
    "FEATURE_CHAT",
    "FEATURE_IMESSAGE",
    "_check_imessage_access",
    "initialize_system",
    # Cleanup
    "cleanup",
]
