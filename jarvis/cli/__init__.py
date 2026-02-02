"""JARVIS CLI package.

This package provides developer tools for JARVIS:
- serve: Start the API server
- health: System health status
- benchmark: Performance benchmarks
- db: Database operations

User-facing features (chat, reply, search, etc.) are in the desktop app.
"""

# Re-export core CLI functions
# Re-export core controllers for testing
from core.health import get_degradation_controller, reset_degradation_controller
from core.memory import get_memory_controller, reset_memory_controller
from jarvis._cli_main import (
    cleanup,
    cmd_benchmark,
    cmd_db,
    cmd_health,
    cmd_serve,
    cmd_version,
    console,
    create_parser,
    main,
    run,
    setup_logging,
)

# Re-export system functions
from jarvis.system import (
    FEATURE_CHAT,
    FEATURE_IMESSAGE,
    _check_imessage_access,
    initialize_system,
)

__all__ = [
    # Commands
    "cmd_benchmark",
    "cmd_db",
    "cmd_health",
    "cmd_serve",
    "cmd_version",
    # Core
    "cleanup",
    "console",
    "create_parser",
    "main",
    "run",
    "setup_logging",
    # Feature constants and system functions
    "FEATURE_CHAT",
    "FEATURE_IMESSAGE",
    "_check_imessage_access",
    "initialize_system",
    # Core controllers (for testing)
    "get_degradation_controller",
    "reset_degradation_controller",
    "get_memory_controller",
    "reset_memory_controller",
]
