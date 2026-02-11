"""JARVIS developer tooling package.

This package provides maintainable, testable tools for:
- Model training (category, mobilization, fact filtering)
- Evaluation and benchmarking
- Data pipeline operations
- Database maintenance
- LLM-assisted labeling

Migration Status:
    This package is replacing the ad-hoc scripts in scripts/.
    See docs/SCRIPTS_MIGRATION_PLAN.md for details.

Usage:
    # Via CLI
    $ jarvis-tools --help
    $ jarvis-tools train category --data-dir data/soc_categories

    # Via Python API
    from jarvis.tools.commands.train import CategoryTrainer
    tool = CategoryTrainer({"data_dir": Path("data/soc_categories")})
    result = tool.run()

Quick Start:
    1. List available tools:
       $ jarvis-tools list

    2. Get help for a specific tool:
       $ jarvis-tools train category --help

    3. Validate configuration without running:
       $ jarvis-tools train category --dry-run
"""

__version__ = "1.0.0"
__all__ = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "tool",
    "get_tool_logger",
]

from jarvis.tools.base import Tool, ToolResult
from jarvis.tools.logging import get_tool_logger
from jarvis.tools.registry import ToolRegistry, tool
