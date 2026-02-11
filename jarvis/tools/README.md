# JARVIS Developer Tools

Unified, maintainable tooling for JARVIS ML operations, data pipelines, and maintenance tasks.

## Overview

This package replaces the ad-hoc scripts in `scripts/` with a structured, testable tool framework. Each tool inherits from `Tool`, uses standardized logging, and produces consistent results.

## Quick Start

```bash
# List available tools
jarvis-tools --list

# Train category classifier
jarvis-tools train category --data-dir data/soc_categories

# Validate configuration without running
jarvis-tools train category --data-dir data/soc_categories --dry-run

# Check database health
jarvis-tools db health

# Get help
jarvis-tools train category --help
```

## Architecture

```
jarvis/tools/
├── __init__.py          # Package exports
├── __main__.py          # Entry point: python -m jarvis.tools
├── cli.py               # Typer CLI application
├── base.py              # Tool base class
├── logging.py           # Standardized logging
├── registry.py          # Tool discovery
├── deprecation.py       # Migration utilities
└── commands/            # Tool implementations
    ├── train.py         # Training commands
    ├── eval.py          # Evaluation commands
    ├── data.py          # Data pipeline commands
    ├── db.py            # Database commands
    └── ...
```

## Creating a New Tool

```python
# jarvis/tools/commands/my_tool.py
from jarvis.tools.base import Tool, ToolResult
from jarvis.tools.registry import tool

@tool("my_tool", "Description of what this tool does", "1.0.0")
class MyTool(Tool):
    """Detailed docstring explaining the tool."""
    
    required_config = ["input_path"]
    path_config_keys = ["input_path", "output_path"]
    
    def validate(self) -> list[str]:
        """Add custom validation."""
        errors = super().validate()
        # Add custom checks
        return errors
    
    def run(self, **kwargs) -> ToolResult:
        """Execute the tool."""
        config = {**self.config, **kwargs}
        
        try:
            # Do work
            self.logger.info("Starting...")
            
            return ToolResult(
                success=True,
                message="Completed successfully",
                metrics={"processed": 100},
            )
        except Exception as e:
            self.logger.exception("Tool failed")
            return ToolResult(
                success=False,
                message=f"Failed: {e}",
            )
```

The tool is automatically registered and available via:
```bash
jarvis-tools my-tool --help
```

## Testing

```bash
# Run all tool tests
pytest tests/tools/ -v

# Unit tests only (fast)
pytest tests/tools/unit/ -v

# Integration tests
pytest tests/tools/integration/ -v

# Migration parity tests
pytest tests/tools/e2e/ -v

# With coverage
pytest tests/tools/unit/ --cov=jarvis.tools --cov-report=html
```

## Migration from Scripts

This package is actively replacing scripts in `scripts/`. See:
- `docs/SCRIPTS_MIGRATION_PLAN.md` - Full migration plan
- `docs/DEPRECATION.md` - Deprecation timeline

### Legacy Script Wrapper

To wrap a legacy script during migration:

```python
# scripts/legacy/my_script.py
"""[DEPRECATED] Use `jarvis-tools my-tool` instead."""

from jarvis.tools.deprecation import emit_deprecation_warning
emit_deprecation_warning("my_script.py")

# Delegate to new tool
from jarvis.tools.commands.my_tool import MyTool

def main():
    import argparse
    parser = argparse.ArgumentParser()
    # ... old args
    args = parser.parse_args()
    
    tool = MyTool({...})
    result = tool.run()
    
    if not result:
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Configuration

Tools use a configuration dictionary passed at initialization:

```python
tool = CategoryTrainer({
    "data_dir": Path("data/soc_categories"),
    "label_map": "4class",
    "seed": 42,
})
```

CLI options override config values:

```python
# Config value
tool = CategoryTrainer({"seed": 42})

# Override at runtime
result = tool.run(seed=100)  # Uses 100, not 42
```

## Logging

Each tool gets its own logger that writes to:
- Console (INFO and above)
- `logs/tools/{tool_name}.log` (DEBUG and above)

```python
self.logger.info("Processing...")
self.logger.debug("Detailed debug info")
```

## Contributing

1. Create tool in `jarvis/tools/commands/`
2. Add tests in `tests/tools/unit/test_{command}.py`
3. Update `docs/SCRIPTS_MIGRATION_PLAN.md`
4. Run `make check` before committing
