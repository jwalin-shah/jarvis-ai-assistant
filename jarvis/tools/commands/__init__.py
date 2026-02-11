"""Tool command implementations.

This package contains concrete tool implementations organized by category:
- train: Model training commands
- eval: Evaluation commands
- data: Data pipeline commands
- label: Labeling commands
- db: Database commands
"""

# Import modules to trigger registration
# These will be imported lazily when needed via the CLI
__all__ = [
    "CategoryTrainer",
]


# Deferred imports to avoid circular dependencies
def __getattr__(name):
    if name == "CategoryTrainer":
        from jarvis.tools.commands.train import CategoryTrainer

        return CategoryTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
