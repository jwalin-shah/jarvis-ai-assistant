"""Tool registry for discovery and CLI integration.

Provides automatic tool registration and discovery.
"""

from __future__ import annotations

import importlib
import pkgutil

from jarvis.tools.base import Tool


class ToolRegistry:
    """Registry for all available tools.

    Tools are automatically registered when imported or when
    using the @tool decorator.

    Example:
        # Register via decorator
        @tool("my_tool", "Does something useful")
        class MyTool(Tool):
            def run(self, **kwargs) -> ToolResult:
                ...

        # Lookup
        tool_class = ToolRegistry.get("my_tool")
        tool = tool_class(config={"key": "value"})
        result = tool.run()
    """

    _tools: dict[str, type[Tool]] = {}
    _discovered: bool = False

    @classmethod
    def register(cls, tool_class: type[Tool]) -> type[Tool]:
        """Register a tool class.

        Args:
            tool_class: Tool class to register

        Returns:
            The registered class (for decorator use)
        """
        if not issubclass(tool_class, Tool):
            raise TypeError(f"Tool must inherit from Tool: {tool_class}")

        if not tool_class.name:
            raise ValueError(f"Tool must have a name: {tool_class}")

        cls._tools[tool_class.name] = tool_class
        return tool_class

    @classmethod
    def get(cls, name: str) -> type[Tool] | None:
        """Get a tool by name.

        Args:
            name: Tool identifier

        Returns:
            Tool class or None if not found
        """
        return cls._tools.get(name)

    @classmethod
    def list_tools(cls) -> dict[str, type[Tool]]:
        """List all registered tools.

        Returns:
            Dictionary mapping tool names to classes
        """
        return cls._tools.copy()

    @classmethod
    def discover(cls, force: bool = False) -> None:
        """Auto-discover tools from commands directory.

        Imports all modules in jarvis.tools.commands to trigger
        decorator registration.

        Args:
            force: Rediscover even if already done
        """
        if cls._discovered and not force:
            return

        from jarvis.tools import commands

        for importer, modname, ispkg in pkgutil.iter_modules(commands.__path__):
            try:
                importlib.import_module(f"jarvis.tools.commands.{modname}")
            except ImportError as e:
                import logging

                logging.getLogger(__name__).warning(f"Failed to import commands.{modname}: {e}")

        cls._discovered = True

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (mainly for testing)."""
        cls._tools.clear()
        cls._discovered = False


def tool(name: str, description: str, version: str = "1.0.0"):
    """Decorator to register a tool with metadata.

    Args:
        name: Tool identifier (snake_case, unique)
        description: Short human-readable description
        version: Semver string

    Returns:
        Decorator function

    Example:
        @tool("train_category", "Train category classifier", "1.0.0")
        class CategoryTrainer(Tool):
            def run(self, **kwargs) -> ToolResult:
                ...
    """

    def decorator(cls: type[Tool]) -> type[Tool]:
        cls.name = name
        cls.description = description
        cls.version = version
        return ToolRegistry.register(cls)

    return decorator


def list_commands() -> list[tuple[str, str, str]]:
    """List all available commands for CLI help.

    Returns:
        List of (name, description, version) tuples
    """
    ToolRegistry.discover()
    return [
        (name, cls.description, cls.version)
        for name, cls in sorted(ToolRegistry.list_tools().items())
    ]
