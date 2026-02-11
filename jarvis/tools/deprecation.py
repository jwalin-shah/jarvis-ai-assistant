"""Deprecation utilities for script migration.

Helps manage the transition from ad-hoc scripts to the tools package.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from pathlib import Path


class DeprecationConfig:
    """Configuration for script deprecation.

    Maps legacy script names to their replacements and removal timeline.
    """

    # script_name -> (new_command, removal_version, migration_guide)
    DEPRECATED_SCRIPTS: dict[str, tuple[str, str, str]] = {
        "train_category_svm.py": (
            "jarvis-tools train category",
            "2.0.0",
            "docs/TOOLS_MIGRATION.md#train-category",
        ),
        "eval_classifiers.py": (
            "jarvis-tools eval classifiers",
            "2.0.0",
            "docs/TOOLS_MIGRATION.md#eval-classifiers",
        ),
        "db_maintenance.py": (
            "jarvis-tools db maintain",
            "2.0.0",
            "docs/TOOLS_MIGRATION.md#db-maintain",
        ),
        "check_prompt_version.py": (
            "jarvis-tools check prompt",
            "2.0.0",
            "docs/TOOLS_MIGRATION.md#check-prompt",
        ),
        "verify_contracts.py": (
            "jarvis-tools check contracts",
            "2.0.0",
            "docs/TOOLS_MIGRATION.md#check-contracts",
        ),
        "generate_ft_configs.py": (
            "jarvis-tools config generate",
            "2.0.0",
            "docs/TOOLS_MIGRATION.md#config-generate",
        ),
        "code_owner.py": (
            "jarvis-tools util codeowners",
            "2.0.0",
            "docs/TOOLS_MIGRATION.md#util-codeowners",
        ),
        # Add more as they are migrated
    }

    @classmethod
    def warn(cls, script_name: str) -> None:
        """Emit deprecation warning for a script.

        Args:
            script_name: Name of the script being run
        """
        if script_name not in cls.DEPRECATED_SCRIPTS:
            return

        new_cmd, removal, guide = cls.DEPRECATED_SCRIPTS[script_name]

        message = (
            f"{script_name} is deprecated and will be removed in v{removal}. "
            f"Use `{new_cmd}` instead. See {guide} for migration guide."
        )

        warnings.warn(
            message,
            DeprecationWarning,
            stacklevel=3,
        )

    @classmethod
    def is_deprecated(cls, script_name: str) -> bool:
        """Check if a script is deprecated."""
        return script_name in cls.DEPRECATED_SCRIPTS

    @classmethod
    def get_replacement(cls, script_name: str) -> str | None:
        """Get the replacement command for a deprecated script."""
        if script_name not in cls.DEPRECATED_SCRIPTS:
            return None
        return cls.DEPRECATED_SCRIPTS[script_name][0]

    @classmethod
    def list_deprecated(cls) -> list[tuple[str, str, str]]:
        """List all deprecated scripts with their replacements.

        Returns:
            List of (script_name, replacement, removal_version) tuples
        """
        return [(name, info[0], info[1]) for name, info in cls.DEPRECATED_SCRIPTS.items()]

    @classmethod
    def generate_status_markdown(cls) -> str:
        """Generate migration status markdown."""
        lines = [
            "# Script Migration Status\n",
            "| Script | Status | New Command | Removal |",
            "|--------|--------|-------------|----------|",
        ]

        scripts_dir = Path("scripts")

        for script_name in sorted(cls.DEPRECATED_SCRIPTS.keys()):
            new_cmd, removal, _ = cls.DEPRECATED_SCRIPTS[script_name]

            # Check if legacy script still exists
            legacy_path = scripts_dir / script_name
            if legacy_path.exists():
                status = "🔄 Migrating"
            else:
                status = "✅ Complete"

            lines.append(f"| {script_name} | {status} | `{new_cmd}` | {removal} |")

        lines.append("\n_Last updated: Auto-generated_")
        return "\n".join(lines)


def deprecated_script(new_command: str, removal_version: str = "2.0.0"):
    """Decorator to mark a function/script as deprecated.

    Args:
        new_command: The replacement command to use
        removal_version: Version when this will be removed

    Example:
        @deprecated_script("jarvis-tools train category", "2.0.0")
        def main():
            # Legacy script implementation
            pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"This script is deprecated and will be removed in v{removal_version}. "
                f"Use `{new_command}` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def add_deprecation_header(
    script_path: Path,
    new_command: str,
    removal_version: str = "2.0.0",
) -> None:
    """Add deprecation warning header to a script file.

    Args:
        script_path: Path to the script to modify
        new_command: Replacement command
        removal_version: When this will be removed
    """
    content = script_path.read_text()

    # Check if already has deprecation warning
    if "[DEPRECATED]" in content or "DeprecationWarning" in content:
        return

    # Create deprecation header
    header = f'''#!/usr/bin/env python3
"""[DEPRECATED] Use `{new_command}` instead.

This script is preserved for backward compatibility during migration.
It will be removed in v{removal_version}.

See docs/SCRIPTS_MIGRATION_PLAN.md for details.
"""

import warnings
warnings.warn(
    "This script is deprecated. Use `{new_command}` instead.",
    DeprecationWarning,
    stacklevel=2,
)

'''

    # Find where docstring ends and insert after
    if '"""' in content:
        # Find end of docstring
        parts = content.split('"""', 2)
        if len(parts) >= 3:
            # Reconstruct with deprecation
            new_content = parts[0] + header + '"""' + parts[1] + '"""' + parts[2]
        else:
            new_content = header + content
    else:
        new_content = header + content

    script_path.write_text(new_content)
    print(f"Added deprecation header to {script_path}")


# Convenience for scripts that import at module level
def emit_deprecation_warning(script_name: str) -> None:
    """Emit deprecation warning for the calling script.

    Call this at the top of a legacy script to warn users.

    Example:
        # At top of scripts/train_category_svm.py
        from jarvis.tools.deprecation import emit_deprecation_warning
        emit_deprecation_warning("train_category_svm.py")
    """
    DeprecationConfig.warn(script_name)
