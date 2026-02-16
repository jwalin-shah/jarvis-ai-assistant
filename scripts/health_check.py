#!/usr/bin/env python3
"""Comprehensive health check for JARVIS system.

Validates:
- Python module imports
- Database connectivity
- API router registration
- iMessage/AppleScript functionality
- Frontend build integrity
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path


class HealthCheckError(Exception):
    """Health check failure with actionable message."""

    def __init__(self, component: str, message: str, fix: str | None = None):
        self.component = component
        self.message = message
        self.fix = fix
        super().__init__(f"[{component}] {message}")


def check_color(success: bool) -> str:
    """Return colored checkmark/x."""
    return "✅" if success else "❌"


def check_module_imports() -> list[HealthCheckError]:
    """Check all critical Python modules can be imported."""
    errors = []
    critical_modules = [
        "api.main",
        "api.routers.conversations",
        "api.routers.drafts",
        "api.routers.health",
        "integrations.imessage.sender",
        "integrations.imessage.reader",
    ]

    for module_name in critical_modules:
        try:
            importlib.import_module(module_name)
        except Exception as e:
            errors.append(
                HealthCheckError(
                    component="backend.imports",
                    message=f"Cannot import {module_name}: {e}",
                    fix=f"Check {module_name.replace('.', '/')}.py for syntax/import errors",
                )
            )

    return errors


def check_api_routers() -> list[HealthCheckError]:
    """Check all API routers are properly registered."""
    errors = []

    try:
        from api.main import create_app

        app = create_app()

        # Get all registered routes
        registered_paths = set()
        for route in app.routes:
            if hasattr(route, "path"):
                registered_paths.add(route.path)

        # Check critical endpoints exist
        critical_endpoints = [
            "/health",
            "/conversations",
            "/drafts",
        ]

        for endpoint in critical_endpoints:
            matching = any(endpoint in path for path in registered_paths)
            if not matching:
                errors.append(
                    HealthCheckError(
                        component="api.routers",
                        message=f"Critical endpoint {endpoint} not registered",
                        fix="Check api/main.py - router may be commented out or missing",
                    )
                )

    except Exception as e:
        errors.append(
            HealthCheckError(
                component="api.routers",
                message=f"Failed to check routers: {e}",
                fix="Run: python -c 'from api import app'",
            )
        )

    return errors


def check_applescript() -> list[HealthCheckError]:
    """Check AppleScript/iMessage functionality."""
    errors = []

    # Check Messages app is accessible
    try:
        result = subprocess.run(
            ["osascript", "-e", 'tell application "Messages" to return count of chats'],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            errors.append(
                HealthCheckError(
                    component="applescript.messages",
                    message=f"Cannot access Messages app: {result.stderr}",
                    fix="Grant Automation permission to Terminal/your app in "
                    "System Settings > Privacy & Security > Automation",
                )
            )
    except subprocess.TimeoutExpired:
        errors.append(
            HealthCheckError(
                component="applescript.messages",
                message="Messages app check timed out",
                fix="Messages app may be frozen - try restarting it",
            )
        )
    except Exception as e:
        errors.append(
            HealthCheckError(
                component="applescript.messages",
                message=f"AppleScript error: {e}",
                fix="Ensure osascript is available and Messages app is installed",
            )
        )

    return errors


def check_chat_db() -> list[HealthCheckError]:
    """Check iMessage database accessibility."""
    errors = []
    chat_db = Path.home() / "Library" / "Messages" / "chat.db"

    if not chat_db.exists():
        errors.append(
            HealthCheckError(
                component="database.chatdb",
                message=f"chat.db not found at {chat_db}",
                fix="Ensure you're on macOS and iMessage is set up",
            )
        )
    elif not os.access(chat_db, os.R_OK):
        errors.append(
            HealthCheckError(
                component="database.chatdb",
                message=f"chat.db is not readable at {chat_db}",
                fix="Grant Full Disk Access permission in System Settings > "
                "Privacy & Security > Full Disk Access",
            )
        )

    return errors


def check_imessage_sender() -> list[HealthCheckError]:
    """Check iMessage sender configuration."""
    errors = []

    try:
        from integrations.imessage.sender import IMessageSender

        sender = IMessageSender()

        # Validate method signatures haven't changed
        import inspect

        sig = inspect.signature(sender.send_message)
        params = list(sig.parameters.keys())

        expected = ["text", "recipient", "chat_id", "is_group"]
        for param in expected:
            if param not in params:
                errors.append(
                    HealthCheckError(
                        component="imessage.sender",
                        message=f"send_message missing parameter: {param}",
                        fix="Check integrations/imessage/sender.py - API may have changed",
                    )
                )

    except Exception as e:
        errors.append(
            HealthCheckError(
                component="imessage.sender",
                message=f"IMessageSender check failed: {e}",
                fix="Check integrations/imessage/sender.py for import/syntax errors",
            )
        )

    return errors


def check_frontend() -> list[HealthCheckError]:
    """Check frontend build integrity."""
    errors = []

    desktop_dir = Path(__file__).parent.parent / "desktop"

    # Check package.json exists
    if not (desktop_dir / "package.json").exists():
        errors.append(
            HealthCheckError(
                component="frontend.build",
                message="package.json not found",
                fix="Ensure desktop/ directory contains a valid npm project",
            )
        )
        return errors

    # Check TypeScript can compile (optional, may be slow)
    # This is a lightweight check - full type check is done in CI

    # Check critical files exist
    critical_files = [
        "src/lib/api/client.ts",
        "src/lib/db/direct.ts",
        "src/lib/stores/conversations.svelte.ts",
    ]

    for file_path in critical_files:
        full_path = desktop_dir / file_path
        if not full_path.exists():
            errors.append(
                HealthCheckError(
                    component="frontend.build",
                    message=f"Critical file missing: {file_path}",
                    fix=f"Restore {file_path} or check git status",
                )
            )

    return errors


def check_schemas() -> list[HealthCheckError]:
    """Check API schemas are consistent."""
    errors = []

    try:
        from api.schemas import SendMessageRequest

        # Check required fields exist
        fields = SendMessageRequest.model_fields
        required = ["text", "recipient", "is_group"]

        for field in required:
            if field not in fields:
                errors.append(
                    HealthCheckError(
                        component="api.schemas",
                        message=f"SendMessageRequest missing field: {field}",
                        fix="Check api/schemas/drafts.py for schema definition",
                    )
                )

    except Exception as e:
        errors.append(
            HealthCheckError(
                component="api.schemas",
                message=f"Schema validation failed: {e}",
                fix="Check api/schemas/__init__.py for missing exports",
            )
        )

    return errors


def print_report(errors: list[HealthCheckError]) -> int:
    """Print health check report and return exit code."""
    print("\n" + "=" * 60)
    print("JARVIS Health Check Report")
    print("=" * 60)

    if not errors:
        print(f"\n{check_color(True)} All checks passed!")
        print("\nSystem is ready to run. Start with: make launch")
        return 0

    # Group errors by component
    by_component: dict[str, list[HealthCheckError]] = {}
    for error in errors:
        by_component.setdefault(error.component, []).append(error)

    print(f"\n❌ {len(errors)} issue(s) found:\n")

    for component, component_errors in by_component.items():
        print(f"\n[{component}]")
        print("-" * 40)
        for error in component_errors:
            print(f"  • {error.message}")
            if error.fix:
                print(f"    Fix: {error.fix}")

    print("\n" + "=" * 60)
    print("Run this check before committing: python scripts/health_check.py")
    print("=" * 60)

    return 1


def main() -> int:
    """Run all health checks."""
    print("Running JARVIS health checks...")

    all_errors: list[HealthCheckError] = []

    checks = [
        ("Python imports", check_module_imports),
        ("API routers", check_api_routers),
        ("API schemas", check_schemas),
        ("iMessage database", check_chat_db),
        ("iMessage sender", check_imessage_sender),
        ("AppleScript", check_applescript),
        ("Frontend", check_frontend),
    ]

    for name, check_func in checks:
        print(f"\n  Checking {name}...", end=" ")
        try:
            errors = check_func()
            if errors:
                print("❌")
                all_errors.extend(errors)
            else:
                print("✓")
        except Exception as e:
            print(f"❌ (exception: {e})")
            all_errors.append(
                HealthCheckError(
                    component="health_check",
                    message=f"Check {name} crashed: {e}",
                    fix="Check the error output above",
                )
            )

    return print_report(all_errors)


if __name__ == "__main__":
    sys.exit(main())
