#!/usr/bin/env python3
"""Verify prompt version is updated when prompt content changes.

This script compares the current prompt content against the last committed
version to ensure PROMPT_VERSION is incremented when templates change.

Usage:
    python scripts/check_prompt_version.py              # Check current state
    python scripts/check_prompt_version.py --ci        # CI mode (exit codes)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROMPTS_DIR = PROJECT_ROOT / "jarvis" / "prompts"
VERSION_FILE = PROMPTS_DIR / "constants.py"


def _get_file_content_at_ref(ref: str, path: Path) -> str:
    """Get file content at a specific git ref."""
    try:
        rel = path.relative_to(PROJECT_ROOT)
        result = subprocess.run(
            ["git", "show", f"{ref}:{rel}"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            return result.stdout
        return ""
    except Exception:
        return ""


def get_prompts_content_at_ref(ref: str) -> tuple[str, str]:
    """Get version from constants.py and concatenated .py content for template extraction.

    Returns:
        (version_string, full_content_for_templates)
    """
    version_content = _get_file_content_at_ref(ref, VERSION_FILE)
    version = extract_version(version_content)

    parts: list[str] = []
    for py_path in sorted(PROMPTS_DIR.glob("*.py")):
        if py_path.name.startswith("__"):
            continue
        content = _get_file_content_at_ref(ref, py_path)
        if content:
            parts.append(content)
    return version, "\n\n".join(parts)


def extract_version(content: str) -> str:
    """Extract PROMPT_VERSION from content."""
    match = re.search(r'PROMPT_VERSION\s*=\s*["\']([^"\']+)["\']', content)
    return match.group(1) if match else ""


def extract_templates(content: str) -> dict[str, str]:
    """Extract prompt template definitions from content."""
    templates = {}

    # Match PromptTemplate definitions
    pattern = (
        r'(\w+)\s*=\s*PromptTemplate\(\s*name\s*=\s*["\']([^"\']+)'
        r'["\'].*?template\s*=\s*"""(.*?)"""'
    )
    for match in re.finditer(pattern, content, re.DOTALL):
        var_name = match.group(1)
        template_name = match.group(2)
        template_content = match.group(3)
        templates[var_name] = {
            "name": template_name,
            "content": template_content.strip(),
        }

    # Match few-shot example lists
    example_pattern = r"(\w+_EXAMPLES)\s*:\s*list\[.*?\]\s*=\s*\[(.*?)\]"
    for match in re.finditer(example_pattern, content, re.DOTALL):
        var_name = match.group(1)
        examples_content = match.group(2)
        templates[var_name] = {"content": examples_content.strip()}

    return templates


def templates_equal(old: dict, new: dict) -> bool:
    """Compare two template dictionaries."""
    if set(old.keys()) != set(new.keys()):
        return False

    for key in old:
        if old[key].get("content") != new[key].get("content"):
            return False

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Check prompt version compliance")
    parser.add_argument("--ci", action="store_true", help="CI mode (non-zero exit on failure)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if not VERSION_FILE.exists():
        print("ERROR: jarvis/prompts/constants.py not found")
        return 1

    # Read current version and aggregate prompt module content for template comparison
    current_version_content = VERSION_FILE.read_text()
    current_version = extract_version(current_version_content)
    current_content = current_version_content
    for py_path in sorted(PROMPTS_DIR.glob("*.py")):
        if py_path.name.startswith("__"):
            continue
        current_content += "\n\n" + py_path.read_text()

    if args.verbose:
        print(f"Current prompt version: {current_version}")

    # Try to get previous version and content from HEAD
    previous_version, previous_content = get_prompts_content_at_ref("HEAD")
    if not previous_content and not previous_version:
        if args.verbose:
            print("No previous version found (new repo or prompts not in git)")
        return 0

    if args.verbose:
        print(f"Previous prompt version: {previous_version}")

    # Extract templates
    current_templates = extract_templates(current_content)
    previous_templates = extract_templates(previous_content)

    # Check if templates changed
    templates_changed = not templates_equal(previous_templates, current_templates)

    if not templates_changed:
        if args.verbose:
            print("No template changes detected")
        return 0

    if args.verbose:
        print("Template changes detected!")
        # Show which templates changed
        old_keys = set(previous_templates.keys())
        new_keys = set(current_templates.keys())

        added = new_keys - old_keys
        removed = old_keys - new_keys
        common = old_keys & new_keys

        for key in added:
            print(f"  + Added: {key}")
        for key in removed:
            print(f"  - Removed: {key}")
        for key in common:
            if previous_templates[key] != current_templates[key]:
                print(f"  ~ Modified: {key}")

    # Check if version was updated
    version_updated = current_version != previous_version

    if templates_changed and not version_updated:
        print("ERROR: Prompt templates changed but PROMPT_VERSION was not updated!")
        print(f"  Current version: {current_version}")
        print(
            "  Please update PROMPT_VERSION and PROMPT_LAST_UPDATED in jarvis/prompts/constants.py"
        )
        if args.ci:
            return 1
        return 0

    if templates_changed and version_updated:
        print(
            f"OK: Templates changed and version updated ({previous_version} -> {current_version})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
