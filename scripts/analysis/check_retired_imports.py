"""Fail CI if retired facade imports are reintroduced."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIRS = ("jarvis", "api", "models", "integrations", "core", "tests")

BANNED_PATTERNS = [
    re.compile(r"\bfrom\s+jarvis\.(errors|cache|router|socket_server)\s+import\b"),
    re.compile(r"\bimport\s+jarvis\.(errors|cache|router|socket_server)\b"),
]


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for directory in PYTHON_DIRS:
        files.extend((REPO_ROOT / directory).rglob("*.py"))
    return files


def main() -> int:
    violations: list[str] = []
    for path in _iter_python_files():
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in BANNED_PATTERNS):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{i}: {line.strip()}")

    if violations:
        print("Retired facade imports detected:")
        for violation in violations:
            print(f"  {violation}")
        return 1

    print("No retired facade imports found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
