"""Check maximum line limits for key modules to prevent monolith growth."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Start with core hotspots; expand as decomposition continues.
MAX_LINES = {
    "jarvis/reply_service.py": 1200,
    "jarvis/watcher.py": 1200,
    "models/templates.py": 2400,
    "api/routers/websocket.py": 1000,
    "api/routers/graph.py": 850,
    "api/routers/priority.py": 850,
    "api/routers/drafts.py": 750,
}


def main() -> int:
    violations: list[str] = []
    for rel_path, max_lines in MAX_LINES.items():
        path = REPO_ROOT / rel_path
        if not path.exists():
            violations.append(f"{rel_path}: missing file")
            continue

        line_count = sum(1 for _ in path.open("r", encoding="utf-8"))
        if line_count > max_lines:
            violations.append(f"{rel_path}: {line_count} lines (limit {max_lines})")

    if violations:
        print("Module size gate failed:")
        for violation in violations:
            print(f"  {violation}")
        return 1

    print("Module size gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
