#!/usr/bin/env python3
"""Filter a hub task file to exclude already-reviewed files.

Reads .hub-progress.json from a lane's worktree and removes files that
have already been reviewed, outputting a filtered task to stdout.

Usage:
    python3 progress_filter.py <task-file> <lane> <worktree-path>

Exit codes:
    0 - files remain to review
    1 - all files reviewed (or no task for this lane)
"""

import json
import re
import sys
from pathlib import Path


def load_progress(worktree: Path) -> dict:
    """Load .hub-progress.json from worktree, or empty dict if missing."""
    progress_file = worktree / ".hub-progress.json"
    if not progress_file.exists():
        return {}
    try:
        return json.loads(progress_file.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def extract_lane_section(task_content: str, lane: str) -> str:
    """Extract the section for a specific lane from the task markdown."""
    lane_upper = lane.upper()
    pattern = rf"## Lane {lane_upper}\s*\n(.*?)(?=\n## Lane [A-Z]|\n## [^L]|\Z)"
    match = re.search(pattern, task_content, re.DOTALL)
    if not match:
        return ""
    text = match.group(1).strip()
    return "" if text.upper() == "IDLE" else text


def extract_file_list(section: str) -> list[str]:
    """Extract file paths from numbered list items like '1. `path/to/file.py`'."""
    # Match patterns: N. `file/path` or - `file/path`
    pattern = r"(?:^\d+\.\s+|^-\s+)`([^`]+)`"
    return re.findall(pattern, section, re.MULTILINE)


def filter_section(section: str, reviewed_files: set[str]) -> tuple[str, int, int]:
    """Remove already-reviewed files from a task section.

    Returns (filtered_section, total_files, remaining_files).
    """
    lines = section.split("\n")
    filtered_lines = []
    total_files = 0
    remaining_files = 0

    for line in lines:
        # Check if this line contains a file reference
        file_match = re.match(r"(\s*)(?:\d+\.\s+|[-*]\s+)`([^`]+)`(.*)", line)
        if file_match:
            total_files += 1
            filepath = file_match.group(2)
            if filepath in reviewed_files:
                continue  # Skip reviewed file
            remaining_files += 1
        filtered_lines.append(line)

    return "\n".join(filtered_lines), total_files, remaining_files


def build_prior_work_summary(progress: dict) -> str:
    """Build a summary of prior work from progress data."""
    parts = []
    pass_num = progress.get("pass", 0)
    reviewed = progress.get("files_reviewed", [])
    fixes = progress.get("fixes", [])

    parts.append(f"Prior passes completed: {pass_num}")
    parts.append(f"Files already reviewed: {len(reviewed)}")
    if fixes:
        parts.append(f"Fixes made so far: {len(fixes)}")
        for fix in fixes[-5:]:  # Show last 5 fixes
            parts.append(f"  - {fix.get('file', '?')}: {fix.get('description', '?')}")
        if len(fixes) > 5:
            parts.append(f"  ... and {len(fixes) - 5} more")

    return "\n".join(parts)


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <task-file> <lane> <worktree-path>", file=sys.stderr)
        sys.exit(2)

    task_file = Path(sys.argv[1])
    lane = sys.argv[2].lower()
    worktree = Path(sys.argv[3])

    if not task_file.exists():
        print(f"Task file not found: {task_file}", file=sys.stderr)
        sys.exit(2)

    # Load task and progress
    task_content = task_file.read_text()
    progress = load_progress(worktree)

    # Extract lane section
    section = extract_lane_section(task_content, lane)
    if not section:
        sys.exit(1)  # No task for this lane

    # Check if already complete
    if progress.get("status") == "complete":
        sys.exit(1)

    # Get reviewed files
    reviewed_files = set(progress.get("files_reviewed", []))
    if not reviewed_files:
        # No progress yet, output original section
        print(section)
        sys.exit(0)

    # Filter out reviewed files
    filtered, total, remaining = filter_section(section, reviewed_files)

    if remaining == 0:
        sys.exit(1)  # All files reviewed

    # Build output with prior work context
    summary = build_prior_work_summary(progress)
    output = f"""## Prior Work Summary
{summary}

## Remaining Task ({remaining} of {total} files)
{filtered}"""

    print(output)
    sys.exit(0)


if __name__ == "__main__":
    main()
