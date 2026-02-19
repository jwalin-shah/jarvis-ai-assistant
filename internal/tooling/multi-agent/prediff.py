#!/usr/bin/env python3
"""Pre-diff processor for multi-agent hub reviews.

Converts raw git diffs into structured summaries that are:
- 60-80% smaller (fewer tokens for review agents)
- More actionable (functions changed, imports added, ownership flags)
- Easier to parse (structured sections vs raw patch format)

Usage:
    python3 prediff.py <diff-file-or-stdin> [--lane a|b|c] [--contracts-file path]
    git diff main..branch | python3 prediff.py --lane b
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Lane ownership definitions (must match hub_lib.sh)
LANE_OWNED: dict[str, list[str]] = {
    "a": [
        "desktop/",
        "api/",
        "jarvis/router.py",
        "jarvis/prompts.py",
        "jarvis/retrieval/",
        "jarvis/reply_service.py",
    ],
    "b": [
        "models/",
        "jarvis/classifiers/",
        "jarvis/extractors/",
        "jarvis/contacts/",
        "jarvis/graph/",
        "jarvis/search/",
        "scripts/train",
        "scripts/extract",
    ],
    "c": ["tests/", "benchmarks/", "evals/"],
}
SHARED_PATHS = ["jarvis/contracts/"]


@dataclass
class FileDiff:
    path: str
    added: int = 0
    removed: int = 0
    imports_added: list[str] = field(default_factory=list)
    imports_removed: list[str] = field(default_factory=list)
    functions_changed: list[str] = field(default_factory=list)
    classes_changed: list[str] = field(default_factory=list)
    hunks: list[str] = field(default_factory=list)


def parse_diff(diff_text: str) -> list[FileDiff]:
    """Parse a unified diff into structured FileDiff objects."""
    files: list[FileDiff] = []
    current: FileDiff | None = None
    current_hunk_header = ""

    for line in diff_text.splitlines():
        # New file
        if line.startswith("diff --git"):
            match = re.search(r"b/(.+)$", line)
            if match:
                current = FileDiff(path=match.group(1))
                files.append(current)
            continue

        if current is None:
            continue

        # Hunk header - extract function/class context
        if line.startswith("@@"):
            match = re.search(r"@@.*@@\s*(.*)", line)
            if match:
                current_hunk_header = match.group(1).strip()
                if current_hunk_header:
                    # Track function/class changes
                    if re.match(r"(def |async def )", current_hunk_header):
                        func_name = re.match(r"(?:async )?def (\w+)", current_hunk_header)
                        if func_name and func_name.group(1) not in current.functions_changed:
                            current.functions_changed.append(func_name.group(1))
                    elif re.match(r"class ", current_hunk_header):
                        cls_name = re.match(r"class (\w+)", current_hunk_header)
                        if cls_name and cls_name.group(1) not in current.classes_changed:
                            current.classes_changed.append(cls_name.group(1))
                    if current_hunk_header not in current.hunks:
                        current.hunks.append(current_hunk_header)
            continue

        # Added lines
        if line.startswith("+") and not line.startswith("+++"):
            current.added += 1
            stripped = line[1:].strip()
            if re.match(r"(from |import )", stripped):
                current.imports_added.append(stripped)
            # Track new function/class definitions
            func_match = re.match(r"(?:async )?def (\w+)", stripped)
            if func_match and func_match.group(1) not in current.functions_changed:
                current.functions_changed.append(func_match.group(1))
            cls_match = re.match(r"class (\w+)", stripped)
            if cls_match and cls_match.group(1) not in current.classes_changed:
                current.classes_changed.append(cls_match.group(1))

        # Removed lines
        elif line.startswith("-") and not line.startswith("---"):
            current.removed += 1
            stripped = line[1:].strip()
            if re.match(r"(from |import )", stripped):
                current.imports_removed.append(stripped)

    return files


def classify_ownership(path: str, lane: str | None) -> str:
    """Classify a file path as owned/shared/violation."""
    for shared in SHARED_PATHS:
        if path.startswith(shared):
            return "SHARED"

    if lane:
        for owned in LANE_OWNED.get(lane, []):
            if path.startswith(owned):
                return "OWNED"
        # Check if it belongs to another lane
        for other_lane, paths in LANE_OWNED.items():
            if other_lane == lane:
                continue
            for owned in paths:
                if path.startswith(owned):
                    return f"VIOLATION (Lane {other_lane.upper()})"
        return "UNKNOWN"
    return ""


def format_summary(
    files: list[FileDiff], lane: str | None = None, contracts_text: str | None = None
) -> str:
    """Format parsed diff into a structured review summary."""
    lines: list[str] = []

    lines.append("# Diff Summary for Review")
    lines.append("")

    if lane:
        lines.append(f"**Lane:** {lane.upper()}")
        lines.append(f"**Owned paths:** {', '.join(LANE_OWNED.get(lane, []))}")
        lines.append("")

    # Overview table
    lines.append("## Files Changed")
    lines.append("")
    lines.append("| File | +/- | Ownership | Key Changes |")
    lines.append("|------|-----|-----------|-------------|")

    total_added = 0
    total_removed = 0
    violations = []

    for f in files:
        total_added += f.added
        total_removed += f.removed
        ownership = classify_ownership(f.path, lane)
        if "VIOLATION" in ownership:
            violations.append(f.path)

        key_changes = []
        if f.functions_changed:
            key_changes.append(f"fn: {', '.join(f.functions_changed[:5])}")
        if f.classes_changed:
            key_changes.append(f"cls: {', '.join(f.classes_changed[:3])}")
        if f.imports_added:
            key_changes.append(f"+{len(f.imports_added)} imports")
        changes_str = "; ".join(key_changes) if key_changes else "minor changes"

        lines.append(f"| `{f.path}` | +{f.added}/-{f.removed} | {ownership} | {changes_str} |")

    lines.append("")
    lines.append(f"**Total:** +{total_added}/-{total_removed} across {len(files)} files")
    lines.append("")

    # Ownership violations (critical)
    if violations:
        lines.append("## OWNERSHIP VIOLATIONS")
        lines.append("")
        for v in violations:
            lines.append(f"- `{v}`")
        lines.append("")

    # Import changes (often the most important signal)
    all_imports_added = []
    for f in files:
        for imp in f.imports_added:
            all_imports_added.append(f"  {f.path}: `{imp}`")

    if all_imports_added:
        lines.append("## New Imports")
        lines.append("")
        for imp in all_imports_added:
            lines.append(f"- {imp}")
        lines.append("")

    # Functions/classes changed
    all_funcs = []
    for f in files:
        for func in f.functions_changed:
            all_funcs.append(f"`{f.path}:{func}()`")
        for cls in f.classes_changed:
            all_funcs.append(f"`{f.path}:{cls}`")

    if all_funcs:
        lines.append("## Functions/Classes Modified")
        lines.append("")
        for item in all_funcs:
            lines.append(f"- {item}")
        lines.append("")

    # Contracts context
    if contracts_text:
        lines.append("## Contract Definitions (reference)")
        lines.append("")
        lines.append("```python")
        lines.append(contracts_text.strip())
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Pre-process git diffs for efficient agent review")
    parser.add_argument("diff_file", nargs="?", help="Diff file path (reads stdin if omitted)")
    parser.add_argument("--lane", choices=["a", "b", "c"], help="Lane to check ownership against")
    parser.add_argument("--contracts-file", help="Path to contracts/pipeline.py for reference")
    parser.add_argument(
        "--raw-diff", action="store_true", help="Also append raw diff at end (for detailed review)"
    )
    args = parser.parse_args()

    # Read diff
    if args.diff_file:
        diff_text = Path(args.diff_file).read_text()
    elif not sys.stdin.isatty():
        diff_text = sys.stdin.read()
    else:
        print("Error: provide a diff file or pipe diff via stdin", file=sys.stderr)
        sys.exit(1)

    # Read contracts
    contracts_text = None
    if args.contracts_file:
        contracts_path = Path(args.contracts_file)
        if contracts_path.exists():
            contracts_text = contracts_path.read_text()

    # Parse and format
    files = parse_diff(diff_text)
    summary = format_summary(files, lane=args.lane, contracts_text=contracts_text)

    if args.raw_diff:
        summary += "\n## Raw Diff\n\n```diff\n" + diff_text + "\n```\n"

    print(summary)


if __name__ == "__main__":
    main()
