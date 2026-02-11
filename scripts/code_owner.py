#!/usr/bin/env python3
"""Code ownership lookup tool for JARVIS.

Usage:
    python scripts/code_owner.py <file_path>
    python scripts/code_owner.py jarvis/contacts/fact_extractor.py

    # Find reviewers for current changes
    git diff --name-only HEAD~1 | xargs -I {} python scripts/code_owner.py {}
"""

import sys
from pathlib import Path

# Module ownership matrix
# Order matters: more specific paths should come after general ones
OWNERSHIP_MAP: dict[str, list[str]] = {
    # Core infrastructure
    "jarvis/config.py": ["@backend-lead"],
    "jarvis/errors.py": ["@backend-lead"],
    "jarvis/db/": ["@data-lead", "@backend-lead"],
    "jarvis/contracts/": ["@architect", "@backend-lead"],
    "jarvis/utils/": ["@backend-lead"],
    # ML & Models
    "models/": ["@ml-lead", "@backend-lead"],
    "jarvis/features/": ["@ml-lead", "@data-lead"],
    # Data Pipeline
    "jarvis/contacts/": ["@data-lead", "@ml-lead"],
    "jarvis/graph/": ["@data-lead", "@ml-lead"],
    "jarvis/search/": ["@data-lead", "@backend-lead"],
    "jarvis/embedding_adapter.py": ["@data-lead", "@ml-lead"],
    # API & Integration
    "api/": ["@backend-lead", "@frontend-lead"],
    "jarvis/socket_server.py": ["@backend-lead", "@desktop-lead"],
    "jarvis/watcher.py": ["@backend-lead", "@desktop-lead"],
    "jarvis/scheduler/": ["@backend-lead"],
    "jarvis/prefetch/": ["@backend-lead"],
    # Desktop App
    "desktop/src-tauri/": ["@desktop-lead", "@backend-lead"],
    "desktop/src/": ["@frontend-lead", "@desktop-lead"],
    # Testing & Quality
    "tests/integration/": ["@qa-lead", "@backend-lead"],
    "tests/unit/": ["@qa-lead"],
    "benchmarks/": ["@ml-lead", "@qa-lead"],
    # Operations
    "scripts/": ["@devops", "@ml-lead"],
    "deploy/": ["@devops", "@backend-lead"],
    "docs/": ["@tech-writer"],
}

# Default owner for unmatched files
DEFAULT_OWNER = ["@backend-lead"]


def get_owners(file_path: str) -> list[str]:
    """Get owners for a file path.

    Args:
        file_path: Path to the file (relative to repo root)

    Returns:
        List of owner handles (e.g., ["@data-lead", "@ml-lead"])
    """
    path = Path(file_path)

    # Check exact match first
    if file_path in OWNERSHIP_MAP:
        return OWNERSHIP_MAP[file_path]

    # Check directory prefixes (longest match wins)
    best_match: str | None = None
    best_owners = DEFAULT_OWNER

    for prefix, owners in OWNERSHIP_MAP.items():
        if prefix.endswith("/"):
            # Directory prefix - check if file is inside this directory
            prefix_path = Path(prefix)
            try:
                path.relative_to(prefix_path)
                if best_match is None or len(prefix) > len(best_match):
                    best_match = prefix
                    best_owners = owners
            except ValueError:
                continue
        else:
            # File prefix - check if file path starts with this
            if file_path.startswith(prefix):
                if best_match is None or len(prefix) > len(best_match):
                    best_match = prefix
                    best_owners = owners

    return best_owners


def get_reviewers_for_changes(base_ref: str = "HEAD~1") -> dict[str, list[str]]:
    """Get suggested reviewers for changed files.

    Args:
        base_ref: Git ref to compare against (default: HEAD~1)

    Returns:
        Dict mapping file paths to their owners
    """
    import subprocess

    result = subprocess.run(
        ["git", "diff", "--name-only", base_ref],
        capture_output=True,
        text=True,
        check=True,
    )

    changed_files = [f for f in result.stdout.strip().split("\n") if f]

    reviewers: dict[str, list[str]] = {}
    for filepath in changed_files:
        reviewers[filepath] = get_owners(filepath)

    return reviewers


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nOwners for current changes:")
        try:
            reviewers = get_reviewers_for_changes()
            if not reviewers:
                print("  No changes detected.")
                return 0

            all_owners: set[str] = set()
            for filepath, owners in reviewers.items():
                owner_str = ", ".join(owners)
                print(f"  {filepath}: {owner_str}")
                all_owners.update(owners)

            print(f"\nSuggested reviewers: {', '.join(sorted(all_owners))}")
        except Exception as e:
            print(f"  Error: {e}")
            return 1
        return 0

    file_path = sys.argv[1]

    # Special flag for CI/review mode
    if file_path == "--review":
        reviewers = get_reviewers_for_changes()
        all_owners: set[str] = set()
        for owners in reviewers.values():
            all_owners.update(owners)
        print(" ".join(sorted(all_owners)))
        return 0

    owners = get_owners(file_path)
    print(f"Owners for {file_path}:")
    for owner in owners:
        print(f"  - {owner}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
