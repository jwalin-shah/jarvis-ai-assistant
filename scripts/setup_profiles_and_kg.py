#!/usr/bin/env python3
"""Master script to set up contact profiles, group chats, and knowledge graph.

This orchestrates the full setup:
1. Analyzes and tags group chats
2. Builds contact profiles for all contacts
3. Verifies the knowledge graph

Usage:
    uv run python scripts/setup_profiles_and_kg.py
    uv run python scripts/setup_profiles_and_kg.py --skip-groups
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess


def run_script(name: str, args: list[str] | None = None) -> bool:
    """Run a script and return success status."""
    cmd = ["python", f"scripts/{name}.py"]
    if args:
        cmd.extend(args)

    print(f"\n{'=' * 60}")
    print(f"Running: {name}")
    print("=" * 60)

    result = subprocess.run(cmd, capture_output=False)  # noqa: S603
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Set up profiles and knowledge graph")
    parser.add_argument("--skip-groups", action="store_true", help="Skip group chat analysis")
    parser.add_argument("--skip-profiles", action="store_true", help="Skip profile building")
    parser.add_argument("--skip-verify", action="store_true", help="Skip KG verification")
    parser.add_argument("--min-messages", type=int, default=5, help="Min messages for profile")
    args = parser.parse_args()

    print("=" * 60)
    print("JARVIS: Profile & Knowledge Graph Setup")
    print("=" * 60)

    success = True

    # Step 1: Group Chat Analysis
    if not args.skip_groups:
        success = run_script("analyze_group_chats", ["--tag-groups"]) and success
    else:
        print("\n⏭️  Skipping group chat analysis")

    # Step 2: Build Profiles
    if not args.skip_profiles:
        success = run_script("build_all_profiles", [f"--min-messages={args.min_messages}"]) and success
    else:
        print("\n⏭️  Skipping profile building")

    # Step 3: Verify Knowledge Graph
    if not args.skip_verify:
        success = run_script("verify_knowledge_graph") and success
    else:
        print("\n⏭️  Skipping KG verification")

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("✅ Setup complete!")
    else:
        print("⚠️  Setup completed with warnings")
    print("=" * 60)

    print("\nNext steps:")
    print("  1. Start the API:      make api-dev")
    print("  2. View network graph: GET /graph/network")
    print("  3. View contact:       GET /graph/contact/{chat_id}")
    print("  4. Start the UI:       cd desktop && pnpm tauri dev")


if __name__ == "__main__":
    main()
