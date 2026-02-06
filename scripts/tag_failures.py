#!/usr/bin/env python3
"""Tag failure reasons for dismissed/ignored suggestions.

Usage:
    # Interactive tagging of recent failures
    uv run python scripts/tag_failures.py --tag

    # View capability gap analysis
    uv run python scripts/tag_failures.py --analyze

    # View recent feedback summary
    uv run python scripts/tag_failures.py --summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jarvis.eval.evaluation import (
    FailureReason,
    FeedbackAction,
    get_feedback_store,
)


def print_failure_reasons() -> None:
    """Print available failure reasons."""
    print("\nFailure Reasons:")
    print("-" * 50)

    print("\n[Capability Gaps] - Need new features:")
    print("  1. needs_calendar    - Scheduling, availability")
    print("  2. needs_memory      - 'Remember X', 'What did we...'")
    print("  3. needs_tasks       - Commitments, todos, reminders")
    print("  4. needs_contacts    - Contact info lookups")
    print("  5. needs_external    - Weather, news, web lookups")

    print("\n[Classifier/Generation Issues] - Can improve now:")
    print("  6. classifier_wrong  - Wrong trigger type detected")
    print("  7. tone_wrong        - Right content, wrong style")
    print("  8. too_generic       - Response too vague")
    print("  9. context_insufficient - Needed more history")

    print("\n  0. unknown           - Not sure")
    print("  s. skip              - Skip this entry")
    print("  q. quit              - Save and exit")


REASON_MAP = {
    "1": FailureReason.NEEDS_CALENDAR.value,
    "2": FailureReason.NEEDS_MEMORY.value,
    "3": FailureReason.NEEDS_TASK_TRACKING.value,
    "4": FailureReason.NEEDS_CONTACT_INFO.value,
    "5": FailureReason.NEEDS_EXTERNAL_INFO.value,
    "6": FailureReason.CLASSIFIER_WRONG.value,
    "7": FailureReason.TONE_WRONG.value,
    "8": FailureReason.TOO_GENERIC.value,
    "9": FailureReason.CONTEXT_INSUFFICIENT.value,
    "0": FailureReason.UNKNOWN.value,
}


def tag_failures(limit: int = 50) -> None:
    """Interactively tag failure reasons."""
    store = get_feedback_store()
    entries = store.get_recent_entries(limit=limit)

    # Filter to untagged failures
    failures = [
        e
        for e in entries
        if e.action in (FeedbackAction.DISMISSED, FeedbackAction.WROTE_FROM_SCRATCH)
        and not e.metadata.get("failure_reason")
    ]

    if not failures:
        print("No untagged failures found!")
        return

    print(f"\nFound {len(failures)} untagged failures to review")
    print_failure_reasons()

    # Load feedback file for updating
    feedback_file = Path.home() / ".jarvis" / "feedback.jsonl"
    if not feedback_file.exists():
        print("No feedback file found!")
        return

    # Read all entries
    all_entries: list[dict] = []
    with open(feedback_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_entries.append(json.loads(line))

    updates = 0
    for i, entry in enumerate(failures):
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(failures)}] Action: {entry.action.value}")
        print(f"Chat: {entry.chat_id[:30]}...")
        print(f"\nSuggestion: {entry.suggestion_text}")
        if entry.edited_text:
            print(f"User wrote: {entry.edited_text}")
        print("-" * 40)

        choice = input("Reason [1-9, 0, s=skip, q=quit]: ").strip().lower()

        if choice == "q":
            break
        if choice == "s":
            continue
        if choice in REASON_MAP:
            reason = REASON_MAP[choice]
            # Find and update in all_entries
            for e in all_entries:
                if e.get("suggestion_id") == entry.suggestion_id:
                    if "metadata" not in e:
                        e["metadata"] = {}
                    e["metadata"]["failure_reason"] = reason
                    updates += 1
                    print(f"  Tagged: {reason}")
                    break

    # Write back
    if updates > 0:
        with open(feedback_file, "w", encoding="utf-8") as f:
            for e in all_entries:
                f.write(json.dumps(e) + "\n")
        print(f"\nSaved {updates} tags!")
    else:
        print("\nNo changes made.")


def show_analysis() -> None:
    """Show capability gap analysis."""
    store = get_feedback_store()
    gaps = store.get_capability_gaps()

    print("\n" + "=" * 60)
    print("CAPABILITY GAP ANALYSIS")
    print("=" * 60)

    print(f"\nTotal failures: {gaps['total_failures']}")
    print(f"Tagged: {gaps['tagged_failures']}")
    print(f"Untagged: {gaps['untagged_failures']}")

    if gaps["capability_gaps"]:
        print("\n[Capability Gaps] - Need new features:")
        for reason, count in sorted(gaps["capability_gaps"].items(), key=lambda x: -x[1]):
            pct = count / gaps["total_failures"] * 100
            bar = "#" * int(pct / 2)
            print(f"  {reason:25} {count:3} ({pct:5.1f}%) {bar}")
        print(f"  {'TOTAL':25} {gaps['capability_gap_total']:3}")

    if gaps["classifier_issues"]:
        print("\n[Classifier Issues] - Can improve now:")
        for reason, count in sorted(gaps["classifier_issues"].items(), key=lambda x: -x[1]):
            pct = count / gaps["total_failures"] * 100
            bar = "#" * int(pct / 2)
            print(f"  {reason:25} {count:3} ({pct:5.1f}%) {bar}")
        print(f"  {'TOTAL':25} {gaps['classifier_issue_total']:3}")

    if gaps["recommendation"]:
        print(f"\nRecommendation: {gaps['recommendation']}")


def show_summary() -> None:
    """Show feedback summary."""
    store = get_feedback_store()
    stats = store.get_stats()

    print("\n" + "=" * 60)
    print("FEEDBACK SUMMARY")
    print("=" * 60)

    print(f"\nTotal feedback entries: {stats['total_feedback']}")
    print("\nBy action:")
    print(f"  Sent unchanged:    {stats['sent_unchanged']:4}")
    print(f"  Edited:            {stats['edited']:4}")
    print(f"  Dismissed:         {stats['dismissed']:4}")
    print(f"  Wrote from scratch:{stats.get('wrote_from_scratch', 0):4}")
    print(f"  Copied:            {stats['copied']:4}")

    print("\nRates:")
    print(f"  Acceptance rate: {stats['acceptance_rate'] * 100:.1f}%")
    print(f"  Edit rate:       {stats['edit_rate'] * 100:.1f}%")

    if stats.get("failure_reasons"):
        print("\nFailure reasons tagged:")
        for reason, count in sorted(stats["failure_reasons"].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag and analyze failure reasons")
    parser.add_argument("--tag", action="store_true", help="Interactively tag failures")
    parser.add_argument("--analyze", action="store_true", help="Show capability gap analysis")
    parser.add_argument("--summary", action="store_true", help="Show feedback summary")
    parser.add_argument("--limit", type=int, default=50, help="Max entries to review")

    args = parser.parse_args()

    if args.tag:
        tag_failures(args.limit)
    elif args.analyze:
        show_analysis()
    elif args.summary:
        show_summary()
    else:
        # Default: show summary then analyze
        show_summary()
        show_analysis()


if __name__ == "__main__":
    main()
