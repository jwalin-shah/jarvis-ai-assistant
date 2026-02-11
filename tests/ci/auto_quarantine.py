"""Automatically quarantine flaky tests based on historical data.

Usage:
    python -m tests.ci.auto_quarantine --threshold=0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tests.ci.flake_detector import FlakeDetector
from tests.ci.quarantine import QuarantineManager


def auto_quarantine(
    detector: FlakeDetector,
    quarantine: QuarantineManager,
    threshold: float = 0.5,
    min_runs: int = 10,
    lookback_days: int = 14,
    dry_run: bool = False,
) -> list[str]:
    """Automatically quarantine tests above threshold.

    Args:
        detector: FlakeDetector instance
        quarantine: QuarantineManager instance
        threshold: Flakiness threshold for quarantine (0.0-1.0)
        min_runs: Minimum runs before considering quarantine
        lookback_days: Days of history to analyze
        dry_run: If True, don't actually quarantine

    Returns:
        List of newly quarantined test IDs
    """
    flaky_tests = detector.get_flaky_tests(
        min_flakiness=threshold,
        lookback_days=lookback_days,
    )

    newly_quarantined = []

    for report in flaky_tests:
        # Skip if already quarantined
        if quarantine.is_quarantined(report.test_id):
            continue

        # Skip if not enough runs
        if report.total_runs < min_runs:
            continue

        # Quarantine tests with high flakiness or low pass rate
        pass_rate = report.pass_count / report.total_runs
        should_quarantine = report.flakiness_score >= threshold or pass_rate < 0.5

        if should_quarantine:
            if not dry_run:
                quarantine.add(
                    test_id=report.test_id,
                    reason=(
                        f"Auto-quarantined: flakiness_score={report.flakiness_score:.2f}, "
                        f"pass_rate={pass_rate:.1%}, "
                        f"runs={report.total_runs}"
                    ),
                    max_retries=3,
                    auto_unquarantine_days=14,
                    quarantined_by="auto-quarantine-script",
                    failure_pattern="high_flakiness",
                )
            newly_quarantined.append(report.test_id)

    return newly_quarantined


def unquarantine_stable_tests(
    detector: FlakeDetector,
    quarantine: QuarantineManager,
    threshold: float = 0.1,
    lookback_days: int = 7,
    dry_run: bool = False,
) -> list[str]:
    """Remove tests from quarantine that have become stable.

    Args:
        detector: FlakeDetector instance
        quarantine: QuarantineManager instance
        threshold: Maximum flakiness to consider stable
        lookback_days: Days of recent history to check
        dry_run: If True, don't actually remove

    Returns:
        List of unquarantined test IDs
    """
    quarantined = quarantine.get_quarantined_tests()
    unquarantined = []

    for entry in quarantined:
        # Skip if auto-unquarantine date not yet reached
        if entry.auto_unquarantine_date:
            from datetime import datetime

            if datetime.now() < entry.auto_unquarantine_date:
                continue

        # Check recent stability
        report = detector.analyze_test(entry.test_id, lookback_days)

        if report is None:
            # No recent data, keep quarantined
            continue

        if report.flakiness_score <= threshold:
            if not dry_run:
                quarantine.remove(entry.test_id)
            unquarantined.append(entry.test_id)

    return unquarantined


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Automatically manage test quarantine")
    parser.add_argument(
        "--flake-db",
        type=Path,
        default=Path(".flake_history.db"),
        help="Path to flake history database",
    )
    parser.add_argument(
        "--quarantine-file",
        type=Path,
        default=Path(".quarantine.json"),
        help="Path to quarantine file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Flakiness threshold for quarantine (0.0-1.0)",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=10,
        help="Minimum runs before considering quarantine",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=14,
        help="Days of history to analyze",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--unquarantine-stable",
        action="store_true",
        help="Also unquarantine tests that have become stable",
    )

    args = parser.parse_args()

    # Initialize components
    detector = FlakeDetector(args.flake_db)
    quarantine = QuarantineManager(args.quarantine_file)

    print(f"Analyzing test history (last {args.lookback_days} days)...")
    print(f"Quarantine threshold: {args.threshold}")
    print()

    # Auto-quarantine flaky tests
    newly_quarantined = auto_quarantine(
        detector=detector,
        quarantine=quarantine,
        threshold=args.threshold,
        min_runs=args.min_runs,
        lookback_days=args.lookback_days,
        dry_run=args.dry_run,
    )

    print(f"Newly quarantined tests: {len(newly_quarantined)}")
    for test_id in newly_quarantined:
        print(f"  - {test_id}")

    print()

    # Optionally unquarantine stable tests
    if args.unquarantine_stable:
        unquarantined = unquarantine_stable_tests(
            detector=detector,
            quarantine=quarantine,
            threshold=0.1,
            lookback_days=7,
            dry_run=args.dry_run,
        )
        print(f"Unquarantined stable tests: {len(unquarantined)}")
        for test_id in unquarantined:
            print(f"  - {test_id}")
        print()

    # Print summary
    summary = quarantine.get_summary()
    print("Quarantine Summary:")
    print(f"  Total quarantined: {summary['total_quarantined']}")
    print(f"  Expiring within 7 days: {summary['expiring_within_7_days']}")
    print(f"  With tickets: {summary['with_tickets']}")

    if args.dry_run:
        print("\n(Dry run - no changes made)")

    return 0 if not newly_quarantined else 1


if __name__ == "__main__":
    sys.exit(main())
