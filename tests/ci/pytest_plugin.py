"""Pytest plugin for test quarantine and flake detection.

Integrates quarantine management and flake detection directly into pytest.

Usage:
    pytest --quarantine-file=.quarantine.json --flake-db=.flake_history.db
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command-line options for quarantine."""
    group = parser.getgroup("quarantine", "Test quarantine and flake detection")

    group.addoption(
        "--quarantine-file",
        action="store",
        default=".quarantine.json",
        help="Path to quarantine file",
    )
    group.addoption(
        "--flake-db",
        action="store",
        default=".flake_history.db",
        help="Path to flake history database",
    )
    group.addoption(
        "--run-quarantined-only",
        action="store_true",
        default=False,
        help="Only run quarantined tests",
    )
    group.addoption(
        "--skip-quarantined",
        action="store_true",
        default=False,
        help="Skip quarantined tests entirely",
    )
    group.addoption(
        "--flake-detection",
        action="store_true",
        default=False,
        help="Enable flake detection and recording",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Configure quarantine plugin."""
    if config.getoption("--flake-detection"):
        from tests.ci.flake_detector import FlakeDetector

        db_path = Path(config.getoption("--flake-db"))
        config._flake_detector = FlakeDetector(db_path)
    else:
        config._flake_detector = None

    # Load quarantine manager
    from tests.ci.quarantine import QuarantineManager

    quarantine_path = Path(config.getoption("--quarantine-file"))
    config._quarantine_manager = QuarantineManager(quarantine_path)

    # Store CI info
    config._ci_run_id = os.environ.get("GITHUB_RUN_ID")
    config._branch = os.environ.get("GITHUB_REF")


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Modify test collection based on quarantine."""
    quarantine = config._quarantine_manager
    run_quarantined_only = config.getoption("--run-quarantined-only")
    skip_quarantined = config.getoption("--skip-quarantined")

    selected = []
    deselected = []

    for item in items:
        test_id = item.nodeid
        is_quarantined = quarantine.is_quarantined(test_id)

        if run_quarantined_only:
            # Only run quarantined tests
            if is_quarantined:
                selected.append(item)
            else:
                deselected.append(item)
        elif skip_quarantined and is_quarantined:
            # Skip quarantined tests entirely
            deselected.append(item)
        elif is_quarantined:
            # Mark quarantined tests with retries
            entry = quarantine.get_entry(test_id)
            if entry:
                # Add flaky marker with retries
                marker = pytest.mark.flaky(
                    reruns=entry.max_retries,
                    reruns_delay=1,
                )
                item.add_marker(marker)

                # Add quarantine marker for reporting
                item.add_marker(pytest.mark.quarantined)
            selected.append(item)
        else:
            selected.append(item)

    # Update items list
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Track test results for flakiness detection."""
    if report.when != "call":
        return

    config = report.config
    flake_detector = getattr(config, "_flake_detector", None)

    if flake_detector is None:
        return

    # Record the result
    flake_detector.record_from_pytest(
        nodeid=report.nodeid,
        outcome=report.outcome,
        duration=report.duration,
        ci_run_id=getattr(config, "_ci_run_id", None),
        branch=getattr(config, "_branch", None),
    )


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """Print quarantine summary."""
    quarantine = config._quarantine_manager

    quarantined_tests = quarantine.get_quarantined_tests()
    if quarantined_tests:
        terminalreporter.write_sep("=", "Quarantine Status")
        terminalreporter.write_line(f"Quarantined tests: {len(quarantined_tests)}")
        for entry in quarantined_tests[:10]:  # Show first 10
            terminalreporter.write_line(f"  - {entry.test_id}")
        if len(quarantined_tests) > 10:
            terminalreporter.write_line(f"  ... and {len(quarantined_tests) - 10} more")

    # Show flake detection summary
    flake_detector = getattr(config, "_flake_detector", None)
    if flake_detector:
        stats = flake_detector.get_summary_stats(lookback_days=7)
        terminalreporter.write_sep("=", "Flake Detection (7 days)")
        terminalreporter.write_line(f"Total test runs: {stats['total_runs']}")
        terminalreporter.write_line(f"Pass rate: {stats['pass_rate']:.1%}")
        terminalreporter.write_line(f"Flaky tests detected: {stats['flaky_tests']}")


@pytest.fixture
def quarantine_manager(pytestconfig: pytest.Config) -> Any:
    """Provide access to quarantine manager in tests."""
    return pytestconfig._quarantine_manager


@pytest.fixture
def flake_detector(pytestconfig: pytest.Config) -> Any:
    """Provide access to flake detector in tests."""
    return pytestconfig._flake_detector
