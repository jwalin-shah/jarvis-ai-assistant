"""Automated flaky test detection system.

Integrates with CI pipeline to:
1. Track test pass/fail history
2. Identify statistically flaky tests
3. Generate flake reports
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class TestResult:
    """Single test execution result."""

    test_id: str  # file::class::method
    timestamp: datetime
    passed: bool
    duration_ms: float
    error_message: str | None = None
    ci_run_id: str | None = None
    branch: str | None = None


@dataclass
class FlakeReport:
    """Report of flaky test detection."""

    test_id: str
    total_runs: int
    pass_count: int
    fail_count: int
    flakiness_score: float  # 0.0-1.0, 0.5 = maximally flaky
    last_failure: datetime | None
    recommended_action: str  # "quarantine", "monitor", "investigate"


class FlakeDetector:
    """Detect flaky tests from historical results."""

    # Thresholds for flakiness classification
    MIN_RUNS = 10  # Minimum runs before classification
    FLAKY_THRESHOLD = 0.2  # 20% failure rate = flaky
    QUARANTINE_THRESHOLD = 0.5  # 50% failure rate = quarantine

    def __init__(self, db_path: Path = Path(".flake_history.db")):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize result database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    passed BOOLEAN NOT NULL,
                    duration_ms REAL,
                    error_message TEXT,
                    ci_run_id TEXT,
                    branch TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_test_id
                ON test_results(test_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON test_results(timestamp)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ci_run
                ON test_results(ci_run_id)
                """
            )

    def record_result(self, result: TestResult) -> None:
        """Record a test result."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO test_results
                (test_id, timestamp, passed, duration_ms, error_message, ci_run_id, branch)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.test_id,
                    result.timestamp.isoformat(),
                    result.passed,
                    result.duration_ms,
                    result.error_message,
                    result.ci_run_id,
                    result.branch,
                ),
            )

    def record_from_pytest(
        self,
        nodeid: str,
        outcome: str,
        duration: float,
        ci_run_id: str | None = None,
        branch: str | None = None,
    ) -> None:
        """Record a result from pytest report."""
        self.record_result(
            TestResult(
                test_id=nodeid,
                timestamp=datetime.now(),
                passed=(outcome == "passed"),
                duration_ms=duration * 1000,
                ci_run_id=ci_run_id,
                branch=branch,
            )
        )

    def analyze_test(
        self,
        test_id: str,
        lookback_days: int = 14,
    ) -> FlakeReport | None:
        """Analyze test flakiness over time period."""
        cutoff = datetime.now() - timedelta(days=lookback_days)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT passed, timestamp FROM test_results
                WHERE test_id = ? AND timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 100
                """,
                (test_id, cutoff.isoformat()),
            )
            results = cursor.fetchall()

        if len(results) < self.MIN_RUNS:
            return None

        passes = sum(1 for r in results if r[0])
        failures = len(results) - passes

        # Calculate flakiness: closer to 0.5 = more flaky
        pass_rate = passes / len(results) if results else 0
        # Score: 0.5 pass rate = 1.0 flakiness, 0.0 or 1.0 = 0.0 flakiness
        flakiness = 1.0 - abs(pass_rate - 0.5) * 2

        # Find last failure
        last_failure = None
        for passed, timestamp in results:
            if not passed:
                last_failure = datetime.fromisoformat(timestamp)
                break

        # Determine recommended action
        if pass_rate < self.QUARANTINE_THRESHOLD:
            action = "quarantine"
        elif flakiness > self.FLAKY_THRESHOLD:
            action = "investigate"
        else:
            action = "monitor"

        return FlakeReport(
            test_id=test_id,
            total_runs=len(results),
            pass_count=passes,
            fail_count=failures,
            flakiness_score=flakiness,
            last_failure=last_failure,
            recommended_action=action,
        )

    def get_all_test_ids(
        self,
        lookback_days: int = 14,
    ) -> list[str]:
        """Get all test IDs seen in recent history."""
        cutoff = datetime.now() - timedelta(days=lookback_days)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT DISTINCT test_id FROM test_results
                WHERE timestamp > ?
                """,
                (cutoff.isoformat(),),
            )
            return [row[0] for row in cursor.fetchall()]

    def get_quarantine_list(self) -> list[str]:
        """Get list of test IDs that should be quarantined."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT test_id,
                       COUNT(*) as runs,
                       SUM(CASE WHEN passed THEN 1 ELSE 0 END) as passes
                FROM test_results
                WHERE timestamp > datetime('now', '-14 days')
                GROUP BY test_id
                HAVING runs >= ?
                """,
                (self.MIN_RUNS,),
            )

            quarantine = []
            for test_id, runs, passes in cursor.fetchall():
                pass_rate = (passes or 0) / runs
                if pass_rate < self.QUARANTINE_THRESHOLD:
                    quarantine.append(test_id)

            return quarantine

    def get_flaky_tests(
        self,
        min_flakiness: float = 0.2,
        lookback_days: int = 14,
    ) -> list[FlakeReport]:
        """Get all flaky tests sorted by flakiness score."""
        test_ids = self.get_all_test_ids(lookback_days)

        reports = []
        for test_id in test_ids:
            report = self.analyze_test(test_id, lookback_days)
            if report and report.flakiness_score >= min_flakiness:
                reports.append(report)

        # Sort by flakiness (most flaky first)
        reports.sort(key=lambda r: r.flakiness_score, reverse=True)
        return reports

    def get_test_history(
        self,
        test_id: str,
        lookback_days: int = 30,
    ) -> list[tuple[datetime, bool, float]]:
        """Get full history for a specific test."""
        cutoff = datetime.now() - timedelta(days=lookback_days)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT timestamp, passed, duration_ms
                FROM test_results
                WHERE test_id = ? AND timestamp > ?
                ORDER BY timestamp DESC
                """,
                (test_id, cutoff.isoformat()),
            )
            return [
                (datetime.fromisoformat(ts), passed, duration)
                for ts, passed, duration in cursor.fetchall()
            ]

    def get_summary_stats(
        self,
        lookback_days: int = 7,
    ) -> dict:
        """Get summary statistics for the test suite."""
        cutoff = datetime.now() - timedelta(days=lookback_days)

        with sqlite3.connect(self.db_path) as conn:
            # Total runs
            cursor = conn.execute(
                """
                SELECT COUNT(*), SUM(CASE WHEN passed THEN 1 ELSE 0 END)
                FROM test_results
                WHERE timestamp > ?
                """,
                (cutoff.isoformat(),),
            )
            total, passed = cursor.fetchone()

            # Unique tests
            cursor = conn.execute(
                """
                SELECT COUNT(DISTINCT test_id)
                FROM test_results
                WHERE timestamp > ?
                """,
                (cutoff.isoformat(),),
            )
            unique_tests = cursor.fetchone()[0]

            # Failed tests
            cursor = conn.execute(
                """
                SELECT COUNT(DISTINCT test_id)
                FROM test_results
                WHERE timestamp > ? AND passed = 0
                """,
                (cutoff.isoformat(),),
            )
            failed_tests = cursor.fetchone()[0]

        return {
            "total_runs": total or 0,
            "passed_runs": passed or 0,
            "failed_runs": (total or 0) - (passed or 0),
            "pass_rate": (passed / total) if total else 0,
            "unique_tests": unique_tests or 0,
            "tests_with_failures": failed_tests or 0,
            "flaky_tests": len(self.get_flaky_tests(lookback_days=lookback_days)),
        }

    def cleanup_old_results(self, keep_days: int = 90) -> int:
        """Remove old test results. Returns number of rows deleted."""
        cutoff = datetime.now() - timedelta(days=keep_days)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM test_results WHERE timestamp < ?",
                (cutoff.isoformat(),),
            )
            return cursor.rowcount
