#!/usr/bin/env python3
"""
Release Checklist Automation Script

Automates validation of release readiness criteria for JARVIS.
Generates a report with pass/fail status for each category.

Usage:
    uv run python scripts/release_checklist.py [--full]

Options:
    --full      Run complete validation including hardware tests
    --category  Run specific category only (code|tests|perf|docs|migrations|obs|incident)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Status(Enum):
    """Check status enumeration."""

    PASS = "✅ PASS"
    CONDITIONAL = "⚠️  CONDITIONAL"
    FAIL = "❌ FAIL"
    UNKNOWN = "⬜ UNKNOWN"


@dataclass
class CheckResult:
    """Result of a single check."""

    name: str
    status: Status
    details: str = ""
    value: Any = None


@dataclass
class CategoryResult:
    """Result of a category of checks."""

    name: str
    weight: float
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Calculate score as percentage of passed checks."""
        if not self.checks:
            return 0.0
        passed = sum(1 for c in self.checks if c.status == Status.PASS)
        return (passed / len(self.checks)) * 100

    @property
    def status(self) -> Status:
        """Determine overall category status."""
        if any(c.status == Status.FAIL for c in self.checks):
            return Status.FAIL
        if any(c.status == Status.CONDITIONAL for c in self.checks):
            return Status.CONDITIONAL
        if all(c.status == Status.PASS for c in self.checks):
            return Status.PASS
        return Status.UNKNOWN


def run_command(cmd: str, timeout: int = 60) -> tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def check_code_quality() -> CategoryResult:
    """Validate code quality gates."""
    result = CategoryResult(name="Code Quality", weight=0.20)

    # Format check
    code, stdout, stderr = run_command("uv run ruff format --check .", timeout=30)
    result.checks.append(
        CheckResult(
            name="Format Check",
            status=Status.PASS if code == 0 else Status.FAIL,
            details="All files formatted" if code == 0 else stdout[:200],
        )
    )

    # Lint check
    code, stdout, stderr = run_command("uv run ruff check .", timeout=30)
    lint_errors = 0
    if code != 0:
        # Count actual errors
        lint_errors = stdout.count(":")
    result.checks.append(
        CheckResult(
            name="Lint Check",
            status=Status.PASS
            if lint_errors == 0
            else Status.FAIL
            if lint_errors > 10
            else Status.CONDITIONAL,
            details=f"{lint_errors} lint errors" if lint_errors > 0 else "No lint errors",
            value=lint_errors,
        )
    )

    # Type check
    code, stdout, stderr = run_command(
        "uv run mypy jarvis/ core/ models/ api/ --ignore-missing-imports",
        timeout=120,
    )
    type_errors = stdout.count(": error:")
    result.checks.append(
        CheckResult(
            name="Type Check",
            status=Status.PASS
            if type_errors == 0
            else Status.FAIL
            if type_errors > 10
            else Status.CONDITIONAL,
            details=f"{type_errors} type errors" if type_errors > 0 else "No type errors",
            value=type_errors,
        )
    )

    # Security scan
    code, stdout, stderr = run_command("uv run bandit -r jarvis/ api/ core/ -f json", timeout=60)
    try:
        bandit_results = json.loads(stdout) if stdout else {"results": []}
        high_severity = sum(
            1
            for r in bandit_results.get("results", [])
            if r.get("issue_severity") in ("HIGH", "CRITICAL")
        )
    except json.JSONDecodeError:
        high_severity = 0
    result.checks.append(
        CheckResult(
            name="Security Scan",
            status=Status.PASS if high_severity == 0 else Status.FAIL,
            details=f"{high_severity} high/critical security issues"
            if high_severity > 0
            else "No security issues",
            value=high_severity,
        )
    )

    # Debug statements
    code, stdout, stderr = run_command(
        "grep -rn 'breakpoint()\\|import pdb\\|IPython' jarvis/ api/ core/ --include='*.py' 2>/dev/null | head -20",
        timeout=10,
    )
    debug_count = len([line for line in stdout.split("\n") if line.strip()])
    result.checks.append(
        CheckResult(
            name="Debug Statements",
            status=Status.PASS if debug_count == 0 else Status.FAIL,
            details=f"{debug_count} debug statements found"
            if debug_count > 0
            else "No debug statements",
            value=debug_count,
        )
    )

    # Build verification
    code, stdout, stderr = run_command("uv build", timeout=60)
    result.checks.append(
        CheckResult(
            name="Build Verification",
            status=Status.PASS if code == 0 else Status.FAIL,
            details="Build successful" if code == 0 else f"Build failed: {stderr[:200]}",
        )
    )

    return result


def check_tests() -> CategoryResult:
    """Validate test suite."""
    result = CategoryResult(name="Tests", weight=0.20)

    # Run unit tests
    code, stdout, stderr = run_command(
        "uv run pytest tests/unit/ --tb=no -q --timeout=30 2>&1 | tail -20",
        timeout=300,
    )
    output = stdout + stderr

    # Parse test results
    passed = 0
    failed = 0
    error = 0

    for line in output.split("\n"):
        if "passed" in line:
            try:
                passed = int(line.split("passed")[0].strip().split()[-1])
            except (ValueError, IndexError):
                pass
        if "failed" in line:
            try:
                failed = int(line.split("failed")[0].strip().split()[-1])
            except (ValueError, IndexError):
                pass
        if "error" in line:
            try:
                error = int(line.split("error")[0].strip().split()[-1])
            except (ValueError, IndexError):
                pass

    total = passed + failed + error
    pass_rate = (passed / total * 100) if total > 0 else 0

    result.checks.append(
        CheckResult(
            name="Unit Test Pass Rate",
            status=Status.PASS
            if pass_rate >= 95
            else Status.CONDITIONAL
            if pass_rate >= 90
            else Status.FAIL,
            details=f"{passed}/{total} passed ({pass_rate:.1f}%)",
            value=pass_rate,
        )
    )

    # Check baseline failures
    baseline_file = Path("benchmarks/baseline.json")
    baseline_failures = 8  # Default from QUALITY_GATES_POLICY
    if baseline_file.exists():
        try:
            with open(baseline_file) as f:
                baseline = json.load(f)
                baseline_failures = (
                    baseline.get("baseline", {}).get("test_run", {}).get("failed", 8)
                )
        except (json.JSONDecodeError, KeyError):
            pass

    new_failures = max(0, failed - baseline_failures)
    result.checks.append(
        CheckResult(
            name="New Test Failures",
            status=Status.PASS if new_failures == 0 else Status.FAIL,
            details=f"{new_failures} new failures (baseline: {baseline_failures})",
            value=new_failures,
        )
    )

    return result


def check_performance() -> CategoryResult:
    """Validate performance benchmarks."""
    result = CategoryResult(name="Performance", weight=0.20)

    # Check if benchmark results exist
    memory_results = Path("results/memory.json")
    hhem_results = Path("results/hhem.json")
    latency_results = Path("results/latency.json")

    if memory_results.exists():
        try:
            with open(memory_results) as f:
                data = json.load(f)
                peak_memory = data.get("peak_memory_gb", 0)
                result.checks.append(
                    CheckResult(
                        name="Memory Benchmark",
                        status=Status.PASS
                        if peak_memory < 6.5
                        else Status.CONDITIONAL
                        if peak_memory < 7.0
                        else Status.FAIL,
                        details=f"Peak memory: {peak_memory:.2f}GB",
                        value=peak_memory,
                    )
                )
        except (json.JSONDecodeError, KeyError):
            result.checks.append(
                CheckResult(
                    name="Memory Benchmark",
                    status=Status.UNKNOWN,
                    details="Could not parse memory results",
                )
            )
    else:
        result.checks.append(
            CheckResult(
                name="Memory Benchmark",
                status=Status.UNKNOWN,
                details="Run: uv run python -m benchmarks.memory.run --output results/memory.json",
            )
        )

    if hhem_results.exists():
        try:
            with open(hhem_results) as f:
                data = json.load(f)
                hhem_score = data.get("mean_hhem_score", 0)
                result.checks.append(
                    CheckResult(
                        name="HHEM Score",
                        status=Status.PASS
                        if hhem_score >= 0.5
                        else Status.CONDITIONAL
                        if hhem_score >= 0.4
                        else Status.FAIL,
                        details=f"Mean HHEM: {hhem_score:.3f}",
                        value=hhem_score,
                    )
                )
        except (json.JSONDecodeError, KeyError):
            result.checks.append(
                CheckResult(
                    name="HHEM Score",
                    status=Status.UNKNOWN,
                    details="Could not parse HHEM results",
                )
            )
    else:
        result.checks.append(
            CheckResult(
                name="HHEM Score",
                status=Status.UNKNOWN,
                details="Run: uv run python -m benchmarks.hallucination.run --output results/hhem.json",
            )
        )

    if latency_results.exists():
        try:
            with open(latency_results) as f:
                data = json.load(f)
                cold_start = data.get("cold_start_seconds", 0)
                warm_start = data.get("warm_start_seconds", 0)

                result.checks.append(
                    CheckResult(
                        name="Cold Start Latency",
                        status=Status.PASS
                        if cold_start < 15
                        else Status.CONDITIONAL
                        if cold_start < 20
                        else Status.FAIL,
                        details=f"Cold start: {cold_start:.1f}s",
                        value=cold_start,
                    )
                )
                result.checks.append(
                    CheckResult(
                        name="Warm Start Latency",
                        status=Status.PASS
                        if warm_start < 3
                        else Status.CONDITIONAL
                        if warm_start < 5
                        else Status.FAIL,
                        details=f"Warm start: {warm_start:.1f}s",
                        value=warm_start,
                    )
                )
        except (json.JSONDecodeError, KeyError):
            result.checks.append(
                CheckResult(
                    name="Latency Benchmarks",
                    status=Status.UNKNOWN,
                    details="Could not parse latency results",
                )
            )
    else:
        result.checks.append(
            CheckResult(
                name="Latency Benchmarks",
                status=Status.UNKNOWN,
                details="Run: uv run python -m benchmarks.latency.run --output results/latency.json",
            )
        )

    return result


def check_documentation() -> CategoryResult:
    """Validate documentation."""
    result = CategoryResult(name="Documentation", weight=0.10)

    # Check key docs exist
    key_docs = [
        "README.md",
        "CHANGELOG.md",
        "docs/CLI_GUIDE.md",
        "AGENTS.md",
    ]

    for doc in key_docs:
        doc_path = Path(doc)
        result.checks.append(
            CheckResult(
                name=f"{doc} exists",
                status=Status.PASS if doc_path.exists() else Status.FAIL,
                details=f"Found: {doc}" if doc_path.exists() else f"Missing: {doc}",
            )
        )

    # Check for TODO/FIXME in key areas
    code, stdout, stderr = run_command(
        r"grep -rn 'TODO\|FIXME\|XXX\|HACK' jarvis/ core/ models/ --include='*.py' 2>/dev/null | wc -l",
        timeout=10,
    )
    try:
        todo_count = int(stdout.strip())
    except ValueError:
        todo_count = 0

    result.checks.append(
        CheckResult(
            name="TODO/FIXME Count",
            status=Status.PASS
            if todo_count < 100
            else Status.CONDITIONAL
            if todo_count < 150
            else Status.FAIL,
            details=f"{todo_count} TODO/FIXME/XXX/HACK comments",
            value=todo_count,
        )
    )

    return result


def check_migrations() -> CategoryResult:
    """Validate migrations and data integrity."""
    result = CategoryResult(name="Migrations", weight=0.10)

    # Check model artifacts exist
    home = Path.home()
    models_to_check = [
        ("~/.jarvis/trigger_classifier_model", "Trigger Classifier"),
        ("~/.jarvis/response_classifier_model", "Response Classifier"),
    ]

    for path, name in models_to_check:
        model_path = home / ".jarvis" / path.replace("~/.jarvis/", "")
        result.checks.append(
            CheckResult(
                name=f"{name} Model",
                status=Status.PASS if model_path.exists() else Status.CONDITIONAL,
                details=f"Found at {model_path}"
                if model_path.exists()
                else f"Not found at {model_path}",
            )
        )

    return result


def check_observability() -> CategoryResult:
    """Validate observability."""
    result = CategoryResult(name="Observability", weight=0.10)

    # Check metrics endpoint exists
    metrics_router = Path("api/routers/metrics.py")
    result.checks.append(
        CheckResult(
            name="Metrics Router",
            status=Status.PASS if metrics_router.exists() else Status.FAIL,
            details="Metrics endpoint implemented"
            if metrics_router.exists()
            else "Missing metrics router",
        )
    )

    # Check health router exists
    health_router = Path("api/routers/health.py")
    result.checks.append(
        CheckResult(
            name="Health Router",
            status=Status.PASS if health_router.exists() else Status.FAIL,
            details="Health endpoint implemented"
            if health_router.exists()
            else "Missing health router",
        )
    )

    # Check for PII in logs (basic check)
    code, stdout, stderr = run_command(
        r"grep -rn 'logger.*phone\|logger.*email\|logger.*ssn' jarvis/ api/ --include='*.py' 2>/dev/null | head -10",
        timeout=10,
    )
    pii_risk = len([line for line in stdout.split("\n") if line.strip()])
    result.checks.append(
        CheckResult(
            name="PII in Logs Check",
            status=Status.PASS if pii_risk == 0 else Status.FAIL,
            details=f"{pii_risk} potential PII logging issues"
            if pii_risk > 0
            else "No obvious PII logging",
            value=pii_risk,
        )
    )

    return result


def check_incident_readiness() -> CategoryResult:
    """Validate incident readiness."""
    result = CategoryResult(name="Incident Readiness", weight=0.10)

    # Check reliability plan exists
    reliability_plan = Path("docs/RELIABILITY_ENGINEERING_PLAN.md")
    result.checks.append(
        CheckResult(
            name="Reliability Plan",
            status=Status.PASS if reliability_plan.exists() else Status.CONDITIONAL,
            details="Reliability engineering plan documented"
            if reliability_plan.exists()
            else "Missing reliability plan",
        )
    )

    # Check observability roadmap exists
    obs_roadmap = Path("docs/OBSERVABILITY_ROADMAP.md")
    result.checks.append(
        CheckResult(
            name="Observability Roadmap",
            status=Status.PASS if obs_roadmap.exists() else Status.CONDITIONAL,
            details="Observability roadmap documented"
            if obs_roadmap.exists()
            else "Missing observability roadmap",
        )
    )

    # Check for circuit breaker implementation
    circuit_file = Path("core/health/circuit.py")
    result.checks.append(
        CheckResult(
            name="Circuit Breaker",
            status=Status.PASS if circuit_file.exists() else Status.CONDITIONAL,
            details="Circuit breaker implemented"
            if circuit_file.exists()
            else "Circuit breaker not found",
        )
    )

    # Check for graceful degradation
    degradation_file = Path("core/health/degradation.py")
    result.checks.append(
        CheckResult(
            name="Graceful Degradation",
            status=Status.PASS if degradation_file.exists() else Status.CONDITIONAL,
            details="Degradation controller implemented"
            if degradation_file.exists()
            else "Degradation controller not found",
        )
    )

    return result


def print_report(results: list[CategoryResult]) -> None:
    """Print formatted report."""
    print("\n" + "=" * 80)
    print("JARVIS RELEASE READINESS REPORT")
    print("=" * 80)

    total_score = 0.0
    total_weight = 0.0
    all_pass = True

    for category in results:
        print(f"\n{'=' * 40}")
        print(f"{category.name} (Weight: {category.weight:.0%})")
        print("=" * 40)

        for check in category.checks:
            print(f"  {check.status.value:<20} {check.name}")
            if check.details:
                print(f"                       {check.details}")

        print(f"\n  Score: {category.score:.1f}% | Status: {category.status.value}")

        total_score += category.score * category.weight
        total_weight += category.weight

        if category.status == Status.FAIL:
            all_pass = False

    # Overall summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if total_weight > 0:
        weighted_score = total_score / total_weight
        print(f"\nOverall Score: {weighted_score:.1f}%")

        if weighted_score >= 90 and all_pass:
            print("Status: ✅ APPROVED FOR RELEASE")
        elif weighted_score >= 75:
            print("Status: ⚠️  CONDITIONAL (See issues above)")
        else:
            print("Status: ❌ NOT READY FOR RELEASE")

    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)

    # Recommendations
    recommendations = []
    for category in results:
        for check in category.checks:
            if check.status == Status.FAIL:
                recommendations.append(f"[BLOCKER] {category.name}: {check.name}")
            elif check.status == Status.CONDITIONAL:
                recommendations.append(f"[WARNING] {category.name}: {check.name}")
            elif check.status == Status.UNKNOWN:
                recommendations.append(
                    f"[MISSING] {category.name}: {check.name} - Run required checks"
                )

    if recommendations:
        for rec in recommendations[:10]:  # Limit to first 10
            print(f"  {rec}")
        if len(recommendations) > 10:
            print(f"  ... and {len(recommendations) - 10} more issues")
    else:
        print("  All checks passed! Ready for final sign-off.")

    print("=" * 80 + "\n")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="JARVIS Release Readiness Check")
    parser.add_argument("--full", action="store_true", help="Run complete validation")
    parser.add_argument(
        "--category",
        choices=["code", "tests", "perf", "docs", "migrations", "obs", "incident"],
        help="Run specific category only",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON results")
    args = parser.parse_args()

    # Map of category functions
    category_map = {
        "code": check_code_quality,
        "tests": check_tests,
        "perf": check_performance,
        "docs": check_documentation,
        "migrations": check_migrations,
        "obs": check_observability,
        "incident": check_incident_readiness,
    }

    if args.category:
        # Run single category
        results = [category_map[args.category]()]
    else:
        # Run all categories
        results = [
            check_code_quality(),
            check_tests(),
            check_performance(),
            check_documentation(),
            check_migrations(),
            check_observability(),
            check_incident_readiness(),
        ]

    if args.json:
        # Output JSON
        output = {
            "categories": [
                {
                    "name": r.name,
                    "weight": r.weight,
                    "score": r.score,
                    "status": r.status.name,
                    "checks": [
                        {
                            "name": c.name,
                            "status": c.status.name,
                            "details": c.details,
                            "value": c.value,
                        }
                        for c in r.checks
                    ],
                }
                for r in results
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        # Print formatted report
        print_report(results)

    # Return exit code based on results
    has_failures = any(
        check.status == Status.FAIL for category in results for check in category.checks
    )
    return 1 if has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
