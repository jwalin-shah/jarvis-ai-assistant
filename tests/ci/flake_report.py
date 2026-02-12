"""Generate flake analysis reports.

Provides HTML and JSON reports for flaky test analysis.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def generate_html_report(
    detector: Any,
    output_path: Path,
    lookback_days: int = 7,
) -> None:
    """Generate HTML flake report."""
    # Get flaky tests
    flaky_tests = detector.get_flaky_tests(
        min_flakiness=0.1,
        lookback_days=lookback_days,
    )

    # Get summary stats
    stats = detector.get_summary_stats(lookback_days)

    # Sort by flakiness
    flaky_tests.sort(key=lambda r: r.flakiness_score, reverse=True)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>JARVIS Flake Report - {datetime.now().strftime("%Y-%m-%d")}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 40px;
            background: #f5f5f7;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1d1d1f;
            border-bottom: 2px solid #007aff;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #1d1d1f;
            margin-top: 30px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f5f5f7;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #007aff;
        }}
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        th {{
            background: #f5f5f7;
            font-weight: 600;
            color: #1d1d1f;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .quarantine {{
            background: #fff2f2;
        }}
        .investigate {{
            background: #fffbf0;
        }}
        .monitor {{
            background: #f0fff4;
        }}
        .score {{
            font-weight: bold;
            padding: 4px 8px;
            border-radius: 4px;
        }}
        .score-high {{
            background: #ff4444;
            color: white;
        }}
        .score-medium {{
            background: #ffaa00;
            color: white;
        }}
        .score-low {{
            background: #00aa44;
            color: white;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .badge-quarantine {{
            background: #ff4444;
            color: white;
        }}
        .badge-investigate {{
            background: #ffaa00;
            color: white;
        }}
        .badge-monitor {{
            background: #00aa44;
            color: white;
        }}
        .test-id {{
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.9em;
            color: #666;
        }}
        .timestamp {{
            color: #999;
            font-size: 0.9em;
        }}
        .no-flakes {{
            text-align: center;
            padding: 40px;
            color: #00aa44;
        }}
        .no-flakes h3 {{
            font-size: 1.5em;
            margin: 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 JARVIS Flaky Test Report</h1>
        <p class="timestamp">
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")} |
            Period: Last {lookback_days} days
        </p>

        <h2>📊 Summary</h2>
        <div class="summary">
            <div class="stat-card">
                <div class="stat-value">{stats["total_runs"]:,}</div>
                <div class="stat-label">Total Test Runs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats["pass_rate"]:.1%}</div>
                <div class="stat-label">Pass Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats["unique_tests"]}</div>
                <div class="stat-label">Unique Tests</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats["flaky_tests"]}</div>
                <div class="stat-label">Flaky Tests</div>
            </div>
        </div>
"""

    if not flaky_tests:
        html += """
        <div class="no-flakes">
            <h3>✅ No flaky tests detected!</h3>
            <p>All tests are passing consistently. Great job!</p>
        </div>
"""
    else:
        html += f"""
        <h2>⚠️ Flaky Tests ({len(flaky_tests)} detected)</h2>
        <table>
            <thead>
                <tr>
                    <th>Test ID</th>
                    <th>Runs</th>
                    <th>Pass Rate</th>
                    <th>Flakiness Score</th>
                    <th>Last Failure</th>
                    <th>Action</th>
                </tr>
            </thead>
            <tbody>
"""
        for report in flaky_tests:
            css_class = report.recommended_action
            score_class = (
                "score-high"
                if report.flakiness_score > 0.5
                else "score-medium"
                if report.flakiness_score > 0.3
                else "score-low"
            )
            last_failure = (
                report.last_failure.strftime("%Y-%m-%d %H:%M") if report.last_failure else "Never"
            )

            html += f"""
                <tr class="{css_class}">
                    <td class="test-id">{report.test_id}</td>
                    <td>{report.total_runs}</td>
                    <td>{report.pass_count / report.total_runs:.1%}</td>
                    <td>
                        <span class="score {score_class}">
                            {report.flakiness_score:.2f}
                        </span>
                    </td>
                    <td class="timestamp">{last_failure}</td>
                    <td>
                        <span class="badge badge-{report.recommended_action}">
                            {report.recommended_action.upper()}
                        </span>
                    </td>
                </tr>
"""

        html += """
            </tbody>
        </table>

        <h2>📋 Legend</h2>
        <ul>
            <li><span class="badge badge-quarantine">QUARANTINE</span> - Pass rate below 50%</li>
            <li><span class="badge badge-investigate">INVESTIGATE</span> - Needs investigation</li>
            <li><span class="badge badge-monitor">MONITOR</span> - Acceptable flakiness</li>
        </ul>
"""

    html += """
    </div>
</body>
</html>
"""

    output_path.write_text(html)


def generate_json_report(
    detector: Any,
    output_path: Path,
    lookback_days: int = 7,
) -> None:
    """Generate JSON flake report."""
    flaky_tests = detector.get_flaky_tests(
        min_flakiness=0.1,
        lookback_days=lookback_days,
    )
    stats = detector.get_summary_stats(lookback_days)

    data = {
        "generated_at": datetime.now().isoformat(),
        "period_days": lookback_days,
        "summary": stats,
        "flaky_tests": [
            {
                "test_id": r.test_id,
                "total_runs": r.total_runs,
                "pass_count": r.pass_count,
                "fail_count": r.fail_count,
                "pass_rate": r.pass_count / r.total_runs,
                "flakiness_score": r.flakiness_score,
                "last_failure": (r.last_failure.isoformat() if r.last_failure else None),
                "recommended_action": r.recommended_action,
            }
            for r in flaky_tests
        ],
    }

    output_path.write_text(json.dumps(data, indent=2))


def generate_markdown_report(
    detector: Any,
    output_path: Path,
    lookback_days: int = 7,
) -> None:
    """Generate Markdown flake report."""
    flaky_tests = detector.get_flaky_tests(
        min_flakiness=0.1,
        lookback_days=lookback_days,
    )
    stats = detector.get_summary_stats(lookback_days)

    lines = [
        "# 🔍 JARVIS Flaky Test Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Period:** Last {lookback_days} days",
        "",
        "## 📊 Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Test Runs | {stats['total_runs']:,} |",
        f"| Pass Rate | {stats['pass_rate']:.1%} |",
        f"| Unique Tests | {stats['unique_tests']} |",
        f"| Tests with Failures | {stats['tests_with_failures']} |",
        f"| Flaky Tests | {stats['flaky_tests']} |",
        "",
    ]

    if not flaky_tests:
        lines.extend(
            [
                "## ✅ Status",
                "",
                "No flaky tests detected! All tests are passing consistently.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"## ⚠️ Flaky Tests ({len(flaky_tests)} detected)",
                "",
                "| Test ID | Runs | Pass Rate | Flakiness | Action |",
                "|---------|------|-----------|-----------|--------|",
            ]
        )

        for report in flaky_tests:
            pass_rate = report.pass_count / report.total_runs
            lines.append(
                f"| `{report.test_id}` | {report.total_runs} | "
                f"{pass_rate:.1%} | {report.flakiness_score:.2f} | "
                f"{report.recommended_action.upper()} |"
            )

        lines.extend(
            [
                "",
                "### Legend",
                "",
                "- **QUARANTINE** - Pass rate below 50%, should be quarantined",
                "- **INVESTIGATE** - Flakiness above threshold, needs investigation",
                "- **MONITOR** - Acceptable flakiness, continue monitoring",
                "",
            ]
        )

    output_path.write_text("\n".join(lines))


def main() -> None:
    """CLI entry point for report generation."""
    import argparse

    from tests.ci.flake_detector import FlakeDetector

    parser = argparse.ArgumentParser(description="Generate flaky test reports")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(".flake_history.db"),
        help="Path to flake history database",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file path",
    )
    parser.add_argument(
        "--format",
        choices=["html", "json", "markdown"],
        default="html",
        help="Report format",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="Days of history to analyze",
    )

    args = parser.parse_args()

    detector = FlakeDetector(args.db)

    if args.format == "html":
        generate_html_report(detector, args.output, args.lookback_days)
    elif args.format == "json":
        generate_json_report(detector, args.output, args.lookback_days)
    else:
        generate_markdown_report(detector, args.output, args.lookback_days)

    print(f"Report generated: {args.output}")


if __name__ == "__main__":
    main()
