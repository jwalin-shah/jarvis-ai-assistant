#!/usr/bin/env python3
"""Phase eval runner for reply pipeline.

Modes:
- baseline: run eval_pipeline and capture/update baseline artifact
- candidate: run eval_pipeline and check against baseline via quality gate
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> int:
    print(f"[phase-eval] $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=PROJECT_ROOT, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run phase eval workflow")
    parser.add_argument(
        "mode",
        choices=["baseline", "candidate"],
        help="baseline: capture baseline; candidate: run regression gate",
    )
    parser.add_argument("--judge", action="store_true", help="Enable LLM judge in eval pipeline")
    parser.add_argument(
        "--source",
        default="results/eval_pipeline_baseline.json",
        help="Eval output path",
    )
    parser.add_argument(
        "--baseline",
        default="evals/baselines/baseline_20260221.json",
        help="Baseline artifact path",
    )
    args = parser.parse_args()

    eval_cmd = ["uv", "run", "python", "evals/eval_pipeline.py"]
    if args.judge:
        eval_cmd.append("--judge")

    rc = _run(eval_cmd)
    if rc != 0:
        return rc

    if args.mode == "baseline":
        return _run(
            [
                "uv",
                "run",
                "python",
                "evals/capture_baseline.py",
                "--source",
                args.source,
                "--output",
                args.baseline,
            ]
        )

    return _run(
        [
            "uv",
            "run",
            "python",
            "evals/quality_gate.py",
            "--baseline",
            args.baseline,
            "--candidate",
            args.source,
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
