#!/usr/bin/env python3
"""Capture normalized Phase-0 baseline metrics from an eval output JSON."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _extract_metrics(payload: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("judge_avg", "anti_ai_clean_rate", "category_accuracy", "hallucination_rate"):
        val = payload.get(key)
        if isinstance(val, int | float):
            out[key] = float(val)

    latency = payload.get("latency", {})
    if isinstance(latency, dict):
        p95 = latency.get("p95_ms")
        if isinstance(p95, int | float):
            out["latency_p95_ms"] = float(p95)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture reply quality baseline")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("evals/results/eval_pipeline_baseline.json"),
        help="Source eval output JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evals/baselines/baseline_20260221.json"),
        help="Baseline output JSON",
    )
    args = parser.parse_args()

    if not args.source.exists():
        print(f"[capture-baseline] Source file missing: {args.source}")
        return 2

    with args.source.open() as f:
        payload: dict[str, Any] = json.load(f)

    metrics = _extract_metrics(payload)
    if not metrics:
        print("[capture-baseline] No supported metrics found in source file")
        return 2

    baseline = {
        "schema": "jarvis_reply_phase0_baseline_v1",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_file": str(args.source),
        "metrics": metrics,
        "gates": {
            "judge_avg_min_delta": -1.0,
            "anti_ai_clean_rate_min_delta": -0.05,
            "category_accuracy_min_delta": -0.05,
            "latency_p95_ms_max_relative_increase": 0.50,
            "hallucination_rate_max_absolute": 0.08,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"[capture-baseline] Wrote baseline to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
