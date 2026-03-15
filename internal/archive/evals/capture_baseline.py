#!/usr/bin/env python3  # noqa: E501
"""Capture normalized Phase-0 baseline metrics from an eval output JSON."""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import argparse  # noqa: E501
import json  # noqa: E501
import time  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501
from typing import Any  # noqa: E402  # noqa: E501


  # noqa: E501
  # noqa: E501
def _extract_metrics(payload: dict[str, Any]) -> dict[str, float]:  # noqa: E501
    out: dict[str, float] = {}  # noqa: E501
    for key in ("judge_avg", "anti_ai_clean_rate", "category_accuracy", "hallucination_rate"):  # noqa: E501
        val = payload.get(key)  # noqa: E501
        if isinstance(val, int | float):  # noqa: E501
            out[key] = float(val)  # noqa: E501
  # noqa: E501
    latency = payload.get("latency", {})  # noqa: E501
    if isinstance(latency, dict):  # noqa: E501
        p95 = latency.get("p95_ms")  # noqa: E501
        if isinstance(p95, int | float):  # noqa: E501
            out["latency_p95_ms"] = float(p95)  # noqa: E501
  # noqa: E501
    return out  # noqa: E501
  # noqa: E501
  # noqa: E501
def main() -> int:  # noqa: E501
    parser = argparse.ArgumentParser(description="Capture reply quality baseline")  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--source",  # noqa: E501
        type=Path,  # noqa: E501
        default=Path("evals/results/eval_pipeline_baseline.json"),  # noqa: E501
        help="Source eval output JSON",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--output",  # noqa: E501
        type=Path,  # noqa: E501
        default=Path("evals/baselines/baseline_20260221.json"),  # noqa: E501
        help="Baseline output JSON",  # noqa: E501
    )  # noqa: E501
    args = parser.parse_args()  # noqa: E501
  # noqa: E501
    if not args.source.exists():  # noqa: E501
        print(f"[capture-baseline] Source file missing: {args.source}")  # noqa: E501
        return 2  # noqa: E501
  # noqa: E501
    with args.source.open() as f:  # noqa: E501
        payload: dict[str, Any] = json.load(f)  # noqa: E501
  # noqa: E501
    metrics = _extract_metrics(payload)  # noqa: E501
    if not metrics:  # noqa: E501
        print("[capture-baseline] No supported metrics found in source file")  # noqa: E501
        return 2  # noqa: E501
  # noqa: E501
    baseline = {  # noqa: E501
        "schema": "jarvis_reply_phase0_baseline_v1",  # noqa: E501
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),  # noqa: E501
        "source_file": str(args.source),  # noqa: E501
        "metrics": metrics,  # noqa: E501
        "gates": {  # noqa: E501
            "judge_avg_min_delta": -1.0,  # noqa: E501
            "anti_ai_clean_rate_min_delta": -0.05,  # noqa: E501
            "category_accuracy_min_delta": -0.05,  # noqa: E501
            "latency_p95_ms_max_relative_increase": 0.50,  # noqa: E501
            "hallucination_rate_max_absolute": 0.08,  # noqa: E501
        },  # noqa: E501
    }  # noqa: E501
  # noqa: E501
    args.output.parent.mkdir(parents=True, exist_ok=True)  # noqa: E501
    args.output.write_text(json.dumps(baseline, indent=2) + "\n")  # noqa: E501
    print(f"[capture-baseline] Wrote baseline to {args.output}")  # noqa: E501
    return 0  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    raise SystemExit(main())  # noqa: E501
