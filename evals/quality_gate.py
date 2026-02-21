#!/usr/bin/env python3
"""Phase-0 reply quality regression gate.

Compares candidate eval metrics against baseline metrics and fails on regressions.

Supported input formats:
1) Raw eval_pipeline output JSON (e.g. evals/results/eval_pipeline_baseline.json)
2) Normalized baseline/candidate JSON with top-level `metrics` object
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DeltaGate:
    key: str
    mode: str  # "min_delta", "max_relative_increase", "max_absolute"
    value: float


def _extract_metrics(payload: dict[str, Any]) -> dict[str, float]:
    if isinstance(payload.get("metrics"), dict):
        metrics = payload["metrics"]
        return {k: float(v) for k, v in metrics.items() if isinstance(v, int | float)}

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


def _load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _default_gates() -> list[DeltaGate]:
    return [
        DeltaGate(key="judge_avg", mode="min_delta", value=-1.0),
        DeltaGate(key="anti_ai_clean_rate", mode="min_delta", value=-0.05),
        DeltaGate(key="category_accuracy", mode="min_delta", value=-0.05),
        DeltaGate(key="latency_p95_ms", mode="max_relative_increase", value=0.50),
        DeltaGate(key="hallucination_rate", mode="max_absolute", value=0.08),
    ]


def _check_gate(gate: DeltaGate, baseline: float, candidate: float) -> tuple[bool, str]:
    if gate.mode == "min_delta":
        delta = candidate - baseline
        ok = delta >= gate.value
        return ok, f"delta={delta:+.4f} (min {gate.value:+.4f})"

    if gate.mode == "max_relative_increase":
        if baseline <= 0:
            return True, "baseline<=0; skipped"
        rel = (candidate - baseline) / baseline
        ok = rel <= gate.value
        return ok, f"relative_increase={rel:+.4f} (max {gate.value:+.4f})"

    if gate.mode == "max_absolute":
        ok = candidate <= gate.value
        return ok, f"value={candidate:.4f} (max {gate.value:.4f})"

    return False, f"unknown gate mode: {gate.mode}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Reply quality regression gate")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("evals/baselines/baseline_20260221.json"),
        help="Baseline metrics JSON",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path("evals/results/eval_pipeline_baseline.json"),
        help="Candidate metrics JSON",
    )
    parser.add_argument(
        "--allow-missing-candidate",
        action="store_true",
        help="Return success if candidate metrics file is missing",
    )
    args = parser.parse_args()

    if not args.baseline.exists():
        print(f"[quality-gate] Baseline file missing: {args.baseline}")
        return 2

    if not args.candidate.exists():
        msg = f"[quality-gate] Candidate file missing: {args.candidate}"
        if args.allow_missing_candidate:
            print(msg + " (allowed)")
            return 0
        print(msg)
        return 2

    baseline_payload = _load(args.baseline)
    candidate_payload = _load(args.candidate)

    baseline = _extract_metrics(baseline_payload)
    candidate = _extract_metrics(candidate_payload)

    if not baseline:
        print("[quality-gate] Baseline metrics are empty")
        return 2
    if not candidate:
        print("[quality-gate] Candidate metrics are empty")
        return 2

    failed = 0
    skipped = 0
    for gate in _default_gates():
        b = baseline.get(gate.key)
        c = candidate.get(gate.key)
        if b is None or c is None:
            skipped += 1
            print(f"[quality-gate] SKIP {gate.key}: missing metric")
            continue

        ok, detail = _check_gate(gate, b, c)
        status = "PASS" if ok else "FAIL"
        print(f"[quality-gate] {status} {gate.key}: baseline={b:.4f}, candidate={c:.4f} | {detail}")
        if not ok:
            failed += 1

    if failed:
        print(f"[quality-gate] FAILED ({failed} regressions, {skipped} skipped)")
        return 1

    print(f"[quality-gate] PASSED (0 regressions, {skipped} skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
