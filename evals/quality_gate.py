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
    mode: str  # "min_delta", "min_absolute", "max_relative_increase", "max_absolute"
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


def _gates_from_payload(payload: dict[str, Any]) -> list[DeltaGate]:
    """Load gate config from baseline payload if present, else use defaults."""
    cfg = payload.get("gates")
    if not isinstance(cfg, dict):
        return _default_gates()

    gates: list[DeltaGate] = []
    mapping = {
        "judge_avg_min_delta": ("judge_avg", "min_delta"),
        "judge_avg_min_absolute": ("judge_avg", "min_absolute"),
        "anti_ai_clean_rate_min_delta": ("anti_ai_clean_rate", "min_delta"),
        "category_accuracy_min_delta": ("category_accuracy", "min_delta"),
        "latency_p95_ms_max_relative_increase": ("latency_p95_ms", "max_relative_increase"),
        "hallucination_rate_max_absolute": ("hallucination_rate", "max_absolute"),
    }
    for cfg_key, (key, mode) in mapping.items():
        val = cfg.get(cfg_key)
        if isinstance(val, int | float):
            gates.append(DeltaGate(key=key, mode=mode, value=float(val)))

    return gates if gates else _default_gates()


def _check_gate(gate: DeltaGate, baseline: float, candidate: float) -> tuple[bool, str]:
    if gate.mode == "min_delta":
        delta = candidate - baseline
        ok = delta >= gate.value
        return ok, f"delta={delta:+.4f} (min {gate.value:+.4f})"

    if gate.mode == "min_absolute":
        ok = candidate >= gate.value
        return ok, f"value={candidate:.4f} (min {gate.value:.4f})"

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
    parser.add_argument(
        "--require-judge",
        action="store_true",
        help="Fail if candidate judge_avg is missing",
    )
    parser.add_argument(
        "--judge-min-absolute",
        type=float,
        default=None,
        help="Fail if candidate judge_avg is below this absolute threshold",
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
    for gate in _gates_from_payload(baseline_payload):
        b = baseline.get(gate.key)
        c = candidate.get(gate.key)
        # Absolute gates are candidate-only; all others compare baseline vs candidate.
        if gate.mode == "min_absolute":
            if c is None:
                skipped += 1
                print(f"[quality-gate] SKIP {gate.key}: missing candidate metric for absolute gate")
                continue
            ok, detail = _check_gate(gate, 0.0, c)
            status = "PASS" if ok else "FAIL"
            print(f"[quality-gate] {status} {gate.key}: candidate={c:.4f} | {detail}")
            if not ok:
                failed += 1
            continue

        if b is None or c is None:
            if args.require_judge and gate.key == "judge_avg" and c is None:
                failed += 1
                print("[quality-gate] FAIL judge_avg: missing metric (required)")
                continue
            skipped += 1
            print(f"[quality-gate] SKIP {gate.key}: missing metric")
            continue

        ok, detail = _check_gate(gate, b, c)
        status = "PASS" if ok else "FAIL"
        print(f"[quality-gate] {status} {gate.key}: baseline={b:.4f}, candidate={c:.4f} | {detail}")
        if not ok:
            failed += 1

    if args.require_judge and "judge_avg" not in candidate:
        failed += 1
        print("[quality-gate] FAIL judge_avg: missing metric (required)")
    elif args.judge_min_absolute is not None:
        judge_avg = candidate.get("judge_avg")
        if judge_avg is None:
            failed += 1
            print(
                f"[quality-gate] FAIL judge_avg: missing metric for min absolute "
                f"{args.judge_min_absolute:.2f}"
            )
        else:
            ok = judge_avg >= args.judge_min_absolute
            status = "PASS" if ok else "FAIL"
            print(
                f"[quality-gate] {status} judge_avg_min_absolute: candidate={judge_avg:.4f} "
                f"(min {args.judge_min_absolute:.4f})"
            )
            if not ok:
                failed += 1

    if failed:
        print(f"[quality-gate] FAILED ({failed} regressions, {skipped} skipped)")
        return 1

    print(f"[quality-gate] PASSED (0 regressions, {skipped} skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
