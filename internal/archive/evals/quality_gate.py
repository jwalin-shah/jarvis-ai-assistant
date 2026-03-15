#!/usr/bin/env python3  # noqa: E501
"""Phase-0 reply quality regression gate.  # noqa: E501
  # noqa: E501
Compares candidate eval metrics against baseline metrics and fails on regressions.  # noqa: E501
  # noqa: E501
Supported input formats:  # noqa: E501
1) Raw eval_pipeline output JSON (e.g. evals/results/eval_pipeline_baseline.json)  # noqa: E501
2) Normalized baseline/candidate JSON with top-level `metrics` object  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import argparse  # noqa: E501
import json  # noqa: E501
import sys  # noqa: E501
from dataclasses import dataclass  # noqa: E402  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501
from typing import Any  # noqa: E402  # noqa: E501


  # noqa: E501
  # noqa: E501
@dataclass(frozen=True)  # noqa: E501
class DeltaGate:  # noqa: E501
    key: str  # noqa: E501
    mode: str  # "min_delta", "min_absolute", "max_relative_increase", "max_absolute"  # noqa: E501
    value: float  # noqa: E501
  # noqa: E501
  # noqa: E501
def _extract_metrics(payload: dict[str, Any]) -> dict[str, float]:  # noqa: E501
    if isinstance(payload.get("metrics"), dict):  # noqa: E501
        metrics = payload["metrics"]  # noqa: E501
        return {k: float(v) for k, v in metrics.items() if isinstance(v, int | float)}  # noqa: E501
  # noqa: E501
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
def _load(path: Path) -> dict[str, Any]:  # noqa: E501
    with path.open() as f:  # noqa: E501
        return json.load(f)  # noqa: E501
  # noqa: E501
  # noqa: E501
def _default_gates() -> list[DeltaGate]:  # noqa: E501
    return [  # noqa: E501
        DeltaGate(key="judge_avg", mode="min_delta", value=-1.0),  # noqa: E501
        DeltaGate(key="anti_ai_clean_rate", mode="min_delta", value=-0.05),  # noqa: E501
        DeltaGate(key="category_accuracy", mode="min_delta", value=-0.05),  # noqa: E501
        DeltaGate(key="latency_p95_ms", mode="max_relative_increase", value=0.50),  # noqa: E501
        DeltaGate(key="hallucination_rate", mode="max_absolute", value=0.08),  # noqa: E501
    ]  # noqa: E501
  # noqa: E501
  # noqa: E501
def _gates_from_payload(payload: dict[str, Any]) -> list[DeltaGate]:  # noqa: E501
    """Load gate config from baseline payload if present, else use defaults."""  # noqa: E501
    cfg = payload.get("gates")  # noqa: E501
    if not isinstance(cfg, dict):  # noqa: E501
        return _default_gates()  # noqa: E501
  # noqa: E501
    gates: list[DeltaGate] = []  # noqa: E501
    mapping = {  # noqa: E501
        "judge_avg_min_delta": ("judge_avg", "min_delta"),  # noqa: E501
        "judge_avg_min_absolute": ("judge_avg", "min_absolute"),  # noqa: E501
        "anti_ai_clean_rate_min_delta": ("anti_ai_clean_rate", "min_delta"),  # noqa: E501
        "category_accuracy_min_delta": ("category_accuracy", "min_delta"),  # noqa: E501
        "latency_p95_ms_max_relative_increase": ("latency_p95_ms", "max_relative_increase"),  # noqa: E501
        "hallucination_rate_max_absolute": ("hallucination_rate", "max_absolute"),  # noqa: E501
    }  # noqa: E501
    for cfg_key, (key, mode) in mapping.items():  # noqa: E501
        val = cfg.get(cfg_key)  # noqa: E501
        if isinstance(val, int | float):  # noqa: E501
            gates.append(DeltaGate(key=key, mode=mode, value=float(val)))  # noqa: E501
  # noqa: E501
    return gates if gates else _default_gates()  # noqa: E501
  # noqa: E501
  # noqa: E501
def _check_gate(gate: DeltaGate, baseline: float, candidate: float) -> tuple[bool, str]:  # noqa: E501
    if gate.mode == "min_delta":  # noqa: E501
        delta = candidate - baseline  # noqa: E501
        ok = delta >= gate.value  # noqa: E501
        return ok, f"delta={delta:+.4f} (min {gate.value:+.4f})"  # noqa: E501
  # noqa: E501
    if gate.mode == "min_absolute":  # noqa: E501
        ok = candidate >= gate.value  # noqa: E501
        return ok, f"value={candidate:.4f} (min {gate.value:.4f})"  # noqa: E501
  # noqa: E501
    if gate.mode == "max_relative_increase":  # noqa: E501
        if baseline <= 0:  # noqa: E501
            return True, "baseline<=0; skipped"  # noqa: E501
        rel = (candidate - baseline) / baseline  # noqa: E501
        ok = rel <= gate.value  # noqa: E501
        return ok, f"relative_increase={rel:+.4f} (max {gate.value:+.4f})"  # noqa: E501
  # noqa: E501
    if gate.mode == "max_absolute":  # noqa: E501
        ok = candidate <= gate.value  # noqa: E501
        return ok, f"value={candidate:.4f} (max {gate.value:.4f})"  # noqa: E501
  # noqa: E501
    return False, f"unknown gate mode: {gate.mode}"  # noqa: E501
  # noqa: E501
  # noqa: E501
def main() -> int:  # noqa: E501
    parser = argparse.ArgumentParser(description="Reply quality regression gate")  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--baseline",  # noqa: E501
        type=Path,  # noqa: E501
        default=Path("evals/baselines/baseline_20260221.json"),  # noqa: E501
        help="Baseline metrics JSON",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--candidate",  # noqa: E501
        type=Path,  # noqa: E501
        default=Path("evals/results/eval_pipeline_baseline.json"),  # noqa: E501
        help="Candidate metrics JSON",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--allow-missing-candidate",  # noqa: E501
        action="store_true",  # noqa: E501
        help="Return success if candidate metrics file is missing",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--require-judge",  # noqa: E501
        action="store_true",  # noqa: E501
        help="Fail if candidate judge_avg is missing",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--judge-min-absolute",  # noqa: E501
        type=float,  # noqa: E501
        default=None,  # noqa: E501
        help="Fail if candidate judge_avg is below this absolute threshold",  # noqa: E501
    )  # noqa: E501
    args = parser.parse_args()  # noqa: E501
  # noqa: E501
    if not args.baseline.exists():  # noqa: E501
        print(f"[quality-gate] Baseline file missing: {args.baseline}")  # noqa: E501
        return 2  # noqa: E501
  # noqa: E501
    if not args.candidate.exists():  # noqa: E501
        msg = f"[quality-gate] Candidate file missing: {args.candidate}"  # noqa: E501
        if args.allow_missing_candidate:  # noqa: E501
            print(msg + " (allowed)")  # noqa: E501
            return 0  # noqa: E501
        print(msg)  # noqa: E501
        return 2  # noqa: E501
  # noqa: E501
    baseline_payload = _load(args.baseline)  # noqa: E501
    candidate_payload = _load(args.candidate)  # noqa: E501
  # noqa: E501
    baseline = _extract_metrics(baseline_payload)  # noqa: E501
    candidate = _extract_metrics(candidate_payload)  # noqa: E501
  # noqa: E501
    if not baseline:  # noqa: E501
        print("[quality-gate] Baseline metrics are empty")  # noqa: E501
        return 2  # noqa: E501
    if not candidate:  # noqa: E501
        print("[quality-gate] Candidate metrics are empty")  # noqa: E501
        return 2  # noqa: E501
  # noqa: E501
    failed = 0  # noqa: E501
    skipped = 0  # noqa: E501
    for gate in _gates_from_payload(baseline_payload):  # noqa: E501
        b = baseline.get(gate.key)  # noqa: E501
        c = candidate.get(gate.key)  # noqa: E501
        # Absolute gates are candidate-only; all others compare baseline vs candidate.  # noqa: E501
        if gate.mode == "min_absolute":  # noqa: E501
            if c is None:  # noqa: E501
                skipped += 1  # noqa: E501
                print(f"[quality-gate] SKIP {gate.key}: missing candidate metric for absolute gate")  # noqa: E501
                continue  # noqa: E501
            ok, detail = _check_gate(gate, 0.0, c)  # noqa: E501
            status = "PASS" if ok else "FAIL"  # noqa: E501
            print(f"[quality-gate] {status} {gate.key}: candidate={c:.4f} | {detail}")  # noqa: E501
            if not ok:  # noqa: E501
                failed += 1  # noqa: E501
            continue  # noqa: E501
  # noqa: E501
        if b is None or c is None:  # noqa: E501
            if args.require_judge and gate.key == "judge_avg" and c is None:  # noqa: E501
                failed += 1  # noqa: E501
                print("[quality-gate] FAIL judge_avg: missing metric (required)")  # noqa: E501
                continue  # noqa: E501
            skipped += 1  # noqa: E501
            print(f"[quality-gate] SKIP {gate.key}: missing metric")  # noqa: E501
            continue  # noqa: E501
  # noqa: E501
        ok, detail = _check_gate(gate, b, c)  # noqa: E501
        status = "PASS" if ok else "FAIL"  # noqa: E501
        print(f"[quality-gate] {status} {gate.key}: baseline={b:.4f}, candidate={c:.4f} | {detail}")  # noqa: E501
        if not ok:  # noqa: E501
            failed += 1  # noqa: E501
  # noqa: E501
    if args.require_judge and "judge_avg" not in candidate:  # noqa: E501
        failed += 1  # noqa: E501
        print("[quality-gate] FAIL judge_avg: missing metric (required)")  # noqa: E501
    elif args.judge_min_absolute is not None:  # noqa: E501
        judge_avg = candidate.get("judge_avg")  # noqa: E501
        if judge_avg is None:  # noqa: E501
            failed += 1  # noqa: E501
            print(  # noqa: E501
                f"[quality-gate] FAIL judge_avg: missing metric for min absolute "  # noqa: E501
                f"{args.judge_min_absolute:.2f}"  # noqa: E501
            )  # noqa: E501
        else:  # noqa: E501
            ok = judge_avg >= args.judge_min_absolute  # noqa: E501
            status = "PASS" if ok else "FAIL"  # noqa: E501
            print(  # noqa: E501
                f"[quality-gate] {status} judge_avg_min_absolute: candidate={judge_avg:.4f} "  # noqa: E501
                f"(min {args.judge_min_absolute:.4f})"  # noqa: E501
            )  # noqa: E501
            if not ok:  # noqa: E501
                failed += 1  # noqa: E501
  # noqa: E501
    if failed:  # noqa: E501
        print(f"[quality-gate] FAILED ({failed} regressions, {skipped} skipped)")  # noqa: E501
        return 1  # noqa: E501
  # noqa: E501
    print(f"[quality-gate] PASSED (0 regressions, {skipped} skipped)")  # noqa: E501
    return 0  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    sys.exit(main())  # noqa: E501
