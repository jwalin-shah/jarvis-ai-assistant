#!/usr/bin/env python3
"""Evaluate GLiNER candidate extraction against a gold-labeled dataset.

Usage:
    uv run python scripts/eval_gliner_candidates.py [--gold PATH] [--threshold FLOAT]

Outputs:
    - Printed report with per-label and per-type metrics
    - JSON metrics file at training_data/gliner_goldset/gliner_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from jarvis.utils.logging import setup_script_logging

GOLD_PATH = Path("training_data/gliner_goldset/candidate_gold_merged_r4.json")
METRICS_PATH = Path("training_data/gliner_goldset/gliner_metrics.json")
LOG_PATH = Path("eval_gliner_candidates.log")

from eval_shared import DEFAULT_LABEL_ALIASES, spans_match  # noqa: E402
from gliner_shared import enforce_runtime_stack, parse_context_messages  # noqa: E402

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def support(self) -> int:
        return self.tp + self.fn

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "support": self.support,
        }


def compute_metrics(
    gold_records: list[dict],
    predictions: dict[str, list[dict]],
    label_aliases: dict[str, set[str]] | None = None,
) -> dict:
    """Compute span-level precision/recall/F1.

    Args:
        gold_records: list of gold records with expected_candidates
        predictions: sample_id -> list of predicted candidate dicts
            each with at least span_text, span_label
        label_aliases: Optional label alias mapping for flexible matching.
            Defaults to DEFAULT_LABEL_ALIASES if None.

    Returns:
        dict with overall, per_label, per_type, per_slice metrics
    """
    if label_aliases is None:
        label_aliases = DEFAULT_LABEL_ALIASES
    overall = Metrics()
    per_label: dict[str, Metrics] = defaultdict(Metrics)
    per_type: dict[str, Metrics] = defaultdict(Metrics)
    per_slice: dict[str, Metrics] = defaultdict(Metrics)
    errors: list[dict] = []

    for rec in gold_records:
        sid = rec["sample_id"]
        gold_cands = rec.get("expected_candidates") or []
        pred_cands = predictions.get(sid, [])
        slc = rec.get("slice", "unknown")

        # Track which golds and preds are matched
        gold_matched = [False] * len(gold_cands)
        pred_matched = [False] * len(pred_cands)

        # Greedy matching: for each gold, find best matching pred
        for gi, gc in enumerate(gold_cands):
            for pi, pc in enumerate(pred_cands):
                if pred_matched[pi]:
                    continue
                if spans_match(
                    pc.get("span_text", ""),
                    pc.get("span_label", ""),
                    gc.get("span_text", ""),
                    gc.get("span_label", ""),
                    label_aliases=label_aliases,
                ):
                    gold_matched[gi] = True
                    pred_matched[pi] = True
                    # TP
                    overall.tp += 1
                    per_label[gc["span_label"]].tp += 1
                    per_type[gc.get("fact_type", "unknown")].tp += 1
                    per_slice[slc].tp += 1
                    break

        # Unmatched golds = FN
        for gi, gc in enumerate(gold_cands):
            if not gold_matched[gi]:
                overall.fn += 1
                per_label[gc["span_label"]].fn += 1
                per_type[gc.get("fact_type", "unknown")].fn += 1
                per_slice[slc].fn += 1
                errors.append(
                    {
                        "type": "fn",
                        "sample_id": sid,
                        "slice": slc,
                        "message_text": rec["message_text"][:100],
                        "gold_span": gc["span_text"],
                        "gold_label": gc["span_label"],
                    }
                )

        # Unmatched preds = FP
        for pi, pc in enumerate(pred_cands):
            if not pred_matched[pi]:
                overall.fp += 1
                label = pc.get("span_label", "unknown")
                per_label[label].fp += 1
                ftype = pc.get("fact_type", "unknown")
                per_type[ftype].fp += 1
                per_slice[slc].fp += 1
                errors.append(
                    {
                        "type": "fp",
                        "sample_id": sid,
                        "slice": slc,
                        "message_text": rec["message_text"][:100],
                        "pred_span": pc.get("span_text", ""),
                        "pred_label": label,
                    }
                )

    return {
        "overall": overall.to_dict(),
        "per_label": {k: v.to_dict() for k, v in sorted(per_label.items())},
        "per_type": {k: v.to_dict() for k, v in sorted(per_type.items())},
        "per_slice": {k: v.to_dict() for k, v in sorted(per_slice.items())},
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------


def threshold_sweep(
    gold_records: list[dict],
    all_raw_preds: dict[str, list[dict]],
    thresholds: list[float] | None = None,
) -> list[dict]:
    """Sweep GLiNER score thresholds and compute metrics at each level."""
    if thresholds is None:
        thresholds = [round(0.30 + i * 0.05, 2) for i in range(11)]  # 0.30 to 0.80

    results = []
    for thresh in thresholds:
        # Filter predictions by threshold
        filtered = {}
        for sid, preds in all_raw_preds.items():
            filtered[sid] = [p for p in preds if p.get("gliner_score", 1.0) >= thresh]
        metrics = compute_metrics(gold_records, filtered)
        results.append(
            {
                "threshold": thresh,
                **metrics["overall"],
            }
        )
    return results


# ---------------------------------------------------------------------------
# Error slices
# ---------------------------------------------------------------------------


def compute_error_slices(gold_records: list[dict], errors: list[dict]) -> dict:
    """Compute error statistics by message characteristics."""
    short_msgs = [e for e in errors if len(e["message_text"]) < 20]
    fp_on_negatives = [
        e
        for e in errors
        if e["type"] == "fp" and e["slice"] in ("hard_negative", "random_negative")
    ]

    # Count messages by length bucket
    len_buckets = {"<20": 0, "20-50": 0, "50-100": 0, ">100": 0}
    for rec in gold_records:
        ml = len(rec["message_text"])
        if ml < 20:
            len_buckets["<20"] += 1
        elif ml < 50:
            len_buckets["20-50"] += 1
        elif ml < 100:
            len_buckets["50-100"] += 1
        else:
            len_buckets[">100"] += 1

    return {
        "short_message_errors": len(short_msgs),
        "fp_on_negatives": len(fp_on_negatives),
        "message_length_distribution": len_buckets,
        "total_errors": len(errors),
    }


def _percentile(values: list[float], pct: float) -> float:
    """Compute percentile for a non-empty list using linear interpolation."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    idx = (len(s) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def compute_raw_prediction_stats(all_raw_preds: dict[str, list[dict]]) -> dict:
    """Summarize pre-filter GLiNER predictions for debugging/calibration."""
    total = sum(len(v) for v in all_raw_preds.values())
    num_messages = len(all_raw_preds)
    with_preds = sum(1 for v in all_raw_preds.values() if v)

    scores: list[float] = []
    by_label: dict[str, int] = defaultdict(int)

    for preds in all_raw_preds.values():
        for p in preds:
            by_label[p.get("span_label", "unknown")] += 1
            try:
                scores.append(float(p.get("gliner_score", 0.0)))
            except (TypeError, ValueError):
                scores.append(0.0)

    top_labels = sorted(by_label.items(), key=lambda x: x[1], reverse=True)[:15]

    score_summary = {
        "min": round(min(scores), 4) if scores else 0.0,
        "p25": round(_percentile(scores, 0.25), 4) if scores else 0.0,
        "p50": round(_percentile(scores, 0.50), 4) if scores else 0.0,
        "p75": round(_percentile(scores, 0.75), 4) if scores else 0.0,
        "p90": round(_percentile(scores, 0.90), 4) if scores else 0.0,
        "p95": round(_percentile(scores, 0.95), 4) if scores else 0.0,
        "max": round(max(scores), 4) if scores else 0.0,
    }

    return {
        "raw_total_predictions": total,
        "raw_messages_with_predictions": with_preds,
        "raw_messages_total": num_messages,
        "raw_predictions_per_message": round(total / num_messages, 4) if num_messages > 0 else 0.0,
        "raw_score_summary": score_summary,
        "raw_top_labels": [{"label": l, "count": c} for l, c in top_labels],
    }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def print_report(
    metrics: dict,
    sweep: list[dict],
    error_slices: dict,
    raw_stats: dict,
    mode: str,
) -> None:
    """Print a human-readable evaluation report."""
    ov = metrics["overall"]
    print("\n" + "=" * 60, flush=True)
    print("GLiNER Candidate Extraction Evaluation", flush=True)
    print("=" * 60, flush=True)
    print(f"Mode: {mode}", flush=True)

    print("\nRaw GLiNER output (pre-filter):", flush=True)
    print(
        "  predictions: "
        f"{raw_stats['raw_total_predictions']} across "
        f"{raw_stats['raw_messages_with_predictions']}/{raw_stats['raw_messages_total']} messages",
        flush=True,
    )
    print(f"  avg preds/msg: {raw_stats['raw_predictions_per_message']:.3f}", flush=True)
    ss = raw_stats["raw_score_summary"]
    print(
        "  score quantiles: "
        f"min={ss['min']:.3f} p25={ss['p25']:.3f} p50={ss['p50']:.3f} "
        f"p75={ss['p75']:.3f} p90={ss['p90']:.3f} p95={ss['p95']:.3f} max={ss['max']:.3f}",
        flush=True,
    )
    if raw_stats["raw_top_labels"]:
        print("  top labels:", flush=True)
        for row in raw_stats["raw_top_labels"][:8]:
            print(f"    - {row['label']}: {row['count']}", flush=True)

    print(
        f"\nOverall:  P={ov['precision']:.3f}  R={ov['recall']:.3f}  "
        f"F1={ov['f1']:.3f}  (TP={ov['tp']} FP={ov['fp']} FN={ov['fn']})",
        flush=True,
    )

    # Per-label table
    print(f"\n{'Label':<20} {'P':>6} {'R':>6} {'F1':>6} {'Sup':>5}", flush=True)
    print("-" * 45, flush=True)
    for label, m in sorted(metrics["per_label"].items(), key=lambda x: -x[1]["support"]):
        print(
            f"{label:<20} {m['precision']:>6.3f} {m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} {m['support']:>5}",
            flush=True,
        )

    # Per-type table
    print(f"\n{'Fact Type':<25} {'P':>6} {'R':>6} {'F1':>6} {'Sup':>5}", flush=True)
    print("-" * 50, flush=True)
    for ftype, m in sorted(metrics["per_type"].items(), key=lambda x: -x[1]["support"]):
        print(
            f"{ftype:<25} {m['precision']:>6.3f} {m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} {m['support']:>5}",
            flush=True,
        )

    # Per-slice table
    print(f"\n{'Slice':<20} {'P':>6} {'R':>6} {'F1':>6} {'Sup':>5}", flush=True)
    print("-" * 45, flush=True)
    for slc, m in sorted(metrics["per_slice"].items()):
        print(
            f"{slc:<20} {m['precision']:>6.3f} {m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} {m['support']:>5}",
            flush=True,
        )

    # Threshold sweep
    print(f"\n{'Thresh':>7} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}", flush=True)
    print("-" * 45, flush=True)
    for row in sweep:
        print(
            f"{row['threshold']:>7.2f} {row['precision']:>6.3f} {row['recall']:>6.3f} "
            f"{row['f1']:>6.3f} {row['tp']:>5} {row['fp']:>5} {row['fn']:>5}",
            flush=True,
        )

    # Error slices
    print("\nError Slices:", flush=True)
    print(f"  Short message (<20 char) errors: {error_slices['short_message_errors']}", flush=True)
    print(f"  FP on negative messages: {error_slices['fp_on_negatives']}", flush=True)
    print(f"  Total errors: {error_slices['total_errors']}", flush=True)
    print(
        f"  Message length distribution: {error_slices['message_length_distribution']}", flush=True
    )

    # Top FP examples
    fps = [e for e in metrics["errors"] if e["type"] == "fp"][:10]
    if fps:
        print("\nTop False Positives (first 10):", flush=True)
        for e in fps:
            print(
                f'  [{e["slice"]}] "{e["message_text"][:60]}..." '
                f"-> {e.get('pred_span', '')} ({e.get('pred_label', '')})",
                flush=True,
            )

    # Top FN examples
    fns = [e for e in metrics["errors"] if e["type"] == "fn"][:10]
    if fns:
        print("\nTop False Negatives (first 10):", flush=True)
        for e in fns:
            print(
                f'  [{e["slice"]}] "{e["message_text"][:60]}..." '
                f"-> missed {e.get('gold_span', '')} ({e.get('gold_label', '')})",
                flush=True,
            )

    print("\n" + "=" * 60, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_evaluation(
    gold_path: Path,
    override_threshold: float | None = None,
    *,
    mode: str = "pipeline",
    raw_threshold: float = 0.01,
    disable_label_min: bool = False,
    context_window: int = 0,
    label_profile: str = "balanced",
    drop_labels: list[str] | None = None,
    allow_unstable_stack: bool = False,
    dump_candidates: Path | None = None,
) -> dict:
    """Run the full evaluation pipeline."""
    enforce_runtime_stack(allow_unstable_stack)

    # Load gold set
    log.info(f"Loading gold set from {gold_path}")
    with open(gold_path) as f:
        gold_records = json.load(f)
    log.info(f"Loaded {len(gold_records)} gold records")
    if context_window > 0:
        log.warning(
            "context_window=%s can reduce extraction quality due to GLiNER truncation; "
            "defaulting to 0 is recommended.",
            context_window,
        )

    # Sanity checks
    pos = [r for r in gold_records if r["slice"] == "positive"]
    neg = [r for r in gold_records if r["slice"] != "positive"]
    pos_with_cands = sum(1 for r in pos if r.get("expected_candidates"))
    neg_with_cands = sum(1 for r in neg if r.get("expected_candidates"))
    log.info(f"  Positive: {len(pos)} ({pos_with_cands} with candidates)")
    log.info(f"  Negative: {len(neg)} ({neg_with_cands} with candidates)")

    if pos_with_cands < len(pos) * 0.5:
        log.warning(
            f"WARNING: Only {pos_with_cands}/{len(pos)} positive messages "
            "have expected_candidates. Gold set may be incomplete."
        )

    # Initialize CandidateExtractor
    log.info("Initializing CandidateExtractor...")
    from jarvis.contacts.candidate_extractor import CandidateExtractor, labels_for_profile

    active_labels = labels_for_profile(label_profile)
    if drop_labels:
        drop_set = set(drop_labels)
        active_labels = [lbl for lbl in active_labels if lbl not in drop_set]
    if not active_labels:
        log.error("No active labels left after applying label profile/drop-label filters.")
        raise SystemExit(2)

    log.info("  Label profile: %s", label_profile)
    log.info("  Active labels: %s", ", ".join(active_labels))

    # Disable entailment in compat venv (no MLX available)
    try:
        import mlx.core  # noqa: F401
        has_mlx = True
    except ImportError:
        has_mlx = False
    extractor = CandidateExtractor(
        labels=active_labels, label_profile=label_profile, use_entailment=has_mlx,
    )
    extractor._load_model()

    # Run GLiNER on all messages
    log.info(f"Running GLiNER extraction on {len(gold_records)} messages...")
    all_raw_preds: dict[str, list[dict]] = {}
    predictions: dict[str, list[dict]] = {}
    t0 = time.time()

    for i, rec in enumerate(gold_records):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Processing {i + 1}/{len(gold_records)} ({elapsed:.1f}s elapsed)", flush=True)

        prev_messages: list[str] | None = None
        next_messages: list[str] | None = None
        if context_window > 0:
            prev_all = parse_context_messages(rec.get("context_prev"))
            next_all = parse_context_messages(rec.get("context_next"))
            prev_slice = prev_all[-context_window:] if prev_all else []
            next_slice = next_all[:context_window] if next_all else []
            prev_messages = prev_slice or None
            next_messages = next_slice or None

        raw_entities = extractor.predict_raw_entities(
            text=rec["message_text"],
            threshold=raw_threshold,
            prev_messages=prev_messages,
            next_messages=next_messages,
        )
        raw_preds = [
            {
                "span_text": e.get("text", ""),
                "span_label": e.get("label", ""),
                "fact_type": extractor._resolve_fact_type(
                    rec["message_text"],
                    e.get("text", ""),
                    e.get("label", ""),
                ),
                "gliner_score": float(e.get("score", 0.0)),
            }
            for e in raw_entities
        ]
        all_raw_preds[rec["sample_id"]] = raw_preds

        if mode == "raw":
            predictions[rec["sample_id"]] = raw_preds
        else:
            candidates = extractor.extract_candidates(
                text=rec["message_text"],
                message_id=rec.get("message_id", 0),
                is_from_me=rec.get("is_from_me"),
                threshold=override_threshold,
                apply_label_thresholds=not disable_label_min,
                prev_messages=prev_messages,
                next_messages=next_messages,
            )
            predictions[rec["sample_id"]] = [
                {
                    "span_text": c.span_text,
                    "span_label": c.span_label,
                    "fact_type": c.fact_type,
                    "gliner_score": c.gliner_score,
                }
                for c in candidates
            ]

    elapsed = time.time() - t0
    total_preds = sum(len(v) for v in predictions.values())
    raw_total_preds = sum(len(v) for v in all_raw_preds.values())
    log.info(
        f"Extraction complete: {total_preds} candidates "
        f"in {elapsed:.1f}s ({elapsed / len(gold_records) * 1000:.1f}ms/msg)"
    )
    log.info(f"Raw predictions (pre-filter): {raw_total_preds}")

    # Dump intermediate candidates for two-stage eval (GLiNER compat -> MLX entailment)
    if dump_candidates:
        dump_data = {
            "gold_path": str(gold_path),
            "gold_records": gold_records,
            "predictions": predictions,
            "all_raw_preds": all_raw_preds,
            "mode": mode,
            "label_profile": label_profile,
            "extraction_time_s": round(elapsed, 2),
        }
        dump_candidates.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_candidates, "w") as f:
            json.dump(dump_data, f, indent=2)
        log.info(f"Dumped {total_preds} candidates to {dump_candidates}")
        log.info("Run stage 2 with: uv run python scripts/eval_entailment_filter.py")

    # Compute metrics
    log.info("Computing metrics...")
    metrics = compute_metrics(gold_records, predictions)

    # Threshold sweep
    log.info("Running threshold sweep...")
    sweep = threshold_sweep(gold_records, all_raw_preds)

    # Error slices
    error_slices = compute_error_slices(gold_records, metrics["errors"])
    raw_stats = compute_raw_prediction_stats(all_raw_preds)

    # Print report
    print_report(metrics, sweep, error_slices, raw_stats, mode)

    # Save metrics
    output = {
        "gold_path": str(gold_path),
        "num_records": len(gold_records),
        "num_predictions": total_preds,
        "mode": mode,
        "raw_threshold": raw_threshold,
        "disable_label_min": disable_label_min,
        "context_window": context_window,
        "label_profile": label_profile,
        "drop_labels": drop_labels or [],
        "active_labels": active_labels,
        "num_raw_predictions": raw_total_preds,
        "extraction_time_s": round(elapsed, 2),
        "ms_per_message": round(elapsed / len(gold_records) * 1000, 1),
        "overall": metrics["overall"],
        "per_label": metrics["per_label"],
        "per_type": metrics["per_type"],
        "per_slice": metrics["per_slice"],
        "threshold_sweep": sweep,
        "error_slices": error_slices,
        "raw_stats": raw_stats,
    }

    # Don't save full errors to metrics file (too large)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Metrics saved to {METRICS_PATH}")

    return output


def main():
    setup_script_logging("eval_gliner_candidates")
    log.info("Starting eval_gliner_candidates.py")
    parser = argparse.ArgumentParser(description="Evaluate GLiNER candidate extraction")
    parser.add_argument("--gold", type=Path, default=GOLD_PATH, help="Path to gold set JSON")
    parser.add_argument(
        "--label-profile",
        choices=["high_recall", "balanced", "high_precision"],
        default="balanced",
        help="CandidateExtractor label profile",
    )
    parser.add_argument(
        "--drop-label",
        action="append",
        default=[],
        help="Drop a label after profile resolution (repeatable)",
    )
    parser.add_argument(
        "--mode",
        choices=["pipeline", "raw"],
        default="pipeline",
        help=(
            "pipeline = evaluate CandidateExtractor filters; raw = evaluate direct GLiNER output"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override CandidateExtractor model call threshold (pipeline mode only)",
    )
    parser.add_argument(
        "--raw-threshold",
        type=float,
        default=0.01,
        help="Threshold for pre-filter raw GLiNER diagnostics/sweeps",
    )
    parser.add_argument(
        "--no-label-min",
        action="store_true",
        help="Disable per-label minimum score filters in pipeline mode",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=0,
        help="Use up to N previous and N next messages from gold context fields",
    )
    parser.add_argument(
        "--allow-unstable-stack",
        action="store_true",
        help="Allow running outside GLiNER compat runtime (not recommended)",
    )
    parser.add_argument(
        "--dump-candidates",
        type=Path,
        default=None,
        help="Save intermediate candidates to JSON for two-stage eval with entailment",
    )
    args = parser.parse_args()

    if not args.gold.exists():
        log.error(f"Gold set not found: {args.gold}")
        sys.exit(1)

    run_evaluation(
        args.gold,
        args.threshold,
        mode=args.mode,
        raw_threshold=args.raw_threshold,
        disable_label_min=args.no_label_min,
        context_window=args.context_window,
        label_profile=args.label_profile,
        drop_labels=args.drop_label,
        allow_unstable_stack=args.allow_unstable_stack,
        dump_candidates=args.dump_candidates,
    )
    log.info("Finished eval_gliner_candidates.py")


if __name__ == "__main__":
    main()
