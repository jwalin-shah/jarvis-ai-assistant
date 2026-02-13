#!/usr/bin/env python3
"""Stage 2: Apply NLI entailment filter to GLiNER candidates and recompute metrics.

Two-stage eval pipeline:
  Stage 1 (compat venv): scripts/run_gliner_eval_compat.sh --dump-candidates candidates.json
  Stage 2 (main venv):   uv run python scripts/eval_entailment_filter.py \
                         --candidates candidates.json

This loads the intermediate candidates from stage 1, runs NLI entailment filtering
using the MLX cross-encoder, and recomputes precision/recall/F1.
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

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from eval_shared import DEFAULT_LABEL_ALIASES, spans_match

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATES = Path("results/gliner_candidates_dump.json")
OUTPUT_DIR = Path("results/entailment_eval")

# Same thresholds and templates as CandidateExtractor
ENTAILMENT_THRESHOLDS: dict[str, float] = {
    "work.employer": 0.45,
    "work.former_employer": 0.45,
    "work.job_title": 0.55,
    "location.current": 0.30,
    "location.past": 0.30,
    "location.future": 0.20,
    "location.hometown": 0.30,
    "preference.food_like": 0.35,
    "preference.food_dislike": 0.35,
    "preference.activity": 0.30,
    "relationship.family": 0.30,
    "relationship.friend": 0.30,
    "relationship.partner": 0.30,
    "health.condition": 0.45,
    "health.allergy": 0.45,
    "health.dietary": 0.45,
    "personal.school": 0.55,
    "personal.birthday": 0.45,
    "personal.pet": 0.45,
}

HYPOTHESIS_TEMPLATES: dict[str, str] = {
    "relationship.family": "{span} is a family member",
    "relationship.friend": "{span} is a friend",
    "relationship.partner": "{span} is a romantic partner",
    "location.current": "Someone lives in {span}",
    "location.past": "Someone used to live in {span}",
    "location.future": "Someone plans to move to {span}",
    "location.hometown": "Someone grew up in {span}",
    "work.employer": "Someone is employed at {span}",
    "work.former_employer": "Someone used to work at {span}",
    "work.job_title": "Someone works as a {span}",
    "preference.food_like": "Someone enjoys eating {span}",
    "preference.food_dislike": "Someone dislikes eating {span}",
    "preference.activity": "Someone enjoys {span} as a hobby",
    "health.allergy": "Someone is allergic to {span}",
    "health.dietary": "Someone avoids eating {span}",
    "health.condition": "Someone suffers from {span}",
    "personal.birthday": "Someone's birthday is {span}",
    "personal.school": "Someone is a student at {span}",
    "personal.pet": "Someone has a pet named {span}",
}

DEFAULT_ENTAILMENT_THRESHOLD = 0.12


def candidate_to_hypothesis(span_text: str, fact_type: str) -> str:
    template = HYPOTHESIS_TEMPLATES.get(fact_type)
    if template:
        return template.format(span=span_text)
    return f"The message mentions {span_text}"


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


def compute_metrics(
    gold_records: list[dict],
    predictions: dict[str, list[dict]],
) -> dict:
    """Compute span-level precision/recall/F1 with label aliasing."""
    overall = Metrics()
    per_label: dict[str, Metrics] = defaultdict(Metrics)
    per_type: dict[str, Metrics] = defaultdict(Metrics)

    for rec in gold_records:
        sid = rec["sample_id"]
        gold_cands = rec.get("expected_candidates") or []
        pred_cands = predictions.get(sid, [])

        gold_matched = [False] * len(gold_cands)
        pred_matched = [False] * len(pred_cands)

        for gi, gc in enumerate(gold_cands):
            for pi, pc in enumerate(pred_cands):
                if pred_matched[pi]:
                    continue
                if spans_match(
                    pc["span_text"], pc["span_label"],
                    gc["span_text"], gc["span_label"],
                    label_aliases=DEFAULT_LABEL_ALIASES,
                ):
                    gold_matched[gi] = True
                    pred_matched[pi] = True
                    overall.tp += 1
                    per_label[gc["span_label"]].tp += 1
                    ft = gc.get("fact_type") or pc.get("fact_type", "unknown")
                    per_type[ft].tp += 1
                    break

        for gi, matched in enumerate(gold_matched):
            if not matched:
                overall.fn += 1
                per_label[gold_cands[gi]["span_label"]].fn += 1
                ft = gold_cands[gi].get("fact_type", "unknown")
                per_type[ft].fn += 1

        for pi, matched in enumerate(pred_matched):
            if not matched:
                overall.fp += 1
                per_label[pred_cands[pi]["span_label"]].fp += 1
                ft = pred_cands[pi].get("fact_type", "unknown")
                per_type[ft].fp += 1

    return {
        "overall": {
            "precision": overall.precision,
            "recall": overall.recall,
            "f1": overall.f1,
            "tp": overall.tp, "fp": overall.fp, "fn": overall.fn,
        },
        "per_label": {
            k: {"precision": v.precision, "recall": v.recall, "f1": v.f1,
                 "support": v.support, "tp": v.tp, "fp": v.fp, "fn": v.fn}
            for k, v in sorted(per_label.items(), key=lambda x: x[1].support, reverse=True)
        },
        "per_type": {
            k: {"precision": v.precision, "recall": v.recall, "f1": v.f1,
                 "support": v.support, "tp": v.tp, "fp": v.fp, "fn": v.fn}
            for k, v in sorted(per_type.items(), key=lambda x: x[1].support, reverse=True)
        },
    }


def apply_entailment_filter(
    gold_records: list[dict],
    predictions: dict[str, list[dict]],
    global_threshold: float = DEFAULT_ENTAILMENT_THRESHOLD,
) -> dict[str, list[dict]]:
    """Apply NLI entailment filtering to predictions. Returns filtered predictions."""
    # Build message text lookup
    msg_lookup = {rec["sample_id"]: rec["message_text"] for rec in gold_records}

    # Collect all (premise, hypothesis) pairs with tracking info
    all_pairs: list[tuple[str, str]] = []
    pair_info: list[tuple[str, int]] = []  # (sample_id, pred_idx)

    for sid, preds in predictions.items():
        premise = msg_lookup.get(sid, "")
        for idx, pred in enumerate(preds):
            hypothesis = candidate_to_hypothesis(pred["span_text"], pred.get("fact_type", ""))
            all_pairs.append((premise, hypothesis))
            pair_info.append((sid, idx))

    if not all_pairs:
        return predictions

    # Batch NLI scoring - use full score dicts for E-C scoring
    logger.info(f"Running NLI entailment on {len(all_pairs)} candidates...")
    t0 = time.time()
    from models.nli_cross_encoder import get_nli_cross_encoder
    nli = get_nli_cross_encoder()
    full_scores = nli.predict_batch(all_pairs)
    elapsed = time.time() - t0
    logger.info(
        "NLI scoring complete in %.1fs (%.1fms/pair)",
        elapsed, elapsed / len(all_pairs) * 1000,
    )

    # Categories that skip NLI (too destructive)
    NLI_SKIP_CATEGORIES = {
        "preference.activity", "preference.food_like", "preference.food_dislike",
        "health.condition", "health.allergy", "preference.skill",
        "personal.cultural_event",
    }

    # E-C scoring: entailment - contradiction, reject only clear contradictions
    filtered: dict[str, list[dict]] = {sid: [] for sid in predictions}
    kept = 0
    rejected = 0
    skipped = 0

    for (sid, idx), scores in zip(pair_info, full_scores):
        pred = predictions[sid][idx]
        fact_type = pred.get("fact_type", "")

        # Skip NLI for destructive categories
        if fact_type in NLI_SKIP_CATEGORIES:
            filtered[sid].append(dict(pred))
            skipped += 1
            continue

        # E-C scoring: entailment - contradiction
        ec_score = scores["entailment"] - scores["contradiction"]

        if ec_score < -0.5:
            rejected += 1
            logger.debug(
                "Hard-rejected: '%s' (%s) E=%.3f C=%.3f EC=%.3f",
                pred["span_text"], fact_type,
                scores["entailment"], scores["contradiction"], ec_score,
            )
        else:
            multiplier = max(0.4, min(1.0, 0.7 + 0.6 * ec_score))
            pred_copy = dict(pred)
            pred_copy["entailment_score"] = round(scores["entailment"], 4)
            pred_copy["ec_score"] = round(ec_score, 4)
            original_score = pred_copy.get("gliner_score", 1.0)
            pred_copy["gliner_score"] = round(original_score * multiplier, 4)
            filtered[sid].append(pred_copy)
            kept += 1

    logger.info(
        "Entailment: %d skipped, %d kept, %d hard-rejected (%d total)",
        skipped, kept, rejected, skipped + kept + rejected,
    )
    return filtered


def print_report(
    before_metrics: dict,
    after_metrics: dict,
) -> None:
    """Print comparison report."""
    print("\n" + "=" * 70, flush=True)
    print("Entailment Filter Evaluation", flush=True)
    print("=" * 70, flush=True)

    b = before_metrics["overall"]
    a = after_metrics["overall"]

    print(f"\n{'Metric':<12} {'Before':>10} {'After':>10} {'Delta':>10}", flush=True)
    print("-" * 45, flush=True)
    for metric in ["precision", "recall", "f1"]:
        bv = b[metric]
        av = a[metric]
        delta = av - bv
        sign = "+" if delta >= 0 else ""
        print(f"{metric:<12} {bv:>10.3f} {av:>10.3f} {sign}{delta:>9.3f}", flush=True)

    print(f"\n{'':>12} {'Before':>10} {'After':>10}", flush=True)
    print("-" * 35, flush=True)
    for k in ["tp", "fp", "fn"]:
        print(f"{k:<12} {b[k]:>10} {a[k]:>10}", flush=True)

    # Per-label comparison
    print(
        f"\n{'Label':<25} {'P_before':>8} {'P_after':>8} "
        f"{'R_before':>8} {'R_after':>8} {'F1_after':>8} {'Sup':>5}",
        flush=True,
    )
    print("-" * 72, flush=True)
    all_labels = set(before_metrics.get("per_label", {})) | set(after_metrics.get("per_label", {}))
    for label in sorted(
        all_labels,
        key=lambda l: after_metrics.get("per_label", {}).get(
            l, {}
        ).get("support", 0),
        reverse=True,
    ):
        bl = before_metrics.get("per_label", {}).get(label, {})
        al = after_metrics.get("per_label", {}).get(label, {})
        print(
            f"{label:<25} {bl.get('precision', 0):>8.3f} {al.get('precision', 0):>8.3f} "
            f"{bl.get('recall', 0):>8.3f} {al.get('recall', 0):>8.3f} "
            f"{al.get('f1', 0):>8.3f} {al.get('support', 0):>5}",
            flush=True,
        )

    print("\n" + "=" * 70, flush=True)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("entailment_eval.log", mode="w"),
        ],
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path, default=DEFAULT_CANDIDATES,
        help="Path to candidates JSON from stage 1",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_ENTAILMENT_THRESHOLD,
        help=f"Global entailment threshold (default: {DEFAULT_ENTAILMENT_THRESHOLD})",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None,
        help="Save results to JSON",
    )
    args = parser.parse_args()

    if not args.candidates.exists():
        logger.error(f"Candidates file not found: {args.candidates}")
        logger.error(
            "Run stage 1 first: bash scripts/run_gliner_eval_compat.sh"
            " --dump-candidates results/gliner_candidates_dump.json"
        )
        sys.exit(1)

    logger.info(f"Loading candidates from {args.candidates}")
    with open(args.candidates) as f:
        dump = json.load(f)

    gold_records = dump["gold_records"]
    predictions = dump["predictions"]
    logger.info(f"Loaded {len(gold_records)} gold records, "
                f"{sum(len(v) for v in predictions.values())} predictions")

    # Compute before metrics
    before_metrics = compute_metrics(gold_records, predictions)
    logger.info(
        f"Before entailment: P={before_metrics['overall']['precision']:.3f} "
        f"R={before_metrics['overall']['recall']:.3f} F1={before_metrics['overall']['f1']:.3f}"
    )

    # Apply entailment filter
    filtered_predictions = apply_entailment_filter(
        gold_records, predictions, global_threshold=args.threshold,
    )

    # Compute after metrics
    after_metrics = compute_metrics(gold_records, filtered_predictions)
    logger.info(
        f"After entailment: P={after_metrics['overall']['precision']:.3f} "
        f"R={after_metrics['overall']['recall']:.3f} F1={after_metrics['overall']['f1']:.3f}"
    )

    # Print comparison
    print_report(before_metrics, after_metrics)

    # Save results
    if args.output_json:
        output = {
            "before": before_metrics,
            "after": after_metrics,
            "threshold": args.threshold,
            "candidates_path": str(args.candidates),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
