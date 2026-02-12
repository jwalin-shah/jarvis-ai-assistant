#!/usr/bin/env python3
"""Extractor bakeoff evaluation script.

Evaluates multiple extractor backends (GLiNER, GLiNER2, NuExtract, spaCy, Ensemble) against
the same frozen goldset and produces a comparative report.

Usage:
    uv run python scripts/run_extractor_bakeoff.py --gold PATH
    uv run python scripts/run_extractor_bakeoff.py --extractors gliner,spacy,ensemble
    uv run python scripts/run_extractor_bakeoff.py --output-json results.json

Output:
    - Printed comparison report
    - JSON results file with detailed metrics per extractor
    - Per-extractor metrics files for further analysis
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
from typing import Any

# Adjust path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval_shared import DEFAULT_LABEL_ALIASES, spans_match
from gliner_shared import parse_context_messages

from jarvis.contacts.candidate_extractor import CandidateExtractor

logger = logging.getLogger(__name__)

DEFAULT_GOLD_PATH = Path("training_data/goldset_v6/dev.json")
OUTPUT_DIR = Path("results/extractor_bakeoff")

# Named extractor configs: map short names to model_name + backend overrides.
# All use the standard gliner package (PyTorch) unless overridden.
EXTRACTOR_CONFIGS: dict[str, dict[str, Any]] = {
    "gliner": {
        "model_name": "urchade/gliner_medium-v2.1",
    },
    "gliner-small": {
        "model_name": "urchade/gliner_small-v2.1",
    },
    # Knowledgator ModernBERT bi-encoder series (v2.0)
    # Bi-encoder architecture: ModernBERT text encoder + BGE label encoder
    # Up to 4x faster than DeBERTa-based models, 8192 token context
    "knowledgator-edge": {
        "model_name": "knowledgator/gliner-bi-edge-v2.0",
    },
    "knowledgator-small": {
        "model_name": "knowledgator/gliner-bi-small-v2.0",
    },
    "knowledgator-base": {
        "model_name": "knowledgator/gliner-bi-base-v2.0",
    },
    "knowledgator-large": {
        "model_name": "knowledgator/gliner-bi-large-v2.0",
    },
}


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("extractor_bakeoff.log", mode="w"),
        ],
    )


def load_gold_set(gold_path: Path) -> list[dict]:
    """Load gold labeled dataset."""
    logger.info(f"Loading gold set from {gold_path}")
    with open(gold_path) as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} gold records")
    return data


@dataclass
class Metrics:
    """Metrics container."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    total_time_ms: float = 0.0

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
    def f05(self) -> float:
        """F0.5 score - weights precision twice as much as recall."""
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        # F-beta: (1 + beta^2) * (p * r) / (beta^2 * p + r)
        # beta = 0.5 means precision is weighted 2x
        beta_sq = 0.25
        return (1 + beta_sq) * (p * r) / (beta_sq * p + r)

    @property
    def support(self) -> int:
        return self.tp + self.fn

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "f05": round(self.f05, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "support": self.support,
            "avg_time_ms": round(self.total_time_ms / max(self.support, 1), 2),
        }


def compute_metrics(
    gold_records: list[dict],
    predictions: dict[int, list[dict]],
    timing: dict[int, float],
) -> dict:
    """Compute metrics for an extractor run."""
    overall = Metrics()
    per_label: dict[str, Metrics] = defaultdict(Metrics)
    per_type: dict[str, Metrics] = defaultdict(Metrics)
    per_slice: dict[str, Metrics] = defaultdict(Metrics)
    errors: list[dict] = []

    for rec in gold_records:
        msg_id = rec["message_id"]
        gold_cands = rec.get("expected_candidates") or []
        pred_cands = predictions.get(msg_id, [])
        slc = rec.get("slice", "unknown")

        # Track timing
        if msg_id in timing:
            overall.total_time_ms += timing[msg_id]
            per_slice[slc].total_time_ms += timing[msg_id]

        gold_matched = [False] * len(gold_cands)
        pred_matched = [False] * len(pred_cands)

        # Match predictions to gold
        for gi, gc in enumerate(gold_cands):
            for pi, pc in enumerate(pred_cands):
                if pred_matched[pi]:
                    continue
                if spans_match(
                    pc.get("span_text", ""),
                    pc.get("span_label", ""),
                    gc.get("span_text", ""),
                    gc.get("span_label", ""),
                    label_aliases=DEFAULT_LABEL_ALIASES,
                ):
                    gold_matched[gi] = True
                    pred_matched[pi] = True
                    overall.tp += 1
                    per_label[gc["span_label"]].tp += 1
                    per_type[gc.get("fact_type", "unknown")].tp += 1
                    per_slice[slc].tp += 1
                    break

        # Unmatched gold = FN
        for gi, gc in enumerate(gold_cands):
            if not gold_matched[gi]:
                overall.fn += 1
                per_label[gc["span_label"]].fn += 1
                per_type[gc.get("fact_type", "unknown")].fn += 1
                per_slice[slc].fn += 1
                errors.append(
                    {
                        "type": "fn",
                        "sample_id": rec.get("sample_id", ""),
                        "slice": slc,
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
                        "sample_id": rec.get("sample_id", ""),
                        "slice": slc,
                        "pred_span": pc.get("span_text", ""),
                        "pred_label": label,
                    }
                )

    return {
        "overall": overall.to_dict(),
        "per_label": {k: v.to_dict() for k, v in sorted(per_label.items())},
        "per_type": {k: v.to_dict() for k, v in sorted(per_type.items())},
        "per_slice": {k: v.to_dict() for k, v in sorted(per_slice.items())},
        "errors": errors[:100],  # Limit errors stored
    }


def _run_regex_extractor(
    gold_records: list[dict],
    config: dict[str, Any] | None = None,
    limit: int | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Run regex FactExtractor against the gold set and return bakeoff metrics."""
    from regex_to_span_adapter import facts_to_spans

    from jarvis.contacts.fact_extractor import FactExtractor

    extractor = FactExtractor(confidence_threshold=0.3)  # Low threshold; let eval measure quality

    records = gold_records[:limit] if limit else gold_records
    logger.info(f"Running regex extractor on {len(records)} messages...")

    predictions: dict[int, list[dict]] = {}
    timing: dict[int, float] = {}

    out_dir = output_dir or OUTPUT_DIR
    incremental_path = out_dir / "regex_predictions.jsonl"
    incremental_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    with open(incremental_path, "w") as inc_f:
        for i, rec in enumerate(records):
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(records) - i - 1) / rate if rate > 0 else 0
                logger.info(
                    f"  Processed {i + 1}/{len(records)} messages "
                    f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)"
                )

            msg_id = rec["message_id"]
            msg_text = rec["message_text"]

            msg_start = time.perf_counter()
            try:
                msg_dict = {"text": msg_text, "id": msg_id}
                facts = extractor.extract_facts([msg_dict])
                elapsed_ms = (time.perf_counter() - msg_start) * 1000

                pred_list = facts_to_spans(facts, msg_text)
                predictions[msg_id] = pred_list
                timing[msg_id] = elapsed_ms
            except (ValueError, RuntimeError, OSError) as e:
                logger.error(f"Regex extraction failed for message {msg_id}: {e}")
                predictions[msg_id] = []
                timing[msg_id] = 0

            inc_f.write(
                json.dumps(
                    {
                        "message_id": msg_id,
                        "predictions": predictions[msg_id],
                        "elapsed_ms": timing[msg_id],
                    }
                )
                + "\n"
            )
            inc_f.flush()

    total_time = time.time() - start_time
    total_spans = sum(len(v) for v in predictions.values())
    msgs_with_preds = sum(1 for v in predictions.values() if v)
    logger.info(
        f"Regex extraction complete in {total_time:.1f}s: "
        f"{total_spans} spans from {msgs_with_preds}/{len(records)} messages"
    )

    metrics = compute_metrics(records, predictions, timing)
    metrics["extractor_name"] = "regex"
    metrics["config"] = config or {}
    metrics["num_messages"] = len(records)
    metrics["total_time_s"] = round(total_time, 2)
    metrics["ms_per_message"] = round(total_time / len(records) * 1000, 1) if records else 0
    return metrics


def run_extractor(
    extractor_name: str,
    gold_records: list[dict],
    config: dict[str, Any] | None = None,
    limit: int | None = None,
    args_output_dir: Path | None = None,
) -> dict:
    """Run an extractor against the gold set. Dispatches to regex or GLiNER."""
    if extractor_name == "regex":
        return _run_regex_extractor(gold_records, config, limit, args_output_dir)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Running extractor: {extractor_name}")
    logger.info(f"{'=' * 60}")

    cfg = config or {}
    # Merge named extractor config (model_name etc.) with CLI overrides
    if extractor_name in EXTRACTOR_CONFIGS:
        named_cfg = dict(EXTRACTOR_CONFIGS[extractor_name])
        named_cfg.update(cfg)  # CLI overrides take priority
        cfg = named_cfg

    # Create CandidateExtractor with config overrides
    try:
        try:
            import mlx.core  # noqa: F401
            _has_mlx = True
        except ImportError:
            _has_mlx = False
        # Force PyTorch backend for non-default models (no MLX port)
        backend = cfg.get("backend", "auto")
        model_name = cfg.get("model_name", "urchade/gliner_medium-v2.1")
        if model_name != "urchade/gliner_medium-v2.1" and backend == "auto":
            backend = "pytorch"
        # Determine verification backend
        verifier = cfg.get("verifier", "none")
        use_entailment = verifier == "nli" and _has_mlx
        # LLM verifier runs as a separate post-extraction pass to avoid
        # memory pressure (GLiNER + LLM can't coexist on 8GB).
        use_llm_verifier_post = verifier == "llm"

        extractor = CandidateExtractor(
            model_name=model_name,
            label_profile=cfg.get("label_profile"),
            global_threshold=cfg.get("threshold", 0.35),
            use_entailment=use_entailment,
            use_llm_verifier=False,  # LLM runs post-extraction
            backend=backend,
        )
        logger.info("Created CandidateExtractor")
    except (ImportError, ValueError, TypeError) as e:
        logger.error(f"Failed to create extractor: {e}")
        return {"extractor_name": extractor_name, "error": str(e)}

    # Load model (prefer MLX on Apple Silicon, skip PyTorch preload to save memory)
    logger.info("Loading model...")
    try:
        if _has_mlx and backend in ("auto", "mlx"):
            extractor._load_mlx_model()
        else:
            extractor._load_model()
        logger.info("Model loaded")
    except (OSError, RuntimeError, ValueError) as e:
        logger.error(f"Failed to load model: {e}")
        return {"extractor_name": extractor_name, "error": str(e)}

    # Run extraction - write predictions incrementally to JSONL for crash safety
    records = gold_records[:limit] if limit else gold_records
    logger.info(f"Extracting from {len(records)} messages...")

    predictions: dict[int, list[dict]] = {}
    timing: dict[int, float] = {}
    total_raw_entities = 0
    # Store raw candidates for post-extraction LLM verification
    raw_candidates_by_msg: dict[int, list] = {} if use_llm_verifier_post else {}

    # Incremental output file - survives crashes
    out_dir = args_output_dir or OUTPUT_DIR
    incremental_path = out_dir / f"{extractor_name}_predictions.jsonl"
    incremental_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    with open(incremental_path, "w") as inc_f:
        for i, rec in enumerate(records):
            if (i + 1) % 50 == 0:
                elapsed_so_far = time.time() - start_time
                rate = (i + 1) / elapsed_so_far
                eta = (len(records) - i - 1) / rate if rate > 0 else 0
                logger.info(
                    f"  Processed {i + 1}/{len(records)} messages "
                    f"({elapsed_so_far:.0f}s elapsed, ETA {eta:.0f}s)"
                )

            msg_id = rec["message_id"]

            # Extract
            apply_thresh = cfg.get("apply_thresholds", True)
            use_gate = cfg.get("use_gate", True)
            msg_start = time.perf_counter()
            try:
                candidates = extractor.extract_candidates(
                    text=rec["message_text"],
                    message_id=msg_id,
                    is_from_me=rec.get("is_from_me"),
                    prev_messages=parse_context_messages(rec.get("context_prev")),
                    next_messages=parse_context_messages(rec.get("context_next")),
                    apply_label_thresholds=apply_thresh,
                    use_gate=use_gate,
                )
                elapsed = (time.perf_counter() - msg_start) * 1000

                if use_llm_verifier_post:
                    raw_candidates_by_msg[msg_id] = list(candidates)
                pred_list = [
                    {
                        "span_text": c.span_text,
                        "span_label": c.span_label,
                        "fact_type": c.fact_type,
                        "score": c.gliner_score,
                    }
                    for c in candidates
                ]
                predictions[msg_id] = pred_list
                total_raw_entities += len(pred_list)
                timing[msg_id] = elapsed

            except (ValueError, RuntimeError, OSError) as e:
                logger.error(f"Extraction failed for message {msg_id}: {e}")
                predictions[msg_id] = []
                timing[msg_id] = 0
                elapsed = 0

            # Write incrementally - one JSON line per message
            inc_f.write(
                json.dumps(
                    {
                        "message_id": msg_id,
                        "predictions": predictions[msg_id],
                        "elapsed_ms": timing[msg_id],
                    }
                )
                + "\n"
            )
            inc_f.flush()

    total_time = time.time() - start_time
    msgs_with_preds = sum(1 for v in predictions.values() if v)
    logger.info(
        f"Extraction complete in {total_time:.1f}s: "
        f"{total_raw_entities} entities from {msgs_with_preds}/{len(records)} messages"
    )

    # Post-extraction LLM verification (two-pass to avoid memory pressure)
    if use_llm_verifier_post and raw_candidates_by_msg:
        import gc

        # Unload GLiNER models to free memory for LLM
        logger.info("Unloading GLiNER models for LLM verification pass...")
        import jarvis.contacts.candidate_extractor as _ce_mod
        from jarvis.contacts.candidate_extractor import (
            _gliner_model_lock,
        )

        with _gliner_model_lock:
            _ce_mod._gliner_mlx_model = None
            _ce_mod._gliner_pytorch_model = None
        del extractor
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, AttributeError):
            pass
        try:
            import mlx.core as mx
            mx.clear_cache()
        except (ImportError, AttributeError):
            pass
        gc.collect()

        logger.info("Running LLM verification on %d messages...", len(raw_candidates_by_msg))
        from jarvis.contacts.llm_fact_verifier import LLMFactVerifier

        verifier_obj = LLMFactVerifier()
        all_candidates = []
        for msg_id, cands in raw_candidates_by_msg.items():
            all_candidates.extend(cands)

        llm_start = time.time()
        verified = verifier_obj.verify_candidates(all_candidates)
        llm_time = time.time() - llm_start
        logger.info(
            "LLM verification: %d -> %d candidates (%.1fs)",
            len(all_candidates),
            len(verified),
            llm_time,
        )

        # Rebuild predictions from verified candidates
        predictions.clear()
        for c in verified:
            pred_list = predictions.setdefault(c.message_id, [])
            pred_list.append({
                "span_text": c.span_text,
                "span_label": c.span_label,
                "fact_type": c.fact_type,
                "score": c.gliner_score,
            })
        total_raw_entities = len(verified)
        total_time += llm_time

    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(records, predictions, timing)

    # Add run metadata
    metrics["extractor_name"] = extractor_name
    metrics["config"] = config or {}
    metrics["num_messages"] = len(records)
    metrics["total_time_s"] = round(total_time, 2)
    metrics["ms_per_message"] = round(total_time / len(records) * 1000, 1) if records else 0

    return metrics


def print_comparison(results: list[dict]) -> None:
    """Print a comparison table of all extractors."""
    print("\n" + "=" * 80)
    print("EXTRACTOR BAKEOFF RESULTS")
    print("=" * 80)

    # Check if thresholds were disabled
    any_no_thresh = any(
        r.get("config", {}).get("apply_thresholds") is False for r in results if "error" not in r
    )
    if any_no_thresh:
        print("  [Per-label thresholds DISABLED - measuring raw model quality]")

    # Overall metrics table
    print(
        "\n{:20} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}".format(
            "Extractor", "P", "R", "F1", "F0.5", "Time/msg", "Support"
        )
    )
    print("-" * 80)

    for r in results:
        if "error" in r:
            print(f"{r['extractor_name']:<20} ERROR: {r['error'][:50]}")
            continue

        ov = r["overall"]
        print(
            "{:20} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.1f}ms {:>8}".format(
                r["extractor_name"],
                ov["precision"],
                ov["recall"],
                ov["f1"],
                ov["f05"],
                ov["avg_time_ms"],
                ov["support"],
            )
        )

    # Per-slice breakdown
    print("\n\nPer-Slice Breakdown:")
    for r in results:
        if "error" in r:
            continue

        print(f"\n{r['extractor_name']}:")
        print("  {:15} {:>8} {:>8} {:>8} {:>8}".format("Slice", "P", "R", "F1", "Support"))
        print("  " + "-" * 55)

        for slice_name, slice_metrics in sorted(r["per_slice"].items()):
            print(
                "  {:15} {:>8.3f} {:>8.3f} {:>8.3f} {:>8}".format(
                    slice_name[:15],
                    slice_metrics["precision"],
                    slice_metrics["recall"],
                    slice_metrics["f1"],
                    slice_metrics["support"],
                )
            )

    # Recommendations
    print("\n\nRecommendations:")
    print("-" * 80)

    # Find best by F0.5 (precision-weighted)
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        best_f05 = max(valid_results, key=lambda r: r["overall"]["f05"])
        best_f1 = max(valid_results, key=lambda r: r["overall"]["f1"])
        best_p = max(valid_results, key=lambda r: r["overall"]["precision"])
        best_r = max(valid_results, key=lambda r: r["overall"]["recall"])

        print(
            f"  Best F0.5 (precision-weighted): {best_f05['extractor_name']} "
            f"({best_f05['overall']['f05']:.3f})"
        )
        print(
            f"  Best F1 (balanced):             {best_f1['extractor_name']} "
            f"({best_f1['overall']['f1']:.3f})"
        )
        print(
            f"  Best Precision:                 {best_p['extractor_name']} "
            f"({best_p['overall']['precision']:.3f})"
        )
        print(
            f"  Best Recall:                    {best_r['extractor_name']} "
            f"({best_r['overall']['recall']:.3f})"
        )

        # Primary recommendation based on exit gate criteria
        # Phase 1 exit gate: F0.5 score + recall floor
        candidates = [
            r
            for r in valid_results
            if r["overall"]["recall"] >= 0.40  # Recall floor
        ]
        if candidates:
            primary = max(candidates, key=lambda r: r["overall"]["f05"])
            print(f"\n  PRIMARY RECOMMENDATION: {primary['extractor_name']}")
            print(f"    - F0.5: {primary['overall']['f05']:.3f}")
            print(f"    - Recall: {primary['overall']['recall']:.3f} (>= 0.40 floor)")
            print(f"    - Precision: {primary['overall']['precision']:.3f}")
        else:
            print("\n  No extractor meets the recall floor (0.40). Consider:")
            print("    - Tuning thresholds")
            print("    - Using ensemble methods")
            print("    - Collecting more training data")

    print("\n" + "=" * 80)


def main() -> None:
    """Main entry point."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Run extractor bakeoff evaluation")
    parser.add_argument(
        "--gold",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help="Path to gold set JSON",
    )
    parser.add_argument(
        "--extractors",
        type=str,
        default="gliner",
        help="Comma-separated extractor names: 'regex' (rule-based) or GLiNER label names",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for results",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path for combined JSON output (default: output_dir/bakeoff_results.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to first N messages",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override confidence threshold for all extractors",
    )
    parser.add_argument(
        "--no-thresholds",
        action="store_true",
        default=False,
        help="Disable per-label threshold filtering (measure raw model quality)",
    )
    parser.add_argument(
        "--label-profile",
        type=str,
        default=None,
        help="Override label profile (e.g., high_recall, balanced, high_precision)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override model name for compatible extractors (e.g., urchade/gliner_small-v2.1)",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        default=False,
        help="Disable message gate pre-filter (evaluate raw extractor quality)",
    )
    parser.add_argument(
        "--verifier",
        type=str,
        choices=["nli", "llm", "none"],
        default="nli",
        help="Verification backend: 'nli' (entailment, default), 'llm' (LLM verifier), 'none'",
    )

    args = parser.parse_args()

    # Validate gold path
    if not args.gold.exists():
        logger.error(f"Gold set not found: {args.gold}")
        sys.exit(1)

    # Parse extractors
    extractor_names = [e.strip() for e in args.extractors.split(",")]
    logger.info(f"Evaluating extractors: {', '.join(extractor_names)}")

    # Load gold set
    gold_records = load_gold_set(args.gold)

    # Run each extractor
    results: list[dict] = []

    for name in extractor_names:
        config = {}
        if args.threshold is not None:
            config["threshold"] = args.threshold
        if args.no_thresholds:
            config["apply_thresholds"] = False
        if args.label_profile is not None:
            config["label_profile"] = args.label_profile
        if args.model_name is not None:
            config["model_name"] = args.model_name
        if args.no_gate:
            config["use_gate"] = False
        if args.verifier:
            config["verifier"] = args.verifier

        result = run_extractor(
            name,
            gold_records,
            config=config,
            limit=args.limit,
            args_output_dir=args.output_dir,
        )
        results.append(result)

        # Save individual result
        args.output_dir.mkdir(parents=True, exist_ok=True)
        individual_path = args.output_dir / f"{name}_metrics.json"
        with open(individual_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved {name} metrics to {individual_path}")

    # Print comparison
    print_comparison(results)

    # Save combined results
    output_json = args.output_json or args.output_dir / "bakeoff_results.json"
    with open(output_json, "w") as f:
        json.dump(
            {
                "extractors": results,
                "gold_path": str(args.gold),
                "num_messages": len(gold_records[: args.limit] if args.limit else gold_records),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
        )
    logger.info(f"Saved combined results to {output_json}")

    logger.info("Bakeoff complete!")


if __name__ == "__main__":
    main()
