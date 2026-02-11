#!/usr/bin/env python3
"""5-fold CV threshold sweep for GLiNER per-label and entailment thresholds.

Runs GLiNER once at threshold=0.01 to get the full raw entity pool, then sweeps
per-label score thresholds and entailment thresholds using cross-validation.

Two-step workflow (GLiNER needs compat venv, entailment needs MLX):

    # Step 1: GLiNER + label sweep in compat venv (caches raw pool to disk)
    scripts/run_gliner_compat.sh scripts/sweep_gliner_thresholds.py --no-entailment

    # Step 2: Entailment sweep in main venv (loads cached raw pool, has MLX)
    uv run python scripts/sweep_gliner_thresholds.py --entailment-only

Other usage:
    scripts/run_gliner_compat.sh scripts/sweep_gliner_thresholds.py --label-variants
    scripts/run_gliner_compat.sh scripts/sweep_gliner_thresholds.py --no-cache

Output:
    results/threshold_sweep.json  — recommended thresholds + diagnostics
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis.contacts.candidate_extractor import (
    DIRECT_LABEL_MAP,
    FACT_TYPE_RULES,
    NATURAL_LANGUAGE_LABELS,
    VAGUE,
    CandidateExtractor,
)
from scripts.run_extractor_bakeoff import (
    LABEL_ALIASES,
    Metrics,
    parse_context_messages,
    spans_match,
)

GOLD_PATH = Path("training_data/gliner_goldset/candidate_gold_merged_r4.json")
OUTPUT_PATH = Path("results/threshold_sweep.json")

# Labels with enough gold support (30+) to tune reliably
TUNABLE_LABELS = {"family_member", "health_condition", "activity", "place", "org"}

# Threshold sweep ranges
LABEL_THRESH_RANGE = np.arange(0.20, 0.81, 0.05)  # 0.20 to 0.80
ENTAILMENT_THRESH_RANGE = np.arange(0.01, 0.31, 0.02)  # 0.01 to 0.30

# Label prompt variants to test (Step 2 of plan)
LABEL_VARIANTS: dict[str, list[str]] = {
    "activity": [
        "hobby, sport, or activity",  # current
        "activity or hobby someone does regularly",
        "hobby, sport, game, or pastime",
    ],
    "person_name": [
        "person name",  # current
        "first name or nickname of a person",
        "proper name of a specific person",
    ],
    "place": [
        "place or location",  # current
        "city, town, country, or geographic location",
        "city, state, country, or neighborhood",
    ],
    "org": [
        "organization or company",  # current
        "company, school, university, or employer",
        "company, school, or organization name",
    ],
}


def load_goldset(path: Path) -> list[dict]:
    print(f"Loading goldset from {path}...", flush=True)
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records", flush=True)
    return data


def make_folds(records: list[dict], n_folds: int = 5) -> list[list[int]]:
    """Create stratified fold indices based on slice field."""
    slice_indices: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        slice_indices[rec.get("slice", "unknown")].append(i)

    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for _slice_name, indices in slice_indices.items():
        np.random.seed(42)
        shuffled = np.random.permutation(indices).tolist()
        for i, idx in enumerate(shuffled):
            folds[i % n_folds].append(idx)

    return folds


def resolve_fact_type_static(text: str, span: str, span_label: str) -> str:
    """Static version of _resolve_fact_type for threshold sweep."""
    import re

    for pattern, label_set, fact_type in FACT_TYPE_RULES:
        if span_label in label_set and re.search(pattern, text, re.IGNORECASE):
            return fact_type
    if span_label in DIRECT_LABEL_MAP:
        return DIRECT_LABEL_MAP[span_label]
    return "other_personal_fact"


def _raw_pool_cache_key(gold_path: Path, record_count: int) -> str:
    """Deterministic cache key from goldset path and record count."""
    h = hashlib.sha256(f"{gold_path.resolve()}:{record_count}:0.01".encode()).hexdigest()[:16]
    return h


def _load_raw_pool_cache(gold_path: Path, record_count: int) -> dict[int, list[dict]] | None:
    """Load cached raw pool if it exists and matches the current goldset."""
    cache_path = Path("results/raw_pool_cache.json")
    if not cache_path.exists():
        return None
    try:
        with open(cache_path) as f:
            data = json.load(f)
        if data.get("cache_key") != _raw_pool_cache_key(gold_path, record_count):
            print("  Cache key mismatch, rebuilding...", flush=True)
            return None
        # JSON keys are strings, convert back to int
        pool = {int(k): v for k, v in data["pool"].items()}
        print(f"  Loaded raw pool from cache ({len(pool)} records)", flush=True)
        return pool
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"  Cache load failed ({e}), rebuilding...", flush=True)
        return None


def _save_raw_pool_cache(
    pool: dict[int, list[dict]],
    gold_path: Path,
    record_count: int,
) -> None:
    """Save raw pool to disk cache."""
    cache_path = Path("results/raw_pool_cache.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "cache_key": _raw_pool_cache_key(gold_path, record_count),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "record_count": record_count,
        "pool": {str(k): v for k, v in pool.items()},
    }
    with open(cache_path, "w") as f:
        json.dump(data, f)
    size_mb = cache_path.stat().st_size / 1024 / 1024
    print(f"  Saved raw pool cache ({size_mb:.1f}MB)", flush=True)


def collect_raw_pool(
    extractor: CandidateExtractor,
    records: list[dict],
    batch_size: int = 32,
) -> dict[int, list[dict]]:
    """Run GLiNER at threshold=0.01 on all records using batch inference.

    Returns msg_id -> raw entities.
    """
    # Pre-build merged texts and bounds for all records
    merged_texts: list[str] = []
    current_bounds: list[tuple[int, int]] = []
    for rec in records:
        merged_text, current_start, current_end = extractor._build_context_text(
            rec["message_text"],
            prev_messages=parse_context_messages(rec.get("context_prev")),
            next_messages=parse_context_messages(rec.get("context_next")),
        )
        merged_texts.append(merged_text)
        current_bounds.append((current_start, current_end))

    # Load model once before batching
    use_mlx = extractor._use_mlx()
    if use_mlx:
        extractor._load_mlx_model()
    else:
        extractor._load_model()

    raw_pool: dict[int, list[dict]] = {}
    total = len(records)
    start = time.time()

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_texts = merged_texts[batch_start:batch_end]
        batch_bounds = current_bounds[batch_start:batch_end]
        batch_records = records[batch_start:batch_end]

        # Batch GLiNER inference
        if use_mlx:
            batch_entities = extractor._mlx_model.predict_batch(
                batch_texts,
                extractor._model_labels,
                batch_size=len(batch_texts),
                threshold=0.01,
                flat_ner=True,
            )
        else:
            batch_entities = extractor._model.batch_predict_entities(
                batch_texts,
                extractor._model_labels,
                threshold=0.01,
                flat_ner=True,
            )

        # Process each record's entities
        for j, (rec, bounds, ents) in enumerate(zip(batch_records, batch_bounds, batch_entities)):
            current_start, current_end = bounds
            msg_id = rec["message_id"]
            current_text = rec["message_text"]

            normalized: list[dict] = []
            for entity in ents:
                projected = extractor._project_entity_to_current(
                    entity,
                    current_start=current_start,
                    current_end=current_end,
                    current_text=current_text,
                )
                if projected is None:
                    continue
                span_text, start_char, end_char = projected
                item = dict(entity)
                raw_label = str(item.get("label", ""))
                item["raw_label"] = raw_label
                item["label"] = extractor._canonicalize_label(raw_label)
                item["text"] = span_text
                item["start"] = start_char
                item["end"] = end_char
                item["fact_type"] = resolve_fact_type_static(current_text, span_text, item["label"])
                normalized.append(item)

            raw_pool[msg_id] = normalized

        # Progress
        elapsed = time.time() - start
        done = min(batch_end, total)
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(
            f"  GLiNER batch inference: {done}/{total} ({elapsed:.0f}s, ETA {eta:.0f}s)",
            flush=True,
        )

    elapsed = time.time() - start
    total_ents = sum(len(v) for v in raw_pool.values())
    print(f"  GLiNER inference done in {elapsed:.1f}s ({total_ents} raw entities)", flush=True)
    return raw_pool


def compute_fold_metrics(
    gold_records: list[dict],
    raw_pool: dict[int, list[dict]],
    label: str,
    threshold: float,
) -> Metrics:
    """Compute metrics for a single label at a threshold on a set of records."""
    m = Metrics()

    for rec in gold_records:
        msg_id = rec["message_id"]
        gold_cands = rec.get("expected_candidates") or []
        raw_ents = raw_pool.get(msg_id, [])

        # Filter raw pool by label + threshold + basic filters
        preds = []
        seen: set[str] = set()
        for ent in raw_ents:
            ent_label = str(ent.get("label", ""))
            ent_text = str(ent.get("text", "")).strip()
            ent_score = float(ent.get("score", 0.0))
            ent_fact_type = ent.get("fact_type", "other_personal_fact")

            if ent_label != label:
                continue
            if ent_score < threshold:
                continue
            if ent_text.casefold() in VAGUE or len(ent_text) < 2:
                continue
            if ent_fact_type == "other_personal_fact":
                continue

            dedup_key = ent_text.casefold()
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            preds.append(
                {
                    "span_text": ent_text,
                    "span_label": ent_label,
                    "fact_type": ent_fact_type,
                    "score": ent_score,
                }
            )

        # Match against gold for this label only
        label_gold = [
            gc
            for gc in gold_cands
            if gc["span_label"] == label
            or label in LABEL_ALIASES
            and gc["span_label"] in LABEL_ALIASES.get(label, set())
        ]

        gold_matched = [False] * len(label_gold)
        pred_matched = [False] * len(preds)

        for gi, gc in enumerate(label_gold):
            for pi, pc in enumerate(preds):
                if pred_matched[pi]:
                    continue
                if spans_match(
                    pc["span_text"],
                    pc["span_label"],
                    gc["span_text"],
                    gc["span_label"],
                ):
                    gold_matched[gi] = True
                    pred_matched[pi] = True
                    m.tp += 1
                    break

        m.fn += sum(1 for matched in gold_matched if not matched)
        m.fp += sum(1 for matched in pred_matched if not matched)

    return m


def sweep_label_thresholds(
    records: list[dict],
    raw_pool: dict[int, list[dict]],
    folds: list[list[int]],
) -> dict[str, dict]:
    """5-fold CV sweep of per-label score thresholds."""
    print("\n=== Per-Label Threshold Sweep (5-fold CV) ===", flush=True)
    results: dict[str, dict] = {}

    for label in sorted(TUNABLE_LABELS):
        print(f"\n  Label: {label}", flush=True)
        fold_results: dict[float, list[dict]] = defaultdict(list)

        for fold_idx, test_indices in enumerate(folds):
            test_records = [records[i] for i in test_indices]

            for thresh in LABEL_THRESH_RANGE:
                m = compute_fold_metrics(test_records, raw_pool, label, float(thresh))
                fold_results[float(thresh)].append(m.to_dict())

        # Average across folds
        best_thresh = 0.0
        best_f1 = 0.0
        thresh_details = {}

        for thresh, fold_metrics in sorted(fold_results.items()):
            avg_p = np.mean([fm["precision"] for fm in fold_metrics])
            avg_r = np.mean([fm["recall"] for fm in fold_metrics])
            avg_f1 = np.mean([fm["f1"] for fm in fold_metrics])
            total_tp = sum(fm["tp"] for fm in fold_metrics)
            total_fp = sum(fm["fp"] for fm in fold_metrics)
            total_fn = sum(fm["fn"] for fm in fold_metrics)

            thresh_details[f"{thresh:.2f}"] = {
                "avg_precision": round(float(avg_p), 4),
                "avg_recall": round(float(avg_r), 4),
                "avg_f1": round(float(avg_f1), 4),
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": total_fn,
            }

            if avg_f1 > best_f1:
                best_f1 = float(avg_f1)
                best_thresh = thresh

        results[label] = {
            "recommended_threshold": round(best_thresh, 2),
            "best_cv_f1": round(best_f1, 4),
            "sweep_details": thresh_details,
        }

        print(
            f"    Best: threshold={best_thresh:.2f}, CV F1={best_f1:.4f}",
            flush=True,
        )

    return results


def sweep_entailment_thresholds(
    records: list[dict],
    raw_pool: dict[int, list[dict]],
    folds: list[list[int]],
    label_thresholds: dict[str, float],
) -> dict[str, dict]:
    """Sweep entailment thresholds per fact_type.

    Uses NLI model to score (premise, hypothesis) pairs, then sweeps
    the accept threshold per fact_type.
    """
    print("\n=== Entailment Threshold Sweep ===", flush=True)

    from jarvis.contacts.candidate_extractor import CandidateExtractor as CE

    # Build a dummy extractor to access hypothesis templates
    dummy = CE(use_entailment=False)

    # Collect all (premise, hypothesis, fact_type, fold_idx, record_idx, is_tp) tuples
    print("  Collecting entailment pairs...", flush=True)
    pairs_info: list[dict] = []

    for fold_idx, test_indices in enumerate(folds):
        for rec_idx in test_indices:
            rec = records[rec_idx]
            msg_id = rec["message_id"]
            raw_ents = raw_pool.get(msg_id, [])
            gold_cands = rec.get("expected_candidates") or []

            # Apply label thresholds to get filtered preds
            seen: set[tuple[str, str]] = set()
            for ent in raw_ents:
                ent_label = str(ent.get("label", ""))
                ent_text = str(ent.get("text", "")).strip()
                ent_score = float(ent.get("score", 0.0))
                ent_fact_type = ent.get("fact_type", "other_personal_fact")

                thresh = label_thresholds.get(ent_label, 0.55)
                if ent_score < thresh:
                    continue
                if ent_text.casefold() in VAGUE or len(ent_text) < 2:
                    continue
                if ent_fact_type == "other_personal_fact":
                    continue

                dedup_key = (ent_text.casefold(), ent_label)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                # Check if this pred matches any gold
                is_tp = any(
                    spans_match(ent_text, ent_label, gc["span_text"], gc["span_label"])
                    for gc in gold_cands
                )

                # Build hypothesis
                from jarvis.contacts.candidate_extractor import FactCandidate

                fake_candidate = FactCandidate(
                    message_id=msg_id,
                    span_text=ent_text,
                    span_label=ent_label,
                    gliner_score=ent_score,
                    fact_type=ent_fact_type,
                    start_char=0,
                    end_char=len(ent_text),
                    source_text=rec["message_text"],
                )
                hypothesis = dummy._candidate_to_hypothesis(fake_candidate)

                pairs_info.append(
                    {
                        "premise": rec["message_text"],
                        "hypothesis": hypothesis,
                        "fact_type": ent_fact_type,
                        "fold_idx": fold_idx,
                        "is_tp": is_tp,
                    }
                )

    print(f"  {len(pairs_info)} entailment pairs to score", flush=True)

    if not pairs_info:
        print("  No pairs to evaluate!", flush=True)
        return {}

    # Batch score all pairs
    print("  Running NLI scoring...", flush=True)
    try:
        from jarvis.nlp.entailment import verify_entailment_batch

        all_pairs = [(p["premise"], p["hypothesis"]) for p in pairs_info]
        # Score with very low threshold to get all scores
        nli_results = verify_entailment_batch(all_pairs, threshold=0.0)
    except (ImportError, ModuleNotFoundError) as e:
        print(f"  Skipping entailment sweep: {e}", flush=True)
        return {}

    for i, (_, score) in enumerate(nli_results):
        pairs_info[i]["nli_score"] = score

    print("  NLI scoring done", flush=True)

    # Group by fact_type and sweep thresholds
    fact_type_groups: dict[str, list[dict]] = defaultdict(list)
    for info in pairs_info:
        fact_type_groups[info["fact_type"]].append(info)

    results: dict[str, dict] = {}
    for fact_type in sorted(fact_type_groups.keys()):
        items = fact_type_groups[fact_type]
        total_tp = sum(1 for it in items if it["is_tp"])
        total_fp = sum(1 for it in items if not it["is_tp"])

        if total_tp < 3:
            print(f"  {fact_type}: skipping (only {total_tp} TPs)", flush=True)
            continue

        best_thresh = 0.0
        best_f1 = 0.0
        thresh_details = {}

        for thresh in ENTAILMENT_THRESH_RANGE:
            tp = sum(1 for it in items if it["is_tp"] and it["nli_score"] > thresh)
            fp = sum(1 for it in items if not it["is_tp"] and it["nli_score"] > thresh)
            fn = total_tp - tp  # TPs that were rejected by threshold

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            thresh_details[f"{float(thresh):.2f}"] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)

        results[fact_type] = {
            "recommended_threshold": round(best_thresh, 2),
            "best_f1": round(best_f1, 4),
            "support_tp": total_tp,
            "support_fp": total_fp,
            "sweep_details": thresh_details,
        }

        print(
            f"  {fact_type}: best_thresh={best_thresh:.2f}, "
            f"F1={best_f1:.4f} (TP={total_tp}, FP={total_fp})",
            flush=True,
        )

    return results


def sweep_label_variants(
    records: list[dict],
    folds: list[list[int]],
) -> dict[str, dict]:
    """Test alternative GLiNER prompt labels and measure raw pool recall."""
    print("\n=== Label Prompt Variant Sweep ===", flush=True)
    results: dict[str, dict] = {}

    for label, variants in sorted(LABEL_VARIANTS.items()):
        print(f"\n  Label: {label}", flush=True)
        variant_results = {}

        for variant_idx, variant_prompt in enumerate(variants):
            print(f"    Variant {variant_idx}: '{variant_prompt}'", flush=True)

            # Create extractor with this variant
            custom_labels = dict(NATURAL_LANGUAGE_LABELS)
            custom_labels[label] = variant_prompt

            ext = CandidateExtractor(
                global_threshold=0.01,
                use_entailment=False,
                backend="pytorch",
            )
            # Monkey-patch the label mapping for this variant
            ext._model_labels = [custom_labels.get(lbl, lbl) for lbl in ext._labels_canonical]
            ext._label_to_canonical = {
                custom_labels.get(lbl, lbl): lbl for lbl in ext._labels_canonical
            }
            ext._label_to_canonical.update({lbl: lbl for lbl in ext._labels_canonical})

            # Collect raw pool for positive records only (faster)
            positive_records = [r for r in records if r.get("slice") == "positive"]
            raw_pool = collect_raw_pool(ext, positive_records)

            # Compute recall at low threshold for this label
            total_gold = 0
            total_found = 0
            for rec in positive_records:
                msg_id = rec["message_id"]
                gold_cands = rec.get("expected_candidates") or []
                raw_ents = raw_pool.get(msg_id, [])

                label_gold = [
                    gc
                    for gc in gold_cands
                    if gc["span_label"] == label
                    or gc["span_label"] in LABEL_ALIASES.get(label, set())
                ]

                for gc in label_gold:
                    total_gold += 1
                    found = any(
                        spans_match(
                            str(ent.get("text", "")),
                            str(ent.get("label", "")),
                            gc["span_text"],
                            gc["span_label"],
                        )
                        for ent in raw_ents
                    )
                    if found:
                        total_found += 1

            recall = total_found / total_gold if total_gold > 0 else 0.0
            variant_results[variant_prompt] = {
                "raw_pool_recall": round(recall, 4),
                "gold_found": total_found,
                "gold_total": total_gold,
            }
            print(
                f"      Raw pool recall: {recall:.4f} ({total_found}/{total_gold})",
                flush=True,
            )

        # Pick best variant
        best = max(variant_results.items(), key=lambda x: x[1]["raw_pool_recall"])
        results[label] = {
            "best_prompt": best[0],
            "best_recall": best[1]["raw_pool_recall"],
            "variants": variant_results,
        }
        print(f"    Best: '{best[0]}' (recall={best[1]['raw_pool_recall']:.4f})", flush=True)

    return results


def compute_diagnostics(
    records: list[dict],
    raw_pool: dict[int, list[dict]],
) -> dict[str, Any]:
    """Compute diagnostic info: gate suppression, fact_type mapping gaps."""
    from jarvis.contacts.fact_filter import is_fact_likely

    gate_suppressed = 0
    gate_suppressed_with_gold = 0
    fact_type_fallback = 0
    fact_type_fallback_with_gold = 0

    for rec in records:
        msg_id = rec["message_id"]
        gold_cands = rec.get("expected_candidates") or []
        has_gold = len(gold_cands) > 0

        # Check gate suppression
        if not is_fact_likely(rec["message_text"], is_from_me=rec.get("is_from_me", False)):
            gate_suppressed += 1
            if has_gold:
                gate_suppressed_with_gold += 1

        # Check fact_type mapping gaps
        raw_ents = raw_pool.get(msg_id, [])
        for ent in raw_ents:
            if ent.get("fact_type") == "other_personal_fact":
                fact_type_fallback += 1
                if has_gold:
                    fact_type_fallback_with_gold += 1

    positive_records = sum(1 for r in records if r.get("slice") == "positive")

    return {
        "total_records": len(records),
        "positive_records": positive_records,
        "gate_suppressed_total": gate_suppressed,
        "gate_suppressed_with_gold": gate_suppressed_with_gold,
        "gate_suppression_rate": round(gate_suppressed / len(records), 4),
        "fact_type_fallback_total": fact_type_fallback,
        "fact_type_fallback_with_gold": fact_type_fallback_with_gold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="5-fold CV threshold sweep for GLiNER")
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--no-entailment", action="store_true", help="Skip entailment sweep")
    parser.add_argument(
        "--entailment-only",
        action="store_true",
        help="Run only entailment sweep (requires cached raw pool)",
    )
    parser.add_argument(
        "--label-variants", action="store_true", help="Run label prompt variant sweep"
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit records for testing")
    parser.add_argument("--no-cache", action="store_true", help="Force rebuild raw pool")
    args = parser.parse_args()

    if args.entailment_only and args.no_entailment:
        parser.error("--entailment-only and --no-entailment are mutually exclusive")

    records = load_goldset(args.gold)
    if args.limit:
        records = records[: args.limit]

    folds = make_folds(records)
    for i, fold in enumerate(folds):
        slices = defaultdict(int)
        for idx in fold:
            slices[records[idx].get("slice", "unknown")] += 1
        print(f"  Fold {i}: {len(fold)} records ({dict(slices)})", flush=True)

    # Step 1: Collect raw pool (with disk cache)
    print("\n=== Collecting Raw Entity Pool (threshold=0.01) ===", flush=True)
    raw_pool = None
    if not args.no_cache:
        raw_pool = _load_raw_pool_cache(args.gold, len(records))

    if raw_pool is None:
        if args.entailment_only:
            print("ERROR: --entailment-only requires a cached raw pool.", flush=True)
            print("Run GLiNER sweep first in compat venv:", flush=True)
            print(
                "  scripts/run_gliner_compat.sh scripts/sweep_gliner_thresholds.py --no-entailment",
                flush=True,
            )
            sys.exit(1)
        ext = CandidateExtractor(
            global_threshold=0.01,
            use_entailment=False,
            backend="pytorch",
        )
        raw_pool = collect_raw_pool(ext, records)
        _save_raw_pool_cache(raw_pool, args.gold, len(records))

    # Step 2: Diagnostics
    diagnostics: dict[str, Any] = {}
    if not args.entailment_only:
        print("\n=== Diagnostics ===", flush=True)
        diagnostics = compute_diagnostics(records, raw_pool)
        print(
            f"  Gate suppressed: {diagnostics['gate_suppressed_total']} "
            f"({diagnostics['gate_suppressed_with_gold']} with gold)",
            flush=True,
        )
        print(
            f"  Fact type fallback: {diagnostics['fact_type_fallback_total']} "
            f"({diagnostics['fact_type_fallback_with_gold']} with gold)",
            flush=True,
        )

    # Step 3: Per-label threshold sweep (always needed for entailment thresholds)
    label_results = sweep_label_thresholds(records, raw_pool, folds)

    # Collect recommended thresholds for entailment sweep
    recommended_thresholds = {
        label: info["recommended_threshold"] for label, info in label_results.items()
    }
    # Fill in defaults for non-tunable labels
    from jarvis.contacts.candidate_extractor import PER_LABEL_MIN

    for label, default_thresh in PER_LABEL_MIN.items():
        if label not in recommended_thresholds:
            recommended_thresholds[label] = default_thresh

    # Step 4: Entailment threshold sweep
    entailment_results: dict[str, dict] = {}
    if not args.no_entailment:
        entailment_results = sweep_entailment_thresholds(
            records,
            raw_pool,
            folds,
            recommended_thresholds,
        )

    # Step 5: Label variant sweep (optional, slow)
    label_variant_results: dict[str, dict] = {}
    if args.label_variants and not args.entailment_only:
        label_variant_results = sweep_label_variants(records, folds)

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_records": len(records),
        "num_folds": len(folds),
        "diagnostics": diagnostics,
        "label_thresholds": label_results,
        "recommended_per_label_min": recommended_thresholds,
        "entailment_thresholds": entailment_results,
        "label_variants": label_variant_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}", flush=True)

    # Print summary
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY: Recommended Thresholds", flush=True)
    print("=" * 60, flush=True)

    print("\nPER_LABEL_MIN = {", flush=True)
    for label, thresh in sorted(recommended_thresholds.items()):
        cv_f1 = label_results.get(label, {}).get("best_cv_f1", "n/a")
        marker = " # CV-tuned" if label in label_results else ""
        print(f'    "{label}": {thresh:.2f},  # F1={cv_f1}{marker}', flush=True)
    print("}", flush=True)

    if entailment_results:
        print("\n_ENTAILMENT_THRESHOLDS = {", flush=True)
        for ft, info in sorted(entailment_results.items()):
            print(
                f'    "{ft}": {info["recommended_threshold"]:.2f},  '
                f"# F1={info['best_f1']:.4f} (TP={info['support_tp']})",
                flush=True,
            )
        print("}", flush=True)

    if label_variant_results:
        print("\nBest label prompts:", flush=True)
        for label, info in sorted(label_variant_results.items()):
            print(
                f'  {label}: "{info["best_prompt"]}" (recall={info["best_recall"]:.4f})',
                flush=True,
            )


if __name__ == "__main__":
    main()
