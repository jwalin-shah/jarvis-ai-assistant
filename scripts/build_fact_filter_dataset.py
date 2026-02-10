#!/usr/bin/env python3
"""Build candidate-level training data for the fact filter classifier.

This converts candidate gold records into JSONL rows expected by:
    scripts/train_fact_filter.py

Each output row has required keys:
    - text
    - candidate
    - entity_type
    - label

And additional metadata for debugging/analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path


def _setup_logging() -> logging.Logger:
    """Setup logging with file and stream handlers."""
    log_file = Path("build_fact_filter_dataset.log")
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(__name__)


def _safe_major(version: str) -> int:
    """Best-effort major version parser."""
    try:
        return int(version.split(".", 1)[0])
    except (TypeError, ValueError):
        return -1


def warn_runtime_stack() -> None:
    """Warn when runtime deps are known to degrade GLiNER extraction quality."""
    try:
        import huggingface_hub
        import transformers
    except Exception:
        return

    tver = getattr(transformers, "__version__", "unknown")
    hver = getattr(huggingface_hub, "__version__", "unknown")
    if _safe_major(str(tver)) >= 5:
        print(
            "WARNING: transformers="
            f"{tver}, huggingface_hub={hver}. GLiNER quality may degrade on this stack. "
            "Use scripts/run_gliner_compat.sh scripts/build_fact_filter_dataset.py ...",
            flush=True,
        )


def enforce_runtime_stack(allow_unstable_stack: bool) -> None:
    """Fail fast on unsupported runtime unless explicitly overridden."""
    try:
        import huggingface_hub
        import transformers
    except Exception:
        return

    tver = getattr(transformers, "__version__", "unknown")
    hver = getattr(huggingface_hub, "__version__", "unknown")
    if _safe_major(str(tver)) >= 5:
        msg = (
            f"Detected transformers={tver}, huggingface_hub={hver}. "
            "GLiNER quality may degrade on this stack."
        )
        if allow_unstable_stack:
            print(
                "WARNING: " + msg + " Continuing due --allow-unstable-stack.",
                flush=True,
            )
            return
        raise SystemExit(
            "ERROR: "
            + msg
            + " Re-run via scripts/run_gliner_compat.sh scripts/build_fact_filter_dataset.py ..."
            + " or pass --allow-unstable-stack."
        )


def parse_context_messages(raw: object) -> list[str]:
    """Parse context payloads from gold JSON (string blob or list) into message texts."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if not isinstance(raw, str):
        return []

    payload = raw.strip()
    if not payload:
        return []

    # CSV context format: "id|speaker|text || id|speaker|text".
    chunks = [c.strip() for c in payload.split("||") if c.strip()]
    messages: list[str] = []
    for chunk in chunks:
        parts = chunk.split("|", 2)
        if len(parts) == 3:
            text = parts[2].strip()
        else:
            text = chunk
        if text:
            messages.append(text)
    return messages


def jaccard_tokens(a: str, b: str) -> float:
    """Token-level Jaccard similarity (case-insensitive)."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def spans_match(pred_text: str, pred_label: str, gold_text: str, gold_label: str) -> bool:
    """Check if a predicted span matches a gold span."""
    if pred_label != gold_label:
        return False

    pl = pred_text.lower().strip()
    gl = gold_text.lower().strip()
    if pl in gl or gl in pl:
        return True
    if jaccard_tokens(pred_text, gold_text) >= 0.5:
        return True
    return False


def first_matching_gold(pred: dict, gold_candidates: list[dict]) -> dict | None:
    """Return the first matching gold candidate for a predicted candidate, else None."""
    for gold in gold_candidates:
        if spans_match(
            pred_text=pred.get("span_text", ""),
            pred_label=pred.get("span_label", ""),
            gold_text=gold.get("span_text", ""),
            gold_label=gold.get("span_label", ""),
        ):
            return gold
    return None


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write JSONL records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def split_by_sample(
    rows: list[dict],
    dev_frac: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Split rows into train/dev by sample_id to avoid message leakage."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["sample_id"], []).append(row)

    sample_ids = list(grouped.keys())
    pos_sample_ids = [sid for sid in sample_ids if any(r["label"] == 1 for r in grouped[sid])]
    neg_sample_ids = [sid for sid in sample_ids if sid not in set(pos_sample_ids)]

    rng = random.Random(seed)
    rng.shuffle(pos_sample_ids)
    rng.shuffle(neg_sample_ids)

    n_pos_dev = round(len(pos_sample_ids) * dev_frac)
    n_neg_dev = round(len(neg_sample_ids) * dev_frac)

    dev_ids = set(pos_sample_ids[:n_pos_dev] + neg_sample_ids[:n_neg_dev])

    train_rows: list[dict] = []
    dev_rows: list[dict] = []
    for sid, sample_rows in grouped.items():
        if sid in dev_ids:
            dev_rows.extend(sample_rows)
        else:
            train_rows.extend(sample_rows)

    rng.shuffle(train_rows)
    rng.shuffle(dev_rows)
    return train_rows, dev_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("training_data/gliner_goldset/candidate_gold.json"),
        help="Path to candidate gold JSON",
    )
    parser.add_argument(
        "--output-all",
        type=Path,
        default=Path("training_data/fact_candidates.jsonl"),
        help="Output JSONL path for all candidate rows",
    )
    parser.add_argument(
        "--output-train",
        type=Path,
        default=Path("training_data/fact_candidates_train.jsonl"),
        help="Output JSONL path for train split",
    )
    parser.add_argument(
        "--output-dev",
        type=Path,
        default=Path("training_data/fact_candidates_dev.jsonl"),
        help="Output JSONL path for dev split",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("training_data/fact_candidates_manifest.json"),
        help="Output path for manifest JSON",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="GLiNER extraction threshold",
    )
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
        "--no-label-min",
        action="store_true",
        help="Disable per-label minimum score thresholds",
    )
    parser.add_argument(
        "--no-vague-filter",
        action="store_true",
        help="Disable vague-span filter",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=0,
        help="Number of prev/next messages to include as context",
    )
    parser.add_argument(
        "--dev-frac",
        type=float,
        default=0.2,
        help="Fraction of sample_ids for dev split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N messages",
    )
    parser.add_argument(
        "--allow-unstable-stack",
        action="store_true",
        help="Allow running outside GLiNER compat runtime (not recommended)",
    )
    return parser.parse_args()


def main() -> None:
    logger = _setup_logging()
    args = parse_args()
    warn_runtime_stack()
    enforce_runtime_stack(args.allow_unstable_stack)

    if not args.gold.exists():
        raise SystemExit(f"Gold file not found: {args.gold}")

    from jarvis.contacts.candidate_extractor import CandidateExtractor, labels_for_profile

    records = json.loads(args.gold.read_text(encoding="utf-8"))
    active_labels = labels_for_profile(args.label_profile)
    if args.drop_label:
        drop_set = set(args.drop_label)
        active_labels = [lbl for lbl in active_labels if lbl not in drop_set]
    if not active_labels:
        raise SystemExit("No active labels left after applying label filters.")

    logger.info("Using label profile: %s", args.label_profile)
    logger.info("Active labels: %s", ", ".join(active_labels))
    print(f"Using label profile: {args.label_profile}", flush=True)
    print(f"Active labels: {', '.join(active_labels)}", flush=True)
    if args.context_window > 0:
        print(
            "WARNING: context_window>0 can reduce extraction quality due to GLiNER truncation; "
            "defaulting to 0 is recommended.",
            flush=True,
        )

    extractor = CandidateExtractor(labels=active_labels, label_profile=args.label_profile)

    all_rows: list[dict] = []
    total_gold = 0
    total_pred = 0
    total_pos = 0

    for idx, rec in enumerate(records, start=1):
        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(f"Processed {idx}/{len(records)} messages", flush=True)

        prev_messages: list[str] | None = None
        next_messages: list[str] | None = None
        if args.context_window > 0:
            prev_all = parse_context_messages(rec.get("context_prev"))
            next_all = parse_context_messages(rec.get("context_next"))
            prev_slice = prev_all[-args.context_window:] if prev_all else []
            next_slice = next_all[: args.context_window] if next_all else []
            prev_messages = prev_slice or None
            next_messages = next_slice or None

        preds = extractor.extract_candidates(
            text=rec["message_text"],
            message_id=int(rec["message_id"]),
            threshold=args.threshold,
            apply_label_thresholds=not args.no_label_min,
            apply_vague_filter=not args.no_vague_filter,
            prev_messages=prev_messages,
            next_messages=next_messages,
        )

        pred_dicts = [p.to_dict() for p in preds]
        gold_candidates = rec.get("expected_candidates") or []

        total_gold += len(gold_candidates)
        total_pred += len(pred_dicts)

        for pred in pred_dicts:
            matched_gold = first_matching_gold(pred, gold_candidates)
            label = 1 if matched_gold is not None else 0
            total_pos += label

            row = {
                "text": rec["message_text"],
                "candidate": pred.get("span_text", ""),
                "entity_type": pred.get("span_label", ""),
                "label": label,
                "fact_type": pred.get("fact_type", ""),
                "gliner_score": pred.get("gliner_score", 0.0),
                "sample_id": rec.get("sample_id", ""),
                "message_id": rec.get("message_id"),
                "slice": rec.get("slice", "unknown"),
                "gold_fact_type": (matched_gold or {}).get("fact_type", ""),
                "gold_span_text": (matched_gold or {}).get("span_text", ""),
                "gold_span_label": (matched_gold or {}).get("span_label", ""),
            }
            all_rows.append(row)

    if not all_rows:
        raise SystemExit("No candidate rows generated. Check thresholds/config.")

    train_rows, dev_rows = split_by_sample(rows=all_rows, dev_frac=args.dev_frac, seed=args.seed)

    write_jsonl(args.output_all, all_rows)
    write_jsonl(args.output_train, train_rows)
    write_jsonl(args.output_dev, dev_rows)

    negatives = len(all_rows) - total_pos
    manifest = {
        "gold_path": str(args.gold),
        "num_messages": len(records),
        "num_gold_candidates": total_gold,
        "num_predicted_candidates": total_pred,
        "num_rows": len(all_rows),
        "positives": total_pos,
        "negatives": negatives,
        "positive_rate": round(total_pos / len(all_rows), 4),
        "estimated_candidate_precision": round(total_pos / total_pred, 4) if total_pred else 0.0,
        "estimated_candidate_recall": round(total_pos / total_gold, 4) if total_gold else 0.0,
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "train_positives": sum(1 for r in train_rows if r["label"] == 1),
        "dev_positives": sum(1 for r in dev_rows if r["label"] == 1),
        "settings": {
            "threshold": args.threshold,
            "apply_label_min": not args.no_label_min,
            "apply_vague_filter": not args.no_vague_filter,
            "context_window": args.context_window,
            "label_profile": args.label_profile,
            "drop_labels": args.drop_label,
            "dev_frac": args.dev_frac,
            "seed": args.seed,
        },
    }

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("Built fact-filter candidate dataset")
    logger.info("  all:        %s (%d rows)", args.output_all, len(all_rows))
    logger.info("  train:      %s (%d rows)", args.output_train, len(train_rows))
    logger.info("  dev:        %s (%d rows)", args.output_dev, len(dev_rows))
    logger.info("  manifest:   %s", args.manifest)
    logger.info(
        "  labels:     pos=%d neg=%d (pos_rate=%.1f%%)",
        total_pos, negatives, total_pos / len(all_rows) * 100,
    )
    print("Built fact-filter candidate dataset", flush=True)
    print(f"  all:        {args.output_all} ({len(all_rows)} rows)", flush=True)
    print(f"  train:      {args.output_train} ({len(train_rows)} rows)", flush=True)
    print(f"  dev:        {args.output_dev} ({len(dev_rows)} rows)", flush=True)
    print(f"  manifest:   {args.manifest}", flush=True)
    print(
        "  labels:     "
        f"pos={total_pos} neg={negatives} (pos_rate={total_pos / len(all_rows):.1%})",
        flush=True,
    )


if __name__ == "__main__":
    main()
