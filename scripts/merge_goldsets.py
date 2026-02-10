#!/usr/bin/env python3
"""Merge fact gold sets and build chat-held-out train/dev splits.

This script fixes three common issues in iterative annotation merges:
1. sample_id collisions across rounds
2. train/dev leakage from same chat appearing in both splits
3. schema mismatch for training by exporting explicit message-gate JSONL

Outputs are written to training_data/fact_goldset_merged/ by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _setup_logging() -> None:
    """Configure logging with FileHandler + StreamHandler."""
    log_file = Path("merge_goldsets.log")
    file_handler = logging.FileHandler(log_file, mode="a")
    stream_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[file_handler, stream_handler],
    )


@dataclass(frozen=True)
class ChatStats:
    """Aggregate stats for one chat group."""

    chat_rowid: str
    total: int
    positives: int


DEFAULT_V1 = Path("training_data/fact_goldset/fact_goldset_v1_frozen.csv")
DEFAULT_R2 = Path("training_data/fact_goldset_round2/fact_goldset_400.csv")
DEFAULT_OUT_DIR = Path("training_data/fact_goldset_merged")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge labeled fact goldsets with chat-held-out train/dev split"
    )
    parser.add_argument("--v1", type=Path, default=DEFAULT_V1, help="Path to v1 labeled CSV")
    parser.add_argument("--r2", type=Path, default=DEFAULT_R2, help="Path to round2 labeled CSV")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--dev-frac",
        type=float,
        default=0.20,
        help="Target dev fraction (chat-held-out)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--search-iters",
        type=int,
        default=3000,
        help="Randomized search iterations for best chat split",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def require_files(v1: Path, r2: Path) -> None:
    missing = [str(p) for p in (v1, r2) if not p.exists()]
    if missing:
        print(f"ERROR: missing input files: {missing}", file=sys.stderr, flush=True)
        raise SystemExit(1)


def ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")


def validate_row(row: dict[str, str], source: str) -> None:
    keep = row.get("gold_keep", "")
    sample_id = row.get("sample_id", "<missing>")
    if keep not in {"0", "1"}:
        raise ValueError(
            f"Row {sample_id} from {source} has invalid gold_keep={keep!r}; expected '0' or '1'"
        )


def normalize_source_rows(rows: list[dict[str, str]], source: str) -> list[dict[str, str]]:
    """Namespace sample IDs by source to avoid collisions across rounds."""
    out: list[dict[str, str]] = []
    for row in rows:
        validate_row(row, source)
        normalized = dict(row)
        original_id = normalized.get("sample_id", "")
        normalized["source_round"] = source
        normalized["sample_id_original"] = original_id
        normalized["sample_id"] = f"{source}_{original_id}"
        out.append(normalized)
    return out


def dedupe_by_message_id(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Keep first row per message_id, return (kept, duplicates)."""
    seen: set[str] = set()
    kept: list[dict[str, str]] = []
    dupes: list[dict[str, str]] = []

    for row in rows:
        mid = row.get("message_id", "")
        if not mid:
            raise ValueError(f"Row {row.get('sample_id', '<missing>')} missing message_id")
        if mid in seen:
            dupes.append(row)
            continue
        seen.add(mid)
        kept.append(row)

    return kept, dupes


def group_by_chat(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        chat_key = row.get("chat_rowid", "")
        if not chat_key:
            raise ValueError(f"Row {row.get('sample_id', '<missing>')} missing chat_rowid")
        grouped[chat_key].append(row)
    return grouped


def compute_chat_stats(grouped: dict[str, list[dict[str, str]]]) -> list[ChatStats]:
    stats: list[ChatStats] = []
    for chat_rowid, rows in grouped.items():
        total = len(rows)
        positives = sum(1 for r in rows if r.get("gold_keep") == "1")
        stats.append(ChatStats(chat_rowid=chat_rowid, total=total, positives=positives))
    return stats


def split_loss(
    dev_total: int,
    dev_pos: int,
    target_total: int,
    target_pos: int,
    total_rows: int,
) -> float:
    """Weighted objective for split quality.

    Primary objective: row count close to target.
    Secondary objective: positive count close to target.
    Mild penalty for very small/large dev proportions.
    """
    count_term = abs(dev_total - target_total)
    pos_term = abs(dev_pos - target_pos)
    frac = dev_total / max(total_rows, 1)
    frac_penalty = 0.0
    if frac < 0.12:
        frac_penalty = (0.12 - frac) * 100
    elif frac > 0.35:
        frac_penalty = (frac - 0.35) * 100
    return (2.0 * count_term) + (1.25 * pos_term) + frac_penalty


def choose_dev_chats(
    chat_stats: list[ChatStats],
    target_total: int,
    target_pos: int,
    total_rows: int,
    seed: int,
    search_iters: int,
) -> set[str]:
    """Search for a chat-level dev set close to target size and class balance."""
    rng = random.Random(seed)

    if not chat_stats:
        return set()

    # Ensure at least one chat in dev.
    best_dev: set[str] | None = None
    best_loss: float | None = None

    for _ in range(max(search_iters, 1)):
        shuffled = chat_stats[:]
        rng.shuffle(shuffled)

        dev_set: set[str] = set()
        dev_total = 0
        dev_pos = 0

        # Greedily accumulate chats until we cross target total.
        for cs in shuffled:
            if dev_total >= target_total:
                break
            dev_set.add(cs.chat_rowid)
            dev_total += cs.total
            dev_pos += cs.positives

        # Try local refinement: random swaps in/out.
        current_loss = split_loss(dev_total, dev_pos, target_total, target_pos, total_rows)

        for _ in range(25):
            if not dev_set:
                break
            in_chat = rng.choice(tuple(dev_set))
            out_candidates = [cs.chat_rowid for cs in chat_stats if cs.chat_rowid not in dev_set]
            if not out_candidates:
                break
            out_chat = rng.choice(out_candidates)

            in_stats = next(cs for cs in chat_stats if cs.chat_rowid == in_chat)
            out_stats = next(cs for cs in chat_stats if cs.chat_rowid == out_chat)

            trial_total = dev_total - in_stats.total + out_stats.total
            trial_pos = dev_pos - in_stats.positives + out_stats.positives
            trial_loss = split_loss(trial_total, trial_pos, target_total, target_pos, total_rows)

            if trial_loss < current_loss:
                dev_set.remove(in_chat)
                dev_set.add(out_chat)
                dev_total = trial_total
                dev_pos = trial_pos
                current_loss = trial_loss

        if best_loss is None or current_loss < best_loss:
            best_loss = current_loss
            best_dev = set(dev_set)

    return best_dev or set()


def split_rows_by_chat(
    rows: list[dict[str, str]],
    dev_chats: set[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    dev_rows = [r for r in rows if r.get("chat_rowid") in dev_chats]
    train_rows = [r for r in rows if r.get("chat_rowid") not in dev_chats]
    return train_rows, dev_rows


def to_message_gate_jsonl_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Convert merged CSV rows to explicit message-gate training schema."""
    out: list[dict[str, Any]] = []
    for row in rows:
        keep = int(row["gold_keep"])
        out.append(
            {
                "sample_id": row["sample_id"],
                "message_id": int(row["message_id"]),
                "chat_rowid": row["chat_rowid"],
                "chat_id": row.get("chat_id", ""),
                "is_from_me": row.get("is_from_me", "") == "True",
                "sender_handle": row.get("sender_handle", ""),
                "text": row.get("message_text", ""),
                "bucket": row.get("bucket", ""),
                "label": keep,
                "gold_fact_type": row.get("gold_fact_type", ""),
                "gold_subject": row.get("gold_subject", ""),
                "gold_subject_resolution": row.get("gold_subject_resolution", ""),
                "gold_anchor_message_id": (
                    int(row["gold_anchor_message_id"])
                    if row.get("gold_anchor_message_id", "").strip()
                    else None
                ),
                "source_round": row.get("source_round", ""),
            }
        )
    return out


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    _setup_logging()
    logging.info("Starting merge_goldsets.py")
    args = parse_args()

    if not (0 < args.dev_frac < 1):
        print("ERROR: --dev-frac must be in (0,1)", file=sys.stderr, flush=True)
        return 2

    require_files(args.v1, args.r2)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        v1_rows_raw = read_csv(args.v1)
        r2_rows_raw = read_csv(args.r2)

        v1_rows = normalize_source_rows(v1_rows_raw, "v1")
        r2_rows = normalize_source_rows(r2_rows_raw, "r2")

        merged_rows, dupes = dedupe_by_message_id(v1_rows + r2_rows)

        keeps = sum(1 for r in merged_rows if r["gold_keep"] == "1")
        discards = len(merged_rows) - keeps

        grouped = group_by_chat(merged_rows)
        chat_stats = compute_chat_stats(grouped)

        total_rows = len(merged_rows)
        target_dev_total = max(1, round(total_rows * args.dev_frac))
        target_dev_pos = max(1, round(keeps * args.dev_frac)) if keeps > 0 else 0

        dev_chats = choose_dev_chats(
            chat_stats=chat_stats,
            target_total=target_dev_total,
            target_pos=target_dev_pos,
            total_rows=total_rows,
            seed=args.seed,
            search_iters=args.search_iters,
        )

        train_rows, dev_rows = split_rows_by_chat(merged_rows, dev_chats)

        # Final leakage checks.
        train_chats = {r["chat_rowid"] for r in train_rows}
        dev_chat_set = {r["chat_rowid"] for r in dev_rows}
        chat_overlap = train_chats & dev_chat_set
        if chat_overlap:
            raise RuntimeError(
                f"Chat leakage detected after split; overlapping chats: {sorted(chat_overlap)[:10]}"
            )

        train_msg_ids = {r["message_id"] for r in train_rows}
        dev_msg_ids = {r["message_id"] for r in dev_rows}
        msg_overlap = train_msg_ids & dev_msg_ids
        if msg_overlap:
            raise RuntimeError(
                "Message leakage detected after split; overlapping message_ids: "
                f"{sorted(msg_overlap)[:10]}"
            )

        fieldnames = list(merged_rows[0].keys()) if merged_rows else []

        all_csv = args.out_dir / "all.csv"
        train_csv = args.out_dir / "train.csv"
        dev_csv = args.out_dir / "dev.csv"
        train_jsonl = args.out_dir / "train.jsonl"
        dev_jsonl = args.out_dir / "dev.jsonl"
        train_gate_jsonl = args.out_dir / "train_message_gate.jsonl"
        dev_gate_jsonl = args.out_dir / "dev_message_gate.jsonl"
        dupes_csv = args.out_dir / "duplicates_dropped.csv"
        manifest_json = args.out_dir / "manifest.json"

        for path in [
            all_csv,
            train_csv,
            dev_csv,
            train_jsonl,
            dev_jsonl,
            train_gate_jsonl,
            dev_gate_jsonl,
            dupes_csv,
            manifest_json,
        ]:
            ensure_writable(path, args.overwrite)

        # Write full row schema.
        write_csv(all_csv, fieldnames, merged_rows)
        write_csv(train_csv, fieldnames, train_rows)
        write_csv(dev_csv, fieldnames, dev_rows)
        write_jsonl(train_jsonl, train_rows)
        write_jsonl(dev_jsonl, dev_rows)

        # Write explicit message-gate schema.
        write_jsonl(train_gate_jsonl, to_message_gate_jsonl_rows(train_rows))
        write_jsonl(dev_gate_jsonl, to_message_gate_jsonl_rows(dev_rows))

        # Audit file for dropped duplicates.
        if dupes:
            write_csv(dupes_csv, fieldnames, dupes)
        else:
            with dupes_csv.open("w", encoding="utf-8") as f:
                f.write("sample_id,message_id,reason\n")

        train_keeps = sum(1 for r in train_rows if r["gold_keep"] == "1")
        dev_keeps = sum(1 for r in dev_rows if r["gold_keep"] == "1")

        manifest = {
            "inputs": {
                "v1": str(args.v1),
                "r2": str(args.r2),
            },
            "seed": args.seed,
            "search_iterations": args.search_iters,
            "dev_fraction_target": args.dev_frac,
            "v1_count": len(v1_rows),
            "r2_count": len(r2_rows),
            "duplicates_removed": len(dupes),
            "merged_total": len(merged_rows),
            "keeps": keeps,
            "discards": discards,
            "positive_rate": round(keeps / max(len(merged_rows), 1), 4),
            "train_count": len(train_rows),
            "train_positives": train_keeps,
            "train_positive_rate": round(train_keeps / max(len(train_rows), 1), 4),
            "dev_count": len(dev_rows),
            "dev_positives": dev_keeps,
            "dev_positive_rate": round(dev_keeps / max(len(dev_rows), 1), 4),
            "unique_chats_total": len(chat_stats),
            "unique_chats_train": len(train_chats),
            "unique_chats_dev": len(dev_chat_set),
            "chat_overlap_count": 0,
            "message_overlap_count": 0,
            "notes": {
                "sample_id_namespaced": True,
                "chat_held_out_split": True,
                "message_gate_jsonl_emitted": True,
                "candidate_filter_note": (
                    "train_fact_filter.py expects candidate-level labels; "
                    "message-gate JSONL is emitted for message-level binary models."
                ),
            },
        }

        with manifest_json.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=True)

        print(
            f"V1: {len(v1_rows)}, R2: {len(r2_rows)}, "
            f"Duplicates removed: {len(dupes)}, Merged: {len(merged_rows)}",
            flush=True,
        )
        print(
            f"Total keeps: {keeps}, discards: {discards} "
            f"({(keeps / max(len(merged_rows), 1)) * 100:.1f}% positive rate)",
            flush=True,
        )
        print(
            f"Train: {len(train_rows)} ({train_keeps} pos, {len(train_chats)} chats)",
            flush=True,
        )
        print(
            f"Dev:   {len(dev_rows)} ({dev_keeps} pos, {len(dev_chat_set)} chats)",
            flush=True,
        )

        print(f"\nOutputs in {args.out_dir}/:", flush=True)
        print(f"  all.csv                 ({len(merged_rows)} rows)", flush=True)
        print(f"  train.csv               ({len(train_rows)} rows)", flush=True)
        print(f"  dev.csv                 ({len(dev_rows)} rows)", flush=True)
        print(f"  train.jsonl             ({len(train_rows)} rows)", flush=True)
        print(f"  dev.jsonl               ({len(dev_rows)} rows)", flush=True)
        print(f"  train_message_gate.jsonl ({len(train_rows)} rows)", flush=True)
        print(f"  dev_message_gate.jsonl   ({len(dev_rows)} rows)", flush=True)
        print(f"  duplicates_dropped.csv   ({len(dupes)} rows)", flush=True)
        print("  manifest.json", flush=True)

        return 0

    except FileExistsError as e:
        print(f"ERROR: {e}", file=sys.stderr, flush=True)
        print("Use --overwrite to replace existing output files.", file=sys.stderr, flush=True)
        return 3
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr, flush=True)
        return 4

    logging.info("Finished merge_goldsets.py")


if __name__ == "__main__":
    raise SystemExit(main())
