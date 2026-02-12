#!/usr/bin/env python3
"""Build a labeling-ready fact extraction gold set from local iMessage messages.

Creates a 200-message stratified sample (by default):
- random messages
- likely fact-bearing messages
- hard negatives / short filler messages

Outputs:
- JSONL: full records with context and empty annotation fields
- CSV: spreadsheet-friendly view
- manifest JSON: reproducibility metadata and bucket stats

Usage:
    python3 scripts/build_fact_goldset.py
    python3 scripts/build_fact_goldset.py --total 300 --seed 123
    python3 scripts/build_fact_goldset.py --db-path ~/Library/Messages/chat.db
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jarvis.utils.logging import setup_script_logging

APPLE_EPOCH_UNIX = 978307200  # 2001-01-01 00:00:00 UTC
NANOSECONDS_PER_SECOND = 1_000_000_000

LIKELY_FACT_PATTERNS = [
    "%i love%",
    "%i like%",
    "%i hate%",
    "%allergic%",
    "%obsessed with%",
    "%moving to%",
    "%live in%",
    "%work at%",
    "%started at%",
    "%my sister%",
    "%my brother%",
    "%my mom%",
    "%my dad%",
    "%my wife%",
    "%my husband%",
    "%my girlfriend%",
    "%my boyfriend%",
]

NEGATIVE_EXACT_TEXT = [
    "lol",
    "lmao",
    "haha",
    "same",
    "same lol",
    "yeah same",
    "me too",
    "ok",
    "k",
    "kk",
    "yep",
    "yup",
    "bet",
    "fr",
    "idk",
    "nm",
]

NEGATIVE_SPAM_PATTERNS = [
    "%cvs pharmacy%",
    "%prescription is ready%",
    "%unsubscribe%",
    "%check out this job%",
    "%apply now%",
]


@dataclass(frozen=True)
class SampledMessage:
    """Message selected for gold-set annotation."""

    message_id: int
    chat_rowid: int
    chat_id: str
    chat_display_name: str
    sender_handle: str
    is_from_me: bool
    message_date_raw: int
    message_date_iso: str
    message_text: str
    bucket: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fact extraction gold-set sample from iMessage"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path.home() / "Library" / "Messages" / "chat.db",
        help="Path to chat.db",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_data/fact_goldset"),
        help="Directory for output files",
    )
    parser.add_argument("--total", type=int, default=200, help="Total sampled messages")
    parser.add_argument(
        "--random-count",
        type=int,
        default=120,
        help="Count for random bucket",
    )
    parser.add_argument(
        "--likely-count",
        type=int,
        default=50,
        help="Count for likely fact-bearing bucket",
    )
    parser.add_argument(
        "--negative-count",
        type=int,
        default=30,
        help="Count for hard-negative bucket",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=2,
        help="Number of previous and next messages to attach as context",
    )
    parser.add_argument(
        "--per-chat-cap",
        type=int,
        default=10,
        help="Max sampled messages per chat",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=5,
        help="Minimum trimmed text length",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=30,
        help="Multiplier for candidate pool size per bucket",
    )
    parser.add_argument(
        "--short-neg-length",
        type=int,
        default=12,
        help="Max length for short-message negatives",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    return parser.parse_args()


def parse_apple_timestamp(timestamp: int | float | None) -> datetime:
    """Convert Apple nanoseconds-since-2001 timestamp to datetime (UTC)."""
    if timestamp is None or timestamp == 0:
        return datetime.fromtimestamp(APPLE_EPOCH_UNIX, tz=UTC)

    try:
        seconds = timestamp / NANOSECONDS_PER_SECOND
        return datetime.fromtimestamp(APPLE_EPOCH_UNIX + seconds, tz=UTC)
    except (ValueError, OSError, OverflowError):
        return datetime.fromtimestamp(APPLE_EPOCH_UNIX, tz=UTC)


def normalize_text(text: str) -> str:
    """Normalize message text for compact CSV output."""
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Open chat.db in read-only mode with row mapping enabled."""
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def base_query(bucket_filter_sql: str) -> str:
    """Build bucket query over message/chat tables."""
    return f"""
    SELECT
        m.ROWID AS message_id,
        cmj.chat_id AS chat_rowid,
        COALESCE(chat.chat_identifier, chat.guid, CAST(chat.ROWID AS TEXT)) AS chat_id,
        COALESCE(chat.display_name, chat.chat_identifier, chat.guid, CAST(chat.ROWID AS TEXT))
            AS chat_display_name,
        COALESCE(h.id, 'me') AS sender_handle,
        m.is_from_me AS is_from_me,
        m.date AS message_date_raw,
        m.text AS message_text
    FROM message m
    JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
    JOIN chat ON cmj.chat_id = chat.ROWID
    LEFT JOIN handle h ON m.handle_id = h.ROWID
    WHERE m.text IS NOT NULL
      AND TRIM(m.text) != ''
      AND LENGTH(TRIM(m.text)) >= ?
      {bucket_filter_sql}
    ORDER BY RANDOM()
    LIMIT ?
    """


def fetch_candidates(
    conn: sqlite3.Connection,
    bucket: str,
    min_length: int,
    limit: int,
    short_neg_length: int,
) -> list[sqlite3.Row]:
    """Fetch candidate rows for a sampling bucket."""
    if bucket == "random":
        sql = base_query(bucket_filter_sql="")
        params: list[Any] = [min_length, limit]
    elif bucket == "likely":
        pattern_sql = " OR ".join("LOWER(m.text) LIKE ?" for _ in LIKELY_FACT_PATTERNS)
        sql = base_query(bucket_filter_sql=f"AND ({pattern_sql})")
        params = [min_length, *LIKELY_FACT_PATTERNS, limit]
    elif bucket == "negative":
        exact_placeholders = ",".join("?" for _ in NEGATIVE_EXACT_TEXT)
        spam_sql = " OR ".join("LOWER(m.text) LIKE ?" for _ in NEGATIVE_SPAM_PATTERNS)
        neg_filter = (
            "AND ("
            f"LOWER(TRIM(m.text)) IN ({exact_placeholders}) "
            "OR LENGTH(TRIM(m.text)) <= ? "
            f"OR {spam_sql}"
            ")"
        )
        sql = base_query(bucket_filter_sql=neg_filter)
        params = [
            min_length,
            *NEGATIVE_EXACT_TEXT,
            short_neg_length,
            *NEGATIVE_SPAM_PATTERNS,
            limit,
        ]
    else:
        raise ValueError(f"Unknown bucket: {bucket}")

    rows = conn.execute(sql, params).fetchall()

    # Deduplicate by message_id while preserving randomized order.
    seen: set[int] = set()
    deduped: list[sqlite3.Row] = []
    for row in rows:
        mid = int(row["message_id"])
        if mid in seen:
            continue
        seen.add(mid)
        deduped.append(row)
    return deduped


def pick_bucket(
    rows: list[sqlite3.Row],
    target: int,
    bucket: str,
    used_ids: set[int],
    global_chat_counts: Counter[int],
    per_chat_cap: int,
    rng: random.Random,
) -> list[SampledMessage]:
    """Select messages from one bucket with dedupe + chat cap."""
    shuffled = rows[:]
    rng.shuffle(shuffled)

    def build_sample(row: sqlite3.Row) -> SampledMessage:
        date_raw = int(row["message_date_raw"] or 0)
        return SampledMessage(
            message_id=int(row["message_id"]),
            chat_rowid=int(row["chat_rowid"]),
            chat_id=str(row["chat_id"]),
            chat_display_name=str(row["chat_display_name"]),
            sender_handle=str(row["sender_handle"]),
            is_from_me=bool(row["is_from_me"]),
            message_date_raw=date_raw,
            message_date_iso=parse_apple_timestamp(date_raw).isoformat(),
            message_text=str(row["message_text"]),
            bucket=bucket,
        )

    selected: list[SampledMessage] = []

    # Pass 1: enforce per-chat cap.
    for row in shuffled:
        if len(selected) >= target:
            break

        message_id = int(row["message_id"])
        chat_rowid = int(row["chat_rowid"])

        if message_id in used_ids:
            continue
        if global_chat_counts[chat_rowid] >= per_chat_cap:
            continue

        sample = build_sample(row)
        selected.append(sample)
        used_ids.add(message_id)
        global_chat_counts[chat_rowid] += 1

    # Pass 2: if bucket under-filled, relax cap but keep dedupe.
    if len(selected) < target:
        for row in shuffled:
            if len(selected) >= target:
                break

            message_id = int(row["message_id"])
            chat_rowid = int(row["chat_rowid"])

            if message_id in used_ids:
                continue

            sample = build_sample(row)
            selected.append(sample)
            used_ids.add(message_id)
            global_chat_counts[chat_rowid] += 1

    return selected


def _context_query(direction: str) -> str:
    """Build a context query for the given direction ('prev' or 'next').

    The only differences between prev/next are the comparison operators
    and the sort order.
    """
    if direction == "prev":
        date_cmp, rowid_cmp, sort_order = "<", "<", "DESC"
    else:
        date_cmp, rowid_cmp, sort_order = ">", ">", "ASC"

    return f"""
    SELECT
        m.ROWID AS message_id,
        m.text AS message_text,
        m.date AS message_date_raw,
        m.is_from_me AS is_from_me,
        COALESCE(h.id, 'me') AS sender_handle
    FROM message m
    JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
    LEFT JOIN handle h ON m.handle_id = h.ROWID
    WHERE cmj.chat_id = ?
      AND m.text IS NOT NULL
      AND TRIM(m.text) != ''
      AND (
        m.date {date_cmp} ?
        OR (m.date = ? AND m.ROWID {rowid_cmp} ?)
      )
    ORDER BY m.date {sort_order}, m.ROWID {sort_order}
    LIMIT ?
    """


def fetch_context(
    conn: sqlite3.Connection,
    chat_rowid: int,
    message_date_raw: int,
    message_id: int,
    window: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch surrounding messages (previous and next) for annotation context."""
    if window <= 0:
        return [], []

    params = (chat_rowid, message_date_raw, message_date_raw, message_id, window)
    prev_rows = conn.execute(_context_query("prev"), params).fetchall()
    next_rows = conn.execute(_context_query("next"), params).fetchall()

    def to_context(row: sqlite3.Row) -> dict[str, Any]:
        raw = int(row["message_date_raw"] or 0)
        return {
            "message_id": int(row["message_id"]),
            "is_from_me": bool(row["is_from_me"]),
            "sender_handle": str(row["sender_handle"]),
            "message_date": parse_apple_timestamp(raw).isoformat(),
            "text": str(row["message_text"]),
        }

    prev_context = [to_context(row) for row in reversed(prev_rows)]
    next_context = [to_context(row) for row in next_rows]
    return prev_context, next_context


def context_to_string(items: list[dict[str, Any]]) -> str:
    """Compact context representation for CSV."""
    parts = []
    for item in items:
        speaker = "me" if item["is_from_me"] else "them"
        text = normalize_text(str(item["text"]))
        parts.append(f"{item['message_id']}|{speaker}|{text}")
    return " || ".join(parts)


def validate_counts(total: int, random_count: int, likely_count: int, negative_count: int) -> None:
    expected = random_count + likely_count + negative_count
    if expected != total:
        raise ValueError(
            "Bucket counts must sum to --total. "
            f"Got random+likely+negative={expected}, total={total}."
        )


def ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")


def write_outputs(
    records: list[dict[str, Any]],
    output_dir: Path,
    prefix: str,
    manifest: dict[str, Any],
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / f"{prefix}.jsonl"
    csv_path = output_dir / f"{prefix}.csv"
    manifest_path = output_dir / f"{prefix}_manifest.json"

    ensure_writable(jsonl_path, overwrite)
    ensure_writable(csv_path, overwrite)
    ensure_writable(manifest_path, overwrite)

    # Write all files to temp paths first, then rename atomically
    # This prevents corrupt output if the process crashes mid-write
    tmp_files: list[tuple[Path, Path]] = []

    try:
        tmp_jsonl = jsonl_path.with_suffix(".jsonl.tmp")
        with tmp_jsonl.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")
        tmp_files.append((tmp_jsonl, jsonl_path))

        csv_columns = [
            "sample_id",
            "bucket",
            "message_id",
            "chat_rowid",
            "chat_id",
            "chat_display_name",
            "is_from_me",
            "sender_handle",
            "message_date",
            "message_text",
            "context_prev",
            "context_next",
            "gold_keep",
            "gold_fact_type",
            "gold_subject",
            "gold_subject_resolution",
            "gold_anchor_message_id",
            "gold_notes",
        ]

        tmp_csv = csv_path.with_suffix(".csv.tmp")
        with tmp_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            for rec in records:
                writer.writerow(
                    {
                        "sample_id": rec["sample_id"],
                        "bucket": rec["bucket"],
                        "message_id": rec["message_id"],
                        "chat_rowid": rec["chat_rowid"],
                        "chat_id": rec["chat_id"],
                        "chat_display_name": rec["chat_display_name"],
                        "is_from_me": rec["is_from_me"],
                        "sender_handle": rec["sender_handle"],
                        "message_date": rec["message_date"],
                        "message_text": normalize_text(str(rec["message_text"])),
                        "context_prev": context_to_string(rec["context_prev"]),
                        "context_next": context_to_string(rec["context_next"]),
                        "gold_keep": "",
                        "gold_fact_type": "",
                        "gold_subject": "",
                        "gold_subject_resolution": "",
                        "gold_anchor_message_id": "",
                        "gold_notes": "",
                    }
                )
        tmp_files.append((tmp_csv, csv_path))

        tmp_manifest = manifest_path.with_suffix(".json.tmp")
        with tmp_manifest.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=True)
        tmp_files.append((tmp_manifest, manifest_path))

        # All writes succeeded - atomically rename all temp files
        for tmp_path, final_path in tmp_files:
            os.replace(tmp_path, final_path)
    except BaseException:
        for tmp_path, _ in tmp_files:
            tmp_path.unlink(missing_ok=True)
        raise

    return jsonl_path, csv_path, manifest_path


def build_records(
    conn: sqlite3.Connection,
    samples: list[SampledMessage],
    context_window: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples, start=1):
        prev_ctx, next_ctx = fetch_context(
            conn=conn,
            chat_rowid=sample.chat_rowid,
            message_date_raw=sample.message_date_raw,
            message_id=sample.message_id,
            window=context_window,
        )
        records.append(
            {
                "sample_id": f"fact_gs_{idx:04d}",
                "bucket": sample.bucket,
                "message_id": sample.message_id,
                "chat_rowid": sample.chat_rowid,
                "chat_id": sample.chat_id,
                "chat_display_name": sample.chat_display_name,
                "sender_handle": sample.sender_handle,
                "is_from_me": sample.is_from_me,
                "message_date": sample.message_date_iso,
                "message_text": sample.message_text,
                "context_prev": prev_ctx,
                "context_next": next_ctx,
                "annotation": {
                    "gold_keep": None,
                    "gold_fact_type": "",
                    "gold_subject": "",
                    "gold_subject_resolution": "",
                    "gold_anchor_message_id": None,
                    "gold_notes": "",
                },
            }
        )
    return records


def main() -> int:
    logger = setup_script_logging("build_fact_goldset")
    args = parse_args()

    validate_counts(
        total=args.total,
        random_count=args.random_count,
        likely_count=args.likely_count,
        negative_count=args.negative_count,
    )

    rng = random.Random(args.seed)

    db_path = args.db_path.expanduser()
    if not db_path.exists():
        print(f"ERROR: chat.db not found at {db_path}", flush=True)
        return 1

    conn: sqlite3.Connection | None = None
    try:
        conn = connect_readonly(db_path)

        candidate_multiplier = max(args.candidate_multiplier, 1)
        bucket_targets = {
            "random": args.random_count,
            "likely": args.likely_count,
            "negative": args.negative_count,
        }

        bucket_candidates: dict[str, list[sqlite3.Row]] = {}
        for bucket, target in bucket_targets.items():
            fetch_limit = max(target * candidate_multiplier, target)
            bucket_candidates[bucket] = fetch_candidates(
                conn=conn,
                bucket=bucket,
                min_length=args.min_length,
                limit=fetch_limit,
                short_neg_length=args.short_neg_length,
            )

        used_ids: set[int] = set()
        global_chat_counts: Counter[int] = Counter()

        sampled: list[SampledMessage] = []
        for bucket in ("random", "likely", "negative"):
            selected = pick_bucket(
                rows=bucket_candidates[bucket],
                target=bucket_targets[bucket],
                bucket=bucket,
                used_ids=used_ids,
                global_chat_counts=global_chat_counts,
                per_chat_cap=args.per_chat_cap,
                rng=rng,
            )
            logger.info(
                "Bucket '%s': selected %d/%d", bucket, len(selected), bucket_targets[bucket]
            )
            if len(selected) < bucket_targets[bucket]:
                print(
                    f"WARNING: bucket '{bucket}' requested {bucket_targets[bucket]} "
                    f"but selected {len(selected)}",
                    flush=True,
                )
            sampled.extend(selected)

        if len(sampled) < args.total:
            print(
                "WARNING: sampled fewer than requested total. "
                f"requested={args.total}, got={len(sampled)}",
                flush=True,
            )

        # Stable shuffle before assigning sample IDs so annotators see mixed buckets.
        rng.shuffle(sampled)

        records = build_records(conn=conn, samples=sampled, context_window=args.context_window)

        bucket_counts = Counter(item.bucket for item in sampled)
        chat_counts = Counter(item.chat_rowid for item in sampled)
        from_me_counts = Counter(item.is_from_me for item in sampled)

        now_utc = datetime.now(UTC).isoformat()
        prefix = f"fact_goldset_{len(records)}"

        manifest = {
            "created_at_utc": now_utc,
            "db_path": str(db_path),
            "seed": args.seed,
            "total_records": len(records),
            "config": {
                "total": args.total,
                "random_count": args.random_count,
                "likely_count": args.likely_count,
                "negative_count": args.negative_count,
                "context_window": args.context_window,
                "per_chat_cap": args.per_chat_cap,
                "min_length": args.min_length,
                "candidate_multiplier": args.candidate_multiplier,
                "short_neg_length": args.short_neg_length,
            },
            "bucket_counts": dict(bucket_counts),
            "is_from_me_counts": {str(k): v for k, v in from_me_counts.items()},
            "unique_chats": len(chat_counts),
            "top_chats": [
                {"chat_rowid": chat_id, "count": count}
                for chat_id, count in chat_counts.most_common(15)
            ],
        }

        jsonl_path, csv_path, manifest_path = write_outputs(
            records=records,
            output_dir=args.output_dir,
            prefix=prefix,
            manifest=manifest,
            overwrite=args.overwrite,
        )

        logger.info("Built fact gold-set annotation pack")
        logger.info("  records:      %d", len(records))
        logger.info("  random:       %d", bucket_counts.get("random", 0))
        logger.info("  likely:       %d", bucket_counts.get("likely", 0))
        logger.info("  negative:     %d", bucket_counts.get("negative", 0))
        logger.info("  unique chats: %d", len(chat_counts))
        logger.info("  jsonl:        %s", jsonl_path)
        logger.info("  csv:          %s", csv_path)
        logger.info("  manifest:     %s", manifest_path)
        print("Built fact gold-set annotation pack", flush=True)
        print(f"  records:      {len(records)}", flush=True)
        print(f"  random:       {bucket_counts.get('random', 0)}", flush=True)
        print(f"  likely:       {bucket_counts.get('likely', 0)}", flush=True)
        print(f"  negative:     {bucket_counts.get('negative', 0)}", flush=True)
        print(f"  unique chats: {len(chat_counts)}", flush=True)
        print(f"  jsonl:        {jsonl_path}", flush=True)
        print(f"  csv:          {csv_path}", flush=True)
        print(f"  manifest:     {manifest_path}", flush=True)
        print(
            "\nNext: open the CSV and fill gold_* columns "
            "(keep/fact_type/subject/subject_resolution).",
            flush=True,
        )
        return 0

    except sqlite3.OperationalError as e:
        msg = str(e).lower()
        if "unable to open database" in msg or "operation not permitted" in msg:
            print("ERROR: Could not read chat.db (macOS privacy restriction).", flush=True)
            print(
                "Grant Terminal/iTerm Full Disk Access in System Settings > Privacy & Security.",
                flush=True,
            )
            print(
                f"Then rerun: python3 scripts/build_fact_goldset.py --db-path {db_path}", flush=True
            )
            return 2
        print(f"ERROR: SQLite failure: {e}", flush=True)
        return 2
    except FileExistsError as e:
        print(f"ERROR: {e}", flush=True)
        print("Use --overwrite to replace existing files.", flush=True)
        return 3
    except ValueError as e:
        print(f"ERROR: {e}", flush=True)
        return 4
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
