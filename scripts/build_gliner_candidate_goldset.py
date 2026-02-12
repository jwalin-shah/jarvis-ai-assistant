#!/usr/bin/env python3
"""Build a targeted GLiNER candidate goldset pack for manual annotation.

This script samples iMessage messages focused on weak extraction areas
(e.g., org/location/health), runs CandidateExtractor to provide
"suggested_candidates", and exports JSON/CSV batches for manual labeling.

Output records are compatible with the candidate-gold workflow:
- expected_candidates: []  (to be filled manually)
- suggested_candidates: model suggestions for faster annotation
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gliner_shared import enforce_runtime_stack

from jarvis.utils.logging import setup_script_logging

APPLE_EPOCH_UNIX = 978307200  # 2001-01-01 00:00:00 UTC
NANOSECONDS_PER_SECOND = 1_000_000_000

IMESSAGE_DB = Path.home() / "Library" / "Messages" / "chat.db"
SOURCE_CSV_DEFAULT = Path("training_data/fact_goldset_merged/all.csv")

ORG_PATTERNS = [
    "%work at%",
    "%working at%",
    "%joined%",
    "%company%",
    "%startup%",
    "%intern%",
    "%manager%",
    "%engineer%",
    "%job%",
    "%role%",
    "%team%",
    "%office%",
]

LOCATION_PATTERNS = [
    "%live in%",
    "%living in%",
    "%moving to%",
    "%moved to%",
    "%based in%",
    "%from %",
    "%in sf%",
    "%in nyc%",
    "%in dallas%",
    "%in austin%",
    "%in california%",
    "%in san jose%",
]

HEALTH_PATTERNS = [
    "%hospital%",
    "%doctor%",
    "%pain%",
    "%injury%",
    "%headache%",
    "%pressure%",
    "%depressed%",
    "%anxious%",
    "%allergic%",
    "%emergency room%",
    "%therapy%",
    "%sick%",
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

BUCKET_LABEL_TARGETS = {
    "weak_org": {"org", "job_role", "employer"},
    "weak_location": {"place", "current_location", "future_location", "past_location"},
    "weak_health": {"health_condition", "allergy"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=SOURCE_CSV_DEFAULT,
        help=(
            "Optional CSV source. If this file exists it is used as message source; "
            "otherwise the script falls back to chat.db."
        ),
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=IMESSAGE_DB,
        help="Path to iMessage chat.db",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_data/gliner_goldset_round3"),
        help="Directory for output files",
    )
    parser.add_argument("--total", type=int, default=300, help="Total samples to emit")
    parser.add_argument("--org-count", type=int, default=100, help="weak_org samples")
    parser.add_argument("--location-count", type=int, default=100, help="weak_location samples")
    parser.add_argument("--health-count", type=int, default=70, help="weak_health samples")
    parser.add_argument("--negative-count", type=int, default=30, help="hard_negative samples")
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=12,
        help="Fetch multiplier per bucket before model filtering",
    )
    parser.add_argument("--min-length", type=int, default=5, help="Minimum message text length")
    parser.add_argument(
        "--context-window",
        type=int,
        default=2,
        help="Prev/next messages in context",
    )
    parser.add_argument("--per-chat-cap", type=int, default=8, help="Max selected per chat")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=32, help="Unused; reserved for parity")
    parser.add_argument("--threshold", type=float, default=0.35, help="GLiNER threshold")
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
        help="Disable per-label minimum filters when generating suggestions",
    )
    parser.add_argument(
        "--allow-unstable-stack",
        action="store_true",
        help="Allow running outside GLiNER compat runtime (not recommended)",
    )
    parser.add_argument("--batch-size-out", type=int, default=50, help="Rows per batch_*.json file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files")
    return parser.parse_args()


def parse_apple_timestamp(timestamp: int | float | None) -> datetime:
    if timestamp is None or timestamp == 0:
        return datetime.fromtimestamp(APPLE_EPOCH_UNIX, tz=UTC)
    try:
        seconds = timestamp / NANOSECONDS_PER_SECOND
        return datetime.fromtimestamp(APPLE_EPOCH_UNIX + seconds, tz=UTC)
    except (ValueError, OSError, OverflowError):
        return datetime.fromtimestamp(APPLE_EPOCH_UNIX, tz=UTC)


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def load_source_csv(path: Path) -> list[dict[str, Any]]:
    """Load pre-sampled message rows from CSV."""
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _matches_like(text_lower: str, pattern: str) -> bool:
    """Simple SQL-LIKE-style pattern matcher for %foo% patterns."""
    needle = pattern.lower().replace("%", "").strip()
    if not needle:
        return False
    return needle in text_lower


def _row_text(row: Any) -> str:
    return str(row["message_text"])


def _row_int(row: Any, key: str, default: int = 0) -> int:
    try:
        return int(row[key] or default)
    except Exception:
        return default


def _row_bool(row: Any, key: str) -> bool:
    val = row.get(key) if hasattr(row, "get") else row[key]
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in {"1", "true", "t", "yes"}


def fetch_candidates_from_rows(
    rows: list[dict[str, Any]],
    bucket: str,
    min_length: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Filter candidate rows from a CSV source using bucket patterns."""
    out: list[dict[str, Any]] = []
    seen: set[int] = set()

    def maybe_add(row: dict[str, Any]) -> None:
        mid = int(row["message_id"])
        if mid in seen:
            return
        seen.add(mid)
        out.append(row)

    # Pass 1: pattern-focused shortlist.
    for row in rows:
        text = _row_text(row).strip()
        if len(text) < min_length:
            continue
        text_lower = text.lower()

        keep = False
        if bucket == "weak_org":
            keep = any(_matches_like(text_lower, p) for p in ORG_PATTERNS)
        elif bucket == "weak_location":
            keep = any(_matches_like(text_lower, p) for p in LOCATION_PATTERNS)
        elif bucket == "weak_health":
            keep = any(_matches_like(text_lower, p) for p in HEALTH_PATTERNS)
        elif bucket == "hard_negative":
            keep = (
                text_lower in NEGATIVE_EXACT_TEXT
                or len(text_lower) <= 14
                or any(_matches_like(text_lower, p) for p in NEGATIVE_SPAM_PATTERNS)
            )
        else:
            raise ValueError(f"Unknown bucket: {bucket}")

        if not keep:
            continue

        maybe_add(row)
        if len(out) >= limit:
            break

    # Pass 2: for weak buckets, backfill with broad rows if pattern pass is sparse.
    if bucket in {"weak_org", "weak_location", "weak_health"} and len(out) < limit:
        for row in rows:
            text = _row_text(row).strip()
            if len(text) < min_length:
                continue
            maybe_add(row)
            if len(out) >= limit:
                break

    return out


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def base_query(bucket_filter_sql: str) -> str:
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
) -> list[sqlite3.Row]:
    if bucket == "weak_org":
        pattern_sql = " OR ".join("LOWER(m.text) LIKE ?" for _ in ORG_PATTERNS)
        sql = base_query(bucket_filter_sql=f"AND ({pattern_sql})")
        params: list[Any] = [min_length, *ORG_PATTERNS, limit]
    elif bucket == "weak_location":
        pattern_sql = " OR ".join("LOWER(m.text) LIKE ?" for _ in LOCATION_PATTERNS)
        sql = base_query(bucket_filter_sql=f"AND ({pattern_sql})")
        params = [min_length, *LOCATION_PATTERNS, limit]
    elif bucket == "weak_health":
        pattern_sql = " OR ".join("LOWER(m.text) LIKE ?" for _ in HEALTH_PATTERNS)
        sql = base_query(bucket_filter_sql=f"AND ({pattern_sql})")
        params = [min_length, *HEALTH_PATTERNS, limit]
    elif bucket == "hard_negative":
        exact_placeholders = ",".join("?" for _ in NEGATIVE_EXACT_TEXT)
        spam_sql = " OR ".join("LOWER(m.text) LIKE ?" for _ in NEGATIVE_SPAM_PATTERNS)
        neg_filter = (
            "AND ("
            f"LOWER(TRIM(m.text)) IN ({exact_placeholders}) "
            "OR LENGTH(TRIM(m.text)) <= 14 "
            f"OR {spam_sql}"
            ")"
        )
        sql = base_query(bucket_filter_sql=neg_filter)
        params = [min_length, *NEGATIVE_EXACT_TEXT, *NEGATIVE_SPAM_PATTERNS, limit]
    else:
        raise ValueError(f"Unknown bucket: {bucket}")

    rows = conn.execute(sql, params).fetchall()
    seen: set[int] = set()
    deduped: list[sqlite3.Row] = []
    for row in rows:
        mid = int(row["message_id"])
        if mid in seen:
            continue
        seen.add(mid)
        deduped.append(row)
    return deduped


def fetch_context(
    conn: sqlite3.Connection,
    chat_rowid: int,
    message_date_raw: int,
    message_id: int,
    window: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if window <= 0:
        return [], []

    prev_sql = """
    SELECT
        m.ROWID AS message_id,
        m.text AS message_text,
        m.is_from_me AS is_from_me
    FROM message m
    JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
    WHERE cmj.chat_id = ?
      AND m.text IS NOT NULL
      AND TRIM(m.text) != ''
      AND (
        m.date < ?
        OR (m.date = ? AND m.ROWID < ?)
      )
    ORDER BY m.date DESC, m.ROWID DESC
    LIMIT ?
    """

    next_sql = """
    SELECT
        m.ROWID AS message_id,
        m.text AS message_text,
        m.is_from_me AS is_from_me
    FROM message m
    JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
    WHERE cmj.chat_id = ?
      AND m.text IS NOT NULL
      AND TRIM(m.text) != ''
      AND (
        m.date > ?
        OR (m.date = ? AND m.ROWID > ?)
      )
    ORDER BY m.date ASC, m.ROWID ASC
    LIMIT ?
    """

    prev_rows = conn.execute(
        prev_sql,
        (chat_rowid, message_date_raw, message_date_raw, message_id, window),
    ).fetchall()
    next_rows = conn.execute(
        next_sql,
        (chat_rowid, message_date_raw, message_date_raw, message_id, window),
    ).fetchall()

    def to_ctx(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "message_id": int(row["message_id"]),
            "is_from_me": bool(row["is_from_me"]),
            "text": str(row["message_text"]),
        }

    prev_ctx = [to_ctx(row) for row in reversed(prev_rows)]
    next_ctx = [to_ctx(row) for row in next_rows]
    return prev_ctx, next_ctx


def context_to_string(items: list[dict[str, Any]]) -> str:
    parts = []
    for item in items:
        speaker = "me" if item["is_from_me"] else "them"
        parts.append(f"{item['message_id']}|{speaker}|{normalize_text(str(item['text']))}")
    return " || ".join(parts)


def should_keep_bucket(bucket: str, candidates: list[dict[str, Any]]) -> bool:
    if bucket == "hard_negative":
        return len(candidates) == 0
    wanted = BUCKET_LABEL_TARGETS[bucket]
    labels = {str(c.get("span_label", "")) for c in candidates}
    return bool(labels & wanted)


def select_bucket_samples(
    bucket: str,
    rows: list[Any],
    target: int,
    extractor: Any,
    threshold: float,
    disable_label_min: bool,
    used_ids: set[int],
    global_chat_counts: Counter[int],
    per_chat_cap: int,
    prediction_cache: dict[int, list[dict[str, Any]]],
) -> list[tuple[Any, list[dict[str, Any]]]]:
    selected: list[tuple[Any, list[dict[str, Any]]]] = []

    for row in rows:
        if len(selected) >= target:
            break

        message_id = _row_int(row, "message_id")
        chat_rowid = _row_int(row, "chat_rowid")

        if message_id in used_ids:
            continue
        if global_chat_counts[chat_rowid] >= per_chat_cap:
            continue

        if message_id in prediction_cache:
            candidates = prediction_cache[message_id]
        else:
            cands = extractor.extract_candidates(
                text=str(row["message_text"]),
                message_id=message_id,
                threshold=threshold,
                apply_label_thresholds=not disable_label_min,
                apply_vague_filter=True,
            )
            candidates = [
                {
                    "span_text": c.span_text,
                    "span_label": c.span_label,
                    "fact_type": c.fact_type,
                    "gliner_score": round(float(c.gliner_score), 4),
                }
                for c in cands
            ]
            prediction_cache[message_id] = candidates

        if not should_keep_bucket(bucket, candidates):
            continue

        selected.append((row, candidates))
        used_ids.add(message_id)
        global_chat_counts[chat_rowid] += 1

    return selected


def write_outputs(
    records: list[dict[str, Any]],
    output_dir: Path,
    overwrite: bool,
    batch_size_out: int,
    manifest: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "sampled_messages.json"
    csv_path = output_dir / "sampled_messages.csv"
    manifest_path = output_dir / "manifest.json"

    for p in (json_path, csv_path, manifest_path):
        if p.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {p}")

    # Write all files to temp paths first, then rename atomically
    # This prevents corrupt output if the process crashes mid-write
    tmp_files: list[tuple[Path, Path]] = []  # (tmp_path, final_path)

    try:
        tmp_json = json_path.with_suffix(".json.tmp")
        with tmp_json.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        tmp_files.append((tmp_json, json_path))

        tmp_csv = csv_path.with_suffix(".csv.tmp")
        with tmp_csv.open("w", newline="", encoding="utf-8") as f:
            cols = [
                "sample_id",
                "slice",
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
                "suggested_candidates_json",
                "expected_candidates_json",
                "gold_notes",
            ]
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for r in records:
                writer.writerow(
                    {
                        "sample_id": r["sample_id"],
                        "slice": r["slice"],
                        "message_id": r["message_id"],
                        "chat_rowid": r["chat_rowid"],
                        "chat_id": r["chat_id"],
                        "chat_display_name": r["chat_display_name"],
                        "is_from_me": r["is_from_me"],
                        "sender_handle": r["sender_handle"],
                        "message_date": r["message_date"],
                        "message_text": normalize_text(str(r["message_text"])),
                        "context_prev": r["context_prev"],
                        "context_next": r["context_next"],
                        "suggested_candidates_json": json.dumps(
                            r["suggested_candidates"], ensure_ascii=False
                        ),
                        "expected_candidates_json": json.dumps(
                            r["expected_candidates"], ensure_ascii=False
                        ),
                        "gold_notes": r.get("gold_notes", ""),
                    }
                )
        tmp_files.append((tmp_csv, csv_path))

        # Batch files for manual annotation
        for i in range(0, len(records), batch_size_out):
            batch_idx = i // batch_size_out
            batch_path = output_dir / f"batch_{batch_idx}.json"
            if batch_path.exists() and not overwrite:
                raise FileExistsError(f"Refusing to overwrite existing file: {batch_path}")
            tmp_batch = batch_path.with_suffix(".json.tmp")
            with tmp_batch.open("w", encoding="utf-8") as f:
                json.dump(records[i : i + batch_size_out], f, indent=2, ensure_ascii=False)
            tmp_files.append((tmp_batch, batch_path))

        tmp_manifest = manifest_path.with_suffix(".json.tmp")
        with tmp_manifest.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=True)
        tmp_files.append((tmp_manifest, manifest_path))

        # All writes succeeded - atomically rename all temp files
        for tmp_path, final_path in tmp_files:
            os.replace(tmp_path, final_path)
    except BaseException:
        # Clean up any temp files on failure
        for tmp_path, _ in tmp_files:
            tmp_path.unlink(missing_ok=True)
        raise


def main() -> int:
    logger = setup_script_logging("build_gliner_candidate_goldset")
    args = parse_args()
    enforce_runtime_stack(args.allow_unstable_stack)

    if args.org_count + args.location_count + args.health_count + args.negative_count != args.total:
        total_requested = (
            args.org_count + args.location_count + args.health_count + args.negative_count
        )
        raise SystemExit(
            f"Bucket counts must sum to --total. Got {total_requested} != {args.total}"
        )

    source_csv = args.source_csv.expanduser() if str(args.source_csv).strip() else None
    use_csv_source = source_csv is not None and source_csv.exists()

    db_path = args.db_path.expanduser()
    if not use_csv_source and not db_path.exists():
        raise SystemExit(f"chat.db not found at {db_path} and source CSV not found at {source_csv}")

    from jarvis.contacts.candidate_extractor import CandidateExtractor, labels_for_profile

    labels = labels_for_profile(args.label_profile)
    if args.drop_label:
        drop = set(args.drop_label)
        labels = [lbl for lbl in labels if lbl not in drop]
    if not labels:
        raise SystemExit("No active labels left after applying label filters.")

    logger.info("Using label profile: %s", args.label_profile)
    logger.info("Active labels: %s", ", ".join(labels))
    print(f"Using label profile: {args.label_profile}", flush=True)
    print(f"Active labels: {', '.join(labels)}", flush=True)

    extractor = CandidateExtractor(labels=labels, label_profile=args.label_profile)

    rng = random.Random(args.seed)

    conn: sqlite3.Connection | None = None
    try:
        source_rows: list[dict[str, Any]] | None = None
        if use_csv_source:
            source_rows = load_source_csv(source_csv)
            print(f"Using source CSV: {source_csv} ({len(source_rows)} rows)", flush=True)
        else:
            conn = connect_readonly(db_path)
            print(f"Using chat.db source: {db_path}", flush=True)

        bucket_targets = {
            "weak_org": args.org_count,
            "weak_location": args.location_count,
            "weak_health": args.health_count,
            "hard_negative": args.negative_count,
        }

        bucket_rows: dict[str, list[Any]] = {}
        for bucket, target in bucket_targets.items():
            fetch_limit = max(target * max(args.candidate_multiplier, 1), target)
            if use_csv_source:
                assert source_rows is not None
                rows = fetch_candidates_from_rows(
                    source_rows,
                    bucket=bucket,
                    min_length=args.min_length,
                    limit=fetch_limit,
                )
            else:
                assert conn is not None
                rows = fetch_candidates(
                    conn,
                    bucket=bucket,
                    min_length=args.min_length,
                    limit=fetch_limit,
                )
            rng.shuffle(rows)
            bucket_rows[bucket] = rows
            logger.info("Fetched %d raw candidates for %s", len(rows), bucket)
            print(f"Fetched {len(rows)} raw candidates for {bucket}", flush=True)

        used_ids: set[int] = set()
        global_chat_counts: Counter[int] = Counter()
        prediction_cache: dict[int, list[dict[str, Any]]] = {}

        selected_by_bucket: dict[str, list[tuple[Any, list[dict[str, Any]]]]] = {}
        for bucket in ("weak_org", "weak_location", "weak_health", "hard_negative"):
            sel = select_bucket_samples(
                bucket=bucket,
                rows=bucket_rows[bucket],
                target=bucket_targets[bucket],
                extractor=extractor,
                threshold=args.threshold,
                disable_label_min=args.no_label_min,
                used_ids=used_ids,
                global_chat_counts=global_chat_counts,
                per_chat_cap=args.per_chat_cap,
                prediction_cache=prediction_cache,
            )
            selected_by_bucket[bucket] = sel
            if len(sel) < bucket_targets[bucket]:
                print(
                    f"WARNING: bucket '{bucket}' requested {bucket_targets[bucket]} "
                    f"but selected {len(sel)}",
                    flush=True,
                )

        current_total = sum(len(v) for v in selected_by_bucket.values())
        if current_total < args.total:
            needed = args.total - current_total
            print(
                f"Backfilling {needed} additional rows to reach requested total={args.total}",
                flush=True,
            )

            if use_csv_source:
                assert source_rows is not None
                backfill_pool = source_rows[:]
            else:
                # Use already fetched candidates as a broad backfill pool.
                backfill_pool = []
                for rows in bucket_rows.values():
                    backfill_pool.extend(rows)

            rng.shuffle(backfill_pool)
            backfill_selected: list[tuple[Any, list[dict[str, Any]]]] = []

            for row in backfill_pool:
                if len(backfill_selected) >= needed:
                    break

                message_id = _row_int(row, "message_id")
                chat_rowid = _row_int(row, "chat_rowid")
                if message_id in used_ids:
                    continue
                if global_chat_counts[chat_rowid] >= args.per_chat_cap:
                    continue

                if message_id in prediction_cache:
                    cands = prediction_cache[message_id]
                else:
                    extracted = extractor.extract_candidates(
                        text=str(row["message_text"]),
                        message_id=message_id,
                        threshold=args.threshold,
                        apply_label_thresholds=not args.no_label_min,
                        apply_vague_filter=True,
                    )
                    cands = [
                        {
                            "span_text": c.span_text,
                            "span_label": c.span_label,
                            "fact_type": c.fact_type,
                            "gliner_score": round(float(c.gliner_score), 4),
                        }
                        for c in extracted
                    ]
                    prediction_cache[message_id] = cands

                likely_positive = len(cands) > 0
                if use_csv_source and not likely_positive:
                    likely_positive = str(row.get("gold_keep", "")).strip() == "1"
                if not likely_positive:
                    continue

                used_ids.add(message_id)
                global_chat_counts[chat_rowid] += 1
                backfill_selected.append((row, cands))

            selected_by_bucket["backfill"] = backfill_selected
            if len(backfill_selected) < needed:
                print(
                    f"WARNING: requested backfill {needed} but selected {len(backfill_selected)}",
                    flush=True,
                )

        combined: list[tuple[str, Any, list[dict[str, Any]]]] = []
        for bucket in ("weak_org", "weak_location", "weak_health", "hard_negative", "backfill"):
            if bucket not in selected_by_bucket:
                continue
            for row, cands in selected_by_bucket[bucket]:
                combined.append((bucket, row, cands))

        rng.shuffle(combined)

        records: list[dict[str, Any]] = []
        for idx, (bucket, row, cands) in enumerate(combined, start=1):
            message_id = _row_int(row, "message_id")
            date_raw = _row_int(row, "message_date_raw")

            if use_csv_source:
                context_prev = str(row.get("context_prev", ""))
                context_next = str(row.get("context_next", ""))
                message_date = (
                    str(row.get("message_date", "")) or parse_apple_timestamp(date_raw).isoformat()
                )
            else:
                assert conn is not None
                prev_ctx, next_ctx = fetch_context(
                    conn=conn,
                    chat_rowid=_row_int(row, "chat_rowid"),
                    message_date_raw=date_raw,
                    message_id=message_id,
                    window=args.context_window,
                )
                context_prev = context_to_string(prev_ctx)
                context_next = context_to_string(next_ctx)
                message_date = parse_apple_timestamp(date_raw).isoformat()

            records.append(
                {
                    "sample_id": f"r3_cand_{idx:04d}",
                    "message_id": message_id,
                    "chat_rowid": _row_int(row, "chat_rowid"),
                    "chat_id": str(row["chat_id"]),
                    "chat_display_name": str(row["chat_display_name"]),
                    "sender_handle": str(row["sender_handle"]),
                    "is_from_me": _row_bool(row, "is_from_me"),
                    "message_date": message_date,
                    "message_text": str(row["message_text"]),
                    "slice": bucket,
                    "context_prev": context_prev,
                    "context_next": context_next,
                    "suggested_candidates": sorted(
                        cands,
                        key=lambda x: float(x.get("gliner_score", 0.0)),
                        reverse=True,
                    )[:8],
                    "expected_candidates": [],
                    "gold_notes": "",
                }
            )

        manifest = {
            "source_mode": "csv" if use_csv_source else "chat.db",
            "source_csv": str(source_csv) if use_csv_source and source_csv is not None else None,
            "db_path": str(db_path) if not use_csv_source else None,
            "total": len(records),
            "requested_total": args.total,
            "counts": {b: len(v) for b, v in selected_by_bucket.items()},
            "settings": {
                "label_profile": args.label_profile,
                "drop_labels": args.drop_label,
                "threshold": args.threshold,
                "apply_label_min": not args.no_label_min,
                "context_window": args.context_window,
                "per_chat_cap": args.per_chat_cap,
                "candidate_multiplier": args.candidate_multiplier,
                "seed": args.seed,
            },
            "messages_scored": len(prediction_cache),
            "avg_suggestions_per_scored_message": round(
                sum(len(v) for v in prediction_cache.values()) / max(len(prediction_cache), 1),
                3,
            ),
        }

        write_outputs(
            records=records,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            batch_size_out=max(args.batch_size_out, 1),
            manifest=manifest,
        )

        logger.info("Built round-3 GLiNER candidate goldset pack")
        logger.info("  total:      %d", len(records))
        for b, v in selected_by_bucket.items():
            logger.info("  %s: %d", b, len(v))
        logger.info("  output dir: %s", args.output_dir)
        print("Built round-3 GLiNER candidate goldset pack", flush=True)
        print(f"  total:      {len(records)}", flush=True)
        for b, v in selected_by_bucket.items():
            print(f"  {b}: {len(v)}", flush=True)
        print(f"  output dir: {args.output_dir}", flush=True)
        print("  next: fill expected_candidates manually in CSV or batch JSON files", flush=True)
        return 0

    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
