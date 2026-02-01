#!/usr/bin/env python3
"""Sample trigger texts for manual gold labeling.

Usage:
    uv run python -m scripts.sample_trigger_gold_set
    uv run python -m scripts.sample_trigger_gold_set --limit 400 --output gold_triggers.jsonl
    uv run python -m scripts.sample_trigger_gold_set --include-response --include-context
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from datetime import datetime
from pathlib import Path

from jarvis.db import JARVIS_DB_PATH


def _connect_db(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _format_timestamp(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value).isoformat()
    except ValueError:
        return value


def _load_candidate_ids(
    conn: sqlite3.Connection,
    min_length: int,
    max_length: int,
    min_quality: float,
    group_filter: str,
) -> list[int]:
    conditions = [
        "trigger_text IS NOT NULL",
        "length(trim(trigger_text)) >= ?",
        "length(trim(trigger_text)) <= ?",
        "quality_score >= ?",
    ]
    params: list[object] = [min_length, max_length, min_quality]

    if group_filter == "exclude":
        conditions.append("is_group = 0")
    elif group_filter == "only":
        conditions.append("is_group = 1")

    where_clause = " AND ".join(conditions)
    cursor = conn.execute(
        f"SELECT id FROM pairs WHERE {where_clause}",
        params,
    )
    return [row[0] for row in cursor.fetchall()]


def _fetch_pairs(conn: sqlite3.Connection, pair_ids: list[int]) -> dict[int, sqlite3.Row]:
    if not pair_ids:
        return {}
    placeholders = ",".join("?" for _ in pair_ids)
    cursor = conn.execute(
        f"""
        SELECT id, trigger_text, response_text, context_text, is_group,
               trigger_timestamp, chat_id, contact_id
        FROM pairs
        WHERE id IN ({placeholders})
        """,
        pair_ids,
    )
    return {row[0]: row for row in cursor.fetchall()}


def _build_entry(
    row: sqlite3.Row,
    include_response: bool,
    include_context: bool,
    context_max_chars: int,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "pair_id": row[0],
        "trigger_text": row[1],
        "is_group": bool(row[4]),
        "trigger_timestamp": _format_timestamp(row[5]),
        "chat_id": row[6],
        "contact_id": row[7],
        "label": None,
        "notes": "",
    }

    if include_response:
        entry["response_text"] = row[2]
    if include_context:
        context = row[3] or ""
        if context_max_chars > 0:
            context = context[:context_max_chars]
        entry["context_text"] = context

    return entry


def sample_gold_set(
    db_path: Path,
    output_path: Path,
    limit: int,
    min_length: int,
    max_length: int,
    min_quality: float,
    seed: int,
    include_response: bool,
    include_context: bool,
    context_max_chars: int,
    group_filter: str,
) -> int:
    conn = _connect_db(db_path)
    conn.row_factory = sqlite3.Row
    try:
        candidate_ids = _load_candidate_ids(
            conn,
            min_length=min_length,
            max_length=max_length,
            min_quality=min_quality,
            group_filter=group_filter,
        )
        if not candidate_ids:
            return 0

        random.seed(seed)
        if limit >= len(candidate_ids):
            sampled_ids = candidate_ids
        else:
            sampled_ids = random.sample(candidate_ids, limit)

        rows = _fetch_pairs(conn, sampled_ids)
        entries = [
            _build_entry(
                rows[pair_id],
                include_response=include_response,
                include_context=include_context,
                context_max_chars=context_max_chars,
            )
            for pair_id in sampled_ids
            if pair_id in rows
        ]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=True))
                handle.write("\n")

        return len(entries)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample triggers for gold labeling")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gold_trigger_labels.jsonl"),
        help="Output JSONL file (default: gold_trigger_labels.jsonl)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=JARVIS_DB_PATH,
        help="Path to jarvis.db (default: ~/.jarvis/jarvis.db)",
    )
    parser.add_argument("--limit", type=int, default=500, help="Sample size (default: 500)")
    parser.add_argument(
        "--min-length",
        type=int,
        default=2,
        help="Minimum trigger length (default: 2)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=200,
        help="Maximum trigger length (default: 200)",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.0,
        help="Minimum quality_score (default: 0.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for sampling (default: 13)",
    )
    parser.add_argument(
        "--include-response",
        action="store_true",
        help="Include response_text in output",
    )
    parser.add_argument(
        "--include-context",
        action="store_true",
        help="Include context_text in output",
    )
    parser.add_argument(
        "--context-max-chars",
        type=int,
        default=400,
        help="Max chars for context_text (default: 400)",
    )
    parser.add_argument(
        "--group-only",
        action="store_true",
        help="Only include group chats",
    )
    parser.add_argument(
        "--exclude-group",
        action="store_true",
        help="Exclude group chats",
    )
    args = parser.parse_args()

    group_filter = "any"
    if args.group_only:
        group_filter = "only"
    if args.exclude_group:
        group_filter = "exclude"

    count = sample_gold_set(
        db_path=args.db_path,
        output_path=args.output,
        limit=args.limit,
        min_length=args.min_length,
        max_length=args.max_length,
        min_quality=args.min_quality,
        seed=args.seed,
        include_response=args.include_response,
        include_context=args.include_context,
        context_max_chars=args.context_max_chars,
        group_filter=group_filter,
    )

    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
