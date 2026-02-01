#!/usr/bin/env python3
"""Sample candidate triggers for targeted labeling.

Usage:
    uv run python -m scripts.sample_trigger_candidates \
        --label good_news --limit 120 --output results/trigger_candidates_good_news.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
from pathlib import Path
from typing import Any

from jarvis.db import JARVIS_DB_PATH


LABEL_PATTERNS: dict[str, list[re.Pattern]] = {
    "good_news": [
        re.compile(r"\b(i got|we got|i passed|i made|i won|we won)\b", re.I),
        re.compile(r"\b(good news|great news|so happy|so excited|finally)\b", re.I),
        re.compile(r"\b(i feel better|feeling better|much better)\b", re.I),
    ],
    "bad_news": [
        re.compile(r"\b(i lost|we lost|i failed|i got fired|i'm sick)\b", re.I),
        re.compile(r"\b(bad news|unfortunately|so sad|so upset|this sucks)\b", re.I),
        re.compile(r"\b(i feel awful|feel terrible|feel terrible)\b", re.I),
    ],
    "greeting": [
        re.compile(r"^(hey|hi|hello|yo|sup|what'?s up|wassup|howdy)\b", re.I),
        re.compile(r"\bhow are you\b", re.I),
        re.compile(r"\bhey+\b", re.I),
    ],
}


def _connect_db(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _fetch_candidates(
    conn: sqlite3.Connection,
    label: str,
    limit: int,
    min_length: int,
    max_length: int,
    exclude_group: bool,
    seed: int,
) -> list[dict[str, Any]]:
    patterns = LABEL_PATTERNS.get(label, [])
    if not patterns:
        return []

    conditions = [
        "trigger_text IS NOT NULL",
        "length(trim(trigger_text)) >= ?",
        "length(trim(trigger_text)) <= ?",
    ]
    params: list[Any] = [min_length, max_length]

    if exclude_group:
        conditions.append("is_group = 0")

    where_clause = " AND ".join(conditions)
    cursor = conn.execute(
        f"""
        SELECT id, trigger_text, context_text, is_group, trigger_timestamp, chat_id
        FROM pairs
        WHERE {where_clause}
        """,
        params,
    )

    matches: list[dict[str, Any]] = []
    for row in cursor.fetchall():
        text = (row[1] or "").strip()
        if not text:
            continue
        if not any(pattern.search(text) for pattern in patterns):
            continue
        matches.append(
            {
                "pair_id": row[0],
                "trigger_text": text,
                "context_text": row[2] or "",
                "is_group": bool(row[3]),
                "trigger_timestamp": row[4],
                "chat_id": row[5],
                "label": None,
                "notes": "",
            }
        )

    rng = random.Random(seed)
    if len(matches) <= limit:
        return matches
    return rng.sample(matches, limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample trigger candidates for labeling")
    parser.add_argument(
        "--label",
        choices=sorted(LABEL_PATTERNS.keys()),
        required=True,
        help="Target label to sample",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=JARVIS_DB_PATH,
        help="Path to jarvis.db (default: ~/.jarvis/jarvis.db)",
    )
    parser.add_argument("--limit", type=int, default=120, help="Sample size (default: 120)")
    parser.add_argument("--min-length", type=int, default=2, help="Min trigger length")
    parser.add_argument("--max-length", type=int, default=200, help="Max trigger length")
    parser.add_argument("--exclude-group", action="store_true", help="Exclude group chats")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    args = parser.parse_args()

    conn = _connect_db(args.db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = _fetch_candidates(
            conn,
            label=args.label,
            limit=args.limit,
            min_length=args.min_length,
            max_length=args.max_length,
            exclude_group=args.exclude_group,
            seed=args.seed,
        )
    finally:
        conn.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
