#!/usr/bin/env python3
"""Trace fact generation across pipeline stages for one chat.

Shows:
1) pass-1 claims (fact_pass1_claims_log)
2) raw structured triples (fact_candidates_log)
3) saved facts (contact_facts)
4) pipeline counters (fact_pipeline_metrics)
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from jarvis.db import get_db


def _print_metrics(chat_id: str, stage: str, limit: int) -> None:
    db = get_db()
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT created_at, raw_count, prefilter_rejected, verifier_rejected,
                   semantic_dedup_rejected, unique_conflict_rejected, saved_count
            FROM fact_pipeline_metrics
            WHERE chat_id = ? AND stage = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (chat_id, stage, limit),
        ).fetchall()

    print("\n=== METRICS ===")
    if not rows:
        print("No rows found.")
        return
    for row in rows:
        print(
            f"[{row['created_at']}] raw={row['raw_count']} pref={row['prefilter_rejected']} "
            f"ver={row['verifier_rejected']} sem={row['semantic_dedup_rejected']} "
            f"uniq={row['unique_conflict_rejected']} saved={row['saved_count']}"
        )


def _load_stage_rows(chat_id: str, stage: str, segment_id: int | None, limit: int) -> dict[int, list[str]]:
    db = get_db()
    out: dict[int, list[str]] = defaultdict(list)
    with db.connection() as conn:
        if stage == "pass1":
            if segment_id is None:
                rows = conn.execute(
                    """
                    SELECT segment_id, created_at, claim_text
                    FROM fact_pass1_claims_log
                    WHERE chat_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (chat_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT segment_id, created_at, claim_text
                    FROM fact_pass1_claims_log
                    WHERE chat_id = ? AND segment_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (chat_id, segment_id, limit),
                ).fetchall()
            for row in rows:
                sid = row["segment_id"] or -1
                out[sid].append(f"[{row['created_at']}] {row['claim_text']}")

        elif stage == "raw":
            if segment_id is None:
                rows = conn.execute(
                    """
                    SELECT segment_id, created_at, subject, predicate, value, confidence
                    FROM fact_candidates_log
                    WHERE chat_id = ? AND log_stage = 'segment_pipeline'
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (chat_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT segment_id, created_at, subject, predicate, value, confidence
                    FROM fact_candidates_log
                    WHERE chat_id = ? AND log_stage = 'segment_pipeline' AND segment_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (chat_id, segment_id, limit),
                ).fetchall()
            for row in rows:
                sid = row["segment_id"] or -1
                out[sid].append(
                    f"[{row['created_at']}] {row['subject']} | {row['predicate']} | "
                    f"{row['value']} | c={row['confidence']}"
                )

        elif stage == "saved":
            if segment_id is None:
                rows = conn.execute(
                    """
                    SELECT segment_id, extracted_at, subject, predicate, value, confidence
                    FROM contact_facts
                    WHERE contact_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (chat_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT segment_id, extracted_at, subject, predicate, value, confidence
                    FROM contact_facts
                    WHERE contact_id = ? AND segment_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (chat_id, segment_id, limit),
                ).fetchall()
            for row in rows:
                sid = row["segment_id"] or -1
                out[sid].append(
                    f"[{row['extracted_at']}] {row['subject']} | {row['predicate']} | "
                    f"{row['value']} | c={row['confidence']}"
                )
        else:
            raise ValueError(f"Unknown stage: {stage}")

    return out


def _print_stage(stage_name: str, rows_by_segment: dict[int, list[str]]) -> None:
    print(f"\n=== {stage_name} ===")
    if not rows_by_segment:
        print("No rows found.")
        return
    for sid in sorted(rows_by_segment.keys()):
        print(f"\n[segment_id={sid}]")
        for line in rows_by_segment[sid]:
            print(f"- {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace fact pipeline stages for one chat")
    parser.add_argument("--chat-id", required=True, help="Exact chat_id/contact_id")
    parser.add_argument("--segment-id", type=int, default=None, help="Optional segment_id filter")
    parser.add_argument("--limit", type=int, default=30, help="Rows per stage")
    parser.add_argument(
        "--stages",
        default="pass1,raw,saved,metrics",
        help="Comma-separated stages: pass1,raw,saved,metrics",
    )
    args = parser.parse_args()

    stage_set = {s.strip().lower() for s in args.stages.split(",") if s.strip()}
    print(f"Tracing chat_id={args.chat_id} segment_id={args.segment_id} limit={args.limit}")

    if "metrics" in stage_set:
        _print_metrics(args.chat_id, "segment_pipeline", args.limit)
    if "pass1" in stage_set:
        _print_stage("PASS1 CLAIMS", _load_stage_rows(args.chat_id, "pass1", args.segment_id, args.limit))
    if "raw" in stage_set:
        _print_stage("RAW TRIPLES", _load_stage_rows(args.chat_id, "raw", args.segment_id, args.limit))
    if "saved" in stage_set:
        _print_stage("SAVED FACTS", _load_stage_rows(args.chat_id, "saved", args.segment_id, args.limit))


if __name__ == "__main__":
    main()
