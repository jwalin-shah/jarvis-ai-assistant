#!/usr/bin/env python3
"""Dashboard-style query output for fact pipeline quality and efficiency.

Shows:
1) saved_precision proxy by chat/stage
2) save_rate (saved/raw) by chat/stage
3) reject_reason counters
"""

from __future__ import annotations

import argparse
import sqlite3


def main() -> None:
    parser = argparse.ArgumentParser(description="Fact pipeline metrics dashboard")
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to jarvis.db (default: resolve from jarvis.db.get_db())",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Window size in hours (default: 24)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Rows to show for per-chat table (default: 20)",
    )
    args = parser.parse_args()

    if args.db_path:
        db_path = args.db_path
    else:
        from jarvis.db import get_db

        db_path = str(get_db().db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # Ensure table exists before querying.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fact_pipeline_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contact_id TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                raw_count INTEGER DEFAULT 0,
                prefilter_rejected INTEGER DEFAULT 0,
                verifier_rejected INTEGER DEFAULT 0,
                semantic_dedup_rejected INTEGER DEFAULT 0,
                unique_conflict_rejected INTEGER DEFAULT 0,
                saved_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        window_expr = f"-{max(1, args.hours)} hours"

        print("=" * 96)
        print(f"FACT PIPELINE DASHBOARD (last {args.hours}h)")
        print("=" * 96)

        print("\nOverall")
        overall = conn.execute(
            """
            SELECT
                COALESCE(SUM(raw_count), 0) AS raw_count,
                COALESCE(SUM(prefilter_rejected), 0) AS prefilter_rejected,
                COALESCE(SUM(verifier_rejected), 0) AS verifier_rejected,
                COALESCE(SUM(semantic_dedup_rejected), 0) AS semantic_dedup_rejected,
                COALESCE(SUM(unique_conflict_rejected), 0) AS unique_conflict_rejected,
                COALESCE(SUM(saved_count), 0) AS saved_count
            FROM fact_pipeline_metrics
            WHERE created_at >= datetime('now', ?)
            """,
            (window_expr,),
        ).fetchone()
        raw = overall["raw_count"]
        prefilter = overall["prefilter_rejected"]
        verifier = overall["verifier_rejected"]
        semantic = overall["semantic_dedup_rejected"]
        unique = overall["unique_conflict_rejected"]
        saved = overall["saved_count"]
        kept_after_quality = max(0, raw - prefilter - verifier)
        saved_precision = (saved / kept_after_quality) if kept_after_quality else 0.0
        save_rate = (saved / raw) if raw else 0.0
        print(f"raw={raw} saved={saved} save_rate={save_rate:.3f} saved_precision={saved_precision:.3f}")
        print(
            f"reject_reason: prefilter={prefilter} verifier={verifier} "
            f"semantic_dedup={semantic} unique_conflict={unique}"
        )

        print("\nBy Chat + Stage")
        rows = conn.execute(
            """
            SELECT
                chat_id,
                stage,
                SUM(raw_count) AS raw_count,
                SUM(prefilter_rejected) AS prefilter_rejected,
                SUM(verifier_rejected) AS verifier_rejected,
                SUM(semantic_dedup_rejected) AS semantic_dedup_rejected,
                SUM(unique_conflict_rejected) AS unique_conflict_rejected,
                SUM(saved_count) AS saved_count
            FROM fact_pipeline_metrics
            WHERE created_at >= datetime('now', ?)
            GROUP BY chat_id, stage
            ORDER BY SUM(raw_count) DESC
            LIMIT ?
            """,
            (window_expr, args.limit),
        ).fetchall()
        header = (
            f"{'chat_id':45} {'stage':16} {'raw':>5} {'saved':>5} "
            f"{'save_rate':>9} {'saved_prec':>11} {'pref':>6} {'ver':>6} {'sem':>6} {'uniq':>6}"
        )
        print(header)
        print("-" * len(header))
        for r in rows:
            raw_c = int(r["raw_count"] or 0)
            pref_c = int(r["prefilter_rejected"] or 0)
            ver_c = int(r["verifier_rejected"] or 0)
            sem_c = int(r["semantic_dedup_rejected"] or 0)
            uniq_c = int(r["unique_conflict_rejected"] or 0)
            saved_c = int(r["saved_count"] or 0)
            kept = max(0, raw_c - pref_c - ver_c)
            sr = (saved_c / raw_c) if raw_c else 0.0
            sp = (saved_c / kept) if kept else 0.0
            print(
                f"{(r['chat_id'] or '')[:45]:45} {(r['stage'] or '')[:16]:16} "
                f"{raw_c:5d} {saved_c:5d} {sr:9.3f} {sp:11.3f} {pref_c:6d} {ver_c:6d} {sem_c:6d} {uniq_c:6d}"
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()

