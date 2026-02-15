#!/usr/bin/env python3
"""Compare pass-1 claims vs saved structured facts for a chat."""

from __future__ import annotations

import argparse
import sqlite3


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pass-1 claims with final facts")
    parser.add_argument(
        "--db", default="/Users/jwalinshah/.jarvis/jarvis.db", help="Path to jarvis.db"
    )
    parser.add_argument("--chat-id", required=True, help="chat_id/contact_id to inspect")
    parser.add_argument(
        "--limit-segments", type=int, default=20, help="How many recent segments to show"
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        segs = conn.execute(
            """
            SELECT id, segment_id, start_time, end_time
            FROM conversation_segments
            WHERE chat_id = ?
            ORDER BY start_time DESC
            LIMIT ?
            """,
            (args.chat_id, args.limit_segments),
        ).fetchall()
        if not segs:
            print("No segments found for chat.")
            return

        seg_ids = [str(r["id"]) for r in segs]
        placeholders = ",".join("?" * len(seg_ids))

        try:
            pass1_rows = conn.execute(
                f"""
                SELECT segment_id, claim_text, created_at
                FROM fact_pass1_claims_log
                WHERE chat_id = ? AND segment_id IN ({placeholders})
                ORDER BY created_at DESC
                """,
                [args.chat_id, *seg_ids],
            ).fetchall()
        except sqlite3.OperationalError:
            pass1_rows = []

        final_rows = conn.execute(
            f"""
            SELECT segment_id, subject, predicate, value, confidence, extracted_at
            FROM contact_facts
            WHERE contact_id = ? AND segment_id IN ({placeholders})
            ORDER BY extracted_at DESC
            """,
            [args.chat_id, *seg_ids],
        ).fetchall()

        pass1_by_seg: dict[int, list[str]] = {}
        for r in pass1_rows:
            pass1_by_seg.setdefault(int(r["segment_id"]), []).append(r["claim_text"])

        final_by_seg: dict[int, list[str]] = {}
        for r in final_rows:
            final_by_seg.setdefault(int(r["segment_id"]), []).append(
                f"{r['subject']} | {r['predicate']} | {r['value']} (c={r['confidence']:.3f})"
            )

        for seg in segs:
            sid = int(seg["id"])
            print("=" * 96)
            print(f"Segment DB ID: {sid} | Segment UUID: {seg['segment_id']}")
            print(f"Time: {seg['start_time']} -> {seg['end_time']}")
            print("- PASS 1 CLAIMS -")
            claims = pass1_by_seg.get(sid, [])
            if claims:
                for c in claims[:25]:
                    print(f"  - {c}")
            else:
                print("  (none)")
            print("- FINAL FACTS -")
            facts = final_by_seg.get(sid, [])
            if facts:
                for f in facts[:25]:
                    print(f"  - {f}")
            else:
                print("  (none)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
