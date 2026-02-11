#!/usr/bin/env python3
"""Manual cleanup for round-4 near-miss rows.

This script applies a curated correction map:
- promote selected near_miss rows to positive with explicit expected_candidates
- demote remaining near_miss rows to random_negative (gold_keep=0)

It then rewrites round-4 labeled JSON/CSV and regenerates merged candidate gold.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROUND4_JSON = Path("training_data/gliner_goldset_round4/candidate_gold_labeled.json")
ROUND4_CSV = Path("training_data/gliner_goldset_round4/candidate_gold_labeled.csv")
BASE_MERGED_R3 = Path("training_data/gliner_goldset/candidate_gold_merged_r3.json")
MERGED_R4 = Path("training_data/gliner_goldset/candidate_gold_merged_r4.json")

# Curated keeps from near_miss: sample_id -> one explicit expected candidate.
PROMOTE_NEAR_MISS: dict[str, dict[str, str]] = {
    "r3_cand_0017": {
        "span_text": "canyon creek",
        "span_label": "place",
        "fact_type": "location.current",
    },
    "r3_cand_0024": {
        "span_text": "Boston",
        "span_label": "place",
        "fact_type": "location.future",
    },
    "r3_cand_0030": {
        "span_text": "heads been worse",
        "span_label": "health_condition",
        "fact_type": "health.condition",
    },
    "r3_cand_0048": {
        "span_text": "reading",
        "span_label": "activity",
        "fact_type": "preference.activity",
    },
    "r3_cand_0049": {
        "span_text": "Korea",
        "span_label": "place",
        "fact_type": "location.past",
    },
    "r3_cand_0186": {
        "span_text": "Colorado",
        "span_label": "place",
        "fact_type": "location.past",
    },
    "r3_cand_0206": {
        "span_text": "boston",
        "span_label": "place",
        "fact_type": "location.current",
    },
    "r3_cand_0214": {
        "span_text": "nursing",
        "span_label": "job_role",
        "fact_type": "personal.school",
    },
    "r3_cand_0256": {
        "span_text": "city",
        "span_label": "place",
        "fact_type": "location.current",
    },
    "r3_cand_0297": {
        "span_text": "tufts",
        "span_label": "org",
        "fact_type": "personal.school",
    },
    "r3_cand_0349": {
        "span_text": "fort Worth",
        "span_label": "place",
        "fact_type": "location.current",
    },
}


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=True, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "slice",
        "source_slice",
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
        "gate_score",
        "auto_coarse_type",
        "gold_keep",
        "suggested_candidates_json",
        "expected_candidates_json",
        "gold_notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row.get("sample_id", ""),
                    "slice": row.get("slice", ""),
                    "source_slice": row.get("source_slice", ""),
                    "message_id": row.get("message_id", ""),
                    "chat_rowid": row.get("chat_rowid", ""),
                    "chat_id": row.get("chat_id", ""),
                    "chat_display_name": row.get("chat_display_name", ""),
                    "is_from_me": row.get("is_from_me", ""),
                    "sender_handle": row.get("sender_handle", ""),
                    "message_date": row.get("message_date", ""),
                    "message_text": row.get("message_text", ""),
                    "context_prev": row.get("context_prev", ""),
                    "context_next": row.get("context_next", ""),
                    "gate_score": row.get("gate_score", ""),
                    "auto_coarse_type": row.get("auto_coarse_type", ""),
                    "gold_keep": row.get("gold_keep", ""),
                    "suggested_candidates_json": json.dumps(
                        row.get("suggested_candidates") or [],
                        ensure_ascii=True,
                    ),
                    "expected_candidates_json": json.dumps(
                        row.get("expected_candidates") or [],
                        ensure_ascii=True,
                    ),
                    "gold_notes": row.get("gold_notes", ""),
                }
            )


def merge_expected(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out = list(existing)
    seen = {(c.get("span_text", "").lower(), c.get("span_label", "")) for c in existing}
    for cand in incoming:
        key = (str(cand.get("span_text", "")).lower(), str(cand.get("span_label", "")))
        if key in seen or not key[0]:
            continue
        out.append(
            {
                "span_text": str(cand.get("span_text", "")),
                "span_label": str(cand.get("span_label", "")),
                "fact_type": str(cand.get("fact_type", "other_personal_fact")),
            }
        )
        seen.add(key)
    return out


def main() -> None:
    rows = json.loads(ROUND4_JSON.read_text())

    promoted = 0
    demoted = 0
    for row in rows:
        if row.get("slice") != "near_miss":
            continue
        sid = str(row.get("sample_id", ""))
        if sid in PROMOTE_NEAR_MISS:
            row["expected_candidates"] = [PROMOTE_NEAR_MISS[sid]]
            row["slice"] = "positive"
            row["gold_keep"] = "1"
            promoted += 1
        else:
            row["expected_candidates"] = []
            row["slice"] = "random_negative"
            row["gold_keep"] = "0"
            demoted += 1

    write_json(ROUND4_JSON, rows)
    write_csv(ROUND4_CSV, rows)

    base_rows = json.loads(BASE_MERGED_R3.read_text())
    merged = [dict(r) for r in base_rows]
    idx_by_mid = {int(r["message_id"]): i for i, r in enumerate(merged)}
    added_rows = 0
    overlap_rows = 0

    for row in rows:
        mid = int(row["message_id"])
        if mid not in idx_by_mid:
            merged.append(dict(row))
            idx_by_mid[mid] = len(merged) - 1
            added_rows += 1
            continue
        overlap_rows += 1
        existing = merged[idx_by_mid[mid]]
        existing["expected_candidates"] = merge_expected(
            list(existing.get("expected_candidates") or []),
            list(row.get("expected_candidates") or []),
        )
        if existing["expected_candidates"]:
            existing["slice"] = "positive"

    write_json(MERGED_R4, merged)

    pos = sum(1 for r in rows if r.get("expected_candidates"))
    near = sum(1 for r in rows if r.get("slice") == "near_miss")
    rand_neg = sum(1 for r in rows if r.get("slice") == "random_negative")
    total_cands = sum(len(r.get("expected_candidates") or []) for r in rows)

    print("Round-4 cleanup complete")
    print(f"  promoted near_miss -> positive: {promoted}")
    print(f"  demoted near_miss -> random_negative: {demoted}")
    print(f"  positives: {pos}")
    print(f"  near_miss: {near}")
    print(f"  random_negative: {rand_neg}")
    print(f"  expected candidates total: {total_cands}")
    print(f"  merged rows: {len(merged)} (added={added_rows}, overlap={overlap_rows})")


if __name__ == "__main__":
    main()
