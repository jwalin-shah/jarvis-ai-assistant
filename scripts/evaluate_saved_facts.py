#!/usr/bin/env python3
"""Heuristic evaluator for saved contact facts.

Labels each saved fact as valid/questionable/invalid to give a quick quality readout.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter

from jarvis.db import get_db

LOW_INFO = {
    "me",
    "you",
    "that",
    "this",
    "it",
    "them",
    "him",
    "her",
    "someone",
    "something",
    "anything",
    "nothing",
}

META_MARKERS = (
    " mentions ",
    " says ",
    " said ",
    " asks ",
    " asked ",
    " talks about ",
    " discussing ",
    " planning ",
)

REACTION_MARKERS = (" an attachment", " tapback", " reacted to ", "liked ", "loved ")
SPEAKER_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 _.'-]{0,30}:\s+")


def classify(subject: str, predicate: str, value: str) -> tuple[str, str]:
    ns = " ".join((subject or "").strip().lower().split())
    np = " ".join((predicate or "").strip().lower().split())
    nv = " ".join((value or "").strip().lower().split())

    if not nv or nv in LOW_INFO:
        return "invalid", "low_info_value"
    if any(marker in nv for marker in REACTION_MARKERS):
        return "invalid", "reaction_artifact"
    if SPEAKER_PREFIX_RE.match(value or ""):
        return "invalid", "speaker_prefix_value"
    if "group" in ns or "chat" in ns:
        return "invalid", "group_subject"
    if not np or len(np) < 2:
        return "invalid", "empty_predicate"
    pred_parts = [p for p in np.split("_") if p]
    if len(pred_parts) > 5:
        return "invalid", "overlong_predicate"
    if any(ch.isdigit() for ch in np):
        return "questionable", "numeric_predicate"
    if np == "has_fact":
        wrapped = f" {nv} "
        if len(nv.split()) > 14 or any(m in wrapped for m in META_MARKERS):
            return "invalid", "generic_has_fact_meta"
        return "questionable", "generic_has_fact"
    if len(pred_parts) > 3:
        return "questionable", "complex_predicate"
    if len(nv.split()) > 20:
        return "questionable", "long_value"
    return "valid", "passes_heuristics"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved fact quality (heuristic)")
    parser.add_argument("--hours", type=int, default=24, help="Only facts saved in last N hours")
    parser.add_argument(
        "--per-chat", type=int, default=3, help="Show up to N sample facts per chat"
    )
    args = parser.parse_args()

    db = get_db()
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT contact_id, subject, predicate, value, confidence
            FROM contact_facts
            WHERE extracted_at >= datetime('now', ?)
            ORDER BY extracted_at DESC
            """,
            (f"-{args.hours} hours",),
        ).fetchall()

    if not rows:
        print("No saved facts found in time window.")
        return

    label_counts = Counter()
    reason_counts = Counter()
    per_chat_samples: dict[str, list[str]] = {}

    for row in rows:
        label, reason = classify(row["subject"], row["predicate"], row["value"])
        label_counts[label] += 1
        reason_counts[reason] += 1
        if len(per_chat_samples.get(row["contact_id"], [])) < args.per_chat:
            per_chat_samples.setdefault(row["contact_id"], []).append(
                f"{label.upper()}: {row['subject']} | {row['predicate']} | {row['value']} (c={row['confidence']})"
            )

    total = sum(label_counts.values())
    print("=" * 88)
    print(f"SAVED FACT QUALITY (last {args.hours}h)")
    print("=" * 88)
    for label in ("valid", "questionable", "invalid"):
        count = label_counts[label]
        pct = (count / total) * 100 if total else 0
        print(f"{label:12s} {count:4d}  ({pct:5.1f}%)")

    print("\nTop reasons")
    for reason, count in reason_counts.most_common(8):
        print(f"- {reason}: {count}")

    print("\nSamples by chat")
    for chat_id, samples in per_chat_samples.items():
        print(f"\n[{chat_id}]")
        for s in samples:
            print(f"- {s}")


if __name__ == "__main__":
    main()
