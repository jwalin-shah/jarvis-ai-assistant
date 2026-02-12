#!/usr/bin/env python3
"""Create goldset v5.1: deduplicate overlapping spans within same label.

Changes from v5_no_phantoms:
1. Same-label dedup: When both "dad" and "my dad" exist with same label,
   keep only the shorter form (the actual entity word). The model extracts "dad",
   which matches via substring. Keeping both artificially inflates FN.
   Applies to ALL labels (family_member, activity, health_condition, food_item).
2. Cross-label dedup: When same text has different labels (e.g. "lending tree"
   as both "employer" and "org"), keep the higher-priority label.
"""

from __future__ import annotations

import json
from pathlib import Path

INPUT = Path("training_data/gliner_goldset/goldset_v5_no_phantoms.json")
OUTPUT = Path("training_data/gliner_goldset/goldset_v5.1_deduped.json")

# When both short and long form exist, prefer this label's canonical span
PREFER_SHORT = {"family_member"}

# When same text has multiple labels, prefer this priority (higher = keep)
LABEL_PRIORITY = {
    "org": 10,
    "employer": 9,
    "job_role": 8,
    "activity": 7,
    "health_condition": 6,
    "family_member": 5,
    "food_item": 4,
    "place": 3,
    "current_location": 3,
    "future_location": 3,
    "past_location": 3,
    "friend_name": 2,
    "person_name": 1,
}


def deduplicate_candidates(candidates: list[dict]) -> list[dict]:
    """Remove duplicate spans within a record."""
    if not candidates:
        return candidates

    result = list(candidates)

    # Step 1: Same-label dedup - remove longer form when shorter exists
    # Group by label
    remove_indices = set()
    by_label: dict[str, list[tuple[int, dict]]] = {}
    for i, c in enumerate(result):
        by_label.setdefault(c.get("span_label", ""), []).append((i, c))

    for label, group in by_label.items():
        if len(group) < 2:
            continue
        for i, c1 in group:
            for j, c2 in group:
                if i == j:
                    continue
                t1 = c1["span_text"].lower()
                t2 = c2["span_text"].lower()
                # If t1 is substring of t2 and shorter, remove t2 (the longer one)
                if t1 in t2 and len(t1) < len(t2):
                    remove_indices.add(j)

    # Step 2: Cross-label dedup - same text, different labels -> keep higher priority
    text_to_indices: dict[str, list[int]] = {}
    for i, c in enumerate(result):
        key = c["span_text"].lower().strip()
        text_to_indices.setdefault(key, []).append(i)

    for text, indices in text_to_indices.items():
        if len(indices) <= 1:
            continue
        best_idx = max(indices, key=lambda i: LABEL_PRIORITY.get(result[i]["span_label"], 0))
        for idx in indices:
            if idx != best_idx:
                remove_indices.add(idx)

    # Remove redundant spans
    result = [c for i, c in enumerate(result) if i not in remove_indices]
    return result


def main():
    with open(INPUT) as f:
        data = json.load(f)

    total_before = sum(len(r.get("expected_candidates", [])) for r in data)
    total_removed = 0
    changes = []

    for rec in data:
        cands = rec.get("expected_candidates", [])
        if not cands:
            continue

        new_cands = deduplicate_candidates(cands)
        removed = len(cands) - len(new_cands)
        if removed > 0:
            total_removed += removed
            changes.append(
                f"  {rec['sample_id']}: {len(cands)} -> {len(new_cands)} "
                f"(removed {[c['span_text'] for c in cands if c not in new_cands]})"
            )
            rec["expected_candidates"] = new_cands

    total_after = sum(len(r.get("expected_candidates", [])) for r in data)

    print(f"Records: {len(data)}", flush=True)
    print(f"Spans before: {total_before}", flush=True)
    print(f"Spans after: {total_after}", flush=True)
    print(f"Removed: {total_removed}", flush=True)
    print(f"\nChanges:", flush=True)
    for c in changes:
        print(c, flush=True)

    with open(OUTPUT, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
