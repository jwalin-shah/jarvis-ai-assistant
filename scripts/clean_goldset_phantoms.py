#!/usr/bin/env python3
"""Remove phantom spans from goldset (spans not found in message_text).

Creates a cleaned goldset at training_data/gliner_goldset/goldset_v5_no_phantoms.json
"""

import json
from pathlib import Path

GOLD_PATH = Path("training_data/gliner_goldset/candidate_gold_merged_r4.json")
OUTPUT_PATH = Path("training_data/gliner_goldset/goldset_v5_no_phantoms.json")


def main():
    with open(GOLD_PATH) as f:
        records = json.load(f)

    phantom_count = 0
    records_with_phantoms = 0
    cleaned = []

    for rec in records:
        msg_lower = rec["message_text"].lower()
        candidates = rec.get("expected_candidates") or []

        if not candidates:
            cleaned.append(rec)
            continue

        clean_cands = []
        has_phantom = False
        for cand in candidates:
            span_text = cand.get("span_text", "")
            if span_text.lower() in msg_lower:
                clean_cands.append(cand)
            else:
                phantom_count += 1
                has_phantom = True
                print(
                    f"  PHANTOM: [{rec['sample_id']}] "
                    f"span='{span_text}' not in msg='{rec['message_text'][:60]}...'"
                )

        if has_phantom:
            records_with_phantoms += 1

        # Update record with cleaned candidates
        new_rec = dict(rec)
        new_rec["expected_candidates"] = clean_cands

        # If all candidates were phantoms, update slice to near_miss
        if not clean_cands and candidates:
            new_rec["slice"] = "near_miss"
            new_rec["gold_keep"] = 0

        cleaned.append(new_rec)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    # Stats
    total_spans_before = sum(len(r.get("expected_candidates", [])) for r in records)
    total_spans_after = sum(len(r.get("expected_candidates", [])) for r in cleaned)

    print(f"\nPhantom spans removed: {phantom_count}")
    print(f"Records with phantoms: {records_with_phantoms}")
    print(f"Total spans: {total_spans_before} → {total_spans_after}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
