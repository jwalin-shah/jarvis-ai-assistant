#!/usr/bin/env python3
"""
Clean GLiNER goldset for fact extraction evaluation.

Issues addressed:
1. Remove expected_candidates where span_text not found in message_text
2. Deduplicate entities (same person/place mentioned with different text)
3. Flag records where gold label only makes sense with context
4. Track all changes made

Usage:
    python scripts/clean_goldset.py
"""

import json
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple


def similarity_ratio(a: str, b: str) -> float:
    """Calculate string similarity (0.0 to 1.0)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def deduplicate_candidates(
    candidates: List[Dict[str, Any]],
    message_text: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Deduplicate entities, keeping the more specific version.
    Returns: (cleaned_candidates, dedup_notes)
    """
    if not candidates:
        return [], []

    dedup_notes = []

    # Group by (entity type, label) to find duplicates
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    for idx, candidate in enumerate(candidates):
        span_text = candidate["span_text"].lower().strip()
        span_label = candidate["span_label"]
        groups[(span_text, span_label)].append(idx)

    # Also find near-duplicates (e.g., "brother" vs "My brother")
    indices_to_remove = set()

    for (text1, label1), indices1 in groups.items():
        for (text2, label2), indices2 in groups.items():
            if label1 != label2:
                continue

            # Check for near-duplicates
            sim = similarity_ratio(text1, text2)
            if 0.6 < sim < 1.0:  # Similar but not identical
                # Keep the longer (more specific) one
                if len(text1) > len(text2):
                    indices_to_remove.update(indices2)
                    dedup_notes.append(
                        f"Removed near-duplicate '{text2}' (kept '{text1}' as more specific)"
                    )
                elif len(text2) > len(text1):
                    indices_to_remove.update(indices1)
                    dedup_notes.append(
                        f"Removed near-duplicate '{text1}' (kept '{text2}' as more specific)"
                    )

    # Remove exact duplicates (keep first occurrence)
    seen = set()
    for idx, candidate in enumerate(candidates):
        key = (candidate["span_text"].lower(), candidate["span_label"])
        if key in seen:
            indices_to_remove.add(idx)
            dedup_notes.append(f"Removed exact duplicate: {candidate['span_text']}")
        else:
            seen.add(key)

    cleaned = [c for i, c in enumerate(candidates) if i not in indices_to_remove]
    return cleaned, dedup_notes


def clean_record(record: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Clean a single record.
    Returns: (cleaned_record, changes_dict)
    """
    changes = {
        "record_id": record["sample_id"],
        "removed_candidates": [],
        "deduplicated": [],
        "needs_context": False,
        "context_reason": "",
    }

    message_text = record["message_text"]
    expected_candidates = record.get("expected_candidates", [])

    # Step 1: Remove candidates where span_text not found in message_text
    valid_candidates = []
    for candidate in expected_candidates:
        span_text = candidate["span_text"]
        if span_text in message_text:
            valid_candidates.append(candidate)
        else:
            changes["removed_candidates"].append({
                "span_text": span_text,
                "reason": "NOT_IN_MESSAGE_TEXT",
            })

    # Step 2: Deduplicate remaining candidates
    deduplicated, dedup_notes = deduplicate_candidates(valid_candidates, message_text)
    if dedup_notes:
        changes["deduplicated"] = dedup_notes

    # Step 3: Check if gold label only makes sense with context
    if record.get("gold_keep") == 1:
        message_is_short = len(message_text.split()) <= 5
        gold_notes = record.get("gold_notes", "").lower()

        # Heuristics: flag if message is very short or notes suggest context-dependence
        context_keywords = [
            "context", "previous", "prior", "before", "as mentioned",
            "referring to", "reference to", "implicit", "unclear without",
        ]

        if message_is_short or any(kw in gold_notes for kw in context_keywords):
            # Additional check: single-word messages like "Vestibular"
            if len(message_text.split()) == 1 and message_text[0].isupper():
                changes["needs_context"] = True
                changes["context_reason"] = "Single-word isolated message requires context"
            elif any(kw in gold_notes for kw in context_keywords):
                changes["needs_context"] = True
                changes["context_reason"] = f"Gold notes suggest context-dependence: {gold_notes[:100]}"

    # Create cleaned record
    cleaned_record = record.copy()
    cleaned_record["expected_candidates"] = deduplicated
    if changes["needs_context"]:
        cleaned_record["needs_context"] = True
        cleaned_record["context_reason"] = changes["context_reason"]

    return cleaned_record, changes


def main():
    goldset_path = Path(
        "/Users/jwalinshah/projects/jarvis-ai-assistant/"
        "training_data/gliner_goldset/candidate_gold_merged_r4.json"
    )
    output_path = Path(
        "/Users/jwalinshah/projects/jarvis-ai-assistant/"
        "training_data/gliner_goldset/candidate_gold_merged_r4_clean.json"
    )
    report_path = Path(
        "/Users/jwalinshah/projects/jarvis-ai-assistant/"
        "training_data/gliner_goldset/cleaning_report.md"
    )

    print(f"Loading goldset from {goldset_path}")
    with open(goldset_path) as f:
        records = json.load(f)

    print(f"Processing {len(records)} records...")
    cleaned_records = []
    all_changes = []

    for i, record in enumerate(records):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(records)}", flush=True)

        cleaned_record, changes = clean_record(record)
        cleaned_records.append(cleaned_record)
        all_changes.append(changes)

    # Write cleaned goldset
    print(f"Writing cleaned goldset to {output_path}")
    with open(output_path, "w") as f:
        json.dump(cleaned_records, f, indent=2)

    # Generate cleaning report
    print(f"Generating cleaning report to {report_path}")

    total_records = len(records)
    records_with_removals = sum(
        1 for c in all_changes if c["removed_candidates"]
    )
    total_removed = sum(
        len(c["removed_candidates"]) for c in all_changes
    )
    records_with_dedup = sum(
        1 for c in all_changes if c["deduplicated"]
    )
    total_dedup = sum(
        len(c["deduplicated"]) for c in all_changes
    )
    records_flagged = sum(
        1 for c in all_changes if c["needs_context"]
    )

    report = f"""# GLiNER Goldset Cleaning Report

**Date**: 2026-02-11
**Source**: training_data/gliner_goldset/candidate_gold_merged_r4.json
**Output**: training_data/gliner_goldset/candidate_gold_merged_r4_clean.json

## Summary

- **Total records**: {total_records}
- **Records with removed candidates**: {records_with_removals} ({100*records_with_removals/total_records:.1f}%)
- **Total candidates removed**: {total_removed}
- **Records with deduplications**: {records_with_dedup} ({100*records_with_dedup/total_records:.1f}%)
- **Total duplicates removed**: {total_dedup}
- **Records flagged "needs_context"**: {records_flagged} ({100*records_flagged/total_records:.1f}%)

## Issues Fixed

### 1. Span Text Not in Message (Removed: {total_removed})

Candidates where `span_text` was not found (as substring) in `message_text`.
These are unfair to evaluate against since the model only sees `message_text`.

**Examples of removed candidates:**
"""

    removal_examples = []
    for change in all_changes:
        if change["removed_candidates"]:
            removal_examples.append(change)
            if len(removal_examples) >= 5:
                break

    for change in removal_examples:
        record = next(r for r in records if r["sample_id"] == change["record_id"])
        report += f"\n**{change['record_id']}**: '{record['message_text']}'\n"
        for removed in change["removed_candidates"]:
            report += f"  - Removed '{removed['span_text']}' (not found in message_text)\n"

    report += f"""

### 2. Duplicate Entities (Removed: {total_dedup})

Same person/place mentioned with slightly different text (e.g., "brother" vs "My brother").
Kept the more specific/longer version.

**Examples of deduplication:**
"""

    dedup_examples = []
    for change in all_changes:
        if change["deduplicated"]:
            dedup_examples.append(change)
            if len(dedup_examples) >= 5:
                break

    for change in dedup_examples:
        record = next(r for r in records if r["sample_id"] == change["record_id"])
        report += f"\n**{change['record_id']}**: '{record['message_text']}'\n"
        for dedup_note in change["deduplicated"]:
            report += f"  - {dedup_note}\n"

    report += f"""

### 3. Context-Dependent Records (Flagged: {records_flagged})

Records where the gold label only makes sense with surrounding context.
Flagged with `needs_context=true` for downstream evaluation.

**Examples of context-dependent records:**
"""

    context_examples = []
    for change in all_changes:
        if change["needs_context"]:
            context_examples.append(change)
            if len(context_examples) >= 5:
                break

    for change in context_examples:
        record = next(r for r in records if r["sample_id"] == change["record_id"])
        report += f"\n**{change['record_id']}**: '{record['message_text']}'\n"
        report += f"  - Reason: {change['context_reason']}\n"

    report += f"""

## Evaluation Recommendations

1. **Span extraction evaluation**:
   - Use cleaned dataset to fairly evaluate model span detection
   - Model was trained on full message_text, so this is appropriate ground truth
   - Original dataset penalized model for spans in context_prev/context_next

2. **Context-dependent labels**:
   - When evaluating fact type (hobby, preference, etc.), use records with `needs_context=false`
   - The context-flagged records require human knowledge of prior conversation
   - ML model should NOT be expected to infer these without context

3. **Deduplication impact**:
   - Removed {total_dedup} near-duplicates that would inflate metrics
   - Candidates are now unique per entity mention
   - Fair comparison between exact string matches

## Statistics

### Candidates Removed by Reason
- **span_text not found**: {total_removed} candidates from {records_with_removals} records
- **Exact/near duplicates**: {total_dedup} candidates from {records_with_dedup} records

### Average Candidates per Record
"""

    avg_before = sum(
        len(r.get("expected_candidates", [])) for r in records
    ) / total_records
    avg_after = sum(
        len(r.get("expected_candidates", [])) for r in cleaned_records
    ) / total_records

    report += f"""- **Before cleaning**: {avg_before:.2f}
- **After cleaning**: {avg_after:.2f}
- **Reduction**: {100*(avg_before - avg_after)/avg_before:.1f}%

## Implementation Notes

Cleaning logic applied to all {total_records} records:

1. **Span substring matching**: Case-sensitive substring search in message_text
2. **Deduplication strategy**:
   - Exact duplicates: removed (kept first occurrence)
   - Near-duplicates: removed if similarity > 60% and < 100%
   - Kept longer/more specific version in case of conflict
3. **Context flagging heuristics**:
   - Single-word messages (all-caps, e.g., "Vestibular")
   - Messages with ≤5 words + context-suggestion keywords in gold_notes
   - Keywords: "context", "previous", "prior", "before", "referring to", "implicit"

"""

    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n✓ Cleaning complete!")
    print(f"  - Cleaned goldset: {output_path}")
    print(f"  - Report: {report_path}")
    print(f"\nSummary:")
    print(f"  - Records: {total_records}")
    print(f"  - Candidates removed: {total_removed} (from {records_with_removals} records)")
    print(f"  - Duplicates removed: {total_dedup} (from {records_with_dedup} records)")
    print(f"  - Context-flagged: {records_flagged}")


if __name__ == "__main__":
    main()
