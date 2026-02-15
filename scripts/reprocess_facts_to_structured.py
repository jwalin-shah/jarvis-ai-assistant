#!/usr/bin/env python3
"""Reprocess all existing facts from free-text to structured format.

This script:
1. Loads all existing facts from contact_facts
2. Parses free-text values into structured triples
3. Clears old facts and inserts restructured ones
4. Reports improvement metrics

Usage:
    uv run python scripts/reprocess_facts_to_structured.py
    uv run python scripts/reprocess_facts_to_structured.py --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from collections import Counter

from jarvis.contacts.fact_storage import get_all_facts
from jarvis.contacts.structured_extractor import restructure_existing_fact
from jarvis.db import get_db


def main():
    parser = argparse.ArgumentParser(description="Reprocess facts to structured format")
    parser.add_argument("--dry-run", action="store_true", help="Don't modify DB")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Min confidence to keep")
    args = parser.parse_args()

    print("=" * 60)
    print("REPROCESSING FACTS TO STRUCTURED FORMAT")
    print("=" * 60)

    # Load all existing facts
    print("\n1. Loading existing facts...")
    existing_facts = get_all_facts()
    print(f"   Found {len(existing_facts)} facts")

    # Analyze current state
    print("\n2. Analyzing current state...")
    empty_predicates = sum(1 for f in existing_facts if not f.predicate)
    contact_subjects = sum(1 for f in existing_facts if f.subject == "Contact")
    print(f"   Facts with empty predicates: {empty_predicates}")
    print(f"   Facts with 'Contact' subject: {contact_subjects}")

    # Restructure all facts
    print("\n3. Restructuring facts...")
    restructured = []
    stats = {"parsed": 0, "unparsed": 0, "compound_split": 0}

    for fact in existing_facts:
        if fact.confidence < args.min_confidence:
            continue

        new_facts = restructure_existing_fact(fact)

        # Track stats
        if len(new_facts) > 1:
            stats["compound_split"] += 1

        for nf in new_facts:
            if nf.predicate and nf.predicate != "note":
                stats["parsed"] += 1
            else:
                stats["unparsed"] += 1
            restructured.append(nf)

    print(f"   Restructured into {len(restructured)} facts")
    print(f"   Successfully parsed: {stats['parsed']}")
    print(f"   Unparsed (stored as notes): {stats['unparsed']}")
    print(f"   Compound facts split: {stats['compound_split']}")

    # Deduplicate
    print("\n4. Deduplicating...")
    seen = set()
    unique_facts = []
    for f in restructured:
        key = (f.contact_id, f.category, f.subject, f.predicate, f.value.lower().strip())
        if key not in seen:
            seen.add(key)
            unique_facts.append(f)

    duplicates = len(restructured) - len(unique_facts)
    print(f"   Removed {duplicates} duplicates")
    print(f"   Final fact count: {len(unique_facts)}")

    # Analyze new distribution
    print("\n5. New fact distribution:")
    categories = Counter(f.category for f in unique_facts)
    for cat, count in categories.most_common():
        print(f"   • {cat}: {count}")

    predicates = Counter(f.predicate for f in unique_facts if f.predicate)
    print("\n   Top predicates:")
    for pred, count in predicates.most_common(10):
        print(f"   • {pred}: {count}")

    subjects = Counter(f.subject for f in unique_facts)
    print("\n   Top subjects:")
    for subj, count in subjects.most_common(10):
        print(f"   • {subj}: {count}")

    # Sample facts
    print("\n6. Sample restructured facts:")
    for f in unique_facts[:5]:
        print(f"   • [{f.category}] {f.subject} --{f.predicate}--> {f.value[:50]}")

    # Apply changes
    if args.dry_run:
        print("\n⚠️  DRY RUN - No changes made")
        return

    print("\n7. Applying changes to database...")

    # Group by contact for batch processing
    facts_by_contact = {}
    for f in unique_facts:
        facts_by_contact.setdefault(f.contact_id, []).append(f)

    db = get_db()
    total_inserted = 0

    with db.connection() as conn:
        # Disable foreign key constraints for speed
        conn.execute("PRAGMA foreign_keys = OFF")

        try:
            # Clear existing facts
            print("   Clearing old facts...")
            conn.execute("DELETE FROM contact_facts")

            # Insert restructured facts
            print(f"   Inserting {len(unique_facts)} restructured facts...")

            batch_size = 100
            for i in range(0, len(unique_facts), batch_size):
                batch = unique_facts[i : i + batch_size]

                fact_data = [
                    (
                        f.contact_id,
                        f.category,
                        f.subject,
                        f.predicate,
                        f.value,
                        f.confidence,
                        f.source_message_id,
                        f.source_text[:500] if f.source_text else "",
                        f.extracted_at or "",
                        None,  # linked_contact_id
                        None,  # valid_from
                        None,  # valid_until
                        f.attribution,
                        None,  # segment_id
                    )
                    for f in batch
                ]

                conn.executemany(
                    """
                    INSERT OR IGNORE INTO contact_facts
                    (contact_id, category, subject, predicate, value, confidence,
                     source_message_id, source_text, extracted_at, linked_contact_id,
                     valid_from, valid_until, attribution, segment_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    fact_data,
                )
                total_inserted += len(batch)

                if (i // batch_size) % 10 == 0:
                    print(f"      ... {total_inserted}/{len(unique_facts)}")

            conn.execute("PRAGMA foreign_keys = ON")
            conn.commit()

        except Exception as e:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.rollback()
            print(f"   ❌ Error: {e}")
            raise

    print(f"\n✅ Successfully inserted {total_inserted} structured facts")
    print(f"   Improvement: {len(existing_facts)} → {total_inserted} facts")

    # Verify
    print("\n8. Verification:")
    final_facts = get_all_facts()
    with_predicates = sum(1 for f in final_facts if f.predicate)
    print(
        f"   Facts with predicates: {with_predicates}/{len(final_facts)} ({100 * with_predicates // len(final_facts)}%)"
    )


if __name__ == "__main__":
    main()
