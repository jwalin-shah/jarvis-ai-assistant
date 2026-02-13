#!/usr/bin/env python3
"""One-time cleanup: remove legacy gliner_* facts from the knowledge graph.

These facts were inserted by an earlier extraction run using unmapped GLiNER
predicates (e.g. gliner_person_name, gliner_place). They never rendered in
prompts and add noise to the fact store.

Usage:
    uv run python scripts/cleanup_legacy_facts.py
    uv run python scripts/cleanup_legacy_facts.py --dry-run   # preview only
    uv run python scripts/cleanup_legacy_facts.py --reindex    # also rebuild vec_facts
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove legacy gliner_* facts")
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    parser.add_argument("--reindex", action="store_true", help="Rebuild vec_facts after cleanup")
    args = parser.parse_args()

    from jarvis.db import get_db

    db = get_db()

    # 1. Show current predicate distribution
    print("=== Current fact distribution by predicate ===", flush=True)
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT predicate, COUNT(*) as cnt
            FROM contact_facts
            GROUP BY predicate
            ORDER BY cnt DESC
            """
        ).fetchall()

    total = 0
    gliner_count = 0
    for row in rows:
        predicate, cnt = row["predicate"], row["cnt"]
        marker = " <-- LEGACY" if predicate.startswith("gliner_") else ""
        print(f"  {predicate:30s} {cnt:5d}{marker}", flush=True)
        total += cnt
        if predicate.startswith("gliner_"):
            gliner_count += cnt

    print(f"\nTotal facts: {total}", flush=True)
    print(f"Legacy gliner_* facts: {gliner_count}", flush=True)
    print(f"Facts to keep: {total - gliner_count}", flush=True)

    if gliner_count == 0:
        print("\nNo legacy facts to clean up.", flush=True)
        return

    if args.dry_run:
        print(
            f"\n[DRY RUN] Would delete {gliner_count} facts. Re-run without --dry-run to execute.",
            flush=True,
        )
        return

    # 2. Delete legacy facts
    from jarvis.contacts.fact_storage import delete_facts_by_predicate_prefix

    deleted = delete_facts_by_predicate_prefix("gliner_")
    print(f"\nDeleted {deleted} legacy facts.", flush=True)

    # 3. Verify
    with db.connection() as conn:
        row = conn.execute("SELECT COUNT(*) FROM contact_facts").fetchone()
        remaining = row[0] if row else 0
    print(f"Remaining facts: {remaining}", flush=True)

    # 4. Optionally rebuild vec_facts index
    if args.reindex:
        print("\nRebuilding vec_facts index...", flush=True)
        from jarvis.contacts.fact_index import reindex_all_facts

        indexed = reindex_all_facts()
        print(f"Reindexed {indexed} facts.", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
