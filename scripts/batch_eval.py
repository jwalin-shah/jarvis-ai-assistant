#!/usr/bin/env python3
"""Batch evaluation script - generates responses across diverse contacts.

Picks contacts with varied relationships, formality levels, and message counts,
then routes a recent message from each through the full pipeline. Outputs a
table showing the spread of responses.

Usage:
    uv run python scripts/batch_eval.py [--limit N] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("JARVIS_ENABLE_CONTACT_PROFILE_CONTEXT", "1")


def main():
    parser = argparse.ArgumentParser(description="Batch response evaluation")
    parser.add_argument("--limit", type=int, default=20, help="Max contacts to evaluate")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    print("=" * 80)
    print("JARVIS Batch Response Evaluation")
    print("=" * 80)

    # Step 1: Load database and get contacts with message counts
    print("\n[1/4] Loading contacts and profiles...")
    from jarvis.contacts.contact_profile import ContactProfile, load_profile
    from jarvis.db import get_db

    db = get_db()
    db.init_schema()

    contacts = db.list_contacts(limit=500)
    print(f"  Found {len(contacts)} contacts")

    # Enrich with profile data and pair counts
    # Profiles are keyed by full iMessage IDs (e.g. "SMS;-;+1234") but DB stores
    # just the identifier ("+1234"). Try all common prefixes.
    def try_load_profile(chat_id: str) -> ContactProfile | None:
        if not chat_id:
            return None
        # Try direct match first
        p = load_profile(chat_id)
        if p:
            return p
        # Try with prefixes
        for prefix in ["iMessage;-;", "SMS;-;", "RCS;-;"]:
            p = load_profile(f"{prefix}{chat_id}")
            if p:
                return p
        return None

    enriched = []
    for c in contacts:
        profile = try_load_profile(c.chat_id)
        if not profile:
            continue

        # Get pair count for this contact (pairs keyed by chat_id, not contact_id)
        with db.connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) as cnt FROM pairs WHERE chat_id = ?", (c.chat_id,)
            )
            row = cursor.fetchone()
            pair_count = row["cnt"] if row else 0

        if pair_count < 5:
            continue

        enriched.append(
            {
                "contact": c,
                "profile": profile,
                "pair_count": pair_count,
            }
        )

    print(f"  {len(enriched)} contacts with profiles and 5+ pairs")

    # Step 2: Select diverse sample
    print("\n[2/4] Selecting diverse sample...")

    # Sort by pair count descending, take top contacts
    enriched.sort(key=lambda x: x["pair_count"], reverse=True)

    # Try to get diversity across relationships and formality
    selected = []
    seen_relationships = {}
    seen_formalities = {}

    for item in enriched:
        rel = item["profile"].relationship or "unknown"
        formality = item["profile"].formality or "casual"

        # Prefer contacts we haven't seen the relationship/formality combo for
        combo = f"{rel}:{formality}"
        count = seen_relationships.get(combo, 0)

        if count < 2 and len(selected) < args.limit:
            selected.append(item)
            seen_relationships[combo] = count + 1
            seen_formalities[formality] = seen_formalities.get(formality, 0) + 1

    # Fill remaining slots with highest-pair-count contacts
    if len(selected) < args.limit:
        for item in enriched:
            if item not in selected and len(selected) < args.limit:
                selected.append(item)

    print(f"  Selected {len(selected)} contacts for evaluation")

    # Step 3: Get a recent trigger message for each contact
    print("\n[3/4] Fetching recent messages and generating responses...")
    print("  (Loading model on first generation...)\n")

    from jarvis.router import ReplyRouter

    router = ReplyRouter(db=db)

    results = []
    total_gen_time = 0

    for i, item in enumerate(selected):
        c = item["contact"]
        profile = item["profile"]

        # Get a recent trigger from pairs table (pairs keyed by chat_id)
        with db.connection() as conn:
            cursor = conn.execute(
                """SELECT trigger_text, response_text, chat_id
                   FROM pairs
                   WHERE chat_id = ?
                   ORDER BY trigger_timestamp DESC
                   LIMIT 1""",
                (c.chat_id,),
            )
            row = cursor.fetchone()

        if not row or not row["trigger_text"]:
            continue

        trigger = row["trigger_text"]
        original_response = row["response_text"]
        chat_id = row["chat_id"] or c.chat_id

        # Generate response
        gen_start = time.perf_counter()
        try:
            result = router.route(
                incoming=trigger,
                contact_id=c.id,
                chat_id=chat_id,
            )
            gen_time = (time.perf_counter() - gen_start) * 1000
            total_gen_time += gen_time

            results.append(
                {
                    "contact_name": c.display_name or "(unknown)",
                    "relationship": profile.relationship or "unknown",
                    "formality": profile.formality or "casual",
                    "formality_score": profile.formality_score,
                    "pair_count": item["pair_count"],
                    "msg_count": profile.message_count,
                    "trigger": trigger,
                    "original": original_response,
                    "generated": result.get("response", ""),
                    "confidence": result.get("confidence", ""),
                    "similarity": result.get("similarity_score", 0.0),
                    "type": result.get("type", ""),
                    "gen_time_ms": gen_time,
                }
            )

            # Progress indicator
            name = (c.display_name or "(unknown)")[:15]
            gen_text = result.get("response", "")[:50]
            print(f"  [{i+1}/{len(selected)}] {name:<15} | {gen_time:>6.0f}ms | {gen_text}")

        except Exception as e:
            print(f"  [{i+1}/{len(selected)}] {c.display_name or '?':<15} | ERROR: {e}")
            results.append(
                {
                    "contact_name": c.display_name or "(unknown)",
                    "relationship": profile.relationship or "unknown",
                    "formality": profile.formality or "casual",
                    "formality_score": profile.formality_score,
                    "pair_count": item["pair_count"],
                    "msg_count": profile.message_count,
                    "trigger": trigger,
                    "original": original_response,
                    "generated": f"ERROR: {e}",
                    "confidence": "error",
                    "similarity": 0.0,
                    "type": "error",
                    "gen_time_ms": 0,
                }
            )

    # Step 4: Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    if not results:
        print("No results generated.")
        return

    # Summary stats
    gen_times = [r["gen_time_ms"] for r in results if r["type"] != "error"]
    similarities = [r["similarity"] for r in results if r["type"] != "error"]

    print(f"\nGenerated {len(results)} responses")
    if gen_times:
        print(f"  Avg generation time: {sum(gen_times)/len(gen_times):.0f}ms")
        print(f"  Min/Max: {min(gen_times):.0f}ms / {max(gen_times):.0f}ms")
    if similarities:
        print(f"  Avg similarity: {sum(similarities)/len(similarities):.3f}")

    # Relationship distribution
    rels = {}
    for r in results:
        rels[r["relationship"]] = rels.get(r["relationship"], 0) + 1
    print(f"\n  Relationships: {rels}")

    # Formality distribution
    forms = {}
    for r in results:
        forms[r["formality"]] = forms.get(r["formality"], 0) + 1
    print(f"  Formality: {forms}")

    # Detailed results table
    print("\n" + "-" * 80)
    print(f"{'Contact':<15} {'Rel':<12} {'Form':<8} {'Pairs':>5} {'Sim':>5} {'Time':>6}")
    print("-" * 80)

    for r in results:
        name = r["contact_name"][:14]
        rel = r["relationship"][:11]
        form = r["formality"][:7]
        print(
            f"{name:<15} {rel:<12} {form:<8} {r['pair_count']:>5} "
            f"{r['similarity']:>5.2f} {r['gen_time_ms']:>5.0f}ms"
        )

    # Detailed exchanges
    print("\n" + "=" * 80)
    print("DETAILED EXCHANGES")
    print("=" * 80)

    for i, r in enumerate(results):
        print(f"\n--- [{i+1}] {r['contact_name']} ({r['relationship']}, {r['formality']}) ---")
        print(f"  Trigger:   {r['trigger'][:100]}")
        print(f"  Original:  {r['original'][:100]}")
        print(f"  Generated: {r['generated'][:100]}")
        print(
            f"  Sim: {r['similarity']:.3f} | Conf: {r['confidence']} | Time: {r['gen_time_ms']:.0f}ms"
        )

    # Save full results to JSON
    output_path = Path.home() / ".jarvis" / "batch_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to {output_path}")

    # Cleanup
    router.close()


if __name__ == "__main__":
    main()
