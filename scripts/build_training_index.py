#!/usr/bin/env python3
"""Build FAISS index from extracted conversation files.

Merges threaded_conversations.jsonl and semantic_conversations.jsonl,
deduplicates, imports to JarvisDB, and builds FAISS index.

Usage:
    uv run python -m scripts.build_training_index
    uv run python -m scripts.build_training_index --test-queries
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_pair_hash(trigger: str, response: str) -> str:
    """Compute hash for deduplication."""
    content = f"{trigger.strip().lower()}|{response.strip().lower()}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def parse_date(date_str: str | None) -> datetime:
    """Parse ISO date string to datetime."""
    if not date_str:
        return datetime.now()
    try:
        # Handle various ISO formats
        if "T" in date_str:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return datetime.now()


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file, handling errors gracefully."""
    records = []
    errors = 0
    with open(path) as f:
        for i, line in enumerate(f):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                errors += 1
                if errors <= 5:
                    logger.warning(f"JSON error on line {i + 1}: {line[:50]}...")
    if errors:
        logger.warning(f"Total JSON errors: {errors}")
    return records


def merge_and_dedupe(
    files: list[Path],
    min_trigger_len: int = 3,
    min_response_len: int = 2,
) -> list[dict]:
    """Merge JSONL files and deduplicate by trigger+response hash."""
    seen_hashes: set[str] = set()
    pairs: list[dict] = []

    stats = {"total": 0, "duplicates": 0, "short_trigger": 0, "short_response": 0, "empty": 0}

    for path in files:
        if not path.exists():
            logger.warning(f"File not found: {path}")
            continue

        logger.info(f"Loading {path.name}...")
        records = load_jsonl(path)
        logger.info(f"  Loaded {len(records)} records")

        for rec in records:
            stats["total"] += 1

            # Extract fields (both formats have these)
            trigger = rec.get("immediate_trigger", "").strip()
            response = rec.get("my_response", "").strip()
            context = rec.get("context_formatted", "")
            response_date = rec.get("my_response_date")
            is_group = rec.get("is_group", False)
            chat_id = rec.get("chat_identifier") or rec.get("thread_guid") or "unknown"

            # Quality filters
            if not trigger or not response:
                stats["empty"] += 1
                continue
            if len(trigger) < min_trigger_len:
                stats["short_trigger"] += 1
                continue
            if len(response) < min_response_len:
                stats["short_response"] += 1
                continue

            # Skip special characters only
            if trigger in ["\ufffc", "\u200b"] or response in ["\ufffc", "\u200b"]:
                stats["empty"] += 1
                continue

            # Deduplicate
            pair_hash = compute_pair_hash(trigger, response)
            if pair_hash in seen_hashes:
                stats["duplicates"] += 1
                continue
            seen_hashes.add(pair_hash)

            # Parse timestamp
            response_dt = parse_date(response_date)
            trigger_dt = response_dt  # Use same timestamp (we don't have separate trigger time)

            pairs.append(
                {
                    "trigger_text": trigger,
                    "response_text": response,
                    "context_text": context if context else None,
                    "trigger_timestamp": trigger_dt,
                    "response_timestamp": response_dt,
                    "chat_id": chat_id,
                    "is_group": is_group,
                    "quality_score": 1.0,  # Will be computed later
                }
            )

    logger.info(f"Merge stats: {json.dumps(stats, indent=2)}")
    logger.info(f"Unique pairs: {len(pairs)}")

    return pairs


def import_to_db(pairs: list[dict], batch_size: int = 1000) -> int:
    """Import pairs to JarvisDB."""
    from jarvis.db import get_db

    db = get_db()

    # Clear existing pairs if needed
    existing = db.get_all_pairs()
    if existing:
        logger.info(f"Database has {len(existing)} existing pairs")

    total_added = 0
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        added = db.add_pairs_bulk(batch)
        total_added += added
        if (i + batch_size) % 10000 == 0 or i + batch_size >= len(pairs):
            logger.info(f"  Imported {min(i + batch_size, len(pairs))}/{len(pairs)} pairs...")

    return total_added


def build_index(min_quality: float = 0.5, holdout_ratio: float = 0.2) -> dict:
    """Build FAISS index from database pairs."""
    import random

    from jarvis.db import get_db
    from jarvis.index import build_index_from_db

    db = get_db()

    # Create train/test split - random by pairs for proper ratio
    logger.info(f"Creating {holdout_ratio:.0%} holdout split (random by pairs)...")

    all_pairs = db.get_all_pairs()
    if not all_pairs:
        return {"success": False, "error": "No pairs in database"}

    # Random split by pairs
    random.seed(42)
    pair_ids = [p.id for p in all_pairs]
    random.shuffle(pair_ids)

    holdout_count = int(len(pair_ids) * holdout_ratio)
    holdout_ids = set(pair_ids[:holdout_count])
    training_ids = set(pair_ids[holdout_count:])

    # Update holdout flags in DB (batch to avoid SQL variable limit)
    with db.connection() as conn:
        conn.execute("UPDATE pairs SET is_holdout = 0")  # Reset all

        # Batch update holdout in chunks of 500
        holdout_list = list(holdout_ids)
        batch_size = 500
        for i in range(0, len(holdout_list), batch_size):
            batch = holdout_list[i : i + batch_size]
            placeholders = ",".join("?" * len(batch))
            conn.execute(f"UPDATE pairs SET is_holdout = 1 WHERE id IN ({placeholders})", batch)
        conn.commit()

    logger.info(
        f"Split: {len(training_ids)} training, {len(holdout_ids)} holdout "
        f"({holdout_count / len(pair_ids):.1%})"
    )

    # Build index (excludes holdout by default)
    logger.info("Building FAISS index...")

    def progress(stage: str, pct: float, msg: str):
        logger.info(f"  [{stage}] {pct:.0%} - {msg}")

    result = build_index_from_db(
        db,
        progress_callback=progress,
        min_quality=min_quality,
        include_holdout=False,
    )

    return result


def test_retrieval(queries: list[str] | None = None, k: int = 3) -> None:
    """Test retrieval with sample queries."""
    from jarvis.db import get_db
    from jarvis.index import TriggerIndexSearcher

    db = get_db()
    searcher = TriggerIndexSearcher(db)

    if queries is None:
        queries = [
            "hey what's up",
            "want to grab lunch?",
            "that's hilarious lmao",
            "sounds good to me",
            "are you coming tonight?",
            "thanks for letting me know",
            "I'll be there in 10",
            "did you see the game last night?",
        ]

    logger.info("\n" + "=" * 60)
    logger.info("Testing retrieval:")
    logger.info("=" * 60)

    for query in queries:
        results = searcher.search_with_pairs(query, k=k, threshold=0.3)

        print(f'\nQuery: "{query}"')
        if not results:
            print("  No matches above threshold")
        else:
            for i, r in enumerate(results):
                print(f'  {i + 1}. [{r["similarity"]:.3f}] "{r["trigger_text"][:50]}..."')
                print(f'     -> "{r["response_text"][:50]}..."')


def main():
    parser = argparse.ArgumentParser(description="Build training index from conversation files")
    parser.add_argument(
        "--files",
        nargs="+",
        default=["semantic_conversations.jsonl", "threaded_conversations.jsonl"],
        help="JSONL files to merge",
    )
    parser.add_argument("--min-quality", type=float, default=0.5, help="Min quality score")
    parser.add_argument("--holdout-ratio", type=float, default=0.2, help="Holdout ratio for eval")
    parser.add_argument("--test-queries", action="store_true", help="Test with sample queries")
    parser.add_argument("--skip-import", action="store_true", help="Skip import, only build index")
    parser.add_argument(
        "--clear-db", action="store_true", help="Clear existing pairs before import"
    )
    args = parser.parse_args()

    # Resolve file paths
    base_dir = Path(__file__).parent.parent
    files = [base_dir / f for f in args.files]

    if not args.skip_import:
        # Step 1: Merge and dedupe
        logger.info("=" * 60)
        logger.info("Step 1: Merging and deduplicating files...")
        logger.info("=" * 60)
        pairs = merge_and_dedupe(files)

        if not pairs:
            logger.error("No pairs extracted!")
            sys.exit(1)

        # Step 2: Clear DB if requested
        if args.clear_db:
            from jarvis.db import get_db

            db = get_db()
            logger.info("Clearing existing pairs...")
            with db.connection() as conn:
                # Disable FK checks, clear tables, re-enable
                conn.execute("PRAGMA foreign_keys = OFF")
                query = "SELECT name FROM sqlite_master WHERE type='table'"
                tables = [r[0] for r in conn.execute(query)]
                for table in ["pair_artifacts", "embeddings", "pairs"]:
                    if table in tables:
                        conn.execute(f"DELETE FROM {table}")
                conn.execute("PRAGMA foreign_keys = ON")
                conn.commit()
            logger.info("Cleared.")

        # Step 3: Import to DB
        logger.info("\n" + "=" * 60)
        logger.info("Step 2: Importing to database...")
        logger.info("=" * 60)
        added = import_to_db(pairs)
        logger.info(f"Added {added} pairs to database")

    # Step 4: Build index
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Building FAISS index...")
    logger.info("=" * 60)
    result = build_index(min_quality=args.min_quality, holdout_ratio=args.holdout_ratio)

    if result.get("success"):
        logger.info("\nIndex built successfully!")
        logger.info(f"  Pairs indexed: {result['pairs_indexed']}")
        logger.info(f"  Dimension: {result['dimension']}")
        logger.info(f"  Size: {result['index_size_bytes'] / 1024 / 1024:.1f} MB")
        logger.info(f"  Version: {result['version_id']}")
        logger.info(f"  Path: {result['index_path']}")
    else:
        logger.error(f"Index build failed: {result.get('error')}")
        sys.exit(1)

    # Step 5: Test queries
    if args.test_queries:
        test_retrieval()

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
