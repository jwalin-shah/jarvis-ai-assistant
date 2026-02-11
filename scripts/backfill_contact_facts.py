"""Backfill contact_facts table from historical iMessage conversations.

Extracts facts (relationships, locations, work, preferences) from existing
messages and populates the knowledge graph database.

Two modes:
  --use-segments: Topic segmenter → GLiNER + spaCy NER (higher recall)
  default:        Rule-based FactExtractor (fast, no model load)

Usage:
    uv run python scripts/backfill_contact_facts.py [--max-contacts 50] [--messages-per-contact 500]
    uv run python scripts/backfill_contact_facts.py --use-segments
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("backfill_facts.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def backfill(
    max_contacts: int = 50,
    messages_per_contact: int = 500,
    output_file: str | None = None,
    use_segments: bool = False,
) -> None:
    """Extract facts from historical messages for top contacts."""
    from integrations.imessage import ChatDBReader
    from jarvis.contacts.fact_storage import get_fact_count, save_facts
    from jarvis.db import get_db

    # Ensure schema is up to date
    db = get_db()
    db.init_schema()

    mode = "segments (GLiNER + spaCy)" if use_segments else "rule-based"
    print(
        f"Starting backfill: max_contacts={max_contacts}, "
        f"messages_per_contact={messages_per_contact}, mode={mode}",
        flush=True,
    )

    # Get conversations sorted by message count
    reader = ChatDBReader()
    conversations = reader.get_conversations(limit=max_contacts * 2)
    print(f"Found {len(conversations)} conversations", flush=True)

    # Initialize extractors based on mode
    if use_segments:
        from jarvis.contacts.candidate_extractor import CandidateExtractor
        from jarvis.contacts.segment_extractor import extract_facts_from_segments
        from jarvis.topics.topic_segmenter import TopicSegmenter

        segmenter = TopicSegmenter(normalization_task="extraction")
        candidate_extractor = CandidateExtractor()
    else:
        from jarvis.contacts.fact_extractor import FactExtractor

        fact_extractor = FactExtractor()

    total_facts = 0
    total_inserted = 0
    start_time = time.time()

    # Sort by participant count (prefer 1:1 chats)
    conversations.sort(key=lambda c: len(c.participants))

    processed = 0
    for conv in conversations:
        if processed >= max_contacts:
            break

        chat_id = conv.chat_id
        if not chat_id:
            continue

        try:
            messages = reader.get_messages(chat_id, limit=messages_per_contact)

            if use_segments:
                # Use all messages for segmentation (context matters)
                msgs_with_text = [m for m in messages if m.text]
                if len(msgs_with_text) < 3:
                    continue

                processed += 1
                elapsed = time.time() - start_time
                remaining = min(max_contacts, len(conversations)) - processed
                eta = elapsed / processed * remaining if processed > 0 else 0

                print(
                    f"[{processed}/{max_contacts}] {chat_id[:30]:30s} "
                    f"({len(msgs_with_text)} msgs) "
                    f"ETA: {eta:.0f}s",
                    flush=True,
                )

                # Segment → extract
                segments = segmenter.segment(msgs_with_text)
                candidates = extract_facts_from_segments(segments, candidate_extractor)
                n_facts = len(candidates)
                total_facts += n_facts

                if candidates:
                    # Convert FactCandidates to storage format
                    from jarvis.contacts.fact_storage import save_candidate_facts

                    inserted = save_candidate_facts(candidates, chat_id)
                    total_inserted += inserted
                    if inserted:
                        print(
                            f"  -> {n_facts} candidates, {inserted} new ({len(segments)} segments)",
                            flush=True,
                        )
            else:
                # Original rule-based path
                incoming = [m for m in messages if not m.is_from_me and m.text]
                if len(incoming) < 3:
                    continue

                processed += 1
                elapsed = time.time() - start_time
                remaining = min(max_contacts, len(conversations)) - processed
                eta = elapsed / processed * remaining if processed > 0 else 0

                print(
                    f"[{processed}/{max_contacts}] {chat_id[:30]:30s} "
                    f"({len(incoming)} msgs) "
                    f"ETA: {eta:.0f}s",
                    flush=True,
                )

                facts = fact_extractor.extract_facts(incoming, chat_id)
                total_facts += len(facts)

                if facts:
                    inserted = save_facts(facts, chat_id)
                    total_inserted += inserted
                    if inserted:
                        print(
                            f"  -> {len(facts)} facts extracted, {inserted} new",
                            flush=True,
                        )

        except Exception as e:
            logger.warning("Error processing %s: %s", chat_id[:20], e)
            continue

    elapsed = time.time() - start_time
    total_in_db = get_fact_count()
    print(f"\nBackfill complete in {elapsed:.1f}s:", flush=True)
    print(f"  Mode:               {mode}", flush=True)
    print(f"  Contacts processed: {processed}", flush=True)
    print(f"  Facts extracted:    {total_facts}", flush=True)
    print(f"  New facts saved:    {total_inserted}", flush=True)
    print(f"  Total facts in DB:  {total_in_db}", flush=True)

    # Export to file if requested
    if output_file:
        _export_facts(output_file)


def _export_facts(output_file: str) -> None:
    """Export all facts to a readable file for review."""
    from jarvis.contacts.fact_storage import get_all_facts

    facts = get_all_facts()
    if not facts:
        print("No facts to export.", flush=True)
        return

    # Group by contact
    by_contact: dict[str, list] = {}
    for f in facts:
        by_contact.setdefault(f.contact_id, []).append(f)

    with open(output_file, "w") as fp:
        fp.write(f"# Contact Facts Export ({len(facts)} total)\n")
        fp.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for contact_id in sorted(by_contact):
            contact_facts = by_contact[contact_id]
            fp.write(f"## Contact: {contact_id}\n")
            for f in contact_facts:
                fp.write(
                    f"  [{f.category}] {f.subject} {f.predicate}"
                    f"{' ' + f.value if f.value else ''}"
                    f" (conf={f.confidence:.2f})\n"
                )
                if f.source_text:
                    src = f.source_text[:120].replace("\n", " ")
                    fp.write(f'    src: "{src}"\n')
            fp.write("\n")

    print(f"Exported {len(facts)} facts to {output_file}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill contact facts from iMessage history")
    parser.add_argument("--max-contacts", type=int, default=50, help="Max contacts to process")
    parser.add_argument(
        "--messages-per-contact", type=int, default=500, help="Messages per contact"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Export extracted facts to this file for review",
    )
    parser.add_argument(
        "--use-segments",
        action="store_true",
        help="Use topic segmenter + GLiNER pipeline (higher recall, slower)",
    )
    args = parser.parse_args()

    backfill(
        max_contacts=args.max_contacts,
        messages_per_contact=args.messages_per_contact,
        output_file=args.output,
        use_segments=args.use_segments,
    )


if __name__ == "__main__":
    main()
