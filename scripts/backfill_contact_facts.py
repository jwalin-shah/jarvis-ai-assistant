"""Backfill contact_facts table from historical iMessage conversations.

Extracts facts (relationships, locations, work, preferences) from existing
messages and populates the knowledge graph database.

Three modes:
  default (--use-segments): GLiNER batch extraction with context windows (fast, MLX GPU)
  --use-segments:           Same as default (kept for backwards compat)
  --rule-based:             Rule-based FactExtractor (no model load, lower recall)

Usage:
    uv run python scripts/backfill_contact_facts.py --max-contacts 5 --messages-per-contact 200
    uv run python scripts/backfill_contact_facts.py --max-contacts 50 -o results/facts.txt
    uv run python scripts/backfill_contact_facts.py --rule-based
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
    rule_based: bool = False,
    batch_size: int = 32,
) -> None:
    """Extract facts from historical messages for top contacts."""
    from integrations.imessage import ChatDBReader
    from jarvis.contacts.fact_storage import get_fact_count, save_candidate_facts, save_facts
    from jarvis.db import get_db

    # Ensure schema is up to date
    db = get_db()
    db.init_schema()

    mode = "rule-based" if rule_based else f"GLiNER batch (batch_size={batch_size})"
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
    if not rule_based:
        from jarvis.contacts.candidate_extractor import CandidateExtractor

        extractor = CandidateExtractor()
        # Eagerly load the model so first batch isn't slow
        print("Loading GLiNER model...", flush=True)
        if extractor._use_mlx():
            extractor._load_mlx_model()
            print("  MLX backend (Metal GPU)", flush=True)
        else:
            extractor._load_model()
            print("  PyTorch backend (CPU)", flush=True)
    else:
        from jarvis.contacts.fact_extractor import FactExtractor

        fact_extractor = FactExtractor()

    total_facts = 0
    total_inserted = 0
    total_messages = 0
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

            if rule_based:
                incoming = [m for m in messages if not m.is_from_me and m.text]
                if len(incoming) < 3:
                    continue

                processed += 1
                _print_progress(processed, max_contacts, chat_id, len(incoming), start_time)

                facts = fact_extractor.extract_facts(incoming, chat_id)
                total_facts += len(facts)

                if facts:
                    inserted = save_facts(facts, chat_id)
                    total_inserted += inserted
                    if inserted:
                        print(f"  -> {len(facts)} facts, {inserted} new", flush=True)
            else:
                # GLiNER batch extraction
                msgs_with_text = [m for m in messages if m.text]
                if len(msgs_with_text) < 3:
                    continue

                processed += 1
                total_messages += len(msgs_with_text)
                _print_progress(processed, max_contacts, chat_id, len(msgs_with_text), start_time)

                # Sort by date for context windowing
                msgs_with_text.sort(key=lambda m: m.date)

                # Build batch dicts with context windows
                batch_dicts = _build_batch_dicts(msgs_with_text, chat_id)

                # Run batched extraction
                t0 = time.time()
                candidates = extractor.extract_batch(batch_dicts, batch_size=batch_size)
                dt = time.time() - t0

                n_facts = len(candidates)
                total_facts += n_facts
                rate = len(batch_dicts) / dt if dt > 0 else 0

                if candidates:
                    inserted = save_candidate_facts(candidates, chat_id)
                    total_inserted += inserted
                    print(
                        f"  -> {n_facts} candidates, {inserted} new "
                        f"({len(batch_dicts)} msgs in {dt:.1f}s, {rate:.0f} msgs/sec)",
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
    print(f"  Messages processed: {total_messages}", flush=True)
    print(f"  Facts extracted:    {total_facts}", flush=True)
    print(f"  New facts saved:    {total_inserted}", flush=True)
    print(f"  Total facts in DB:  {total_in_db}", flush=True)
    if total_messages > 0 and elapsed > 0:
        print(f"  Throughput:         {total_messages / elapsed:.0f} msgs/sec", flush=True)

    # Export to file if requested
    if output_file:
        _export_facts(output_file)


def _build_batch_dicts(msgs_with_text: list, chat_id: str) -> list[dict]:
    """Convert iMessage objects to batch dicts with 2-message context windows."""
    batch: list[dict] = []
    texts = [m.text for m in msgs_with_text]

    for i, msg in enumerate(msgs_with_text):
        prev = texts[max(0, i - 2) : i]
        nxt = texts[i + 1 : min(len(texts), i + 2)]

        batch.append({
            "text": msg.text,
            "message_id": msg.id,
            "chat_id": chat_id,
            "is_from_me": msg.is_from_me,
            "message_date": msg.date,
            "context_prev": prev if prev else None,
            "context_next": nxt if nxt else None,
        })

    return batch


def _print_progress(
    processed: int, max_contacts: int, chat_id: str, n_msgs: int, start_time: float
) -> None:
    """Print progress line with ETA."""
    elapsed = time.time() - start_time
    remaining = max_contacts - processed
    eta = elapsed / processed * remaining if processed > 0 else 0
    print(
        f"[{processed}/{max_contacts}] {chat_id[:35]:35s} "
        f"({n_msgs} msgs) elapsed={elapsed:.0f}s ETA={eta:.0f}s",
        flush=True,
    )


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
        "--rule-based",
        action="store_true",
        help="Use rule-based extractor (no model load, lower recall)",
    )
    parser.add_argument(
        "--use-segments",
        action="store_true",
        help="(Deprecated) Same as default GLiNER batch mode",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="GLiNER batch size (default: 32)",
    )
    args = parser.parse_args()

    backfill(
        max_contacts=args.max_contacts,
        messages_per_contact=args.messages_per_contact,
        output_file=args.output,
        rule_based=args.rule_based,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
