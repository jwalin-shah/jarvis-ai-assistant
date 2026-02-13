"""Extract facts from chat history into contact_facts table.

Reads conversations from chat.db via ChatDBReader, runs FactExtractor,
and stores results in the contact_facts table.

Usage:
    uv run python scripts/extract_facts.py [--contact CHAT_ID] [--threshold 0.7] [--use-ner]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    # Setup logging with FileHandler + StreamHandler at top of main()
    log_file = Path("extract_facts.log")
    file_handler = logging.FileHandler(log_file, mode="a")
    stream_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[file_handler, stream_handler],
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting extract_facts.py")

    parser = argparse.ArgumentParser(description="Extract facts from chat history")
    parser.add_argument("--contact", type=str, help="Process only this chat_id")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold")
    parser.add_argument("--use-ner", action="store_true", help="Use spaCy NER (slower, richer)")
    parser.add_argument("--limit", type=int, default=500, help="Messages per contact")
    args = parser.parse_args()

    from integrations.imessage import ChatDBReader
    from jarvis.contacts.contact_profile import save_facts
    from jarvis.contacts.fact_extractor import FactExtractor

    reader = ChatDBReader()
    extractor = FactExtractor(confidence_threshold=args.threshold)

    if args.contact:
        chat_ids = [args.contact]
    else:
        conversations = reader.get_conversations(limit=200)
        chat_ids = [c.chat_id for c in conversations if not c.is_group]

    print(f"Processing {len(chat_ids)} contacts...", flush=True)

    total_facts = 0
    for i, chat_id in enumerate(tqdm(chat_ids, desc="Extracting facts", total=len(chat_ids)), 1):
        try:
            messages = reader.get_messages(chat_id=chat_id, limit=args.limit)
            # Only analyze their messages (facts about them)
            their_messages = [m for m in messages if not m.is_from_me]

            if not their_messages:
                continue

            if args.use_ner:
                facts = extractor.extract_facts_with_ner(their_messages, contact_id=chat_id)
            else:
                facts = extractor.extract_facts(their_messages, contact_id=chat_id)

            # Filter by threshold
            facts = [f for f in facts if f.confidence >= args.threshold]

            if facts:
                count = save_facts(chat_id, facts)
                total_facts += count
                print(
                    f"  [{i}/{len(chat_ids)}] {chat_id[:20]}...: "
                    f"{count} facts from {len(their_messages)} messages",
                    flush=True,
                )
            else:
                print(f"  [{i}/{len(chat_ids)}] {chat_id[:20]}...: no facts", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(chat_ids)}] {chat_id[:20]}...: ERROR {e}", flush=True)

    print(f"\nDone. Extracted {total_facts} total facts.", flush=True)
    logger.info(f"Extracted {total_facts} total facts")
    logger.info("Finished extract_facts.py")


if __name__ == "__main__":
    main()
