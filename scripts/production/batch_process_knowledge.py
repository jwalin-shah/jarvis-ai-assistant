import argparse
import logging

from jarvis.db import get_db
from jarvis.tasks.models import TaskType
from jarvis.tasks.queue import get_task_queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis")


def batch_process_knowledge(force_historical: bool = False, limit: int | None = None):
    """Queue fact extraction for all contacts.

    Args:
        force_historical: If True, re-extract all messages from scratch.
                         If False, only extract new messages since last extraction.
        limit: Maximum number of contacts to process.
    """
    db = get_db()
    queue = get_task_queue()

    with db.connection() as conn:
        if force_historical:
            query = """
                SELECT chat_id, display_name, last_extracted_rowid
                FROM contacts
                ORDER BY last_extracted_rowid NULLS FIRST
            """
        else:
            # Only queue contacts that have new messages
            # (have last_extracted_rowid set but may have new messages)
            # or contacts never extracted
            query = """
                SELECT chat_id, display_name, last_extracted_rowid
                FROM contacts
                ORDER BY last_extracted_rowid NULLS FIRST
            """

        if limit:
            query += f" LIMIT {limit}"

        cursor = conn.execute(query)
        contacts = cursor.fetchall()

    if not contacts:
        print("No contacts found.")
        return

    mode = "historical" if force_historical else "incremental"
    print(f"Found {len(contacts)} contacts. Queuing {mode} extraction tasks...")

    queued = 0
    for contact in contacts:
        chat_id = contact["chat_id"]
        display = contact["display_name"] or chat_id
        last_row = contact["last_extracted_rowid"]

        if force_historical:
            reason = "force historical"
        elif last_row is None:
            reason = "never extracted"
        else:
            reason = f"last extracted row {last_row}"

        print(f"Queuing {display}: {reason}")
        queue.enqueue(
            TaskType.FACT_EXTRACTION, {"chat_id": chat_id, "force_historical": force_historical}
        )
        queued += 1

    print(f"Queued {queued} extraction tasks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process knowledge extraction")
    parser.add_argument(
        "--force", action="store_true", help="Force re-extraction of all messages (historical pass)"
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum contacts to process")
    args = parser.parse_args()

    batch_process_knowledge(force_historical=args.force, limit=args.limit)
