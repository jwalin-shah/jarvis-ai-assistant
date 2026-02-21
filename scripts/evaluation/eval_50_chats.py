import asyncio
import logging
import sys
import time
from datetime import UTC, datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("eval_50")

# Add project root to path
sys.path.insert(0, ".")

from integrations.imessage import ChatDBReader
from jarvis.contracts.pipeline import MessageContext
from jarvis.db import get_db
from jarvis.reply_service import get_reply_service


async def run_eval():
    print("=" * 80)
    print("🚀 STARTING EVALUATION ON TOP 50 CHATS")
    print("Goal: Test variety (3 suggestions), language consistency, and latency.")
    print("=" * 80)

    service = get_reply_service()
    get_db()
    reader = ChatDBReader()

    # 1. Fetch real conversations
    print("Fetching top 50 recent conversations...")
    conversations = reader.get_conversations(limit=50)

    if not conversations:
        print("No conversations found. Ensure Full Disk Access is granted.")
        return

    stats = {
        "total_chats": 0,
        "total_suggestions": 0,
        "chinese_detected": 0,
        "stale_caught": 0,
        "empty_responses": 0,
        "total_latency_ms": 0,
    }

    for i, conv in enumerate(conversations, 1):
        chat_id = conv.chat_id
        display_name = conv.display_name or chat_id[:12]

        # Get messages
        messages = reader.get_messages(chat_id, limit=10)
        if not messages:
            continue

        last_msg = messages[0]
        incoming = last_msg.text or ""
        if not incoming:
            continue

        print(f"\n[{i}/50] CHAT: {display_name}")
        print(f'   Incoming: "{incoming[:60]}..."')

        # Simulate the new _generate_draft multi-suggestion logic
        # We'll use the service directly to see what it produces

        start_time = time.perf_counter()

        suggestions = []
        seen_texts = set()

        # We want 3 suggestions per chat
        num_requested = 3
        attempts = 0

        while len(suggestions) < num_requested and attempts < num_requested + 2:
            attempts += 1

            # Build context for this attempt
            context = MessageContext(
                chat_id=chat_id,
                message_text=incoming,
                is_from_me=last_msg.is_from_me,
                timestamp=datetime.now(UTC),
                metadata={"thread": [m.text for m in reversed(messages) if m.text]},
            )

            # Generate
            try:
                result = service.generate_reply(context)
                text = result.response.strip()
            except Exception as e:
                print(f"   ❌ Generation failed: {e}")
                continue

            if not text:
                stats["empty_responses"] += 1
                continue

            # Check for Chinese
            import re

            if re.search(r"[\u4e00-\u9fff]", text):
                stats["chinese_detected"] += 1
                print(f"   ⚠️ CHINESE DRIFT: {text}")

            if text and text not in seen_texts:
                suggestions.append(text)
                seen_texts.add(text)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        stats["total_latency_ms"] += elapsed_ms
        stats["total_chats"] += 1
        stats["total_suggestions"] += len(suggestions)

        if suggestions:
            for j, s in enumerate(suggestions, 1):
                print(f"   Suggestion {j}: {s}")
        else:
            print("   ❌ No suggestions generated")

        print(f"   Latency: {elapsed_ms:.1f}ms")

    # Final Stats
    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY")
    print("-" * 80)
    print(f"Total Chats Processed:    {stats['total_chats']}")
    print(f"Total Suggestions Made:   {stats['total_suggestions']}")
    print(
        f"Avg Suggestions/Chat:     {stats['total_suggestions'] / max(1, stats['total_chats']):.1f}"
    )
    print(
        f"Avg Latency/Chat:         {stats['total_latency_ms'] / max(1, stats['total_chats']):.1f}ms"
    )
    print(f"Chinese Drift Events:     {stats['chinese_detected']}")
    print(f"Empty Response Events:    {stats['empty_responses']}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_eval())
