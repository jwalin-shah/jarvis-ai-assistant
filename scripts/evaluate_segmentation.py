import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.imessage.reader import ChatDBReader
from jarvis.topics.topic_segmenter import segment_conversation


def evaluate_segmentation():
    reader = ChatDBReader()

    # 1. Pick a chat with a decent number of messages
    convs = reader.get_conversations(limit=50)
    target_chats = [c for c in convs if c.message_count > 40 and c.display_name]

    if not target_chats:
        print("No suitable chats found for evaluation.")
        return

    for chat in target_chats[:3]:
        print(
            f"\n{'=' * 80}\nSEGMENTATION CHECK: {chat.display_name} "
            f"({chat.message_count} msgs)\n{'=' * 80}"
        )

        messages = reader.get_messages(chat.chat_id, limit=100)
        # segment_conversation expects sorted oldest-first
        messages.reverse()

        segments = segment_conversation(messages, contact_id=chat.chat_id)

        print(f"Total Segments Found: {len(segments)}")

        for i, seg in enumerate(segments):
            print(f"\n--- SEGMENT {i} ({len(seg.messages)} msgs) | Topic: {seg.topic_label} ---")
            print(f"Summary: {seg.summary}")
            print("Sample Messages:")
            # Show first 2 and last 2 messages
            sample = seg.messages[:2] + (seg.messages[-2:] if len(seg.messages) > 4 else [])
            for m in sample:
                sender = "Me" if m.is_from_me else chat.display_name
                print(f"  [{m.date.strftime('%H:%M')}] {sender}: {m.text[:100]}...")
            if len(seg.messages) > 4:
                print(f"  ... (+{len(seg.messages) - 4} more)")


if __name__ == "__main__":
    evaluate_segmentation()
