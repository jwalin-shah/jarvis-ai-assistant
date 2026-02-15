import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.imessage import ChatDBReader
from jarvis.contacts.instruction_extractor import get_instruction_extractor


def test_subset():
    reader = ChatDBReader()
    extractor = get_instruction_extractor(tier="0.7b")

    # 1. Pick two distinct chats (one individual, one group)
    # Robert (Individual) and The Ochos (Group)
    targets = ["RCS;-;+17252177891", "iMessage;+;chat497445309743717804"]

    for chat_id in targets:
        print(f"\n{'='*80}\nTESTING CHAT: {chat_id}\n{'='*80}")
        messages = reader.get_messages(chat_id, limit=30)
        messages.reverse()

        # Turn grouping
        from dataclasses import dataclass
        @dataclass
        class Seg:
            def __init__(self, msgs, text):
                self.messages = msgs
                self.text = text

        seg = Seg(messages, "\n".join([m.text or "" for m in messages]))

        facts = extractor.extract_facts_from_segment(
            seg,
            contact_id=chat_id,
            contact_name="Contact",
            user_name="Jwalin"
        )

        print(f"\nFINAL VERIFIED FACTS FOR {chat_id}:")
        if not facts:
            print("  (None found - Correct behavior for noisy/logistical chats)")
        for f in facts:
            print(f"  ✓ [{f.subject}] {f.value} (Conf: {f.confidence:.2f})")

if __name__ == "__main__":
    test_subset()
