import logging

from integrations.imessage.reader import ChatDBReader
from jarvis.contacts.fact_storage import save_facts
from jarvis.contacts.instruction_extractor import get_instruction_extractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis")

TARGET_CHATS = [
    {"chat_id": "iMessage;-;+14087867207", "default_name": "Mateo"},
    {"chat_id": "iMessage;-;+14084643141", "default_name": "Friend"},
]


class MockSegment:
    def __init__(self, messages):
        self.messages = messages
        # Safer join to avoid syntax issues in prompt rendering
        self.text = "\n".join([" ".join((m.text or "").splitlines()) for m in messages])


def process_top_contacts():
    reader = ChatDBReader()
    user_name = "Jwalin"
    extractor = get_instruction_extractor(tier="1.2b")

    for target in TARGET_CHATS:
        chat_id = target["chat_id"]

        # 1. Resolve Contact Name
        contact_name = target["default_name"]
        conv = reader.get_conversation(chat_id)
        if conv and conv.display_name:
            contact_name = conv.display_name.split()[0]

        print("\n" + "=" * 60)
        print(f"PROCESSING: {contact_name} (Chat ID: {chat_id})")
        print("=" * 60)

        # 2. Fetch messages
        messages = reader.get_messages(chat_id, limit=300)
        messages.reverse()
        if not messages:
            print(f"No messages found for {chat_id}")
            continue

        # 3. Create Sliding Windows
        window_size = 25
        overlap = 5
        windows = []
        for i in range(0, len(messages), window_size - overlap):
            window = messages[i : i + window_size]
            if len(window) < 5:
                break
            windows.append(MockSegment(window))

        print(f"Created {len(windows)} windows from {len(messages)} messages.")

        # 4. Extract
        all_facts = []
        for i, seg in enumerate(windows[:5]):
            print(f"\nProcessing window {i + 1}/5...")
            facts = extractor.extract_facts_from_segment(
                seg, contact_id=chat_id, contact_name=contact_name, user_name=user_name
            )
            if facts:
                print(f"  Verified {len(facts)} facts.")
                for f in facts:
                    print(f"    - {f.subject}: {f.value}")
                all_facts.extend(facts)
                save_facts(
                    facts,
                    chat_id,
                    log_raw_facts=True,
                    log_chat_id=chat_id,
                    log_stage="process_top_contacts",
                )

        print(f"\nCompleted {contact_name}. Total verified facts: {len(all_facts)}")


if __name__ == "__main__":
    process_top_contacts()
