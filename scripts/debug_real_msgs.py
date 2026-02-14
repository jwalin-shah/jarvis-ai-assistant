from integrations.imessage import ChatDBReader
from jarvis.contacts.candidate_extractor import CandidateExtractor
import logging

logging.basicConfig(level=logging.INFO)

chat_id = 'iMessage;-;+17204963920'
with ChatDBReader() as reader:
    messages = reader.get_messages(chat_id, limit=50)

if not messages:
    print("No messages found for Sangati Shah")
    exit()

extractor = CandidateExtractor(label_profile="high_recall")
seg_msgs_data = []
for m in messages:
    if m.text:
        seg_msgs_data.append({
            "text": m.text,
            "message_id": m.id,
            "is_from_me": m.is_from_me,
            "chat_id": chat_id
        })

print(f"Testing extraction on {len(seg_msgs_data)} messages...")
candidates = extractor.extract_batch(seg_msgs_data)
if not candidates:
    print("No candidates extracted from actual messages.")
else:
    for c in candidates:
        print(f"  - Found: {c.span_text} ({c.span_label}) -> {c.fact_type} [score: {c.gliner_score:.2f}]")
        print(f"    Source: {c.source_text}")
