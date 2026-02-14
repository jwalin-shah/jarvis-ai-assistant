from integrations.imessage import ChatDBReader
from jarvis.contacts.instruction_extractor import get_instruction_extractor
import logging
import os
import json

# Ensure we stay offline for local model
os.environ["HF_HUB_OFFLINE"] = "1"

# Enable debug logging for our extractor
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("jarvis.contacts.instruction_extractor")

chat_id = 'iMessage;-;+17204963920' # Sangati Shah
with ChatDBReader() as reader:
    messages = reader.get_messages(chat_id, limit=10)

lines = []
for m in messages:
    if m.text:
        lines.append(m.text)

segment_text = "\n".join(lines)
print(f"--- DEBUG Extraction on Segment ---\n{segment_text}\n")

extractor = get_instruction_extractor(tier="350m")
facts = extractor.extract_facts_from_segment(segment_text, contact_id=chat_id)

print(f"\nExtracted {len(facts)} facts.")
for f in facts:
    print(f"  {f.category} | {f.subject} | {f.predicate}")

extractor.unload()
