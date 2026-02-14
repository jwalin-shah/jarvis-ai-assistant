#!/usr/bin/env python3
"""Debug extraction to see raw messages and facts side by side."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.imessage import ChatDBReader
from dataclasses import dataclass

@dataclass
class SimpleSegment:
    messages: list
    text: str

CHAT_ID = "iMessage;+;chat449740849439005391"

reader = ChatDBReader()
messages = reader.get_messages(CHAT_ID, limit=100)
messages.reverse()

print(f"Total messages: {len(messages)}\n")
print("="*60)
print("RAW MESSAGES:")
print("="*60)

for i, m in enumerate(messages[:30]):  # First 30 messages
    sender = "Jwalin" if m.is_from_me else "Unknown"
    text = (m.text or "")[:100]
    print(f"{i}: {sender}: {text}")

# Create segments like the extractor does
WINDOW_SIZE = 25
OVERLAP = 5
segments = []
for j in range(0, len(messages), WINDOW_SIZE - OVERLAP):
    window = messages[j:j + WINDOW_SIZE]
    if len(window) < 5:
        break
    seg_text = "\n".join([f"{'Jwalin' if m.is_from_me else 'Unknown'}: {(m.text or '')}" for m in window])
    segments.append(SimpleSegment(messages=window, text=seg_text))

print(f"\n{'='*60}")
print(f"SEGMENTS CREATED: {len(segments)}")
print(f"{'='*60}")

for i, seg in enumerate(segments):
    print(f"\n--- SEGMENT {i} ({len(seg.messages)} messages) ---")
    print(seg.text[:1500])  # First 1500 chars
    print("...")
