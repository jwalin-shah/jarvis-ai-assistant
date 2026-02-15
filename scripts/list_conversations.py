#!/usr/bin/env python3
"""List all iMessage conversations to see what's available."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.imessage import ChatDBReader

reader = ChatDBReader()
conversations = reader.get_conversations(limit=100)

print(f"\nFound {len(conversations)} conversations:\n")
print(f"{'Type':<15} {'Display Name':<30} {'Message Count':<15} {'Chat ID (first 50)'}")
print("-" * 110)

for c in conversations[:30]:  # Show first 30
    chat_type = "SMS" if "SMS" in c.chat_id else "iMessage" if "iMessage" in c.chat_id else "RCS" if "RCS" in c.chat_id else "Other"
    name = (c.display_name or "Unknown")[:28]
    msg_count = c.message_count
    chat_id = c.chat_id[:50] if c.chat_id else ""
    print(f"{chat_type:<15} {name:<30} {msg_count:<15} {chat_id}")

print(f"\n{'='*60}")
print("Top 10 by message count:")
conversations.sort(key=lambda x: x.message_count, reverse=True)
for c in conversations[:10]:
    name = c.display_name or "Unknown"
    print(f"  {name}: {c.message_count} messages ({c.chat_id[:40]}...)")
