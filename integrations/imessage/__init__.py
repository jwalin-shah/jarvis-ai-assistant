"""iMessage chat.db integration (Workstream 10).

Provides read-only access to the macOS iMessage database and
message sending via AppleScript.

Example:
    from integrations.imessage import ChatDBReader, iMessageSender

    reader = ChatDBReader()
    if reader.check_access():
        conversations = reader.get_conversations(limit=10)
        for conv in conversations:
            messages = reader.get_messages(conv.chat_id)

    # Send a message
    sender = iMessageSender()
    result = sender.send_message("+1234567890", "Hello!")
"""

from .reader import CHAT_DB_PATH, ChatDBReader
from .sender import IMessageSender, SendResult, TapbackType

__all__ = ["ChatDBReader", "CHAT_DB_PATH", "IMessageSender", "SendResult", "TapbackType"]
