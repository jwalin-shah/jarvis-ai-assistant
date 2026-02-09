"""iMessage sender using AppleScript.

Sends iMessages via the Messages app using osascript subprocess calls.

Requirements:
- macOS with Messages.app
- Automation permission (user will be prompted on first use)
- Messages.app must be running

Usage:
    from integrations.imessage import IMessageSender

    sender = IMessageSender()
    result = sender.send_message("Hello!", recipient="+1234567890")
    if result.success:
        print("Sent!")
    else:
        print(f"Failed: {result.error}")
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


def _validate_file_path(file_path: Path) -> bool:
    """Validate file path is within allowed directories.

    SECURITY: Prevents path traversal attacks by ensuring resolved path
    is within allowed directories (home, temp, common system dirs).

    Args:
        file_path: Path to validate (should be resolved)

    Returns:
        True if path is in allowed directory, False otherwise
    """
    try:
        # Resolve to absolute path to handle symlinks and ../ traversal
        resolved = file_path.resolve()

        # Define allowed base directories
        allowed_bases = [
            Path.home(),  # User's home directory
            Path("/tmp"),  # Temp directory
            Path("/var/tmp"),  # Alternate temp
            Path("/private/tmp"),  # macOS temp
            Path("/Users"),  # macOS user directories
        ]

        # Check if resolved path is within any allowed base
        for base in allowed_bases:
            try:
                # is_relative_to() checks if path is under base
                if resolved.is_relative_to(base):
                    return True
            except (ValueError, AttributeError):
                # is_relative_to() not available in Python <3.9, or path issues
                # Fallback: check if base is in parents
                if base in resolved.parents or resolved == base:
                    return True

        return False
    except (OSError, RuntimeError):
        # Error resolving path (circular symlink, etc.)
        return False


class TapbackType(Enum):
    """iMessage tapback reaction types."""

    LOVE = 0
    LIKE = 1
    DISLIKE = 2
    LAUGH = 3
    EMPHASIS = 4  # !!
    QUESTION = 5  # ?


@dataclass
class SendResult:
    """Result of sending a message."""

    success: bool
    error: str | None = None


class IMessageSender:
    """Send iMessages via AppleScript.

    Supports sending text messages and attachments to both individual
    recipients and group chats.
    """

    def __init__(self) -> None:
        """Initialize the sender."""
        pass

    def send_message(
        self,
        text: str,
        *,
        recipient: str | None = None,
        chat_id: str | None = None,
        is_group: bool = False,
    ) -> SendResult:
        """Send a message to a conversation.

        For individual chats, provide recipient (phone/email).
        For group chats, provide chat_id and set is_group=True.

        Args:
            text: Message text to send
            recipient: Phone number or email (for individual chats)
            chat_id: The chat ID from the database (for group chats)
            is_group: Whether this is a group chat

        Returns:
            SendResult with success status and optional error message
        """
        if not text:
            return SendResult(success=False, error="Message text is required")

        if is_group:
            return self._send_to_group(text, chat_id)
        else:
            return self._send_to_individual(text, recipient)

    def send_attachment(
        self,
        file_path: str | Path,
        *,
        recipient: str | None = None,
        chat_id: str | None = None,
        is_group: bool = False,
    ) -> SendResult:
        """Send a file attachment to a conversation.

        Args:
            file_path: Path to the file to send
            recipient: Phone number or email (for individual chats)
            chat_id: The chat ID from the database (for group chats)
            is_group: Whether this is a group chat

        Returns:
            SendResult with success status and optional error message
        """
        path = Path(file_path)
        if not path.exists():
            return SendResult(success=False, error=f"File not found: {file_path}")

        # SECURITY: Validate path is within allowed directories to prevent path traversal
        if not _validate_file_path(path):
            logger.warning("Rejected file path outside allowed directories: %s", path)
            return SendResult(
                success=False,
                error="File path is outside allowed directories for security reasons",
            )

        # Get absolute POSIX path for AppleScript
        abs_path = str(path.resolve())
        escaped_path = abs_path.replace("\\", "\\\\").replace('"', '\\"')

        if is_group:
            if not chat_id:
                return SendResult(success=False, error="Chat ID required for group")
            # Extract the actual chat identifier from the guid format
            actual_chat_id = chat_id
            if ";" in chat_id:
                parts = chat_id.split(";")
                actual_chat_id = parts[-1] if len(parts) > 1 else chat_id
            escaped_chat_id = actual_chat_id.replace("\\", "\\\\").replace('"', '\\"')
            applescript = f"""
tell application "Messages"
    set targetChat to chat id "{escaped_chat_id}"
    set theFile to POSIX file "{escaped_path}"
    send theFile to targetChat
end tell
"""
        else:
            if not recipient:
                return SendResult(success=False, error="Recipient required")
            escaped_recipient = recipient.replace("\\", "\\\\").replace('"', '\\"')
            applescript = f"""
tell application "Messages"
    set targetService to 1st account whose service type = iMessage
    set targetBuddy to participant "{escaped_recipient}" of targetService
    set theFile to POSIX file "{escaped_path}"
    send theFile to targetBuddy
end tell
"""
        return self._run_applescript(applescript, f"attachment to {chat_id or recipient}")

    @staticmethod
    def _validate_recipient_format(recipient: str) -> bool:
        """Validate that recipient is a valid phone number or email address.

        Args:
            recipient: Phone number or email address to validate

        Returns:
            True if valid format, False otherwise
        """
        import re

        recipient = recipient.strip()
        if not recipient:
            return False

        # Check for email format (basic validation)
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if re.match(email_pattern, recipient):
            return True

        # Check for phone number format (various formats)
        # Accepts: +1234567890, (123) 456-7890, 123-456-7890, 1234567890
        phone_pattern = r"^[\+]?[\d\s\(\)\-\.]{10,20}$"
        if re.match(phone_pattern, recipient):
            # Ensure it has at least 10 digits
            digits = re.sub(r"\D", "", recipient)
            return len(digits) >= 10

        return False

    @staticmethod
    def _escape_for_applescript(text: str) -> str:
        """Escape text for safe embedding in AppleScript strings.

        Handles backslashes, quotes, and control characters that have
        special meaning in AppleScript.

        Args:
            text: Raw text to escape

        Returns:
            Escaped text safe for AppleScript double-quoted strings
        """
        # Order matters: escape backslashes first
        text = text.replace("\\", "\\\\")
        text = text.replace('"', '\\"')
        text = text.replace("\r", "\\r")
        text = text.replace("\n", "\\n")
        text = text.replace("\t", "\\t")
        return text

    def _send_to_individual(self, text: str, recipient: str | None) -> SendResult:
        """Send a message to an individual recipient.

        Validates recipient format before sending to prevent AppleScript errors.
        """
        if not recipient:
            return SendResult(success=False, error="Recipient is required")

        # Validate recipient format (phone number or email)
        if not self._validate_recipient_format(recipient):
            return SendResult(
                success=False,
                error=f"Invalid recipient format: {recipient}. Must be phone number or email.",
            )

        # Escape special characters for AppleScript
        escaped_text = self._escape_for_applescript(text)
        escaped_recipient = recipient.replace("\\", "\\\\").replace('"', '\\"')

        applescript = f"""
tell application "Messages"
    set targetService to 1st account whose service type = iMessage
    set targetBuddy to participant "{escaped_recipient}" of targetService
    send "{escaped_text}" to targetBuddy
end tell
"""
        return self._run_applescript(applescript, f"individual: {recipient}")

    def _send_to_group(self, text: str, chat_id: str | None) -> SendResult:
        """Send a message to a group chat."""
        if not chat_id:
            return SendResult(success=False, error="Chat ID is required for group chats")

        # Extract the actual chat identifier from the guid format
        # Database guid format: "iMessage;+;chat<numbers>" or "iMessage;-;+<phone>"
        # We need just the "chat<numbers>" part for AppleScript
        actual_chat_id = chat_id
        if ";" in chat_id:
            # Split by semicolon and take the last part
            parts = chat_id.split(";")
            actual_chat_id = parts[-1] if len(parts) > 1 else chat_id

        # Escape special characters for AppleScript
        escaped_text = self._escape_for_applescript(text)
        escaped_chat_id = actual_chat_id.replace("\\", "\\\\").replace('"', '\\"')

        # For group chats, we find the chat by its ID and send directly to it
        applescript = f"""
tell application "Messages"
    set targetChat to chat id "{escaped_chat_id}"
    send "{escaped_text}" to targetChat
end tell
"""
        return self._run_applescript(applescript, f"group: {chat_id}")

    def _run_applescript(self, script: str, target_desc: str) -> SendResult:
        """Execute an AppleScript and return the result."""
        try:
            logger.info("Sending iMessage to %s", target_desc)
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                check=True,
                text=True,
                timeout=10,
            )
            logger.info("Message sent successfully")
            return SendResult(success=True)

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else "AppleScript execution failed"
            logger.error("Failed to send message: %s", error_msg)
            return SendResult(success=False, error=error_msg)

        except subprocess.TimeoutExpired:
            logger.error("Message send timed out")
            return SendResult(success=False, error="Timeout sending message")

        except Exception as e:
            logger.exception("Unexpected error sending message")
            return SendResult(success=False, error=str(e))
