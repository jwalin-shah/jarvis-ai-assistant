"""MCP Tool definitions with JSON schemas.

Defines all tools exposed by the JARVIS MCP server following the
Model Context Protocol specification.

Each tool has:
- name: Unique identifier for the tool
- description: Human-readable description
- inputSchema: JSON Schema for the tool's parameters
"""

from typing import Any

# Tool definitions following MCP specification
TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_messages",
        "description": (
            "Search through iMessage conversations. Supports filtering by query text, "
            "sender, date range, and attachments. Returns matching messages with context."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to match against message text (required)",
                    "minLength": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 20, max: 100)",
                    "default": 20,
                    "minimum": 1,
                    "maximum": 100,
                },
                "sender": {
                    "type": "string",
                    "description": (
                        "Filter by sender phone number or email. "
                        "Use 'me' to search only your sent messages."
                    ),
                },
                "start_date": {
                    "type": "string",
                    "description": "Only include messages after this date (ISO 8601 format)",
                    "format": "date-time",
                },
                "end_date": {
                    "type": "string",
                    "description": "Only include messages before this date (ISO 8601 format)",
                    "format": "date-time",
                },
                "has_attachments": {
                    "type": "boolean",
                    "description": "Filter to messages with (true) or without (false) attachments",
                },
                "chat_id": {
                    "type": "string",
                    "description": "Limit search to a specific conversation by chat_id",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_summary",
        "description": (
            "Get an AI-generated summary of a conversation. Returns a brief summary "
            "and key points extracted from the conversation history."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "person_name": {
                    "type": "string",
                    "description": (
                        "Name of the person to summarize conversation with. "
                        "Matched against iMessage contacts/conversations."
                    ),
                },
                "chat_id": {
                    "type": "string",
                    "description": (
                        "Specific conversation ID to summarize. Use this OR person_name, not both."
                    ),
                },
                "num_messages": {
                    "type": "integer",
                    "description": "Number of messages to include in summary (default: 50)",
                    "default": 50,
                    "minimum": 10,
                    "maximum": 200,
                },
            },
            "oneOf": [
                {"required": ["person_name"]},
                {"required": ["chat_id"]},
            ],
        },
    },
    {
        "name": "generate_reply",
        "description": (
            "Generate AI-powered reply suggestions for a conversation. "
            "Analyzes conversation context to suggest contextually appropriate responses."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "person_name": {
                    "type": "string",
                    "description": (
                        "Name of the person to generate reply for. "
                        "Matched against iMessage contacts/conversations."
                    ),
                },
                "chat_id": {
                    "type": "string",
                    "description": (
                        "Specific conversation ID to generate reply for. "
                        "Use this OR person_name, not both."
                    ),
                },
                "instruction": {
                    "type": "string",
                    "description": (
                        "Optional instruction to guide the reply tone/content. "
                        "Examples: 'accept enthusiastically', 'politely decline', 'be brief'"
                    ),
                },
                "num_suggestions": {
                    "type": "integer",
                    "description": "Number of reply suggestions to generate (default: 3)",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 5,
                },
                "context_messages": {
                    "type": "integer",
                    "description": ("Number of previous messages to use for context (default: 20)"),
                    "default": 20,
                    "minimum": 5,
                    "maximum": 50,
                },
            },
            "oneOf": [
                {"required": ["person_name"]},
                {"required": ["chat_id"]},
            ],
        },
    },
    {
        "name": "get_contact_info",
        "description": (
            "Get contact information for a phone number or email address. "
            "Returns display name, initials, and avatar availability."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "description": (
                        "Phone number (e.g., '+15551234567') or email address. "
                        "Phone numbers can include or exclude country code."
                    ),
                },
            },
            "required": ["identifier"],
        },
    },
    {
        "name": "list_conversations",
        "description": (
            "List recent iMessage conversations sorted by last message date. "
            "Returns conversation metadata including participants and message counts."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of conversations to return (default: 20)",
                    "default": 20,
                    "minimum": 1,
                    "maximum": 100,
                },
                "since": {
                    "type": "string",
                    "description": (
                        "Only return conversations with messages after this date (ISO 8601 format)"
                    ),
                    "format": "date-time",
                },
            },
        },
    },
    {
        "name": "get_conversation_messages",
        "description": (
            "Get recent messages from a specific conversation. "
            "Returns messages with sender info, timestamps, and attachments."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "person_name": {
                    "type": "string",
                    "description": (
                        "Name of the person whose conversation to retrieve. "
                        "Matched against iMessage contacts/conversations."
                    ),
                },
                "chat_id": {
                    "type": "string",
                    "description": ("Specific conversation ID. Use this OR person_name, not both."),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of messages to return (default: 20)",
                    "default": 20,
                    "minimum": 1,
                    "maximum": 100,
                },
            },
            "oneOf": [
                {"required": ["person_name"]},
                {"required": ["chat_id"]},
            ],
        },
    },
]


def get_tool_definitions() -> list[dict[str, Any]]:
    """Get all tool definitions.

    Returns:
        List of tool definition dictionaries following MCP specification.
    """
    return TOOLS


def get_tool_by_name(name: str) -> dict[str, Any] | None:
    """Get a specific tool definition by name.

    Args:
        name: The tool name to look up.

    Returns:
        Tool definition dictionary or None if not found.
    """
    for tool in TOOLS:
        if tool["name"] == name:
            return tool
    return None
