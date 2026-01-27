"""JARVIS MCP Server - Model Context Protocol server for Claude Code integration.

This module provides an MCP-compliant server that exposes JARVIS functionality
as tools that can be used by Claude Code or other MCP clients.

Available tools:
- search_messages: Search iMessage conversations with filters
- get_summary: Get AI-generated conversation summary
- generate_reply: Generate reply suggestions for a conversation
- get_contact_info: Retrieve contact information
- list_conversations: List recent conversations

Usage:
    # Start the MCP server
    jarvis mcp-serve

    # Or run directly
    python -m mcp_server.server
"""

from mcp_server.server import MCPServer

__all__ = ["MCPServer"]
__version__ = "1.0.0"
