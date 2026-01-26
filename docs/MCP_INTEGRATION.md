# JARVIS MCP Integration Guide

This guide explains how to use JARVIS as a tool provider for Claude Code and other MCP (Model Context Protocol) compatible clients.

## Overview

The Model Context Protocol (MCP) is Anthropic's standard for connecting AI assistants with external tools and data sources. JARVIS implements an MCP server that exposes its iMessage functionality as tools that Claude Code can use.

## Quick Start

### 1. Start the MCP Server

```bash
# Start in stdio mode (for Claude Code integration)
jarvis mcp-serve

# Or start in HTTP mode (for network access)
jarvis mcp-serve --transport http --port 8765
```

### 2. Configure Claude Code

Add JARVIS to your Claude Code MCP settings. Create or edit `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "jarvis": {
      "command": "jarvis",
      "args": ["mcp-serve"]
    }
  }
}
```

If you installed JARVIS in a specific Python environment, use the full path:

```json
{
  "mcpServers": {
    "jarvis": {
      "command": "/path/to/your/venv/bin/jarvis",
      "args": ["mcp-serve"]
    }
  }
}
```

### 3. Restart Claude Code

After updating the configuration, restart Claude Code for the changes to take effect.

## Available Tools

JARVIS exposes the following tools through MCP:

### search_messages

Search through iMessage conversations with powerful filtering.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| query | string | Yes | Search query to match against message text |
| limit | integer | No | Max results (default: 20, max: 100) |
| sender | string | No | Filter by sender phone/email |
| start_date | string | No | Only messages after this date (ISO 8601) |
| end_date | string | No | Only messages before this date (ISO 8601) |
| has_attachments | boolean | No | Filter by attachment presence |
| chat_id | string | No | Limit to specific conversation |

**Example Usage in Claude Code:**
```
Search my messages for "dinner plans" from the last week
```

### get_summary

Get an AI-generated summary of a conversation.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| person_name | string | One of these | Name of person to summarize conversation with |
| chat_id | string | One of these | Specific conversation ID |
| num_messages | integer | No | Messages to include (default: 50, max: 200) |

**Example Usage in Claude Code:**
```
Summarize my recent conversation with John
```

### generate_reply

Generate AI-powered reply suggestions for a conversation.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| person_name | string | One of these | Name of person to reply to |
| chat_id | string | One of these | Specific conversation ID |
| instruction | string | No | Guide the reply tone (e.g., "accept enthusiastically") |
| num_suggestions | integer | No | Number of suggestions (default: 3, max: 5) |
| context_messages | integer | No | Messages for context (default: 20, max: 50) |

**Example Usage in Claude Code:**
```
Generate a reply to Sarah's last message - I want to accept her invitation
```

### get_contact_info

Retrieve contact information for a phone number or email.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| identifier | string | Yes | Phone number or email address |

**Example Usage in Claude Code:**
```
Get contact info for +15551234567
```

### list_conversations

List recent iMessage conversations.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| limit | integer | No | Max conversations (default: 20, max: 100) |
| since | string | No | Only conversations with messages after this date |

**Example Usage in Claude Code:**
```
Show my recent iMessage conversations
```

### get_conversation_messages

Get recent messages from a specific conversation.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| person_name | string | One of these | Name of person whose conversation to retrieve |
| chat_id | string | One of these | Specific conversation ID |
| limit | integer | No | Max messages (default: 20, max: 100) |

**Example Usage in Claude Code:**
```
Show me my last 10 messages with Mom
```

## Transport Modes

### Stdio Mode (Default)

The stdio transport is designed for direct integration with Claude Code. Communication happens through stdin/stdout using newline-delimited JSON-RPC 2.0.

```bash
jarvis mcp-serve
# or explicitly
jarvis mcp-serve --transport stdio
```

### HTTP Mode

The HTTP transport exposes the MCP server as an HTTP endpoint, useful for network access or testing.

```bash
jarvis mcp-serve --transport http
jarvis mcp-serve --transport http --host 0.0.0.0 --port 9000
```

**Endpoints:**
- `POST /mcp` - Main MCP endpoint
- `GET /health` - Health check endpoint

## Protocol Details

JARVIS implements the Model Context Protocol specification (version 2024-11-05).

### JSON-RPC 2.0 Format

All communication uses JSON-RPC 2.0:

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "search_messages",
    "arguments": {
      "query": "dinner",
      "limit": 10
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"messages\": [...], \"count\": 5}"
      }
    ]
  }
}
```

### MCP Methods

| Method | Description |
|--------|-------------|
| `initialize` | Initialize the connection |
| `initialized` | Notification that client is ready |
| `tools/list` | List available tools |
| `tools/call` | Execute a tool |
| `ping` | Health check |
| `shutdown` | Graceful shutdown |

## Permissions

JARVIS MCP server requires the same permissions as the main JARVIS application:

1. **Full Disk Access** - Required to read the iMessage database
   - Grant in: System Settings > Privacy & Security > Full Disk Access
   - Add your terminal application (Terminal, iTerm2, etc.)

2. **Apple Silicon Required** - MLX models run on Apple Silicon
   - The AI generation features (summary, reply) require Apple Silicon

## Troubleshooting

### Server Not Starting

1. Ensure JARVIS is installed correctly:
   ```bash
   jarvis version
   ```

2. Check the MCP server directly:
   ```bash
   jarvis mcp-serve -v
   ```

### Tools Not Working

1. Verify Full Disk Access permission:
   ```bash
   jarvis health
   ```

2. Check the logs (stderr in stdio mode):
   ```bash
   jarvis mcp-serve 2>mcp.log
   ```

### Connection Issues

1. Verify the configuration path:
   ```bash
   cat ~/.claude/claude_desktop_config.json
   ```

2. Test the command directly:
   ```bash
   /path/to/jarvis mcp-serve --transport http
   curl http://localhost:8765/health
   ```

### Debug Mode

Enable verbose logging:
```bash
jarvis -v mcp-serve
```

## Testing the MCP Server

### Manual Testing

You can test the MCP server manually using HTTP mode:

```bash
# Start server
jarvis mcp-serve --transport http --port 8765

# In another terminal, test the health endpoint
curl http://localhost:8765/health

# Test tools/list
curl -X POST http://localhost:8765/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'

# Test a tool
curl -X POST http://localhost:8765/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"list_conversations","arguments":{"limit":5}}}'
```

### Integration Testing

Run the JARVIS test suite to verify MCP functionality:
```bash
make test
```

## Security Considerations

1. **Local Only by Default** - The HTTP transport binds to `127.0.0.1` by default
2. **No Authentication** - The MCP server does not implement authentication
3. **Read-Only iMessage Access** - The server only reads messages, never sends
4. **Data Privacy** - All processing happens locally, no data sent to cloud services

If exposing the HTTP endpoint on `0.0.0.0`, consider:
- Using a reverse proxy with authentication
- Restricting access via firewall rules
- Running in a trusted network environment

## Architecture

```
┌─────────────────┐     JSON-RPC 2.0      ┌─────────────────┐
│   Claude Code   │◄────────────────────►│  JARVIS MCP     │
│                 │      (stdio/http)     │    Server       │
└─────────────────┘                       └────────┬────────┘
                                                   │
                                         ┌─────────┼─────────┐
                                         │         │         │
                                    ┌────▼───┐ ┌───▼───┐ ┌───▼────┐
                                    │iMessage│ │  MLX  │ │Contacts│
                                    │ Reader │ │ Model │ │ Access │
                                    └────────┘ └───────┘ └────────┘
```

## Additional Resources

- [Model Context Protocol Specification](https://modelcontextprotocol.io/specification)
- [JARVIS CLI Guide](CLI_GUIDE.md)
- [JARVIS README](../README.md)
