# V2 Architecture: Performance Optimizations

## The Problem with V1 (HTTP Polling)

```
Desktop App ←─ HTTP (50-150ms) ─→ FastAPI Server
                    │
                    └─ Polling every 10s for new messages
                       Up to 10 second delay!
```

**Pain points:**

- 50-150ms latency for every operation
- Polling wastes resources
- Up to 10 second delay for new messages

## V2 Solution: Direct SQLite + Unix Sockets

```
┌─────────────────────────────────────────────────────┐
│ Tauri App                                           │
│                                                     │
│  SQLite Plugin        Socket Client                 │
│  (direct reads)       (JSON-RPC + push)             │
│       │                      │                      │
└───────┼──────────────────────┼──────────────────────┘
        │ ~1-5ms               │ ~1-5ms + inference
        ▼                      ▼
   chat.db / jarvis.db    ~/.jarvis/jarvis.sock + File Watcher
```

## Performance Comparison

| Operation                | V1 (HTTP)         | V2 (Socket)        | Improvement           |
| ------------------------ | ----------------- | ------------------ | --------------------- |
| Load conversations       | ~100-150ms        | ~1-5ms             | **30-50x**            |
| Load messages            | ~100-150ms        | ~1-5ms             | **30-50x**            |
| New message notification | Up to 10s         | Instant            | **Instant**           |
| Generate draft           | ~50ms + inference | ~1-5ms + inference | **10x less overhead** |
| Idle resource usage      | Constant polling  | Zero               | **No waste**          |

## Unix Socket Protocol

JSON-RPC 2.0 over newline-delimited JSON.

### Available Methods

| Method                | Streaming | Description                               |
| --------------------- | --------- | ----------------------------------------- |
| `ping`                | No        | Health check                              |
| `generate_draft`      | Yes       | Generate reply draft for a conversation   |
| `summarize`           | Yes       | Summarize a conversation                  |
| `get_smart_replies`   | No        | Get quick reply suggestions               |
| `semantic_search`     | No        | Search message history by meaning         |
| `batch`               | No        | Execute multiple RPC calls in one request |
| `resolve_contacts`    | No        | Resolve contact info from handles         |
| `list_conversations`  | No        | List recent conversations                 |
| `get_routing_metrics` | No        | Get routing/classification metrics        |
| `prefetch_stats`      | No        | Get prefetch cache statistics             |
| `prefetch_invalidate` | No        | Invalidate prefetch cache entries         |
| `prefetch_focus`      | No        | Signal that a conversation is focused     |
| `prefetch_hover`      | No        | Signal that a conversation is hovered     |

### Message Format

```json
// Request
{"jsonrpc": "2.0", "method": "generate_draft", "params": {"chat_id": "..."}, "id": 1}

// Response
{"jsonrpc": "2.0", "result": {"suggestions": [...]}, "id": 1}

// Push notification (no id)
{"jsonrpc": "2.0", "method": "new_message", "params": {"chat_id": "...", "text": "..."}}
```

## Real Token Streaming

For LLM generation, tokens stream as they're generated:

```json
{"jsonrpc": "2.0", "method": "stream.token", "params": {"token": "Sure", "index": 0}}
{"jsonrpc": "2.0", "method": "stream.token", "params": {"token": ",", "index": 1}}
{"jsonrpc": "2.0", "method": "stream.token", "params": {"token": " I'm", "index": 2}}
...
// Final response
{"jsonrpc": "2.0", "result": {"suggestions": [...]}, "id": 1}
```

## Why Unix Sockets Over HTTP?

| Option           | Pros                         | Cons              |
| ---------------- | ---------------------------- | ----------------- |
| HTTP (FastAPI)   | Simple, works                | 50-150ms overhead |
| gRPC             | Typed, fast                  | Complex setup     |
| **Unix sockets** | **Fast, simple, local-only** | Local only        |

We keep HTTP for CLI and potential remote clients.
