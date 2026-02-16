# JARVIS Architecture V2: Direct SQLite + Unix Sockets

> **Last Updated:** 2026-02-12

## Overview

This document describes the optimized architecture for JARVIS, replacing HTTP polling with direct SQLite reads and Unix socket communication.

**Status:** All Phases Complete (1-4)

## Quick Start

```bash
# Launch includes socket server automatically
./scripts/production/launch.sh
```

## Socket Server API

The socket server supports two protocols:

- **Unix Socket:** `~/.jarvis/jarvis.sock` (for Tauri app)
- **WebSocket:** `ws://localhost:8743` (for browser/Playwright)

Protocol: JSON-RPC 2.0 over newline-delimited JSON

```
Request:  {"jsonrpc": "2.0", "method": "...", "params": {...}, "id": 1}
Response: {"jsonrpc": "2.0", "result": {...}, "id": 1}
Error:    {"jsonrpc": "2.0", "error": {"code": -32600, "message": "..."}, "id": 1}
```

**Available Methods:**
| Method | Description | Parameters | Streaming |
|--------|-------------|------------|-----------|
| `ping` | Health check | None | No |
| `generate_draft` | Generate reply suggestions | `chat_id`, `instruction?`, `num_suggestions?`, `stream?` | Yes |
| `summarize` | Summarize conversation | `chat_id`, `num_messages?`, `stream?` | Yes |
| `get_smart_replies` | Quick reply suggestions | `last_message`, `num_suggestions?` | No |
| `semantic_search` | Search messages | `query`, `chat_id?`, `limit?`, `threshold?` | No |
| `list_conversations` | List recent conversations | `limit?` | No |
| `batch` | Execute multiple RPC calls | `requests` (array) | No |
| `resolve_contacts` | Resolve contact info | `handles` | No |
| `get_contacts` | Get contacts | `limit?`, `search?` | No |
| `chat` | Direct chat with SLM | `message`, `context?`, `stream?` | Yes |
| `get_routing_metrics` | Get routing metrics | None | No |
| `get_performance_slo` | Get SLO performance | None | No |
| `get_draft_metrics` | Get draft quality metrics | `chat_id?` | No |
| `prefetch_stats` | Get prefetch cache stats | None | No |
| `prefetch_focus` | Signal conversation focused | `chat_id` | No |
| `prefetch_hover` | Signal conversation hovered | `chat_id` | No |

**Streaming:**
For streaming methods (`generate_draft`, `summarize`, `chat`), set `"stream": true` in params. Tokens are sent as notifications, followed by final response.

**Push Notifications:**
| Event | Description | Data |
|-------|-------------|------|
| `new_message` | New message received | `message_id`, `chat_id`, `sender`, `text`, `date`, `is_from_me` |
| `streaming_token` | Token generated (streaming) | `token`, `done` |
| `prefetch_complete` | Prefetch finished | `chat_id` |

## Current Architecture (V1)

```
┌─────────────────────────────────────────────────────────────────────┐
│  Tauri App (Frontend)                                               │
│  └── Every operation goes through HTTP                              │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ HTTP (50-150ms per request)
┌─────────────────────────────────────────────────────────────────────┐
│  FastAPI (Python)                                                   │
│  ├── Reads chat.db for messages                                     │
│  ├── Reads/writes jarvis.db for embeddings                          │
│  └── Runs LLM for drafts, classification, etc.                      │
└─────────────────────────────────────────────────────────────────────┘

Problems:
- 50-150ms latency for every operation (including simple reads)
- Polling every 10s for new messages (wasteful)
- Up to 10 second delay for new message notifications
- HTTP overhead for local communication
```

## New Architecture (V2)

```
┌─────────────────────────────────────────────────────────────────────┐
│  Tauri App (Frontend)                                               │
│                                                                     │
│  ┌──────────────────────┐     ┌──────────────────────────────────┐ │
│  │  SQLite Plugin       │     │  Socket Client                   │ │
│  │  (tauri-plugin-sql)  │     │  (JSON-RPC over Unix socket)     │ │
│  └──────────┬───────────┘     └──────────────┬───────────────────┘ │
│             │                                │                      │
└─────────────┼────────────────────────────────┼──────────────────────┘
              │ ~1-5ms                         │ ~1-5ms + inference
              ▼                                ▼
       ┌──────────────┐              ┌─────────────────────┐
       │   chat.db    │              │  ~/.jarvis/jarvis.sock   │
       │  (read-only) │              │  (Python daemon)    │
       └──────────────┘              └──────────┬──────────┘
       ┌──────────────┐                         │
       │  jarvis.db   │◄────────────────────────┤
       │(contacts/vec)│                         │
       └──────────────┘              ┌──────────┴──────────┐
                                     │  - LLM inference    │
                                     │  - sqlite-vec search│
                                     │  - Classification   │
                                     │  - File watcher     │
                                     │  - Speculative prefetch
                                     └─────────────────────┘
```

## Component Details

### 1. Direct SQLite Reads (Tauri → chat.db / jarvis.db)

**What:** Tauri reads directly from SQLite databases, bypassing HTTP entirely.

**Why:** Reading 50 messages takes ~1-5ms via SQLite vs ~100-150ms via HTTP.

**Databases:**
| Database | Location | Access | Contains |
|----------|----------|--------|----------|
| chat.db | ~/Library/Messages/chat.db | Read-only | Messages, conversations, attachments |
| jarvis.db | ~/.jarvis/jarvis.db | Read-write | Contacts, segments, facts, and scheduled messages |
| Vec Index | ~/.jarvis/jarvis.db | Read-write | Vector index (sqlite-vec) in vec_messages/vec_chunks |

**Implementation:**

```typescript
// desktop/src/lib/db/direct.ts
import Database from '@tauri-apps/plugin-sql';

let chatDb: Database | null = null;
let jarvisDb: Database | null = null;

export async function initDatabases() {
  const homeDir = await homeDir();

  // Apple's iMessage database (read-only)
  chatDb = await Database.load(`sqlite:${homeDir}/Library/Messages/chat.db?mode=ro`);

  // Our database
  jarvisDb = await Database.load(`sqlite:${homeDir}/.jarvis/jarvis.db`);
}

export async function getMessages(chatId: string, limit = 50): Promise<Message[]> {
  // Direct read - ~1-5ms
  return chatDb.select(
    `
    SELECT
      m.ROWID as id,
      m.text,
      m.date,
      m.is_from_me,
      m.cache_has_attachments,
      h.id as sender
    FROM message m
    LEFT JOIN handle h ON m.handle_id = h.ROWID
    WHERE m.chat_id = ?
    ORDER BY m.date DESC
    LIMIT ?
  `,
    [chatId, limit]
  );
}

export async function getConversations(): Promise<Conversation[]> {
  // Direct read - ~1-5ms
  return chatDb.select(`
    SELECT
      c.ROWID as id,
      c.chat_identifier as chat_id,
      c.display_name,
      c.group_id
    FROM chat c
    ORDER BY c.last_message_date DESC
  `);
}

// Vector search is handled server-side via sqlite-vec
// Use the socket client for semantic_search instead of direct embedding access
```

**Tauri Config (src-tauri/tauri.conf.json):**

```json
{
  "plugins": {
    "sql": {
      "preload": ["sqlite:~/.jarvis/jarvis.db"]
    }
  }
}
```

**Cargo.toml:**

```toml
[dependencies]
tauri-plugin-sql = { version = "2", features = ["sqlite"] }
```

---

### 2. Unix Socket Communication (Tauri ↔ Python)

**What:** Bidirectional JSON-RPC communication over Unix socket.

**Why:**

- ~1-5ms latency vs ~50ms HTTP
- Enables push notifications (no polling)
- Persistent connection (no connection overhead)

**Socket Locations:**

- Unix Socket: `~/.jarvis/jarvis.sock`
- WebSocket: `ws://localhost:8743` (with auth token at `~/.jarvis/ws_token`)

**Protocol:** JSON-RPC 2.0 over newline-delimited JSON

**Python Server (jarvis/socket_server.py):**

```python
import asyncio
import websockets
from websockets.server import ServerConnection

# Configuration
SOCKET_PATH = Path.home() / ".jarvis" / "jarvis.sock"
WS_PORT = 8743

class JarvisSocketServer:
    def __init__(self, enable_watcher: bool = True, preload_models: bool = True):
        self._methods: dict[str, Callable] = {}
        self._streaming_methods: set[str] = set()
        self._clients: set[asyncio.StreamWriter] = set()
        self._ws_clients: set[ServerConnection] = set()
        self._rate_limiter = RateLimiter(max_requests=100, window_seconds=1.0)

        # Register built-in methods
        self._register_methods()

    def _register_methods(self) -> None:
        """Register available RPC methods."""
        self.register("ping", self._ping)
        self.register("generate_draft", self._generate_draft, streaming=True)
        self.register("summarize", self._summarize, streaming=True)
        self.register("get_smart_replies", self._get_smart_replies)
        self.register("semantic_search", self._semantic_search)
        self.register("batch", self._batch)
        self.register("resolve_contacts", self._resolve_contacts)
        self.register("get_contacts", self._get_contacts)
        self.register("list_conversations", self._list_conversations)
        self.register("chat", self._chat, streaming=True)
        self.register("get_routing_metrics", self._get_routing_metrics)
        self.register("get_performance_slo", self._get_performance_slo)
        self.register("get_draft_metrics", self._get_draft_metrics)
        self.register("prefetch_stats", self._prefetch_stats)
        self.register("prefetch_focus", self._prefetch_focus)
        self.register("prefetch_hover",_hover)

    self._prefetch def register(self, name: str, handler, streaming: bool = False) -> None:
        """Register a method handler."""
        self._methods[name] = handler
        if streaming:
            self._streaming_methods.add(name)

    async def start(self):
        # Start Unix socket server
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
        self._server = await asyncio.start_unix_server(
            self._handle_client, path=SOCKET_PATH
        )
        SOCKET_PATH.chmod(0o600)

        # Start WebSocket server
        self._ws_server = await websockets.serve(
            self._handle_ws_client, "localhost", WS_PORT
        )

        # Start file watcher and model preload
        await asyncio.gather(
            self._server.serve_forever(),
            self._watcher.watch() if self._watcher else asyncio.sleep(float('inf'))
        )

    async def _handle_client(self, reader, writer):
        """Handle Unix socket client."""
        # Length-prefixed JSON-RPC protocol
        length_bytes = await reader.readexactly(4)
        length = int.from_bytes(length_bytes, 'big')
        data = await reader.readexactly(length)
        request = json.loads(data.decode())

        response = await self._handle_request(request)
        if response:
            await self._send(writer, response)

    async def _handle_ws_client(self, websocket: ServerConnection):
        """Handle WebSocket client."""
        self._ws_clients.add(websocket)
        try:
            async for message in websocket:
                request = json.loads(message)
                response = await self._handle_request(request)
                if response:
                    await websocket.send(json.dumps(response))
        finally:
            self._ws_clients.discard(websocket)

    async def _handle_request(self, request: dict) -> dict:
        """Process JSON-RPC request with rate limiting."""
        # Rate limit check
        client_id = request.get("client_id", "unknown")
        if not self._rate_limiter.is_allowed(client_id):
            return {"jsonrpc": "2.0", "error": {"code": -32001, "message": "Rate limited"}, "id": request.get("id")}

        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        handler = self._methods.get(method)
        if not handler:
            return {"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Method not found: {method}"}, "id": request_id}

        try:
            result = await handler(params)
            return {"jsonrpc": "2.0", "result": result, "id": request_id}
        except JsonRpcError as e:
            return {"jsonrpc": "2.0", "error": {"code": e.code, "message": e.message}, "id": request_id}
        except Exception as e:
            return {"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}, "id": request_id}

    async def broadcast(self, method: str, params: dict):
        """Push notification to all connected clients."""
        message = {"jsonrpc": "2.0", "method": method, "params": params}

        # Unix socket clients
        for writer in self._clients:
            try:
                await self._send(writer, message)
            except:
                self._clients.discard(writer)

        # WebSocket clients
        for ws in self._ws_clients:
            try:
                await ws.send(json.dumps(message))
            except:
                self._ws_clients.discard(ws)

    async def _generate_draft(self, params: dict) -> dict:
        """Generate draft replies with optional streaming."""
        from jarvis.reply_service import get_reply_service

        stream = params.get("stream", False)
        chat_id = params["chat_id"]

        if stream:
            # Register streaming callback for token notifications
            async def on_token(token: str):
                await self.broadcast("streaming_token", {"token": token, "done": False})

            service = get_reply_service()
            result = await asyncio.to_thread(service.route_legacy, incoming="", chat_id=chat_id)
            return {"drafts": result}

        service = get_reply_service()
        result = await asyncio.to_thread(service.route_legacy, incoming="", chat_id=chat_id)
        return {"drafts": result}
```

**TypeScript Client (desktop/src/lib/socket/client.ts):**

```typescript
import { invoke } from '@tauri-apps/api/core';

type JsonRpcRequest = {
  jsonrpc: '2.0';
  method: string;
  params?: Record<string, unknown>;
  id?: number;
};

type JsonRpcResponse = {
  jsonrpc: '2.0';
  result?: unknown;
  error?: { code: number; message: string };
  id?: number;
};

type PushHandler = (params: Record<string, unknown>) => void;

class JarvisSocket {
  private requestId = 0;
  private pendingRequests = new Map<
    number,
    {
      resolve: (value: unknown) => void;
      reject: (error: Error) => void;
    }
  >();
  private pushHandlers = new Map<string, PushHandler>();
  private connected = false;

  async connect(): Promise<void> {
    // Tauri command connects to Unix socket
    await invoke('connect_jarvis_socket');
    this.connected = true;

    // Start listening for messages
    this.startListening();
  }

  async call<T>(method: string, params?: Record<string, unknown>): Promise<T> {
    const id = ++this.requestId;
    const request: JsonRpcRequest = {
      jsonrpc: '2.0',
      method,
      params,
      id,
    };

    return new Promise((resolve, reject) => {
      this.pendingRequests.set(id, { resolve: resolve as (v: unknown) => void, reject });
      invoke('send_jarvis_message', { message: JSON.stringify(request) });
    });
  }

  on(method: string, handler: PushHandler): void {
    this.pushHandlers.set(method, handler);
  }

  private async startListening(): Promise<void> {
    // Listen for messages from Tauri backend
    await listen('jarvis_message', (event) => {
      const message = JSON.parse(event.payload as string) as JsonRpcResponse;

      if (message.id !== undefined) {
        // Response to a request
        const pending = this.pendingRequests.get(message.id);
        if (pending) {
          this.pendingRequests.delete(message.id);
          if (message.error) {
            pending.reject(new Error(message.error.message));
          } else {
            pending.resolve(message.result);
          }
        }
      } else {
        // Push notification
        const handler = this.pushHandlers.get(message.method!);
        if (handler) {
          handler(message.params as Record<string, unknown>);
        }
      }
    });
  }
}

export const jarvis = new JarvisSocket();
```

---

### 3. File Watcher (Python watches chat.db)

**What:** Python daemon watches chat.db for changes, computes embeddings, pushes notifications.

**Why:**

- Instant new message notifications (no polling)
- Embeddings computed in background
- Tauri UI updates immediately

**Technology Choice:** `watchfiles` (Rust-based, uses the `notify` crate internally). On macOS this uses FSEvents natively, giving ~100ms detection latency with our 50ms debounce. Alternatives considered:

- `pyobjc` FSEvents wrapper: More code for no advantage over `watchfiles`
- Raw kqueue: Lower-level than needed; FSEvents is better suited for file watching
- SQLite WAL polling: Introduces unnecessary polling latency

**Implementation (jarvis/watcher.py):**

```python
import asyncio
import sqlite3
from watchfiles import awatch
from pathlib import Path

CHAT_DB = Path.home() / "Library/Messages/chat.db"

class ChatDBWatcher:
    def __init__(self, socket_server: JarvisSocketServer):
        self.socket_server = socket_server
        self.last_message_id = self._get_last_message_id()

    def _get_last_message_id(self) -> int:
        conn = sqlite3.connect(f"file:{CHAT_DB}?mode=ro", uri=True)
        row = conn.execute("SELECT MAX(ROWID) FROM message").fetchone()
        conn.close()
        return row[0] or 0

    async def watch(self):
        async for changes in awatch(CHAT_DB):
            await self._check_new_messages()

    async def _check_new_messages(self):
        conn = sqlite3.connect(f"file:{CHAT_DB}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row

        # Get new messages since last check
        rows = conn.execute("""
            SELECT m.ROWID as id, m.text, m.chat_id, m.is_from_me,
                   h.id as sender
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            WHERE m.ROWID > ?
            ORDER BY m.ROWID
        """, (self.last_message_id,)).fetchall()

        conn.close()

        for row in rows:
            self.last_message_id = row["id"]

            # Compute embedding in background
            if row["text"]:
                asyncio.create_task(self._index_message(row))

            # Push notification to Tauri immediately
            await self.socket_server.broadcast("new_message", {
                "message_id": row["id"],
                "chat_id": row["chat_id"],
                "is_from_me": bool(row["is_from_me"]),
                "sender": row["sender"],
                "text_preview": (row["text"] or "")[:100]
            })

    async def _index_message(self, row):
        """Compute and store embedding for new message."""
        from models.bert_embedder import get_embedder

        embedder = get_embedder()
        embedding = await asyncio.to_thread(
            embedder.encode, row["text"]
        )
        # Store via vec_search for sqlite-vec indexing
```

---

### 4. Updated Frontend Store

**desktop/src/lib/stores/conversations.ts:**

```typescript
import { writable } from 'svelte/store';
import { db } from '../db/direct';
import { jarvis } from '../socket/client';

export const conversationsStore = writable<ConversationsState>(initialState);

// Initialize: direct reads + socket connection
export async function initialize() {
  await db.init();
  await jarvis.connect();

  // Load conversations directly from SQLite
  const conversations = await db.getConversations();
  conversationsStore.update((s) => ({ ...s, conversations }));

  // Listen for push notifications
  jarvis.on('new_message', async (data) => {
    const { chat_id, message_id } = data;

    // Read full message directly from chat.db
    const message = await db.getMessage(message_id);

    // Update store
    conversationsStore.update((s) => {
      if (s.selectedChatId === chat_id) {
        return { ...s, messages: [...s.messages, message] };
      }
      // Mark as having new messages
      const newSet = new Set(s.conversationsWithNewMessages);
      newSet.add(chat_id);
      return { ...s, conversationsWithNewMessages: newSet };
    });
  });
}

// Select conversation - direct read, no HTTP
export async function selectConversation(chatId: string) {
  conversationsStore.update((s) => ({
    ...s,
    selectedChatId: chatId,
    loadingMessages: true,
  }));

  // Direct SQLite read - ~1-5ms
  const messages = await db.getMessages(chatId, 50);

  conversationsStore.update((s) => ({
    ...s,
    messages,
    loadingMessages: false,
  }));
}

// Generate draft - needs LLM, use socket
export async function generateDraft(chatId: string): Promise<string[]> {
  const result = await jarvis.call<{ drafts: string[] }>('generate_draft', {
    chat_id: chatId,
  });
  return result.drafts;
}

// Semantic search - needs embeddings, use socket
export async function semanticSearch(query: string, chatId?: string) {
  return jarvis.call('semantic_search', { query, chat_id: chatId });
}
```

---

## Repository Modernization (Phases 1-3)

The codebase underwent a 3-phase modernization effort to reduce complexity:

### Phase 1: Foundation Cleanup

- Removed 100+ experimental scripts and result files from root
- Pruned obsolete model artifacts (category_svm_v2, old LightGBM variants)
- Consolidated scripts directory from 116 to ~25 production scripts
- Removed unused dependencies from pyproject.toml
- Stabilized test suite (0 failures baseline)

### Phase 2: Structural Reorganization

- Established `contracts/` for Protocol-based interfaces (Classifier, Embedder, Generator)
- Decomposed large modules:
  - `socket_server.py` remains single file but with clear handler sections
  - Lifecycle management extracted where applicable
- Enforced clean architecture layers (core -> interfaces -> infrastructure)
- Standardized error responses across API routers

### Phase 3: Pipeline Simplification

- Unified model access through registry pattern
- Simplified feature extraction pipeline
- Consolidated cache implementations into `jarvis/cache.py` (TTLCache)
- Removed dead prefetch paths and redundant warming logic

### Phase 4: Stabilization & Hardoff

- Performance baseline tests (`tests/test_performance_baselines.py`)
- Coverage threshold enforcement (`--cov-fail-under=60`)
- Security hardening: rate limiting, path validation, timing-safe token comparison
- Documentation updates (SECURITY.md, TROUBLESHOOTING.md)

### Phase 4: V4 Fact Extraction & Contact Profiling (2026-02-13)

- **Turn-Based Extraction**: Switched from segment-based to Turn-Based grouping. Consecutive messages from the same sender are combined into single turns, providing coherent context for the LLM.
- **Dynamic Identity Anchor**: Automatically resolves the user's name (e.g., "Jwalin Shah") from the Address Book, eliminating hardcoded identity references.
- **AddressBook Integration**: Generalizable name resolution for all contacts (including group chats) by resolving participant numbers against macOS AddressBook.
- **Targeted NLI Verification**: Each candidate fact is verified against the specific source message turn it originated from, using full-sentence hypotheses for maximum accuracy. (Note: Restored 2-pass LLM verification for superior casual chat handling).
- **Enriched Contact Profiles**: `ContactProfile` now stores `extracted_facts` and `relationship_reasoning` (LLM-derived justification for relationship labels).

### Phase 5: Pipeline Optimization & Resilience (2026-02-14)

- **Streaming Delta Ingestion**: Replaced bulk message loading with a memory-efficient per-chat streaming approach. Uses end-time checkpoints to fetch only new messages (deltas), making updates near-instant.
- **Automatic Preference Population**: Integrated computation of `contact_style_targets` and `contact_timing_prefs` (including quiet hours and optimal weekdays) directly into the ingestion loop.
- **Resource Management (`ModelManager`)**: Centralized model lifecycle management to coordinate transitions between Embedding and LLM phases within strict memory constraints.
- **Observability (`PipelineMonitor`)**: Real-time tracking of throughput, success/failure rates, and rejection metrics across all ingestion stages.
- **Database Resilience (`SafeChatReader`)**: Implemented exponential backoff and retries for SQLite "Database Locked" errors using a specialized `sqlite_retry` decorator.
- **Semantic Deduplication**: Added `FactDeduplicator` using BERT embeddings to prevent redundant facts from cluttering the Knowledge Graph.

---

## Performance Comparison

| Operation                | V1 (HTTP)           | V2 (Direct + Socket) |
| ------------------------ | ------------------- | -------------------- |
| Load conversations       | ~100-150ms          | ~1-5ms               |
| Load messages            | ~100-150ms          | ~1-5ms               |
| New message notification | Up to 10s (polling) | Instant (push)       |
| Generate draft           | ~50ms + inference   | ~1-5ms + inference   |
| Semantic search          | ~50ms + compute     | ~1-5ms + compute     |
| Idle resource usage      | Polling requests    | Zero                 |

---

## Implementation Status

### Phase 1: Direct SQLite Reads ✅ COMPLETE

1. ✅ Install `tauri-plugin-sql` - Added to `Cargo.toml` and `package.json`
2. ✅ Create `db/direct.ts` with query functions - `desktop/src/lib/db/direct.ts`
3. ✅ Create `db/queries.ts` with SQL queries - Ported from `integrations/imessage/queries.py`
4. ✅ Update stores to use direct reads - `desktop/src/lib/stores/conversations.ts`
5. ✅ HTTP API fallback when SQLite unavailable

### Phase 2: Unix Socket Server ✅ COMPLETE

1. ✅ Create `jarvis/socket_server.py` - JSON-RPC over `~/.jarvis/jarvis.sock`
2. ✅ Create Rust socket bridge - `desktop/src-tauri/src/socket.rs`
3. ✅ Create TypeScript client - `desktop/src/lib/socket/client.ts`
4. ✅ Update `scripts/launch.sh` to start socket server
5. ✅ Model preloading at startup for faster first request

### Phase 3: File Watcher + Push ✅ COMPLETE

1. ✅ Create `jarvis/watcher.py` - Watches chat.db for changes
2. ✅ Integrate watcher with socket server
3. ✅ Push `new_message` notifications to clients
4. ✅ Dynamic polling intervals (longer when socket connected)

### Phase 4: Deprecate HTTP for Tauri ✅ COMPLETE

1. ✅ HTTP API remains for CLI and other clients
2. ✅ Tauri uses direct reads + socket (fallback to HTTP when needed)
3. ✅ Key operations migrated to socket/direct DB with HTTP fallback:
   - `ping()` / `getHealth()` - Socket preferred (~50x faster)
   - `getConversations()` / `getMessages()` - Direct SQLite preferred (~30-50x faster)
   - `getDraftReplies()` - Socket preferred (LLM via Unix socket)
   - `getSummary()` - Socket preferred (LLM via Unix socket)
   - `semanticSearch()` - Socket preferred (~30-50x faster)
   - `getSmartReplySuggestions()` - Socket preferred (LLM via Unix socket)

---

## File Changes Summary

**New Files (Created):**

```
jarvis/interfaces/desktop/server.py  # Unix socket JSON-RPC server with model preloading
jarvis/watcher.py                 # chat.db file watcher for real-time notifications
desktop/src/lib/db/direct.ts      # Direct SQLite access layer
desktop/src/lib/db/queries.ts     # SQL queries ported from Python
desktop/src/lib/db/index.ts       # DB module exports
desktop/src/lib/socket/client.ts  # TypeScript socket client with auto-reconnect
desktop/src/lib/socket/index.ts   # Socket module exports
desktop/src-tauri/src/socket.rs   # Rust Unix socket bridge
tests/integration/test_socket_server.py  # Socket server tests
tests/unit/test_watcher.py        # File watcher tests (8 tests)
```

**Modified Files:**

```
desktop/src-tauri/Cargo.toml           # Added tauri-plugin-sql, tokio
desktop/src-tauri/src/lib.rs           # Registered SQL plugin and socket commands
desktop/src/lib/stores/conversations.ts # Uses direct reads + socket with HTTP fallback
desktop/src/lib/stores/health.ts       # Phase 4: Uses socket ping with HTTP fallback
desktop/src/lib/api/client.ts          # Phase 4: Key methods use socket/direct DB
desktop/package.json                    # Added @tauri-apps/plugin-sql
scripts/launch.sh                       # Starts socket server alongside API
```

**Unchanged (kept for CLI/other clients):**

```
api/main.py                  # HTTP API still works
api/routers/*.py             # Still functional for CLI, web clients
```

---

## Performance Optimizations

This section tracks performance improvements beyond the core V2 architecture changes.

### Phase 1: Backend Quick Wins (Pre-V2) ✅ COMPLETE

These optimizations are independent of V2 and improve base performance:

| Optimization               | File                          | Status  | Impact                                         |
| -------------------------- | ----------------------------- | ------- | ---------------------------------------------- |
| Vectorized semantic search | `jarvis/search/vec_search.py` | ✅ Done | 10-50x faster for large searches               |
| Scale thread pool          | `jarvis/router.py`            | ✅ Done | `max_workers=min(4, cpu_count)` for multi-core |
| Batch message indexing     | `jarvis/search/vec_search.py` | ✅ Done | Uses `executemany` + batch encoding            |
| Embedding result cache     | `jarvis/embedding_adapter.py` | ✅ Done | Increased LRU cache to 1000 entries            |
| Intent classification      | `jarvis/router.py`            | ✅ Done | Streamlined to single intent classifier        |
| Stop words optimization    | `jarvis/text_normalizer.py`   | ✅ Done | Module-level frozenset for O(1) lookup         |
| Quick reply O(1) lookup    | `jarvis/router.py`            | ✅ Done | Dict lookup instead of linear search           |

**Details:**

1. **Vectorized Semantic Search** - Replaced per-message similarity loop with single NumPy matrix operation:

   ```python
   # Before: O(n) Python loop
   for msg in messages:
       similarity = np.dot(query, msg.embedding)

   # After: Single vectorized operation
   similarities = np.dot(embedding_matrix, query)  # All at once
   ```

2. **Thread Pool Scaling** - Dynamic worker count based on CPU:

   ```python
   ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 2))
   ```

3. **Embedding Cache** - Singleton `CachedEmbedder` with 1000-entry LRU cache keyed by text hash.

---

### Phase 2: Frontend Optimizations (During V2)

Apply these when V2 rewrites the relevant Svelte components:

| Optimization       | Files                                           | Status | Description                                     |
| ------------------ | ----------------------------------------------- | ------ | ----------------------------------------------- |
| Reduce animations  | `MessageView.svelte`, `ConversationList.svelte` | ⏳ V2  | Remove decorative animations, keep functional   |
| Optimistic sending | `stores/conversations.ts`                       | ⏳ V2  | Show message instantly with "sending" state     |
| Skeleton states    | `ConversationList.svelte`, `MessageView.svelte` | ⏳ V2  | Placeholder skeletons during loading            |
| Granular stores    | `stores/conversations.ts`                       | ⏳ V2  | Split into focused stores for fewer re-renders  |
| Instant app shell  | `App.svelte`                                    | ⏳ V2  | Render structure immediately, load data into it |

**Animation Strategy:**
| Animation Type | Action | Reason |
|---------------|--------|--------|
| Loading spinners | Keep | Shows activity |
| Progress bars | Keep | Shows progress |
| New message highlight | Reduce to 1s | Draw attention without delay |
| Skeleton loaders | Add | Perceived speed |
| bounceIn on buttons | Remove | Decorative delay |
| 0.15s hover transitions | Reduce to 0.05s | Snappier feel |
| avatarPulse | Remove | Distracting |

---

### Phase 3: Advanced Optimizations (After V2)

These require V2's socket infrastructure:

| Optimization            | File                      | Status     | Description                                                               |
| ----------------------- | ------------------------- | ---------- | ------------------------------------------------------------------------- |
| Speculative prefetching | `jarvis/prefetch/`        | ✅ Done    | ML predictor + 3-tier cache (L1/L2/L3) + executor with resource awareness |
| Pre-warm common queries | `jarvis/socket_server.py` | ✅ Done    | Model preloading at startup via `model_warmer.py`                         |
| Message virtualization  | `MessageView.svelte`      | ⏳ Post-V2 | Cache heights in localStorage, reduce buffer                              |
| WebSocket multiplexing  | `socket/client.ts`        | ⏳ Post-V2 | Multiple streams over one connection                                      |

**Speculative Prefetching Example:**

```typescript
// When user hovers on a conversation for 200ms, prefetch messages
function prefetchOnHover(chatId: string) {
  if (!messageCache.has(chatId)) {
    requestIdleCallback(() => fetchMessages(chatId));
  }
}
```

**Pre-warm Example:**

```python
async def warmup():
    await get_conversations(limit=20)  # Likely first request
    get_embedder()  # Pre-load model
    await get_top_contacts(limit=10)
```

---

### Skipped Optimizations

| Optimization         | Reason                            |
| -------------------- | --------------------------------- |
| HTTP caching headers | V2 removes HTTP for Tauri clients |

---

### Performance Targets

| Metric                         | Current                | Target               |
| ------------------------------ | ---------------------- | -------------------- |
| Cold start time                | ~4-5s                  | <2s to interactive   |
| Conversation switch            | ~1-5ms (direct SQLite) | <100ms perceived ✅  |
| Message send feedback          | ~200ms                 | Instant (optimistic) |
| Search response (10k messages) | ~800ms                 | <500ms               |

### Verification

After each optimization:

1. `make test` - Ensure no regressions
2. Manual test - Measure perceived speed improvement
3. Profile if needed - `py-spy` for Python, Chrome DevTools for frontend
