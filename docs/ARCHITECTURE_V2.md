# JARVIS Architecture V2: Direct SQLite + Unix Sockets

## Overview

This document describes the optimized architecture for JARVIS, replacing HTTP polling with direct SQLite reads and Unix socket communication.

**Status:** Phases 1-3 Complete, Phase 4 In Progress

## Quick Start

```bash
# Launch includes socket server automatically
./scripts/launch.sh
```

## Socket Server API

The socket server runs at `/tmp/jarvis.sock` with JSON-RPC 2.0 protocol.

**Available Methods:**
| Method | Description | Parameters |
|--------|-------------|------------|
| `ping` | Health check | None |
| `generate_draft` | Generate reply suggestions | `chat_id`, `instruction?`, `num_suggestions?` |
| `summarize` | Summarize conversation | `chat_id`, `num_messages?` |
| `get_smart_replies` | Quick reply suggestions | `last_message`, `num_suggestions?` |
| `semantic_search` | Search messages | `query`, `limit?`, `threshold?`, `filters?` |
| `classify_intent` | Classify message intent | `text` |

**Push Notifications:**
| Event | Description | Data |
|-------|-------------|------|
| `new_message` | New message received | `message_id`, `chat_id`, `sender`, `text`, `date`, `is_from_me` |

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
       │   chat.db    │              │  /tmp/jarvis.sock   │
       │  (read-only) │              │  (Python daemon)    │
       └──────────────┘              └──────────┬──────────┘
       ┌──────────────┐                         │
       │  jarvis.db   │◄────────────────────────┤
       │  (our data)  │                         │
       └──────────────┘              ┌──────────┴──────────┐
                                     │  - LLM inference    │
                                     │  - Embeddings       │
                                     │  - Classification   │
                                     │  - Semantic search  │
                                     │  - File watcher     │
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
| jarvis.db | ~/.jarvis/jarvis.db | Read-write | Embeddings, contacts, cached analysis |

**Implementation:**
```typescript
// desktop/src/lib/db/direct.ts
import Database from '@tauri-apps/plugin-sql';

let chatDb: Database | null = null;
let jarvisDb: Database | null = null;

export async function initDatabases() {
  const homeDir = await homeDir();

  // Apple's iMessage database (read-only)
  chatDb = await Database.load(
    `sqlite:${homeDir}/Library/Messages/chat.db?mode=ro`
  );

  // Our database
  jarvisDb = await Database.load(
    `sqlite:${homeDir}/.jarvis/jarvis.db`
  );
}

export async function getMessages(chatId: string, limit = 50): Promise<Message[]> {
  // Direct read - ~1-5ms
  return chatDb.select(`
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
  `, [chatId, limit]);
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

export async function getEmbedding(messageId: number): Promise<Float32Array | null> {
  const row = await jarvisDb.select(
    'SELECT embedding FROM message_embeddings WHERE message_id = ?',
    [messageId]
  );
  return row[0]?.embedding;
}
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

**Socket Location:** `/tmp/jarvis.sock`

**Protocol:** JSON-RPC 2.0

```
Request:  {"jsonrpc": "2.0", "method": "generate_draft", "params": {...}, "id": 1}
Response: {"jsonrpc": "2.0", "result": {...}, "id": 1}
Push:     {"jsonrpc": "2.0", "method": "new_message", "params": {...}}
```

**Python Server (jarvis/socket_server.py):**
```python
import asyncio
import json
import os
from pathlib import Path

SOCKET_PATH = "/tmp/jarvis.sock"

class JarvisSocketServer:
    def __init__(self):
        self.clients: set[asyncio.StreamWriter] = set()
        self.handlers = {
            "generate_draft": self.handle_generate_draft,
            "semantic_search": self.handle_semantic_search,
            "classify_intent": self.handle_classify_intent,
            "get_smart_replies": self.handle_smart_replies,
        }

    async def start(self):
        # Clean up stale socket
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)

        server = await asyncio.start_unix_server(
            self.handle_client,
            path=SOCKET_PATH
        )

        # Set permissions (owner only)
        os.chmod(SOCKET_PATH, 0o600)

        async with server:
            await server.serve_forever()

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.clients.add(writer)
        try:
            while True:
                # Read length-prefixed message
                length_bytes = await reader.readexactly(4)
                length = int.from_bytes(length_bytes, 'big')
                data = await reader.readexactly(length)

                request = json.loads(data.decode())
                response = await self.handle_request(request)

                if response:  # Don't respond to notifications
                    await self.send(writer, response)
        except asyncio.IncompleteReadError:
            pass  # Client disconnected
        finally:
            self.clients.discard(writer)
            writer.close()

    async def handle_request(self, request: dict) -> dict | None:
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        handler = self.handlers.get(method)
        if not handler:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Method not found: {method}"},
                "id": request_id
            }

        try:
            result = await handler(params)
            return {"jsonrpc": "2.0", "result": result, "id": request_id}
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": str(e)},
                "id": request_id
            }

    async def broadcast(self, method: str, params: dict):
        """Push notification to all connected clients."""
        message = {"jsonrpc": "2.0", "method": method, "params": params}
        for writer in self.clients:
            try:
                await self.send(writer, message)
            except:
                self.clients.discard(writer)

    async def send(self, writer: asyncio.StreamWriter, data: dict):
        encoded = json.dumps(data).encode()
        length = len(encoded).to_bytes(4, 'big')
        writer.write(length + encoded)
        await writer.drain()

    # --- Handlers ---

    async def handle_generate_draft(self, params: dict) -> dict:
        from jarvis.router import generate_response
        chat_id = params["chat_id"]
        result = await asyncio.to_thread(generate_response, chat_id)
        return {"drafts": result}

    async def handle_semantic_search(self, params: dict) -> dict:
        from jarvis.semantic_search import search
        results = await asyncio.to_thread(
            search,
            query=params["query"],
            chat_id=params.get("chat_id"),
            limit=params.get("limit", 20)
        )
        return {"results": results}

    async def handle_classify_intent(self, params: dict) -> dict:
        from jarvis.intent import classify
        result = await asyncio.to_thread(classify, params["text"])
        return {"intent": result}

    async def handle_smart_replies(self, params: dict) -> dict:
        from jarvis.router import get_smart_replies
        result = await asyncio.to_thread(
            get_smart_replies,
            chat_id=params["chat_id"]
        )
        return {"replies": result}
```

**TypeScript Client (desktop/src/lib/socket/client.ts):**
```typescript
import { invoke } from '@tauri-apps/api/core';

type JsonRpcRequest = {
  jsonrpc: "2.0";
  method: string;
  params?: Record<string, unknown>;
  id?: number;
};

type JsonRpcResponse = {
  jsonrpc: "2.0";
  result?: unknown;
  error?: { code: number; message: string };
  id?: number;
};

type PushHandler = (params: Record<string, unknown>) => void;

class JarvisSocket {
  private requestId = 0;
  private pendingRequests = new Map<number, {
    resolve: (value: unknown) => void;
    reject: (error: Error) => void;
  }>();
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
      jsonrpc: "2.0",
      method,
      params,
      id
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
        from jarvis.embedding_adapter import get_embedder
        from jarvis.embeddings import EmbeddingStore

        embedder = get_embedder()
        store = EmbeddingStore()

        embedding = await asyncio.to_thread(
            embedder.encode, row["text"]
        )

        await asyncio.to_thread(
            store.add_embedding,
            message_id=row["id"],
            chat_id=row["chat_id"],
            embedding=embedding,
            text_preview=row["text"][:200]
        )
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
  conversationsStore.update(s => ({ ...s, conversations }));

  // Listen for push notifications
  jarvis.on('new_message', async (data) => {
    const { chat_id, message_id } = data;

    // Read full message directly from chat.db
    const message = await db.getMessage(message_id);

    // Update store
    conversationsStore.update(s => {
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
  conversationsStore.update(s => ({
    ...s,
    selectedChatId: chatId,
    loadingMessages: true
  }));

  // Direct SQLite read - ~1-5ms
  const messages = await db.getMessages(chatId, 50);

  conversationsStore.update(s => ({
    ...s,
    messages,
    loadingMessages: false
  }));
}

// Generate draft - needs LLM, use socket
export async function generateDraft(chatId: string): Promise<string[]> {
  const result = await jarvis.call<{ drafts: string[] }>('generate_draft', {
    chat_id: chatId
  });
  return result.drafts;
}

// Semantic search - needs embeddings, use socket
export async function semanticSearch(query: string, chatId?: string) {
  return jarvis.call('semantic_search', { query, chat_id: chatId });
}
```

---

## Performance Comparison

| Operation | V1 (HTTP) | V2 (Direct + Socket) |
|-----------|-----------|----------------------|
| Load conversations | ~100-150ms | ~1-5ms |
| Load messages | ~100-150ms | ~1-5ms |
| New message notification | Up to 10s (polling) | Instant (push) |
| Generate draft | ~50ms + inference | ~1-5ms + inference |
| Semantic search | ~50ms + compute | ~1-5ms + compute |
| Idle resource usage | Polling requests | Zero |

---

## Implementation Status

### Phase 1: Direct SQLite Reads ✅ COMPLETE
1. ✅ Install `tauri-plugin-sql` - Added to `Cargo.toml` and `package.json`
2. ✅ Create `db/direct.ts` with query functions - `desktop/src/lib/db/direct.ts`
3. ✅ Create `db/queries.ts` with SQL queries - Ported from `integrations/imessage/queries.py`
4. ✅ Update stores to use direct reads - `desktop/src/lib/stores/conversations.ts`
5. ✅ HTTP API fallback when SQLite unavailable

### Phase 2: Unix Socket Server ✅ COMPLETE
1. ✅ Create `jarvis/socket_server.py` - JSON-RPC over `/tmp/jarvis.sock`
2. ✅ Create Rust socket bridge - `desktop/src-tauri/src/socket.rs`
3. ✅ Create TypeScript client - `desktop/src/lib/socket/client.ts`
4. ✅ Update `scripts/launch.sh` to start socket server
5. ✅ Model preloading at startup for faster first request

### Phase 3: File Watcher + Push ✅ COMPLETE
1. ✅ Create `jarvis/watcher.py` - Watches chat.db for changes
2. ✅ Integrate watcher with socket server
3. ✅ Push `new_message` notifications to clients
4. ✅ Dynamic polling intervals (longer when socket connected)

### Phase 4: Deprecate HTTP for Tauri ⏳ IN PROGRESS
1. ✅ HTTP API remains for CLI and other clients
2. ⏳ Tauri uses direct reads + socket (fallback to HTTP still enabled)
3. ⏳ Remove HTTP calls from Tauri frontend (gradual migration)

---

## File Changes Summary

**New Files (Created):**
```
jarvis/socket_server.py           # Unix socket JSON-RPC server with model preloading
jarvis/watcher.py                 # chat.db file watcher for real-time notifications
desktop/src/lib/db/direct.ts      # Direct SQLite access layer
desktop/src/lib/db/queries.ts     # SQL queries ported from Python
desktop/src/lib/db/index.ts       # DB module exports
desktop/src/lib/socket/client.ts  # TypeScript socket client with auto-reconnect
desktop/src/lib/socket/index.ts   # Socket module exports
desktop/src-tauri/src/socket.rs   # Rust Unix socket bridge
tests/unit/test_socket_server.py  # Socket server tests (17 tests)
tests/unit/test_watcher.py        # File watcher tests (8 tests)
```

**Modified Files:**
```
desktop/src-tauri/Cargo.toml           # Added tauri-plugin-sql, tokio
desktop/src-tauri/src/lib.rs           # Registered SQL plugin and socket commands
desktop/src/lib/stores/conversations.ts # Uses direct reads + socket with HTTP fallback
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

| Optimization | File | Status | Impact |
|-------------|------|--------|--------|
| Vectorized semantic search | `jarvis/semantic_search.py`, `jarvis/embeddings.py` | ✅ Done | 10-50x faster for large searches |
| Scale thread pool | `jarvis/router.py` | ✅ Done | `max_workers=min(4, cpu_count)` for multi-core |
| Batch message indexing | `jarvis/embeddings.py` | ✅ Already had | Uses `executemany` + batch encoding |
| Embedding result cache | `jarvis/embedding_adapter.py` | ✅ Done | Increased LRU cache to 1000 entries |
| Intent classification | `jarvis/router.py` | ✅ Done | Streamlined to single intent classifier |
| Profile caching | `jarvis/embeddings.py` | ✅ Done | 5-min TTL cache for relationship profiles |
| Stop words optimization | `jarvis/embeddings.py` | ✅ Done | Module-level frozenset for O(1) lookup |
| Quick reply O(1) lookup | `jarvis/router.py` | ✅ Done | Dict lookup instead of linear search |

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

| Optimization | Files | Status | Description |
|-------------|-------|--------|-------------|
| Reduce animations | `MessageView.svelte`, `ConversationList.svelte` | ⏳ V2 | Remove decorative animations, keep functional |
| Optimistic sending | `stores/conversations.ts` | ⏳ V2 | Show message instantly with "sending" state |
| Skeleton states | `ConversationList.svelte`, `MessageView.svelte` | ⏳ V2 | Placeholder skeletons during loading |
| Granular stores | `stores/conversations.ts` | ⏳ V2 | Split into focused stores for fewer re-renders |
| Instant app shell | `App.svelte` | ⏳ V2 | Render structure immediately, load data into it |

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

| Optimization | File | Status | Description |
|-------------|------|--------|-------------|
| Speculative prefetching | `stores/conversations.ts` | ⏳ Post-V2 | Prefetch likely-next conversations on hover |
| Pre-warm common queries | `jarvis/socket_server.py` | ⏳ Post-V2 | Cache common queries on startup |
| Message virtualization | `MessageView.svelte` | ⏳ Post-V2 | Cache heights in localStorage, reduce buffer |
| WebSocket multiplexing | `socket/client.ts` | ⏳ Post-V2 | Multiple streams over one connection |

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

| Optimization | Reason |
|-------------|--------|
| HTTP caching headers | V2 removes HTTP for Tauri clients |

---

### Performance Targets

| Metric | Current | Target |
|--------|---------|--------|
| Cold start time | ~4-5s | <2s to interactive |
| Conversation switch | ~100-150ms | <100ms perceived |
| Message send feedback | ~200ms | Instant (optimistic) |
| Search response (10k messages) | ~800ms | <500ms |

### Verification

After each optimization:
1. `make test` - Ensure no regressions
2. Manual test - Measure perceived speed improvement
3. Profile if needed - `py-spy` for Python, Chrome DevTools for frontend
