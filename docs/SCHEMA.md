# JARVIS Database Schema

This document describes the database schemas used by JARVIS.

## iMessage Database (Read-Only)

JARVIS reads from the macOS iMessage database at `~/Library/Messages/chat.db`. This is a SQLite database managed by macOS.

### Key Tables

#### `message`
Primary message storage.

| Column | Type | Description |
|--------|------|-------------|
| ROWID | INTEGER | Primary key |
| guid | TEXT | Unique message identifier |
| text | TEXT | Message content |
| handle_id | INTEGER | FK to handle table |
| service | TEXT | "iMessage" or "SMS" |
| date | INTEGER | Timestamp (nanoseconds since 2001-01-01) |
| is_from_me | INTEGER | 1 if sent, 0 if received |
| is_read | INTEGER | Read status |
| is_delivered | INTEGER | Delivery status |
| is_sent | INTEGER | Send status |
| cache_has_attachments | INTEGER | Has attachments flag |
| associated_message_guid | TEXT | For reactions/replies |
| associated_message_type | INTEGER | Reaction type |

#### `chat`
Conversation containers.

| Column | Type | Description |
|--------|------|-------------|
| ROWID | INTEGER | Primary key |
| guid | TEXT | Unique chat identifier |
| style | INTEGER | 43=group, 45=1:1 |
| chat_identifier | TEXT | Phone/email or group ID |
| display_name | TEXT | Group name (if set) |

#### `handle`
Contact identifiers.

| Column | Type | Description |
|--------|------|-------------|
| ROWID | INTEGER | Primary key |
| id | TEXT | Phone number or email |
| service | TEXT | "iMessage" or "SMS" |

#### `chat_message_join`
Links chats to messages.

| Column | Type | Description |
|--------|------|-------------|
| chat_id | INTEGER | FK to chat |
| message_id | INTEGER | FK to message |

#### `attachment`
File attachments.

| Column | Type | Description |
|--------|------|-------------|
| ROWID | INTEGER | Primary key |
| guid | TEXT | Unique identifier |
| filename | TEXT | File path |
| mime_type | TEXT | MIME type |
| transfer_name | TEXT | Display name |

### Common Queries

Get messages for a chat:
```sql
SELECT m.* FROM message m
JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
WHERE cmj.chat_id = ?
ORDER BY m.date DESC
LIMIT 50;
```

Get recent conversations:
```sql
SELECT c.*, MAX(m.date) as last_date FROM chat c
JOIN chat_message_join cmj ON c.ROWID = cmj.chat_id
JOIN message m ON cmj.message_id = m.ROWID
GROUP BY c.ROWID
ORDER BY last_date DESC
LIMIT 50;
```

### Date Conversion

iMessage dates are stored as nanoseconds since January 1, 2001:
```python
from datetime import datetime, timezone

def imessage_to_datetime(imessage_date: int) -> datetime:
    # Reference: 2001-01-01 00:00:00 UTC
    APPLE_EPOCH = 978307200
    unix_timestamp = (imessage_date / 1_000_000_000) + APPLE_EPOCH
    return datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)
```

## JARVIS Internal Storage

JARVIS uses several internal databases for caching and state. Primary storage is located in `~/.jarvis/`.

### Vector Search (sqlite-vec)

JARVIS uses the `sqlite-vec` extension for high-performance vector search directly within SQLite.

**Location**: `~/.jarvis/jarvis.db`

**Tables**:
- `vec_chunks`: `int8[384]` quantized embeddings with metadata for per-contact partitioned search.
- `vec_binary`: `bit[384]` sign-bit quantized embeddings for fast cross-contact hamming scans.

| Column | Type | Description |
|--------|------|-------------|
| rowid | INTEGER | Primary key |
| embedding | BLOB | Quantized vector |
| contact_id | INTEGER | Partition key for fast per-contact search |
| chat_id | TEXT | Source conversation ID |
| topic_label | TEXT | Detected topic or summary |
| trigger_text | TEXT | Last trigger in the chunk |
| response_text | TEXT | Last response in the chunk |

### JARVIS Primary Database

**Location**: `~/.jarvis/jarvis.db`

#### `contacts`
Stores contact relationship metadata and handle mappings.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| chat_id | TEXT | Primary iMessage chat_id |
| display_name | TEXT | Resolved name |
| relationship | TEXT | sister, coworker, boss, etc. |
| handles_json | TEXT | JSON list of associated handles |

#### `scheduled_drafts`
Stores automated and scheduled messages (formerly `scheduler.json`).

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT | UUID primary key |
| chat_id | TEXT | Target conversation |
| message_text | TEXT | Content to send |
| send_at | TIMESTAMP | Scheduled execution time |
| status | TEXT | pending, queued, sent, failed |

#### `relationship_profiles`
Aggregated communication patterns (cached in model-specific `embeddings.db`).

| Column | Type | Description |
|--------|------|-------------|
| contact_id | TEXT | Primary key |
| display_name | TEXT | Name |
| profile_data | TEXT | JSON blob of style, tone, and topics |

### Quality Metrics (In-Memory)

Quality tracking data is kept in memory during runtime and optionally persisted to `jarvis.db`.

**Location**: Memory-only (resets on restart)

Tracked events:
- Response generation (template vs model)
- HHEM scores
- User acceptance/rejection
- Edit distance

## Data Flow

```
iMessage DB (read-only)
        │
        ▼
    Message Reader
        │
        ├──▶ sqlite-vec Index (semantic search)
        │
        ├──▶ Contact Profiler (style analysis)
        │
        └──▶ Quality Metrics (tracking)
```

## Migration Notes

### Schema Changes

JARVIS does not modify the iMessage database. Internal schemas (in `jarvis.db`) are versioned and migrated automatically.

When upgrading:
1. Internal caches in `~/.jarvis/embeddings/` can be safely deleted.
2. Vector indexes will be rebuilt automatically using `sqlite-vec`.
3. Scheduler state in `jarvis.db` is migrated automatically.

### Backup

Only `~/.jarvis/` needs backup, specifically:
- `jarvis.db` - Contacts, pairs, and scheduled messages.
- `config.json` - User configuration.

Artifacts in `~/.jarvis/embeddings/` can be regenerated from `chat.db`.
