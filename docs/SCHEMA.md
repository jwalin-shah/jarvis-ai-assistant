# JARVIS Database Schema

> **Last Updated:** 2026-02-12

This document describes the database schemas used by JARVIS.

## iMessage Database (Read-Only)

JARVIS reads from the macOS iMessage database at `~/Library/Messages/chat.db`. This is a SQLite database managed by macOS.

### Key Tables

#### `message`

Primary message storage.

| Column                  | Type    | Description                              |
| ----------------------- | ------- | ---------------------------------------- |
| ROWID                   | INTEGER | Primary key                              |
| guid                    | TEXT    | Unique message identifier                |
| text                    | TEXT    | Message content                          |
| handle_id               | INTEGER | FK to handle table                       |
| service                 | TEXT    | "iMessage" or "SMS"                      |
| date                    | INTEGER | Timestamp (nanoseconds since 2001-01-01) |
| is_from_me              | INTEGER | 1 if sent, 0 if received                 |
| is_read                 | INTEGER | Read status                              |
| is_delivered            | INTEGER | Delivery status                          |
| is_sent                 | INTEGER | Send status                              |
| cache_has_attachments   | INTEGER | Has attachments flag                     |
| associated_message_guid | TEXT    | For reactions/replies                    |
| associated_message_type | INTEGER | Reaction type                            |

#### `chat`

Conversation containers.

| Column          | Type    | Description             |
| --------------- | ------- | ----------------------- |
| ROWID           | INTEGER | Primary key             |
| guid            | TEXT    | Unique chat identifier  |
| style           | INTEGER | 43=group, 45=1:1        |
| chat_identifier | TEXT    | Phone/email or group ID |
| display_name    | TEXT    | Group name (if set)     |

#### `handle`

Contact identifiers.

| Column  | Type    | Description           |
| ------- | ------- | --------------------- |
| ROWID   | INTEGER | Primary key           |
| id      | TEXT    | Phone number or email |
| service | TEXT    | "iMessage" or "SMS"   |

#### `chat_message_join`

Links chats to messages.

| Column     | Type    | Description   |
| ---------- | ------- | ------------- |
| chat_id    | INTEGER | FK to chat    |
| message_id | INTEGER | FK to message |

#### `attachment`

File attachments.

| Column        | Type    | Description       |
| ------------- | ------- | ----------------- |
| ROWID         | INTEGER | Primary key       |
| guid          | TEXT    | Unique identifier |
| filename      | TEXT    | File path         |
| mime_type     | TEXT    | MIME type         |
| transfer_name | TEXT    | Display name      |

### Common Queries

Get messages for a chat:

```sql
SELECT m.* FROM message m
JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
WHERE cmj.chat_id = ?
ORDER BY m.date DESC
LIMIT 50;
```

Get recent conversations (uses `ROW_NUMBER()` to guarantee one message per chat, avoiding duplicate-timestamp multiplication):

```sql
WITH chat_stats AS (
    SELECT cmj.chat_id,
           MAX(m.date) as last_date,
           COUNT(*) as msg_count
    FROM chat_message_join cmj
    JOIN message m ON cmj.message_id = m.ROWID
    GROUP BY cmj.chat_id
),
last_msgs AS (
    SELECT cmj.chat_id, m.text, m.is_from_me,
           ROW_NUMBER() OVER (PARTITION BY cmj.chat_id ORDER BY m.date DESC) as rn
    FROM chat_message_join cmj
    JOIN message m ON cmj.message_id = m.ROWID
)
SELECT c.*, cs.last_date, cs.msg_count, lm.text as last_message_text
FROM chat c
JOIN chat_stats cs ON c.ROWID = cs.chat_id
LEFT JOIN last_msgs lm ON c.ROWID = lm.chat_id AND lm.rn = 1
ORDER BY cs.last_date DESC
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

| Column        | Type    | Description                               |
| ------------- | ------- | ----------------------------------------- |
| rowid         | INTEGER | Primary key                               |
| embedding     | BLOB    | Quantized vector                          |
| contact_id    | INTEGER | Partition key for fast per-contact search |
| chat_id       | TEXT    | Source conversation ID                    |
| topic_label   | TEXT    | Detected topic or summary                 |
| trigger_text  | TEXT    | Last trigger in the chunk                 |
| response_text | TEXT    | Last response in the chunk                |

### JARVIS Primary Database

**Location**: `~/.jarvis/jarvis.db`
**Current Schema Version**: 20

#### `contacts`

Stores contact relationship metadata and handle mappings.

| Column                 | Type      | Description                                             |
| ---------------------- | --------- | ------------------------------------------------------- |
| id                     | INTEGER   | Primary key                                             |
| chat_id                | TEXT      | Primary iMessage chat_id (unique)                       |
| display_name           | TEXT      | Resolved name                                           |
| phone_or_email         | TEXT      | Primary contact method                                  |
| handles_json           | TEXT      | JSON array of associated handles                        |
| relationship           | TEXT      | sister, coworker, friend, boss                          |
| relationship_reasoning | TEXT      | LLM-derived justification for relationship label (v18+) |
| style_notes            | TEXT      | e.g. "casual, uses emojis"                              |
| last_extracted_rowid   | INTEGER   | iMessage ROWID of last extracted message (v18)          |
| last_extracted_at      | TIMESTAMP | when extraction was last run (v18)                      |
| created_at             | TIMESTAMP | Record creation time                                    |
| updated_at             | TIMESTAMP | Last update time                                        |

#### `contact_style_targets`

Computed style targets from a contact's message history.

| Column              | Type      | Description                      |
| ------------------- | --------- | -------------------------------- |
| contact_id          | INTEGER   | FK to contacts(id)               |
| median_reply_length | INTEGER   | Median word count                |
| punctuation_rate    | REAL      | Fraction with ending punctuation |
| emoji_rate          | REAL      | Fraction containing emojis       |
| greeting_rate       | REAL      | Fraction starting with greeting  |
| updated_at          | TIMESTAMP | Last update time                 |

#### `contact_timing_prefs`

Contact timing preferences for smart scheduling (v8+).

| Column                 | Type      | Description                            |
| ---------------------- | --------- | -------------------------------------- |
| contact_id             | INTEGER   | FK to contacts(id)                     |
| timezone               | TEXT      | Contact's timezone (IANA format)       |
| quiet_hours_json       | TEXT      | JSON with quiet hours config           |
| preferred_hours_json   | TEXT      | JSON array of preferred hours (0-23)   |
| optimal_weekdays_json  | TEXT      | JSON array of preferred weekdays (0-6) |
| avg_response_time_mins | REAL      | Average response time in minutes       |
| last_interaction       | TIMESTAMP | Last interaction timestamp             |
| updated_at             | TIMESTAMP | Last update time                       |

#### `fact_candidates_log` (v16+)

Keeps every extracted fact candidate for auditing and quality tracking.

| Column      | Type      | Description                 |
| ----------- | --------- | --------------------------- |
| id          | INTEGER   | Primary key                 |
| contact_id  | TEXT      | Contact ID                  |
| chat_id     | TEXT      | Chat ID                     |
| message_id  | INTEGER   | Source message ID           |
| subject     | TEXT      | Fact subject                |
| predicate   | TEXT      | Fact predicate              |
| value       | TEXT      | Fact value                  |
| category    | TEXT      | Fact category               |
| confidence  | REAL      | Extraction confidence       |
| source_text | TEXT      | Original message text       |
| attribution | TEXT      | Who fact is about           |
| segment_id  | INTEGER   | FK to conversation_segments |
| log_stage   | TEXT      | Stage where fact was logged |
| created_at  | TIMESTAMP | Logging time                |

#### `reply_logs` (v20+)

Full traceability for every AI-generated response.

| Column              | Type      | Description                |
| ------------------- | --------- | -------------------------- |
| id                  | INTEGER   | Primary key                |
| chat_id             | TEXT      | Chat ID                    |
| contact_id          | TEXT      | Contact ID                 |
| incoming_text       | TEXT      | Incoming user message      |
| classification_json | TEXT      | Category, urgency, etc.    |
| rag_context_json    | TEXT      | Retrieved document content |
| final_prompt        | TEXT      | Exact prompt sent to LLM   |
| response_text       | TEXT      | Generated response         |
| confidence          | REAL      | Generation confidence      |
| metadata_json       | TEXT      | Latency, model info, etc.  |
| created_at          | TIMESTAMP | Log time                   |

#### `pairs`

Extracted message pairs (trigger + response) from iMessage history.

| Column                | Type      | Description                                         |
| --------------------- | --------- | --------------------------------------------------- |
| id                    | INTEGER   | Primary key                                         |
| contact_id            | INTEGER   | FK to contacts(id)                                  |
| trigger_text          | TEXT      | What they said                                      |
| response_text         | TEXT      | What you said                                       |
| trigger_timestamp     | TIMESTAMP | Timestamp of trigger message                        |
| response_timestamp    | TIMESTAMP | Timestamp of response message                       |
| chat_id               | TEXT      | Source conversation                                 |
| trigger_msg_id        | INTEGER   | Primary trigger message ID                          |
| response_msg_id       | INTEGER   | Primary response message ID                         |
| trigger_msg_ids_json  | TEXT      | JSON array for multi-message triggers               |
| response_msg_ids_json | TEXT      | JSON array for multi-message responses              |
| context_text          | TEXT      | Previous messages before trigger                    |
| quality_score         | REAL      | 0.0-1.0 quality rating                              |
| flags_json            | TEXT      | JSON: {"attachment_only":true, ...}                 |
| is_group              | BOOLEAN   | True if from group chat                             |
| is_holdout            | BOOLEAN   | True if reserved for evaluation                     |
| gate_a_passed         | BOOLEAN   | Rule gate result (v6+)                              |
| gate_b_score          | REAL      | Embedding similarity score (v6+)                    |
| gate_c_verdict        | TEXT      | NLI verdict: accept/reject/uncertain (v6+)          |
| validity_status       | TEXT      | Final: valid/invalid/uncertain (v6+)                |
| trigger_da_type       | TEXT      | Dialogue act type, e.g. WH_QUESTION (v7+)           |
| trigger_da_conf       | REAL      | Dialogue act classifier confidence (v7+)            |
| response_da_type      | TEXT      | Response dialogue act type (v7+)                    |
| response_da_conf      | REAL      | Response dialogue act confidence (v7+)              |
| cluster_id            | INTEGER   | HDBSCAN cluster assignment (v7+)                    |
| usage_count           | INTEGER   | Times this pair was used for generation             |
| last_used_at          | TIMESTAMP | Last time pair was used                             |
| source_timestamp      | TIMESTAMP | Original message timestamp (for decay)              |
| content_hash          | TEXT      | MD5 of normalized trigger\|response for dedup (v9+) |

#### `pair_artifacts`

Heavy artifacts split from pairs to keep the main table lean.

| Column             | Type    | Description                           |
| ------------------ | ------- | ------------------------------------- |
| pair_id            | INTEGER | FK to pairs(id)                       |
| context_json       | TEXT    | Structured context window (JSON list) |
| gate_a_reason      | TEXT    | Why Gate A rejected (if rejected)     |
| gate_c_scores_json | TEXT    | Raw NLI scores (JSON dict)            |
| raw_trigger_text   | TEXT    | Original text before normalization    |
| raw_response_text  | TEXT    | Original text before normalization    |

#### `clusters`

Clustered intent groups for analytics.

| Column            | Type    | Description                         |
| ----------------- | ------- | ----------------------------------- |
| id                | INTEGER | Primary key                         |
| name              | TEXT    | e.g. INVITATION, GREETING, SCHEDULE |
| description       | TEXT    | Cluster description                 |
| example_triggers  | TEXT    | JSON array                          |
| example_responses | TEXT    | JSON array                          |

#### `pair_embeddings`

Links pairs to vector index positions. (`faiss_id` column name is legacy; now backed by sqlite-vec.)

| Column        | Type    | Description                            |
| ------------- | ------- | -------------------------------------- |
| pair_id       | INTEGER | FK to pairs(id)                        |
| faiss_id      | INTEGER | Position in vector index (legacy name) |
| cluster_id    | INTEGER | FK to clusters(id)                     |
| index_version | TEXT    | Which index version this belongs to    |

#### `index_versions`

Tracks vector index rebuilds for safe swaps.

| Column        | Type    | Description                      |
| ------------- | ------- | -------------------------------- |
| id            | INTEGER | Primary key                      |
| version_id    | TEXT    | e.g. "20240115-143022"           |
| model_name    | TEXT    | e.g. "BAAI/bge-small-en-v1.5"    |
| embedding_dim | INTEGER | e.g. 384                         |
| num_vectors   | INTEGER | Total vectors in index           |
| index_path    | TEXT    | Relative path to index file      |
| is_active     | BOOLEAN | Whether this is the active index |
| normalized    | BOOLEAN | Whether vectors are normalized   |

#### `scheduled_drafts`

Stores automated and scheduled messages (v8+).

| Column        | Type      | Description                                          |
| ------------- | --------- | ---------------------------------------------------- |
| id            | TEXT      | UUID primary key                                     |
| draft_id      | TEXT      | Reference to the draft                               |
| contact_id    | INTEGER   | FK to contacts(id)                                   |
| chat_id       | TEXT      | Target conversation                                  |
| message_text  | TEXT      | Content to send                                      |
| send_at       | TIMESTAMP | Scheduled execution time                             |
| priority      | TEXT      | urgent/normal/low                                    |
| status        | TEXT      | pending/queued/sending/sent/failed/cancelled/expired |
| timezone      | TEXT      | Contact's timezone (IANA format)                     |
| depends_on    | TEXT      | ID of item this depends on                           |
| retry_count   | INTEGER   | Number of retry attempts                             |
| max_retries   | INTEGER   | Maximum retries allowed                              |
| expires_at    | TIMESTAMP | When this schedule expires                           |
| result_json   | TEXT      | JSON with send result details                        |
| metadata_json | TEXT      | Additional metadata                                  |

#### `contact_timing_prefs`

Contact timing preferences for smart scheduling (v8+).

| Column                 | Type      | Description                            |
| ---------------------- | --------- | -------------------------------------- |
| contact_id             | INTEGER   | FK to contacts(id)                     |
| timezone               | TEXT      | Contact's timezone (IANA format)       |
| quiet_hours_json       | TEXT      | JSON with quiet hours config           |
| preferred_hours_json   | TEXT      | JSON array of preferred hours (0-23)   |
| optimal_weekdays_json  | TEXT      | JSON array of preferred weekdays (0-6) |
| avg_response_time_mins | REAL      | Average response time in minutes       |
| last_interaction       | TIMESTAMP | Last interaction timestamp             |

#### `send_queue`

Tracks message delivery status (v8+).

| Column             | Type      | Description                 |
| ------------------ | --------- | --------------------------- |
| id                 | TEXT      | UUID primary key            |
| scheduled_draft_id | TEXT      | FK to scheduled_drafts(id)  |
| status             | TEXT      | pending/sending/sent/failed |
| queued_at          | TIMESTAMP | When queued                 |
| sent_at            | TIMESTAMP | When actually sent          |
| error              | TEXT      | Error message if failed     |
| attempts           | INTEGER   | Number of send attempts     |
| next_retry_at      | TIMESTAMP | When to retry next          |

### Embeddings Database

**Location**: `~/.jarvis/embeddings/<model_name>/embeddings.db`

Per-model embedding cache with its own schema, managed by `jarvis/search/embeddings.py`.

#### `relationship_profiles`

Aggregated communication patterns cached per model.

| Column       | Type | Description                          |
| ------------ | ---- | ------------------------------------ |
| contact_id   | TEXT | Primary key                          |
| display_name | TEXT | Name                                 |
| profile_data | TEXT | JSON blob of style, tone, and topics |

#### `contact_facts`

Extracted facts about contacts from message history. Used by the knowledge graph for relationship/location/work/preference tracking.

| Column            | Type      | Description                                               |
| ----------------- | --------- | --------------------------------------------------------- |
| id                | INTEGER   | Primary key                                               |
| contact_id        | TEXT      | FK to contacts (chat_id)                                  |
| category          | TEXT      | relationship, location, work, preference, event           |
| subject           | TEXT      | Entity name (e.g., "Sarah", "Austin", "Google")           |
| predicate         | TEXT      | Relation type (optional, can be empty)                    |
| value             | TEXT      | Primary fact content (e.g., "is a software engineer")     |
| confidence        | REAL      | 0.0-1.0, from NLI verification                            |
| source_message_id | INTEGER   | Which message this fact was extracted from                |
| source_text       | TEXT      | Source message text (truncated to 500 chars)              |
| extracted_at      | TIMESTAMP | When the fact was extracted                               |
| linked_contact_id | TEXT      | Resolved contact reference from NER person linking (v13+) |
| valid_from        | TIMESTAMP | When the fact became true (v14+)                          |
| valid_until       | TIMESTAMP | When the fact stopped being true (v14+)                   |
| attribution       | TEXT      | Who fact is about: contact/user/third_party (v16+)        |
| segment_id        | INTEGER   | FK to conversation_segments(id) (v17+)                    |

**Unique constraint**: `(contact_id, category, subject, predicate)` prevents duplicate facts.

**Indexes**: `idx_facts_contact` on contact_id, `idx_facts_category` on category, `idx_facts_linked_contact` on linked_contact_id, `idx_facts_lookup` on (contact_id, predicate, subject).

#### `vec_facts` (sqlite-vec, v15+)

Semantic vector index over contact facts for similarity search.

| Column    | Type    | Description                            |
| --------- | ------- | -------------------------------------- |
| rowid     | INTEGER | Primary key (matches contact_facts.id) |
| embedding | BLOB    | 384-dim embedding of fact text         |

#### `conversation_segments` (v17+)

Persistent topic segments from message history.

| Column          | Type      | Description                       |
| --------------- | --------- | --------------------------------- |
| id              | INTEGER   | Primary key                       |
| segment_id      | TEXT      | Unique segment identifier         |
| chat_id         | TEXT      | Source conversation               |
| contact_id      | TEXT      | Associated contact                |
| start_time      | TIMESTAMP | Segment start time                |
| end_time        | TIMESTAMP | Segment end time                  |
| topic_label     | TEXT      | Detected topic/summary            |
| keywords_json   | TEXT      | JSON array of keywords            |
| entities_json   | TEXT      | JSON array of named entities      |
| message_count   | INTEGER   | Number of messages in segment     |
| confidence      | REAL      | Segment confidence score          |
| vec_chunk_rowid | INTEGER   | FK to vector chunk                |
| facts_extracted | BOOLEAN   | Whether facts have been extracted |
| created_at      | TIMESTAMP | Creation timestamp                |
| updated_at      | TIMESTAMP | Last update timestamp             |

#### `segment_messages` (v17+)

Links messages to their parent segments.

| Column        | Type    | Description                      |
| ------------- | ------- | -------------------------------- |
| segment_id    | INTEGER | FK to conversation_segments      |
| message_rowid | INTEGER | FK to message ROWID              |
| position      | INTEGER | Message position in segment      |
| is_from_me    | BOOLEAN | Whether message was sent by user |

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
        ├──▶ Fact Extractor → contact_facts (knowledge graph)
        │
        ├──▶ Contact Profiler (style analysis)
        │
        └──▶ Quality Metrics (tracking)
```

## Schema Version History

| Version | Changes                                                                 |
| ------- | ----------------------------------------------------------------------- |
| v12     | `contact_facts` table for knowledge graph                               |
| v13     | `linked_contact_id` column + NER person linking                         |
| v14     | Temporal fields: `valid_from`, `valid_until` on contact_facts           |
| v15     | `vec_facts` semantic index for contact fact similarity search           |
| v16     | `attribution` column on contact_facts (contact/user/third_party)        |
| v17     | `conversation_segments` and `segment_messages` tables + `segment_id` FK |

## Migration Notes

### Schema Changes

JARVIS does not modify the iMessage database. Internal schemas (in `jarvis.db`) are versioned and migrated automatically via `jarvis/db/migration.py`.

When upgrading:

1. Internal caches in `~/.jarvis/embeddings/` can be safely deleted.
2. Vector indexes will be rebuilt automatically using `sqlite-vec`.
3. Scheduler state in `jarvis.db` is migrated automatically.

### Backup

Only `~/.jarvis/` needs backup, specifically:

- `jarvis.db` - Contacts, pairs, and scheduled messages.
- `config.json` - User configuration.

Artifacts in `~/.jarvis/embeddings/` can be regenerated from `chat.db`.
