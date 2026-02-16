"""Database schema SQL and migration constants for JARVIS."""

# Schema SQL - Version 11 (validity gates, dialogue acts, scheduling, sqlite-vec)
SCHEMA_SQL = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Contacts with relationship labels and multiple handles
CREATE TABLE IF NOT EXISTS contacts (
    id INTEGER PRIMARY KEY,
    chat_id TEXT UNIQUE,              -- primary iMessage chat_id
    display_name TEXT NOT NULL,
    phone_or_email TEXT,              -- primary contact method
    handles_json TEXT,                -- JSON array: ["+15551234567", "email@x.com"]
    relationship TEXT,                -- 'sister', 'coworker', 'friend', 'boss'
    relationship_reasoning TEXT,      -- LLM-derived justification (v18+)
    style_notes TEXT,                 -- 'casual, uses emojis'
    last_extracted_rowid INTEGER,    -- iMessage ROWID of last extracted message (v18)
    last_extracted_at TIMESTAMP,      -- when extraction was last run (v18)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Style targets for contacts (computed from their pairs)
CREATE TABLE IF NOT EXISTS contact_style_targets (
    contact_id INTEGER PRIMARY KEY REFERENCES contacts(id),
    median_reply_length INTEGER DEFAULT 10,   -- median word count
    punctuation_rate REAL DEFAULT 0.5,        -- fraction with ending punctuation
    emoji_rate REAL DEFAULT 0.1,              -- fraction containing emojis
    greeting_rate REAL DEFAULT 0.2,           -- fraction starting with greeting
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Vector index versions for safe rebuilds
CREATE TABLE IF NOT EXISTS index_versions (
    id INTEGER PRIMARY KEY,
    version_id TEXT UNIQUE NOT NULL,  -- e.g., "20240115-143022"
    model_name TEXT NOT NULL,         -- e.g., "BAAI/bge-small-en-v1.5"
    embedding_dim INTEGER NOT NULL,   -- e.g., 384
    num_vectors INTEGER NOT NULL,
    index_path TEXT NOT NULL,         -- relative path to index file
    is_active BOOLEAN DEFAULT FALSE,
    normalized BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast lookup
CREATE INDEX IF NOT EXISTS idx_contacts_chat ON contacts(chat_id);
CREATE INDEX IF NOT EXISTS idx_contacts_id ON contacts(id);

-- Scheduled drafts for automated sending (v8+)
CREATE TABLE IF NOT EXISTS scheduled_drafts (
    id TEXT PRIMARY KEY,                -- UUID
    draft_id TEXT NOT NULL,             -- Reference to the draft
    contact_id INTEGER REFERENCES contacts(id),
    chat_id TEXT NOT NULL,              -- Chat to send to
    message_text TEXT NOT NULL,         -- The message content
    send_at TIMESTAMP NOT NULL,         -- Scheduled send time
    priority TEXT DEFAULT 'normal',     -- urgent/normal/low
    status TEXT DEFAULT 'pending',      -- pending/queued/sending/sent/failed/cancelled/expired
    timezone TEXT,                      -- Contact's timezone (IANA format)
    depends_on TEXT,                    -- ID of item this depends on
    retry_count INTEGER DEFAULT 0,      -- Number of retry attempts
    max_retries INTEGER DEFAULT 3,      -- Maximum retries allowed
    expires_at TIMESTAMP,               -- When this schedule expires
    result_json TEXT,                   -- JSON with send result details
    metadata_json TEXT,                 -- Additional metadata (instructions, etc.)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Contact timing preferences for smart scheduling (v8+)
CREATE TABLE IF NOT EXISTS contact_timing_prefs (
    contact_id INTEGER PRIMARY KEY REFERENCES contacts(id),
    timezone TEXT,                      -- Contact's timezone (IANA format)
    quiet_hours_json TEXT,              -- JSON with quiet hours config
    preferred_hours_json TEXT,          -- JSON array of preferred hours (0-23)
    optimal_weekdays_json TEXT,         -- JSON array of preferred weekdays (0-6)
    avg_response_time_mins REAL,        -- Average response time in minutes
    last_interaction TIMESTAMP,         -- Last interaction timestamp
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Send queue for tracking message delivery (v8+)
CREATE TABLE IF NOT EXISTS send_queue (
    id TEXT PRIMARY KEY,                -- UUID
    scheduled_draft_id TEXT REFERENCES scheduled_drafts(id),
    status TEXT DEFAULT 'pending',      -- pending/sending/sent/failed
    queued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sent_at TIMESTAMP,                  -- When actually sent
    error TEXT,                         -- Error message if failed
    attempts INTEGER DEFAULT 0,         -- Number of send attempts
    next_retry_at TIMESTAMP             -- When to retry next
);

-- Contact facts for knowledge graph (v12+, v13 adds linked_contact_id, v14 adds temporal)
CREATE TABLE IF NOT EXISTS contact_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contact_id TEXT NOT NULL,
    category TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    value TEXT DEFAULT '',
    confidence REAL DEFAULT 1.0,
    source_message_id INTEGER,
    source_text TEXT DEFAULT '',
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    linked_contact_id TEXT,                  -- resolved contact reference (v13+)
    valid_from TIMESTAMP,                    -- when fact became true (v14+)
    valid_until TIMESTAMP,                   -- when fact stopped being true (v14+)
    attribution TEXT DEFAULT 'contact',      -- who fact is about: contact/user/third_party (v16+)
    segment_id INTEGER,                      -- FK to conversation_segments(id) (v17+)
    UNIQUE(contact_id, category, subject, predicate, attribution)
);

CREATE INDEX IF NOT EXISTS idx_facts_contact ON contact_facts(contact_id);
CREATE INDEX IF NOT EXISTS idx_facts_category ON contact_facts(category);
CREATE INDEX IF NOT EXISTS idx_facts_linked_contact ON contact_facts(linked_contact_id);
CREATE INDEX IF NOT EXISTS idx_facts_lookup ON contact_facts(contact_id, predicate, subject);

-- Raw facts log (keeps every extracted candidate for auditing)
CREATE TABLE IF NOT EXISTS fact_candidates_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contact_id TEXT,
    chat_id TEXT,
    message_id INTEGER,
    subject TEXT,
    predicate TEXT,
    value TEXT,
    category TEXT,
    confidence REAL,
    source_text TEXT,
    attribution TEXT DEFAULT 'contact',
    segment_id INTEGER,
    log_stage TEXT DEFAULT 'extraction',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fact_candidates_contact ON fact_candidates_log(contact_id);
CREATE INDEX IF NOT EXISTS idx_fact_candidates_chat ON fact_candidates_log(chat_id);

-- Conversation segments (v17+, simplified v19 - removed topic metadata)
-- Stores message boundaries without low-quality topic labels/keywords
CREATE TABLE IF NOT EXISTS conversation_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id TEXT UNIQUE NOT NULL,      -- UUID for the segment
    chat_id TEXT NOT NULL,                -- iMessage chat identifier
    contact_id TEXT,                      -- FK to contacts
    start_time TIMESTAMP NOT NULL,        -- First message timestamp
    end_time TIMESTAMP NOT NULL,          -- Last message timestamp
    message_count INTEGER NOT NULL,       -- Number of messages
    preview TEXT,                         -- First 100 chars (for UI display)
    vec_chunk_rowid INTEGER,              -- FK to vec_chunks for embedding
    facts_extracted BOOLEAN DEFAULT FALSE, -- Whether facts were extracted
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS segment_messages (
    segment_id INTEGER NOT NULL REFERENCES conversation_segments(id),
    message_rowid INTEGER NOT NULL,
    position INTEGER NOT NULL,
    is_from_me BOOLEAN NOT NULL,
    PRIMARY KEY (segment_id, message_rowid)
);

CREATE INDEX IF NOT EXISTS idx_segments_chat ON conversation_segments(chat_id);
CREATE INDEX IF NOT EXISTS idx_segments_chat_time
    ON conversation_segments(chat_id, start_time DESC);
CREATE INDEX IF NOT EXISTS idx_segments_segment_id ON conversation_segments(segment_id);
CREATE INDEX IF NOT EXISTS idx_segments_contact ON conversation_segments(contact_id);
CREATE INDEX IF NOT EXISTS idx_segmsg_message ON segment_messages(message_rowid);

-- Indexes for scheduling tables
CREATE INDEX IF NOT EXISTS idx_scheduled_contact ON scheduled_drafts(contact_id);
CREATE INDEX IF NOT EXISTS idx_scheduled_status ON scheduled_drafts(status);
CREATE INDEX IF NOT EXISTS idx_scheduled_send_at ON scheduled_drafts(send_at);
CREATE INDEX IF NOT EXISTS idx_scheduled_priority ON scheduled_drafts(priority, send_at);
CREATE INDEX IF NOT EXISTS idx_send_queue_status ON send_queue(status);
CREATE INDEX IF NOT EXISTS idx_send_queue_scheduled ON send_queue(scheduled_draft_id);

-- Reply logs for full traceability (v20+)
CREATE TABLE IF NOT EXISTS reply_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT,
    contact_id TEXT,
    incoming_text TEXT,
    classification_json TEXT,         -- category, urgency, etc.
    rag_context_json TEXT,           -- full content of retrieved documents
    final_prompt TEXT,               -- the actual prompt sent to LLM
    response_text TEXT,
    confidence REAL,
    metadata_json TEXT,              -- latency, model info, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_reply_logs_chat ON reply_logs(chat_id);
CREATE INDEX IF NOT EXISTS idx_reply_logs_created ON reply_logs(created_at);
"""

# Expected indices for verification
EXPECTED_INDICES = {
    "idx_contacts_chat",
    "idx_contacts_id",
    # Scheduling indexes (v8+)
    "idx_scheduled_contact",
    "idx_scheduled_status",
    "idx_scheduled_send_at",
    "idx_scheduled_priority",
    "idx_send_queue_status",
    "idx_send_queue_scheduled",
    # Contact facts indexes (v12+)
    "idx_facts_contact",
    "idx_facts_category",
    "idx_facts_linked_contact",
    "idx_facts_lookup",
    # Segment indexes (v17+)
    "idx_segments_chat",
    "idx_segments_chat_time",
    "idx_segments_segment_id",
    "idx_segments_contact",
    "idx_segmsg_message",
    "idx_fact_candidates_contact",
    "idx_fact_candidates_chat",
    "idx_reply_logs_chat",
    "idx_reply_logs_created",
}

CURRENT_SCHEMA_VERSION = 21  # added relationship_reasoning column

# Allowlist of valid column names for ALTER TABLE migrations (prevent SQL injection)
VALID_MIGRATION_COLUMNS = {
    # v3: context
    "context_text",
    # v4: group chat
    "is_group",
    # v5: holdout
    "is_holdout",
    # v6: validity gates
    "gate_a_passed",
    "gate_b_score",
    "gate_c_verdict",
    "validity_status",
    # v7: dialogue acts + clustering
    "trigger_da_type",
    "trigger_da_conf",
    "response_da_type",
    "response_da_conf",
    "cluster_id",
    # v9: content hash
    "content_hash",
    # v13: NER person linking
    "linked_contact_id",
    # v16: attribution
    "attribution",
    # v17: segment traceability
    "segment_id",
    # v18: extraction tracking
    "last_extracted_rowid",
    "last_extracted_at",
    # v18: relationship
    "relationship_reasoning",
}

# Allowlist of valid column types for ALTER TABLE migrations
VALID_COLUMN_TYPES = {
    "TEXT",
    "TEXT DEFAULT 'contact'",
    "REAL",
    "INTEGER",
    "BOOLEAN",
    "BOOLEAN DEFAULT FALSE",
}
