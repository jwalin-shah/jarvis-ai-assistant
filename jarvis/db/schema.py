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
    style_notes TEXT,                 -- 'casual, uses emojis'
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

-- Extracted message pairs from history (lean table)
CREATE TABLE IF NOT EXISTS pairs (
    id INTEGER PRIMARY KEY,
    contact_id INTEGER REFERENCES contacts(id),
    trigger_text TEXT NOT NULL,       -- what they said (may be multi-message joined)
    response_text TEXT NOT NULL,      -- what you said (may be multi-message joined)
    trigger_timestamp TIMESTAMP,      -- timestamp of first trigger message
    response_timestamp TIMESTAMP,     -- timestamp of first response message
    chat_id TEXT,                     -- source conversation
    -- Message IDs for debugging and deduplication
    trigger_msg_id INTEGER,           -- primary trigger message ID
    response_msg_id INTEGER,          -- primary response message ID
    trigger_msg_ids_json TEXT,        -- JSON array for multi-message triggers
    response_msg_ids_json TEXT,       -- JSON array for multi-message responses
    -- Conversation context (legacy, use pair_artifacts for v6+)
    context_text TEXT,                -- previous messages before trigger (for LLM context)
    -- Quality and filtering
    quality_score REAL DEFAULT 1.0,   -- 0.0-1.0, lower = worse
    flags_json TEXT,                  -- JSON: {"attachment_only":true, "short":true}
    is_group BOOLEAN DEFAULT FALSE,   -- True if from group chat (for filtering)
    is_holdout BOOLEAN DEFAULT FALSE, -- True if reserved for evaluation (not in training)
    -- Validity gate results (v6+)
    gate_a_passed BOOLEAN,            -- Rule gate result
    gate_b_score REAL,                -- Embedding similarity score
    gate_c_verdict TEXT,              -- NLI verdict (accept/reject/uncertain)
    validity_status TEXT,             -- Final: valid/invalid/uncertain
    -- Dialogue act classification (v7+)
    trigger_da_type TEXT,             -- e.g., WH_QUESTION, INFO_STATEMENT
    trigger_da_conf REAL,             -- Classifier confidence 0-1
    response_da_type TEXT,            -- e.g., STATEMENT, AGREE, ACKNOWLEDGE
    response_da_conf REAL,            -- Classifier confidence 0-1
    cluster_id INTEGER,               -- HDBSCAN cluster assignment (-1 for noise)
    -- Freshness and usage tracking
    usage_count INTEGER DEFAULT 0,    -- times this pair was used for generation
    last_used_at TIMESTAMP,           -- last time pair was used
    last_verified_at TIMESTAMP,       -- for re-indexing workflows
    source_timestamp TIMESTAMP,       -- original message timestamp (for decay)
    -- Content-based deduplication (v9+)
    content_hash TEXT,                -- MD5 of normalized trigger|response for dedup
    -- Uniqueness: use primary message IDs
    UNIQUE(trigger_msg_id, response_msg_id)
);

-- Heavy artifacts for pairs (split table to keep pairs lean)
CREATE TABLE IF NOT EXISTS pair_artifacts (
    pair_id INTEGER PRIMARY KEY REFERENCES pairs(id),
    context_json TEXT,                -- Structured context window (JSON list of messages)
    gate_a_reason TEXT,               -- Why Gate A rejected (if rejected)
    gate_c_scores_json TEXT,          -- Raw NLI scores (JSON dict)
    raw_trigger_text TEXT,            -- Original text before normalization
    raw_response_text TEXT            -- Original text before normalization
);

-- Clustered intent groups (optional, for later analytics)
CREATE TABLE IF NOT EXISTS clusters (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,        -- 'INVITATION', 'GREETING', 'SCHEDULE'
    description TEXT,
    example_triggers TEXT,            -- JSON array
    example_responses TEXT,           -- JSON array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Links pairs to vector index positions (keyed by pair_id for stability)
-- Note: faiss_id column name is legacy; now backed by sqlite-vec
CREATE TABLE IF NOT EXISTS pair_embeddings (
    pair_id INTEGER PRIMARY KEY REFERENCES pairs(id),
    faiss_id INTEGER UNIQUE,          -- position in vector index (legacy name)
    cluster_id INTEGER REFERENCES clusters(id),
    index_version TEXT                -- which index version this belongs to
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
CREATE INDEX IF NOT EXISTS idx_pairs_contact ON pairs(contact_id);
CREATE INDEX IF NOT EXISTS idx_pairs_chat ON pairs(chat_id);
CREATE INDEX IF NOT EXISTS idx_pairs_quality ON pairs(quality_score);
CREATE INDEX IF NOT EXISTS idx_pairs_validity ON pairs(validity_status);
CREATE INDEX IF NOT EXISTS idx_contacts_chat ON contacts(chat_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_index ON pair_embeddings(index_version);
CREATE INDEX IF NOT EXISTS idx_embeddings_faiss ON pair_embeddings(faiss_id);

-- Indexes for trigger pattern lookups (acknowledgment checks in router)
CREATE INDEX IF NOT EXISTS idx_pairs_trigger_text ON pairs(contact_id, LOWER(TRIM(trigger_text)));
CREATE INDEX IF NOT EXISTS idx_pairs_timestamp ON pairs(trigger_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_pairs_source_timestamp ON pairs(source_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_contacts_id ON contacts(id);

-- Composite indexes for common query patterns (faster filtered queries)
CREATE INDEX IF NOT EXISTS idx_pairs_chat_timestamp ON pairs(chat_id, trigger_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_pairs_contact_quality ON pairs(contact_id, quality_score DESC);
-- Index for content-based deduplication (v9+)
CREATE INDEX IF NOT EXISTS idx_pairs_content_hash ON pairs(content_hash);

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
    UNIQUE(contact_id, category, subject, predicate)
);

CREATE INDEX IF NOT EXISTS idx_facts_contact ON contact_facts(contact_id);
CREATE INDEX IF NOT EXISTS idx_facts_category ON contact_facts(category);
CREATE INDEX IF NOT EXISTS idx_facts_linked_contact ON contact_facts(linked_contact_id);

-- Indexes for scheduling tables
CREATE INDEX IF NOT EXISTS idx_scheduled_contact ON scheduled_drafts(contact_id);
CREATE INDEX IF NOT EXISTS idx_scheduled_status ON scheduled_drafts(status);
CREATE INDEX IF NOT EXISTS idx_scheduled_send_at ON scheduled_drafts(send_at);
CREATE INDEX IF NOT EXISTS idx_scheduled_priority ON scheduled_drafts(priority, send_at);
CREATE INDEX IF NOT EXISTS idx_send_queue_status ON send_queue(status);
CREATE INDEX IF NOT EXISTS idx_send_queue_scheduled ON send_queue(scheduled_draft_id);
"""

# Expected indices for verification
EXPECTED_INDICES = {
    "idx_pairs_contact",
    "idx_pairs_chat",
    "idx_pairs_quality",
    "idx_pairs_validity",
    "idx_contacts_chat",
    "idx_embeddings_index",
    "idx_embeddings_faiss",
    "idx_pairs_trigger_text",
    "idx_pairs_timestamp",
    "idx_pairs_source_timestamp",
    "idx_contacts_id",
    # Composite indexes for faster filtered queries
    "idx_pairs_chat_timestamp",
    "idx_pairs_contact_quality",
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
}

CURRENT_SCHEMA_VERSION = 15  # vec_facts semantic index for contact facts

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
}

# Allowlist of valid column types for ALTER TABLE migrations
VALID_COLUMN_TYPES = {"TEXT", "REAL", "INTEGER", "BOOLEAN", "BOOLEAN DEFAULT FALSE"}
