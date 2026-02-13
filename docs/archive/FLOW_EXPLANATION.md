# JARVIS Flow: From Messages to Real-Time Generation

This document explains the complete flow from preprocessing iMessage data to generating real-time responses.

## Overview

JARVIS uses a **chunk-based RAG (Retrieval-Augmented Generation)** approach:
1. **Preprocessing**: Extract topic chunks from conversations
2. **Indexing**: Embed and index chunks in FAISS
3. **Real-time**: Retrieve similar chunks → Generate response

---

## Phase 1: Preprocessing (Offline)

### Step 1.1: Read Messages from iMessage

**File**: `integrations/imessage/reader.py`

```python
ChatDBReader.get_messages(chat_id, limit=100)
```

- Reads from macOS `~/Library/Messages/chat.db` (read-only)
- Extracts: text, sender, timestamp, attachments, reactions
- Returns `Message` objects

**Storage**: None (read-only from macOS DB)

---

### Step 1.2: Extract Pairs (Optional - for analysis)

**File**: `jarvis/_cli_main.py` → `_cmd_db_extract()`

**Command**: `jarvis db extract`

- Groups messages into (trigger, response) pairs
- Trigger = message from them
- Response = your reply within time window
- Stores in `jarvis.db` → `pairs` table

**Why pairs?** Used for:
- Quality analysis
- Clustering (intent discovery)
- Template mining
- **NOT used for real-time generation**

---

### Step 1.3: Create Topic Chunks (The Real Preprocessing)

**File**: `jarvis/topic_chunker.py`

**Command**: `uv run python scripts/preprocess_chunks.py`

**Process**:
1. **Topic Detection** (`jarvis/topic_detector.py`):
   - Analyzes linguistic features (nouns, entities, time gaps)
   - Detects topic boundaries
   - Splits conversation into topic-coherent segments

2. **Chunk Creation**:
   ```python
   chunks = chunk_conversation(messages, contact_id="chat123")
   ```

   Each `TopicChunk` contains:
   - All messages in a topic segment
   - Time range (start_time, end_time)
   - Topic keywords (extracted nouns/entities)
   - Label (human-readable: "Planning Lunch", "Work Discussion")
   - Formatted text (all messages formatted together)

**Example Chunk**:
```
Topic: Planning Lunch
Keywords: ["lunch", "restaurant", "tomorrow", "12pm"]
Messages: [
  "Want to grab lunch tomorrow?",
  "Sure! What time works?",
  "How about 12pm?",
  "Perfect, see you then!"
]
Formatted: "Planning Lunch\n\nWant to grab lunch tomorrow?\nSure! What time works?\n..."
```

**Storage**: `jarvis.db` → `topic_chunks` table

---

### Step 1.4: Build Chunk Index

**File**: `jarvis/chunk_index.py` → `ChunkIndexBuilder.build_index()`

**Command**: `uv run python scripts/preprocess_chunks.py --rebuild-index`

**Process**:
1. **Load chunks** from database
2. **Compute embeddings**:
   ```python
   texts = [chunk.text_for_embedding for chunk in chunks]
   embeddings = embedder.encode(texts, normalize=True)  # Shape: (n_chunks, 384)
   ```
   - Uses MLX embedding service (bge-small/gte-tiny/etc)
   - Each chunk → 384-dim vector

3. **Create FAISS index**:
   ```python
   # Default: IVFPQ 4x (3.8x compression, 92% recall)
   index = _create_faiss_index(
       dimension=384,
       num_vectors=len(embeddings),
       embeddings=embeddings,
       index_type="ivfpq_4x"  # Options: flat, ivf, ivfpq_4x, ivfpq_8x
   )
   index.add(embeddings)
   ```
   
   **Index Types**:
   - `flat`: IndexFlatIP (brute force) - 100% recall, highest memory
   - `ivf`: IndexIVFFlat - 93% recall, same size (clustering for speed)
   - `ivfpq_4x`: IVFPQ 384x8 - **92% recall, 3.8x compression (DEFAULT)**
   - `ivfpq_8x`: IVFPQ 192x8 - 88% recall, 7.2x compression

4. **Save to disk**:
   - `~/.jarvis/indexes/{model_name}/{version_id}/chunks.faiss`
   - `~/.jarvis/indexes/{model_name}/{version_id}/metadata.json`

5. **Register in DB**:
   - `jarvis.db` → `chunk_index_versions` table
   - Marks as `is_active=True`

**Storage**:
- FAISS index: `~/.jarvis/indexes/`
- Metadata: JSON with chunk IDs, labels, formatted text
- DB: Version tracking

---

## Phase 2: Incremental Updates (When New Messages Arrive)

### Step 2.1: Detect New Messages

**File**: `jarvis/watcher.py` or `desktop/src/lib/stores/conversations.ts`

- File watcher monitors `chat.db` for changes
- Polls for new messages (delta detection)
- Broadcasts `new_message` events

### Step 2.2: Incremental Chunking

**File**: `jarvis/incremental_chunking.py`

**Function**: `add_new_messages_to_chunks()`

**Process**:
1. Get last chunk for conversation (`db.get_last_chunk_for_chat()`)
2. Check time gap:
   - If < 30 minutes: Check if topic is same
   - If same topic: **Extend last chunk**
   - If different topic: **Create new chunk**
3. If time gap > 30 minutes: **Create new chunk**

**Storage**: New/updated chunks stored in `jarvis.db` → `chunks` table

**Note**: After adding new chunks, you may want to rebuild the FAISS index:
```bash
uv run python scripts/preprocess_chunks.py --rebuild-index
```

---

## Phase 3: Real-Time Generation

### Step 3.1: Receive Incoming Message

**Entry Points**:
- `api/routers/drafts.py` → `generate_smart_reply()`
- `jarvis/socket_server.py` → `_get_smart_replies()`
- Desktop app → API call

**Input**: 
- `incoming`: "Want to grab lunch tomorrow?"
- `chat_id`: "chat123"
- `thread`: Recent messages for context

---

### Step 3.2: Retrieve Context Chunks

**File**: `jarvis/router.py` → `ReplyRouter.route()`

```python
context = self.retriever.get_context(
    query=incoming,
    k=3,  # Top 3 chunks
    threshold=0.35,  # Minimum similarity
    chat_id=chat_id,  # Boost same conversation
)
```

**File**: `jarvis/retrieval.py` → `ChunkRetriever.get_context()`

**Process**:
1. **Encode query**:
   ```python
   query_embedding = embedder.encode([incoming], normalize=True)  # (1, 384)
   ```

2. **Load FAISS index** (lazy, first call):
   ```python
   # File: jarvis/chunk_index.py → ChunkIndexSearcher._load_index()
   # - Gets active index from DB: db.get_active_chunk_index()
   # - Loads FAISS index from disk: ~/.jarvis/indexes/{model}/{version}/chunks.faiss
   # - Loads metadata JSON: ~/.jarvis/indexes/{model}/{version}/metadata.json
   # - Sets nprobe for IVF indexes (default: 128)
   # - Thread-safe, cached per version
   ```

3. **Search FAISS index**:
   ```python
   # File: jarvis/chunk_index.py → ChunkIndexSearcher.search()
   scores, indices = index.search(query_embedding, k=6)  # Get 6, filter to top 3
   ```

4. **Filter by threshold**:
   ```python
   results = [
       ChunkSearchResult(
           chunk_id=chunk_id,
           similarity=score,  # Cosine similarity (0-1)
           label="Planning Lunch",
           formatted_text="Want to grab lunch tomorrow?\nSure!...",
           ...
       )
       for score, idx in zip(scores[0], indices[0])
       if score >= threshold
   ]
   ```

5. **Boost same conversation**:
   ```python
   if chunk.chat_id == chat_id:
       score *= 1.2  # 20% boost
   ```

6. **Return top k chunks**:
   ```python
   return GenerationContext(
       query=incoming,
       chunks=results[:k]  # Top 3
   )
   ```

**Output**: `GenerationContext` with 0-3 similar chunks

---

### Step 3.3: Build Prompt

**File**: `jarvis/router.py` → `ReplyRouter._build_prompt()`

**Process**:
1. **Get relationship context** (cached):
   ```python
   # File: jarvis/router.py → _get_relationship_type()
   relationship = RelationshipClassifier.classify_contact(chat_id)
   # Returns: "close friend", "family", "coworker", etc.
   ```

2. **Build prompt with relationship context**:
   ```python
   prompt_parts = [
       "You are helping compose a reply to a message.",
       "Keep responses natural, casual, and concise.",
   ]
   
   # Add relationship context
   if relationship == "close friend":
       prompt_parts.append("\nRelationship context: close friend - keep tone very casual and friendly.")
   elif relationship == "coworker":
       prompt_parts.append("\nRelationship context: coworker - keep tone professional but friendly.")
   # ... etc
   ```

if context.has_context:
    prompt_parts.append("\nSimilar past conversations for style reference:")
    prompt_parts.append(context.format_for_prompt(max_chunks=2))
    # Example:
    # --- Planning Lunch ---
    # Want to grab lunch tomorrow?
    # Sure! What time works?
    # How about 12pm?
    # Perfect, see you then!
    #
    # --- Work Discussion ---
    # ...

if thread:
    prompt_parts.append("\nRecent conversation:")
    for msg in thread[-5:]:
        prompt_parts.append(f"  {msg}")

prompt_parts.append(f"\nReply to: {incoming}")
prompt_parts.append("\nYour reply:")
```

**Example Prompt**:
```
You are helping compose a reply to a message.
Keep responses natural, casual, and concise.

Similar past conversations for style reference:
--- Planning Lunch ---
Want to grab lunch tomorrow?
Sure! What time works?
How about 12pm?
Perfect, see you then!

Recent conversation:
  Hey, how's it going?
  Pretty good, thanks!

Reply to: Want to grab lunch tomorrow?

Your reply:
```

---

### Step 3.4: Generate Response

**File**: `jarvis/router.py` → `ReplyRouter._generate()`

```python
response = self.generator.generate(
    prompt=prompt,
    max_tokens=150,
    temperature=0.7,
)
```

**File**: `models/generator.py` → `MLXGenerator.generate()`

**Process**:
1. Load LLM model (LFM2.5-1.2B-4bit) if not loaded
2. Tokenize prompt
3. Run inference (MLX on Apple Silicon GPU)
4. Decode tokens → text
5. Return: `"Sure! What time works?"`

**Output**: Generated response text

---

### Step 3.5: Return Result

**File**: `jarvis/router.py` → `ReplyRouter.route()`

```python
return RouteResult(
    response="Sure! What time works?",
    similarity=0.87,  # Best chunk similarity
    chunk_id="chunk_abc123",
    chunk_label="Planning Lunch",
    latency_ms=245.3,
    context_used=True,
)
```

**Returned to**:
- API endpoint → JSON response
- Socket server → JSON-RPC response
- Desktop app → Displayed as draft suggestion

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: PREPROCESSING (Offline)                           │
└─────────────────────────────────────────────────────────────┘

iMessage DB (read-only)
    │
    ├─→ ChatDBReader.get_messages()
    │   └─→ List[Message]
    │
    ├─→ Extract Pairs (optional)
    │   └─→ jarvis.db.pairs table
    │
    └─→ Topic Chunker
        ├─→ Topic Detection (linguistic features)
        ├─→ Split into TopicChunk[]
        └─→ jarvis.db.topic_chunks table

Topic Chunks
    │
    └─→ ChunkIndexBuilder
        ├─→ Embed chunks (MLX service)
        ├─→ Create FAISS IndexFlatIP
        ├─→ Save: ~/.jarvis/indexes/{model}/{version}/chunks.faiss
        └─→ Register: jarvis.db.chunk_index_versions

┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: REAL-TIME GENERATION                               │
└─────────────────────────────────────────────────────────────┘

Incoming Message: "Want to grab lunch?"
    │
    └─→ ReplyRouter.route()
        │
        ├─→ ChunkRetriever.get_context()
        │   ├─→ Encode query → embedding (384-dim)
        │   ├─→ FAISS search → top k chunks
        │   └─→ GenerationContext (chunks + metadata)
        │
        ├─→ Build prompt
        │   ├─→ System instructions
        │   ├─→ Similar chunks (style reference)
        │   ├─→ Recent thread context
        │   └─→ Query message
        │
        └─→ MLXGenerator.generate()
            ├─→ Load model (if needed)
            ├─→ Tokenize prompt
            ├─→ Run inference (GPU)
            └─→ Decode → response text

Response: "Sure! What time works?"
```

---

## Key Components

### 1. Topic Chunks (Not Pairs!)

**Why chunks?**
- Richer context: Full topic flow, not isolated Q→A
- Better style matching: LLM sees HOW you discuss topics
- Handles multi-turn conversations naturally

**Storage**:
- `jarvis.db.topic_chunks` table
- FAISS index: `~/.jarvis/indexes/{model}/{version}/chunks.faiss`
- Metadata JSON: chunk labels, formatted text, keywords

### 2. FAISS Index

**Type**: `IVFPQ 4x` (default) - IVF clustering + Product Quantization compression
- **Memory**: 3.8x compression (saves 73% vs flat)
- **Recall**: ~92% (8% loss vs 100% flat)
- **Speed**: Scales better than flat (clusters reduce search space)
- **Training**: One-time cost (~10-20s), persists to disk

**Options**:
- `flat`: IndexFlatIP - 100% recall, no compression
- `ivf`: IndexIVFFlat - 93% recall, clustering only (no compression)
- `ivfpq_4x`: IVFPQ 384x8 - **92% recall, 3.8x compression (DEFAULT)**
- `ivfpq_8x`: IVFPQ 192x8 - 88% recall, 7.2x compression

**Location**: `~/.jarvis/indexes/chunks/{model_name}/{version_id}/`

### 3. Embeddings

**Model**: Configurable (bge-small, gte-tiny, etc.)
- Default: `bge-small` (384 dims, ~100-150ms)
- Faster: `gte-tiny` (384 dims, ~50-70ms)
- All models output 384 dims (index compatible)

**Service**: MLX embedding microservice
- Unix socket: `~/.jarvis/mlx-embed-service/socket`
- GPU-accelerated on Apple Silicon
- Caching: `CachedEmbedder` (LRU, 1000 entries)

### 4. Generation

**Model**: LFM2.5-1.2B-Instruct-MLX-4bit
- Small, fast, runs on-device
- ~600-3000ms for 50 tokens
- Uses retrieved chunks as few-shot examples

---

## What's NOT Used in Real-Time Flow

### ❌ Clustering (`jarvis/clustering.py`)
- **Purpose**: Analyze pairs to discover intent clusters
- **Usage**: CLI only (`jarvis db cluster`)
- **Why not used**: Router uses chunks, not clusters

### ❌ Embedding Profiles (`archive/embedding_profile.py`)
- **Purpose**: Build relationship profiles with topic clusters
- **Usage**: Archived script (`archive/build_embedding_profiles.py`)
- **Why not used**: Router uses chunk index directly

### ❌ Trigger Index (`jarvis/index.py`)
- **Purpose**: FAISS index of trigger messages (for pair matching)
- **Usage**: Legacy/analysis only
- **Why not used**: Router uses chunk index, not trigger index

### ❌ Semantic Search (`jarvis/semantic_search.py`)
- **Purpose**: Search individual messages by semantic similarity
- **Usage**: Separate feature (search UI), not used in generation
- **Why not used**: Router uses chunks, not individual messages

---

## Performance Characteristics

### Preprocessing (One-time)
- **Chunking**: ~100ms per conversation
- **Embedding**: ~50-150ms per chunk (batch processing)
- **Index Training**: ~10-20s (one-time, IVFPQ clustering)
- **Indexing**: ~10ms per chunk (FAISS add)

### Real-Time (Per Request)
- **Query embedding**: ~50-150ms (cached if repeated)
- **FAISS search**: ~1-5ms (exact search, very fast)
- **Prompt building**: <1ms
- **Generation**: ~600-3000ms (model inference)
- **Total**: ~700-3200ms end-to-end

---

## Configuration

**Chunk Index**:
- Location: `~/.jarvis/indexes/{model_name}/`
- Active version: `jarvis.db.chunk_index_versions.is_active=True`

**Embedding Model**:
- Config: `~/.jarvis/config.json` → `embedding.model_name`
- Options: `bge-small`, `gte-tiny`, `minilm-l6`, `bge-micro`

**Generation**:
- Model: LFM2.5-1.2B-Instruct-MLX-4bit (hardcoded)
- Max tokens: 150
- Temperature: 0.7

---

## Summary

**The Real Flow**:
1. **Preprocessing**: Messages → Topic Chunks → FAISS Index
2. **Real-time**: Incoming message → Embed → Search FAISS → Retrieve chunks → Generate

**Key Insight**: JARVIS uses **topic chunks** (not pairs, not clusters) for RAG. Chunks provide richer context and better style matching than isolated Q→A pairs.
