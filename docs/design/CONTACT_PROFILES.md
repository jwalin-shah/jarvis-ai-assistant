# Contact Profiles & Topic Discovery

> **Last Updated:** 2026-02-10

## Design Decisions & Rationale

### Why Per-Contact Topics (Not Global)?

| Approach | Pros | Cons |
|----------|------|------|
| **Global topics** ("sports", "food") | Simple, one model | Generic, misses YOUR topics |
| **Per-contact topics** | Learns "paradigms-class", "valorant" | More storage (~15KB/contact) |

**Decision**: Per-contact. Your conversations have specific topics (professor names, games you play, friend groups) that global topics miss. 15KB/contact is trivial.

### Why HDBSCAN (Not K-Means, LDA, BERTopic)?

| Algorithm | Pros | Cons | Memory |
|-----------|------|------|--------|
| **K-Means** | Fast | Need to specify K upfront | Low |
| **LDA** | Interpretable | Bag-of-words, misses semantics | Low |
| **BERTopic** | Best quality | [2+ GB RAM, slow](https://github.com/MaartenGr/BERTopic/issues/484) | High |
| **HDBSCAN** | Auto-detects K, handles noise | Slightly slower | Low |

**Decision**: HDBSCAN. It auto-detects number of topics, handles variable density clusters, and works on embeddings. ~800ms for 1000 messages is acceptable for one-time extraction.

### Why bge-small (Not MiniLM, OpenAI)?

| Model | Dimensions | MTEB Score | Memory | Speed |
|-------|------------|------------|--------|-------|
| OpenAI ada-002 | 1536 | ~61 | API cost | Slow (network) |
| MiniLM-L6 | 384 | ~58 | ~90MB | Fast |
| **bge-small** | 384 | ~62 | ~130MB | Fast |
| bge-large | 1024 | ~64 | ~1.3GB | Slow |

**Decision**: bge-small. Best quality-to-size ratio, already used for FAISS search (no extra model load), runs locally via MLX server.

### Why Hybrid Approach for Style (Regex + Preprocessing Features)?

Research shows ["hybrid approaches combining traditional stylometric features with modern embeddings tend to outperform either approach alone"](https://ceur-ws.org/Vol-4038/paper_283.pdf) (PAN 2025).

We use **regex patterns + features from preprocessing we already run**:

| Feature | Source | What it tells us |
|---------|--------|------------------|
| `uses_lowercase` | Regex | Casual texters don't capitalize |
| `uses_abbreviations` | TEXT_ABBREVIATIONS set | "u rn gonna" = informal |
| `slang_frequency` | jarvis/slang.py (already run) | Higher = more casual |
| `spell_error_rate` | symspellpy (already run) | Fast typers make more errors |
| `vocabulary_diversity` | Word analysis | Formal = richer vocabulary |
| `emoji_frequency` | EMOJI_PATTERN | Emoji usage patterns |

**Why not pure ML/embeddings for style?**
- Embeddings capture *meaning*, not *style* ("u wanna" and "do you want to" have same meaning, different style)
- Need labeled training data for classifiers
- Our hybrid approach: 0.5ms, no training data

**Results:**

| Metric | Casual Texter | Formal Texter |
|--------|---------------|---------------|
| slang_frequency | 1.1/msg | 0.0/msg |
| spell_error_rate | 17.1% | 1.8% |
| avg_words_per_msg | 3.5 | 5.6 |
| formality | very_casual | casual |

**Decision**: Hybrid (regex + preprocessing features). Leverages work we already do, no extra compute, richer signals than pure regex.

### Why Unsupervised (Not Fine-tuned Classifier)?

| Approach | Pros | Cons |
|----------|------|------|
| **Fine-tuned classifier** | High accuracy | Need labeled data, retraining |
| **LightGBM + BERT embeddings** | High accuracy, fast | Requires labeled training data |
| **Unsupervised clustering** | No labels, adapts | Slightly lower precision |

**Decision**: Unsupervised. No labeled data collection needed, automatically adapts to new contacts and topics, minimal memory overhead.

### Why Cache Profiles (Not Compute On-Demand)?

Old approach in `prompts.py`:
```python
# Every prompt generation (10x/minute):
style = analyze_user_style(messages)  # Fetches messages, analyzes
```

Problems:
- Redundant: style doesn't change between messages
- Wasteful: same computation repeated
- Slow startup: need to fetch messages

New approach:
```python
# Once per contact (during extraction):
profile = build_contact_profile(contact_id, messages, embeddings)

# Every prompt (instant):
profile = get_contact_profile(contact_id)  # 0.002ms from cache
```

**Decision**: Cache. Build once (~800ms), load forever (~0.002ms).

### Why JSON Files (Not SQLite)?

| Storage | Pros | Cons |
|---------|------|------|
| **SQLite** | ACID, queries | Centroid BLOBs awkward, migration needed |
| **JSON files** | Simple, human-readable | No queries, separate files |

**Decision**: JSON files for now. Simple to implement, easy to debug (can read profiles), ~15KB/file is trivial. Can migrate to SQLite later if needed.

---

## Overview

Per-contact learned profiles that capture **writing style** and **conversation topics** without any labeled training data. Everything is unsupervised and adapts automatically to new contacts.

## Architecture

```
EXTRACTION (once per contact, ~800ms)
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Messages from DB ──┬──► Text Preprocessing ──► Style Analysis          │
│                     │    (spaCy/NER for                                 │
│                     │     entity extraction)   - avg length             │
│                     │                          - formality              │
│                     │                          - abbreviations          │
│                     │                          - emoji frequency        │
│                     │                                                   │
│                     └──► Embeddings ──► HDBSCAN ──► Topic Discovery    │
│                         (bge-small,        │                            │
│                          already have)     └──► Topic centroids         │
│                                               └──► Keywords per topic   │
│                                                                         │
│  Output: ContactProfile (JSON, ~15KB per contact)                       │
│          - StyleProfile                                                 │
│          - DiscoveredTopics (centroids + keywords)                      │
└─────────────────────────────────────────────────────────────────────────┘

GENERATION (every prompt, ~0.002ms)
┌─────────────────────────────────────────────────────────────────────────┐
│  Load cached ContactProfile ──► Style instructions for LLM             │
│                             └──► Topic context for relevance           │
└─────────────────────────────────────────────────────────────────────────┘

CHUNKING (per message, ~0.03ms)
┌─────────────────────────────────────────────────────────────────────────┐
│  New message ──► Embed ──► Cosine sim to topic centroids ──► Topic ID  │
│              └──► If topic changed from previous ──► Chunk boundary    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Why This Approach?

### Problem with Old Approach

```python
# OLD: prompts.py - computed EVERY prompt generation
style = analyze_user_style(user_messages)  # 0.22ms each time

# Called 10x/minute = 600x/hour = wasted compute
```

### New Approach Benefits

| Aspect | Old | New |
|--------|-----|-----|
| Style computation | Every prompt | Once per contact |
| Topic detection | Fixed keywords | Learned per person |
| Adapts to new people | ❌ No | ✅ Automatically |
| Labeled data needed | N/A | ❌ None |
| Storage | None | ~15KB/contact |

## No Labeled Data Required

Everything is **unsupervised**:

| Component | Method | Training Data? |
|-----------|--------|----------------|
| **Style Analysis** | Regex patterns + word counting | ❌ None |
| **Topic Discovery** | HDBSCAN clustering on embeddings | ❌ None |
| **Topic Classification** | Cosine similarity to centroids | ❌ None |
| **Chunk Detection** | Topic change = boundary | ❌ None |

### Adapts to New Contacts Automatically

```
New contact "Alice" sends first 20 messages:
  1. Embed messages with bge-small
  2. Cluster with HDBSCAN → discovers HER topics:
     - Topic 0: "grad school", "thesis", "advisor"
     - Topic 1: "hiking", "trails", "weekend"
     - Topic 2: "cooking", "recipes", "dinner"
  3. Extract HER style:
     - formality: "casual"
     - uses_lowercase: True
     - emoji_frequency: 0.4
  4. Save profile → done

No retraining, no labeled data, works immediately.
```

## How It Connects to Existing Code

### 1. Text Preprocessing (jarvis/text_normalizer.py)

```
┌─────────────────────────────────────────────────────────────────┐
│ Raw Message                                                      │
│ "u wanna grab dinner w Jake rn? 😂"                             │
│                                                                  │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ EXTRACTION PROFILE (stored for LLM)                         │ │
│ │   expand_slang: OFF   → keeps "u wanna rn"                  │ │
│ │   spell_check: ON     → fixes typos                         │ │
│ │   normalize_emojis: OFF → keeps 😂                          │ │
│ │   OUTPUT: "u wanna grab dinner w Jake rn? 😂"               │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ CLASSIFICATION PROFILE (for NER/embeddings)                 │ │
│ │   expand_slang: ON    → "you want to right now"             │ │
│ │   ner_enabled: ON     → extracts "Jake" as PERSON           │ │
│ │   OUTPUT: cleaner text for ML                               │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2. NER/spaCy (jarvis/ner_client.py)

NER is used for **entity extraction**, NOT topic detection:

```
NER extracts:    "Jake" → PERSON, "Main St" → LOC
Topics extract:  "dinner plans", "basketball", "homework" (from clustering)

Different purposes:
- NER: WHO/WHERE/WHEN (entities for context)
- Topics: WHAT (conversation subject for chunking)
```

### 3. Embeddings (bge-small via MLX server)

```
Same embeddings used for:
1. Vector search (FAISS index)     ← already computed
2. Topic discovery (HDBSCAN)       ← reuse same embeddings!
3. Topic classification (cosine)   ← reuse same embeddings!

No additional embedding computation needed.
```

## Integration Points

### jarvis/extract.py

```python
# After indexing messages for a contact
from jarvis.contact_profile import build_contact_profile

profile = build_contact_profile(
    contact_id=contact_id,
    messages=[m.text for m in messages],
    embeddings=embeddings,  # Already computed for FAISS
)
```

### jarvis/prompts.py

```python
# OLD
style = analyze_user_style(user_messages)

# NEW
from jarvis.contact_profile import get_contact_profile
profile = get_contact_profile(contact_id)
style = profile.style if profile else analyze_user_style(user_messages)
```

### jarvis/threading.py

```python
# Use topic-based chunking
from jarvis.contact_profile import get_contact_profile

profile = get_contact_profile(contact_id)
if profile and profile.topics:
    for i, emb in enumerate(embeddings):
        assignment = profile.topics.classify(emb)
        if assignment.is_chunk_start:
            # Start new conversation chunk
```

## Performance

| Operation | Time | When |
|-----------|------|------|
| Build profile | ~800ms | Once per contact (extraction) |
| Load cached | 0.002ms | Every prompt |
| Classify topic | 0.03ms | Per message |
| Style analysis (old) | 0.22ms | Was every prompt |

## Storage

```
~/.jarvis/profiles/
├── a1b2c3d4e5f6.json   # hashed contact_id
├── f7e8d9c0b1a2.json
└── ...

Each file: ~15KB
100 contacts: ~1.5MB
```

## Quality Comparison

### Topic Detection: Fixed vs Learned

| Message | Fixed Keywords | Learned Topics |
|---------|---------------|----------------|
| "elden ring dlc is insane" | unknown | gaming/elden-ring |
| "paradigms hw due friday" | information | school/paradigms |
| "lakers traded for harden" | unknown | nba/trades |
| "salazar's class is brutal" | unknown | school/salazar |

**Fixed keywords: 31% coverage → Learned: 100% coverage**

### Style Matching

Learned per contact, not generic:

```
Contact A: very_casual, uses "u rn gonna", lots of 😂
Contact B: casual, proper spelling, occasional emoji
Contact C: formal, full sentences, no emoji

LLM adapts response style to match each contact.
```

## Files

| File | Purpose |
|------|---------|
| `jarvis/contact_profile.py` | Main ContactProfile class, StyleProfile, storage |
| `jarvis/topic_discovery.py` | HDBSCAN clustering, centroid computation |
| `jarvis/text_normalizer.py` | Preprocessing (slang, NER integration) |
