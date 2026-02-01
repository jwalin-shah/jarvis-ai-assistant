# JARVIS From-Scratch Rebuild Plan

**Date**: 2026-01-30
**Status**: Ready for execution (after Day 1 validation)

---

## Executive Summary

Rebuild JARVIS with a focused, research-backed architecture:

1. **Extract gold pairs** using Apple's explicit threading (`thread_originator_guid`)
2. **Two-stage retrieval** (FAISS + optional reranker) instead of complex heuristics
3. **Simple LLM generation** only when retrieval confidence is low
4. **Minimal codebase** (~2,500 lines core vs current ~15,000)

**Critical**: This plan requires Day 1 validation before any implementation begins.

---

## ⚠️ Known Risks & Mitigations

This section documents critical risks identified through external review. Each must be addressed.

### Risk 1: "100% Confidence" Gold Pairs is False

**The Assumption**:
> 9,868 gold pairs with 100% confidence from `thread_originator_guid`

**Why It's Wrong**:
- **Sarcastic replies**: "Sure..." meaning NO
- **Context from other channels**: Reply after a phone call; text alone is nonsense
- **Accidental wrong reply**: Clicked wrong message
- **Multi-message context**: Reply to "Movies?" when real context was "Dinner?" + "Movies?"
- **Context blindness**: Them: "Dinner?" → Them: "Movies?" → You (Reply to "Movies?"): "Sure." The bot learns "Movies?" → "Sure" but misses that "Sure" applied to both

**Realistic Confidence**: 85-90%, not 100%

**Mitigation**:
- [ ] Run Day 1 validation: manually review 100 random gold pairs
- [ ] If quality < 85%, redesign extraction before proceeding
- [ ] Store confidence as 0.85-0.95, not 1.0
- [ ] Add context window: store N previous messages with each pair

### Risk 2: Evaluation Too Late (Week 4)

**The Problem**: If gold pairs are garbage or reranker is too slow, you won't know until Week 4 after building the entire infrastructure.

**Mitigation**: Evaluation is Day 1, not Week 4. See [Day 1 Actions](#day-1-actions-before-any-code).

### Risk 3: Latency Trap (2s Generation is DOA)

**The Math**:
```
FAISS (5ms) + Reranker (200ms) + LLM Generation (2s) = ~2.2s latency
```

**The Reality**: In 2.2 seconds, users have already typed "ok" or "sounds good". Smart Reply must be sub-200ms to feel responsive.

**Mitigation**:
- [ ] Template path (FAISS match) must be the default, not exception
- [ ] LLM generation runs **asynchronously** to populate draft buffer
- [ ] Never block request path on LLM generation
- [ ] V1: Skip reranker entirely (add in V2 if needed)

### Risk 4: Semantic Chunking is O(N²) Cold Start

**The Problem**: Embedding 50k-100k+ messages takes hours/days on M-series chips. The plan treats this as a "quick script."

**Mitigation**:
- [ ] Build incremental processing queue (not batch job)
- [ ] Handle messages arriving during index rebuild
- [ ] Process in chunks with progress checkpoints
- [ ] Add estimated time remaining to CLI output

### Risk 5: 16 Features ≠ "Minimal"

**The Problem**: Features 8-16 are each complete projects hidden in a "minimal" plan.

**Mitigation**: V1 includes only 4 core features. Features 5-16 moved to [Future Enhancements](#future-enhancements-post-v1).

### Risk 6: Arbitrary Thresholds Without Calibration

**The Problem**:
```python
if best_score >= 0.65:   # ← Why 0.65?
elif best_score >= 0.45:  # ← Why 0.45?
```

Cross-encoder outputs aren't calibrated probabilities. Thresholds vary by model, corpus, message length.

**Mitigation**:
- [ ] Day 1: Establish baseline metrics before setting thresholds
- [ ] Start with conservative thresholds (template only above 0.85)
- [ ] Collect data on actual score distributions before tuning
- [ ] Build calibration curves from holdout evaluation

### Risk 7: Cross-Encoder May Not Help for Short Text

**The Assumption**: "+20-40% accuracy improvement" from document retrieval benchmarks.

**The Reality**: iMessages are ~15 words average, often literal ("Sure", "What time?"). For short casual text, bi-encoder may be sufficient.

**Mitigation**:
- [ ] V1: FAISS only, no reranker
- [ ] V2: A/B test reranker to measure actual lift
- [ ] Only add complexity if measured improvement > 15%

### Risk 8: Reranker Infrastructure Assumptions

| Model | Problem |
|-------|---------|
| ms-marco-MiniLM-L-12-v2 | PyTorch, not MLX native |
| Qwen3-Reranker-0.6B | May not fit in 8GB with LFM-2.5 loaded |
| FlashRank | Which implementation? CPU/GPU? Quantization? |

**Mitigation**:
- [ ] V1: Skip reranker entirely
- [ ] V2: Benchmark memory usage before adding reranker
- [ ] Test on actual 8GB M2 hardware, not specs

### Risk 9: No Baseline = No Way to Measure Improvement

**The Problem**: "Top-5 accuracy: 73%" means nothing without comparison.

**Mitigation**:
- [ ] Day 1: Measure current system accuracy (even if broken)
- [ ] Define null hypothesis baselines:
  - Random response from history
  - Most common response ("haha", "ok")
  - Current system accuracy
- [ ] All metrics must show: `baseline → new system`

### Risk 10: Cold Start Problem Not Addressed

**The Problem**: New contact has no history, no relationship classification, no style profile.

**Mitigation**:
- [ ] Add `Route → COLD_START` for contacts with < 10 messages
- [ ] Fallback to relationship-type templates (not contact-specific)
- [ ] Prompt user: "I don't know [Name] well yet. Generic response?"
- [ ] Learn quickly from first few interactions

### Risk 11: No "Don't Respond" Route

**The Problem**: Some messages don't need responses:
- Read receipts
- Reactions to your message
- "lol" in group chat
- Announcements requiring no action

**Mitigation**:
- [ ] Add `Route → SKIP` with confidence score
- [ ] Detect: reactions, acknowledgments in group chat, broadcast messages
- [ ] UI shows "No response needed" instead of forcing suggestion

### Risk 12: Human Validation is Unsustainable

**The Problem**: 500+ manual reviews before evaluation. Confirmation bias at review #30.

**Mitigation**:
- [ ] Automate what can be automated (length, timing, pattern checks)
- [ ] Sample-based validation: 50 items, not 500
- [ ] Use inter-rater reliability (get second opinion on 20%)
- [ ] Build validation into normal usage (accept/reject tracking)

### Risk 13: "Delete Everything" is Nuclear

**The Problem**: Those 15k lines contain years of bug fixes, SQLite locking workarounds, edge case handling.

**Mitigation**:
- [ ] Move to `legacy/` instead of delete
- [ ] Reference when hitting familiar bugs
- [ ] Only delete after V1 is stable (3+ weeks)

### Risk 14: Quality Filters Reject Valid Responses

**The Problem**:
```python
r"^(ok|okay|k|kk)$",      # Often the CORRECT response
r"^(yes|no|yeah)$",       # Often the CORRECT response
r"^(Loved|Liked|...)"     # Catches normal sentences too
```

**Mitigation**:
- [ ] Filter on (trigger, response) coherence, not response alone
- [ ] Keep "ok" responses for "Got it?" triggers
- [ ] Use actual tapback detection (attributed body), not regex
- [ ] Review filter rejection rate: if > 40%, filters are too aggressive

### Risk 15: Verification Dashboard Shows Fictional Numbers

**The Problem**: Checkmarks on numbers that haven't been measured.

**Mitigation**:
- [ ] Dashboard shows only measured values
- [ ] Use `--` or `pending` for unmeasured metrics
- [ ] All claimed numbers must have measurement script + date

---

## Day 1 Actions (Before Any Code)

**These must be completed before writing any implementation code.**

### Action 1: Validate Gold Pairs Quality

```bash
# Run the gold pairs query
uv run python -m scripts.count_threaded_pairs --export gold_pairs_sample.json --limit 200

# Manually review 100 random pairs
# For each pair, mark: GOOD / BAD / AMBIGUOUS
# Calculate: good_count / total
```

**Gate**: If quality < 85%, STOP. Redesign extraction approach before proceeding.

**Review checklist for each pair**:
- [ ] Does the response make sense given only the trigger?
- [ ] Is there sarcasm/irony that reverses meaning?
- [ ] Would this response work for a similar trigger from someone else?
- [ ] Is essential context missing (phone call, previous messages)?

### Action 2: Measure Current System Baseline

```bash
# Get accuracy of current (broken) system
uv run python -m scripts.eval_pipeline --baseline --limit 100

# Record:
# - Current top-1 accuracy: ___%
# - Current top-5 accuracy: ___%
# - Current rejection rate: ___%
```

**Purpose**: You need a baseline to prove improvement. Even if current system is broken, measure it.

### Action 3: Define Null Hypothesis Baselines

```python
BASELINES = {
    "random": "Random response from same contact's history",
    "most_common": "Most frequent response ('ok', 'sounds good', etc.)",
    "current_system": "Existing JARVIS implementation",
    "no_system": "User types themselves (0% suggestion acceptance)",
}
```

Run each baseline on 100 holdout pairs, record accuracy.

### Action 4: Validate Infrastructure Assumptions

```bash
# Check FAISS works
uv run python -c "import faiss; print('FAISS OK')"

# Check embedding memory
uv run python -c "
from models.embeddings import get_mlx_embedder
e = get_mlx_embedder()
import psutil
print(f'Memory after embedder: {psutil.Process().memory_info().rss / 1e9:.2f} GB')
"

# Check total memory budget
# Must have room for: embedder + LLM + FAISS index + working memory
```

### Action 5: Count Actual Message Volume

```bash
# How many messages need processing?
uv run python -c "
from integrations.imessage.reader import ChatDBReader
reader = ChatDBReader()
# Count total messages
# Count messages per contact (distribution)
# Estimate embedding time at 100 msg/sec
"
```

**Gate**: If > 100k messages and no incremental processing plan, add it before proceeding.

---

## V1 Scope (4 Core Features Only)

V1 is intentionally minimal. Additional features are in [Future Enhancements](#future-enhancements-post-v1).

### Feature 1: Gold Pair Extraction

Extract pairs using `thread_originator_guid` with quality filters.

**Includes**:
- SQL extraction from iMessage database
- Quality filters (length, coherence, not reactions)
- Confidence scoring (0.85-0.95, not 1.0)
- Context storage (N previous messages per pair)

**Does NOT include**:
- Semantic topic segmentation (Future V2)
- Relationship classification (Future V2)
- Intent classification (Future V2)

### Feature 2: FAISS Retrieval (No Reranker)

Fast vector search for similar triggers.

**Includes**:
- Embed triggers with bge-small-en-v1.5
- FAISS index build and search
- Return top-K candidates with scores
- Incremental index updates

**Does NOT include**:
- Cross-encoder reranking (Future V2)
- Two-stage pipeline (Future V2)

### Feature 3: Simple Router

Route to template, generate, or skip.

**Includes**:
- Template path: high similarity → return past response
- Generate path: medium similarity → async LLM draft
- Skip path: very low similarity or no-response-needed patterns
- Cold start path: new contact fallback

**Routing thresholds (conservative start)**:
```python
TEMPLATE_THRESHOLD = 0.85  # Very high confidence required
GENERATE_THRESHOLD = 0.50  # Medium confidence → LLM
SKIP_THRESHOLD = 0.30      # Very low → don't suggest
```

### Feature 4: Minimal API

Essential endpoints for desktop app.

**Includes**:
- `GET /health` - System status
- `GET /conversations` - List conversations
- `POST /drafts/smart-reply` - Get suggestion (non-blocking)
- `GET /drafts/{id}` - Poll for async draft result

**Does NOT include**:
- Export endpoints (Future)
- Metrics endpoints (Future)
- Search endpoints (Future)

---

## V1 Architecture

```
User sends message
        │
        ▼
┌───────────────────┐
│  Cold Start Check │  ← Contact has < 10 messages?
└─────────┬─────────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌────────┐  ┌───────────────┐
│ COLD   │  │  FAISS Search │  ← Embed query, search index
│ START  │  │  (top 10, ~5ms)│
└────────┘  └───────┬───────┘
    │               │
    ▼               ▼
┌────────┐  ┌───────────────────┐
│Generic │  │   Route by Score  │
│Response│  ├───────────────────┤
└────────┘  │ >= 0.85 → Template│  Return best match response
            │ 0.50-0.85 → LLM   │  Async generate with examples
            │ < 0.50 → Skip     │  No suggestion (or clarify)
            └───────────────────┘
```

---

## V1 Implementation Phases

### Phase 0: Day 1 Validation (Today)

**Goal**: Validate assumptions before writing code

- [ ] Complete [Day 1 Actions](#day-1-actions-before-any-code)
- [ ] Document results in `docs/DAY1_VALIDATION.md`
- [ ] Go/No-Go decision based on gold pair quality

**Deliverable**: Written validation report with metrics

**Gate**: Gold pair quality >= 85% AND baseline established

### Phase 1: Foundation (Days 2-4)

**Goal**: Clean slate with gold pair extraction

- [ ] Move `mcp_server/`, `jarvis/prompts/`, `scripts/experiments/` to `legacy/`
- [ ] Create minimal CLI (~300 lines): serve, db, health
- [ ] Implement `extraction/threaded.py` using `thread_originator_guid`
- [ ] Add context storage (5 previous messages per pair)
- [ ] Export gold pairs to database with confidence scores

**Deliverable**: `jarvis db extract-gold` produces pairs with measured quality

**Verification**:
```bash
jarvis db stats
# Shows: X pairs extracted, Y% passed quality filters
# Compare to Day 1 manual review results
```

### Phase 2: Retrieval (Days 5-7)

**Goal**: Fast, accurate retrieval (FAISS only, no reranker)

- [ ] Implement `retrieval/embedder.py` (use existing MLX embeddings)
- [ ] Implement `retrieval/index.py` (simplified FAISS wrapper)
- [ ] Build index from gold pairs
- [ ] Add incremental update support

**Deliverable**: Query → top 10 relevant (trigger, response) pairs in <50ms

**Verification**:
```bash
jarvis db build-index
jarvis retrieval test "Want to grab lunch?"
# Shows: top 10 matches with scores and latency
```

### Phase 3: Router + API (Days 8-10)

**Goal**: Smart routing with async generation

- [ ] Implement simple `router.py` with 4 routes (template/generate/skip/cold)
- [ ] Implement async generation (non-blocking, polls for result)
- [ ] Wire up minimal API endpoints
- [ ] Add skip detection patterns

**Deliverable**: `POST /drafts/smart-reply` returns appropriate response

**Verification**:
```bash
curl -X POST http://localhost:8000/drafts/smart-reply \
  -d '{"message": "Want to grab lunch?", "contact": "John"}'
# Returns: route taken, suggestion (or skip reason), latency
```

### Phase 4: Evaluation & Calibration (Days 11-14)

**Goal**: Measure accuracy, tune thresholds

- [ ] Create holdout set (20% of gold pairs, by contact)
- [ ] Run retrieval accuracy evaluation
- [ ] Compare to Day 1 baselines
- [ ] Tune thresholds based on actual score distributions
- [ ] Document results

**Deliverable**: Measured accuracy with comparison to baselines

**Verification**:
```bash
jarvis eval --holdout
# Shows:
#   Baseline (random): 5%
#   Baseline (current): 32%
#   New system (top-1): 58%
#   New system (top-5): 74%
```

---

## What to Move to `legacy/`

**Do NOT delete. Move to `legacy/` folder for reference.**

| Component | Lines | Reason |
|-----------|-------|--------|
| `mcp_server/` | 1,412 | Not needed for V1 |
| `jarvis/prompts/` package | 900 | Replaced by minimal prompts |
| `jarvis/_prompts.py` | 1,940 | Replaced by retrieved pairs |
| `jarvis/_cli_main.py` | 3,151 | Replaced by minimal CLI |
| `jarvis/cli/` package | 637 | Replaced by minimal CLI |
| `scripts/experiments/` | 9,193 | Reference for failed approaches |

**Move command**:
```bash
mkdir -p legacy
mv mcp_server/ legacy/
mv jarvis/prompts/ legacy/jarvis_prompts/
mv jarvis/_prompts.py legacy/
mv jarvis/_cli_main.py legacy/
mv jarvis/cli/ legacy/jarvis_cli/
mv scripts/experiments/ legacy/
```

## What to Keep

| Component | Lines | Why |
|-----------|-------|-----|
| `integrations/imessage/` | ~2,000 | Solid iMessage reading |
| `models/loader.py` | ~400 | MLX model loading |
| `models/embeddings.py` | ~300 | MLX embeddings |
| `jarvis/db.py` | ~1,800 | Database (needs cleanup) |
| `jarvis/index.py` | ~400 | FAISS index (keep core) |
| `core/memory/` | ~300 | Memory controller |
| `core/health/` | ~400 | Health monitoring |
| `benchmarks/` | ~1,500 | Validation gates |

---

## V1 Directory Structure

```
jarvis-ai-assistant/
│
├── jarvis/                          # Core application (~1,500 lines for V1)
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py                       # Minimal CLI (300 lines)
│   ├── config.py                    # Config (200 lines)
│   ├── errors.py                    # Errors (100 lines)
│   ├── db.py                        # Database (500 lines)
│   │
│   ├── extraction/                  # Pair extraction (~300 lines)
│   │   ├── __init__.py
│   │   ├── threaded.py              # Gold pairs from thread_originator_guid
│   │   └── filters.py               # Quality filters
│   │
│   ├── retrieval/                   # FAISS retrieval (~300 lines)
│   │   ├── __init__.py
│   │   ├── embedder.py              # Bi-encoder wrapper
│   │   └── index.py                 # FAISS index
│   │
│   └── router.py                    # Route: template/generate/skip/cold (200 lines)
│
├── api/                             # Minimal API (~400 lines)
│   ├── __init__.py
│   ├── main.py
│   └── routers/
│       ├── health.py
│       └── drafts.py                # Main endpoint
│
├── integrations/imessage/           # Keep as-is
├── models/                          # Keep as-is
├── core/                            # Keep as-is
├── benchmarks/                      # Keep as-is
├── contracts/                       # Keep as-is
├── tests/                           # Add V1 tests
├── legacy/                          # Moved old code (reference only)
└── docs/
    ├── FROM_SCRATCH_PLAN.md         # This plan
    ├── DAY1_VALIDATION.md           # Day 1 results (to be created)
    └── EVALUATION_AND_KNOWN_ISSUES.md
```

---

## V1 Success Metrics

| Metric | Target | Baseline | How to Measure |
|--------|--------|----------|----------------|
| Gold pairs extracted | 8,000+ | N/A | `jarvis db stats` |
| Gold pair quality | >= 85% | Day 1 manual review | Sample validation |
| Retrieval accuracy (top-5) | > baseline + 20% | Day 1 baseline | Holdout evaluation |
| Response latency (template) | < 50ms | N/A | Benchmark |
| Response latency (generate) | < 3s (async) | N/A | Benchmark |
| Skip accuracy | > 80% | N/A | Manual review of skips |
| Cold start handled | 100% | 0% (current crashes) | Test new contacts |
| Codebase size | < 3,000 lines | ~15,000 | `wc -l` |

---

## Key Research Findings (Reference)

### 1. The Current Extraction is Broken

**Problem**: The existing 40,061 pairs are corrupted due to:
- Substring matching bug (incorrectly associates triggers with responses)
- No topic boundaries (creates cross-topic nonsense pairs)
- Ignores Apple's explicit threading

**Solution**: Rebuild extraction from scratch using `thread_originator_guid`

### 2. Apple Provides Threading Data (Not "Ground Truth")

Since iOS 14, when someone uses the "reply" feature in iMessage, Apple stores an explicit link:

```sql
SELECT
    original.text AS trigger,
    reply.text AS response
FROM message reply
JOIN message original ON reply.thread_originator_guid = original.guid
WHERE reply.is_from_me = 1 AND original.is_from_me = 0
```

**Your data**: ~9,868 threaded pairs (them→me)
**Realistic confidence**: 85-95% (NOT 100%)

### 3. Two-Stage Retrieval is Industry Standard

| Stage | Purpose | Latency |
|-------|---------|---------|
| **Stage 1: FAISS** | Fast vector search, retrieve top 50 candidates | ~5ms |
| **Stage 2: Reranker** | Cross-encoder scoring, return top 5 | ~50-200ms |

**Note**: V1 uses Stage 1 only. Stage 2 added in V2 if measured improvement > 15%.

### 4. Quality Filters

```python
QUALITY_FILTERS = {
    # Length constraints
    "trigger_min_chars": 3,
    "trigger_max_chars": 500,
    "response_min_chars": 2,
    "response_max_chars": 1000,

    # Semantic filters (NOT pattern-based rejection of valid responses)
    "min_semantic_coherence": 0.2,  # Trigger-response must be somewhat related

    # Actual tapback detection (use attributed_body, not regex)
    "use_tapback_detection": True,
}
```

**Important**: Don't reject "ok", "yes", "no" responses—they're often correct. Filter on (trigger, response) coherence instead.

---

## Future Enhancements (Post-V1)

These features are explicitly OUT OF SCOPE for V1. Each is a separate project.

### V2 Features (After V1 Stable)

#### Feature: Two-Stage Retrieval with Reranker

**Prerequisite**: V1 accuracy measured, baseline established

**What it adds**:
- Cross-encoder reranking (ms-marco-MiniLM or Qwen3-0.6B)
- Two-stage pipeline (FAISS → Reranker)
- Potentially +20-40% accuracy

**When to add**: Only if V1 accuracy is insufficient AND memory budget allows

**How to validate**:
```bash
# A/B test: FAISS-only vs FAISS+Reranker
# Measure: accuracy lift, latency cost, memory cost
# Gate: Add only if lift > 15% AND latency < 200ms
```

#### Feature: Semantic Topic Segmentation

**Prerequisite**: Gold pairs extracted and validated

**What it adds**:
- Chunk conversations by meaning (not time gaps)
- Extract "silver" pairs from non-threaded messages
- Potentially 2-3x more training data

**When to add**: After V1 if gold pairs alone are insufficient

**Implementation notes**:
- Must be incremental (not batch)
- Handle messages arriving during processing
- Add progress checkpoints and resume capability

#### Feature: Intent Classification

**What it adds**:
- Classify incoming message intent (question, invitation, logistics, etc.)
- Filter retrieval by intent match
- Better routing decisions

**When to add**: V2, after basic routing is working

#### Feature: Relationship Classification

**What it adds**:
- Classify contacts (friend, family, coworker, etc.)
- Style adaptation per relationship type
- Better cold start handling

**When to add**: V2, with user confirmation workflow

### V3 Features (Future Projects)

Each of these is a complete project requiring its own planning:

| Feature | What It Does | Complexity |
|---------|--------------|------------|
| Commitment Tracking | Detect promises made | High (NLP extraction) |
| Response Time Modeling | Learn reply speed patterns | Medium (statistics) |
| Tone/Energy Matching | Match incoming message energy | Medium (feature engineering) |
| Conversation Memory | Remember recent context | High (state management) |
| Learning from Rejections | Improve from rejected suggestions | High (feedback loop) |
| Urgency Detection | Flag urgent messages | Medium (classification) |
| Relationship Health | Detect drifting relationships | High (time series) |
| Group Chat Dynamics | Understand group behavior | Very High (multi-party) |
| Style Drift Over Time | Weight recent messages more | Medium (recency decay) |

**Rule**: Do not start V3 features until V2 is stable and measured.

---

## Appendix: Gold Pairs Query

```sql
-- Extract gold pairs from explicit threading
SELECT
    original.text AS trigger_text,
    reply.text AS response_text,
    original.date AS trigger_date,
    reply.date AS response_date,
    original.is_from_me AS trigger_from_me,
    reply.is_from_me AS response_from_me,
    cmj.chat_id
FROM message reply
JOIN message original ON reply.thread_originator_guid = original.guid
LEFT JOIN chat_message_join cmj ON reply.ROWID = cmj.message_id
WHERE reply.text IS NOT NULL
  AND reply.text != ''
  AND original.text IS NOT NULL
  AND original.text != ''
  -- them→me pairs (most useful for suggestions)
  AND original.is_from_me = 0
  AND reply.is_from_me = 1
```

---

## Appendix: Clarification Responses (V1 Minimal Set)

```python
# Only handle the most obvious vague patterns
# Don't over-engineer—most messages don't need clarification

CLARIFICATION_RESPONSES = {
    r"^hey\??$": "Hey! What's up?",
    r"^\?+$": "What's the question?",
    r"^(you there|hello)\??$": "Yeah, what's up?",
}

# Default for very low similarity: return SKIP, not clarification
# User can type their own response
```

---

## Appendix: Cold Start Handling

```python
def handle_cold_start(contact_id: str, message: str) -> Response:
    """Handle contacts with insufficient history."""

    message_count = db.count_messages(contact_id)

    if message_count < 10:
        return Response(
            route="COLD_START",
            suggestion=None,
            explanation=f"I don't know {contact_name} well yet (only {message_count} messages).",
            fallback_options=[
                "Generic friendly response",
                "Skip suggestion",
            ]
        )

    # Enough history, proceed normally
    return None
```

---

## Appendix: Async Generation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    ASYNC GENERATION (V1)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  POST /drafts/smart-reply                                       │
│       │                                                         │
│       ├── Route = TEMPLATE (>= 0.85)                           │
│       │   └── Return immediately with suggestion               │
│       │                                                         │
│       ├── Route = GENERATE (0.50 - 0.85)                       │
│       │   ├── Return immediately: {"status": "generating", "id": "abc123"} │
│       │   └── Background: queue LLM generation                 │
│       │                                                         │
│       ├── Route = SKIP (< 0.50)                                │
│       │   └── Return immediately: {"status": "skip", "reason": "low confidence"} │
│       │                                                         │
│       └── Route = COLD_START                                   │
│           └── Return immediately: {"status": "cold_start", "message_count": 5} │
│                                                                 │
│  GET /drafts/{id}                                              │
│       └── Poll for generation result                           │
│           ├── {"status": "generating"}                         │
│           └── {"status": "complete", "suggestion": "..."}      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key**: Never block on LLM generation. Return immediately, let client poll.

---

## Implementation Progress (2026-01-31)

### Phase 1: Data Extraction - COMPLETE

**Threaded Conversations (Apple's reply feature)**:
- Script: `scripts/extract_threaded_conversations.py`
- Output: `threaded_conversations.jsonl` - 17,918 pairs (23 MB)
- Quality: ~100% confidence (explicit reply threading)

**Semantic Chunking (contextual pairs)**:
- Script: `scripts/extract_semantic_conversations.py`
- Output: `semantic_conversations.jsonl` - 122,567 pairs (344 MB)
- Method: Chunk conversations by semantic similarity (embedding distance), not time gaps
- Each record includes:
  - `context` - List of up to 50 prior messages with timestamps
  - `context_formatted` - "Me: X\nThem: Y\n..." format for LLM prompts
  - `immediate_trigger` - The message triggering my response (what we embed)
  - `my_response` - My actual reply
  - `is_group` - Group chat flag

**Performance**: MLX embeddings encode ~666 texts/second on Apple Silicon.

### Phase 2: Merge & Index - COMPLETE

**Data Pipeline** (`scripts/build_training_index.py`):
1. Merge threaded + semantic files
2. Deduplicate by hash of (trigger, response)
3. Filter short/empty pairs
4. Import to JarvisDB

**Merge Stats**:
```
Total records: 140,485
Duplicates:      773
Short trigger:  5,375
Short response: 2,262
Unique pairs: 132,073
```

**Train/Test Split**:
- Method: Random split by pairs (NOT by contact - see below)
- Split: 105,659 training / 26,414 holdout (80/20)
- Stored in `is_holdout` column in `pairs` table

**Why random split instead of by-contact**:
The original plan was 20% of *contacts* for holdout, but data is heavily skewed:
- Top contact: 14,059 pairs
- Median contact: ~200 pairs
- 20% of contacts = only 4% of pairs

Random pair split gives true 80/20 ratio and tests pattern matching accuracy.

**FAISS Index**:
- 100,000 vectors indexed (filtered by quality)
- 384 dimensions (bge-small-en-v1.5)
- 146.5 MB index file
- Version: `20260131-101355`
- Build time: ~2.5 minutes (MLX accelerated)

### Initial Retrieval Test Results

Sample queries against training index (self-retrieval):

| Query | Top Score | Top Match |
|-------|-----------|-----------|
| "hey what's up" | 1.000 | Exact match found |
| "sounds good to me" | 1.000 | Exact match found |
| "are you coming tonight?" | 0.982 | Near-exact match |
| "did you see the game last night?" | 0.887 | Semantic match |

These high scores are expected for self-retrieval. Holdout evaluation will show real performance.

### Holdout Evaluation Results (2026-01-31)

**Full evaluation on 26,414 holdout pairs:**

| Metric | Value |
|--------|-------|
| Pairs evaluated | 26,414 |
| Semantic match rate (≥0.5) | **100.0%** |
| Exact response match | 0.2% (expected) |
| Mean top score | **0.877** |
| Median top score | 0.866 |
| 90th percentile | 1.000 |
| 10th percentile | 0.779 |

**Score Distribution:**
```
0.9+    : 35.9% ████████████████████
0.8-0.9 : 46.1% █████████████████████████
0.7-0.8 : 17.6% ██████████
0.6-0.7 :  0.4%
<0.6    :  0.0%
```

**Performance:**
- Encode: 42s for 26k queries (629 texts/sec)
- Search: 2.2s for 26k queries (12,079 queries/sec)
- Total: 44 seconds

**Key Finding**: The index has extremely high coverage. For virtually any incoming message, we can find a semantically similar trigger in the training set.

**But**: High trigger similarity ≠ appropriate response. Response similarity analysis showed:

| Trigger Score | Response Similarity | Implication |
|---------------|---------------------|-------------|
| 0.9+ | 0.564 | Even perfect trigger match → only 56% appropriate |
| 0.8-0.9 | 0.558 | Barely different from lower scores |
| 0.7-0.8 | 0.546 | |
| 0.6-0.7 | 0.541 | |

**Critical Insight**: Trigger matching alone cannot determine response appropriateness. The same trigger can have different appropriate responses based on:
- **Context** - What was discussed before
- **Relationship** - Friend vs coworker vs family
- **Intent** - Question vs invitation vs acknowledgment
- **Time** - Same question, different day = different answer

**Strategy Change**: Use retrieval for EXAMPLES, not direct responses
- FAISS finds similar patterns → use as few-shot examples for LLM
- Direct template responses disabled (threshold raised to 0.95)
- LLM generates contextually appropriate response using examples

### Response Type Clustering Experiment (2026-01-31)

**Goal**: Discover natural response types (agree, decline, defer, etc.) from data.

**Method**: UMAP + HDBSCAN pipeline (same as BERTopic)
1. Embed 105k training responses (bge-small, 384 dims)
2. UMAP reduce to 5 dims (cosine metric)
3. HDBSCAN cluster (min_cluster_size=100)

**Results**:
```
Clusters found: 240
Noise (outliers): 43% (45,411 responses)
Largest cluster: 2.5% (2,668 responses)
```

**Key Finding**: Semantic clustering finds **TOPICS**, not response **TYPES**:

| Cluster | Size | What it found | Category |
|---------|------|---------------|----------|
| 73 | 2,668 | Food/eating discussions | TOPIC |
| 55 | 1,813 | iMessage reactions ("Loved", "Liked") | TYPE ✓ |
| 233 | 1,765 | Games/sports | TOPIC |
| 63 | 1,305 | Sleep-related | TOPIC |
| 216 | 1,073 | Numbers/times | TOPIC |
| 203 | 733 | Dating/relationships | TOPIC |
| 198 | 716 | Scheduling (days of week) | TOPIC |
| 221 | 642 | "Don't wanna" negations | TYPE ✓ |

**Insight**: Embeddings encode SEMANTIC MEANING (what you're talking about), not FUNCTIONAL TYPE (how you're responding). "Yeah let's eat" and "No I'm not hungry" are both about FOOD, so they cluster together despite being opposite response types.

**Implications for Response Generation**:

Instead of: `classify_response_type() → generate(type)`

Consider: `get_topic() + get_response_options(topic)`
- Topic: "food invitation"
- Options: ["Yeah I'm down!", "Not hungry rn", "Maybe later?"]

Or use **structural features** for type classification:
- Starts with "yeah/yes/sure" → AGREE
- Starts with "no/can't/sorry" → DECLINE
- Ends with "?" → QUESTION
- Contains "maybe/idk/let me check" → DEFER

**Performance Notes**:
- Embedding: 760 texts/sec (MLX GPU-accelerated)
- UMAP: ~2 min for 105k vectors (CPU-only, no GPU acceleration on Mac)
- HDBSCAN: <1 sec (CPU)
- Memory: ~1.1GB peak for 105k × 384 embeddings

**Files Created**:
- `~/.jarvis/response_clusters/response_embeddings.npy` (155MB) - raw embeddings
- `~/.jarvis/response_clusters/reduced_embeddings.npy` - UMAP 5D
- `~/.jarvis/response_clusters/hdbscan_labels.npy` - cluster assignments

### Dialogue Act Classification Analysis (2026-01-31)

**Goal**: Classify WHAT TYPE of trigger/response, not just semantic similarity.

**Approach**: k-NN classifier over SWDA (Switchboard Dialog Act corpus) + iMessage-specific exemplars.

**Script**: `scripts/build_da_classifier.py`

**Categories Defined**:

| Trigger Types | Response Types |
|---------------|----------------|
| INVITATION | AGREE |
| YN_QUESTION | DECLINE |
| WH_QUESTION | DEFER |
| INFO_STATEMENT | ACKNOWLEDGE |
| OPINION | ANSWER |
| REQUEST | QUESTION |
| GOOD_NEWS | REACT_POSITIVE |
| BAD_NEWS | REACT_SYMPATHY |
| GREETING | STATEMENT |
| ACKNOWLEDGE | GREETING |

**Distribution Problem (Our Data)**:

```
TRIGGER TYPES                    RESPONSE TYPES
INFO_STATEMENT    67.9%          STATEMENT       77.7%
ACKNOWLEDGE       14.8%          ACKNOWLEDGE      6.2%
OPINION            8.4%          QUESTION         5.7%
WH_QUESTION        3.5%          REACT_POSITIVE   5.0%
YN_QUESTION        2.8%          AGREE            2.5%
REQUEST            1.3%          ...
```

**Key Finding**: STATEMENT is a catch-all (77.7% of responses). This is problematic because:
1. SWDA maps `sd` (statement-non-opinion) → STATEMENT
2. Many actual ANSWERs get classified as STATEMENT ("2pm" → STATEMENT, should be ANSWER)
3. AGREE/ACKNOWLEDGE distinction is unclear

**Cross-Validation with HDBSCAN Clusters**:

We compared natural HDBSCAN clusters with DA classifier labels:

| Cluster (HDBSCAN) | DA Classifier Says | Mismatch? |
|-------------------|-------------------|-----------|
| "ACKNOWLEDGE" cluster | 77% REACT_POSITIVE | ⚠️ |
| "DECLINE" cluster | 64% AGREE | ❌ Major! |
| "AGREE_ELABORATE" cluster | 39% GREETING | ❌ |
| "AGREE_SHORT" cluster | 85% ACKNOWLEDGE | ~ Close |

**Root Cause**: Two issues:
1. **Heuristic cluster naming is bad** - `suggest_cluster_names()` uses keyword patterns that don't match content
2. **DA classifier disagrees with natural clusters** - SWDA patterns don't match iMessage patterns

**Coherence Analysis** (Semantic Similarity Check):

Even though DA types seem wrong, semantic coherence is actually decent:

| Response DA Type | Avg Trigger-Response Similarity | Low Similarity (<0.4) |
|------------------|-------------------------------|----------------------|
| STATEMENT | 0.59 | 0% |
| ACKNOWLEDGE | 0.60 | 0% |
| AGREE | 0.63 | 0% |
| ANSWER | 0.66 | 0% |

**Insight**: Pairs are semantically coherent (related topics), but DA labels are wrong.

When we look at "mismatched" pairs (STATEMENT when other type expected):

| Similarity | DA Mismatch | Interpretation |
|------------|-------------|----------------|
| High (>0.6) | Yes | **Misclassification** - response is good, DA label is wrong |
| Low (<0.5) | Yes | **Bad pair** - actual topic shift, should filter |

**Examples of Misclassification**:
```
"at what?" → "Everything"     [sim=0.63] Should be ANSWER, got STATEMENT
"Forward me email" → "Sent"   [sim=0.81] Should be AGREE, got STATEMENT
"im free"                     Should be AGREE, got STATEMENT
```

**Examples of Bad Coherence**:
```
"How's work?" → "Picnic tomorrow?" [sim=0.45] Topic shift, bad pair
"hello" → "what size jersey?"      [sim=0.52] Topic shift, bad pair
```

**Database Schema Update** (v7):

Added columns to persist DA classifications:
```sql
trigger_da_type TEXT,      -- e.g., WH_QUESTION, INFO_STATEMENT
trigger_da_conf REAL,      -- Classifier confidence 0-1
response_da_type TEXT,     -- e.g., STATEMENT, AGREE
response_da_conf REAL,     -- Classifier confidence 0-1
cluster_id INTEGER,        -- HDBSCAN cluster (-1 for noise)
```

**New Methods**:
- `db.update_da_classifications()` - Bulk update DA labels
- `db.update_cluster_assignments()` - Bulk update cluster IDs
- `db.get_da_distribution()` - Get DA type counts

**Script**: `scripts/populate_da_and_clusters.py` - Populates all 105k pairs (~300 pairs/sec)

**Next Steps**:
1. Add better exemplars to DA classifier (from actual data, not just SWDA)
2. Use cluster centroids as new exemplars
3. Combine DA type + semantic similarity for quality scoring
4. Consider training a simple classifier on our labeled data

### Combined Classification Architecture (2026-01-31)

**Key Insight**: We have four complementary classification systems:

| System | Classifies | Method | Best For |
|--------|------------|--------|----------|
| DA Classifier | Functional type (INVITATION, AGREE, DECLINE) | k-NN over exemplars | Understanding WHAT response is needed |
| HDBSCAN Clusters | Response topics (food, games, sleep) | UMAP + HDBSCAN | Discovering WHAT you talk about |
| Intent Classifier | User commands (REPLY, SUMMARIZE, SEARCH) | Centroid similarity | Routing user requests |
| FAISS Index | Trigger similarity | Vector search | Finding similar past exchanges |

**Critical Realization**: Semantic clustering finds TOPICS, not response TYPES.
- "Yeah let's eat" and "No I'm not hungry" cluster together (FOOD topic)
- Despite being OPPOSITE response types (AGREE vs DECLINE)
- Embeddings encode semantic meaning, not functional intent

**Proposed Pipeline**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     INCOMING MESSAGE                                 │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Message Understanding (parallel)                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │  DA CLASSIFIER  │  │  TOPIC CLUSTER  │  │  FAISS SEARCH   │     │
│  │  (trigger type) │  │  (semantic)     │  │  (similar pairs)│     │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘     │
│           ▼                    ▼                    ▼               │
│  "INVITATION"           "FOOD/DINING"        Top-5 similar         │
│  conf: 0.85             cluster: 73          triggers + responses  │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Response Strategy Selection                               │
├─────────────────────────────────────────────────────────────────────┤
│  DA Type: INVITATION → Valid: AGREE, DECLINE, DEFER, QUESTION       │
│  Topic: FOOD/DINING → Filter FAISS to food-related examples         │
│  Similar pairs (filtered):                                          │
│  ├── "lunch?" → "Yeah I'm down!" (AGREE)                           │
│  ├── "dinner?" → "Can't tonight" (DECLINE)                         │
│  └── "brunch?" → "Maybe, let me check" (DEFER)                     │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3: Multi-Option Generation                                   │
├─────────────────────────────────────────────────────────────────────┤
│  For trigger_da=INVITATION, generate 3 response options:            │
│  Option 1 (AGREE):  "Yeah I'm down! What time?"                    │
│  Option 2 (DECLINE): "Can't today, swamped with work"              │
│  Option 3 (DEFER):  "Maybe, let me check my sched"                 │
│  Each option uses same-type examples from FAISS as few-shot        │
└─────────────────────────────────────────────────────────────────────┘
```

**Implementation Components**:

1. **DA-Filtered Retrieval**: Filter FAISS results by `response_da_type` to get functionally relevant examples
2. **Topic-Aware Generation**: Use `cluster_id` to add topic context to prompts
3. **Multi-Option Generation**: Generate one response per valid response type (AGREE/DECLINE/DEFER)
4. **Hybrid Classification**: Combine semantic clusters (topics) + structural features (starts with "yeah" → AGREE)

**Key Functions to Implement**:

```python
def get_examples_by_response_type(trigger, target_da, faiss_results):
    """Filter FAISS results to only those matching target response DA."""
    return [r for r in faiss_results if r["response_da_type"] == target_da][:5]

def generate_response_options(trigger, trigger_da, faiss_results, contact):
    """Generate one response option per valid response type."""
    valid_types = TRIGGER_TO_VALID_RESPONSES.get(trigger_da, ["STATEMENT"])
    options = []
    for response_type in valid_types[:3]:
        examples = get_examples_by_response_type(trigger, response_type, faiss_results)
        response = generator.generate(build_typed_prompt(trigger, response_type, examples))
        options.append({"type": response_type, "response": response})
    return options
```

**Cluster Purity Analysis** (to validate approach):

```sql
-- If cluster purity > 70% → clusters can infer response type
-- If cluster purity < 70% → use DA classifier for types, clusters for topics only
SELECT cluster_id, response_da_type, COUNT(*) as count,
       100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY cluster_id) as purity
FROM pairs WHERE cluster_id IS NOT NULL AND response_da_type IS NOT NULL
GROUP BY cluster_id, response_da_type ORDER BY cluster_id, count DESC;
```

**Files Created/Modified**:
- `jarvis/db.py` - Added DA columns (trigger_da_type, response_da_type, cluster_id)
- `scripts/populate_da_and_clusters.py` - Batch populate DA + cluster fields
- `scripts/build_da_classifier.py` - k-NN DA classifier with SWDA + manual exemplars

### Updated Thresholds

Based on overnight evaluation (2026-01-30), lower thresholds perform better:

```python
# Original (conservative)
TEMPLATE_THRESHOLD = 0.90  # Too strict, missed good matches
GENERATE_THRESHOLD = 0.50

# Updated (from evaluation)
TEMPLATE_THRESHOLD = 0.65  # More matches, still high quality
GENERATE_THRESHOLD = 0.45  # Catch more borderline cases
```

See `docs/PLAN.md` Phase 2 for threshold tuning rationale.

### Scripts Created

| Script | Purpose |
|--------|---------|
| `scripts/extract_threaded_conversations.py` | Extract pairs from Apple's reply threading |
| `scripts/extract_semantic_conversations.py` | Extract pairs using semantic chunking |
| `scripts/build_training_index.py` | Merge, dedupe, import, build FAISS |
| `scripts/evaluate_retrieval.py` | Batched evaluation on holdout set |

### Multi-Option Response Strategy (Key Insight)

**Problem**: System can't know user's answer to commitment questions ("Want to grab lunch?")

**Solution**: Generate multiple response OPTIONS, let user pick:

```
"Want to grab lunch?"
├── Option 1: "Yeah I'm down! What time?"      (AGREE)
├── Option 2: "Can't today, swamped with work" (DECLINE)
└── Option 3: "Maybe, let me check my sched"   (DEFER)
```

**Response Type Templates by Message Category**:

| Incoming Type | Option 1 | Option 2 | Option 3 |
|---------------|----------|----------|----------|
| Invitation | Agree + enthusiasm | Decline + reason | Defer + check |
| Yes/No Question | Affirmative | Negative | Need more info |
| Opinion Request | Positive take | Neutral/unsure | Ask for context |
| Plan Question | Suggest time | Can't commit | Defer decision |
| Good News | Congrats + question | - | - |
| Bad News | Sympathy + offer help | - | - |

This is how Gmail Smart Reply works - 3 semantically diverse options.

### Next Steps: Rebuild Classifiers from Scratch

Based on our findings (trigger similarity ≠ response appropriateness), we need classifiers that determine:
1. **Does this need context?** - Can't answer "what time?" without knowing what event
2. **What's the intent?** - Question, invitation, acknowledgment, etc.
3. **What's the relationship?** - Affects tone and formality
4. **Should we skip?** - Some messages don't need responses

**Approach**: Test each classifier individually before integration.

#### Phase 3.1: Context Requirement Classifier (NEXT)

**Goal**: Detect messages that need prior context to respond appropriately.

**Categories**:
- `STANDALONE` - Can respond with just this message ("Want to grab lunch?")
- `NEEDS_CONTEXT` - Requires prior messages ("What time?", "Where?", "Which one?")
- `REFERENCE` - References something specific ("That's hilarious", "Sounds good")

**Training Data**: Use our 132k pairs
- Pairs where `context_formatted` is essential → NEEDS_CONTEXT
- Pairs where trigger is self-contained → STANDALONE

**Test**: Evaluate on holdout, measure if context detection improves response quality.

#### Phase 3.2: Intent Classifier

**Goal**: Classify what kind of response is needed.

**Categories** (simplified from current):
- `QUESTION` - Needs informational answer
- `INVITATION` - Needs RSVP (yes/no/maybe + details)
- `STATEMENT` - May need acknowledgment or follow-up
- `ACKNOWLEDGMENT` - Usually doesn't need response
- `REQUEST` - Needs action or commitment

**Training Data**: Label subset of 132k pairs by intent, train classifier.

**Test**: Measure intent accuracy on labeled holdout.

#### Phase 3.3: Relationship Classifier

**Goal**: Determine relationship type for tone adaptation.

**Categories**:
- `CLOSE_FRIEND` - Casual, jokes, slang OK
- `FAMILY` - Warm but context-dependent
- `ACQUAINTANCE` - Friendly but more formal
- `PROFESSIONAL` - Formal, no slang

**Training Data**: Use chat patterns from our data:
- Message frequency, response times, emoji usage, message length variance
- Cluster contacts by communication patterns

**Test**: Manual review of classifications, measure if tone adaptation improves quality.

#### Phase 3.4: Skip Classifier

**Goal**: Detect messages that don't need responses.

**Categories**:
- `NEEDS_RESPONSE` - Should suggest a reply
- `OPTIONAL` - Could respond but not necessary (reactions, "lol" in group)
- `NO_RESPONSE` - Definitely don't respond (read receipts, typing indicators)

**Training Data**: Analyze our pairs - which triggers got responses vs didn't.

**Test**: Measure precision (don't skip when response needed).

#### Phase 3.5: Integration & End-to-End Test

Only after each classifier passes individual testing:
1. Wire classifiers into router
2. Test full pipeline on holdout set
3. Measure improvement over baseline (retrieval-only)

**Success Metric**: Response appropriateness > 0.65 (up from 0.56 baseline)

---

## Research Synthesis: Industry Approaches (2026-01-31)

External research on combining clustering, DA classification, and embeddings for response generation.

### Google Smart Reply Architecture (KDD 2016)

**Source**: [Smart Reply: Automated Response Suggestion for Email](https://arxiv.org/abs/1606.04870)

Google's Smart Reply faced the **exact same problem** we discovered:
> "The LSTM has a strong tendency towards producing positive responses, whereas negative responses typically receive low scores."

**Their Three-Stage Pipeline**:

```
STAGE 1: SEMANTIC CLUSTERING
├── Graph-based clustering using EXPANDER algorithm
├── Nodes = response messages + feature nodes (n-grams, skip-grams)
├── Human raters validate clusters
└── Result: 380 semantic clusters covering 12.9k unique suggestions

STAGE 2: RESPONSE SELECTION
├── LSTM generates candidates per cluster
└── Only take top-1 from each cluster (prevents redundancy)

STAGE 3: DIVERSITY ENFORCEMENT (Critical!)
├── If top-2 are both positive → force negative into slot 3
├── If top-2 are both negative → force positive into slot 3
└── Uses second LSTM pass restricted to positive/negative target set
```

**Key Insight**: They solved topic-vs-type problem by **enforcing diversity by functional type**, not by semantic similarity alone.

**Production Stats**:
- 12.9k unique suggestions daily
- 380 semantic clusters
- 31.9% of suggestions actually used
- Slot 3 (diverse option) is crucial for system quality

**Reference**: [The Morning Paper Deep Dive](https://blog.acolyer.org/2016/11/24/smart-reply-automated-response-suggestion-for-email/)

### ChatRouter: Hierarchical Intent Classification (HP, 2025)

**Source**: [HP Technical Disclosure](https://www.tdcommons.org/cgi/viewcontent.cgi?article=9923&context=dpubs_series)

**Multi-Intent Detection + Priority Routing**:
- LLM with few-shot prompting for intent detection
- Detects **multiple intents** simultaneously with confidence scores
- **Hierarchical Priority Logic Engine** evaluates intent combinations
- Routes to appropriate skill/handler based on priority

**Relevance**: Our DA classifier + clusters = multi-signal system that could use similar priority logic.

### LLM-based Smart Reply (LSR, 2023-2024)

**Source**: [LSR Research](https://arxiv.org/html/2306.11980v5)

**Two-Step Generation**:
1. Generate response **type first** (Agree, Disagree, Question, etc.)
2. Then generate full response conditioned on type

This validates our approach: DA classifier → determines type → LLM uses type as guidance.

### Hybrid Embeddings + Structural Features (2024)

**Source**: [ScienceDirect Intent Recognition](https://www.sciencedirect.com/science/article/pii/S0925231223011773)

**Best Practice: Concatenate Features**:
```python
# Step 1: Extract semantic features (BERT/sentence-transformers)
semantic_features = embedder.encode(text)

# Step 2: Extract structural features (rule-based)
structural_features = [
    starts_with_yes,      # "yeah", "sure", "yes"
    starts_with_no,       # "no", "can't", "nah"
    ends_with_question,   # contains "?"
    is_short,             # < 5 words
    has_emoji,
]

# Step 3: Concatenate and classify
combined = np.concatenate([semantic_features, structural_features])
```

**Finding**: Combining semantic + structural outperforms either alone.

### Dual-Encoder vs Cross-Encoder Trade-offs

**Source**: [Sentence Transformers Documentation](https://www.sbert.net/examples/applications/cross-encoder/README.html)

**Industry Standard Pipeline**:
1. Bi-encoder (FAISS) retrieves top-100 candidates (~5ms)
2. Cross-encoder reranks to top-5 (~50-200ms)

**Our Decision**: Skip cross-encoder for V1 (latency concerns for short text).

---

## Cluster Purity Analysis Results (2026-01-31)

Analyzed whether HDBSCAN clusters can predict response DA type.

### Summary Statistics

```
High purity clusters (>=70%): 189
Low purity clusters (<70%):    51

Total clusters: 240
Noise points: 45,411 (43%)
```

### Key Finding: STATEMENT Dominates Everything

| Observation | Implication |
|-------------|-------------|
| 189 clusters have >=70% purity | Looks good... |
| BUT STATEMENT is 78% of all data | "Purity" is misleading |
| Most clusters dominated by STATEMENT | Clusters find topics, DA classifier needed for types |

### Useful Non-STATEMENT Clusters

These clusters have distinct functional types (not just STATEMENT):

| Cluster | Purity | Dominant Type | Size | Use Case |
|---------|--------|---------------|------|----------|
| 9 | 100% | AGREE | 124 | Agreement exemplars |
| 7 | 99% | AGREE | 181 | Agreement exemplars |
| 32 | 100% | REACT_POSITIVE | 164 | Congratulations exemplars |
| 12 | 99% | REACT_POSITIVE | 101 | Excitement exemplars |
| 45 | 99% | QUESTION | 112 | Clarification exemplars |
| 80 | 87% | QUESTION | 136 | Clarification exemplars |
| 19 | 96% | ACKNOWLEDGE | 177 | Acknowledgment exemplars |
| 11 | 85% | ACKNOWLEDGE | 189 | Got-it exemplars |

### Low Purity Clusters (Mixed Types)

These clusters show topic grouping with mixed functional types:

| Cluster | Purity | Dominant | Secondary | Interpretation |
|---------|--------|----------|-----------|----------------|
| 53 | 49% | GREETING | QUESTION (201) | "Hey" vs "Hey, what's up?" |
| 213 | 47% | STATEMENT | ANSWER (204) | Informational responses |
| 49 | 32% | ACKNOWLEDGE | REACT_POSITIVE (72) | Positive acknowledgments |
| 23 | 32% | AGREE | STATEMENT (45) | Agreement with elaboration |
| 74 | 54% | AGREE | ACKNOWLEDGE (67) | "Yeah" variants |

**Insight**: Low-purity clusters often contain related-but-different functional types.

### Strategy Update Based on Analysis

```
PREVIOUS ASSUMPTION:
- Clusters might predict response type
- High purity = usable for type inference

ACTUAL FINDING:
- Clusters predict TOPIC (food, games, scheduling)
- High purity is artifact of STATEMENT dominance (78%)
- Only a few clusters have distinct non-STATEMENT types

UPDATED STRATEGY:
1. Use DA classifier for response type (AGREE/DECLINE/DEFER)
2. Use clusters for topic context (filter FAISS to same-topic examples)
3. Mine non-STATEMENT clusters for better DA exemplars
```

---

## Updated Combined Architecture (2026-01-31)

Based on research + cluster analysis, here's the refined pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│  INCOMING MESSAGE: "Want to grab lunch?"                            │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Multi-Signal Classification (Parallel)                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │
│  │ DA CLASSIFIER │  │ HDBSCAN       │  │ STRUCTURAL    │           │
│  │ trigger type  │  │ topic cluster │  │ FEATURES      │           │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘           │
│          │                  │                  │                    │
│          ▼                  ▼                  ▼                    │
│   INVITATION           Cluster 73         is_question=True         │
│   (conf: 0.85)         (FOOD topic)       is_invitation=True       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Valid Response Types (from DA taxonomy)                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  TRIGGER_TO_VALID_RESPONSES["INVITATION"] = [                       │
│      "AGREE",    # "Yeah I'm down!"                                 │
│      "DECLINE",  # "Can't today"                                    │
│      "DEFER",    # "Let me check"                                   │
│      "QUESTION"  # "What time?"                                     │
│  ]                                                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3: Type-Filtered FAISS Retrieval                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  For each valid response type, find examples:                       │
│                                                                     │
│  FAISS search → filter by response_da_type = "AGREE"                │
│              → further filter by cluster_id = 73 (FOOD topic)       │
│              → get: "Yeah let's do it!", "I'm down for lunch"       │
│                                                                     │
│  FAISS search → filter by response_da_type = "DECLINE"              │
│              → further filter by cluster_id = 73 (FOOD topic)       │
│              → get: "Can't today, swamped", "Not hungry rn"         │
│                                                                     │
│  FAISS search → filter by response_da_type = "DEFER"                │
│              → further filter by cluster_id = 73 (FOOD topic)       │
│              → get: "Maybe, let me check my sched"                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 4: Multi-Option Generation (Gmail Smart Reply Style)         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Option 1 (AGREE):   "Yeah I'm down! What time?"                   │
│  Option 2 (DECLINE): "Can't today, swamped with work"              │
│  Option 3 (DEFER):   "Maybe, let me check my schedule"             │
│                                                                     │
│  DIVERSITY ENFORCEMENT (from Smart Reply research):                 │
│  - If slots 1-2 are both positive → force negative/neutral in 3    │
│  - If slots 1-2 are both negative → force positive in 3            │
│                                                                     │
│  User picks one → system learns preference                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Implementation Components

#### 1. Unified Response Classifier

```python
# jarvis/response_classifier.py

class UnifiedResponseClassifier:
    """Combines DA classification + cluster assignment + structural features."""

    def __init__(self):
        self.da_classifier = DialogueActClassifier("response")

    def classify(self, response: str, embedder=None) -> dict:
        # 1. DA classification (functional type)
        da_result = self.da_classifier.classify(response)

        # 2. Structural features (rule-based backup)
        structural_type = self._structural_classify(response)

        # 3. Combine with confidence weighting
        if da_result.confidence >= 0.7:
            final_type = da_result.label
        elif structural_type != "UNKNOWN":
            final_type = structural_type
        else:
            final_type = da_result.label

        return {
            "type": final_type,
            "da_type": da_result.label,
            "da_confidence": da_result.confidence,
            "structural_type": structural_type,
        }

    def _structural_classify(self, text: str) -> str:
        """Rule-based classification for high-precision cases."""
        text_lower = text.lower().strip()

        # AGREE patterns (high precision)
        if text_lower.startswith(("yeah", "yes", "sure", "definitely",
                                   "absolutely", "sounds good", "i'm down",
                                   "count me in", "let's do it")):
            return "AGREE"

        # DECLINE patterns
        if text_lower.startswith(("no", "can't", "cannot", "sorry",
                                   "unfortunately", "i can't", "not")):
            return "DECLINE"

        # DEFER patterns
        if text_lower.startswith(("maybe", "possibly", "let me check",
                                   "i'll see", "not sure", "might")):
            return "DEFER"

        # QUESTION patterns
        if "?" in text:
            return "QUESTION"

        # REACT_POSITIVE patterns
        if text_lower.startswith(("congrats", "amazing", "awesome",
                                   "that's great", "so happy", "wow")):
            return "REACT_POSITIVE"

        return "UNKNOWN"
```

#### 2. Type-Filtered FAISS Retrieval

```python
# jarvis/retrieval.py

def get_typed_examples(
    trigger: str,
    target_response_type: str,
    db: JarvisDB,
    embedder,
    k: int = 5,
    topic_cluster: int | None = None,
) -> list[dict]:
    """Get examples filtered by response type (and optionally topic)."""

    # Get more candidates than needed, then filter
    search_results = index_searcher.search_with_pairs(
        query=trigger,
        k=k * 3,  # Get 3x to have filtering room
        embedder=embedder,
    )

    # Filter by response_da_type
    typed_results = [
        r for r in search_results
        if r.get("response_da_type") == target_response_type
    ]

    # Optionally filter by topic cluster
    if topic_cluster is not None and topic_cluster != -1:
        same_topic = [r for r in typed_results if r.get("cluster_id") == topic_cluster]
        if len(same_topic) >= 2:
            typed_results = same_topic

    return typed_results[:k]
```

#### 3. Multi-Option Generation with Diversity

```python
# jarvis/multi_option.py

from scripts.build_da_classifier import TRIGGER_TO_VALID_RESPONSES

def generate_response_options(
    trigger: str,
    trigger_da: str,
    faiss_results: list[dict],
    contact,
    generator,
    max_options: int = 3,
) -> list[dict]:
    """Generate diverse response options by type."""

    valid_types = TRIGGER_TO_VALID_RESPONSES.get(trigger_da, ["STATEMENT"])
    options = []

    for response_type in valid_types[:max_options]:
        # Get type-specific examples
        typed_examples = [
            (r["trigger_text"], r["response_text"])
            for r in faiss_results
            if r.get("response_da_type") == response_type
        ][:3]

        if not typed_examples:
            # Fall back to any examples + type hint
            typed_examples = [(r["trigger_text"], r["response_text"])
                             for r in faiss_results[:2]]

        # Generate with type guidance
        prompt = build_typed_prompt(
            trigger=trigger,
            response_type=response_type,
            examples=typed_examples,
            contact=contact,
        )

        response = generator.generate(prompt)

        options.append({
            "type": response_type,
            "response": response.text.strip(),
            "examples_used": len(typed_examples),
        })

    return enforce_diversity(options)


def enforce_diversity(options: list[dict]) -> list[dict]:
    """Ensure response options are functionally diverse (Smart Reply approach)."""

    if len(options) < 3:
        return options

    types = [o["type"] for o in options]

    positive_types = {"AGREE", "REACT_POSITIVE"}
    negative_types = {"DECLINE", "REACT_SYMPATHY"}

    has_positive = any(t in positive_types for t in types)
    has_negative = any(t in negative_types for t in types)

    # If all positive, try to include DECLINE or DEFER
    if has_positive and not has_negative:
        for opt in options:
            if opt["type"] in {"DECLINE", "DEFER"}:
                options.remove(opt)
                options.insert(2, opt)  # Move to slot 3
                break

    return options[:3]
```

### Component Responsibility Summary

| Component | Role | When to Use |
|-----------|------|-------------|
| **DA Classifier** | Determines functional type (AGREE/DECLINE/DEFER) | Always - primary routing signal |
| **HDBSCAN Clusters** | Identifies topic (food/games/scheduling) | Filter FAISS to same-topic examples |
| **Structural Rules** | High-precision backup for clear patterns | When DA confidence < 0.7 |
| **FAISS Retrieval** | Finds similar past exchanges | Filter by DA type + cluster for best examples |
| **Multi-Option Gen** | Generates diverse response choices | For INVITATION, REQUEST, YN_QUESTION triggers |
| **Diversity Enforcement** | Ensures positive/negative balance | Applied to final 3 options |

### Files to Create/Modify

| File | Purpose | Status |
|------|---------|--------|
| `jarvis/response_classifier.py` | Unified classifier (DA + structural) | TODO |
| `jarvis/multi_option.py` | Multi-option generation with diversity | TODO |
| `jarvis/retrieval.py` | Type-filtered FAISS retrieval | TODO |
| `jarvis/router.py` | Wire up new pipeline | MODIFY |
| `api/routers/drafts.py` | Return multiple options to frontend | MODIFY |

### Research Sources

- [Smart Reply: Automated Response Suggestion for Email (KDD 2016)](https://arxiv.org/abs/1606.04870)
- [Efficient Smart Reply, now for Gmail (Google Research)](https://research.google/blog/efficient-smart-reply-now-for-gmail/)
- [Smart Reply Deep Dive (The Morning Paper)](https://blog.acolyer.org/2016/11/24/smart-reply-automated-response-suggestion-for-email/)
- [ChatRouter: Hierarchical Intent Classification (HP, 2025)](https://www.tdcommons.org/cgi/viewcontent.cgi?article=9923&context=dpubs_series)
- [Intent Recognition with Structural Features (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0925231223011773)
- [LLM-based Smart Reply (LSR)](https://arxiv.org/html/2306.11980v5)
- [Sentence Transformers Clustering Documentation](https://sbert.net/examples/sentence_transformer/applications/clustering/README.html)
- [Cross-Encoder Reranking Best Practices](https://www.sbert.net/examples/applications/cross-encoder/README.html)

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-30 | V1: FAISS only, no reranker | Latency concerns, uncertain gains for short text |
| 2026-01-30 | V1: 4 features only | Prevent scope creep, ship minimal viable |
| 2026-01-30 | Move to `legacy/`, don't delete | Preserve reference for bug fixes |
| 2026-01-30 | Day 1 validation required | Prevent building on bad assumptions |
| 2026-01-30 | Async generation, never blocking | 2s LLM generation kills UX |
| 2026-01-30 | Add SKIP route | Not all messages need suggestions |
| 2026-01-30 | Add COLD_START route | Handle new contacts gracefully |
| 2026-01-30 | Conservative thresholds (0.85/0.50) | Start strict, loosen with data |
| 2026-01-30 | Store context with pairs | Address "context blindness" problem |
| 2026-01-31 | Lower template threshold to 0.65 | Overnight eval showed 0.90 too strict |
| 2026-01-31 | Lower generate threshold to 0.45 | Catch more borderline cases |
| 2026-01-31 | Random pair split (not by contact) | Skewed contact sizes gave 96/4 split |
| 2026-01-31 | Semantic chunking extraction | 122k pairs vs 18k threaded - 7x more data |
| 2026-01-31 | MLX batch encoding (500/batch) | ~666 texts/sec, 100k in 2.5 min |
| 2026-01-31 | Multi-option responses | System offers 3 options (agree/decline/defer), user picks |
| 2026-01-31 | UMAP is CPU-only on Mac | No GPU acceleration available, ~2min for 105k vectors |
| 2026-01-31 | Semantic clustering = topics not types | Need structural features for response type classification |
| 2026-01-31 | Add DA classification columns to DB | Persist trigger_da_type, response_da_type, cluster_id |
| 2026-01-31 | DA classifier uses SWDA + manual exemplars | k-NN over labeled utterances, ~300 pairs/sec |
| 2026-01-31 | STATEMENT is catch-all (78% of responses) | Need better ANSWER/AGREE exemplars |
| 2026-01-31 | Combine DA + semantic similarity | High similarity + wrong DA = misclassification; low similarity = bad pair |
| 2026-01-31 | Multi-stage architecture | DA classifier || FAISS search || topic cluster → multi-option generation |
| 2026-01-31 | DA-filtered retrieval | Filter FAISS results by response_da_type for type-specific examples |
| 2026-01-31 | Cluster purity analysis complete | 189 high-purity clusters, but STATEMENT dominates (78%) |
| 2026-01-31 | Clusters = topics, not types | Use clusters for topic filtering, DA classifier for response types |
| 2026-01-31 | Adopt Smart Reply diversity approach | Force positive/negative balance in slot 3 (from Google research) |
| 2026-01-31 | Add structural feature classification | Rule-based backup when DA confidence < 0.7 |
| 2026-01-31 | Mine non-STATEMENT clusters | Clusters 7,9 (AGREE), 32,12 (REACT_POSITIVE), 45,80 (QUESTION) for better exemplars |
| 2026-01-31 | Two-step generation (from LSR research) | Generate response TYPE first, then full response conditioned on type |

---

## Questions Resolved

| Question | Answer |
|----------|--------|
| Include reranker in V1? | **No** - add in V2 if measured lift > 15% |
| Delete old code? | **No** - move to `legacy/` |
| When to evaluate? | **Day 1** before any implementation |
| What confidence for gold pairs? | **0.85-0.95** not 1.0 |
| How to handle new contacts? | **COLD_START route** with fallback |
| Block on LLM generation? | **Never** - async with polling |
| How to ensure response diversity? | **Smart Reply approach**: Force positive/negative balance, max 1 per cluster |
| Should clusters predict response type? | **No** - clusters find topics; use DA classifier for types |
| How to combine semantic + structural? | **Concatenate features**: DA classifier + rule-based backup (from 2024 research) |
| What's the industry standard? | **Gmail Smart Reply** (2016): 380 clusters, diversity enforcement, 3 options |

## Questions Still Open

1. **Memory budget**: Does FAISS + embedder + LLM fit in 8GB? (Measure in Phase 2)
2. **Incremental updates**: Best strategy for adding new pairs? (Design in Phase 2)
3. **Desktop app needs**: What endpoints does Tauri app actually require? (Clarify with frontend)
4. **STATEMENT problem**: 78% of responses are STATEMENT - need better ANSWER/AGREE/DECLINE exemplars
5. **Cross-encoder for V2?**: Research suggests +20-40% accuracy, but latency concerns for short text
6. **User preference learning**: How to learn from which option user picks over time?
7. **Topic cluster granularity**: 240 clusters vs fewer broader topics - what's optimal?
