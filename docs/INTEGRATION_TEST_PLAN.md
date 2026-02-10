# JARVIS Integration Test Plan

**Version:** 1.0  
**Last Updated:** 2026-02-10  
**Scope:** Complete end-to-end and boundary testing strategy

---

## Executive Summary

This document provides a comprehensive integration test plan for the JARVIS AI assistant. It covers:

1. **End-to-End Flows:** Complete message processing pipelines
2. **Boundary Tests:** Module interface contracts and error handling
3. **Regression Scenarios:** Historical bug prevention
4. **Missing Coverage:** Gaps in existing test suite

---

## 1. END-TO-END FLOWS

### 1.1 Core Reply Generation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MESSAGE RECEIPT → REPLY GENERATION                   │
└─────────────────────────────────────────────────────────────────────────────┘

Entry Points:
├── Socket Server: JarvisSocketServer._generate_draft()
├── API Router:   api/routers/drafts.py::generate_smart_reply()
├── Direct:       ReplyRouter.route()
└── ReplyService: ReplyService.generate_reply()

Complete Call Chain:
────────────────────
Entry Point
    ↓
ReplyRouter.route(incoming, contact_id, thread, chat_id)
    ├── [PARALLEL] ContextService.get_contact(contact_id, chat_id)
    │       └── JarvisDB.get_contact() OR JarvisDB.get_contact_by_chat_id()
    │
    ├── [PARALLEL] classify_with_cascade(incoming)
    │       └── ResponseMobilizationClassifier.classify()
    │           └── Pattern matching → LightGBM fallback
    │
    ├── [PARALLEL] ContextService.search_examples(incoming, chat_id, contact_id, embedder)
    │       └── VecSearcher.search_with_pairs() OR search_with_pairs_global()
    │           ├── Embedder.encode(query)
    │           └── sqlite-vec MATCH query
    │
    └── ReplyService.generate_reply()
            ├── classify_category(incoming, context, mobilization)
            │       ├── Fast path: reactions/acknowledgments
            │       ├── LightGBM prediction (BERT + context + hand-crafted)
            │       └── Heuristic corrections
            │
            ├── ReplyService.can_use_llm() → Health check
            │
            └── ReplyService._generate_llm_reply()
                    ├── build_generation_request()
                    │       ├── ContextService.get_relationship_profile()
                    │       ├── resolve_category() → get_optimized_instruction()
                    │       ├── Reranker.rerank() [if results > 1]
                    │       └── _build_chat_prompt() [tokenizer chat template]
                    │
                    └── generator.generate(GenerationRequest)
                            └── MLXModelLoader.generate_sync()
                                ├── Tokenize with chat template
                                ├── mlx_lm.generate() OR stream_generate()
                                └── Post-process response

Expected Outputs at Each Stage:
───────────────────────────────
1. ReplyRouter.route() → {
       type: "generated" | "clarify" | "acknowledge" | "closing" | "fallback",
       response: str,
       confidence: "high" | "medium" | "low",
       similarity_score: float,
       cluster_name: str | None,
       similar_triggers: list[str] | None
   }

2. classify_category() → CategoryResult(category, confidence, method)

3. ContextService.search_examples() → list[dict] with keys:
       trigger_text, response_text, similarity, topic

4. generator.generate() → GenerationResult(
       text, tokens_generated, generation_time_ms, tokens_per_second
   )

What to Mock vs. Test Live:
───────────────────────────
✅ MOCK:
   - MLX model generation (use MagicMock with deterministic response)
   - BERT embeddings (use MockEmbedder from conftest.py)
   - iMessage database (use in-memory SQLite + mock Message objects)
   - ChatDBReader (mock get_messages(), check_access())

✅ TEST LIVE:
   - Category classification (fast, deterministic)
   - Response mobilization (fast, rule-based)
   - Vector search via sqlite-vec (use test database)
   - Prompt assembly and formatting
   - Confidence computation logic
```

### 1.2 Real-Time Message Watcher Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FILE WATCHER → BROADCAST PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────────┘

Entry Points:
├── Socket Server: JarvisSocketServer.start() → ChatDBWatcher.start()
└── Standalone:   jarvis/watcher.py::run_watcher()

Complete Call Chain:
────────────────────
ChatDBWatcher.start()
    ├── _validate_schema() → Check chat.db tables/columns exist
    ├── _get_last_rowid() → Initialize cursor position
    └── [ASYNC TASK] _watch_fsevents() OR _watch_polling()
            ↓
            FSEvents/polling detects chat.db modification
            ↓
            _debounced_check() (50ms debounce)
            ↓
            _check_new_messages()
                ├── _get_new_messages() → Query messages by ROWID
                │       └── _query_new_messages() [batched, 500/msg batch]
                │
                ├── [ASYNC TASK] _index_new_messages()
                │       └── VecSearcher.index_messages()
                │
                ├── [ASYNC TASK] _extract_facts()
                │       └── FactExtractor.extract_facts()
                │           └── NER + rule-based patterns + NLI verification
                │
                └── JarvisSocketServer.broadcast("new_message", params)
                        └── Send to all connected WebSocket/Unix clients

Expected Outputs:
─────────────────
1. broadcast notification → {
       "jsonrpc": "2.0",
       "method": "new_message",
       "params": {
           "message_id": int,
           "chat_id": str,
           "sender": str,
           "text": str,
           "date": str (ISO),
           "is_from_me": bool
       }
   }

2. VecSearcher.index_messages() → int (count indexed)

3. FactExtractor.extract_facts() → list[Fact]

What to Mock vs. Test Live:
───────────────────────────
✅ MOCK:
   - Actual FSEvents (use manual trigger or polling mode)
   - chat.db file system (use temporary database file)
   - spaCy NER (expensive, slow)
   - NLI model (expensive, slow)

✅ TEST LIVE:
   - SQLite ROWID queries
   - Broadcast delivery to mock clients
   - Debounce logic
   - Batch message fetching
```

### 1.3 API Draft Generation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    API ENDPOINT → DRAFT GENERATION                          │
└─────────────────────────────────────────────────────────────────────────────┘

Entry Point: api/routers/drafts.py

Flows:
────────────────────

A) /drafts/reply (Full MLX Generation)
   ────────────────────────────────────
   generate_draft_reply(DraftReplyRequest)
       ├── get_imessage_reader() → ChatDBReader
       ├── reader.get_messages(chat_id, limit) [threadpool]
       ├── get_warm_generator() → MLXGenerator [threadpool]
       ├── _build_reply_prompt() per suggestion
       ├── _generate_single_suggestion() [threadpool x N]
       │       └── generator.generate(GenerationRequest)
       └── DraftReplyResponse(suggestions, context_used)

B) /drafts/smart-reply (Routed Generation)
   ───────────────────────────────────────
   generate_smart_reply(RoutedReplyRequest)
       ├── get_imessage_reader() → ChatDBReader
       ├── reader.get_messages(chat_id, limit) [threadpool]
       ├── _route_reply_sync() [threadpool]
       │       └── ReplyRouter.route()
       └── RoutedReplyResponse(response, type, confidence, metadata)

C) /drafts/summarize (Conversation Summary)
   ─────────────────────────────────────────
   summarize_conversation(DraftSummaryRequest)
       ├── reader.get_messages(chat_id, limit) [threadpool]
       ├── get_warm_generator() [threadpool]
       ├── _build_summary_prompt()
       ├── _generate_summary() [threadpool]
       │       └── generator.generate()
       └── DraftSummaryResponse(summary, key_points, date_range)

Expected Outputs:
─────────────────
1. DraftReplyResponse → {
       suggestions: [{text: str, confidence: float}],
       context_used: {num_messages, participants, last_message}
   }

2. RoutedReplyResponse → {
       response: str,
       response_type: "generated" | "clarify",
       confidence: "high" | "medium" | "low",
       similarity_score: float,
       context_used: ContextInfo
   }

3. DraftSummaryResponse → {
       summary: str,
       key_points: list[str],
       date_range: {start: str, end: str}
   }

Error Cases to Test:
────────────────────
- 403: Full Disk Access not granted
- 404: No messages found for chat_id
- 408: Request timed out
- 429: Rate limit exceeded
- 500: Generation failed
- 503: Model service unavailable
```

### 1.4 Socket Server RPC Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SOCKET SERVER RPC METHODS                                │
└─────────────────────────────────────────────────────────────────────────────┘

Entry Point: JarvisSocketServer._process_message()

Methods:
────────
ping                    → Health check with model readiness
generate_draft          → Draft reply with streaming support
summarize               → Conversation summary with streaming
get_smart_replies       → Quick template-based suggestions
semantic_search         → Vector search over messages
batch                   → Parallel batch RPC execution
resolve_contacts        → Contact name resolution
list_conversations      → Recent conversation list
get_routing_metrics     → Performance metrics
prefetch_stats          → Prefetch cache statistics
prefetch_invalidate     → Cache invalidation
prefetch_focus          → Focus mode for specific chat
prefetch_hover          → Pre-compute on UI hover

Complete Call Chain (generate_draft with streaming):
────────────────────────────────────────────────────
_process_message({method: "generate_draft", params: {stream: true}})
    ├── Deep copy params
    ├── Extract stream flag
    ├── handler(**params, _writer=writer, _request_id=id)
    │       └── _generate_draft_streaming()
    │               ├── ChatDBReader.get_messages()
    │               ├── reply_service.prepare_streaming_context() [thread]
    │               │       └── Full RAG pipeline
    │               └── generator.generate_stream(request)
    │                       └── Stream tokens via _send_stream_token()
    └── Return None (streaming handled by handler)

Expected Outputs:
─────────────────
1. Normal response: {"jsonrpc": "2.0", "result": {...}, "id": 1}
2. Stream token: {"jsonrpc": "2.0", "method": "stream.token", "params": {...}}
3. Error: {"jsonrpc": "2.0", "error": {code, message}, "id": 1}

JSON-RPC Error Codes:
─────────────────────
-32700: Parse error (invalid JSON)
-32600: Invalid request
-32601: Method not found
-32602: Invalid params
-32603: Internal error
```

---

## 2. BOUNDARY TESTS

### 2.1 Router → Classifier Boundary

```python
# Input Contract
classify_with_cascade(incoming: str) -> MobilizationResult

# Test Cases:
├── Empty string → MobilizationResult(pressure=NONE, confidence=0.0)
├── Whitespace only → Same as empty
├── Very long message (>1000 chars) → Should not crash
├── Unicode/emoji → Should handle gracefully
├── Special characters → Should not affect classification
└── None input → TypeError or graceful handling

# Output Handling in Router:
├── mobilization.pressure → Maps to confidence level
├── mobilization.response_type → Affects prompt instruction
└── mobilization.confidence → Weighted in final confidence

# Error Cases:
├── Classifier returns None → Router should use default (ResponsePressure.NONE)
└── Classifier raises exception → Caught, logged, use default

# Performance Expectation:
└── Latency: < 5ms for pattern-based, < 50ms for ML fallback
```

### 2.2 Classifier → Reply Service Boundary

```python
# Input Contract
ReplyService.generate_reply(
    incoming: str,
    contact: Contact | None,
    search_results: list[dict] | None,
    mobilization: MobilizationResult | None,
)

# Test Cases - Category Routing:
├── Category=acknowledge → Skip LLM, return template
├── Category=closing → Skip LLM, return template
├── Category=question → LLM with question-optimized prompt
├── Category=request → LLM with action-oriented prompt
├── Category=emotion → LLM with empathetic tone
└── Category=statement → Standard LLM generation

# Search Results Handling:
├── Empty search_results → Fallback to base generation
├── search_results[0] missing keys → KeyError handling
├── Invalid similarity scores → Clamp to [0, 1]
└── Very large search_results (>100) → Truncate gracefully

# Mobilization Handling:
├── mobilization=None → Default behavior
├── mobilization.pressure=HIGH → Urgent tone, lower token limit
├── mobilization.pressure=NONE → Brief acknowledgment, 20 tokens
└── Invalid mobilization fields → AttributeError handling
```

### 2.3 Reply Service → Prompts Boundary

```python
# Input Contract
build_generation_request(
    incoming: str,
    search_results: list[dict],
    contact: Contact | None,
    mobilization: MobilizationResult,
) -> GenerationRequest

# Test Cases - Prompt Assembly:
├── No tokenizer available → Fallback to XML prompt
├── Chat template available → Use tokenizer.apply_chat_template()
├── Very long context (>10000 chars) → Truncation needed
├── Empty few_shot_examples → No examples in prompt
├── Invalid examples format → Skip malformed examples
└── Special chars in contact name → Escape properly

# Reranker Integration:
├── Cross-encoder not loaded → Skip reranking
├── Reranker returns empty → Use original order
├── Reranker scores all negative → Still use results
└── Reranker exception → Log warning, continue without

# Performance Expectation:
└── Latency: < 100ms for prompt assembly (excluding LLM)
```

### 2.4 Reply Service → MLX Generator Boundary

```python
# Input Contract
generator.generate(GenerationRequest) -> GenerationResponse

# Test Cases:
├── Empty prompt → ModelGenerationError
├── Prompt > max context length → Truncation or error
├── Generation timeout → ModelGenerationError with timeout code
├── Model not loaded → ModelGenerationError (MDL_LOAD_FAILED)
├── Out of memory during generation → ModelGenerationError (MDL_OUT_OF_MEMORY)
└── MLX Metal crash → Exception caught, model unloaded

# Response Handling:
├── Empty response.text → Fallback message
├── Response contains only stop sequence → Regenerate or fallback
├── Very long generation (>max_tokens) → Truncated
└── Invalid UTF-8 in response → Replace/sanitize

# Performance Expectations:
├── First token (TTFT): < 500ms
├── Tokens per second: > 20 tps on Apple Silicon
└── Total generation time: < 2000ms for 40 tokens
```

### 2.5 Context Service → Database Boundary

```python
# Input Contract
ContextService.get_contact(contact_id, chat_id) -> Contact | None

# Test Cases:
├── contact_id valid → Return Contact with all fields
├── chat_id valid → Lookup by chat_id, return Contact
├── Neither provided → Return None
├── Both provided → contact_id takes precedence
├── Contact not found → Return None
└── Database locked/timeout → Retry or return None

# Input Contract
ContextService.search_examples(incoming, chat_id, contact_id, embedder)

# Test Cases:
├── contact_id provided → Use partition-filtered search (~0.2ms)
├── No contact_id → Use global search with hamming pre-filter (~5ms)
├── VecSearcher raises exception → Return [], log warning
├── sqlite-vec extension unavailable → Return []
└── Empty query string → Return []
```

### 2.6 Watcher → Fact Extractor Boundary

```python
# Input Contract
FactExtractor.extract_facts(messages, contact_id) -> list[Fact]

# Test Cases:
├── Empty messages list → Return []
├── All messages from me → Return [] (only incoming)
├── spaCy not available → Fallback to regex-only
├── NLI model not available → Skip verification pass
├── All messages filtered as bot/professional → Return []
└── Message with no extractable content → Return []

# Fact Quality Thresholds:
├── confidence < 0.5 → Filtered out
├── vague subject (pronouns) → Filtered out
├── too short (< 3 words for preference) → Filtered out
├── incoherent subject → Filtered out
└── NLI contradiction → Filtered out

# Performance Expectation:
└── Batch processing: > 100 messages/second
```

### 2.7 API Router → Dependencies Boundary

```python
# Input Contract
depends(get_imessage_reader) -> ChatDBReader

# Test Cases:
├── Full Disk Access granted → Return working reader
├── Full Disk Access denied → Raise HTTPException 403
├── chat.db not accessible → Raise HTTPException 403
└── Multiple concurrent requests → Reader per request (DI cleanup)

# Input Contract
depends(get_db) -> JarvisDB

# Test Cases:
├── Database file exists → Return connection
├── Database needs schema init → Create tables
├── sqlite-vec extension fails → Continue without vector tables
└── Database locked → Wait/timeout per PRAGMA settings
```

---

## 3. REGRESSION SCENARIOS

### 3.1 N+1 Query Prevention

**Historical Bug:** Conversations endpoint made O(n) queries for n conversations (900ms → fixed to <100ms)

```python
# Test: Verify single query per endpoint call
@pytest.mark.integration
def test_conversations_list_no_n_plus_one():
    """Regression test: conversations list should use single query."""
    # Setup: Create 50 conversations with messages
    # Execute: GET /conversations
    # Assert: Query count < 5 (not 50+)
    pass

# Additional test fixtures needed:
# - Database with 100+ contacts, each with 10+ messages
# - Query counter fixture using sqlite trace callback
```

### 3.2 MLX Thread Safety

**Historical Bug:** Concurrent MLX inference crashed with "command encoder already encoding"

```python
# Test: Verify serialized GPU access
@pytest.mark.integration
@pytest.mark.timeout(30)
def test_concurrent_generation_serialized():
    """Regression test: Concurrent generation must not crash MLX."""
    # Setup: Load generator
    # Execute: Start 5 concurrent generation requests
    # Assert: All complete without crashing, results are valid
    pass

# Implementation: Use ThreadPoolExecutor + MLXModelLoader._mlx_load_lock verification
```

### 3.3 Fact Extraction Precision

**Historical Bug:** Initial fact extraction had 0% precision on real messages

```python
# Test: Verify quality filters work
@pytest.mark.integration
def test_fact_extraction_quality_filters():
    """Regression test: Low quality facts should be filtered."""
    # Setup: Messages with known bad extractions (pronouns, fragments)
    bad_messages = [
        "I like it",  # "it" should be filtered as vague
        "in August",  # Bare preposition should be filtered
        "sm rn",      # Too many abbreviations
    ]
    # Execute: FactExtractor.extract_facts()
    # Assert: All filtered out, returned facts = 0
    pass
```

### 3.4 Memory Leak Prevention

**Historical Bug:** Model loading without proper cleanup caused OOM

```python
# Test: Verify unload frees memory
@pytest.mark.integration
def test_model_unload_frees_memory():
    """Regression test: Model unload must free GPU memory."""
    # Setup: Load model, record memory
    # Execute: Generate, then unload
    # Assert: Memory returned to within 10% of pre-load level
    pass
```

### 3.5 Response Router Singleton Leak

**Historical Bug:** reset_reply_router() didn't close iMessage reader

```python
# Test: Verify proper cleanup on reset
@pytest.mark.integration
def test_router_reset_closes_resources():
    """Regression test: Router reset must close DB connections."""
    # Setup: Create router, trigger reader creation
    # Execute: reset_reply_router()
    # Assert: Reader is closed, can be garbage collected
    pass
```

### 3.6 Template Analytics Deadlock

**Historical Bug:** get_stats() acquired lock then called methods that also acquired lock

```python
# Test: Verify no deadlock
@pytest.mark.integration
@pytest.mark.timeout(5)
def test_template_analytics_no_deadlock():
    """Regression test: get_stats must not deadlock."""
    # Setup: Create TemplateAnalytics with data
    # Execute: Call get_stats() from multiple threads
    # Assert: All return successfully within timeout
    pass
```

### 3.7 AppleScript Injection

**Historical Bug:** chat_id in scheduler could inject arbitrary AppleScript

```python
# Test: Verify input sanitization
@pytest.mark.integration
def test_applescript_injection_prevented():
    """Regression test: Malicious chat_id must be escaped."""
    # Setup: chat_id with injection attempt: '"; do shell script "rm -rf /";
    # Execute: Schedule message
    # Assert: No shell execution, script properly escaped
    pass
```

### 3.8 Batch Request Limit

**Historical Bug:** No limit on batch size could cause DoS

```python
# Test: Verify batch size limit enforced
@pytest.mark.integration
def test_batch_request_size_limit():
    """Regression test: Batch must reject >50 requests."""
    # Setup: Create 51 requests
    # Execute: Send batch RPC
    # Assert: Returns INVALID_PARAMS error
    pass
```

### 3.9 Settings Partial Update

**Historical Bug:** Partial update overwrote all sub-object fields

```python
# Test: Verify partial update merges
@pytest.mark.integration
def test_settings_partial_update_preserves_fields():
    """Regression test: Partial update must merge, not replace."""
    # Setup: Settings with temperature=0.7, max_tokens=100
    # Execute: Update with {"temperature": 0.8} only
    # Assert: max_tokens still 100, temperature now 0.8
    pass
```

### 3.10 Concurrent Router Access

**Historical Bug:** Race condition in get_generator() double-checked locking

```python
# Test: Verify thread-safe singleton access
@pytest.mark.integration
@pytest.mark.timeout(10)
def test_router_thread_safe():
    """Regression test: Concurrent route() calls must be safe."""
    # Setup: Shared ReplyRouter
    # Execute: 10 threads call route() simultaneously
    # Assert: All return valid results, no exceptions
    pass
```

---

## 4. MISSING COVERAGE

### 4.1 Modules with 0 Test Coverage

| Module | Priority | Test Type Needed |
|--------|----------|------------------|
| `jarvis/contacts/fact_storage.py` | HIGH | Integration with DB |
| `jarvis/contacts/contact_profile_context.py` | HIGH | Profile serialization |
| `jarvis/prefetch/*.py` (4 modules) | MEDIUM | Cache behavior, invalidation |
| `jarvis/scheduler/*.py` (4 modules) | MEDIUM | Task scheduling, execution |
| `jarvis/graph/*.py` (5 modules) | LOW | Graph algorithms, layout |
| `jarvis/analytics/*.py` (4 modules) | LOW | Aggregation, reports |
| `jarvis/tags/*.py` (4 modules) | LOW | Tag rules, auto-tagger |
| `jarvis/eval/*.py` (3 modules) | LOW | Evaluation pipeline |
| `integrations/calendar/*.py` (2 modules) | MEDIUM | AppleScript integration |
| `integrations/imessage/bridge.py` | HIGH | Message sending |

### 4.2 Functions with Error Paths but No Error Tests

| Function | File | Missing Tests |
|----------|------|---------------|
| `JarvisDB._cleanup_stale_connections` | db/core.py | DB error during cleanup |
| `VecSearcher.index_messages` | search/vec_search.py | Partial batch failure |
| `CategoryClassifier._load_pipeline` | classifiers/category_classifier.py | Corrupted model file |
| `ReplyService._dedupe_examples` | reply_service.py | Embedding compute failure |
| `MLXModelLoader.load` | models/loader.py | Network error, disk error |
| `FactExtractor._verify_facts_nli` | contacts/fact_extractor.py | NLI model failure |
| `ContextService.fetch_conversation_context` | services/context_service.py | iMessage read error |

### 4.3 Integration Points Only Unit-Tested

| Integration Point | Current Test | Missing Integration Test |
|-------------------|--------------|-------------------------|
| Router → ReplyService | Unit mocked | Full pipeline with real services |
| ReplyService → VecSearcher | Unit mocked | Live sqlite-vec search |
| Watcher → FactExtractor | Not tested | End-to-end fact extraction flow |
| API → Socket Server | Not tested | Concurrent API/socket access |
| Socket → Prefetch | Not tested | Cache hit/miss scenarios |
| Classifier → Prompts | Unit mocked | Full category → prompt flow |
| Contact Profile → Generation | Not tested | Profile injection in prompts |

### 4.4 Specific Test Cases to Add

```python
# A. Streaming generation integration
def test_streaming_generation_full_pipeline():
    """Test complete streaming flow from socket request to token delivery."""
    pass

# B. Prefetch cache behavior
def test_prefetch_predicts_and_caches():
    """Test prefetch predicts need and serves from cache."""
    pass

# C. Fact extraction end-to-end
def test_fact_extraction_watcher_to_storage():
    """Test facts flow from watcher detection to DB storage."""
    pass

# D. Multi-contact vector search
def test_vector_search_multi_contact_isolation():
    """Test that contact A's chunks don't leak to contact B search."""
    pass

# E. Health degradation
def test_health_degradation_reduces_quality():
    """Test that high memory pressure triggers fallback responses."""
    pass

# F. Model warm-up
def test_model_warmer_preloads_on_startup():
    """Test that model warmer preloads before first request."""
    pass

# G. Chat template compatibility
def test_chat_template_applies_correctly():
    """Test various tokenizer chat template formats."""
    pass

# H. Reranker fallback
def test_reranker_failure_continues_without():
    """Test graceful degradation when cross-encoder fails."""
    pass

# I. Schema migration
def test_database_schema_migration():
    """Test migration from old schema versions."""
    pass

# J. Embedding binary format
def test_embedding_binary_roundtrip():
    """Test float → binary → float preserves similarity."""
    pass
```

---

## 5. TEST FIXTURES NEEDED

### 5.1 Database Fixtures

```python
@pytest.fixture
def populated_db(tmp_path):
    """Create JarvisDB with realistic test data."""
    # 100 contacts, 1000 message pairs, 500 indexed chunks
    pass

@pytest.fixture
def empty_vec_db(tmp_path):
    """Create JarvisDB with schema but no data."""
    pass

@pytest.fixture
def corrupted_model_file(tmp_path):
    """Create corrupted .joblib file for error testing."""
    pass
```

### 5.2 Mock Fixtures

```python
@pytest.fixture
def mock_chatdb_reader():
    """Mock ChatDBReader with predefined conversations."""
    pass

@pytest.fixture
def mock_mlx_generator():
    """Mock MLXGenerator with deterministic responses."""
    pass

@pytest.fixture
def slow_mock_generator():
    """Mock generator that simulates slow generation for timeout testing."""
    pass

@pytest.fixture
def failing_mock_generator():
    """Mock generator that raises exceptions for error testing."""
    pass
```

### 5.3 Performance Fixtures

```python
@pytest.fixture
def query_counter():
    """Count database queries during test execution."""
    pass

@pytest.fixture
def latency_tracker():
    """Track and report latency of operations."""
    pass

@pytest.fixture
def memory_profiler():
    """Track memory usage before/after operations."""
    pass
```

---

## 6. IMPLEMENTATION PRIORITY

### Phase 1: Critical Boundaries (Week 1)
1. Router → Classifier boundary tests
2. Reply Service → MLX Generator boundary tests
3. Context Service → Database boundary tests
4. N+1 query regression tests

### Phase 2: Historical Bugs (Week 2)
1. MLX thread safety regression tests
2. Memory leak prevention tests
3. Router singleton cleanup tests
4. Template analytics deadlock tests

### Phase 3: End-to-End Flows (Week 3)
1. Complete message receipt → reply generation flow
2. Real-time watcher → broadcast flow
3. API draft generation flow
4. Socket server RPC flow

### Phase 4: Missing Coverage (Week 4)
1. Uncovered modules (fact_storage, prefetch, scheduler)
2. Error path tests
3. Performance/integration tests for unit-only components

---

## 7. TEST EXECUTION

### Running Integration Tests

```bash
# Run all integration tests
make test

# Run specific integration test file
pytest tests/integration/test_message_flow.py -v

# Run with real embeddings (slow)
pytest tests/integration/ -m "requires_real_embeddings" -v

# Run with coverage report
pytest tests/integration/ --cov=jarvis --cov-report=html

# Run regression tests only
pytest tests/integration/ -k "regression" -v

# Run with performance profiling
pytest tests/integration/ --profile-svg
```

### CI Integration

```yaml
# .github/workflows/test.yml snippet
- name: Integration Tests
  run: |
    pytest tests/integration/ -v --tb=short
  timeout-minutes: 30
  env:
    JARVIS_TEST_DB_PATH: /tmp/jarvis_test.db
```

---

## 8. APPENDICES

### Appendix A: File Coverage Matrix

| File | Unit | Integration | Notes |
|------|------|-------------|-------|
| jarvis/router.py | ✅ | ✅ | Well covered |
| jarvis/reply_service.py | ✅ | ⚠️ | Missing streaming E2E |
| jarvis/socket_server.py | ✅ | ✅ | Well covered |
| jarvis/watcher.py | ✅ | ⚠️ | Missing FSEvents test |
| jarvis/classifiers/*.py | ✅ | ✅ | Well covered |
| jarvis/search/vec_search.py | ✅ | ⚠️ | Missing multi-contact test |
| jarvis/contacts/fact_*.py | ✅ | ❌ | No integration tests |
| jarvis/prefetch/*.py | ❌ | ❌ | No tests |
| jarvis/scheduler/*.py | ❌ | ❌ | No tests |

### Appendix B: Known Flaky Tests

| Test | Issue | Mitigation |
|------|-------|------------|
| test_concurrent_generation | Timing-dependent | Increase timeout, retry |
| test_watcher_debounce | FSEvents timing | Use polling mode in tests |
| test_memory_unload | Memory measurement variance | Allow 15% tolerance |

### Appendix C: External Dependencies for Tests

| Dependency | Required For | Fallback |
|------------|--------------|----------|
| MLX | Model generation | MagicMock generator |
| sentence-transformers | Real embeddings | MockEmbedder |
| sqlite-vec | Vector search | Skip tests |
| spaCy | Fact extraction NER | Regex-only mode |
| watchfiles | FSEvents testing | Polling fallback |

---

*Document maintained by QA team. Update when adding new flows or fixing regressions.*
