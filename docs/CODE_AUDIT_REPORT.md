# JARVIS AI Assistant - Comprehensive Code Audit Report

**Audit Date:** 2026-01-26
**Total Files Reviewed:** ~150 Python files
**Total Lines of Code:** ~25,000+
**Auditor:** Claude Code (Automated Analysis)

---

## Executive Summary

| Severity | Count | Description |
|----------|-------|-------------|
| Critical | 1 | Must fix immediately - causes crashes or data loss |
| High | 20 | Should fix soon - significant bugs or security issues |
| Medium | 84 | Should fix - logic errors, resource leaks, edge cases |
| Low | 68 | Consider fixing - code quality, documentation, style |
| **Total** | **173** | |

### Issues by Module

| Module | Critical | High | Medium | Low | Total |
|--------|----------|------|--------|-----|-------|
| jarvis/ | 1 | 2 | 7 | 3 | 13 |
| api/ | 0 | 2 | 13 | 8 | 23 |
| models/ | 0 | 2 | 7 | 7 | 16 |
| integrations/ | 0 | 1 | 6 | 5 | 12 |
| core/ | 0 | 2 | 7 | 8 | 17 |
| mcp_server/ | 0 | 3 | 13 | 6 | 22 |
| benchmarks/ | 0 | 3 | 10 | 11 | 24 |
| contracts/scripts/ | 0 | 0 | 12 | 14 | 26 |
| tests/ | 0 | 5 | 9 | 6 | 20 |

---

## Critical Issues (1)

### 1. Type Unpacking Error in API Chat Endpoint
**File:** `jarvis/api.py`
**Lines:** 250-270
**Issue:** The `/chat` endpoint's degradation controller `execute()` can return either a string or a tuple. The code attempts to unpack with `text, metadata = execute()`, which raises `ValueError: not enough values to unpack` when a string is returned.

```python
# Problematic code
text, metadata = deg_controller.execute(FEATURE_CHAT, generate_response, request.message)

# Handle case where degradation controller returns a simple string
if isinstance(text, str) and not metadata:
    return ChatResponse(...)
```

**Fix:** Check return type before unpacking:
```python
result = deg_controller.execute(FEATURE_CHAT, generate_response, request.message)
if isinstance(result, str):
    text, metadata = result, {}
else:
    text, metadata = result
```

---

## High Severity Issues (20)

### API Module (2)

#### 2. IMessageSender Resource Leak
**File:** `api/routers/conversations.py`
**Lines:** 557, 674
**Issue:** IMessageSender instances are created in `send_message()` and `send_attachment()` but never closed/disposed of, causing resource leaks.

**Fix:** Add cleanup in finally block or use context manager.

#### 3. WebSocket Task Exception Handling Missing
**File:** `api/routers/websocket.py`
**Lines:** 464, 467
**Issue:** Background tasks created with `asyncio.create_task()` for generation without exception handlers. Exceptions are silently dropped.

```python
asyncio.create_task(_handle_generate(client, data, stream=False))
```

**Fix:** Add done callback for error handling:
```python
task = asyncio.create_task(_handle_generate(client, data, stream=False))
task.add_done_callback(lambda t: _log_task_exception(t, client.client_id))
```

### Models Module (2)

#### 4. Thread Safety Issue in CustomTemplateStore
**File:** `models/templates.py`
**Lines:** 313-327
**Issue:** `_load()` is called in `__init__` without acquiring the lock, but modifies `self._templates`, creating a race condition during initialization.

**Fix:** Load templates before creating the lock or ensure thread-safe initialization.

#### 5. Missing Type Validation for ThreadTopic
**File:** `models/generator.py`
**Lines:** 416-438
**Issue:** `_get_thread_examples()` uses `topic.value` without verifying `topic` is a ThreadTopic enum, causing silent failures with incorrect data.

**Fix:** Add validation: `assert isinstance(topic, ThreadTopic)`

### Integrations Module (1)

#### 6. Unsafe Regex Match Assumption
**File:** `integrations/calendar/detector.py`
**Lines:** 264, 275, 299
**Issue:** Calls `re.search(pattern, text_lower).group()` without null check. If the second `re.search()` returns None, calling `.group()` raises `AttributeError`.

**Fix:** Store match object from first search and reuse it:
```python
match = re.search(pattern, text_lower)
if match:
    original_text = match.group()
```

### Core Module (2)

#### 7. Fallback Exception Not Caught
**File:** `core/health/degradation.py`
**Lines:** 273-290
**Issue:** `_execute_fallback()` doesn't wrap the fallback call in try/except. If fallback also fails, exception propagates uncaught.

**Fix:** Add exception handling in `_execute_fallback()`.

#### 8. Cache Stampede Risk in Schema Detector
**File:** `core/health/schema.py`
**Lines:** 69-124
**Issue:** Double-lock pattern releases lock before expensive database operation, allowing multiple threads to start detection simultaneously.

**Fix:** Mark cache entry as "in-progress" to prevent stampede.

### MCP Server Module (3)

#### 9. Generator Not Thread-Safe
**File:** `mcp_server/handlers.py`
**Lines:** 214, 309, 324
**Issue:** `get_generator()` is thread-safe, but `generator.generate()` calls may race if MLX model isn't thread-safe, potentially corrupting state.

**Fix:** Use lock around generation calls or document thread safety requirements.

#### 10. Assertions Used for Runtime Validation
**File:** `mcp_server/handlers.py`
**Lines:** 193, 303, 540
**Issue:** Uses `assert chat_id is not None` for runtime validation. Assertions can be disabled with `-O` flag, causing crashes later.

**Fix:** Replace with explicit runtime checks:
```python
if chat_id is None:
    return ToolResult(success=False, error="Failed to resolve conversation ID")
```

#### 11. Missing Initialization Check
**File:** `mcp_server/server.py`
**Lines:** 189-238
**Issue:** `handle_request()` doesn't verify server is initialized before processing tool calls, allowing clients to bypass initialization handshake.

**Fix:** Add initialization check for non-initialize requests.

### Benchmarks Module (3)

#### 12. Protocol Mismatch in HHEMEvaluator
**File:** `benchmarks/hallucination/hhem.py`
**Lines:** 48-51
**Issue:** `run_benchmark()` signature takes `dataset: list[tuple[str, str, str]]` but contract expects `dataset_path: str`.

**Fix:** Align implementation with contract or update contract.

#### 13. Accessing Non-Existent Field
**File:** `benchmarks/memory/dashboard.py`
**Lines:** 238-243
**Issue:** Export references `s.available_gb` which doesn't exist on MemorySample dataclass.

**Fix:** Use correct field names from MemorySample.

#### 14. Hardcoded Latency Split
**File:** `benchmarks/latency/run.py`
**Lines:** 185-188
**Issue:** Uses hardcoded 10%/90% split for prefill/generation time which is inaccurate.

**Fix:** Measure prefill and generation separately or remove approximations.

### Tests Module (5)

#### 15. Race Condition in Timing Tests
**File:** `tests/unit/test_degradation.py`
**Lines:** 106-118
**Issue:** Test sleeps 150ms to wait for 100ms timeout - margin too small for slow CI systems.

**Fix:** Increase sleep to 5x timeout value.

#### 16. Mock Side Effect Doesn't Propagate Exceptions
**File:** `tests/integration/test_api_integration.py`
**Lines:** 302-304
**Issue:** Mock `side_effect = lambda feature, func, *args: func(*args)` doesn't properly handle exceptions from func.

**Fix:** Add proper error handling in mock.

#### 17. WebSocket Resource Leak in Tests
**File:** `tests/unit/test_websocket.py`
**Lines:** 40-182
**Issue:** Async tests create WebSocket connections but never disconnect them.

**Fix:** Add cleanup in fixture or use context managers.

#### 18. Wrong Expected Value in Circuit Breaker Test
**File:** `tests/unit/test_degradation.py`
**Lines:** 150-176
**Issue:** Test expects `record_success()` in CLOSED state to NOT reset failure count, which may not match actual behavior.

**Fix:** Verify expected behavior against implementation.

#### 19. Rate Limit Test Always Passes
**File:** `tests/integration/test_api_integration.py`
**Lines:** 226-258
**Issue:** Test ends with `assert True` which always passes regardless of rate limiting behavior.

**Fix:** Replace with actual assertion about rate limit responses.

### Jarvis Core Module (2)

#### 20. Incorrect Unanswered Conversation Logic
**File:** `jarvis/digest.py`
**Lines:** 287-341
**Issue:** `_find_unanswered()` checks if there are NO replies after last message from others, but doesn't properly track the last exchange.

**Fix:** Restructure logic to properly track conversation flow.

#### 21. Race Condition in Embedding Model Loading
**File:** `jarvis/embeddings.py`
**Lines:** 141-189
**Issue:** Double-check locking doesn't handle failed loads - no sentinel value stored to prevent repeated expensive failures.

**Fix:** Store sentinel value on failure.

---

## Medium Severity Issues (84)

### API Module (13)

| # | File | Lines | Issue |
|---|------|-------|-------|
| 1 | `api/routers/websocket.py` | 240-260 | No type/range validation for WebSocket parameters |
| 2 | `api/routers/attachments.py` | 344-361 | Path traversal uses string prefix (use `relative_to()`) |
| 3 | `api/routers/health.py` | 94-106 | Missing null checks on loader/info |
| 4 | `api/routers/settings.py` | 152-163 | Overly broad exception handling |
| 5 | `api/routers/batch.py` | 246-259 | Resource leak potential in batch operations |
| 6 | `api/routers/contacts.py` | 135-140 | Race condition in double-checked locking |
| 7 | `api/routers/attachments.py` | 457 | Undefined variable after exception |
| 8 | `api/dependencies.py` | 23-51 | Singleton without async locking |
| 9 | `api/ratelimit.py` | 257-258 | Timeout constants won't update with config |
| 10 | `api/routers/search.py` | 379, 413 | Unchecked cache close |
| 11 | `api/routers/custom_templates.py` | 291-292 | Missing type validation for group sizes |
| 12 | `api/routers/contacts.py` | 136-140 | Threading lock in async context |
| 13 | `api/routers/topics.py` | Various | Integer division issues in score calculations |

### Models Module (7)

| # | File | Lines | Issue |
|---|------|-------|-------|
| 1 | `models/__init__.py` | 75-93 | Race condition in singleton pattern |
| 2 | `models/loader.py` | 439-579 | Resource leak in generate_stream exception handling |
| 3 | `models/templates.py` | 2130-2134 | Similarity boost allows sub-threshold matches |
| 4 | `models/loader.py` | 608-622 | Fragile null check for model spec |
| 5 | `models/templates.py` | 1900-1938 | Non-atomic state assignments |
| 6 | `models/generator.py` | 395-396 | Missing null check for message text |
| 7 | `models/templates.py` | 1927-1938 | Memory leak if exception during encoding |

### Integrations Module (6)

| # | File | Lines | Issue |
|---|------|-------|-------|
| 1 | `integrations/imessage/avatar.py` | 140-167 | Dead code block (first query unused) |
| 2 | `integrations/calendar/detector.py` | 369-372 | Off-by-one in weekday calculation |
| 3 | `integrations/imessage/parser.py` | 472-513 | Inconsistent null handling in phone normalization |
| 4 | `integrations/imessage/avatar.py` | 136-198 | Cursor not closed on exception |
| 5 | `integrations/imessage/reader.py` | 260-261 | Type ignore comments hide errors |
| 6 | `integrations/imessage/reader.py` | 714-755 | Brittle string-based query fallback |

### Core Module (7)

| # | File | Lines | Issue |
|---|------|-------|-------|
| 1 | `core/health/circuit.py` | 139-162 | TOCTOU race in can_execute() |
| 2 | `core/health/degradation.py` | 145-197 | Fragile TypeError detection via string |
| 3 | `core/health/permissions.py` | 153-182 | Returns True when chat.db doesn't exist |
| 4 | `core/health/permissions.py` | 104-109 | Cache timestamp sync issue |
| 5 | `core/health/schema.py` | 233-234 | Missing schema version validation |
| 6 | `core/memory/controller.py` | 68-86 | Pressure callback without lock |
| 7 | `core/memory/controller.py` | 221-237 | Callbacks receive stale pressure level |

### MCP Server Module (13)

| # | File | Lines | Issue |
|---|------|-------|-------|
| 1 | `mcp_server/handlers.py` | 62-69 | Broad exception handling masks errors |
| 2 | `mcp_server/handlers.py` | 26-53 | Datetime parsing returns None silently |
| 3 | `mcp_server/handlers.py` | 256, 366 | Missing null safety for participant_names |
| 4 | `mcp_server/handlers.py` | 338 | Temperature can exceed model limits |
| 5 | `mcp_server/handlers.py` | 108 | Missing type validation for limit |
| 6 | `mcp_server/handlers.py` | 112 | Boolean filter logic ambiguity |
| 7 | `mcp_server/handlers.py` | 154, 262, 374 | Raw exceptions exposed to clients |
| 8 | `mcp_server/handlers.py` | 283-284 | Unvalidated num_suggestions |
| 9 | `mcp_server/server.py` | 276-281 | No validation JSON-RPC method is string |
| 10 | `mcp_server/handlers.py` | 82 | Import inside function |
| 11 | `mcp_server/server.py` | 164-168 | JSON serialization assumptions |
| 12 | `mcp_server/handlers.py` | 214 | Generator state not checked |
| 13 | `mcp_server/server.py` | 327-336 | No error rate limiting |

### Benchmarks Module (10)

| # | File | Lines | Issue |
|---|------|-------|-------|
| 1 | `benchmarks/memory/dashboard.py` | 98-105 | Race condition modifying sampler._interval |
| 2 | `benchmarks/memory/dashboard.py` | 174-175 | Off-by-one in chart rendering |
| 3 | `benchmarks/memory/dashboard.py` | 156-157 | Division by zero edge case |
| 4 | `benchmarks/latency/run.py` | 268-270 | JIT outlier exclusion inconsistent |
| 5 | `benchmarks/latency/run.py` | 259-261 | Failed iterations silently excluded |
| 6 | `benchmarks/latency/run.py` | 275-277 | P99 unreliable with 10 samples |
| 7 | `benchmarks/latency/run.py` | 268-279 | Std dev on different sample set |
| 8 | `benchmarks/latency/run.py` | 493-504 | Cleanup not called on failure |
| 9 | `benchmarks/memory/profiler.py` | 261-299 | Exceptions not logged |
| 10 | `benchmarks/latency/timer.py` | 153-162 | Ineffective warmup timer |

### Contracts/Scripts (12)

| # | File | Lines | Issue |
|---|------|-------|-------|
| 1 | `contracts/__init__.py` | 55-96 | Missing exports for Attachment, Reaction |
| 2 | `contracts/health.py` | 51-59 | Callable signature mismatch |
| 3 | `contracts/models.py` | 11-19 | No temperature validation |
| 4 | `contracts/models.py` | 17 | No max_tokens bounds |
| 5 | `contracts/models.py` | 32 | finish_reason should be Literal |
| 6 | `contracts/hallucination.py` | 11-19 | HHEM score range not enforced |
| 7 | `contracts/latency.py` | 29-39 | Percentile ordering not enforced |
| 8 | `contracts/calendar.py` | 20-21 | No end >= start validation |
| 9 | `contracts/calendar.py` | 41-42 | No end >= start validation |
| 10 | `contracts/calendar.py` | 92-106 | Batch ordering not guaranteed |
| 11 | `scripts/check_gates.py` | 74 | Missing p95_ms field check |
| 12 | `scripts/generate_report.py` | 24 | Wrong type annotation |

### Jarvis Core Module (7)

| # | File | Lines | Issue |
|---|------|-------|-------|
| 1 | `jarvis/evaluation.py` | 99-127 | Missing index bounds check in percentile |
| 2 | `jarvis/experiments.py` | 399-410 | Variant assignment logic concerns |
| 3 | `jarvis/evaluation.py` | 537-540 | Chi-squared assumptions not validated |
| 4 | `jarvis/context.py` | 265-266 | Timezone mismatch |
| 5 | `jarvis/config.py` | 213-229 | Config migration not handling empty sections |
| 6 | `jarvis/embeddings.py` | 392-434 | No limit on total messages indexed |
| 7 | `jarvis/experiments.py` | 617-625 | Incomplete gamma bounds checking |

### Tests Module (9)

| # | File | Lines | Issue |
|---|------|-------|-------|
| 1 | `tests/unit/test_latency.py` | 77-84 | Timing-dependent assertion |
| 2 | `tests/unit/test_websocket.py` | 31-34 | Mock not resetting singletons |
| 3 | `tests/integration/test_websocket_api.py` | 239-263 | Test accepts multiple outcomes |
| 4 | `tests/integration/test_api_integration.py` | 618-642 | No roundtrip persistence verification |
| 5 | `tests/unit/test_memory_profiler.py` | 103-124 | Mock doesn't cover error paths |
| 6 | `tests/integration/test_api_integration.py` | 998 | Empty placeholder test |
| 7 | `tests/integration/test_api_integration.py` | 814-826 | Concurrent test doesn't verify concurrency |
| 8 | `tests/unit/test_config.py` | 56-67 | Missing lite_mode_mb assertion |
| 9 | `tests/unit/test_websocket.py` | 98-114 | Broadcast doesn't verify message content |

---

## Low Severity Issues (68)

Low severity issues include:
- Unused imports and dead code
- Missing documentation and docstrings
- Inconsistent logging formats
- Code style inconsistencies
- Type annotation improvements
- Performance micro-optimizations
- Unreachable code branches
- Inefficient cache key generation
- Missing test edge cases

See individual module sections for complete details.

---

## Key Patterns Identified

### 1. Resource Leaks
Multiple instances of database cursors, WebSocket connections, and model instances not being properly closed:
- `api/routers/conversations.py` - IMessageSender
- `integrations/imessage/avatar.py` - SQLite cursors
- `tests/unit/test_websocket.py` - WebSocket connections
- `benchmarks/latency/run.py` - Model cleanup on exception

### 2. Race Conditions
Singleton patterns and shared state without proper synchronization:
- `api/routers/contacts.py` - Avatar cache
- `api/dependencies.py` - Reader singleton
- `models/__init__.py` - Generator singleton
- `core/memory/controller.py` - Pressure callbacks

### 3. Silent Failures
Broad exception handling that swallows errors:
- `mcp_server/handlers.py` - Multiple handlers
- `api/routers/settings.py` - Model check
- `integrations/imessage/reader.py` - Database close

### 4. Type Safety Gaps
Missing validation on dictionary access and type assumptions:
- `api/routers/websocket.py` - Generation parameters
- `mcp_server/handlers.py` - Tool parameters
- `api/routers/custom_templates.py` - Group sizes

### 5. Path Security
String-based path validation instead of proper path resolution:
- `api/routers/attachments.py` - Use `Path.relative_to()` instead

### 6. Flaky Tests
Tests with timing dependencies or assertions that always pass:
- `tests/unit/test_degradation.py` - Tight sleep margins
- `tests/integration/test_api_integration.py` - `assert True`

---

## Recommendations

### Immediate Actions (This Sprint)
1. Fix the critical type unpacking error in `jarvis/api.py`
2. Add resource cleanup for IMessageSender in `api/routers/conversations.py`
3. Add exception handling to WebSocket tasks
4. Replace `assert` with proper runtime checks in MCP handlers
5. Fix path traversal vulnerability in attachments router

### Short-Term Actions (Next 2 Sprints)
1. Implement proper async singleton pattern for shared resources
2. Add type validation for all external inputs (WebSocket, MCP)
3. Fix all race conditions in singleton patterns
4. Add missing null checks throughout codebase
5. Fix flaky tests with proper timing margins

### Long-Term Improvements
1. Add comprehensive integration tests for threading scenarios
2. Implement proper async database interface
3. Add request concurrency limits with semaphores
4. Create coding guidelines document for common patterns
5. Add mutation testing to catch tests that pass for wrong reasons

---

## Appendix: Files Reviewed

```
api/__init__.py
api/dependencies.py
api/errors.py
api/main.py
api/ratelimit.py
api/schemas.py
api/routers/__init__.py
api/routers/attachments.py
api/routers/batch.py
api/routers/calendar.py
api/routers/contacts.py
api/routers/conversations.py
api/routers/custom_templates.py
api/routers/digest.py
api/routers/drafts.py
api/routers/embeddings.py
api/routers/experiments.py
api/routers/export.py
api/routers/feedback.py
api/routers/health.py
api/routers/insights.py
api/routers/metrics.py
api/routers/pdf_export.py
api/routers/priority.py
api/routers/quality.py
api/routers/relationships.py
api/routers/search.py
api/routers/settings.py
api/routers/stats.py
api/routers/suggestions.py
api/routers/tasks.py
api/routers/template_analytics.py
api/routers/threads.py
api/routers/topics.py
api/routers/websocket.py
benchmarks/__init__.py
benchmarks/hallucination/__init__.py
benchmarks/hallucination/datasets.py
benchmarks/hallucination/hhem.py
benchmarks/hallucination/run.py
benchmarks/latency/__init__.py
benchmarks/latency/run.py
benchmarks/latency/scenarios.py
benchmarks/latency/timer.py
benchmarks/memory/__init__.py
benchmarks/memory/dashboard.py
benchmarks/memory/models.py
benchmarks/memory/profiler.py
benchmarks/memory/run.py
contracts/__init__.py
contracts/calendar.py
contracts/hallucination.py
contracts/health.py
contracts/imessage.py
contracts/latency.py
contracts/memory.py
contracts/models.py
core/__init__.py
core/health/__init__.py
core/health/circuit.py
core/health/degradation.py
core/health/permissions.py
core/health/schema.py
core/memory/__init__.py
core/memory/controller.py
core/memory/monitor.py
integrations/__init__.py
integrations/calendar/__init__.py
integrations/calendar/detector.py
integrations/calendar/reader.py
integrations/calendar/writer.py
integrations/imessage/__init__.py
integrations/imessage/avatar.py
integrations/imessage/parser.py
integrations/imessage/queries.py
integrations/imessage/reader.py
integrations/imessage/sender.py
jarvis/__init__.py
jarvis/__main__.py
jarvis/api_models.py
jarvis/api.py
jarvis/cli_examples.py
jarvis/cli.py
jarvis/config.py
jarvis/context.py
jarvis/digest.py
jarvis/embeddings.py
jarvis/errors.py
jarvis/evaluation.py
jarvis/experiments.py
jarvis/export.py
jarvis/fallbacks.py
jarvis/generation.py
jarvis/insights.py
jarvis/intent.py
jarvis/metrics.py
jarvis/pdf_generator.py
jarvis/priority.py
jarvis/prompts.py
jarvis/quality_metrics.py
jarvis/relationships.py
jarvis/retry.py
jarvis/semantic_search.py
jarvis/setup.py
jarvis/system.py
jarvis/tasks/__init__.py
jarvis/tasks/models.py
jarvis/tasks/queue.py
jarvis/tasks/worker.py
jarvis/threading.py
mcp_server/__init__.py
mcp_server/handlers.py
mcp_server/server.py
mcp_server/tools.py
models/__init__.py
models/generator.py
models/loader.py
models/prompt_builder.py
models/registry.py
models/templates.py
scripts/check_gates.py
scripts/generate_report.py
tests/__init__.py
tests/conftest.py
tests/fixtures/__init__.py
tests/integration/__init__.py
tests/integration/conftest.py
tests/integration/test_api_integration.py
tests/integration/test_api.py
tests/integration/test_cli.py
tests/integration/test_e2e_script.py
tests/integration/test_export_api.py
tests/integration/test_rag_flow.py
tests/integration/test_websocket_api.py
tests/test_mcp_server.py
tests/test_tasks.py
tests/unit/__init__.py
tests/unit/test_api_async_ratelimit.py
tests/unit/test_avatar.py
tests/unit/test_calendar.py
tests/unit/test_config.py
tests/unit/test_context.py
tests/unit/test_custom_templates.py
tests/unit/test_degradation.py
tests/unit/test_digest.py
tests/unit/test_drafts_api.py
tests/unit/test_embeddings.py
tests/unit/test_errors.py
tests/unit/test_evaluation.py
tests/unit/test_export.py
tests/unit/test_fallbacks.py
tests/unit/test_feedback_api.py
tests/unit/test_generation.py
tests/unit/test_generator.py
tests/unit/test_health_api.py
tests/unit/test_hhem.py
tests/unit/test_imessage.py
tests/unit/test_insights.py
tests/unit/test_intent.py
tests/unit/test_latency.py
tests/unit/test_memory_controller.py
tests/unit/test_memory_profiler.py
tests/unit/test_metrics_api.py
tests/unit/test_metrics.py
tests/unit/test_permissions.py
tests/unit/test_priority.py
tests/unit/test_prompts.py
tests/unit/test_quality_api.py
tests/unit/test_quality_metrics.py
tests/unit/test_registry.py
tests/unit/test_relationships.py
tests/unit/test_retry.py
tests/unit/test_schema.py
tests/unit/test_semantic_search.py
tests/unit/test_settings_api.py
tests/unit/test_setup.py
tests/unit/test_suggestions_api.py
tests/unit/test_templates.py
tests/unit/test_threaded_generation.py
tests/unit/test_threading.py
tests/unit/test_websocket.py
```

---

*Report generated by Claude Code automated analysis. Manual review recommended for critical and high severity issues before implementing fixes.*
