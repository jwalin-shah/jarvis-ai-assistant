# JARVIS AI Assistant - Comprehensive Code Review

**Review Date:** 2026-02-09
**Reviewer:** Claude Code (Opus 4.6)
**Scope:** Full codebase (~355 Python files, ~98K LOC)

---

## Executive Summary

Comprehensive review of the entire JARVIS codebase covering: `jarvis/`, `api/`, `core/`, `contracts/`, `models/`, `integrations/`, `evals/`, `experiments/`, `scripts/`, and root-level files. Each file was read and analyzed for bugs, security, dead code, error handling, performance, type safety, and code quality.

### Findings by Severity

| Severity | Count |
|----------|-------|
| **Critical** | 7 |
| **High** | 22 |
| **Medium** | 38 |
| **Low** | 43 |
| **Total** | **110** |

---

## Critical Issues (7)

### C-1. NameError: `logger` undefined in text_normalizer.py
**File:** `jarvis/text_normalizer.py:319`
**Category:** Bug (Runtime Crash)

`logger.warning(...)` is called but `logger` is never defined in this module. No `import logging` or `logger = logging.getLogger(...)` exists. **Will raise `NameError` at runtime** when the spell checker fails to load.

**Proposed fix:** Add `import logging` and `logger = logging.getLogger(__name__)` at the top of the file.

---

### C-2. get_imessage_reader() called directly instead of via Depends()
**File:** `api/routers/priority.py:446`
**Category:** Bug (Runtime Crash)

`get_imessage_reader()` is called directly as a function instead of via FastAPI's `Depends()`. Since `get_imessage_reader` is a generator function (uses `yield`), calling it directly returns a generator object, NOT a `ChatDBReader`. Every method call on `reader` (e.g., `reader.check_access()` at line 449, `reader.get_conversations()` at line 458) **will fail with `AttributeError`**. The reader will also never be closed since the generator's `finally` block is never triggered.

**Proposed fix:** Change the endpoint signature to accept `reader` via `Depends(get_imessage_reader)`.

---

### C-3. Deadlock in TemplateAnalytics.get_stats()
**File:** `jarvis/metrics.py:812-813`
**Category:** Bug (Deadlock)

`get_stats()` acquires `self._lock` (line 805), then calls `self.get_hit_rate()` and `self.get_cache_hit_rate()` which each also try to acquire `self._lock`. Since `threading.Lock` is non-reentrant, **this will deadlock** on every call to `get_stats()`.

**Proposed fix:** Use `threading.RLock()` instead of `threading.Lock()`, or inline the hit rate calculations directly in `get_stats()` to avoid nested lock acquisition.

---

### C-4. AppleScript injection via unescaped chat_id
**File:** `jarvis/scheduler/executor.py:311`
**Category:** Security (Code Injection)

`chat_id` is interpolated directly into an AppleScript string without escaping. A malformed `chat_id` containing `"` characters could break the script or execute arbitrary AppleScript (which can run shell commands via `do shell script`).

**Proposed fix:** Apply the same escaping to `chat_id` as is applied to `escaped_text` on lines 301-303.

---

### C-5. AppleScript injection in calendar reader
**File:** `integrations/calendar/reader.py:183`
**Category:** Security (Code Injection)

`calendar_id` is interpolated into AppleScript via `.format(calendar_id=calendar_id)` without escaping. The template at line 69 uses `'set targetCalendars to (calendars whose id is "{calendar_id}")'`. A malicious `calendar_id` containing `"` followed by AppleScript commands could escape the string literal and execute arbitrary code.

**Proposed fix:** Apply `_escape_applescript()` to `calendar_id` before interpolation.

---

### C-6. AppleScript injection in calendar writer
**File:** `integrations/calendar/writer.py:144`
**Category:** Security (Code Injection)

`calendar_id` and `title` are interpolated into AppleScript template via `template.format()` without escaping. Other fields (`location`, `notes`, `url`) are properly escaped via `self._escape_applescript()` at lines 130-134, but `calendar_id` and `title` are passed raw.

**Proposed fix:** Escape both: `calendar_id=self._escape_applescript(calendar_id), title=self._escape_applescript(title)`.

---

### C-7. Protocol/implementation callback signature mismatch
**File:** `contracts/memory.py:157` vs `core/memory/controller.py:186`
**Category:** Bug (Type Contract Violation)

Protocol defines `register_pressure_callback(callback: Callable[[], None])` (zero args), but the implementation defines `register_pressure_callback(callback: Callable[[str], None])` (receives pressure level string). Code written against the Protocol will crash when the controller invokes callbacks with a string argument via `_notify_callbacks()` at line 239.

**Proposed fix:** Update Protocol in `contracts/memory.py:157` to `Callable[[str], None]` to match the implementation.

---

## High Severity Issues (22)

### H-1. Race condition in get_generator() double-checked locking
**File:** `models/__init__.py:115-134`
**Category:** Concurrency

The unprotected check `if _generator is None` at line 115 races with `reset_generator()` which sets `_generator = None` under lock. The `else` branch can return `None` instead of an `MLXGenerator`.

**Proposed fix:** Always acquire the lock before inspecting `_generator`.

---

### H-2. Lock initialized as None (race condition)
**File:** `api/routers/search.py:44-45`
**Category:** Concurrency

`_searcher_lock` is initialized as `None` and conditionally set inside `_get_searcher()`. Two threads could simultaneously see `None`, each create their own lock, and both enter the critical section.

**Proposed fix:** Initialize at module level: `_searcher_lock = threading.Lock()`.

---

### H-3. Lock created inside function (useless)
**File:** `api/routers/search.py:413, 457`
**Category:** Concurrency

`_cache_stats_lock = threading.Lock()` is created INSIDE `get_cache_stats()`, meaning a **new lock per call**. Same at line 457 with `_cache_clear_lock`.

**Proposed fix:** Move both lock definitions to module level.

---

### H-4. Route ordering: DELETE /completed/clear shadowed
**File:** `api/routers/tasks.py:515`
**Category:** Bug

`DELETE /completed/clear` is registered AFTER `DELETE /{task_id}` at line 412. FastAPI matches in registration order, so `DELETE /tasks/completed/clear` matches `/{task_id}` with `task_id="completed"` first. **The clear endpoint is unreachable.**

**Proposed fix:** Move `/completed/clear` route before `/{task_id}`.

---

### H-5. Settings partial update overwrites all fields
**File:** `api/routers/settings.py:361-366`
**Category:** Bug

Partial update for generation/behavior settings replaces the entire sub-object with `model_dump()`. Sending `{"generation": {"temperature": 0.8}}` will reset `max_tokens_reply` and `max_tokens_summary` to Pydantic defaults, overwriting the user's saved values.

**Proposed fix:** Merge with `model_dump(exclude_unset=True)` instead of full replacement.

---

### H-6. WebSocket auth token logged at INFO level
**File:** `api/routers/websocket.py:37`
**Category:** Security

`logger.info("Generated WebSocket auth token: %s", _WS_AUTH_TOKEN)` exposes the authentication credential in log files.

**Proposed fix:** Log that a token was generated without the actual value, or use DEBUG level.

---

### H-7. Arbitrary file path in send_attachment
**File:** `api/routers/conversations.py:690`
**Category:** Security

`send_attachment` accepts arbitrary `file_path` with no validation. A caller could specify `/etc/passwd` or `~/.ssh/id_rsa` and have it sent via iMessage, enabling data exfiltration.

**Proposed fix:** Validate path against an allow-list of directories.

---

### H-8. Arbitrary output_dir in batch export
**File:** `api/routers/batch.py:53`
**Category:** Security

`output_dir` field accepts arbitrary filesystem paths with no validation.

**Proposed fix:** Validate against an allow-list or restrict to `~/.jarvis/exports/`.

---

### H-9. Singleton searcher holds closed reader
**File:** `api/routers/search.py:362`
**Category:** Bug (Resource Leak)

Singleton `_searcher_instance` is created with a `reader` from dependency injection. That reader is closed when the request ends (DI's `finally` block). The singleton then holds a closed DB connection; subsequent searches fail.

**Proposed fix:** Create a dedicated reader for the singleton (not from DI), or recreate per-request.

---

### H-10. reset_metrics() doesn't reset TTL caches
**File:** `jarvis/metrics.py:582`
**Category:** Bug

`reset_metrics()` resets `_template_analytics` but NOT the TTL cache globals (`_conversation_cache`, `_health_cache`, `_model_info_cache`). A "full reset" leaves stale cached data.

**Proposed fix:** Add these caches to the reset function.

---

### H-11. Debug print() statements in production code
**File:** `models/bert_embedder.py:219-280, 384-411`
**Category:** Code Quality

Ten `print()` statements with `[BERT]` prefix dump verbose memory diagnostics to stdout on every BERT model load. Uses `import psutil` solely for these debug prints.

**Proposed fix:** Remove all debug `print()` calls and the `psutil` imports. Use `logger.debug()` if needed.

---

### H-12. Thread-unsafe write in memory controller
**File:** `core/memory/controller.py:220`
**Category:** Concurrency

`set_model_loaded()` writes `self._model_loaded = loaded` without lock protection. Called from external components (any thread).

**Proposed fix:** Wrap in `self._lock`.

---

### H-13. Unreliable __del__ for resource cleanup
**File:** `jarvis/router.py:251-253`
**Category:** Resource Leak

`__del__` calls `self.close()` but `__del__` is unreliable in Python - may never be called, or may run during interpreter shutdown when referenced objects are already gone.

**Proposed fix:** Remove `__del__` and rely on explicit `close()` calls.

---

### H-14. reset_reply_router() leaks iMessage reader
**File:** `jarvis/router.py:478-486`
**Category:** Resource Leak

Sets `_router = None` without calling `close()` on the existing router instance. The iMessage reader DB connection leaks.

**Proposed fix:** Call `_router.close()` before setting `_router = None`.

---

### H-15. Hard import of optional mem0 dependency
**File:** `jarvis/memory_layer.py:12`
**Category:** Bug (Import Crash)

`from mem0 import Memory` at module level will crash any module that imports `memory_layer` if `mem0` is not installed.

**Proposed fix:** Wrap in try/except or make lazy import.

---

### H-16. O(n^2) Louvain implementation
**File:** `jarvis/graph/clustering.py:215-249`
**Category:** Performance

Inner loops iterate all nodes for `sum_to_current`, neighbor communities, and `sum_to_comm` calculations. Inside a `while improved` loop (up to 100 iterations), effective complexity is O(100 * n^2).

**Proposed fix:** Pre-build neighbor adjacency dict; only iterate non-zero weight neighbors.

---

### H-17. O(n^2) modularity calculation
**File:** `jarvis/graph/clustering.py:263-268`
**Category:** Performance

Double `for i in range(n): for j in range(n):` loop over all node pairs for final modularity.

**Proposed fix:** Compute incrementally using community_weight accumulators.

---

### H-18. O(n^2) force-directed layout
**File:** `jarvis/graph/layout.py:127-152`
**Category:** Performance

Nested all-pairs repulsion per iteration (100 iterations default). O(100 * n^2).

**Proposed fix:** Barnes-Hut approximation or spatial hashing for >100 nodes.

---

### H-19. O(n*m) cross-reference scan
**File:** `jarvis/observability/metrics_validation.py:77-84`
**Category:** Performance

For each audit entry, every DB entry is scanned linearly.

**Proposed fix:** Build dict keyed by `query_hash` for O(1) lookup.

---

### H-20. Inconsistent error handling for missing API key
**File:** `evals/judge_config.py:43-49 vs 53-63`
**Category:** Error Handling

`get_judge_api_key()` calls `sys.exit(1)` (hard crash), while `get_judge_client()` returns `None` for the same condition.

**Proposed fix:** Make `get_judge_api_key()` return `None` or raise an exception instead of `sys.exit(1)`.

---

### H-21. WebSocket auth token via URL query parameter
**File:** `jarvis/socket_server.py:577`
**Category:** Security

Auth token passed via `?token=...`. Query parameters appear in access logs, browser history, and referrer headers.

**Proposed fix:** Use WebSocket subprotocol header or first-message authentication.

---

### H-22. Duplicate text silently dropped in prepare_data
**File:** `experiments/trigger/prepare_data.py:108`
**Category:** Bug

Building `text_to_example` dict silently drops duplicate texts with different labels.

**Proposed fix:** Detect and warn on duplicates before overwriting.

---

## Medium Severity Issues (38)

### M-1. Prompt injection sanitization has gaps
**File:** `api/routers/drafts.py:106-113`
**Category:** Security

Detection uses `lower_instruction` (case-insensitive) but replacement uses `instruction.replace(pattern, ...)` (case-sensitive). Mixed-case patterns like `"Ignore previous"` won't be replaced.

**Proposed fix:** Use `re.sub()` with `re.IGNORECASE` flag.

---

### M-2. WebSocket auth via query parameter logged
**File:** `api/routers/websocket.py:65`
**Category:** Security

Auth token in query params appears in server access logs, proxy logs, and referrer headers.

**Proposed fix:** Document header-based auth (`X-WS-Token`) as preferred; deprecate query param auth.

---

### M-3. WebSocket get_client() reads without lock
**File:** `api/routers/websocket.py:248`
**Category:** Concurrency

Accesses `self._clients` dict without acquiring the async lock while mutations use the lock.

**Proposed fix:** Acquire `self._lock` in `get_client()`.

---

### M-4. Generic exception handler disabled
**File:** `api/errors.py:366`
**Category:** Security

Generic handler is commented out. Unhandled exceptions may expose stack traces.

**Proposed fix:** Uncomment or set `debug=False` on the app.

---

### M-5. Timezone-naive datetime throughout codebase
**Files:** `api/routers/embeddings.py:311`, `api/routers/calendar.py:137`, `jarvis/priority.py:566`, `jarvis/watcher.py:614`, `jarvis/graph/builder.py:468`, `models/templates.py:262-263`
**Category:** Bug

`datetime.now()` and `datetime.fromtimestamp()` used without timezone info. Mixing with timezone-aware values raises `TypeError`.

**Proposed fix:** Use `datetime.now(UTC)` and `datetime.fromtimestamp(ts, tz=UTC)` consistently.

---

### M-6. Thread API contract violation
**File:** `api/routers/threads.py:380-383`
**Category:** Bug

Returns empty list when thread not found, but OpenAPI spec declares 404. API contract violation.

**Proposed fix:** Raise `HTTPException(status_code=404)` or remove 404 from OpenAPI spec.

---

### M-7. Batch export format not validated
**File:** `api/routers/batch.py:49`
**Category:** Bug

`format` is a plain `str` with default `"json"`. No validation against supported formats.

**Proposed fix:** Use an enum or validator to restrict to `["json", "csv", "txt"]`.

---

### M-8. O(n) conversation scan for single chat_id
**Files:** `api/routers/stats.py:537-542`, `api/routers/export.py:69-74`
**Category:** Performance

Fetches 100-500 conversations via `get_conversations()` then linearly scans for target `chat_id`.

**Proposed fix:** Add `get_conversation(chat_id)` method for direct lookup.

---

### M-9. NER service stderr to DEVNULL
**File:** `jarvis/_cli_main.py:359`
**Category:** Error Handling

NER subprocess started with stderr to DEVNULL, making startup failure diagnosis impossible.

**Proposed fix:** Redirect stderr to `~/.jarvis/ner_server.log`.

---

### M-10. Config file written without restricted permissions
**File:** `jarvis/config.py:807`
**Category:** Security

`~/.jarvis/config.json` may be readable by other users on multi-user systems.

**Proposed fix:** Set `os.chmod(path, 0o600)` after writing.

---

### M-11. ValidationError shadows Pydantic's
**File:** `jarvis/errors.py:449`
**Category:** Code Quality

`jarvis.errors.ValidationError` shadows `pydantic.ValidationError` used in `jarvis/config.py:29`.

**Proposed fix:** Rename to `InputValidationError` or `JarvisValidationError`.

---

### M-12. Unsynchronized embeddings cache access
**File:** `jarvis/threading.py:508-511`
**Category:** Concurrency

`_embeddings_cache` (OrderedDict) read/written without lock. Concurrent calls can corrupt it.

**Proposed fix:** Wrap in `self._lock`.

---

### M-13. Thread-unsafe _handled_items mutation
**File:** `jarvis/priority.py:651`
**Category:** Concurrency

`mark_handled` modifies `self._handled_items` set without `self._lock`.

**Proposed fix:** Wrap mutations/reads with `self._lock`.

---

### M-14. Thread-unsafe optimized programs singleton
**File:** `jarvis/prompts.py:1746-1748`
**Category:** Concurrency

`_optimized_programs` accessed without lock. Two threads could trigger double initialization.

**Proposed fix:** Add `threading.Lock()` around initialization.

---

### M-15. Fragile locals() check
**File:** `jarvis/nlp/validity_gate.py:265`
**Category:** Bug

`if "response_words" not in locals()` depends on control flow analysis.

**Proposed fix:** Initialize `response_words` unconditionally at method start.

---

### M-16. Single-letter slang false positives
**File:** `jarvis/nlp/slang.py:23-24`
**Category:** Bug

Entries like `"y"`, `"n"`, `"b"`, `"c"`, `"k"` cause aggressive false-positive expansions. `"I got a B in math"` → `"I got a be in math"`.

**Proposed fix:** Remove single-letter entries or add context-aware filtering.

---

### M-17. Private lock access across classes
**File:** `jarvis/prefetch/executor.py:609, 650, 721, 730`
**Category:** Coupling

Accesses `MLXModelLoader._mlx_load_lock` (private) from another class.

**Proposed fix:** Expose a public `MLXModelLoader.gpu_lock()` accessor.

---

### M-18. Duplicated serialize/deserialize logic
**File:** `jarvis/prefetch/cache.py:472-531, 742-783`
**Category:** Code Quality

`L2Cache._serialize()` and `L3Cache._serialize()` are identical. Same for `_deserialize()`.

**Proposed fix:** Extract shared utility function.

---

### M-19. Global RNG state mutation
**Files:** `jarvis/graph/clustering.py:136`, `jarvis/graph/layout.py:70`
**Category:** Bug

`random.seed()` mutates global `random` state. Multi-threaded or concurrent calls produce unpredictable results.

**Proposed fix:** Use `random.Random(seed)` for isolated RNG.

---

### M-20. Deprecated preexec_fn
**File:** `jarvis/services/base.py:311`
**Category:** Deprecation

Uses `preexec_fn` in `subprocess.Popen()`, deprecated in Python 3.12.

**Proposed fix:** Use `process_group=0` on Python 3.11+.

---

### M-21. ProcessLookupError race
**File:** `jarvis/services/base.py:329, 342`
**Category:** Bug

`os.getpgid(self._process.pid)` can raise `ProcessLookupError` if process exited between check and kill.

**Proposed fix:** Wrap `os.killpg()` in `try/except ProcessLookupError`.

---

### M-22. Memory-unbounded backfill
**File:** `jarvis/search/vec_search.py:770-789`
**Category:** Performance

Accumulates ALL rows in `batch` list before `executemany`. For large tables, loads everything into memory.

**Proposed fix:** Chunk batches into groups of ~1000.

---

### M-23. signal.SIGALRM is Unix-only and main-thread-only
**File:** `integrations/imessage/reader.py:790-796`
**Category:** Portability/Bug

`signal.SIGALRM` raises `AttributeError` on non-Unix platforms and `ValueError` when called from non-main threads (e.g., web server worker threads).

**Proposed fix:** Use `threading.Timer` or `concurrent.futures` with timeout.

---

### M-24. Schema detection holds lock during I/O
**File:** `core/health/schema.py:72-166`
**Category:** Performance

`detect()` holds `self._lock` for the entire method including SQLite connect/query/close. Serializes concurrent first-time calls.

**Proposed fix:** Do I/O outside the lock; re-check cache inside lock.

---

### M-25. O(n^2) valid_indices check
**File:** `models/reranker.py:91`
**Category:** Performance

`if i in valid_indices` scans a list per iteration.

**Proposed fix:** Use a `set` for O(1) membership check.

---

### M-26. StopIteration on empty snapshots dir
**Files:** `models/cross_encoder.py:198`, `models/bert_embedder.py:335`
**Category:** Bug

`next(snapshots_dir.iterdir())` raises `StopIteration` if directory is empty.

**Proposed fix:** `snapshots = list(dir.iterdir()); if not snapshots: raise FileNotFoundError(...)`.

---

### M-27. Double-wrapped CachedEmbedder
**File:** `jarvis/reply_service.py:423`
**Category:** Performance

Creates `CachedEmbedder(get_embedder())` but `get_embedder()` already returns a `CachedEmbedder`. Double-wrapping wastes memory.

**Proposed fix:** Use `get_embedder()` directly.

---

### M-28. Approximate token count for DSPy
**File:** `jarvis/dspy_client.py:71`
**Category:** Bug

`completion_tokens = len(text.split())` counts words, not tokens. DSPy relies on accurate counts.

**Proposed fix:** Use tokenizer or document as approximation.

---

### M-29. Temp file leak on exception
**File:** `jarvis/embedding_adapter.py:259`
**Category:** Resource Leak

`tempfile.NamedTemporaryFile(delete=False)` only cleaned at method end (line 307). Exception during processing leaks the file.

**Proposed fix:** Use `try/finally` or `contextlib.ExitStack`.

---

### M-30. Race in model_warmer unload
**File:** `jarvis/model_warmer.py:343-348`
**Category:** Concurrency

`unload()` checks `generator.is_loaded()` and calls `generator.unload()` without holding lock.

**Proposed fix:** Wrap check-and-unload in `with self._lock:`.

---

### M-31. Private message text in debug logs
**File:** `jarvis/watcher.py:307`
**Category:** Security

Debug log includes `msg['text'][:50]`, exposing private iMessage content.

**Proposed fix:** Log message hash/length instead of content.

---

### M-32. WebSocket connection limit TOCTOU
**File:** `jarvis/socket_server.py:590`
**Category:** Concurrency

Connection limit check outside lock; multiple connections could bypass simultaneously.

**Proposed fix:** Move check inside `async with self._clients_lock:`.

---

### M-33. In-place request dict mutation
**File:** `jarvis/socket_server.py:1313-1316`
**Category:** Bug

`params.pop("stream", None)` mutates the caller's request dict.

**Proposed fix:** Copy params before popping keys.

---

### M-34. Docstring defaults don't match actual defaults
**File:** `experiments/scripts/prepare_data.py:56-69`
**Category:** Documentation/Bug

Docstring says `confidence_threshold` defaults to 0.90 but actual default is 0.80. Same for `minority_threshold`.

**Proposed fix:** Update docstrings to match actual defaults.

---

### M-35. Division by zero risk
**File:** `evals/eval_pipeline.py:292`
**Category:** Bug

`cat_matches / n * 100` crashes if `n == 0`.

**Proposed fix:** Guard: `if n == 0: return`.

---

### M-36. Exception hierarchy conflation
**File:** `jarvis/fallbacks.py:147-153, 155-160`
**Category:** Design

`GenerationTimeoutError` and `GenerationError` both inherit from `ModelLoadError`. Catching `ModelLoadError` also catches timeouts.

**Proposed fix:** Create a common `GenerationError` base class instead.

---

### M-37. Non-reproducible bootstrap CI
**File:** `experiments/trigger/final_eval.py:135`
**Category:** Bug

`bootstrap_ci` uses `random_state=None`, making CI bounds non-reproducible.

**Proposed fix:** Use seeded RNG.

---

### M-38. Stale data on resume in LLM labeling
**File:** `llm_label_dialog_clean.py:151-156`
**Category:** Bug

File mode logic always appends after first batch, even on fresh runs that could have stale data from prior interrupted runs.

**Proposed fix:** Truncate file on fresh runs (when existing_count == 0).

---

## Low Severity Issues (43)

### L-1. Duplicate emojis in frozenset
**File:** `jarvis/observability/insights.py:200,222`
Duplicates "😰" and "😭" in NEGATIVE_EMOJIS. Remove duplicates.

### L-2. MD5 for non-security hashing
**File:** `jarvis/search/embeddings.py:475`, `jarvis/threading.py:507`
Uses MD5 where blake2b would be faster. Replace for consistency.

### L-3. Misleading comment about lock scope
**File:** `jarvis/observability/metrics_router.py:253-254`
Comment says "write outside lock" but write is inside lock. Update comment.

### L-4. Non-deterministic cluster colors
**File:** `jarvis/graph/clustering.py:85-86`
`random.random()` in `get_cluster_colors()` is not seeded. Accept optional RNG parameter.

### L-5. Assumes contiguous rowid allocation
**File:** `jarvis/search/vec_search.py:453-455`
Computes chunk_rowids from range. Could fail under concurrent writers.

### L-6. Dead code: unused rate limit functions
**File:** `api/ratelimit.py:93-126`
`_rate_limit_enabled()` and `_get_rate_limit()` never called.

### L-7. Dead code: empty TYPE_CHECKING blocks
**Files:** `api/errors.py:38-39`, `api/routers/calendar.py:36-37`, `contracts/memory.py:13-14`, `contracts/health.py:13-14`, `contracts/latency.py:11-12`
Empty `if TYPE_CHECKING: pass` blocks.

### L-8. Private attribute access
**Files:** `api/routers/settings.py:125,165` (`generator._model`), `api/routers/priority.py:764` (`scorer._important_contacts`), `api/routers/health.py:98` (`generator._loader`), `jarvis/socket_server.py:343` (`generator._loader`)
Access private attrs instead of public methods.

### L-9. f-strings in logger calls
**Files:** `api/routers/settings.py:96,595,721`, `api/routers/attachments.py:360,400,457`, `integrations/imessage/reader.py:183`, `jarvis/graph/builder.py:258,351`
Eager f-string evaluation even when log level disabled. Use `%s` formatting.

### L-10. Missing exception chain (from e)
**Files:** `api/routers/attachments.py:355`, `api/routers/feedback.py:546`
`raise HTTPException(...)` inside except without `from e`.

### L-11. Redundant slice
**File:** `api/routers/priority.py:474`
`conversations[:15]` redundant since already fetched with `limit=15`.

### L-12. Duplicate fallback logic
**File:** `api/routers/priority.py:487-494`
Outer except re-executes identical loop as inner fallback.

### L-13. Read timeout used for write operation
**File:** `api/routers/conversations.py:565, 686`
`send_message` uses `get_timeout_read()` instead of write timeout.

### L-14. Dead code: unused RouteResult and RoutingResponse
**File:** `jarvis/router.py:77-131`
TypedDicts defined but never used as return types.

### L-15. Missing jitter in RetryContext
**File:** `jarvis/retry.py:211`
No jitter unlike decorator versions. Susceptible to thundering herd.

### L-16. Incorrect async type annotations
**File:** `jarvis/retry.py:97`
Return type should use `Awaitable[T]` or `Coroutine` for async wrapper.

### L-17. Duplicate docstring parameter
**File:** `jarvis/prompts.py:1496-1516`
`contact_context` listed twice in Args section.

### L-18. No-op migration function
**File:** `jarvis/config.py:695-697`
`_migrate_v12_to_v13` returns data unchanged.

### L-19. Silent save failure during migration
**File:** `jarvis/config.py:780-782`
If `save_config` fails during migration, failure silently ignored.

### L-20. O(n) participant lookup
**File:** `jarvis/context.py:99-106`
Linear scan through cached conversations. Build dict for O(1) lookup.

### L-21. Loose phone number matching
**File:** `jarvis/context.py:340-348`
Short digit sequences like "1234" match too many numbers.

### L-22. O(N^2) message grouping
**File:** `jarvis/priority.py:636`
Builds filtered list per message. Pre-group by chat_id.

### L-23. Overly broad emoji regex
**File:** `jarvis/relationships.py:53`
Range `\U000024c2-\U0001f251` includes many non-emoji characters.

### L-24. list[Any] instead of list[Message]
**File:** `jarvis/relationships.py:480,512,629,643,663,690,705,735,756,777`
Multiple functions lose type safety with `list[Any]`.

### L-25. Inline import in function body
**File:** `jarvis/relationships.py:807`
`import heapq` inside function runs every call.

### L-26. capitalize() lowercases subsequent chars
**File:** `jarvis/relationships.py:1220`
`guide.capitalize()` lowercases all except first char.

### L-27. Double CachedEmbedder wrapping in class import
**File:** `jarvis/reply_service.py:643`
Class-level import at definition time.

### L-28. Empty prompt sent to model
**File:** `jarvis/dspy_client.py:127-128`
When both `prompt` and `messages` are empty, sends empty string.

### L-29. Thread-unsafe computation counter
**File:** `jarvis/embedding_adapter.py:414`
`self._computations += len(missing_texts)` not protected by lock.

### L-30. No-op string replacement
**File:** `models/cross_encoder.py:107`
`name = hf_name.replace("bert.", "bert.")` replaces with itself.

### L-31. Ambiguous variable name 'l'
**File:** `models/cross_encoder.py:290`
`[-l for l in lengths]` confusable with digit `1`.

### L-32. Missing rerank_score on empty filter
**File:** `models/reranker.py:78`
Returns candidates without `rerank_score` field when all are filtered.

### L-33. Fragile delegation in ThreadAwareGenerator
**File:** `models/generator.py:654-685`
Manual delegation of 5 methods; new methods on inner class won't be forwarded.

### L-34. ParagraphStyle created per message
**File:** `jarvis/pdf_generator.py:418-423`
New style per message in loop. Create once and reuse.

### L-35. sys.path manipulation
**Files:** `scripts/generate_preference_pairs.py:29`, `scripts/label_soc_categories.py:36`, `scripts/evaluate_personal_ft.py:26`
Fragile `sys.path.insert(0, ...)`.

### L-36. Makefile typo
**File:** `Makefile:424`
`mlx_lm_lora.lora` should be `mlx_lm.lora`.

### L-37. ANTI_AI_PHRASES duplicated
**Files:** `evals/eval_pipeline.py:34-44`, `evals/batch_eval.py:41-51`
Identical list in two files.

### L-38. Unused CATEGORIES list
**File:** `evals/batch_eval.py:54-59`
Defined but never used for validation.

### L-39. Wrong API key in error message
**File:** `evals/rag_eval.py:345`
Says "CEREBRAS_API_KEY" but should reference actual key from judge_config.

### L-40. Inconsistent version bounds
**File:** `pyproject.toml:84-85`
Training extras have lower bounds than main deps for xgboost/lightgbm.

### L-41. Return type annotation mismatch
**File:** `experiments/scripts/dailydialog_sweep.py:65`
Says returns 4-tuple but returns 7 values.

### L-42. Duplicate .env loading logic
**Files:** `evals/batch_eval.py:26-32`, `evals/eval_pipeline.py:26-32`, `evals/rag_eval.py:30-37`, `scripts/generate_preference_pairs.py:32-38`
Same hand-rolled parser duplicated 4+ times.

### L-43. Missing KeyError guard for cache files
**File:** `experiments/scripts/coarse_search.py:59-61`
No validation of required keys in embeddings cache file.

---

## Resolution Tracking

| Issue | Severity | Status | Notes |
|-------|----------|--------|-------|
| C-1 | Critical | Pending | |
| C-2 | Critical | Pending | |
| C-3 | Critical | Pending | |
| C-4 | Critical | Pending | |
| C-5 | Critical | Pending | |
| C-6 | Critical | Pending | |
| C-7 | Critical | Pending | |
| H-1 through H-22 | High | Pending | |
| M-1 through M-38 | Medium | Pending | |
| L-1 through L-43 | Low | Pending | |

---

**End of Review**
