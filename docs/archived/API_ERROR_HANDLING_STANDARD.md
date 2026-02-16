# JARVIS API Error Handling Standard

**Version:** 1.0.0  
**Status:** Active  
**Last Updated:** 2026-02-10  
**Owner:** Backend Architecture Team

---

## 1. Overview

This document defines the comprehensive error-handling standard for the JARVIS API. It establishes a consistent taxonomy, error codes, payload schemas, retryability semantics, and mapping rules across all layers of the application.

### 1.1 Goals

- **Consistency:** All errors follow a unified structure across the API
- **Debuggability:** Rich error context for troubleshooting
- **Client Guidance:** Machine-readable codes enable programmatic handling
- **Retry Safety:** Clear semantics for which errors are safe to retry
- **Observability:** Structured logging and metrics integration

### 1.2 Scope

This standard applies to:
- All FastAPI routers (`api/routers/`)
- All service layers (`jarvis/services/`)
- All model operations (`jarvis/models/`, `jarvis/classifiers/`)
- All database access (`jarvis/db/`)
- All external integrations (`integrations/`)

---

## 2. Error Taxonomy

### 2.1 Hierarchy Overview

```
JarvisError (base)
├── ConfigurationError          # CFG_* codes
├── ModelError                  # MDL_* codes
│   ├── ModelLoadError
│   └── ModelGenerationError
├── iMessageError               # MSG_* codes
│   ├── iMessageAccessError
│   └── iMessageQueryError
├── ValidationError             # VAL_* codes
├── ResourceError               # RES_* codes
│   ├── MemoryResourceError
│   └── DiskResourceError
├── TaskError                   # TSK_* codes
│   ├── TaskNotFoundError
│   └── TaskExecutionError
├── CalendarError               # CAL_* codes
│   ├── CalendarAccessError
│   ├── CalendarCreateError
│   └── EventParseError
├── ExperimentError             # EXP_* codes
│   ├── ExperimentNotFoundError
│   └── ExperimentConfigError
└── FeedbackError               # FBK_* codes
    ├── FeedbackNotFoundError
    └── FeedbackInvalidActionError
```

### 2.2 Error Categories

| Category | Prefix | HTTP Status | Description |
|----------|--------|-------------|-------------|
| Configuration | `CFG_*` | 500 | Config file issues, migration failures |
| Model | `MDL_*` | 400-503 | Model loading, generation, timeouts |
| iMessage | `MSG_*` | 403-500 | Database access, query failures |
| Validation | `VAL_*` | 400 | Input validation, type errors |
| Resource | `RES_*` | 503 | Memory, disk exhaustion |
| Task | `TSK_*` | 400-500 | Task queue operations |
| Calendar | `CAL_*` | 400-500 | Calendar integration |
| Experiment | `EXP_*` | 400-404 | A/B testing operations |
| Feedback | `FBK_*` | 400-500 | User feedback tracking |

---

## 3. Error Codes

### 3.1 Code Format

All error codes follow the pattern: `CATEGORY_SPECIFIC_DESCRIPTOR`

- **CATEGORY:** 3-letter uppercase category identifier
- **SPECIFIC:** Specific error type
- **DESCRIPTOR:** Detailed error condition

### 3.2 Complete Code Reference

#### Configuration Errors (`CFG_*`)

| Code | HTTP | Retryable | Description |
|------|------|-----------|-------------|
| `CFG_INVALID` | 500 | No | Invalid configuration value |
| `CFG_MISSING` | 404 | No | Required configuration missing |
| `CFG_MIGRATION_FAILED` | 500 | No | Database migration failed |

#### Model Errors (`MDL_*`)

| Code | HTTP | Retryable | Description |
|------|------|-----------|-------------|
| `MDL_LOAD_FAILED` | 503 | Yes (30s) | Model failed to load |
| `MDL_NOT_FOUND` | 404 | No | Model file not found |
| `MDL_GENERATION_FAILED` | 500 | No | Generation process failed |
| `MDL_TIMEOUT` | 408 | Yes (5s) | Generation timed out |
| `MDL_INVALID_REQUEST` | 400 | No | Invalid generation parameters |

#### iMessage Errors (`MSG_*`)

| Code | HTTP | Retryable | Description |
|------|------|-----------|-------------|
| `MSG_ACCESS_DENIED` | 403 | No* | Full Disk Access not granted |
| `MSG_DB_NOT_FOUND` | 404 | No | chat.db file not found |
| `MSG_QUERY_FAILED` | 500 | Yes (1s) | SQL query execution failed |
| `MSG_SCHEMA_UNSUPPORTED` | 500 | No | Unsupported database schema |
| `MSG_SEND_FAILED` | 500 | Yes (2s) | Failed to send message |

\* `MSG_ACCESS_DENIED` is not retryable without user action.

#### Validation Errors (`VAL_*`)

| Code | HTTP | Retryable | Description |
|------|------|-----------|-------------|
| `VAL_INVALID_INPUT` | 400 | No | Input failed validation |
| `VAL_MISSING_REQUIRED` | 400 | No | Required field missing |
| `VAL_TYPE_ERROR` | 400 | No | Type mismatch |
| `VAL_RANGE_ERROR` | 400 | No | Value out of range |
| `VAL_FORMAT_ERROR` | 400 | No | Invalid format |

#### Resource Errors (`RES_*`)

| Code | HTTP | Retryable | Description |
|------|------|-----------|-------------|
| `RES_MEMORY_LOW` | 503 | Yes (60s) | Memory pressure warning |
| `RES_MEMORY_EXHAUSTED` | 503 | Yes (60s) | Out of memory |
| `RES_DISK_FULL` | 503 | No | Disk space exhausted |
| `RES_DISK_ACCESS` | 503 | Yes (5s) | Disk access denied |
| `RES_RATE_LIMITED` | 429 | Yes (see Retry-After) | Rate limit exceeded |

#### Task Errors (`TSK_*`)

| Code | HTTP | Retryable | Description |
|------|------|-----------|-------------|
| `TSK_NOT_FOUND` | 404 | No | Task ID not found |
| `TSK_INVALID_STATUS` | 400 | No | Invalid status transition |
| `TSK_EXECUTION_FAILED` | 500 | Yes (10s) | Task execution failed |
| `TSK_CANCELLED` | 200 | No | Task was cancelled |
| `TSK_QUEUE_FULL` | 503 | Yes (30s) | Task queue at capacity |

#### Calendar Errors (`CAL_*`)

| Code | HTTP | Retryable | Description |
|------|------|-----------|-------------|
| `CAL_ACCESS_DENIED` | 403 | No* | Calendar permission denied |
| `CAL_NOT_AVAILABLE` | 503 | Yes (5s) | Calendar app not available |
| `CAL_CREATE_FAILED` | 500 | No | Failed to create event |
| `CAL_PARSE_FAILED` | 400 | No | Failed to parse event text |

\* `CAL_ACCESS_DENIED` is not retryable without user action.

#### Experiment Errors (`EXP_*`)

| Code | HTTP | Retryable | Description |
|------|------|-----------|-------------|
| `EXP_NOT_FOUND` | 404 | No | Experiment not found |
| `EXP_INVALID_CONFIG` | 400 | No | Invalid experiment config |
| `EXP_VARIANT_NOT_FOUND` | 404 | No | Variant not found |
| `EXP_ALREADY_EXISTS` | 409 | No | Experiment already exists |

#### Feedback Errors (`FBK_*`)

| Code | HTTP | Retryable | Description |
|------|------|-----------|-------------|
| `FBK_NOT_FOUND` | 404 | No | Feedback record not found |
| `FBK_INVALID_ACTION` | 400 | No | Invalid action type |
| `FBK_STORE_ERROR` | 500 | Yes (1s) | Failed to store feedback |

---

## 4. Payload Schema

### 4.1 Standard Error Response

All API errors return a consistent JSON structure:

```json
{
  "error": "ErrorClassName",
  "code": "ERROR_CODE",
  "detail": "Human-readable error message",
  "details": {
    // Optional additional context
  },
  "request_id": "uuid-for-tracing",
  "timestamp": "2026-02-10T10:14:31Z"
}
```

### 4.2 Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `error` | string | Yes | Exception class name for client type handling |
| `code` | string | Yes | Machine-readable error code |
| `detail` | string | Yes | Human-readable description |
| `details` | object | No | Additional context (field names, paths, etc.) |
| `request_id` | string | No | Request ID for log correlation |
| `timestamp` | string | No | ISO 8601 timestamp of error |

### 4.3 Example Error Responses

#### Validation Error (400)

```json
{
  "error": "ValidationError",
  "code": "VAL_MISSING_REQUIRED",
  "detail": "Missing required field: conversation_id",
  "details": {
    "field": "conversation_id",
    "provided_fields": ["message_text"],
    "required_fields": ["conversation_id", "message_text"]
  },
  "request_id": "req_abc123",
  "timestamp": "2026-02-10T10:14:31Z"
}
```

#### Permission Error (403)

```json
{
  "error": "iMessageAccessError",
  "code": "MSG_ACCESS_DENIED",
  "detail": "Full Disk Access is required to read iMessages",
  "details": {
    "requires_permission": true,
    "permission_instructions": [
      "Open System Settings",
      "Go to Privacy & Security > Full Disk Access",
      "Add and enable your terminal application",
      "Restart JARVIS"
    ]
  },
  "request_id": "req_def456",
  "timestamp": "2026-02-10T10:14:31Z"
}
```

#### Resource Exhaustion (503)

```json
{
  "error": "MemoryResourceError",
  "code": "RES_MEMORY_EXHAUSTED",
  "detail": "Insufficient memory to load model: qwen-3b",
  "details": {
    "resource_type": "memory",
    "available_mb": 1024,
    "required_mb": 2048,
    "model_name": "qwen-3b"
  },
  "request_id": "req_ghi789",
  "timestamp": "2026-02-10T10:14:31Z"
}
```

#### Rate Limited (429)

```json
{
  "error": "RateLimitExceeded",
  "code": "RES_RATE_LIMITED",
  "detail": "Rate limit exceeded: 10 requests per minute",
  "details": {
    "limit": 10,
    "window": "1 minute",
    "retry_after": 45
  },
  "request_id": "req_jkl012",
  "timestamp": "2026-02-10T10:14:31Z"
}
```

---

## 5. Retryability Semantics

### 5.1 Retry Decision Matrix

| HTTP Status | Default Retryable | Strategy | Notes |
|-------------|-------------------|----------|-------|
| 400 Bad Request | ❌ No | Fix request | Client error |
| 401 Unauthorized | ❌ No | Re-authenticate | Credentials issue |
| 403 Forbidden | ❌ No | User action | Permission issue |
| 404 Not Found | ❌ No | Fix request | Resource doesn't exist |
| 408 Request Timeout | ✅ Yes | 5s backoff | Transient timeout |
| 409 Conflict | ❌ No | Resolve conflict | State conflict |
| 429 Too Many Requests | ✅ Yes | Use Retry-After | Rate limited |
| 500 Internal Error | ⚠️ Depends | See error code | Server error |
| 502 Bad Gateway | ✅ Yes | 1s backoff | Upstream issue |
| 503 Service Unavailable | ✅ Yes | 30-60s backoff | Temporary unavailable |
| 504 Gateway Timeout | ✅ Yes | 5s backoff | Upstream timeout |

### 5.2 Retry Strategies

#### Exponential Backoff (Standard)

```python
def calculate_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """Calculate exponential backoff with jitter.
    
    Formula: min(base * 2^attempt, max_delay) * jitter(0.5-1.5)
    """
    import random
    delay = min(base_delay * (2 ** attempt), max_delay)
    return delay * random.uniform(0.5, 1.5)
```

| Attempt | Base 1s | Base 2s | Max |
|---------|---------|---------|-----|
| 1 | 0.5-1.5s | 1-3s | 60s |
| 2 | 1-3s | 2-6s | 60s |
| 3 | 2-6s | 4-12s | 60s |
| 4 | 4-12s | 8-24s | 60s |
| 5 | 8-24s | 16-48s | 60s |

#### Server-Specified Retry-After

When present, always use the server's `Retry-After` header value:

```python
retry_after = response.headers.get("Retry-After", default_backoff)
```

#### Model-Specific Retry Delays

| Error Code | Retry After | Reason |
|------------|-------------|--------|
| `MDL_LOAD_FAILED` | 30s | Model reload time |
| `MDL_TIMEOUT` | 5s | Generation retry |
| `RES_MEMORY_EXHAUSTED` | 60s | Memory recovery |
| `RES_MEMORY_LOW` | 60s | GC time |
| `TSK_QUEUE_FULL` | 30s | Queue drain |

### 5.3 Non-Retryable Conditions

**Never retry these errors without client changes:**

- `VAL_*` - Fix the input
- `CFG_INVALID` - Fix configuration
- `MSG_ACCESS_DENIED` - Grant permissions
- `CAL_ACCESS_DENIED` - Grant permissions
- `MDL_NOT_FOUND` - Install model
- `EXP_NOT_FOUND` - Create experiment first

### 5.4 Idempotency Considerations

For retry safety, ensure idempotency:

| Operation | Idempotent? | Strategy |
|-----------|-------------|----------|
| GET | ✅ Yes | Safe to retry |
| POST /generate | ❌ No | Use idempotency key |
| POST /tasks | ✅ Yes | Task deduplication |
| PUT | ✅ Yes | Safe to retry |
| DELETE | ✅ Yes | Safe to retry (404 OK) |
| PATCH | ⚠️ Partial | Use ETag/version |

---

## 6. Mapping Rules

### 6.1 Layer Mapping Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         API Layer                            │
│                    (HTTP Status Codes)                       │
└───────────────────────┬─────────────────────────────────────┘
                        │ raise JarvisError
┌───────────────────────▼─────────────────────────────────────┐
│                      Router Layer                            │
│              (FastAPI Exception Handlers)                    │
└───────────────────────┬─────────────────────────────────────┘
                        │ raise/convert
┌───────────────────────▼─────────────────────────────────────┐
│                     Service Layer                            │
│              (Business Logic, Orchestration)                 │
└───────────────────────┬─────────────────────────────────────┘
                        │ raise/convert
┌───────────────────────▼─────────────────────────────────────┐
│                      Model Layer                             │
│           (MLX Models, Classifiers, Embeddings)              │
└───────────────────────┬─────────────────────────────────────┘
                        │ raise/convert
┌───────────────────────▼─────────────────────────────────────┐
│                   Integration Layer                          │
│         (iMessage DB, Calendar, File System)                 │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Integration Layer → Model Layer

**Rule:** Convert low-level exceptions to domain-specific errors.

```python
# jarvis/db/core.py
import sqlite3
from jarvis.errors import iMessageQueryError, iMessageAccessError

def execute_query(query: str, params: tuple = ()) -> list[dict]:
    try:
        conn.execute(query, params)
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e):
            raise iMessageQueryError(
                "Database is locked by another process",
                query=query,
                cause=e
            )
        elif "no such table" in str(e):
            raise iMessageQueryError(
                "Invalid database schema",
                query=query,
                code=ErrorCode.MSG_SCHEMA_UNSUPPORTED,
                cause=e
            )
        raise iMessageQueryError(
            f"Query failed: {e}",
            query=query,
            cause=e
        )
    except sqlite3.DatabaseError as e:
        raise iMessageAccessError(
            "Database access error",
            cause=e
        )
```

### 6.3 Model Layer → Service Layer

**Rule:** Enrich errors with context, preserve original cause.

```python
# jarvis/services/context_service.py
from jarvis.errors import ModelGenerationError, model_generation_timeout

def generate_reply(context: ConversationContext) -> str:
    try:
        return model.generate(context.to_prompt())
    except TimeoutError as e:
        # Convert to domain error with context
        raise model_generation_timeout(
            model_name=model.name,
            timeout_seconds=30.0,
            prompt=context.to_prompt()
        ) from e
    except Exception as e:
        # Wrap unexpected errors
        raise ModelGenerationError(
            f"Generation failed: {e}",
            model_name=model.name,
            prompt=context.to_prompt(),
            cause=e
        ) from e
```

### 6.4 Service Layer → Router Layer

**Rule:** Use JarvisError subclasses directly, let handlers convert to HTTP.

```python
# api/routers/drafts.py
from fastapi import APIRouter
from jarvis.errors import ValidationError, ModelGenerationError

@router.post("/reply")
async def generate_reply(request: DraftReplyRequest) -> DraftReplyResponse:
    # Validation - raises 400
    if not request.conversation_id:
        raise ValidationError(
            "conversation_id is required",
            field="conversation_id",
            code=ErrorCode.VAL_MISSING_REQUIRED
        )
    
    # Service call - may raise 503, 408, 500
    try:
        reply = await context_service.generate_reply(context)
    except ModelGenerationError:
        # Re-raise to be handled by exception handler
        raise
    
    return DraftReplyResponse(text=reply)
```

### 6.5 Router Layer → HTTP Response

**Rule:** Exception handlers convert JarvisError to JSONResponse.

```python
# api/errors.py
async def jarvis_error_handler(request: Request, exc: JarvisError) -> JSONResponse:
    status_code = get_status_code_for_error(exc)
    response_body = build_error_response(exc)
    
    # Add request ID for tracing
    response_body["request_id"] = getattr(request.state, "request_id", None)
    response_body["timestamp"] = datetime.utcnow().isoformat() + "Z"
    
    headers = {}
    if status_code == 429:
        headers["Retry-After"] = str(exc.details.get("retry_after", 60))
    elif status_code == 503 and exc.code in RETRYABLE_ERROR_CODES:
        headers["Retry-After"] = str(get_retry_after(exc.code))
    
    return JSONResponse(
        status_code=status_code,
        content=response_body,
        headers=headers
    )
```

### 6.6 HTTP Status Code Mapping

| Error Class | Default HTTP | Override Conditions |
|-------------|--------------|---------------------|
| `ValidationError` | 400 | Always |
| `iMessageAccessError` | 403 | Always |
| `iMessageQueryError` | 500 | `MSG_DB_NOT_FOUND` → 404 |
| `ModelLoadError` | 503 | Always |
| `ModelGenerationError` | 500 | `MDL_TIMEOUT` → 408 |
| `ResourceError` | 503 | Always |
| `TaskNotFoundError` | 404 | Always |
| `ExperimentNotFoundError` | 404 | Always |
| `FeedbackNotFoundError` | 404 | Always |

---

## 7. Implementation Guidelines

### 7.1 Raising Errors

**DO:**
```python
from jarvis.errors import ValidationError, ErrorCode

# Use specific error types
raise ValidationError(
    "Invalid email format",
    field="email",
    value=email,
    expected="valid email address",
    code=ErrorCode.VAL_INVALID_INPUT
)

# Include context for debugging
raise ModelGenerationError(
    "Generation failed after retries",
    model_name="qwen-1.5b",
    prompt=prompt[:200],  # Truncate long values
    cause=original_exception
)
```

**DON'T:**
```python
# Don't use generic exceptions
raise Exception("Something went wrong")

# Don't use HTTP exceptions in business logic
raise HTTPException(status_code=400, detail="Bad request")

# Don't lose the original exception context
try:
    risky_operation()
except Exception:
    raise ModelError("Failed")  # Lost the cause!
```

### 7.2 Catching and Converting

**DO:**
```python
try:
    db_result = db.query(sql)
except sqlite3.Error as e:
    # Convert to domain error with full context
    raise iMessageQueryError(
        "Database query failed",
        query=sql,
        cause=e  # Preserve original
    ) from e  # Use 'from' for exception chaining
```

**DON'T:**
```python
try:
    db_result = db.query(sql)
except Exception:
    # Swallowing the error!
    return None

try:
    db_result = db.query(sql)
except sqlite3.Error as e:
    # Lost the original exception
    raise JarvisError(str(e))
```

### 7.3 Error Response Builder

```python
# api/errors.py
def build_error_response(error: JarvisError) -> dict[str, Any]:
    """Build standardized error response."""
    response: dict[str, Any] = {
        "error": error.__class__.__name__,
        "code": error.code.value,
        "detail": error.message,
    }
    
    # Add details if present
    if error.details:
        response["details"] = error.details
    
    return response
```

### 7.4 Request Context Middleware

```python
# api/main.py
import uuid
from fastapi import Request

@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Add request ID for error tracing."""
    request.state.request_id = str(uuid.uuid4())
    
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response
    except Exception as e:
        # Log with request ID for correlation
        logger.exception(
            "Request %s failed: %s",
            request.state.request_id,
            str(e)
        )
        raise
```

---

## 8. Testing Error Handling

### 8.1 Unit Testing Errors

```python
# tests/test_errors.py
import pytest
from jarvis.errors import ValidationError, ErrorCode

def test_validation_error_includes_field():
    error = ValidationError(
        "Invalid value",
        field="email",
        value="not-an-email",
        code=ErrorCode.VAL_INVALID_INPUT
    )
    
    assert error.code == ErrorCode.VAL_INVALID_INPUT
    assert error.details["field"] == "email"
    assert error.details["value"] == "not-an-email"

def test_error_to_dict_structure():
    error = ValidationError("Test error")
    result = error.to_dict()
    
    assert "error" in result
    assert "code" in result
    assert "detail" in result
```

### 8.2 Integration Testing Error Responses

```python
# tests/api/test_error_responses.py
import pytest
from fastapi.testclient import TestClient

def test_validation_error_response_format(client: TestClient):
    response = client.post("/drafts/reply", json={})  # Missing required fields
    
    assert response.status_code == 400
    data = response.json()
    
    # Verify standard structure
    assert "error" in data
    assert "code" in data
    assert "detail" in data
    assert data["code"].startswith("VAL_")

def test_rate_limit_returns_retry_after(client: TestClient):
    # Make requests until rate limited
    for _ in range(100):
        response = client.post("/drafts/reply", json={"conversation_id": "123"})
        if response.status_code == 429:
            break
    
    assert response.status_code == 429
    assert "Retry-After" in response.headers
    assert "retry_after" in response.json()["details"]
```

### 8.3 Testing Retry Behavior

```python
# tests/test_retry.py
import pytest
from unittest.mock import Mock, patch
from jarvis.retry import retry_with_backoff

def test_retry_on_model_load_error():
    mock_func = Mock(side_effect=[ModelLoadError("Failed"), ModelLoadError("Failed"), "success"])
    
    @retry_with_backoff(max_retries=3, exceptions=(ModelLoadError,))
    def load_model():
        return mock_func()
    
    result = load_model()
    
    assert result == "success"
    assert mock_func.call_count == 3

def test_no_retry_on_validation_error():
    mock_func = Mock(side_effect=ValidationError("Invalid"))
    
    @retry_with_backoff(max_retries=3, exceptions=(ValidationError,))
    def validate():
        return mock_func()
    
    with pytest.raises(ValidationError):
        validate()
    
    assert mock_func.call_count == 1  # No retry
```

---

## 9. Observability

### 9.1 Structured Error Logging

```python
# jarvis/observability/errors.py
import logging
import json
from jarvis.errors import JarvisError

logger = logging.getLogger("jarvis.errors")

def log_error(error: JarvisError, request_id: str | None = None) -> None:
    """Log error with structured context."""
    log_entry = {
        "event": "error",
        "error_type": error.__class__.__name__,
        "error_code": error.code.value,
        "message": error.message,
        "request_id": request_id,
        "details": error.details,
    }
    
    if error.cause:
        log_entry["cause"] = str(error.cause)
    
    if error.code.value.startswith("RES_") or error.code.value.startswith("MDL_"):
        logger.error(json.dumps(log_entry))
    else:
        logger.warning(json.dumps(log_entry))
```

### 9.2 Error Metrics

```python
# jarvis/metrics.py
from prometheus_client import Counter, Histogram

error_counter = Counter(
    "jarvis_api_errors_total",
    "Total API errors by code",
    ["error_code", "error_type", "endpoint"]
)

error_latency = Histogram(
    "jarvis_error_response_time_seconds",
    "Time to generate error response",
    ["error_code"]
)

def record_error(error: JarvisError, endpoint: str) -> None:
    error_counter.labels(
        error_code=error.code.value,
        error_type=error.__class__.__name__,
        endpoint=endpoint
    ).inc()
```

### 9.3 Error Alerting Rules

```yaml
# monitoring/alerts.yml
alerts:
  - name: HighErrorRate
    condition: |
      rate(jarvis_api_errors_total[5m]) > 10
    severity: warning
    
  - name: ModelLoadFailures
    condition: |
      increase(jarvis_api_errors_total{error_code="MDL_LOAD_FAILED"}[10m]) > 5
    severity: critical
    
  - name: MemoryExhaustion
    condition: |
      increase(jarvis_api_errors_total{error_code="RES_MEMORY_EXHAUSTED"}[5m]) > 0
    severity: critical
    
  - name: PermissionErrors
    condition: |
      increase(jarvis_api_errors_total{error_code="MSG_ACCESS_DENIED"}[1h]) > 10
    severity: info
```

---

## 10. Client SDK Guidelines

### 10.1 Error Handling Pattern

```typescript
// desktop/src/lib/api/errors.ts
export interface ApiError {
  error: string;
  code: string;
  detail: string;
  details?: Record<string, any>;
  request_id?: string;
  timestamp?: string;
}

export class JarvisApiError extends Error {
  constructor(
    public code: string,
    message: string,
    public details?: Record<string, any>,
    public requestId?: string
  ) {
    super(message);
    this.name = 'JarvisApiError';
  }
  
  get isRetryable(): boolean {
    return RETRYABLE_CODES.includes(this.code);
  }
  
  get retryAfter(): number {
    return this.details?.retry_after || 5;
  }
}

const RETRYABLE_CODES = [
  'MDL_LOAD_FAILED',
  'MDL_TIMEOUT',
  'RES_MEMORY_LOW',
  'RES_MEMORY_EXHAUSTED',
  'MSG_QUERY_FAILED',
  'RES_RATE_LIMITED',
];
```

### 10.2 Retry Logic

```typescript
// desktop/src/lib/api/client.ts
async function fetchWithRetry<T>(
  url: string,
  options: RequestInit,
  maxRetries: number = 3
): Promise<T> {
  let lastError: JarvisApiError | null = null;
  
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url, options);
      
      if (!response.ok) {
        const error: ApiError = await response.json();
        throw new JarvisApiError(
          error.code,
          error.detail,
          error.details,
          error.request_id
        );
      }
      
      return await response.json();
    } catch (error) {
      if (error instanceof JarvisApiError && error.isRetryable && attempt < maxRetries) {
        const delay = error.retryAfter * Math.pow(2, attempt);
        await sleep(delay * 1000);
        lastError = error;
      } else {
        throw error;
      }
    }
  }
  
  throw lastError;
}
```

---

## 11. Appendix

### 11.1 Error Code Quick Reference

| Code | HTTP | Retry | Category |
|------|------|-------|----------|
| `CFG_INVALID` | 500 | ❌ | Config |
| `CFG_MISSING` | 404 | ❌ | Config |
| `MDL_LOAD_FAILED` | 503 | ✅ 30s | Model |
| `MDL_TIMEOUT` | 408 | ✅ 5s | Model |
| `MSG_ACCESS_DENIED` | 403 | ❌ | iMessage |
| `MSG_QUERY_FAILED` | 500 | ✅ 1s | iMessage |
| `VAL_INVALID_INPUT` | 400 | ❌ | Validation |
| `VAL_MISSING_REQUIRED` | 400 | ❌ | Validation |
| `RES_MEMORY_EXHAUSTED` | 503 | ✅ 60s | Resource |
| `RES_RATE_LIMITED` | 429 | ✅ Header | Resource |
| `TSK_NOT_FOUND` | 404 | ❌ | Task |
| `TSK_EXECUTION_FAILED` | 500 | ✅ 10s | Task |
| `CAL_ACCESS_DENIED` | 403 | ❌ | Calendar |
| `EXP_NOT_FOUND` | 404 | ❌ | Experiment |
| `FBK_STORE_ERROR` | 500 | ✅ 1s | Feedback |

### 11.2 HTTP Status to Error Type Mapping

```python
# api/errors.py - Reference mapping
HTTP_STATUS_ERROR_TYPES: dict[int, list[str]] = {
    400: ["ValidationError", "EventParseError"],
    403: ["iMessageAccessError", "CalendarAccessError"],
    404: ["ModelLoadError", "TaskNotFoundError", "ExperimentNotFoundError"],
    408: ["ModelGenerationError"],  # timeout variant
    429: ["RateLimitExceeded"],
    500: ["JarvisError", "ModelGenerationError", "iMessageQueryError"],
    503: ["ModelLoadError", "ResourceError"],
}
```

### 11.3 Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-10 | Initial comprehensive standard |

---

## 12. References

- [FastAPI Exception Handling](https://fastapi.tiangolo.com/tutorial/handling-errors/)
- [RFC 7807: Problem Details for HTTP APIs](https://tools.ietf.org/html/rfc7807)
- [Google API Design Guide: Errors](https://cloud.google.com/apis/design/errors)
- [Microsoft REST API Guidelines](https://github.com/microsoft/api-guidelines/blob/vNext/Guidelines.md#7102-error-condition-responses)
