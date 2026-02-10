# API Error Handling Standard

## Error Response Format

All API errors MUST return a JSON response with this structure:

```json
{
    "error": "ErrorClassName",
    "code": "MACHINE_READABLE_CODE",
    "detail": "Human-readable error message",
    "details": {}
}
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `error` | string | Yes | Exception class name (e.g., `ModelLoadError`) |
| `code` | string | Yes | Machine-readable error code from `ErrorCode` enum |
| `detail` | string | Yes | Human-readable description of the error |
| `details` | object | No | Additional context (field names, paths, etc.) |

## HTTP Status Code Mapping

| Status | When |
|--------|------|
| 400 | Validation errors, invalid input |
| 403 | Permission denied (iMessage, Calendar) |
| 404 | Resource not found (model, database) |
| 408 | Request timeout |
| 429 | Rate limit exceeded |
| 500 | Internal server errors |
| 503 | Service unavailable (model loading, resource exhaustion) |

## Required Headers

- **503 responses** MUST include `Retry-After` header (seconds until retry is reasonable)
  - Model load failures: `Retry-After: 30`
  - Resource exhaustion: `Retry-After: 60`
- **429 responses** MUST include `Retry-After` header (from rate limiter)
- All responses include `X-Response-Time` header

## Error Hierarchy

All JARVIS errors extend `JarvisError` (defined in `jarvis/errors.py`).
Exception handlers are registered in `api/errors.py` via `register_exception_handlers()`.

### Handler Priority (most specific first)

1. `ValidationError` -> 400
2. `iMessageAccessError` -> 403
3. `ModelLoadError` -> 503 + Retry-After: 30
4. `ResourceError` -> 503 + Retry-After: 60
5. `JarvisError` (base) -> mapped via `ERROR_CODE_STATUS_CODES`
6. `TimeoutError` -> 408
7. `Exception` (catch-all) -> 500

## Implementation

Error handlers are in `api/errors.py` and registered in `api/main.py`:

```python
from api.errors import register_exception_handlers

app = FastAPI()
register_exception_handlers(app)
```
