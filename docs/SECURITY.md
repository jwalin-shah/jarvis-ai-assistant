# JARVIS Security Model

> **Last Updated:** 2026-02-10

## Threat Model

JARVIS is a **local-first** application. All data stays on-device, all inference runs locally via MLX. The primary attack surface is the IPC layer between the Tauri desktop app and the Python backend.

### Trust Boundaries

```
┌───────────────────────────────────────────────┐
│  User's macOS session (trusted)               │
│                                               │
│  ┌─────────────┐      ┌──────────────────┐   │
│  │ Tauri App   │◄────►│ Python Backend   │   │
│  │ (frontend)  │ IPC  │ (socket server)  │   │
│  └─────────────┘      └────────┬─────────┘   │
│                                │              │
│  ┌─────────────┐      ┌───────┴──────────┐   │
│  │ chat.db     │      │ jarvis.db        │   │
│  │ (read-only) │      │ (read-write)     │   │
│  └─────────────┘      └──────────────────┘   │
└───────────────────────────────────────────────┘
```

### Assets Protected

| Asset | Sensitivity | Protection |
|-------|------------|------------|
| iMessage database (chat.db) | High | Read-only access, no modifications |
| Contact profiles (jarvis.db) | Medium | User-only file permissions (0600) |
| WebSocket auth token | High | Generated per-session, file permissions (0600) |
| Unix socket | Medium | User-only permissions (0600), `~/.jarvis/` directory (0700) |
| Config file | Low | User-only permissions (0600) |

## IPC Security

### Unix Socket (`~/.jarvis/jarvis.sock`)

- **Location:** `~/.jarvis/` directory with `0700` permissions (user-only access)
- **Socket permissions:** `0600` (owner read/write only)
- **Protocol:** JSON-RPC 2.0 over newline-delimited JSON
- **Max message size:** 1MB (prevents memory exhaustion)
- **Idle timeout:** 5 minutes per client
- **Rate limiting:** 100 requests/second per client (sliding window)

### WebSocket (`ws://127.0.0.1:8743`)

- **Binding:** `127.0.0.1` only (not `0.0.0.0`) - loopback only
- **Authentication:** Per-session random token (32 bytes, URL-safe base64)
- **Token storage:** `~/.jarvis/ws_token` with `0600` permissions
- **Token comparison:** Timing-safe (`hmac.compare_digest`)
- **Origin validation:** Only accepts `tauri://localhost`, `http://localhost`, `http://127.0.0.1`
- **Connection limit:** Max 10 concurrent WebSocket connections
- **Rate limiting:** 100 requests/second per client

### Database Access

- **chat.db:** Opened with `?mode=ro` (SQLite read-only flag). No writes possible.
- **jarvis.db:** User-only permissions. Path validated against traversal attacks.
- **SQL queries:** All parameterized (no string interpolation). See `integrations/imessage/queries.py`.

## Input Validation

### Path Validation

All filesystem paths are validated via `jarvis.config.validate_path()`:
- Rejects `../` traversal sequences
- Rejects null bytes (`\x00`)
- Resolves to absolute paths

### JSON-RPC Input

- Message size capped at 1MB
- Batch requests capped at 50 per call
- Contact resolution capped at 500 identifiers per call
- All parameters validated by handler function signatures (TypeError on mismatch)

## Security Scanning

### Bandit

Run static security analysis:
```bash
uv run bandit -r jarvis/ api/ core/ -x tests/,tools/ -ll
```

### Known Exceptions

| Finding | Location | Justification |
|---------|----------|--------------|
| B104 (bind all) | N/A | WebSocket binds to 127.0.0.1 only |
| B608 (SQL injection) | `integrations/imessage/queries.py` | All queries use parameterized `?` placeholders |

## Vulnerability Disclosure

If you discover a security vulnerability:

1. **Do not** open a public GitHub issue
2. Email the maintainer directly (see repository contact info)
3. Include: description, reproduction steps, potential impact
4. Allow 30 days for a fix before public disclosure

## Security Checklist for Contributors

- [ ] All SQL queries use parameterized placeholders (`?`)
- [ ] All file paths validated with `validate_path()` before use
- [ ] No secrets hardcoded (tokens generated at runtime)
- [ ] Socket/file permissions set to owner-only (`0600`/`0700`)
- [ ] WebSocket token comparison uses `hmac.compare_digest()`
- [ ] New RPC methods validate input parameters
- [ ] Rate limiting applies to new endpoints
