# JARVIS Troubleshooting Guide

> **Last Updated:** 2026-02-10

## Common Issues

### MLX Out of Memory (OOM)

**Symptoms:**
- Process killed by macOS
- `malloc` errors in logs
- System becomes unresponsive during model operations

**Causes:**
- MLX allocates GPU memory without limits by default
- On 8GB systems, this can consume 4-5GB for batch encoding alone

**Solutions:**

1. **Memory limits are set automatically** in `jarvis/embedding_adapter.py`. If you see OOM, verify they're applied:
   ```python
   import mlx.core as mx
   mx.metal.set_memory_limit(1024 * 1024 * 1024)  # 1GB
   mx.metal.set_cache_limit(512 * 1024 * 1024)     # 512MB
   ```

2. **Reduce batch sizes** in config:
   - Embedding batches: 64 for large models, 128 for base models
   - Never process >500MB of data in RAM at once

3. **Check memory pressure** (not just swap):
   ```bash
   # Real memory pressure (0 = good, >50 = warn, >100 = critical)
   uv run python -c "from jarvis.utils.memory import get_memory_pressure; print(get_memory_pressure())"
   ```

4. **One model at a time**: Unload the previous model before loading the next.

---

### Socket Connection Refused

**Symptoms:**
- Desktop app shows "Connection failed"
- `ConnectionRefusedError` when connecting to `~/.jarvis/jarvis.sock`

**Solutions:**

1. **Check if server is running:**
   ```bash
   ls -la ~/.jarvis/jarvis.sock
   # If file exists but no server: stale socket
   ```

2. **Remove stale socket and restart:**
   ```bash
   rm -f ~/.jarvis/jarvis.sock
   uv run python -m jarvis.socket_server
   ```

3. **Check permissions:**
   ```bash
   ls -la ~/.jarvis/
   # Directory should be drwx------ (0700)
   # Socket should be srw------- (0600)
   ```

4. **Check for port conflicts** (WebSocket):
   ```bash
   lsof -i :8743
   # Kill any existing process using the port
   ```

---

### Database Locked

**Symptoms:**
- `sqlite3.OperationalError: database is locked`
- Slow queries or timeouts

**Solutions:**

1. **chat.db should always be read-only:**
   ```python
   # Correct: read-only mode prevents lock conflicts
   sqlite3.connect("file:path/to/chat.db?mode=ro", uri=True)
   ```

2. **jarvis.db WAL mode**: The database uses WAL (Write-Ahead Logging) which allows concurrent readers:
   ```bash
   sqlite3 ~/.jarvis/jarvis.db "PRAGMA journal_mode;"
   # Should output: wal
   ```

3. **If locked, check for stale connections:**
   ```bash
   lsof ~/.jarvis/jarvis.db
   # Identify and investigate processes holding the lock
   ```

4. **Timeout configuration**: Database connections use a 5-second busy timeout. If you see lock errors, another process may be holding a write transaction too long.

---

### WebSocket Authentication Failed

**Symptoms:**
- WebSocket connection closes with code `4001`
- "Unauthorized" error in logs

**Solutions:**

1. **Token file must be readable:**
   ```bash
   cat ~/.jarvis/ws_token
   # Should contain a base64 token string
   ```

2. **Token regenerates each server start.** If the desktop app has a stale token, restart the app.

3. **Check origin headers**: WebSocket only accepts connections from:
   - `tauri://localhost`
   - `http://localhost`
   - `http://127.0.0.1`

---

### Model Loading Fails

**Symptoms:**
- "Model not available" errors
- Preload timeout warnings in logs

**Solutions:**

1. **Check if model is downloaded:**
   ```bash
   ls -la ~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct-MLX-4bit/
   ```

2. **Download model manually:**
   ```bash
   uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit')"
   ```

3. **Skip preloading for faster startup** (models load on first request):
   ```bash
   uv run python -m jarvis.socket_server --no-preload
   ```

---

### Tests Failing

**Symptoms:**
- `make test` reports failures
- Collection errors in test output

**Solutions:**

1. **Always check test_results.txt first:**
   ```bash
   tail -50 test_results.txt
   ```

2. **Known collection errors**: `CentroidMixin` import errors in `jarvis/classifiers/mixins.py` are pre-existing and don't affect production code.

3. **Run a specific test:**
   ```bash
   uv run pytest tests/test_specific.py -v
   ```

4. **Run performance benchmarks separately:**
   ```bash
   uv run pytest -m benchmark tests/test_performance_baselines.py -v
   ```

---

### Slow Performance

**Symptoms:**
- Conversation list takes >500ms to load
- Draft generation takes >30s

**Diagnostic Steps:**

1. **Check if using direct SQLite (fast) or HTTP fallback (slow):**
   - Direct SQLite: ~1-5ms per query
   - HTTP fallback: ~100-150ms per query

2. **Check model preload status:**
   ```bash
   # Via socket
   echo '{"jsonrpc":"2.0","method":"ping","id":1}' | socat - UNIX-CONNECT:~/.jarvis/jarvis.sock
   # Look for "models_ready": true
   ```

3. **Profile hot paths:**
   ```bash
   uv run python -m cProfile -s cumulative -m jarvis.socket_server
   ```

4. **Check prefetch cache hit rate:**
   ```bash
   echo '{"jsonrpc":"2.0","method":"prefetch_stats","id":1}' | socat - UNIX-CONNECT:~/.jarvis/jarvis.sock
   ```

---

## Environment Requirements

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| macOS | 13.0+ | 14.0+ |
| Apple Silicon | M1 | M1 Pro+ |
| RAM | 8GB | 16GB |
| Disk | 5GB free | 10GB free |
| Python | 3.11 | 3.12 |

## Getting Help

- Check [docs/HOW_IT_WORKS.md](HOW_IT_WORKS.md) for system overview
- Check [docs/ARCHITECTURE.md](ARCHITECTURE.md) for technical details
- File issues at the project repository
