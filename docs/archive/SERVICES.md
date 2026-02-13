# JARVIS Services Architecture

## Current Services Overview

JARVIS runs multiple services that work together:

| Service | Location | Communication | Purpose |
|---------|----------|---------------|---------|
| **FastAPI Backend** | Main `.venv` | HTTP `localhost:8742` | REST API for CLI and web clients |
| **Socket Server** | Main `.venv` | Unix socket `/tmp/jarvis.sock` + WebSocket `ws://localhost:8743` | Desktop app IPC, LLM generation, search |
| **MLX Embedding Service** | `~/.jarvis/mlx-embed-service/` | HTTP `localhost:8766` + Unix socket `/tmp/jarvis-embed.sock` | GPU-accelerated embeddings (MLX) |
| **NER Server** | `~/.jarvis/venvs/ner/` | Unix socket `/tmp/jarvis-ner.sock` | Named entity recognition (spaCy) |
| **Tauri Desktop** | `desktop/` | Direct SQLite + Socket | Frontend UI |

## Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Tauri Desktop App                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Direct SQLite│  │ Socket Client │  │ HTTP Client  │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
└─────────┼─────────────────┼─────────────────┼────────────┘
          │                 │                 │
          │ ~1-5ms         │ ~1-5ms          │ ~50-150ms
          ▼                 ▼                 ▼
    ┌──────────┐    ┌──────────────┐   ┌─────────────┐
    │ chat.db  │    │ Socket Server │   │ FastAPI     │
    │ jarvis.db│    │ (Main .venv)  │   │ (Main .venv)│
    └──────────┘    └──────┬───────┘   └─────────────┘
                           │
                           │ JSON-RPC calls
                           ▼
              ┌──────────────────────────┐
              │  MLX Embedding Service   │
              │  (~/.jarvis/mlx-embed-   │
              │   service/, uv run)      │
              └──────────────────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │  NER Server              │
              │  (~/.jarvis/venvs/ner/)  │
              └──────────────────────────┘
```

## Service Details

### 1. FastAPI Backend (`api/main.py`)
- **Port**: 8742
- **Environment**: Main `.venv`
- **Purpose**: REST API endpoints for CLI tools and web clients
- **Start**: `uvicorn api.main:app --port 8742`
- **Status**: ✅ Operational

### 2. Socket Server (`jarvis/socket_server.py`)
- **Unix Socket**: `/tmp/jarvis.sock`
- **WebSocket**: `ws://localhost:8743`
- **Environment**: Main `.venv`
- **Purpose**: 
  - Desktop app IPC (Unix socket)
  - Browser/Playwright testing (WebSocket)
  - LLM generation with streaming
  - Semantic search
  - Intent classification
  - Real-time message notifications (via file watcher)
- **Start**: `python -m jarvis.socket_server`
- **Status**: ✅ Operational

### 3. MLX Embedding Service
- **HTTP Port**: 8766
- **Unix Socket**: `/tmp/jarvis-embed.sock`
- **Location**: `~/.jarvis/mlx-embed-service/`
- **Environment**: Uses `uv run` in that directory (not a venv)
- **Purpose**: GPU-accelerated embeddings via MLX on Apple Silicon
- **Start**: `cd ~/.jarvis/mlx-embed-service && uv run python server.py`
- **Status**: ✅ Operational (optional - falls back to CPU if not running)

### 4. NER Server (`scripts/ner_server.py`)
- **Unix Socket**: `/tmp/jarvis-ner.sock`
- **Location**: `~/.jarvis/venvs/ner/`
- **Environment**: Separate venv (spaCy dependencies)
- **Purpose**: Named entity recognition for contact extraction
- **Start**: `~/.jarvis/venvs/ner/bin/python scripts/ner_server.py`
- **CLI**: `jarvis ner start`
- **Status**: ✅ Operational (optional)

## Current Issues

### 1. Inconsistent Environment Management
- **Main services**: Use `.venv` (uv-managed)
- **MLX service**: Uses `uv run` in separate directory (not a venv)
- **NER service**: Uses separate venv (`~/.jarvis/venvs/ner/`)

### 2. Manual Process Management
- `scripts/launch.sh` manually tracks PIDs
- Cleanup via bash traps (fragile)
- No unified service manager
- Hard to restart individual services

### 3. No Health Monitoring
- Services can die silently
- No automatic restart
- No status checking

### 4. Scattered Configuration
- Service paths hardcoded in multiple places
- No central config for service locations

## Proposed Cleaner Architecture

### Option 1: Unified Service Manager (Python)

Create a Python-based service manager that handles all services:

```python
# jarvis/services/manager.py
class ServiceManager:
    """Manages all JARVIS services with unified lifecycle."""
    
    def __init__(self):
        self.services = {
            "api": FastAPIService(port=8742, venv=".venv"),
            "socket": SocketService(venv=".venv"),
            "embedding": EmbeddingService(venv="~/.jarvis/embed_venv"),
            "ner": NERService(venv="~/.jarvis/venvs/ner"),
        }
    
    def start_all(self):
        """Start all services in dependency order."""
        # 1. Embedding (no deps)
        # 2. NER (no deps)
        # 3. Socket (needs embedding)
        # 4. API (needs socket)
    
    def stop_all(self):
        """Stop all services gracefully."""
    
    def restart(self, service_name: str):
        """Restart a single service."""
    
    def status(self) -> dict:
        """Get status of all services."""
```

**Benefits:**
- Single source of truth for service management
- Consistent venv handling
- Better error handling and logging
- Health checks and auto-restart

### Option 2: Standardize All Services to Separate Venvs

Move everything to separate venvs for complete isolation:

```
~/.jarvis/
├── venvs/
│   ├── main/          # FastAPI + Socket server
│   ├── embedding/     # MLX embedding service
│   └── ner/           # NER server (already exists)
```

**Benefits:**
- Complete dependency isolation
- Easier to update individual services
- Clearer separation of concerns

**Drawbacks:**
- More setup complexity
- Larger disk footprint

### Option 3: Use a Process Manager (systemd/launchd)

Create systemd user services or launchd plists:

```ini
# ~/.config/systemd/user/jarvis-api.service
[Unit]
Description=JARVIS API Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/jarvis-ai-assistant
ExecStart=/path/to/.venv/bin/uvicorn api.main:app --port 8742
Restart=on-failure

[Install]
WantedBy=default.target
```

**Benefits:**
- OS-level process management
- Automatic restart on failure
- Logging via journald

**Drawbacks:**
- Platform-specific (macOS uses launchd, Linux uses systemd)
- More complex setup

## Recommended Approach

**Hybrid: Python Service Manager + Standardized Venvs**

1. **Create `jarvis/services/manager.py`** - Unified Python service manager
2. **Standardize venvs**:
   - Main: `.venv` (FastAPI + Socket)
   - Embedding: `~/.jarvis/venvs/embedding/` (MLX service)
   - NER: `~/.jarvis/venvs/ner/` (already exists, just move)
3. **Update `scripts/launch.sh`** to use the Python manager
4. **Add health checks** and auto-restart for critical services

This gives us:
- ✅ Clean Python-based management
- ✅ Consistent venv handling
- ✅ Better error handling
- ✅ Easy to extend with new services
- ✅ Cross-platform (no OS-specific dependencies)

## New Service Commands

Use the unified service manager from the CLI:

```bash
uv run python -m jarvis services start
uv run python -m jarvis services stop
uv run python -m jarvis services status
uv run python -m jarvis services restart

uv run python -m jarvis services start-service embedding
uv run python -m jarvis services stop-service ner
uv run python -m jarvis services status-service socket
```

Makefile shortcuts:

```bash
make services-start
make services-stop
make services-status
make services-restart
```

## Implementation Plan

1. **Phase 1**: Create `jarvis/services/manager.py` with basic start/stop
2. **Phase 2**: Migrate MLX service to venv (`~/.jarvis/venvs/embedding/`)
3. **Phase 3**: Update `launch.sh` to use Python manager
4. **Phase 4**: Add health checks and status reporting
5. **Phase 5**: Add auto-restart for critical services

## Migration Notes

- Keep `launch.sh` as a wrapper for backward compatibility
- Add `jarvis services start/stop/status` CLI commands
- Document service dependencies clearly
- Add tests for service lifecycle
