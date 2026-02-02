#!/bin/bash
# JARVIS Launcher Script
# Starts the FastAPI backend and Tauri desktop app together.
# Ensures proper cleanup when the app closes.

set -e

# Configuration
API_PORT=8742
EMBED_PORT=8766
SOCKET_PATH="/tmp/jarvis.sock"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DESKTOP_DIR="$PROJECT_ROOT/desktop"
EMBED_SERVICE_DIR="$HOME/.jarvis/mlx-embed-service"
API_PID=""
EMBED_PID=""
SOCKET_PID=""
TAURI_PID=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Cleanup function - called on exit
cleanup() {
    log_info "Shutting down JARVIS..."

    # Kill the API server if running
    if [ -n "$API_PID" ] && kill -0 "$API_PID" 2>/dev/null; then
        log_info "Stopping API server (PID: $API_PID)..."
        kill "$API_PID" 2>/dev/null || true
        wait "$API_PID" 2>/dev/null || true
    fi

    # Kill the socket server if we started it
    if [ -n "$SOCKET_PID" ] && kill -0 "$SOCKET_PID" 2>/dev/null; then
        log_info "Stopping socket server (PID: $SOCKET_PID)..."
        kill "$SOCKET_PID" 2>/dev/null || true
        wait "$SOCKET_PID" 2>/dev/null || true
    fi

    # Kill the embedding service if we started it
    if [ -n "$EMBED_PID" ] && kill -0 "$EMBED_PID" 2>/dev/null; then
        log_info "Stopping embedding service (PID: $EMBED_PID)..."
        kill "$EMBED_PID" 2>/dev/null || true
        wait "$EMBED_PID" 2>/dev/null || true
    fi

    # Kill any remaining processes on the API port
    kill_port_process "$API_PORT"

    # Remove socket file if it exists
    if [ -e "$SOCKET_PATH" ]; then
        rm -f "$SOCKET_PATH"
    fi

    # Clean up Python/MLX memory by triggering garbage collection
    # This runs a small Python script to clear any cached models
    if command -v python &> /dev/null; then
        python -c "
import gc
try:
    import mlx.core as mx
    mx.metal.clear_cache()
except ImportError:
    pass
gc.collect()
" 2>/dev/null || true
    fi

    log_success "Cleanup complete"
}

# Kill any process using a specific port
kill_port_process() {
    local port=$1
    local pids=$(lsof -ti:$port 2>/dev/null || true)

    if [ -n "$pids" ]; then
        log_warn "Found existing process(es) on port $port: $pids"
        for pid in $pids; do
            log_info "Killing process $pid..."
            kill -9 "$pid" 2>/dev/null || true
        done
        sleep 1
    fi
}

# Check if port is available
check_port() {
    local port=$1
    if lsof -ti:$port >/dev/null 2>&1; then
        return 1
    fi
    return 0
}

# Wait for API server to be ready
wait_for_api() {
    local max_attempts=30
    local attempt=0

    log_info "Waiting for API server to be ready..."

    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$API_PORT/health" >/dev/null 2>&1; then
            log_success "API server is ready"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
    done

    log_error "API server failed to start within ${max_attempts}s"
    return 1
}

# Check if embedding service is running
check_embed_service() {
    curl -s "http://localhost:$EMBED_PORT/health" >/dev/null 2>&1
}

# Start embedding service if not running
start_embed_service() {
    # Check if already running
    if check_embed_service; then
        log_success "Embedding service already running on port $EMBED_PORT"
        return 0
    fi

    # Check if service directory exists
    if [ ! -d "$EMBED_SERVICE_DIR" ]; then
        log_warn "Embedding service not installed at $EMBED_SERVICE_DIR"
        log_warn "Some features may be unavailable"
        return 1
    fi

    log_info "Starting MLX embedding service..."
    cd "$EMBED_SERVICE_DIR"

    # Start in background, redirect output to log file
    uv run python server.py >> "$EMBED_SERVICE_DIR/server.log" 2>&1 &
    EMBED_PID=$!
    log_info "Embedding service started (PID: $EMBED_PID)"

    # Wait for it to be ready
    local max_attempts=15
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if check_embed_service; then
            log_success "Embedding service is ready"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
    done

    log_warn "Embedding service failed to start (check $EMBED_SERVICE_DIR/server.log)"
    return 1
}

# Main function
main() {
    log_info "Starting JARVIS..."
    log_info "Project root: $PROJECT_ROOT"

    # Set up signal handlers for cleanup
    trap cleanup EXIT INT TERM

    # Step 1: Clean up any existing processes on the API port
    log_info "Checking port $API_PORT..."
    if ! check_port "$API_PORT"; then
        log_warn "Port $API_PORT is in use, cleaning up..."
        kill_port_process "$API_PORT"
    fi
    log_success "Port $API_PORT is available"

    # Step 2: Activate virtual environment if it exists
    if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
        log_info "Activating virtual environment..."
        source "$PROJECT_ROOT/.venv/bin/activate"
        log_success "Virtual environment activated"
    else
        log_warn "No virtual environment found at $PROJECT_ROOT/.venv"
    fi

    # Step 3: Start embedding service (optional, some features need it)
    start_embed_service || true  # Don't fail if embedding service doesn't start

    # Step 4: Start the FastAPI backend
    log_info "Starting FastAPI backend on port $API_PORT..."
    cd "$PROJECT_ROOT"
    uvicorn api.main:app --port "$API_PORT" --host 127.0.0.1 &
    API_PID=$!
    log_info "API server started (PID: $API_PID)"

    # Wait for API to be ready
    if ! wait_for_api; then
        log_error "Failed to start API server"
        exit 1
    fi

    # Step 5: Start the socket server for direct desktop communication
    log_info "Starting socket server..."
    cd "$PROJECT_ROOT"
    uv run python -m jarvis.socket_server &
    SOCKET_PID=$!
    log_info "Socket server started (PID: $SOCKET_PID)"

    # Wait briefly for socket to be ready
    sleep 1
    if [ -e "$SOCKET_PATH" ]; then
        log_success "Socket server is ready at $SOCKET_PATH"
    else
        log_warn "Socket server may not be ready yet"
    fi

    # Step 6: Start the Tauri desktop app
    log_info "Starting JARVIS desktop app..."
    cd "$DESKTOP_DIR"

    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        log_info "Installing desktop dependencies..."
        npm install
    fi

    # Run Tauri in dev mode and wait for it to exit
    npm run tauri dev
    TAURI_EXIT_CODE=$?

    log_info "Desktop app closed (exit code: $TAURI_EXIT_CODE)"

    # Cleanup happens automatically via trap
}

# Run main function
main "$@"
