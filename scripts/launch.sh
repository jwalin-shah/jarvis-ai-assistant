#!/bin/bash
# JARVIS Launcher Script
# Starts the FastAPI backend and Tauri desktop app together.
# Ensures proper cleanup when the app closes.

set -e

# Configuration
API_PORT=8742
SOCKET_PORT=8743
FRONTEND_PORT=1420
SOCKET_PORT=8743
SOCKET_PATH="$HOME/.jarvis/jarvis.sock"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DESKTOP_DIR="$PROJECT_ROOT/desktop"
API_PID=""
SOCKET_PID=""
WORKER_PID=""
TAURI_PID=""
SOCKET_MONITOR_PID=""
SOCKET_RESTART_COUNT=0
MAX_SOCKET_RESTARTS=5
SOCKET_RESTART_DELAY=3

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

    # Kill the socket monitor if running
    if [ -n "$SOCKET_MONITOR_PID" ] && kill -0 "$SOCKET_MONITOR_PID" 2>/dev/null; then
        log_info "Stopping socket monitor (PID: $SOCKET_MONITOR_PID)..."
        kill "$SOCKET_MONITOR_PID" 2>/dev/null || true
        wait "$SOCKET_MONITOR_PID" 2>/dev/null || true
    fi

    # Kill the socket server if we started it
    if [ -n "$SOCKET_PID" ] && kill -0 "$SOCKET_PID" 2>/dev/null; then
        log_info "Stopping socket server (PID: $SOCKET_PID)..."
        kill "$SOCKET_PID" 2>/dev/null || true
        wait "$SOCKET_PID" 2>/dev/null || true
    fi

    # Kill the background worker if we started it
    if [ -n "$WORKER_PID" ] && kill -0 "$WORKER_PID" 2>/dev/null; then
        log_info "Stopping background worker (PID: $WORKER_PID)..."
        kill "$WORKER_PID" 2>/dev/null || true
        wait "$WORKER_PID" 2>/dev/null || true
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

# Check if socket server is healthy
is_socket_healthy() {
    # Check if process is running
    if [ -z "$SOCKET_PID" ] || ! kill -0 "$SOCKET_PID" 2>/dev/null; then
        log_warn "Socket health: process ${SOCKET_PID:-<unknown>} is not running"
        return 1
    fi
    # Check if socket file exists
    if [ ! -S "$SOCKET_PATH" ]; then
        log_warn "Socket health: socket file $SOCKET_PATH is missing"
        return 1
    fi
    return 0
}

# Start socket server
start_socket_server() {
    log_info "Starting socket server (models load on-demand)..."
    log_info "Ensuring socket port $SOCKET_PORT is free..."
    if ! check_port "$SOCKET_PORT"; then
        log_warn "Socket port $SOCKET_PORT already in use, killing existing listener..."
        kill_port_process "$SOCKET_PORT"
    fi
    cd "$PROJECT_ROOT"
    uv run python -m jarvis.socket_server --no-preload &
    SOCKET_PID=$!
    log_info "Socket server started (PID: $SOCKET_PID)"

    # Wait up to 5 seconds for socket to be ready
    local socket_wait=0
    while [ $socket_wait -lt 50 ] && [ ! -e "$SOCKET_PATH" ]; do
        sleep 0.1
        socket_wait=$((socket_wait + 1))
    done

    if [ -e "$SOCKET_PATH" ]; then
        log_success "Socket server ready at $SOCKET_PATH (${socket_wait}00ms)"
    else
        log_warn "Socket server starting in background (app will use SQLite fallback)"
    fi
}

# Monitor socket server health and restart if needed
monitor_socket_server() {
    while true; do
        sleep 10

        # Skip check if we've exceeded max restarts
        if [ $SOCKET_RESTART_COUNT -ge $MAX_SOCKET_RESTARTS ]; then
            log_error "Socket server has crashed $MAX_SOCKET_RESTARTS times. Giving up."
            break
        fi

        # Check if socket is healthy
        if ! is_socket_healthy; then
            log_warn "Socket server appears unhealthy (restart #$((SOCKET_RESTART_COUNT + 1))/$MAX_SOCKET_RESTARTS)"

            # Clean up old socket file if it exists
            if [ -e "$SOCKET_PATH" ]; then
                rm -f "$SOCKET_PATH"
            fi

            # Kill old process if still running
            if [ -n "$SOCKET_PID" ] && kill -0 "$SOCKET_PID" 2>/dev/null; then
                kill "$SOCKET_PID" 2>/dev/null || true
                wait "$SOCKET_PID" 2>/dev/null || true
            fi

            SOCKET_RESTART_COUNT=$((SOCKET_RESTART_COUNT + 1))

            # Wait before restarting
            sleep $SOCKET_RESTART_DELAY

            # Restart the socket server
            start_socket_server
        fi
    done
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

    # Step 3: Start the FastAPI backend
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

    # Step 4: Start the socket server for direct desktop communication
    # Use --no-preload to avoid blocking startup with model loading
    # Models will load on-demand when first used (first message generation, etc.)
    start_socket_server

    # Start socket health monitor in background
    log_info "Starting socket health monitor..."
    monitor_socket_server &
    SOCKET_MONITOR_PID=$!

    # Step 5: Start the background task worker
    log_info "Starting background task worker..."
    cd "$PROJECT_ROOT"
    uv run python scripts/start_worker_loop.py &
    WORKER_PID=$!
    log_info "Background worker started (PID: $WORKER_PID)"

    # Step 6: Start the Tauri desktop app
    log_info "Starting JARVIS desktop app..."
    log_info "Ensuring frontend port $FRONTEND_PORT is free..."
    if ! check_port "$FRONTEND_PORT"; then
        log_warn "Frontend port $FRONTEND_PORT already in use, killing existing listener..."
        kill_port_process "$FRONTEND_PORT"
    fi
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
