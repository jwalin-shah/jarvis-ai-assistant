#!/bin/bash
# Start mlx_lm server exposing LFM-0.7B via OpenAI-compatible endpoints
# Endpoints: /v1/chat/completions, /v1/completions, /v1/models
#
# Usage: ./scripts/start_mlx_server.sh [--port PORT]

set -euo pipefail

PORT="${1:-8000}"
MODEL_PATH="$(dirname "$0")/../models/lfm-0.7b-4bit"
MODEL_PATH="$(cd "$MODEL_PATH" && pwd)"

echo "Starting mlx_lm server..."
echo "  Model: $MODEL_PATH"
echo "  Port:  $PORT"
echo "  Endpoints:"
echo "    GET  http://127.0.0.1:$PORT/v1/models"
echo "    POST http://127.0.0.1:$PORT/v1/chat/completions"
echo "    POST http://127.0.0.1:$PORT/v1/completions"
echo ""

cd "$(dirname "$0")/.."

exec uv run python -m mlx_lm server \
    --model "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --trust-remote-code
