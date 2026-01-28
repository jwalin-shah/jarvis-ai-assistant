.PHONY: dev api tauri both stop test lint format clean help

# Default target
help:
	@echo "JARVIS v2 Commands:"
	@echo ""
	@echo "  make dev       - Start both API and Tauri (recommended)"
	@echo "  make api       - Start v2 API server only (port 8000)"
	@echo "  make tauri     - Start Tauri dev server only (port 1420)"
	@echo "  make stop      - Stop all running servers"
	@echo ""
	@echo "  make test      - Run tests"
	@echo "  make test-gen  - Test generation interactively"
	@echo "  make lint      - Check code style"
	@echo "  make format    - Format code"
	@echo ""
	@echo "  make clean     - Remove cache files"

# Start v2 API server
api:
	@echo "Starting v2 API on http://localhost:8000..."
	cd .. && uv run uvicorn v2.api.main:app --port 8000 --reload

# Start Tauri dev server
tauri:
	@echo "Starting Tauri on http://localhost:1420..."
	cd ../desktop && npm run dev

# Start both servers (API in background, Tauri in foreground)
dev:
	@echo "Starting JARVIS v2..."
	@echo "- API: http://localhost:8000"
	@echo "- Tauri: http://localhost:1420"
	@echo ""
	@make stop 2>/dev/null || true
	@cd .. && uv run uvicorn v2.api.main:app --port 8000 & echo $$! > /tmp/jarvis-api.pid
	@sleep 2
	@echo "API started (PID: $$(cat /tmp/jarvis-api.pid))"
	@echo "Starting Tauri... (Ctrl+C to stop both)"
	@cd ../desktop && npm run dev || (make stop && exit 1)

# Start both in background
both:
	@echo "Starting both servers in background..."
	@make stop 2>/dev/null || true
	@cd .. && uv run uvicorn v2.api.main:app --port 8000 & echo $$! > /tmp/jarvis-api.pid
	@cd ../desktop && npm run dev & echo $$! > /tmp/jarvis-tauri.pid
	@sleep 3
	@echo ""
	@echo "Servers running:"
	@echo "  API:   http://localhost:8000 (PID: $$(cat /tmp/jarvis-api.pid))"
	@echo "  Tauri: http://localhost:1420 (PID: $$(cat /tmp/jarvis-tauri.pid))"
	@echo ""
	@echo "Run 'make stop' to stop both servers"

# Stop all servers
stop:
	@echo "Stopping servers..."
	@-pkill -f "uvicorn v2.api" 2>/dev/null || true
	@-pkill -f "vite.*desktop" 2>/dev/null || true
	@-rm -f /tmp/jarvis-api.pid /tmp/jarvis-tauri.pid 2>/dev/null || true
	@echo "Stopped"

# Run tests
test:
	cd .. && uv run pytest v2/tests -v

# Test generation interactively
test-gen:
	cd .. && uv run python -m v2.scripts.test_generation

# Quick generation test (no prompts)
test-quick:
	@cd .. && uv run python -c " \
from v2.core.imessage import MessageReader; \
from v2.core.models import get_model_loader; \
from v2.core.generation import ReplyGenerator; \
reader = MessageReader(); \
convs = reader.get_conversations(limit=1); \
msgs = reader.get_messages(convs[0].chat_id, limit=10); \
reader.close(); \
messages = [{'text': m.text, 'sender': m.sender, 'is_from_me': m.is_from_me} for m in reversed(msgs) if m.text]; \
print(f'Last message: {messages[-1][\"text\"]}'); \
loader = get_model_loader('qwen-1.5b'); \
gen = ReplyGenerator(loader); \
result = gen.generate_replies(messages); \
print(f'Replies ({result.generation_time_ms:.0f}ms):'); \
[print(f'  {i+1}. {r.text}') for i, r in enumerate(result.replies)]; \
"

# Lint code
lint:
	cd .. && uv run ruff check v2/

# Format code
format:
	cd .. && uv run ruff format v2/

# Clean cache files
clean:
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned"
