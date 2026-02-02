# Makefile for JARVIS AI Assistant
# Single source of truth for all development commands
# All commands use uv for package management

.PHONY: help install hooks setup \
        test test-fast test-verbose test-coverage test-file \
        lint format format-check typecheck check \
        verify review health \
        clean clean-all \
        launch api-dev desktop-setup desktop-dev desktop-build frontend-dev

# ============================================================================
# HELP
# ============================================================================

help:
	@echo "JARVIS AI Assistant - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install dependencies via uv sync"
	@echo "  make hooks         Configure git to use .githooks/ directory"
	@echo "  make setup         Full dev setup (install + hooks)"
	@echo ""
	@echo "Testing (all output captured to test_results.txt):"
	@echo "  make test          Run full test suite"
	@echo "  make test-fast     Run tests, stop at first failure"
	@echo "  make test-verbose  Run tests with extra verbosity (-vvv)"
	@echo "  make test-coverage Run tests with coverage report"
	@echo "  make test-file FILE=path  Run single test file"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          Run linter (ruff check)"
	@echo "  make format        Auto-format code (ruff format)"
	@echo "  make format-check  Check formatting without changes"
	@echo "  make typecheck     Run type checker (mypy)"
	@echo "  make check         Run all static checks (lint + format-check + typecheck)"
	@echo ""
	@echo "Verification:"
	@echo "  make verify        Full verification (check + test)"
	@echo "  make review        Generate codebase summary for review"
	@echo "  make health        Print project health summary"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         Remove generated files and caches"
	@echo "  make clean-all     Clean + remove .venv"
	@echo ""
	@echo "Desktop App (Tauri + Svelte):"
	@echo "  make launch        Launch full app (API + desktop) with auto-cleanup"
	@echo "  make api-dev       Start API server on port 8742"
	@echo "  make desktop-setup Install desktop app npm dependencies"
	@echo "  make desktop-dev   Instructions for dev mode"
	@echo "  make desktop-build Build production desktop app"
	@echo "  make frontend-dev  Run Svelte frontend in dev mode"

# ============================================================================
# SETUP
# ============================================================================

install:
	uv sync --extra dev

hooks:
	git config core.hooksPath .githooks
	chmod +x .githooks/*
	@echo "Git hooks enabled from .githooks/"

setup: install hooks
	@echo "Development environment ready!"

# ============================================================================
# TESTING
# All test commands capture output to test_results.txt for full tracebacks
# ============================================================================

test:
	uv run pytest tests/ --tb=long -v --junit-xml=test_results.xml --timeout=30 --timeout-method=thread 2>&1 | tee test_results.txt
	@echo ""
	@echo "Results saved to test_results.txt and test_results.xml"

test-fast:
	uv run pytest tests/ --tb=long -v --maxfail=1 --junit-xml=test_results.xml --timeout=30 --timeout-method=thread 2>&1 | tee test_results.txt
	@echo ""
	@echo "Results saved to test_results.txt and test_results.xml"

test-verbose:
	uv run pytest tests/ --tb=long -vvv --junit-xml=test_results.xml --timeout=30 --timeout-method=thread 2>&1 | tee test_results.txt
	@echo ""
	@echo "Results saved to test_results.txt and test_results.xml"

test-coverage:
	uv run pytest tests/ --tb=long -v --cov=core --cov=models --cov=integrations --cov=benchmarks \
		--cov-report=html --cov-report=term --junit-xml=test_results.xml --timeout=30 --timeout-method=thread 2>&1 | tee test_results.txt
	@echo ""
	@echo "Results saved to test_results.txt, test_results.xml, and htmlcov/"

test-file:
ifndef FILE
	$(error FILE is required. Usage: make test-file FILE=tests/unit/test_foo.py)
endif
	uv run pytest $(FILE) --tb=long -v --junit-xml=test_results.xml --timeout=30 --timeout-method=thread 2>&1 | tee test_results.txt
	@echo ""
	@echo "Results saved to test_results.txt and test_results.xml"

# ============================================================================
# CODE QUALITY
# ============================================================================

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy core/ models/ integrations/ benchmarks/ api/ --ignore-missing-imports

check: lint format-check typecheck
	@echo ""
	@echo "All static checks passed!"

# ============================================================================
# VERIFICATION & REVIEW
# ============================================================================

verify: check test
	@echo ""
	@echo "============================================"
	@echo "VERIFICATION COMPLETE"
	@echo "============================================"
	@echo "All checks passed. Safe to commit/push."

review:
	@echo "============================================"
	@echo "CODEBASE REVIEW SUMMARY"
	@echo "============================================"
	@echo ""
	@echo "## File Statistics"
	@echo "Python files: $$(find core models integrations benchmarks -name '*.py' 2>/dev/null | wc -l | tr -d ' ')"
	@echo "Test files: $$(find tests -name 'test_*.py' 2>/dev/null | wc -l | tr -d ' ')"
	@echo "Total lines: $$(find core models integrations benchmarks -name '*.py' -exec cat {} + 2>/dev/null | wc -l | tr -d ' ')"
	@echo ""
	@echo "## Git Status"
	@git status --short
	@echo ""
	@echo "## Recent Commits"
	@git log --oneline -5
	@echo ""
	@echo "## Lint Status"
	@uv run ruff check . --statistics 2>/dev/null || echo "Run 'make lint' for details"

health:
	@echo "============================================"
	@echo "PROJECT HEALTH SUMMARY"
	@echo "============================================"
	@echo ""
	@echo "## Git Status"
	@echo "Branch: $$(git branch --show-current)"
	@echo "Uncommitted changes: $$(git status --porcelain | wc -l | tr -d ' ') files"
	@echo "Untracked files: $$(git status --porcelain | grep '^??' | wc -l | tr -d ' ')"
	@echo ""
	@echo "## Test Results"
	@uv run pytest tests/ --collect-only -q 2>/dev/null | tail -1 || echo "Run 'make test' to see test results"
	@if [ -f test_results.xml ]; then \
		failed=$$(grep -o 'failures="[0-9]*"' test_results.xml 2>/dev/null | head -1 | grep -o '[0-9]*'); \
		errors=$$(grep -o 'errors="[0-9]*"' test_results.xml 2>/dev/null | head -1 | grep -o '[0-9]*'); \
		if [ -n "$$failed" ] && [ "$$failed" != "0" ]; then \
			echo "Failed tests: $$failed"; \
		elif [ -n "$$errors" ] && [ "$$errors" != "0" ]; then \
			echo "Test errors: $$errors"; \
		else \
			echo "Last run: all passed"; \
		fi; \
	else \
		echo "No test results. Run 'make test'"; \
	fi
	@echo ""
	@echo "## Coverage"
	@if [ -f .coverage ]; then uv run coverage report --format=total 2>/dev/null || echo "No coverage data"; else echo "No coverage data. Run 'make test'"; fi
	@echo ""
	@echo "## Lint Errors (ruff)"
	@lint_output=$$(uv run ruff check . 2>/dev/null); \
	if echo "$$lint_output" | grep -q "All checks passed"; then \
		echo "0 errors"; \
	else \
		lint_count=$$(echo "$$lint_output" | grep "Found [0-9]* error" | grep -o "[0-9]*" | head -1); \
		if [ -n "$$lint_count" ]; then \
			echo "$$lint_count errors (run 'make lint' for details)"; \
		else \
			echo "0 errors"; \
		fi; \
	fi
	@echo ""
	@echo "## Unused Imports (dead code)"
	@unused_output=$$(uv run ruff check . --select=F401 2>/dev/null); \
	if echo "$$unused_output" | grep -q "All checks passed"; then \
		echo "0 unused imports"; \
	else \
		unused=$$(echo "$$unused_output" | grep "Found [0-9]* error" | grep -o "[0-9]*" | head -1); \
		if [ -n "$$unused" ]; then \
			echo "$$unused unused imports"; \
		else \
			echo "0 unused imports"; \
		fi; \
	fi
	@echo ""
	@echo "## Type Errors (mypy)"
	@mypy_output=$$(uv run mypy jarvis/ core/ models/ api/ --ignore-missing-imports 2>&1); \
	if echo "$$mypy_output" | grep -q "Success"; then \
		echo "0 errors"; \
	else \
		mypy_errors=$$(echo "$$mypy_output" | grep ": error:" | wc -l | tr -d ' '); \
		if [ "$$mypy_errors" = "0" ]; then \
			echo "0 errors"; \
		else \
			echo "$$mypy_errors errors (run 'make typecheck' for details)"; \
		fi; \
	fi
	@echo ""
	@echo "## TODOs in Code"
	@todo_count=$$(grep -rn 'TODO\|FIXME\|XXX\|HACK' jarvis/ core/ models/ integrations/ benchmarks/ scripts/ --include='*.py' 2>/dev/null | wc -l | tr -d ' '); \
	echo "$$todo_count TODO/FIXME/XXX/HACK comments"
	@echo ""
	@echo "## Security Scan"
	@secrets=$$(grep -rn 'password\s*=\s*["\x27][^"\x27]*["\x27]\|api_key\s*=\s*["\x27][^"\x27]*["\x27]\|secret\s*=\s*["\x27][^"\x27]*["\x27]' jarvis/ core/ models/ --include='*.py' 2>/dev/null | grep -v 'test_\|_test\|mock\|example\|placeholder\|TODO\|None\|""' | wc -l | tr -d ' '); \
	if [ "$$secrets" = "0" ]; then \
		echo "No hardcoded secrets detected"; \
	else \
		echo "WARNING: $$secrets potential hardcoded secrets found"; \
	fi
	@env_files=$$(find . -name '.env' -o -name '.env.*' 2>/dev/null | grep -v '.venv' | wc -l | tr -d ' '); \
	if [ "$$env_files" != "0" ]; then \
		echo "WARNING: $$env_files .env files found (ensure not committed)"; \
	fi
	@echo ""
	@echo "## ML Models Status"
	@if [ -d "$$HOME/.jarvis/trigger_classifier_model" ]; then \
		echo "Trigger classifier (SVM): present"; \
	else \
		echo "Trigger classifier (SVM): MISSING (~/.jarvis/trigger_classifier_model/)"; \
	fi
	@if [ -d "$$HOME/.jarvis/response_classifier_model" ]; then \
		echo "Response classifier (SVM): present"; \
	else \
		echo "Response classifier (SVM): MISSING (~/.jarvis/response_classifier_model/)"; \
	fi
	@echo ""
	@echo "## Dependencies"
	@echo "Lock file: $$(if [ -f uv.lock ]; then echo 'present'; else echo 'MISSING - run uv sync'; fi)"
	@echo "Venv: $$(if [ -d .venv ]; then echo 'present'; else echo 'MISSING - run make install'; fi)"
	@echo ""
	@echo "## Outdated Packages (top 5)"
	@uv pip list --outdated 2>/dev/null | head -6 || echo "Run 'uv pip list --outdated' to check"

# ============================================================================
# CLEANUP
# ============================================================================

clean:
	rm -f test_results.txt
	rm -f test_results.xml
	rm -f .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -not -path "./.venv/*" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -not -path "./.venv/*" -delete 2>/dev/null || true
	@echo "Cleaned generated files and caches"

clean-all: clean
	rm -rf .venv/
	@echo "Cleaned everything including .venv/"

# ============================================================================
# DESKTOP APP (Tauri + Svelte)
# ============================================================================

launch:
	@./scripts/launch.sh

api-dev:
	uv run uvicorn api.main:app --reload --port 8742

desktop-setup:
	cd desktop && npm install

desktop-dev:
	@echo "Starting API server and Tauri dev mode..."
	@echo "Run 'make api-dev' in a separate terminal first, then run:"
	@echo "  cd desktop && npm run tauri dev"

desktop-build:
	cd desktop && npm run tauri build

frontend-dev:
	cd desktop && npm run dev
