# Makefile for JARVIS AI Assistant
# Single source of truth for all development commands
# All commands use uv for package management

.PHONY: help install hooks setup \
        test test-fast test-verbose test-coverage test-file \
        lint format format-check typecheck check \
        verify review health \
        clean clean-all \
        launch api-dev desktop-setup desktop-dev desktop-build frontend-dev \
        download-models eval-setup eval eval-view eval-batch \
        dspy-optimize dspy-optimize-mipro dspy-optimize-category dspy-eval dspy-eval-category \
        eval-rag eval-rag-relevance eval-rag-ablation eval-rag-audit \
        benchmark-spec \
        prepare-data finetune-sft finetune-draft fuse-models \
        generate-prefs finetune-orpo fuse-orpo finetune-embedder \
        prepare-dailydialog dailydialog-sweep dailydialog-sweep-quick dailydialog-analyze train-category-svm \
        personalize extract-personal prepare-personal generate-ft-configs finetune-personal evaluate-personal personalize-report fuse-personal \
        db-backup db-backup-export db-backup-migration \
        db-restore db-restore-force db-restore-backup \
        db-health db-health-json db-list \
        db-maintain db-maintain-full db-vacuum db-analyze db-cleanup db-cleanup-dry \
        db-test-migrations db-test-migration db-rollback-test \
        db-drill db-recover db-recover-force \
        db-pre-release db-verify-full

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
	@echo ""
	@echo "Evals & Benchmarks:"
	@echo "  make download-models         Download cross-encoder + draft model"
	@echo "  make eval-setup              Install promptfoo CLI via pnpm"
	@echo "  make eval                    Run promptfoo evaluation"
	@echo "  make eval-view               Open promptfoo results viewer"
	@echo "  make eval-batch              Run batch eval with local checks"
	@echo "  make dspy-optimize           Run DSPy BootstrapFewShot optimization"
	@echo "  make dspy-optimize-mipro     Run DSPy MIPROv2 optimization (global)"
	@echo "  make dspy-optimize-category  Run per-category MIPROv2 optimization"
	@echo "  make dspy-eval               Evaluate saved DSPy program (global)"
	@echo "  make dspy-eval-category      Evaluate per-category DSPy programs"
	@echo "  make eval-rag                Run full RAG quality evaluation"
	@echo "  make eval-rag-relevance      RAG retrieval relevance only"
	@echo "  make eval-rag-ablation       RAG generation ablation only"
	@echo "  make eval-rag-audit          RAG pair quality audit only"
	@echo "  make benchmark-spec          Run speculative decoding A/B benchmark"
	@echo ""
	@echo "Fine-Tuning Pipeline:"
	@echo "  make prepare-data            Download SOC-2508 and convert to SFT format"
	@echo "  make finetune-sft            Fine-tune LFM 1.2B on SOC data (QLoRA)"
	@echo "  make finetune-draft          Fine-tune LFM 0.3B draft model (QLoRA)"
	@echo "  make fuse-models             Fuse LoRA adapters into base models"
	@echo "  make generate-prefs          Generate ORPO preference pairs via Gemini"
	@echo "  make finetune-orpo           Run ORPO preference alignment"
	@echo "  make fuse-orpo               Fuse ORPO adapter into final model"
	@echo "  make finetune-embedder       Fine-tune BGE-small on conversation triplets"
	@echo "  make train-category-svm      Train production SVM (requires --label-map flag)"
	@echo ""
	@echo "Database Reliability & Maintenance:"
	@echo "  make db-backup               Create hot backup of database"
	@echo "  make db-restore              Restore from latest backup"
	@echo "  make db-health               Run database health check"
	@echo "  make db-maintain             Run daily maintenance (ANALYZE, checkpoint)"
	@echo "  make db-maintain-full        Run full maintenance (includes VACUUM, backup)"
	@echo "  make db-test-migrations      Test all database migrations"
	@echo "  make db-drill                Run backup/restore drill"
	@echo "  make db-recover              Attempt database recovery"
	@echo "  make db-list                 List available backups"
	@echo "  make db-cleanup              Clean up old backups"
	@echo "  make db-pre-release          Run pre-release database checks"
	@echo "  make db-verify-full          Complete database verification"

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

upgrade:
	@echo "Upgrading Python dependencies..."
	uv lock --upgrade
	uv sync
	@echo "Upgrading Desktop dependencies..."
	cd desktop && pnpm update
	@echo "Upgrade complete!"

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
	uv run pytest tests/ --tb=long -v --cov=jarvis --cov=api --cov=core --cov=models --cov=integrations --cov=contracts --cov=benchmarks \
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
# TIERED TEST TARGETS (New Architecture)
# ============================================================================

test-deps:
	@uv run python -c "from tests.dependencies import print_dependency_report; print_dependency_report()"

test-unit:
	@echo "Running unit tests (no external deps)..."
	uv run pytest tests/unit/ -v --tb=short -m "not slow" 2>&1 | tee test_results.txt

test-integration:
	@echo "Running integration tests (mocked deps)..."
	uv run pytest tests/integration/ -v --tb=short -m "not slow" 2>&1 | tee test_results.txt

test-hardware:
	@echo "Running hardware tests (requires Apple Silicon, 16GB RAM)..."
	uv run pytest tests/hardware/ -v --tb=short 2>&1 | tee test_results.txt

test-slow:
	@echo "Running slow tests..."
	RUN_SLOW_TESTS=1 uv run pytest tests/ -v --tb=short -m "slow" 2>&1 | tee test_results.txt

test-ci:
	@echo "Running CI test suite (unit + integration, no hardware)..."
	uv run pytest tests/unit/ tests/integration/ -v --tb=short -m "not hardware and not slow" 2>&1 | tee test_results.txt

test-full:
	@echo "Running full test suite (all tiers)..."
	uv run pytest tests/ -v --tb=short 2>&1 | tee test_results.txt

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
	uv run mypy jarvis/ core/ models/ integrations/ api/ --ignore-missing-imports

check: lint format-check typecheck svelte-check
	@echo ""
	@echo "All static checks passed!"

svelte-check:
	@echo "Running svelte-check..."
	@cd desktop && pnpm check

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
# EVALS & BENCHMARKS
# ============================================================================

# --- Model Management ---

download-models:
	@echo "Downloading cross-encoder model..."
	uv run huggingface-cli download cross-encoder/ms-marco-MiniLM-L-6-v2
	@echo ""
	@echo "Downloading draft model for speculative decoding..."
	uv run huggingface-cli download mlx-community/LFM2-350M-4bit
	@echo ""
	@echo "Models downloaded successfully."

# --- Promptfoo Evals ---

eval-setup:
	pnpm install -g promptfoo
	@echo "promptfoo installed. Run 'make eval' to evaluate."

eval:
	OPENAI_API_KEY=$$(grep CEREBRAS_API_KEY .env | cut -d= -f2) \
	OPENAI_BASE_URL=https://api.cerebras.ai/v1 \
	uv run npx promptfoo eval -c evals/promptfoo.yaml

eval-view:
	uv run npx promptfoo view

eval-batch:
	uv run python evals/batch_eval.py $(ARGS)

# --- DSPy Optimization ---

dspy-optimize:
	uv run python evals/dspy_optimize.py

dspy-optimize-mipro:
	uv run python evals/dspy_optimize.py --optimizer mipro

dspy-optimize-category:
	uv run python evals/dspy_optimize.py --per-category --optimizer mipro

dspy-eval:
	uv run python evals/dspy_optimize.py --eval-only

dspy-eval-category:
	uv run python evals/dspy_optimize.py --eval-only --per-category

# --- RAG Quality Evaluation ---

eval-rag:
	uv run python evals/rag_eval.py

eval-rag-relevance:
	uv run python evals/rag_eval.py --relevance-only

eval-rag-ablation:
	uv run python evals/rag_eval.py --ablation-only

eval-rag-audit:
	uv run python evals/rag_eval.py --audit-only

# --- Benchmarks ---

benchmark-spec:
	uv run python evals/speculative_benchmark.py

# ============================================================================
# CATEGORY CLASSIFIER TRAINING
# ============================================================================

# Train LinearSVC on labeled data
train-category-svm:
	uv run python scripts/train_category_svm.py

# ============================================================================
# FINE-TUNING PIPELINE
# ============================================================================

# Step 1: Prepare SOC-2508 training data
prepare-data:
	uv run python scripts/prepare_soc_data.py

# Step 2: SFT fine-tune LFM 1.2B
finetune-sft:
	uv run mlx_lm.lora --config fine_tune_config.yaml

# Step 3: SFT fine-tune LFM 0.3B draft model
finetune-draft:
	uv run mlx_lm.lora --config fine_tune_config_draft.yaml

# Step 4: Fuse LoRA adapters into base models
fuse-models:
	uv run mlx_lm.fuse \
		--model LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit \
		--adapter-path adapters/lfm-1.2b-soc-sft \
		--save-path models/lfm-1.2b-soc-fused
	uv run mlx_lm.fuse \
		--model mlx-community/LFM2-350M-4bit \
		--adapter-path adapters/lfm-0.3b-soc-sft \
		--save-path models/lfm-0.3b-soc-fused

# Step 5: Generate ORPO preference pairs with Gemini
generate-prefs:
	uv run python scripts/generate_preference_pairs.py

# Step 6: ORPO preference alignment on SFT-fused model
finetune-orpo:
	uv run mlx_lm.lora \
		--model models/lfm-1.2b-soc-fused \
		--data data/soc_orpo \
		--train \
		--training-mode orpo \
		--batch-size 2 \
		--grad-checkpoint \
		--iters 1000 \
		--adapter-path adapters/lfm-1.2b-orpo

# Step 7: Fuse ORPO adapter into final model
fuse-orpo:
	uv run mlx_lm.fuse \
		--model models/lfm-1.2b-soc-fused \
		--adapter-path adapters/lfm-1.2b-orpo \
		--save-path models/lfm-1.2b-final


# ============================================================================
# PERSONAL FINE-TUNING PIPELINE
# ============================================================================

# Full pipeline: extract -> prepare -> generate configs -> fine-tune -> evaluate
personalize: extract-personal prepare-personal generate-ft-configs finetune-personal evaluate-personal
	@echo "Personalization complete! Run 'make personalize-report' for results."

# Step 1: Extract iMessage pairs
extract-personal:
	uv run python scripts/extract_personal_data.py

# Step 2: Prepare training data (both variants)
prepare-personal:
	uv run python scripts/prepare_personal_data.py --both

# Step 3: Generate fine-tuning configs
generate-ft-configs:
	uv run python scripts/generate_ft_configs.py

# Step 4: Fine-tune all variants
finetune-personal:
	@echo "Fine-tuning all personal variants..."
	@for config in ft_configs/personal_*.yaml; do \
		echo ""; \
		echo "========================================"; \
		echo "Training: $$config"; \
		echo "========================================"; \
		uv run python scripts/train_personal.py --config "$$config" || echo "FAILED: $$config"; \
	done

# Step 5: Evaluate all variants
evaluate-personal:
	uv run python scripts/evaluate_personal_ft.py

# View evaluation results
personalize-report:
	uv run python scripts/evaluate_personal_ft.py --report-only

# Fuse adapters into base models for fast inference
fuse-personal:
	@for adapter_dir in adapters/personal/*/; do \
		adapter_name=$$(basename "$$adapter_dir"); \
		echo "Fusing $$adapter_name..."; \
		config_file=$$(ls ft_configs/personal_*$$(echo "$$adapter_name" | sed 's/-/_/g')*.yaml 2>/dev/null | head -1); \
		if [ -n "$$config_file" ]; then \
			model=$$(grep '^model:' "$$config_file" | sed 's/model: *//; s/"//g'); \
			uv run mlx_lm.fuse --model "$$model" --adapter-path "$$adapter_dir" --save-path "models/personal/$$adapter_name" || echo "FAILED: $$adapter_name"; \
		else \
			echo "  No config found for $$adapter_name, skipping"; \
		fi; \
	done

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
	cd desktop && pnpm install

desktop-dev:
	@echo "Starting API server and Tauri dev mode..."
	@echo "Run 'make api-dev' in a separate terminal first, then run:"
	@echo "  cd desktop && npm run tauri dev"

desktop-build:
	cd desktop && npm run tauri build

frontend-dev:
	cd desktop && npm run dev


# ============================================================================
# DEVELOPER PRODUCTIVITY ENHANCEMENTS
# See docs/DEV_PRODUCTIVITY_PLAN.md for full documentation
# ============================================================================

# --- Cache Configuration ---

PYTHON_FILES_HASH := $(shell find jarvis models core api -name '*.py' -type f -exec md5 -q {} \; 2>/dev/null | md5 -q)
DEPS_HASH := $(shell md5 -q pyproject.toml uv.lock 2>/dev/null)
JARVIS_CACHE_DIR := $(HOME)/.cache/jarvis

# --- Bootstrap Speedups ---

install-fast:
	@echo "Installing dependencies (using uv cache)..."
	uv sync --extra dev --frozen

restore-venv:
	@if [ -d "$(JARVIS_CACHE_DIR)/venvs/$(shell git rev-parse --short HEAD)" ]; then \
		echo "Restoring venv from cache..."; \
		cp -r "$(JARVIS_CACHE_DIR)/venvs/$(shell git rev-parse --short HEAD)" .venv; \
		echo "Venv restored!"; \
	else \
		echo "No cached venv found, running fresh install..."; \
		make install && make cache-venv; \
	fi

cache-venv:
	@mkdir -p "$(JARVIS_CACHE_DIR)/venvs"
	@if [ -d ".venv" ]; then \
		cp -r .venv "$(JARVIS_CACHE_DIR)/venvs/$(shell git rev-parse --short HEAD)"; \
		echo "Venv cached for commit $(shell git rev-parse --short HEAD)"; \
	fi

setup-parallel:
	@echo "Running parallel setup..."
	@(make install-fast > /tmp/install.log 2>&1 && echo "✓ Dependencies installed") &
	@(make hooks > /tmp/hooks.log 2>&1 && echo "✓ Git hooks configured") &
	@wait
	@echo "Parallel setup complete!"

# --- Test Acceleration ---

test-changed:
	@changed_files=$$(git diff --name-only HEAD~1 | grep '\.py$$'); \
	if [ -z "$$changed_files" ]; then \
		echo "No Python files changed since last commit"; \
		exit 0; \
	fi; \
	echo "Testing changed files: $$changed_files"; \
	uv run pytest $$changed_files --tb=short -v --timeout=30 2>&1 | tee test_results.txt

test-ff:
	@echo "Running tests (fail-fast mode)..."
	uv run pytest tests/ --tb=line --maxfail=1 --timeout=30 -q 2>&1 | tee test_results.txt

test-parallel:
	@echo "Running tests in parallel (2 workers for 8GB RAM constraint)..."
	uv run pytest tests/ -n 2 --tb=short --timeout=30 2>&1 | tee test_results.txt

test-failed:
	@echo "Re-running previously failed tests..."
	uv run pytest tests/ --lf --tb=short -v --timeout=30 2>&1 | tee test_results.txt

test-smart:
	@echo "Running failed tests first, then others..."
	uv run pytest tests/ --ff --tb=short -v --timeout=30 2>&1 | tee test_results.txt

test-module:
	@if [ -z "$(MODULE)" ]; then \
		echo "Usage: make test-module MODULE=jarvis/contacts"; \
		exit 1; \
	fi
	@echo "Testing module: $(MODULE)"
	@base=$$(basename $(MODULE)); \
	uv run pytest tests/unit/test_$${base}*.py tests/integration/test_$${base}*.py --tb=short -v --timeout=30 2>&1 | tee test_results.txt

test-watch:
	@echo "Starting test watch mode (requires pytest-watch)..."
	@uv run ptw tests/ --onpass "echo '✓ All tests passed'" --onfail "echo '✗ Tests failed'" -q 2>/dev/null || \
		(echo "Installing pytest-watch..."; uv add --dev pytest-watch; uv run ptw tests/ -q)

test-profile:
	@echo "Profiling test execution times..."
	@uv run pytest tests/ --durations=20 --tb=no -q 2>&1 | tee test_profile.txt
	@echo ""
	@echo "Top 10 slowest tests:"
	@head -15 test_profile.txt | tail -10

# --- Incremental Caching ---

lint-incremental:
	@mkdir -p .cache/lint
	@if [ -f .cache/lint/last_run_hash ] && [ "$$(cat .cache/lint/last_run_hash)" = "$(PYTHON_FILES_HASH)" ]; then \
		echo "✓ Lint cache valid (no Python changes since last run)"; \
	else \
		echo "Running linter..."; \
		uv run ruff check . && echo "$(PYTHON_FILES_HASH)" > .cache/lint/last_run_hash; \
	fi

typecheck-incremental:
	@mkdir -p .cache/typecheck
	@if [ -f .cache/typecheck/last_run_hash ] && [ "$$(cat .cache/typecheck/last_run_hash)" = "$(PYTHON_FILES_HASH)" ]; then \
		echo "✓ Type check cache valid (no Python changes since last run)"; \
	else \
		echo "Running type checker..."; \
		uv run mypy jarvis/ core/ models/ api/ --ignore-missing-imports && echo "$(PYTHON_FILES_HASH)" > .cache/typecheck/last_run_hash; \
	fi

format-incremental:
	@mkdir -p .cache/format
	@if [ -f .cache/format/last_run_hash ] && [ "$$(cat .cache/format/last_run_hash)" = "$(PYTHON_FILES_HASH)" ]; then \
		echo "✓ Format check cache valid"; \
	else \
		echo "Checking format..."; \
		uv run ruff format --check . && echo "$(PYTHON_FILES_HASH)" > .cache/format/last_run_hash; \
	fi

verify-fast: format-incremental lint-incremental typecheck-incremental test-ff
	@echo ""
	@echo "✅ Fast verification complete!"


code-review-checklist:
	@echo "========================================"
	@echo "Code Review Checklist"
	@echo "========================================"
	@echo ""
	@echo "Before requesting review:"
	@echo "  [ ] make verify-fast passes"
	@echo "  [ ] Self-review: git diff --cached"
	@echo "  [ ] Test coverage maintained/improved"
	@echo "  [ ] Documentation updated (if needed)"
	@echo "  [ ] CHANGELOG.md updated (if user-facing)"
	@echo ""
	@echo "For ML changes:"
	@echo "  [ ] Benchmarks run and documented"
	@echo "  [ ] Memory usage checked (8GB constraint)"
	@echo ""
	@echo "For API changes:"
	@echo "  [ ] Contracts updated"
	@echo "  [ ] Integration tests pass"
	@echo ""

# --- Cache Management ---

clear-caches:
	@echo "Clearing all development caches..."
	@rm -rf .cache/
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@rm -rf .ruff_cache/
	@echo "Caches cleared. Next verify will run full checks."

clear-venv-cache:
	@echo "Clearing venv cache..."
	@rm -rf $(JARVIS_CACHE_DIR)/venvs/
	@echo "Venv cache cleared."

# --- Performance Profiling ---

profile-imports:
	@echo "Profiling import times..."
	@python -X importtime -c "import jarvis" 2>&1 | tail -20

profile-tests:
	@echo "Running tests with profiling..."
	@uv run pytest tests/ --profile-svg --tb=no -q 2>/dev/null || \
		(echo "Installing pytest-profiling..."; uv add --dev pytest-profiling; uv run pytest tests/ --profile-svg --tb=no -q)
	@echo "Profile saved to prof/"

# --- Development Health Check ---

dev-health:
	@echo "========================================"
	@echo "Developer Environment Health"
	@echo "========================================"
	@echo ""
	@echo "Git Status:"
	@echo "  Branch: $$(git branch --show-current)"
	@echo "  Uncommitted: $$(git status --porcelain | wc -l | tr -d ' ') files"
	@echo ""
	@echo "Virtual Environment:"
	@echo "  Status: $$(if [ -d .venv ]; then echo '✓ present'; else echo '✗ missing'; fi)"
	@echo "  Size: $$(du -sh .venv 2>/dev/null | cut -f1 || echo 'N/A')"
	@echo ""
	@echo "Cache Status:"
	@echo "  Lint cache: $$(if [ -f .cache/lint/last_run_hash ]; then echo '✓ valid'; else echo '✗ none'; fi)"
	@echo "  Type cache: $$(if [ -f .cache/typecheck/last_run_hash ]; then echo '✓ valid'; else echo '✗ none'; fi)"
	@echo ""
	@echo "Recent Test Results:"
	@if [ -f test_results.txt ]; then \
		if grep -q "passed" test_results.txt; then \
			passed=$$(grep -o "[0-9]* passed" test_results.txt | tail -1 | grep -o "[0-9]*" || echo "0"); \
			failed=$$(grep -o "[0-9]* failed" test_results.txt | tail -1 | grep -o "[0-9]*" || echo "0"); \
			echo "  ✓ $$passed tests passed"; \
			if [ "$$failed" != "0" ] && [ "$$failed" != "" ]; then \
				echo "  ✗ $$failed tests failed"; \
			fi; \
		else \
			echo "  ⚠ No test results found. Run: make test"; \
		fi; \
	else \
		echo "  ⚠ No test results found. Run: make test"; \
	fi
	@echo ""
	@echo "Last Commit:"
	@git log -1 --oneline
	@echo ""

# ============================================================================
# DATABASE RELIABILITY & MAINTENANCE
# See docs/DATABASE_RELIABILITY_PLAN.md for full documentation
# ============================================================================

db-backup:
	@uv run python scripts/db_maintenance.py backup --type hot

db-backup-export:
	@uv run python scripts/db_maintenance.py backup --type export

db-backup-migration:
	@uv run python scripts/db_maintenance.py backup --type migration

db-restore:
	@uv run python scripts/db_maintenance.py restore

db-restore-force:
	@uv run python scripts/db_maintenance.py restore --force

db-restore-backup:
ifndef BACKUP
	$(error BACKUP is required. Usage: make db-restore-backup BACKUP=path/to/backup.db)
endif
	@uv run python scripts/db_maintenance.py restore --backup $(BACKUP) --force

# ============================================================================
# DATABASE HEALTH & MONITORING
# ============================================================================

db-health:
	@uv run python scripts/db_maintenance.py health

db-health-json:
	@uv run python scripts/db_maintenance.py health --json

db-list:
	@uv run python scripts/db_maintenance.py list

# ============================================================================
# DATABASE MAINTENANCE
# ============================================================================

db-maintain:
	@uv run python scripts/db_maintenance.py maintain --daily

db-maintain-full:
	@uv run python scripts/db_maintenance.py maintain --full

db-vacuum:
	@sqlite3 ~/.jarvis/jarvis.db "VACUUM"
	@echo "Database vacuumed successfully"

db-analyze:
	@sqlite3 ~/.jarvis/jarvis.db "ANALYZE"
	@echo "Database analyzed successfully"

db-cleanup:
	@uv run python scripts/db_maintenance.py cleanup --max-age 7

db-cleanup-dry:
	@uv run python scripts/db_maintenance.py cleanup --max-age 7 --dry-run

# ============================================================================
# DATABASE MIGRATION TESTING
# ============================================================================

db-test-migrations:
	@uv run python scripts/db_maintenance.py test-migrations

db-test-migration:
ifndef VERSION
	$(error VERSION is required. Usage: make db-test-migration VERSION=5)
endif
	@uv run python scripts/db_maintenance.py test-migrations --from-version $(VERSION)

db-rollback-test:
ifndef FROM_VERSION
	$(error FROM_VERSION is required. Usage: make db-rollback-test FROM_VERSION=7 TO_VERSION=6)
endif
ifndef TO_VERSION
	$(error TO_VERSION is required. Usage: make db-rollback-test FROM_VERSION=7 TO_VERSION=6)
endif
	@uv run python scripts/db_maintenance.py rollback-test --from-version $(FROM_VERSION) --to-version $(TO_VERSION)

# ============================================================================
# DATABASE RELIABILITY DRILLS & RECOVERY
# ============================================================================

db-drill:
	@uv run python scripts/db_maintenance.py drill

db-recover:
	@uv run python scripts/db_maintenance.py recover

db-recover-force:
	@uv run python scripts/db_maintenance.py recover --force

# ============================================================================
# PRE-RELEASE DATABASE CHECKLIST
# ============================================================================

db-pre-release: db-backup db-test-migrations db-drill
	@echo ""
	@echo "========================================"
	@echo "PRE-RELEASE DATABASE CHECKLIST"
	@echo "========================================"
	@echo ""
	@echo "✓ Backup created"
	@echo "✓ Migrations tested"
	@echo "✓ Backup/restore drill completed"
	@echo ""
	@echo "Database is ready for release."

# ============================================================================
# COMPLETE DATABASE RELIABILITY VERIFICATION
# ============================================================================

db-verify-full: db-health db-backup db-test-migrations db-drill db-maintain
	@echo ""
	@echo "========================================"
	@echo "COMPLETE DATABASE VERIFICATION"
	@echo "========================================"
	@echo ""
	@echo "All database reliability checks passed!"

# ============================================================================
# AUTONOMOUS LOOP RUNNER
# ============================================================================

auto-loop:
ifndef PROMPT
	$(error PROMPT is required. Usage: make auto-loop PROMPT="Fix issues" END_CONDITION="All resolved" [AGENT=claude] [REVIEWER=claude-haiku])
endif
ifndef END_CONDITION
	$(error END_CONDITION is required.)
endif
	@./scripts/autonomous_loop.sh \
		--prompt "$(PROMPT)" \
		--end-condition "$(END_CONDITION)" \
		--max-iterations $(or $(MAX_ITER),20) \
		--cooldown $(or $(COOLDOWN),10) \
		$(if $(AGENT),--agent $(AGENT)) \
		$(if $(MODEL),--model $(MODEL)) \
		$(if $(REVIEWER),--reviewer $(REVIEWER)) \
		$(if $(REVIEWER_PROMPT),--reviewer-prompt $(REVIEWER_PROMPT))

auto-loop-file:
ifndef PROMPT_FILE
	$(error PROMPT_FILE is required. Usage: make auto-loop-file PROMPT_FILE=tasks/prompt.md END_CONDITION="Done")
endif
ifndef END_CONDITION
	$(error END_CONDITION is required.)
endif
	@./scripts/autonomous_loop.sh \
		--prompt-file "$(PROMPT_FILE)" \
		--end-condition "$(END_CONDITION)" \
		--max-iterations $(or $(MAX_ITER),20) \
		--cooldown $(or $(COOLDOWN),10) \
		$(if $(AGENT),--agent $(AGENT)) \
		$(if $(MODEL),--model $(MODEL)) \
		$(if $(REVIEWER),--reviewer $(REVIEWER)) \
		$(if $(REVIEWER_PROMPT),--reviewer-prompt $(REVIEWER_PROMPT))

auto-loop-stop:
	@touch tasks/.stop-loop
	@echo "Stop signal sent. Loop will exit after current iteration."

auto-loop-status:
	@if [ -f tasks/loop-status.md ]; then \
		echo "=== Loop Status ==="; \
		head -5 tasks/loop-status.md; \
		echo ""; \
		echo "=== Recent Activity ==="; \
		tail -20 tasks/loop-status.md; \
	else \
		echo "No active loop. Start one with: make auto-loop"; \
	fi

auto-loop-dry:
ifndef PROMPT
	$(error PROMPT is required.)
endif
	@./scripts/autonomous_loop.sh \
		--prompt "$(PROMPT)" \
		--end-condition "$(or $(END_CONDITION),Task complete)" \
		$(if $(AGENT),--agent $(AGENT)) \
		$(if $(REVIEWER),--reviewer $(REVIEWER)) \
		--dry-run
