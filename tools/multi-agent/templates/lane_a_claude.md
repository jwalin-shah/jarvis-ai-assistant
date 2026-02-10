# CLAUDE.md - Lane A: App + Orchestration

You are working in a **multi-agent worktree** for Lane A (App + Orchestration).

## Your Ownership

YOU OWN (can modify freely):
- `desktop/` - Tauri desktop app (Rust + Svelte)
- `api/` - API layer
- `jarvis/router.py` - Message routing
- `jarvis/prompts.py` - Prompt templates
- `jarvis/retrieval/` - RAG retrieval
- `jarvis/reply_service.py` - Reply generation service

## Restrictions

YOU MUST NOT MODIFY (owned by other lanes):
- `jarvis/classifiers/` - Lane B (ML)
- `jarvis/extractors/` - Lane B (ML)
- `jarvis/contacts/` - Lane B (ML)
- `jarvis/graph/` - Lane B (ML)
- `models/` - Lane B (ML)
- `scripts/train*` - Lane B (ML)
- `scripts/extract*` - Lane B (ML)
- `tests/` - Lane C (QA)
- `benchmarks/` - Lane C (QA)
- `evals/` - Lane C (QA)

SHARED (requires all-lane approval):
- `jarvis/contracts/pipeline.py` - Modify only if necessary, document changes

## Key Files to Read First

Before making changes, read these to understand the current architecture:
- `jarvis/contracts/pipeline.py` - Shared type definitions
- `jarvis/router.py` - Current routing logic
- `jarvis/reply_service.py` - Current reply generation
- `jarvis/prompts.py` - Prompt building

## Commands

- **Run tests**: `make test` (reads output from `test_results.txt`)
- **Format code**: `make format`
- **Python**: Always use `uv run python ...`
- **Git commit**: `git add <files> && git commit -m "message"`
- **Check types**: `make typecheck`

## Completion Protocol

When your work is done:
1. Run `make format` to fix style
2. Run `make test` and verify tests pass (read `test_results.txt`)
3. Commit all changes: `git add <your files> && git commit -m "descriptive message"`
4. Create done sentinel: `touch .agent-done`
5. ALL steps are required for the hub to detect completion
