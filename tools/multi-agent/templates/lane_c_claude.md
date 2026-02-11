# CLAUDE.md - Lane C: Quality + Regression Gates

You are working in a **multi-agent worktree** for Lane C (Quality + Regression Gates).

## Your Ownership

YOU OWN (can modify freely):
- `tests/` - All test files
- `benchmarks/` - Performance benchmarks
- `evals/` - Evaluation suites

## Restrictions

YOU MUST NOT MODIFY (owned by other lanes):
- `desktop/` - Lane A (App)
- `api/` - Lane A (App)
- `jarvis/router.py` - Lane A (App)
- `jarvis/prompts.py` - Lane A (App)
- `jarvis/retrieval/` - Lane A (App)
- `jarvis/reply_service.py` - Lane A (App)
- `models/` - Lane B (ML)
- `jarvis/classifiers/` - Lane B (ML)
- `jarvis/extractors/` - Lane B (ML)
- `jarvis/contacts/` - Lane B (ML)
- `jarvis/graph/` - Lane B (ML)
- `scripts/train*` - Lane B (ML)
- `scripts/extract*` - Lane B (ML)

SHARED (requires all-lane approval):
- `jarvis/contracts/pipeline.py` - Modify only if necessary, document changes

## Allowed Exceptions

- Test-only fixtures and mocks that reference other lanes' code (read-only usage)
- You CAN import from `jarvis/` in tests to verify behavior

## Key Files to Read First

Before writing tests, read these to understand what you're testing:
- `jarvis/contracts/pipeline.py` - Shared type definitions (test these!)
- `tests/` - Existing test patterns and fixtures
- `benchmarks/` - Existing benchmarks (if any)

## Commands

- **Run tests**: `make test` (reads output from `test_results.txt`)
- **Run fast tests**: `make test-fast` (stops at first failure)
- **Format code**: `make format`
- **Python**: Always use `uv run python ...`
- **Git commit**: `git add <files> && git commit -m "message"`

## Completion Protocol

When your work is done:
1. Run `make format` to fix style
2. Run `make test` and verify your new tests pass (read `test_results.txt`)
3. Commit all changes: `git add <your files> && git commit -m "descriptive message"`
4. Create done sentinel: `touch .agent-done`
5. ALL steps are required for the hub to detect completion

## Pre-existing Failures

These tests fail BEFORE your changes (not your fault, ignore them):
- Category classifier tests may fail due to missing training data
