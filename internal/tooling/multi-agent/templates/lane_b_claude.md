# CLAUDE.md - Lane B: ML + Extraction + Classification

You are working in a **multi-agent worktree** for Lane B (ML + Extraction + Classification).

## Your Ownership

YOU OWN (can modify freely):

- `models/` - Model weights and configs
- `jarvis/classifiers/` - Classification pipeline
- `jarvis/extractors/` - Entity/fact extraction
- `jarvis/contacts/` - Contact profiles and fact extraction
- `jarvis/graph/` - Knowledge graph
- `jarvis/search/` - Semantic search and embeddings
- `scripts/train*` - Training scripts
- `scripts/extract*` - Extraction scripts

## Restrictions

YOU MUST NOT MODIFY (owned by other lanes):

- `desktop/` - Lane A (App)
- `api/` - Lane A (App)
- `jarvis/router.py` - Lane A (App)
- `jarvis/prompts.py` - Lane A (App)
- `jarvis/retrieval/` - Lane A (App)
- `jarvis/reply_service.py` - Lane A (App)
- `tests/` - Lane C (QA)
- `benchmarks/` - Lane C (QA)
- `evals/` - Lane C (QA)

SHARED (requires all-lane approval):

- `jarvis/contracts/pipeline.py` - Modify only if necessary, document changes

## Key Files to Read First

Before making changes, read these to understand the current architecture:

- `jarvis/contracts/pipeline.py` - Shared type definitions (the contract)
- `jarvis/classifiers/response_mobilization.py` - Main classifier
- `jarvis/contacts/fact_extractor.py` - Fact extraction
- `jarvis/search/embeddings.py` - Semantic search

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
