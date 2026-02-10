# CLAUDE.md - Lane A: App + Orchestration

You are working in a **multi-agent worktree** for Lane A (App + Orchestration).

## Your Ownership

YOU OWN (can modify freely):
- `desktop/` - Tauri desktop app (Rust + Svelte)
- `api/` - API layer
- `jarvis/router.py` - Message routing
- `jarvis/prompts.py` - Prompt templates
- `jarvis/retrieval/` - RAG retrieval

## Restrictions

YOU MUST NOT MODIFY (owned by other lanes):
- `jarvis/classifiers/` - Lane B (ML)
- `jarvis/extractors/` - Lane B (ML)
- `jarvis/graph/` - Lane B (ML)
- `models/` - Lane B (ML)
- `scripts/train*` - Lane B (ML)
- `scripts/extract*` - Lane B (ML)
- `tests/` - Lane C (QA)
- `benchmarks/` - Lane C (QA)
- `evals/` - Lane C (QA)

SHARED (requires all-lane approval):
- `jarvis/contracts/pipeline.py` - Modify only if necessary, document changes

## Completion Protocol

When your work is done:
1. Commit all changes to the current branch
2. Create a file called `.agent-done` in the worktree root: `touch .agent-done`
3. Both steps are required for the hub to detect completion

## Task

Your task is defined in `.hub-task.md` in this worktree root.
