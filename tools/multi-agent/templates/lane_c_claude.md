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
- `models/` - Lane B (ML)
- `jarvis/classifiers/` - Lane B (ML)
- `jarvis/extractors/` - Lane B (ML)
- `jarvis/graph/` - Lane B (ML)
- `scripts/train*` - Lane B (ML)
- `scripts/extract*` - Lane B (ML)

SHARED (requires all-lane approval):
- `jarvis/contracts/pipeline.py` - Modify only if necessary, document changes

## Allowed Exceptions

- Test-only fixtures and mocks that reference other lanes' code (read-only usage)

## Completion Protocol

When your work is done:
1. Commit all changes to the current branch
2. Create a file called `.agent-done` in the worktree root: `touch .agent-done`
3. Both steps are required for the hub to detect completion

## Task

Your task is defined in `.hub-task.md` in this worktree root.
