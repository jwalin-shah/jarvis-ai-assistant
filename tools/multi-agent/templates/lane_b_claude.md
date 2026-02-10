# CLAUDE.md - Lane B: ML + Extraction + Classification

You are working in a **multi-agent worktree** for Lane B (ML + Extraction + Classification).

## Your Ownership

YOU OWN (can modify freely):
- `models/` - Model weights and configs
- `jarvis/classifiers/` - Classification pipeline
- `jarvis/extractors/` - Entity/fact extraction
- `jarvis/graph/` - Knowledge graph
- `scripts/train*` - Training scripts
- `scripts/extract*` - Extraction scripts

## Restrictions

YOU MUST NOT MODIFY (owned by other lanes):
- `desktop/` - Lane A (App)
- `api/` - Lane A (App)
- `jarvis/router.py` - Lane A (App)
- `jarvis/prompts.py` - Lane A (App)
- `jarvis/retrieval/` - Lane A (App)
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
