# Jarvis Parallel Workstreams

This document defines how to split work across multiple agents so frontend, ML, and testing can move in parallel without collisions.

## 1) Goals

- Keep `main` stable while major refactor work is in progress.
- Enable parallel delivery across app orchestration, ML pipeline, and QA gates.
- Reduce integration risk by enforcing one shared contract and strict ownership.

## 2) Single Source of Truth (Contracts)

All lanes depend on one file:

- `jarvis/contracts/pipeline.py`

Rules:

- Any signature/type change here requires explicit cross-lane approval.
- Lane owners must not consume undocumented fields.
- Contract version must be bumped on breaking changes.

Recommended contract versioning:

- Add `PIPELINE_CONTRACT_VERSION = "v1"` in `jarvis/contracts/pipeline.py`.
- Use additive changes when possible (new optional fields first).

## 3) Lane Ownership

Each lane owns specific folders and is responsible for code quality within them.

### Lane A: App + Orchestration (Codex + Kimi)

Primary goal: make the end-to-end pipeline run reliably and keep user-facing behavior stable.

Owned paths:

- `desktop/`
- `api/`
- `jarvis/router.py` (or replacement `jarvis/pipeline.py`)
- `jarvis/prompts.py`
- `jarvis/retrieval/` (or current retrieval implementation path)

Allowed changes outside ownership:

- `jarvis/contracts/pipeline.py` only with approval and compatibility note.

Deliverables:

- Pipeline orchestration (`process_message`) calling extract -> classify -> retrieve -> generate.
- Prompt assembly using unified request objects.
- Feature flags for old/new path rollout.

### Lane B: ML + Extraction + Classification (Claude)

Primary goal: produce consistent structured outputs that satisfy pipeline contracts.

Owned paths:

- `models/`
- `jarvis/classifiers/`
- `jarvis/extractors/`
- `jarvis/graph/`
- `scripts/` for training/extraction utilities

Allowed changes outside ownership:

- `jarvis/contracts/pipeline.py` only with approval and migration plan.

Deliverables:

- `UnifiedClassifier.classify()` returning `ClassificationResult`.
- Entity/fact/relationship extraction returning `ExtractionResult`.
- Training artifacts + reproducible training command logs.

### Lane C: Quality + Regression Gates (Gemini)

Primary goal: prevent regressions and provide objective merge decisions.

Owned paths:

- `tests/`
- `benchmarks/`
- `evals/`
- CI and verification scripts in `scripts/`

Allowed changes outside ownership:

- Test-only fixtures/mocks where required.

Deliverables:

- Contract tests for all stage boundaries.
- Regression suite for reply quality, latency, and error rates.
- Pass/fail reports attached to every integration PR.

## 4) PR and Merge Policy

### Branches

- `main` (protected)
- `codex/pipeline-contracts`
- `codex/app-orchestration`
- `codex/ml-unified-classifier`
- `codex/qa-regression-gates`

### Merge order

1. Contract PR (`codex/pipeline-contracts`)
2. Lane PRs in parallel (`app-orchestration`, `ml-unified-classifier`, `qa-regression-gates`)
3. Integration PR that enables the full new pipeline path

### Required PR template sections

- Scope (what changed, what did not)
- Contract impact (none/additive/breaking)
- Tests added/updated
- Benchmark/eval impact
- Rollback plan

### Required checks before merge

- `make check`
- `make test-fast`
- Lane C regression gate status is green

## 5) Handoff Contracts Between Lanes

### B -> A handoff

- Stable `ClassificationResult` and `ExtractionResult` schemas
- Known confidence semantics
- Example fixtures for expected outputs

### A -> C handoff

- End-to-end scenarios and acceptance criteria
- Feature flag behavior for new vs legacy path

### C -> all lanes handoff

- Failing test report with minimal repro
- Regression delta (what got worse, by how much)

## 6) Conflict-Prevention Rules

- One lane per file owner by default.
- Cross-lane edits require issue comment + owner acknowledgement before merge.
- No direct commits to `main`.
- Rebase daily from `main`.
- If contract changes are proposed, freeze merges until compatibility is confirmed.

## 7) 7-Day Execution Schedule

### Day 1: Contract freeze

- Finalize dataclasses and enum values in `jarvis/contracts/pipeline.py`.
- Add contract tests in `tests/` for serialization and required fields.

### Day 2: Lane scaffolding

- Lane A builds orchestrator skeleton with stubs.
- Lane B scaffolds unified classifier/extractor interfaces.
- Lane C scaffolds regression harness and baseline snapshots.

### Day 3: First functional pass

- Lane B returns real classification/extraction outputs.
- Lane A wires outputs to prompt builder/generator request.
- Lane C runs baseline-vs-new dry comparison.

### Day 4: Reliability hardening

- Fix schema mismatches and edge cases.
- Add fallback behavior when extraction/classification fails.
- Expand regression scenarios.

### Day 5: Performance and memory pass

- Validate batching and memory constraints (8GB target).
- Remove per-message heavyweight loads.
- Confirm no obvious latency regressions in hot path.

### Day 6: Integration PR

- Merge lane branches into integration branch.
- Enable feature flag in shadow mode.
- Run full verification (`make verify`) and benchmark gates.

### Day 7: Cutover

- Enable new pipeline path by default.
- Keep fallback switch for one release window.
- Publish post-cutover report: quality, latency, errors, rollback status.

## 8) Definition of Done

A workstream is done only when:

- Contract-compatible code is merged.
- Tests and regression gates are green.
- Benchmarks are non-regressive or regression is explicitly accepted.
- Operational fallback exists and is documented.

## 9) Hub-Spoke Orchestration

Use `tools/multi-agent/hub.sh` to manage the multi-agent workflow:

```bash
# Set up isolated worktrees for each lane
./tools/multi-agent/hub.sh setup

# Dispatch tasks to lane agents (parses ## Lane A/B/C sections)
./tools/multi-agent/hub.sh dispatch tasks/my-task.md

# Monitor agent progress
./tools/multi-agent/hub.sh status

# Cross-review completed lanes (ownership enforcement + agent review)
./tools/multi-agent/hub.sh review

# Re-dispatch a lane with rejection feedback
./tools/multi-agent/hub.sh rework <lane>

# Merge all approved lanes to main (runs make verify)
./tools/multi-agent/hub.sh merge

# Clean up worktrees and state
./tools/multi-agent/hub.sh teardown
```

Task files use `## Lane A` / `## Lane B` / `## Lane C` sections. See `tools/multi-agent/templates/task_template.md` for the format.

## 10) Immediate Next Steps

1. Run `hub.sh setup` to create lane worktrees.
2. Write a task file with lane-specific instructions.
3. Run `hub.sh dispatch <task-file>` to start parallel work.
4. Use `hub.sh status` and `hub.sh review` to coordinate.
