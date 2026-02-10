# Multi-Agent Hub Orchestration Guide

A hub-spoke system for coordinating multiple AI agents in isolated git worktrees with cross-review and merge gating.

## Quick Start

```bash
# 1. Setup worktrees for 3 lanes
bash tools/multi-agent/hub.sh setup

# 2. Create a task file
cat > tasks/refactor.md << 'EOF'
## Lane A
Refactor jarvis/router.py to use async patterns

## Lane B
IDLE

## Lane C
Add tests for the new async router
EOF

# 3. Dispatch tasks to lanes
bash tools/multi-agent/hub.sh dispatch tasks/refactor.md

# 4. Watch progress (auto-refreshes, stops when done)
bash tools/multi-agent/hub.sh watch

# 5. Review cross-lane approvals (auto-triggered by default)
bash tools/multi-agent/hub.sh review

# 6. Merge approved lanes (runs make verify)
bash tools/multi-agent/hub.sh merge

# 7. Cleanup
bash tools/multi-agent/hub.sh teardown
```

## Lane Architecture

| Lane | Default Agent | Branch | Ownership |
|------|---------------|--------|-----------|
| A | codex | lane-a/app | `desktop/`, `api/`, `jarvis/router.py`, `jarvis/prompts.py`, `jarvis/retrieval/` |
| B | claude | lane-b/ml | `models/`, `jarvis/classifiers/`, `jarvis/extractors/`, `jarvis/graph/`, `scripts/train`, `scripts/extract` |
| C | gemini | lane-c/qa | `tests/`, `benchmarks/`, `evals/` |

Shared paths (require cross-lane approval): `jarvis/contracts/`

## Standalone Tasks

Run one-off tasks outside the lane system:

```bash
# Basic usage
bash tools/multi-agent/hub.sh run kimi "Generate test fixtures"

# With model override
bash tools/multi-agent/hub.sh run codex -m o3 "Deep audit of jarvis/db/"

# With label for tracking
bash tools/multi-agent/hub.sh run claude -l "docs-update" "Update README"
```

Supported agents: `claude`, `codex`, `gemini`, `kimi`, `opencode`

## Monitoring

```bash
# Show all lanes + standalone tasks
bash tools/multi-agent/hub.sh status

# Auto-refresh status every 5 seconds
bash tools/multi-agent/hub.sh watch

# Custom refresh interval (seconds)
bash tools/multi-agent/hub.sh watch 10

# View lane logs
bash tools/multi-agent/hub.sh logs a          # Lane A
bash tools/multi-agent/hub.sh logs a -f       # Follow/tail mode

# View standalone task logs
bash tools/multi-agent/hub.sh logs 3          # Task #3

# Session summary (stats, agent usage, events)
bash tools/multi-agent/hub.sh summary
```

Status colors: `idle` → `working` → `done` → `reviewing` → `approved` → `merged`

## Configuring Lane Agents

Override default agents per lane via environment variables:

```bash
# Use Claude for Lane A (App)
LANE_A_AGENT=claude bash tools/multi-agent/hub.sh setup

# Use Kimi for Lane B (ML)
LANE_B_AGENT=kimi bash tools/multi-agent/hub.sh dispatch tasks/ml.md

# Use OpenCode for Lane C (QA)
LANE_C_AGENT=opencode bash tools/multi-agent/hub.sh setup
```

## Auto-Review and Auto-Rework

The system automatically reviews completed lanes and reworks rejected ones:

| Variable | Default | Description |
|----------|---------|-------------|
| `HUB_AUTO_REVIEW` | `1` | Auto-trigger review when lane completes |
| `HUB_AUTO_REWORK` | `1` | Auto-rework rejected lanes with feedback |
| `HUB_MAX_RETRIES` | `3` | Max rework attempts before manual intervention |

Disable for manual control:
```bash
HUB_AUTO_REVIEW=0 HUB_AUTO_REWORK=0 bash tools/multi-agent/hub.sh dispatch tasks/big.md
```

### Manual Rework

If auto-rework is disabled or max retries reached:

```bash
# Check rejection feedback
cat .hub/reviews/a_feedback.md

# Rework a specific lane
bash tools/multi-agent/hub.sh rework a
```

## Task File Format

```markdown
## Lane A
Specific task for Lane A. Use markdown.
- Bullet points
- Code examples

## Lane B
IDLE

## Lane C
Another task here.

## Shared Notes (optional)
Any content after all 3 lanes is ignored by the parser.
```

Use `IDLE` (case-insensitive) to skip a lane.

## Common Workflows

### Parallel Feature Development
```bash
bash tools/multi-agent/hub.sh setup
bash tools/multi-agent/hub.sh dispatch tasks/feature-x.md
bash tools/multi-agent/hub.sh watch        # Wait for completion
bash tools/multi-agent/hub.sh merge
```

### Fire-and-Forget Standalone Tasks
```bash
# Run multiple audits in parallel
bash tools/multi-agent/hub.sh run codex "Audit SQL queries"
bash tools/multi-agent/hub.sh run kimi "Generate test plan"
bash tools/multi-agent/hub.sh run gemini "Review docs"
bash tools/multi-agent/hub.sh watch        # Monitor all tasks
```

### Mixed Workflow (Lanes + Standalone)
```bash
bash tools/multi-agent/hub.sh setup
bash tools/multi-agent/hub.sh dispatch tasks/refactor.md
bash tools/multi-agent/hub.sh run codex -m o3 "Security audit" &
bash tools/multi-agent/hub.sh watch
```

## Tips

1. **Worktree locations**: Lanes work in sibling directories (`../jarvis-lane-a`, etc.), not subdirectories
2. **Ownership violations**: Auto-rejected during review. Check `CLAUDE.md` in each worktree for ownership rules
3. **Lane completion**: Agents signal completion by creating `.agent-done` file
4. **Merge order**: Contracts first, then B (ML), A (App), C (QA)
5. **Notifications**: macOS notifications on lane/task completion
6. **Logs**: All output saved to `.hub/logs/` with timestamps
7. **State**: Stored in `.hub/state.json` (JSON, human-readable)

## Teardown Options

```bash
# Clean removal (stashes uncommitted changes if prompted)
bash tools/multi-agent/hub.sh teardown

# Force cleanup if state is corrupted
rm -rf .hub ../jarvis-lane-*
git worktree prune
```
