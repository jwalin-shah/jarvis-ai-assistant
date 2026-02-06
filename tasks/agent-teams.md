# Agent Teams Tracking

## Active Worktrees

| Branch | CLI | Path | Status | Task |
|--------|-----|------|--------|------|
| main | claude | `~/coding/jarvis-ai-assistant` | coordination | Management hub |

## How to Use

### Create a worktree for a CLI agent
```bash
./scripts/manage_worktrees.sh create feature/<name> <cli>
# cli options: claude, gemini, kimi, opencode, codex, agent
```

### Run multi-agent orchestration
```bash
# Parallel: same prompt, all agents, compare outputs
./tools/multi-agent/orchestrator.sh "your prompt here"

# Debate: agents critique each other over rounds
./tools/multi-agent/orchestrator.sh -m debate -r 3 "your prompt"

# Relay: chain agents, each builds on previous
./tools/multi-agent/orchestrator.sh -m relay -o claude,codex,gemini "your prompt"

# Pick specific agents
./tools/multi-agent/orchestrator.sh -a claude,gemini "your prompt"
```

### Check status
```bash
./scripts/manage_worktrees.sh list
```

### Merge completed work
```bash
cd ~/coding/jarvis-ai-assistant
git fetch origin
git merge origin/feature/<branch>
make verify
```

## Task Assignments

### Pending
_No tasks assigned yet._

### In Progress
_None._

### Completed
_None._

## Session Log

### $(date +%Y-%m-%d)
- Enabled `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`
- Set up agent teams tracking doc
- Available CLIs: claude, gemini, codex, opencode, kimi, agent (Cursor)
