#!/usr/bin/env bash
# Launch extraction optimization loop in background with nohup+caffeinate
set -euo pipefail

cd "$(dirname "$0")/.."

LOG="tasks/sessions/extraction-opt-nohup.log"
mkdir -p tasks/sessions

caffeinate -i nohup /bin/bash scripts/autonomous_loop.sh \
  --prompt-file tasks/prompts/extraction-optimizer.md \
  --end-condition "F1 >= 0.90 on full goldset OR 50 iterations completed" \
  --agent claude \
  --model opus \
  --reviewer gemini \
  --reviewer-prompt tasks/prompts/extraction-reviewer.md \
  --max-iterations 50 \
  --max-turns 100 \
  --cooldown 15 \
  --work-timeout 3600 \
  --review-timeout 600 \
  --session-name extraction-opt \
  --status-file tasks/extraction-opt-status.md \
  --worktree extraction-opt \
  --allowed-tools 'Edit,Write,Read,Glob,Grep,Bash(uv\ run\ *),Bash(git\ diff*),Bash(git\ status*),Bash(git\ log*),Bash(git\ add*),Bash(git\ commit*),Bash(ls\ *),Bash(wc\ *),Bash(mkdir\ *),Bash(cat\ *),Bash(head\ *),Bash(tail\ *),WebSearch,WebFetch' \
  > "$LOG" 2>&1 &

PID=$!
echo "Extraction optimization loop launched!"
echo "  PID: $PID"
echo "  Log: $LOG"
echo ""
echo "Monitor:"
echo "  tail -f $LOG"
echo "  cat ../jarvis-ai-assistant-extraction-opt/tasks/extraction-opt-status.md"
echo ""
echo "Stop:"
echo "  touch ../jarvis-ai-assistant-extraction-opt/tasks/.stop-loop"
echo ""
echo "Tomorrow - review and merge:"
echo "  cd ../jarvis-ai-assistant-extraction-opt && git log --oneline main..extraction-opt"
echo "  cd ~/projects/jarvis-ai-assistant && git merge extraction-opt"
echo "  git worktree remove ../jarvis-ai-assistant-extraction-opt"
echo "  git branch -d extraction-opt"
