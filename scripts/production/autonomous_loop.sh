#!/usr/bin/env bash
# Autonomous multi-agent loop runner with optional cross-review.
#
# Runs any AI CLI agent (claude, gemini, codex, kimi, opencode) in a loop,
# tracking state across iterations via a status file. Optionally runs a
# reviewer agent after each work iteration to provide feedback.
#
# Usage:
#   # Simple: Claude fixes issues
#   ./scripts/autonomous_loop.sh \
#     --prompt "Fix all issues in tasks/performance-audit-2025-02-11.md" \
#     --end-condition "All H-priority issues resolved and make test passes"
#
#   # With review: Gemini works, Claude reviews
#   ./scripts/autonomous_loop.sh \
#     --agent gemini \
#     --reviewer claude \
#     --prompt "Refactor the search module" \
#     --end-condition "All functions <50 lines, tests pass"
#
#   # From prompt file with custom settings
#   ./scripts/autonomous_loop.sh \
#     --prompt-file tasks/my-prompt.md \
#     --end-condition "All done" \
#     --max-iterations 30 \
#     --agent codex \
#     --reviewer gemini
#
#   # Parallel overnight runs in isolated worktrees
#   ./scripts/autonomous_loop.sh \
#     --worktree perf-fixes --reviewer claude-haiku \
#     --prompt "Fix all perf issues" --end-condition "All queries <100ms"
#   ./scripts/autonomous_loop.sh \
#     --worktree refactor --reviewer claude-haiku \
#     --prompt "Refactor large functions" --end-condition "All functions <50 lines"
#
# Agents: claude (default), gemini, codex, kimi, opencode
# Reviewers: claude-haiku (default), claude-sonnet, claude, gemini, codex, kimi, opencode
#
# Control:
#   touch tasks/.stop-loop       Graceful stop after current iteration
#   cat tasks/loop-status.md     Check progress
#   tail -f tasks/sessions/*.log Follow live output

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ============================================================================
# Colors
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()  { echo -e "${CYAN}[loop]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[loop]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[loop]${NC} $1"; }
log_error() { echo -e "${RED}[loop]${NC} $1"; }
log_step()  { echo -e "${BOLD}[iter $1/$MAX_ITERATIONS]${NC} $2"; }

# ============================================================================
# Defaults
# ============================================================================
MAX_ITERATIONS=20
COOLDOWN=10
PROMPT=""
PROMPT_FILE=""
END_CONDITION=""
DRY_RUN=false
SESSION_NAME="loop-$(date +%Y%m%d-%H%M%S)"
STATUS_FILE="tasks/loop-status.md"
STOP_FILE="tasks/.stop-loop"
LOG_DIR="tasks/sessions"
WORK_TIMEOUT=3600    # 1 hour per work iteration
REVIEW_TIMEOUT=300   # 5 min per review

# Agent config
WORKER_AGENT="claude"
WORKER_MODEL=""            # empty = agent default, e.g. "opus", "sonnet", "haiku"
REVIEWER_AGENT=""          # empty = no review
REVIEWER_PROMPT_FILE=""    # custom review prompt template

# Worktree config
WORKTREE_NAME=""           # empty = run in current dir
WORKTREE_BASE=""           # empty = auto (sibling of repo root)
WORKTREE_BRANCH=""         # empty = same as WORKTREE_NAME

# Claude-specific
ALLOWED_TOOLS='Edit,Write,Read,Glob,Grep,Bash(make\ *),Bash(uv\ run\ *),Bash(git\ diff*),Bash(git\ status*),Bash(git\ log*),Bash(git\ add*),Bash(git\ commit*),Bash(cargo\ *),Bash(cd\ desktop\ &&\ pnpm\ *),Bash(ls\ *),Bash(wc\ *),Bash(mkdir\ *)'
MAX_TURNS=100

# ============================================================================
# Parse args
# ============================================================================
while [[ $# -gt 0 ]]; do
  case $1 in
    --prompt)              PROMPT="$2"; shift 2 ;;
    --prompt-file)         PROMPT_FILE="$2"; shift 2 ;;
    --end-condition)       END_CONDITION="$2"; shift 2 ;;
    --max-iterations)      MAX_ITERATIONS="$2"; shift 2 ;;
    --cooldown)            COOLDOWN="$2"; shift 2 ;;
    --session-name)        SESSION_NAME="$2"; shift 2 ;;
    --status-file)         STATUS_FILE="$2"; shift 2 ;;
    --agent)               WORKER_AGENT="$2"; shift 2 ;;
    --model)               WORKER_MODEL="$2"; shift 2 ;;
    --reviewer)            REVIEWER_AGENT="$2"; shift 2 ;;
    --reviewer-prompt)     REVIEWER_PROMPT_FILE="$2"; shift 2 ;;
    --worktree)            WORKTREE_NAME="$2"; shift 2 ;;
    --worktree-path)       WORKTREE_BASE="$2"; shift 2 ;;
    --worktree-branch)     WORKTREE_BRANCH="$2"; shift 2 ;;
    --allowed-tools)       ALLOWED_TOOLS="$2"; shift 2 ;;
    --max-turns)           MAX_TURNS="$2"; shift 2 ;;
    --work-timeout)        WORK_TIMEOUT="$2"; shift 2 ;;
    --review-timeout)      REVIEW_TIMEOUT="$2"; shift 2 ;;
    --dry-run)             DRY_RUN=true; shift ;;
    --help|-h)
      cat <<'HELP'
Usage: autonomous_loop.sh --prompt <text> --end-condition <text> [options]

Required:
  --prompt <text>           What the agent should work on each iteration
  --prompt-file <path>      Read prompt from file instead of --prompt
  --end-condition <text>    When to stop (agent checks this each iteration)

Agent Selection:
  --agent <name>            Worker agent: claude (default), gemini, codex, kimi, opencode
  --model <name>            Model for claude worker: opus, sonnet, haiku (default: your CLI default)
  --reviewer <name>         Reviewer agent (enables review step). Use a different
                            agent than --agent for cross-agent review.
                            Options: claude-haiku, claude-sonnet, claude, gemini,
                            codex, kimi, opencode
  --reviewer-prompt <path>  Custom review prompt template file

Worktree Isolation:
  --worktree <name>         Run in an isolated git worktree. Creates branch + worktree
                            automatically. Safe for parallel overnight runs.
  --worktree-path <dir>     Base directory for worktree (default: sibling of repo root)
  --worktree-branch <name>  Branch name (default: same as worktree name)

Loop Control:
  --max-iterations <n>      Max loop iterations (default: 20)
  --cooldown <secs>         Pause between iterations (default: 10)
  --work-timeout <secs>     Timeout per work iteration (default: 3600)
  --review-timeout <secs>   Timeout per review step (default: 300)

Claude-specific:
  --allowed-tools <list>    Comma-separated tool allowlist
  --max-turns <n>           Max agent turns per iteration (default: 100)

Other:
  --session-name <name>     Session identifier (default: loop-YYYYMMDD-HHMMSS)
  --status-file <path>      State file path (default: tasks/loop-status.md)
  --dry-run                 Print config without executing

Runtime Control:
  touch tasks/.stop-loop    Graceful stop after current iteration
  cat tasks/loop-status.md  Check progress
  tail -f tasks/sessions/loop-*.log  Follow live output

Examples:
  # Claude works alone
  autonomous_loop.sh --prompt "Fix the N+1 queries" --end-condition "All queries <100ms"

  # Gemini works, Claude reviews
  autonomous_loop.sh --agent gemini --reviewer claude-haiku \
    --prompt "Refactor search module" --end-condition "Tests pass, <50 LOC per fn"

  # Codex works, Gemini reviews, custom review criteria
  autonomous_loop.sh --agent codex --reviewer gemini \
    --reviewer-prompt tasks/review-criteria.md \
    --prompt "Add input validation" --end-condition "All endpoints validated"

  # Parallel overnight runs in isolated worktrees
  autonomous_loop.sh --worktree perf-fixes --reviewer claude-haiku \
    --prompt "Fix all H-priority perf issues" --end-condition "All queries <100ms"
  autonomous_loop.sh --worktree refactor --reviewer claude-haiku \
    --prompt "Refactor functions >50 lines" --end-condition "All functions <50 lines"
HELP
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ============================================================================
# Validate inputs
# ============================================================================
if [[ -z "$PROMPT" && -z "$PROMPT_FILE" ]]; then
  log_error "--prompt or --prompt-file is required"
  exit 1
fi

if [[ -z "$END_CONDITION" ]]; then
  log_error "--end-condition is required"
  exit 1
fi

if [[ -n "$PROMPT_FILE" ]]; then
  if [[ ! -f "$PROMPT_FILE" ]]; then
    log_error "Prompt file not found: $PROMPT_FILE"
    exit 1
  fi
  PROMPT=$(cat "$PROMPT_FILE")
fi

if [[ -n "$REVIEWER_PROMPT_FILE" && ! -f "$REVIEWER_PROMPT_FILE" ]]; then
  log_error "Reviewer prompt file not found: $REVIEWER_PROMPT_FILE"
  exit 1
fi

# ============================================================================
# Worktree setup
# ============================================================================
if [[ -n "$WORKTREE_NAME" ]]; then
  WORKTREE_BRANCH="${WORKTREE_BRANCH:-$WORKTREE_NAME}"

  if [[ -z "$WORKTREE_BASE" ]]; then
    REPO_BASENAME=$(basename "$REPO_ROOT")
    WORKTREE_BASE="$(dirname "$REPO_ROOT")/${REPO_BASENAME}-${WORKTREE_NAME}"
  fi

  log_info "Setting up worktree: $WORKTREE_BASE (branch: $WORKTREE_BRANCH)"

  # Create branch from current HEAD if it doesn't exist
  if ! git rev-parse --verify "$WORKTREE_BRANCH" &>/dev/null; then
    git branch "$WORKTREE_BRANCH" HEAD
    log_ok "Created branch: $WORKTREE_BRANCH"
  fi

  # Create worktree if it doesn't exist
  if [[ ! -d "$WORKTREE_BASE" ]]; then
    git worktree add "$WORKTREE_BASE" "$WORKTREE_BRANCH"
    log_ok "Created worktree: $WORKTREE_BASE"
  else
    log_info "Worktree already exists: $WORKTREE_BASE"
  fi

  # Copy prompt file into worktree if it's a relative path
  if [[ -n "$PROMPT_FILE" && ! "$PROMPT_FILE" = /* ]]; then
    PROMPT_FILE_ABS="${REPO_ROOT}/${PROMPT_FILE}"
    if [[ -f "$PROMPT_FILE_ABS" ]]; then
      PROMPT_FILE_DEST="${WORKTREE_BASE}/${PROMPT_FILE}"
      mkdir -p "$(dirname "$PROMPT_FILE_DEST")"
      /bin/cp -f "$PROMPT_FILE_ABS" "$PROMPT_FILE_DEST"
    fi
  fi

  # Copy reviewer prompt file into worktree if needed
  if [[ -n "$REVIEWER_PROMPT_FILE" && ! "$REVIEWER_PROMPT_FILE" = /* ]]; then
    REVIEWER_FILE_ABS="${REPO_ROOT}/${REVIEWER_PROMPT_FILE}"
    if [[ -f "$REVIEWER_FILE_ABS" ]]; then
      REVIEWER_FILE_DEST="${WORKTREE_BASE}/${REVIEWER_PROMPT_FILE}"
      mkdir -p "$(dirname "$REVIEWER_FILE_DEST")"
      /bin/cp -f "$REVIEWER_FILE_ABS" "$REVIEWER_FILE_DEST"
    fi
  fi

  # Switch into the worktree for the rest of execution
  REPO_ROOT="$WORKTREE_BASE"
  cd "$REPO_ROOT"
  log_ok "Working directory: $(pwd)"
fi

# ============================================================================
# Agent dispatch functions
# ============================================================================

# Run a worker agent with full write permissions
# Args: agent, prompt, log_file, timeout_secs
run_worker() {
  local agent=$1
  local prompt=$2
  local log_file=$3
  local timeout_secs=$4
  local exit_code=0

  case $agent in
    claude)
      local claude_args=(claude --print --max-turns "$MAX_TURNS" --allowedTools "$ALLOWED_TOOLS")
      if [[ -n "$WORKER_MODEL" ]]; then
        claude_args+=(--model "$WORKER_MODEL")
      fi
      claude_args+=(-p "$prompt")
      timeout --signal=TERM "$timeout_secs" "${claude_args[@]}" > "$log_file" 2>&1 || exit_code=$?
      ;;
    codex)
      timeout --signal=TERM "$timeout_secs" \
        codex exec --full-auto "$prompt" > "$log_file" 2>&1 || exit_code=$?
      ;;
    gemini)
      timeout --signal=TERM "$timeout_secs" \
        gemini -p "$prompt" --yolo > "$log_file" 2>&1 || exit_code=$?
      ;;
    opencode)
      timeout --signal=TERM "$timeout_secs" \
        opencode run "$prompt" > "$log_file" 2>&1 || exit_code=$?
      ;;
    kimi)
      timeout --signal=TERM "$timeout_secs" \
        kimi --quiet -p "$prompt" --yolo > "$log_file" 2>&1 || exit_code=$?
      ;;
    *)
      log_error "Unknown worker agent: $agent"
      return 1
      ;;
  esac

  case $exit_code in
    0)   echo "[worker:$agent] Completed successfully" >> "$log_file" ;;
    124) echo "[worker:$agent] TIMEOUT after ${timeout_secs}s" >> "$log_file" ;;
    137) echo "[worker:$agent] KILLED (SIGKILL)" >> "$log_file" ;;
    *)   echo "[worker:$agent] Exited with code $exit_code" >> "$log_file" ;;
  esac
  return $exit_code
}

# Run a reviewer agent in read-only mode (text output only)
# Args: agent, prompt, review_file, timeout_secs
run_reviewer() {
  local agent=$1
  local prompt=$2
  local review_file=$3
  local timeout_secs=$4

  case $agent in
    claude-haiku)
      timeout "$timeout_secs" claude -p "$prompt" --model haiku --print > "$review_file" 2>&1 || true
      ;;
    claude-sonnet)
      timeout "$timeout_secs" claude -p "$prompt" --model sonnet --print > "$review_file" 2>&1 || true
      ;;
    claude)
      timeout "$timeout_secs" claude -p "$prompt" --print > "$review_file" 2>&1 || true
      ;;
    codex)
      timeout "$timeout_secs" codex exec "$prompt" > "$review_file" 2>&1 || true
      ;;
    gemini)
      timeout "$timeout_secs" gemini -p "$prompt" > "$review_file" 2>&1 || true
      ;;
    opencode)
      timeout "$timeout_secs" opencode run "$prompt" > "$review_file" 2>&1 || true
      ;;
    kimi)
      timeout "$timeout_secs" kimi --quiet -p "$prompt" > "$review_file" 2>&1 || true
      ;;
    *)
      log_error "Unknown reviewer agent: $agent"
      return 1
      ;;
  esac
}

# Parse APPROVE/REJECT from review output
parse_verdict() {
  local review_file=$1
  if [[ ! -f "$review_file" ]] || [[ ! -s "$review_file" ]]; then
    echo "REJECT"
    return
  fi
  if grep -qiE "^APPROVE:" "$review_file" 2>/dev/null; then
    echo "APPROVE"
  elif grep -qiE "APPROVE:" "$review_file" 2>/dev/null; then
    echo "APPROVE"
  else
    echo "REJECT"
  fi
}

# Extract feedback text from review output (everything after the verdict line)
extract_feedback() {
  local review_file=$1
  if [[ ! -f "$review_file" ]] || [[ ! -s "$review_file" ]]; then
    echo "No review output"
    return
  fi
  cat "$review_file"
}

# ============================================================================
# Prompt builders
# ============================================================================

build_work_prompt() {
  local iteration=$1
  local review_feedback="${2:-}"

  cat <<PROMPT_EOF
You are running in an autonomous loop (iteration ${iteration}/${MAX_ITERATIONS}).

## State File
Read \`${STATUS_FILE}\` first. It tracks what's been done, what's in progress, and what's left.
If it doesn't exist yet, create it with an initial plan based on the task below.

## Your Task
${PROMPT}

## End Condition
${END_CONDITION}
PROMPT_EOF

  # Include reviewer feedback if available
  if [[ -n "$review_feedback" ]]; then
    cat <<FEEDBACK_EOF

## Reviewer Feedback (from previous iteration)
A reviewer agent evaluated your last iteration's work. Address their feedback:

${review_feedback}
FEEDBACK_EOF
  fi

  cat <<RULES_EOF

## Rules for This Iteration
1. Read the status file to understand current state
2. Pick ONE concrete next step (don't try to do everything)
3. Do the work. Run tests if you changed code (\`make test\`, then read \`test_results.txt\`)
4. Update the status file with what you did, what worked, what failed
5. If the end condition is met, write "## STATUS: DONE" at the top of the status file
6. If you're blocked on something, write "## STATUS: BLOCKED - <reason>" at the top
7. If there's more work to do, write "## STATUS: IN_PROGRESS" at the top

Keep changes surgical. One issue per iteration. Verify before marking done.
RULES_EOF
}

build_review_prompt() {
  local iteration=$1
  local worker_agent=$2
  local worker_log=$3

  # Use custom review prompt if provided
  local custom_criteria=""
  if [[ -n "$REVIEWER_PROMPT_FILE" ]]; then
    custom_criteria=$(cat "$REVIEWER_PROMPT_FILE")
  fi

  cat <<REVIEW_EOF
You are reviewing work done by the "${worker_agent}" agent (iteration ${iteration}).

## Context
- Status file: \`${STATUS_FILE}\`
- End condition: ${END_CONDITION}
- The worker just completed one iteration of work on this task

## What to Review
1. Read \`${STATUS_FILE}\` to see what the worker claims to have done
2. Check \`git diff\` to see actual code changes
3. If tests were run, check \`test_results.txt\` for results

## Review Criteria
- Is the change correct? Does it actually fix what it claims to fix?
- Is it surgical? No unnecessary changes or scope creep?
- Are there regressions? Did it break anything?
- Is the approach sound? Or is there a better way?
- Does the status file accurately reflect what was done?
${custom_criteria:+
## Additional Criteria
$custom_criteria
}

## Your Output Format
Start your response with exactly one of:
  APPROVE: <one-line summary of what's good>
  REJECT: <one-line summary of what needs fixing>

Then provide detailed feedback:
- What was done well
- What needs improvement
- Specific suggestions for the next iteration

Your feedback will be passed to the worker agent in the next iteration.
REVIEW_EOF
}

# ============================================================================
# Dry run
# ============================================================================
if [[ "$DRY_RUN" == true ]]; then
  echo -e "${BOLD}=== DRY RUN ===${NC}"
  echo ""
  echo "Session:         $SESSION_NAME"
  if [[ -n "$WORKTREE_NAME" ]]; then
  echo "Worktree:        $WORKTREE_BASE (branch: $WORKTREE_BRANCH)"
  fi
  echo "Worker agent:    $WORKER_AGENT"
  echo "Worker model:    ${WORKER_MODEL:-default}"
  echo "Reviewer agent:  ${REVIEWER_AGENT:-none}"
  echo "Max iterations:  $MAX_ITERATIONS"
  echo "Cooldown:        ${COOLDOWN}s"
  echo "Work timeout:    ${WORK_TIMEOUT}s"
  echo "Review timeout:  ${REVIEW_TIMEOUT}s"
  echo "Status file:     $STATUS_FILE"
  echo "Log dir:         $LOG_DIR"
  echo "Stop file:       $STOP_FILE"
  if [[ "$WORKER_AGENT" == "claude" ]]; then
    echo "Allowed tools:   $ALLOWED_TOOLS"
    echo "Max turns:       $MAX_TURNS"
  fi
  echo ""
  echo -e "${BOLD}--- Work Prompt (iteration 1) ---${NC}"
  build_work_prompt 1
  if [[ -n "$REVIEWER_AGENT" ]]; then
    echo ""
    echo -e "${BOLD}--- Review Prompt (iteration 1) ---${NC}"
    build_review_prompt 1 "$WORKER_AGENT" "tasks/sessions/work-iter-1.log"
  fi
  echo ""
  echo "Would run up to $MAX_ITERATIONS iterations."
  exit 0
fi

# ============================================================================
# Setup
# ============================================================================
mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$STATUS_FILE")"
rm -f "$STOP_FILE"

LOG_FILE="${LOG_DIR}/${SESSION_NAME}.log"
REVIEWS_DIR="${LOG_DIR}/${SESSION_NAME}-reviews"
if [[ -n "$REVIEWER_AGENT" ]]; then
  mkdir -p "$REVIEWS_DIR"
fi

# ============================================================================
# Banner
# ============================================================================
echo ""
echo -e "${BOLD}=== Autonomous Loop Started ===${NC}"
echo -e "  Session:    ${CYAN}$SESSION_NAME${NC}"
if [[ -n "$WORKTREE_NAME" ]]; then
echo -e "  Worktree:   ${BLUE}$WORKTREE_BASE${NC} (branch: $WORKTREE_BRANCH)"
fi
echo -e "  Worker:     ${GREEN}$WORKER_AGENT${NC} (${WORKER_MODEL:-default model})"
if [[ -n "$REVIEWER_AGENT" ]]; then
echo -e "  Reviewer:   ${BLUE}$REVIEWER_AGENT${NC}"
fi
echo -e "  Iterations: ${MAX_ITERATIONS} max"
echo -e "  Status:     $STATUS_FILE"
echo -e "  Log:        $LOG_FILE"
echo -e "  Stop with:  ${YELLOW}touch $STOP_FILE${NC}"
echo ""

{
  echo "=== Autonomous Loop: $SESSION_NAME ==="
  echo "Started: $(date)"
  echo "Worker: $WORKER_AGENT"
  echo "Reviewer: ${REVIEWER_AGENT:-none}"
  echo "Task: ${PROMPT:0:200}..."
  echo "End condition: $END_CONDITION"
  echo "Max iterations: $MAX_ITERATIONS"
  echo ""
} >> "$LOG_FILE"

# ============================================================================
# Main loop
# ============================================================================
LAST_REVIEW_FEEDBACK=""
STATS_WORK_OK=0
STATS_WORK_FAIL=0
STATS_REVIEWS=0
STATS_APPROVALS=0
STATS_REJECTIONS=0

for ((i = 1; i <= MAX_ITERATIONS; i++)); do
  # ── Pre-flight checks ────────────────────────────────────────────────────
  if [[ -f "$STOP_FILE" ]]; then
    log_warn "Stop signal detected. Exiting gracefully."
    echo "[iter $i] Stop signal detected at $(date)" >> "$LOG_FILE"
    rm -f "$STOP_FILE"
    break
  fi

  if [[ -f "$STATUS_FILE" ]] && head -5 "$STATUS_FILE" | grep -q "STATUS: DONE"; then
    log_ok "End condition met (STATUS: DONE). Exiting."
    echo "[iter $i] End condition met at $(date)" >> "$LOG_FILE"
    break
  fi

  if [[ -f "$STATUS_FILE" ]] && head -5 "$STATUS_FILE" | grep -q "STATUS: BLOCKED"; then
    log_error "Loop is BLOCKED. Check $STATUS_FILE for details."
    echo "[iter $i] BLOCKED at $(date)" >> "$LOG_FILE"
    break
  fi

  # ── Work phase ────────────────────────────────────────────────────────────
  WORK_LOG="${LOG_DIR}/${SESSION_NAME}-work-iter-${i}.log"

  log_step "$i" "Work phase (${WORKER_AGENT})..."
  echo "=== Work: Iteration $i - $(date) ===" >> "$LOG_FILE"

  WORK_PROMPT=$(build_work_prompt "$i" "$LAST_REVIEW_FEEDBACK")

  ITER_START=$(date +%s)
  set +e
  run_worker "$WORKER_AGENT" "$WORK_PROMPT" "$WORK_LOG" "$WORK_TIMEOUT"
  WORK_EXIT=$?
  set -e
  ITER_END=$(date +%s)
  WORK_DURATION=$((ITER_END - ITER_START))

  # Append work output to main log
  {
    echo "--- Work output (${WORK_DURATION}s, exit: $WORK_EXIT) ---"
    tail -50 "$WORK_LOG" 2>/dev/null || echo "(no output)"
    echo ""
  } >> "$LOG_FILE"

  if [[ $WORK_EXIT -eq 0 ]]; then
    log_step "$i" "Work completed in ${WORK_DURATION}s"
    ((STATS_WORK_OK++)) || true
  else
    log_step "$i" "Work exited with code $WORK_EXIT (${WORK_DURATION}s)"
    ((STATS_WORK_FAIL++)) || true
  fi

  # ── Review phase (optional) ──────────────────────────────────────────────
  LAST_REVIEW_FEEDBACK=""

  if [[ -n "$REVIEWER_AGENT" ]]; then
    # Re-check stop signal before review
    if [[ -f "$STOP_FILE" ]]; then
      log_warn "Stop signal detected before review. Exiting."
      rm -f "$STOP_FILE"
      break
    fi

    REVIEW_FILE="${REVIEWS_DIR}/review-iter-${i}.md"
    log_step "$i" "Review phase (${REVIEWER_AGENT})..."
    echo "--- Review: Iteration $i - $(date) ---" >> "$LOG_FILE"

    REVIEW_PROMPT=$(build_review_prompt "$i" "$WORKER_AGENT" "$WORK_LOG")

    REVIEW_START=$(date +%s)
    run_reviewer "$REVIEWER_AGENT" "$REVIEW_PROMPT" "$REVIEW_FILE" "$REVIEW_TIMEOUT"
    REVIEW_END=$(date +%s)
    REVIEW_DURATION=$((REVIEW_END - REVIEW_START))

    ((STATS_REVIEWS++)) || true

    # Parse verdict
    VERDICT=$(parse_verdict "$REVIEW_FILE")
    LAST_REVIEW_FEEDBACK=$(extract_feedback "$REVIEW_FILE")

    # Append review to main log
    {
      echo "--- Review verdict: $VERDICT (${REVIEW_DURATION}s) ---"
      head -20 "$REVIEW_FILE" 2>/dev/null || echo "(no review output)"
      echo ""
    } >> "$LOG_FILE"

    if [[ "$VERDICT" == "APPROVE" ]]; then
      log_step "$i" "Review: ${GREEN}APPROVED${NC} (${REVIEW_DURATION}s)"
      ((STATS_APPROVALS++)) || true
    else
      log_step "$i" "Review: ${YELLOW}REJECTED${NC} - feedback will be passed to next iteration (${REVIEW_DURATION}s)"
      ((STATS_REJECTIONS++)) || true
    fi

    # Append review summary to status file so worker can see it
    if [[ -f "$STATUS_FILE" ]]; then
      {
        echo ""
        echo "### Review (iteration $i) - $VERDICT"
        echo "Reviewer: $REVIEWER_AGENT"
        head -5 "$REVIEW_FILE" 2>/dev/null | sed 's/^/> /'
        echo ""
      } >> "$STATUS_FILE"
    fi
  fi

  # ── Cooldown ──────────────────────────────────────────────────────────────
  if [[ $i -lt $MAX_ITERATIONS ]]; then
    log_info "Cooling down ${COOLDOWN}s..."
    sleep "$COOLDOWN"
  fi
done

# ============================================================================
# Summary
# ============================================================================
echo "" >> "$LOG_FILE"
echo "=== Loop Complete ===" >> "$LOG_FILE"
echo "Completed: $(date)" >> "$LOG_FILE"

echo ""
echo -e "${BOLD}=== Loop Complete ===${NC}"
echo -e "  Iterations:   $((i < MAX_ITERATIONS ? i : MAX_ITERATIONS))"
echo -e "  Work OK:      $STATS_WORK_OK"
echo -e "  Work Failed:  $STATS_WORK_FAIL"
if [[ -n "$REVIEWER_AGENT" ]]; then
echo -e "  Reviews:      $STATS_REVIEWS"
echo -e "  Approvals:    ${GREEN}$STATS_APPROVALS${NC}"
echo -e "  Rejections:   ${YELLOW}$STATS_REJECTIONS${NC}"
fi
echo -e "  Status:       $STATUS_FILE"
echo -e "  Log:          $LOG_FILE"
if [[ -n "$REVIEWER_AGENT" ]]; then
echo -e "  Reviews dir:  $REVIEWS_DIR"
fi

# Final status
if [[ -f "$STATUS_FILE" ]] && head -5 "$STATUS_FILE" | grep -q "STATUS: DONE"; then
  echo ""
  log_ok "Task completed successfully!"
elif [[ -f "$STATUS_FILE" ]] && head -5 "$STATUS_FILE" | grep -q "STATUS: BLOCKED"; then
  echo ""
  log_error "Task blocked. Review $STATUS_FILE for details."
else
  echo ""
  log_warn "Loop exhausted iterations without completing. Review $STATUS_FILE."
fi

# Worktree merge instructions
if [[ -n "$WORKTREE_NAME" ]]; then
  echo ""
  echo -e "${BOLD}=== Worktree Merge Instructions ===${NC}"
  echo -e "  Review changes:  ${CYAN}cd $WORKTREE_BASE && git log --oneline main..$WORKTREE_BRANCH${NC}"
  echo -e "  Diff from main:  ${CYAN}git diff main...$WORKTREE_BRANCH${NC}"
  echo -e "  Merge to main:   ${CYAN}cd $REPO_ROOT && git checkout main && git merge $WORKTREE_BRANCH${NC}"
  echo -e "  Cleanup:         ${CYAN}git worktree remove $WORKTREE_BASE && git branch -d $WORKTREE_BRANCH${NC}"
fi

# macOS notification
if command -v osascript &>/dev/null; then
  osascript -e "display notification \"Loop $SESSION_NAME finished\" with title \"Autonomous Loop\"" 2>/dev/null || true
fi
