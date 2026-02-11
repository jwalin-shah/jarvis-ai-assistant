#!/usr/bin/env bash
# hub_lib.sh - Helper functions for hub.sh multi-agent orchestration
# Provides: state management, logging, ownership checking, task parsing

# Prevent double-sourcing
[[ -n "${_HUB_LIB_LOADED:-}" ]] && return 0
_HUB_LIB_LOADED=1

# ── Paths ──────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
HUB_DIR="$REPO_ROOT/.hub"
STATE_FILE="$HUB_DIR/state.json"
REVIEWS_DIR="$HUB_DIR/reviews"
LOGS_DIR="$HUB_DIR/logs"
REVIEW_TIMEOUT="${REVIEW_TIMEOUT:-300}"
REVIEW_AGENT="${REVIEW_AGENT:-claude-haiku}"  # Agent for cross-lane reviews (cheap/fast)
PREDIFF="$SCRIPT_DIR/prediff.py"
CONTRACTS_FILE="$REPO_ROOT/jarvis/contracts/pipeline.py"

# ── Colors ─────────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Lane Definitions ──────────────────────────────────────────────────────────

# Default agent assignments (override via env: LANE_A_AGENT=kimi hub.sh dispatch ...)
declare -A LANE_AGENTS=(
    [a]="${LANE_A_AGENT:-codex}"
    [b]="${LANE_B_AGENT:-claude}"
    [c]="${LANE_C_AGENT:-gemini}"
)

declare -A LANE_BRANCHES=(
    [a]="lane-a/app"
    [b]="lane-b/ml"
    [c]="lane-c/qa"
)

declare -A LANE_WORKTREES=(
    [a]="jarvis-lane-a"
    [b]="jarvis-lane-b"
    [c]="jarvis-lane-c"
)

declare -A LANE_LABELS=(
    [a]="App + Orchestration"
    [b]="ML + Extraction"
    [c]="Quality + Regression"
)

# Ownership paths per lane (space-separated)
declare -A LANE_OWNED_PATHS=(
    [a]="desktop/ api/ jarvis/router.py jarvis/prompts.py jarvis/retrieval/ jarvis/reply_service.py"
    [b]="models/ jarvis/classifiers/ jarvis/extractors/ jarvis/contacts/ jarvis/graph/ jarvis/search/ scripts/train scripts/extract"
    [c]="tests/ benchmarks/ evals/"
)

SHARED_PATHS="jarvis/contracts/"

ALL_LANES="a b c"

# ── Logging ────────────────────────────────────────────────────────────────────

hub_log() {
    echo -e "${CYAN}[hub]${NC} $1"
}

hub_warn() {
    echo -e "${YELLOW}[hub]${NC} $1"
}

hub_error() {
    echo -e "${RED}[hub]${NC} $1" >&2
}

hub_success() {
    echo -e "${GREEN}[hub]${NC} $1"
}

get_lane_agent() {
    # Read agent from state.json (persisted during setup), fall back to env/default
    local lane=$1
    if [[ -f "$STATE_FILE" ]]; then
        local saved
        saved=$(python3 -c "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
print(state['lanes']['$lane']['agent'])
" 2>/dev/null)
        if [[ -n "$saved" && "$saved" != "None" && "$saved" != "null" ]]; then
            echo "$saved"
            return
        fi
    fi
    echo "${LANE_AGENTS[$lane]}"
}

lane_log() {
    local lane=$1
    local msg=$2
    local agent
    agent=$(get_lane_agent "$lane")
    echo -e "${BLUE}[lane-${lane}/${agent}]${NC} $msg"
}

# ── State Locking ─────────────────────────────────────────────────────────────

_state_lock() {
    local timeout="${1:-30}"
    mkdir -p "$HUB_DIR/locks"
    local start=$SECONDS
    while ! mkdir "$HUB_DIR/locks/state" 2>/dev/null; do
        if (( SECONDS - start > timeout )); then
            hub_error "Timeout waiting for state lock"
            return 1
        fi
        sleep 0.1
    done
}

_state_unlock() {
    rmdir "$HUB_DIR/locks/state" 2>/dev/null || true
}

# Atomic state update via Python with file locking
update_state() {
    local python_code="$1"
    _state_lock || return 1
    python3 -c "$python_code"
    local rc=$?
    _state_unlock
    return $rc
}

# ── State Management ──────────────────────────────────────────────────────────

init_state() {
    mkdir -p "$HUB_DIR" "$REVIEWS_DIR" "$LOGS_DIR"

    # If state already exists, preserve it (don't overwrite running tasks)
    if [[ -f "$STATE_FILE" ]]; then
        # Just ensure lanes exist
        python3 -c "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
for lane in ['a', 'b', 'c']:
    if lane not in state.get('lanes', {}):
        state.setdefault('lanes', {})[lane] = {'status': 'idle', 'agent': '', 'pid': None, 'last_commit': None}
if 'tasks' not in state:
    state['tasks'] = []
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=4)
"
        return
    fi

    local ts
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    cat > "$STATE_FILE" << STATEEOF
{
    "lanes": {
        "a": {"status": "idle", "agent": "${LANE_A_AGENT:-codex}", "pid": null, "last_commit": null, "started_at": null, "finished_at": null, "retries": 0},
        "b": {"status": "idle", "agent": "${LANE_B_AGENT:-claude}", "pid": null, "last_commit": null, "started_at": null, "finished_at": null, "retries": 0},
        "c": {"status": "idle", "agent": "${LANE_C_AGENT:-gemini}", "pid": null, "last_commit": null, "started_at": null, "finished_at": null, "retries": 0}
    },
    "tasks": [],
    "reviews": {},
    "created_at": "$ts"
}
STATEEOF
}

get_lane_status() {
    local lane=$1
    python3 -c "
import json, sys
with open('$STATE_FILE') as f:
    state = json.load(f)
print(state['lanes']['$lane']['status'])
" 2>/dev/null || echo "unknown"
}

set_lane_status() {
    local lane=$1
    local status=$2
    update_state "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
state['lanes']['$lane']['status'] = '$status'
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=4)
"
}

set_lane_pid() {
    local lane=$1
    local pid=$2
    update_state "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
state['lanes']['$lane']['pid'] = $pid
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=4)
"
}

set_lane_commit() {
    local lane=$1
    local commit=$2
    update_state "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
state['lanes']['$lane']['last_commit'] = '$commit'
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=4)
"
}

# ── Standalone Task Tracking ──────────────────────────────────────────────────

# Add a standalone task to the state file. Returns the task ID.
add_task() {
    local agent=$1
    local description=$2
    local pid=$3
    local log_file=$4
    local model=${5:-"default"}

    _state_lock || return 1
    python3 -c "
import json, time
with open('$STATE_FILE') as f:
    state = json.load(f)
tasks = state.setdefault('tasks', [])
task_id = len(tasks) + 1
tasks.append({
    'id': task_id,
    'agent': '$agent',
    'model': '$model',
    'description': '''$description''',
    'pid': $pid,
    'log': '$log_file',
    'status': 'working',
    'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
})
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=4)
print(task_id)
"
    local rc=$?
    _state_unlock
    return $rc
}

# Update a standalone task's status
set_task_status() {
    local task_id=$1
    local status=$2
    update_state "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
for t in state.get('tasks', []):
    if t['id'] == $task_id:
        t['status'] = '$status'
        break
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=4)
"
}

# ── Notifications ────────────────────────────────────────────────────────────

# Send macOS notification
notify() {
    local title=$1
    local message=$2
    local sound=${3:-"Glass"}

    # macOS native notification
    osascript -e "display notification \"$message\" with title \"$title\" sound name \"$sound\"" 2>/dev/null &

    # Also log to events
    echo "[$(date '+%H:%M:%S')] NOTIFY: $title - $message" >> "$LOGS_DIR/hub_events.log" 2>&1
}

# ── Log Summary Extraction ───────────────────────────────────────────────────

# Extract a short summary from an agent's log file
extract_log_summary() {
    local log_file=$1
    local max_lines=${2:-5}

    if [[ ! -f "$log_file" ]] || [[ ! -s "$log_file" ]]; then
        echo "(empty log)"
        return
    fi

    # Get last meaningful lines, skip blank lines
    tail -20 "$log_file" 2>/dev/null | grep -v '^$' | tail -"$max_lines"
}

# Extract token/cost info from agent logs (best-effort parsing)
extract_token_usage() {
    local log_file=$1

    if [[ ! -f "$log_file" ]]; then
        echo "tokens: unknown"
        return
    fi

    # Try to find token counts in various formats
    local tokens
    tokens=$(grep -iE "tokens?[: ]+[0-9]|total.*(tokens|cost)|usage" "$log_file" 2>/dev/null | tail -3)
    if [[ -n "$tokens" ]]; then
        echo "$tokens"
    else
        # Fallback: log file size as rough proxy
        local size
        size=$(wc -c < "$log_file" 2>/dev/null | tr -d ' ')
        echo "log size: ${size} bytes (token count unavailable)"
    fi
}

# ── Duration Formatting ──────────────────────────────────────────────────────

format_duration() {
    local started=$1
    local ended=${2:-"now"}

    python3 -c "
from datetime import datetime, timezone
started = datetime.fromisoformat('$started'.replace('Z', '+00:00'))
if '$ended' == 'now':
    ended = datetime.now(timezone.utc)
else:
    ended = datetime.fromisoformat('$ended'.replace('Z', '+00:00'))
delta = ended - started
secs = int(delta.total_seconds())
if secs < 60:
    print(f'{secs}s')
elif secs < 3600:
    print(f'{secs // 60}m {secs % 60}s')
else:
    print(f'{secs // 3600}h {(secs % 3600) // 60}m')
" 2>/dev/null || echo "?"
}

set_review_result() {
    local source_lane=$1
    local reviewer_lane=$2
    local result=$3  # "approve" or "reject"
    local reason=$4
    update_state "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
key = '${source_lane}_by_${reviewer_lane}'
if 'reviews' not in state:
    state['reviews'] = {}
state['reviews'][key] = {'result': '$result', 'reason': '''$reason'''}
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=4)
"
}

# ── Worktree Helpers ──────────────────────────────────────────────────────────

get_worktree_path() {
    local lane=$1
    local parent
    parent="$(dirname "$REPO_ROOT")"
    echo "$parent/${LANE_WORKTREES[$lane]}"
}

worktree_exists() {
    local lane=$1
    local wt_path
    wt_path=$(get_worktree_path "$lane")
    [[ -d "$wt_path" ]]
}

worktree_has_uncommitted() {
    local lane=$1
    local wt_path
    wt_path=$(get_worktree_path "$lane")
    [[ -n "$(git -C "$wt_path" status --porcelain 2>/dev/null)" ]]
}

worktree_last_commit() {
    local lane=$1
    local wt_path
    wt_path=$(get_worktree_path "$lane")
    git -C "$wt_path" log -1 --format="%h %s" 2>/dev/null || echo "none"
}

agent_done_exists() {
    local lane=$1
    local wt_path
    wt_path=$(get_worktree_path "$lane")
    [[ -f "$wt_path/.agent-done" ]]
}

# ── Auto-Update on Exit ──────────────────────────────────────────────────────

# Called after a lane agent exits to auto-update state and trigger review
on_lane_exit() {
    local lane=$1
    local wt_path
    wt_path=$(get_worktree_path "$lane")

    # Check if agent created .agent-done
    if [[ -f "$wt_path/.agent-done" ]]; then
        local commits
        commits=$(git -C "$wt_path" log main..HEAD --oneline 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$commits" -gt 0 ]]; then
            set_lane_status "$lane" "done"
            local last
            last=$(git -C "$wt_path" log -1 --format="%h" 2>/dev/null)
            set_lane_commit "$lane" "$last"
            echo "[$(date '+%H:%M:%S')] Lane ${lane^^} completed (commit: $last)" >> "$LOGS_DIR/hub_events.log" 2>&1
        else
            set_lane_status "$lane" "done"
            echo "[$(date '+%H:%M:%S')] Lane ${lane^^} finished (no commits)" >> "$LOGS_DIR/hub_events.log" 2>&1
        fi
    else
        set_lane_status "$lane" "done"
        echo "[$(date '+%H:%M:%S')] Lane ${lane^^} exited (no .agent-done sentinel)" >> "$LOGS_DIR/hub_events.log" 2>&1
    fi

    # Record finish time
    python3 -c "
import json, time
with open('$STATE_FILE') as f:
    state = json.load(f)
state['lanes']['$lane']['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=4)
" 2>/dev/null

    # macOS notification
    notify "Hub: Lane ${lane^^} Done" "${LANE_LABELS[$lane]} finished"

    # Auto-trigger review if HUB_AUTO_REVIEW=1
    if [[ "${HUB_AUTO_REVIEW:-1}" == "1" ]]; then
        echo "[$(date '+%H:%M:%S')] Auto-reviewing Lane ${lane^^}..." >> "$LOGS_DIR/hub_events.log" 2>&1
        auto_review_lane "$lane"
    fi
}

# Auto-review a single lane (called on exit, runs in background)
auto_review_lane() {
    local lane=$1
    local wt_path
    wt_path=$(get_worktree_path "$lane")
    local label="${LANE_LABELS[$lane]}"

    # Get diff and create structured summary via prediff processor
    local diff_content
    diff_content=$(git -C "$wt_path" diff main...HEAD 2>/dev/null)
    if [[ -z "$diff_content" ]]; then
        echo "[$(date '+%H:%M:%S')] Lane ${lane^^}: no changes to review" >> "$LOGS_DIR/hub_events.log" 2>&1
        return
    fi

    # Save raw diff
    echo "$diff_content" > "$REVIEWS_DIR/lane_${lane}_diff.patch"

    # Generate structured summary (90% fewer tokens for review agents)
    local review_summary
    if [[ -f "$PREDIFF" ]]; then
        review_summary=$(echo "$diff_content" | python3 "$PREDIFF" --lane "$lane" --contracts-file "$CONTRACTS_FILE" 2>/dev/null)
    fi
    if [[ -z "$review_summary" ]]; then
        review_summary="$diff_content"
    fi

    # Layer 1: Ownership auto-check
    local violations
    violations=$(check_ownership_violations "$lane")
    if [[ -n "$violations" ]]; then
        echo "[$(date '+%H:%M:%S')] Lane ${lane^^}: OWNERSHIP VIOLATION - $violations" >> "$LOGS_DIR/hub_events.log" 2>&1
        set_lane_status "$lane" "needs_revision"
        set_review_result "$lane" "hub" "reject" "Ownership violation: $violations"
        cat > "$REVIEWS_DIR/${lane}_feedback.md" << FEEDBACKEOF
# Auto-Rejection: Ownership Violation

Lane ${lane^^} modified files outside its ownership boundaries.

## Violating Files
$(for f in $violations; do echo "- \`$f\`"; done)

## Required Action
Remove or revert changes to the above files.
FEEDBACKEOF
        return
    fi

    echo "[$(date '+%H:%M:%S')] Lane ${lane^^}: ownership check passed, spawning cross-reviews..." >> "$LOGS_DIR/hub_events.log" 2>&1
    set_lane_status "$lane" "reviewing"

    # Load review template
    local review_template
    review_template=$(cat "$SCRIPT_DIR/templates/review_prompt.md" 2>/dev/null || echo "Review the following diff and respond with APPROVE: or REJECT: on the first line.")

    # Spawn cross-reviews in parallel (background)
    for reviewer_lane in $ALL_LANES; do
        [[ "$reviewer_lane" == "$lane" ]] && continue

        local reviewer_agent="$REVIEW_AGENT"
        local reviewer_label="${LANE_LABELS[$reviewer_lane]}"
        local review_file="$REVIEWS_DIR/${lane}_reviewed_by_${reviewer_lane}.md"

        # Build review prompt (use structured summary, not raw diff)
        local review_prompt="$review_template"
        review_prompt="${review_prompt//\{SOURCE_LANE\}/${lane^^}}"
        review_prompt="${review_prompt//\{SOURCE_LABEL\}/$label}"
        review_prompt="${review_prompt//\{REVIEWER_LANE\}/${reviewer_lane^^}}"
        review_prompt="${review_prompt//\{REVIEWER_LABEL\}/$reviewer_label}"
        review_prompt="${review_prompt//\{DIFF_CONTENT\}/$review_summary}"

        # Spawn review agent in background
        (
            local reviewer_wt
            reviewer_wt=$(get_worktree_path "$reviewer_lane")
            cd "$reviewer_wt" 2>/dev/null || cd "$REPO_ROOT"
            run_agent_review "$reviewer_agent" "$review_prompt" "$review_file" "${REVIEW_TIMEOUT:-300}"

            # Parse and record verdict
            local verdict
            verdict=$(parse_review_verdict "$review_file")
            local reason
            reason=$(parse_review_reason "$review_file")

            if [[ "$verdict" == "APPROVE" ]]; then
                set_review_result "$lane" "$reviewer_lane" "approve" "$reason"
                echo "[$(date '+%H:%M:%S')] Lane ${lane^^} reviewed by ${reviewer_lane^^}: APPROVED - $reason" >> "$LOGS_DIR/hub_events.log" 2>&1
            else
                set_review_result "$lane" "$reviewer_lane" "reject" "$reason"
                echo "[$(date '+%H:%M:%S')] Lane ${lane^^} reviewed by ${reviewer_lane^^}: REJECTED - $reason" >> "$LOGS_DIR/hub_events.log" 2>&1
            fi

            # Check if all reviews are in for this lane
            local all_in
            all_in=$(python3 -c "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
reviews = state.get('reviews', {})
needed = [l for l in '$ALL_LANES'.split() if l != '$lane']
results = []
for n in needed:
    key = '${lane}_by_' + n
    if key in reviews:
        results.append(reviews[key]['result'])
if len(results) == len(needed):
    if all(r == 'approve' for r in results):
        print('all_approved')
    else:
        print('has_rejections')
" 2>/dev/null)

            if [[ "$all_in" == "all_approved" ]]; then
                set_lane_status "$lane" "approved"
                echo "[$(date '+%H:%M:%S')] Lane ${lane^^}: ALL REVIEWS APPROVED" >> "$LOGS_DIR/hub_events.log" 2>&1
                notify "Hub: Lane ${lane^^} APPROVED" "All reviewers approved - ready to merge" "Hero"
            elif [[ "$all_in" == "has_rejections" ]]; then
                set_lane_status "$lane" "needs_revision"
                echo "[$(date '+%H:%M:%S')] Lane ${lane^^}: NEEDS REVISION - triggering auto-rework" >> "$LOGS_DIR/hub_events.log" 2>&1
                notify "Hub: Lane ${lane^^} REJECTED" "Auto-reworking with feedback" "Basso"

                # Auto-rework if enabled and under retry limit
                if [[ "${HUB_AUTO_REWORK:-1}" == "1" ]]; then
                    auto_rework_lane "$lane"
                fi
            fi
        ) &
    done
}

# Auto-rework a lane that was rejected, with retry limit
auto_rework_lane() {
    local lane=$1
    local max_retries="${HUB_MAX_RETRIES:-3}"

    # Track retry count
    local retries
    retries=$(python3 -c "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
count = state['lanes']['$lane'].get('retries', 0)
print(count)
" 2>/dev/null)

    if [[ "$retries" -ge "$max_retries" ]]; then
        echo "[$(date '+%H:%M:%S')] Lane ${lane^^}: MAX RETRIES ($max_retries) reached - needs manual intervention" >> "$LOGS_DIR/hub_events.log" 2>&1
        set_lane_status "$lane" "needs_revision"
        return
    fi

    # Increment retry count
    python3 -c "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
state['lanes']['$lane']['retries'] = state['lanes']['$lane'].get('retries', 0) + 1
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=4)
" 2>/dev/null

    local new_retry=$((retries + 1))
    echo "[$(date '+%H:%M:%S')] Lane ${lane^^}: Auto-rework attempt $new_retry/$max_retries" >> "$LOGS_DIR/hub_events.log" 2>&1

    local wt_path
    wt_path=$(get_worktree_path "$lane")
    local agent
    agent=$(get_lane_agent "$lane")
    local label="${LANE_LABELS[$lane]}"

    # Collect all rejection feedback
    local feedback=""
    for reviewer_lane in $ALL_LANES; do
        [[ "$reviewer_lane" == "$lane" ]] && continue
        local review_file="$REVIEWS_DIR/${lane}_reviewed_by_${reviewer_lane}.md"
        if [[ -f "$review_file" ]]; then
            local verdict
            verdict=$(parse_review_verdict "$review_file")
            if [[ "$verdict" == "REJECT" ]]; then
                feedback="${feedback}

## Feedback from Lane ${reviewer_lane^^} ($(get_lane_agent "$reviewer_lane"))
$(cat "$review_file")
"
            fi
        fi
    done

    # Write combined feedback
    cat > "$REVIEWS_DIR/${lane}_feedback.md" << RFEOF
# Review Feedback for Lane ${lane^^} (Attempt $new_retry/$max_retries)
$feedback
RFEOF

    # Read original task
    local original_task=""
    if [[ -f "$wt_path/.hub-task.md" ]]; then
        original_task=$(cat "$wt_path/.hub-task.md")
    fi

    # Remove done sentinel
    rm -f "$wt_path/.agent-done"

    # Build rework prompt
    local prompt="You are working on Lane ${lane^^} ($label) of a multi-agent project.

Your previous work was REJECTED during cross-review (attempt $new_retry/$max_retries). Fix the issues.

## Original Task
$original_task

## Review Feedback
$feedback

## Instructions
1. Read the feedback carefully
2. Fix ONLY the issues identified - don't rewrite everything
3. Commit your changes
4. Run: touch .agent-done

Only modify files you own (see CLAUDE.md for ownership rules)."

    local log_file="$LOGS_DIR/lane_${lane}_rework${new_retry}_$(date +%Y%m%d_%H%M%S).log"

    # Clear old reviews
    python3 -c "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
state['reviews'] = {k: v for k, v in state.get('reviews', {}).items()
                    if not k.startswith('${lane}_')}
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=4)
" 2>/dev/null

    set_lane_status "$lane" "working"

    # Spawn agent with auto-update on exit (which triggers another review)
    (
        cd "$wt_path"
        run_agent_work "$agent" "$prompt" "$log_file" "${DISPATCH_TIMEOUT:-3600}"
        on_lane_exit "$lane"
    ) &
    local pid=$!

    set_lane_pid "$lane" "$pid"
    echo "[$(date '+%H:%M:%S')] Lane ${lane^^}: Rework dispatched to $agent (PID: $pid)" >> "$LOGS_DIR/hub_events.log" 2>&1
}

# Called after a standalone task agent exits to auto-update state
on_task_exit() {
    local task_id=$1
    local log_file
    log_file=$(python3 -c "
import json, time
with open('$STATE_FILE') as f:
    state = json.load(f)
for t in state.get('tasks', []):
    if t['id'] == $task_id:
        t['status'] = 'finished'
        t['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        # Extract summary from log
        log = t.get('log', '')
        print(log)
        break
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=4)
" 2>/dev/null)

    echo "[$(date '+%H:%M:%S')] Task #$task_id finished" >> "$LOGS_DIR/hub_events.log" 2>&1

    # Save summary
    if [[ -n "$log_file" ]] && [[ -f "$log_file" ]]; then
        extract_log_summary "$log_file" 3 > "$LOGS_DIR/task_${task_id}_summary.txt" 2>/dev/null
    fi

    # macOS notification
    local desc
    desc=$(python3 -c "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
for t in state.get('tasks', []):
    if t['id'] == $task_id:
        print(t.get('description', 'Task')[:50])
        break
" 2>/dev/null)
    notify "Hub: Task #$task_id Done" "$desc"
}

# ── Ownership Enforcement ─────────────────────────────────────────────────────

# Check if a file path belongs to a lane
file_owned_by_lane() {
    local file=$1
    local lane=$2
    local owned_paths="${LANE_OWNED_PATHS[$lane]}"

    for owned in $owned_paths; do
        # Match if file starts with owned path
        if [[ "$file" == "$owned"* ]]; then
            return 0
        fi
    done
    return 1
}

# Check if a file is in shared paths
file_is_shared() {
    local file=$1
    for shared in $SHARED_PATHS; do
        if [[ "$file" == "$shared"* ]]; then
            return 0
        fi
    done
    return 1
}

# Check diff for ownership violations. Returns list of violating files.
check_ownership_violations() {
    local lane=$1
    local wt_path
    wt_path=$(get_worktree_path "$lane")

    local violations=""
    local changed_files
    changed_files=$(git -C "$wt_path" diff main...HEAD --name-only 2>/dev/null)

    for file in $changed_files; do
        if file_owned_by_lane "$file" "$lane"; then
            continue
        fi
        if file_is_shared "$file"; then
            continue
        fi
        # Check if it belongs to another lane
        violations="$violations $file"
    done

    echo "$violations" | xargs  # trim whitespace
}

# ── Task Parsing ──────────────────────────────────────────────────────────────

# Extract section for a lane from a task file
parse_lane_task() {
    local task_file=$1
    local lane=$2
    local lane_upper
    lane_upper=$(echo "$lane" | tr '[:lower:]' '[:upper:]')

    python3 -c "
import re, sys

with open('$task_file') as f:
    content = f.read()

# Find section for Lane $lane_upper
pattern = r'## Lane ${lane_upper}\s*\n(.*?)(?=\n## Lane [A-Z]|\n## [^L]|\Z)'
match = re.search(pattern, content, re.DOTALL)
if match:
    text = match.group(1).strip()
    if text.upper() != 'IDLE' and text:
        print(text)
" 2>/dev/null
}

# ── Agent Invocation ──────────────────────────────────────────────────────────

# Run an agent with full write permissions (for dispatch/rework)
# Returns exit status: 0=success, 124=timeout, other=error
run_agent_work() {
    local agent=$1
    local prompt=$2
    local log_file=$3
    local timeout_secs=$4
    local exit_code=0

    case $agent in
        claude)
            timeout --signal=TERM "$timeout_secs" claude -p "$prompt" --dangerously-skip-permissions > "$log_file" 2>&1 || exit_code=$?
            ;;
        codex)
            timeout --signal=TERM "$timeout_secs" codex exec --full-auto "$prompt" > "$log_file" 2>&1 || exit_code=$?
            ;;
        gemini)
            timeout --signal=TERM "$timeout_secs" gemini -p "$prompt" --yolo > "$log_file" 2>&1 || exit_code=$?
            ;;
        opencode)
            timeout --signal=TERM "$timeout_secs" opencode run "$prompt" > "$log_file" 2>&1 || exit_code=$?
            ;;
        kimi)
            timeout --signal=TERM "$timeout_secs" kimi --quiet -p "$prompt" --yolo > "$log_file" 2>&1 || exit_code=$?
            ;;
        *)
            hub_error "Unknown agent: $agent"
            return 1
            ;;
    esac

    # Log exit status
    case $exit_code in
        0)   echo "[agent] Completed successfully" >> "$log_file" ;;
        124) echo "[agent] TIMEOUT after ${timeout_secs}s" >> "$log_file" ;;
        137) echo "[agent] KILLED (SIGKILL)" >> "$log_file" ;;
        *)   echo "[agent] Exited with code $exit_code" >> "$log_file" ;;
    esac
    return $exit_code
}

# Run an agent in read-only mode (for reviews - only needs text output)
run_agent_review() {
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
            hub_error "Unknown agent: $agent"
            return 1
            ;;
    esac
}

# ── Review Parsing ─────────────────────────────────────────────────────────────

# Parse review output for APPROVE/REJECT verdict
parse_review_verdict() {
    local review_file=$1

    if [[ ! -f "$review_file" ]] || [[ ! -s "$review_file" ]]; then
        echo "REJECT"
        return
    fi

    local content
    content=$(cat "$review_file")

    # Look for APPROVE: or REJECT: at start of a line
    if echo "$content" | grep -qiE "^APPROVE:"; then
        echo "APPROVE"
    elif echo "$content" | grep -qiE "^REJECT:"; then
        echo "REJECT"
    else
        # Fallback: search anywhere in content
        if echo "$content" | grep -qiE "APPROVE:"; then
            echo "APPROVE"
        else
            echo "REJECT"
        fi
    fi
}

# Extract reason from review verdict line
parse_review_reason() {
    local review_file=$1

    if [[ ! -f "$review_file" ]] || [[ ! -s "$review_file" ]]; then
        echo "No review output"
        return
    fi

    local content
    content=$(cat "$review_file")

    # Extract reason after APPROVE: or REJECT:
    local reason
    reason=$(echo "$content" | grep -iE "^(APPROVE|REJECT):" | head -1 | sed 's/^[^:]*: *//')
    if [[ -n "$reason" ]]; then
        echo "$reason"
    else
        echo "No explicit reason given"
    fi
}
