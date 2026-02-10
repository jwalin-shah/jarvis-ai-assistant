#!/usr/bin/env bash
# hub_lib.sh - Helper functions for hub.sh multi-agent orchestration
# Provides: state management, logging, ownership checking, task parsing

# Prevent double-sourcing
[[ -n "${_HUB_LIB_LOADED:-}" ]] && return 0
_HUB_LIB_LOADED=1

# ── Paths ──────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HUB_DIR="$REPO_ROOT/.hub"
STATE_FILE="$HUB_DIR/state.json"
REVIEWS_DIR="$HUB_DIR/reviews"
LOGS_DIR="$HUB_DIR/logs"

# ── Colors ─────────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Lane Definitions ──────────────────────────────────────────────────────────

declare -A LANE_AGENTS=(
    [a]="codex"
    [b]="claude"
    [c]="gemini"
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
    [a]="desktop/ api/ jarvis/router.py jarvis/prompts.py jarvis/retrieval/"
    [b]="models/ jarvis/classifiers/ jarvis/extractors/ jarvis/graph/ scripts/train scripts/extract"
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

lane_log() {
    local lane=$1
    local msg=$2
    local agent="${LANE_AGENTS[$lane]}"
    echo -e "${BLUE}[lane-${lane}/${agent}]${NC} $msg"
}

# ── State Management ──────────────────────────────────────────────────────────

init_state() {
    mkdir -p "$HUB_DIR" "$REVIEWS_DIR" "$LOGS_DIR"
    cat > "$STATE_FILE" << 'STATEEOF'
{
    "lanes": {
        "a": {"status": "idle", "agent": "codex", "pid": null, "last_commit": null},
        "b": {"status": "idle", "agent": "claude", "pid": null, "last_commit": null},
        "c": {"status": "idle", "agent": "gemini", "pid": null, "last_commit": null}
    },
    "reviews": {},
    "created_at": "TIMESTAMP"
}
STATEEOF
    # Fill in timestamp
    local ts
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    sed -i '' "s/TIMESTAMP/$ts/" "$STATE_FILE"
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
    python3 -c "
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
    python3 -c "
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
    python3 -c "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
state['lanes']['$lane']['last_commit'] = '$commit'
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=4)
"
}

set_review_result() {
    local source_lane=$1
    local reviewer_lane=$2
    local result=$3  # "approve" or "reject"
    local reason=$4
    python3 -c "
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

build_agent_cmd() {
    local agent=$1
    local prompt=$2

    case $agent in
        claude)
            echo "claude -p $(printf '%q' "$prompt") --print"
            ;;
        codex)
            echo "codex e $(printf '%q' "$prompt")"
            ;;
        gemini)
            echo "gemini -p $(printf '%q' "$prompt")"
            ;;
        opencode)
            echo "opencode run $(printf '%q' "$prompt")"
            ;;
        kimi)
            echo "kimi --quiet -p $(printf '%q' "$prompt")"
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
