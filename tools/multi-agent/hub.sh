#!/usr/bin/env bash
# hub.sh - Multi-Agent Hub-Spoke Orchestration
#
# Spawns worker agents into isolated git worktrees, coordinates cross-review,
# and gates merges on mutual approval.
#
# Usage:
#   hub.sh setup                  Create worktrees and state directory
#   hub.sh dispatch <task-file>   Send tasks to lane agents
#   hub.sh status                 Show lane statuses
#   hub.sh review                 Cross-review completed lanes
#   hub.sh rework <lane>          Re-dispatch with rejection feedback
#   hub.sh merge                  Merge all approved lanes to main
#   hub.sh teardown               Remove worktrees and state

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/hub_lib.sh"

DISPATCH_TIMEOUT=3600   # 1 hour for main work
REVIEW_TIMEOUT=300      # 5 minutes for reviews

# ── Signal Handling ───────────────────────────────────────────────────────────

declare -a HUB_CHILD_PIDS=()

cleanup_children() {
    if [[ ${#HUB_CHILD_PIDS[@]} -gt 0 ]]; then
        hub_log "Shutting down, terminating ${#HUB_CHILD_PIDS[@]} child processes..."
        for pid in "${HUB_CHILD_PIDS[@]}"; do
            kill -TERM "$pid" 2>/dev/null || true
        done
        sleep 1
        for pid in "${HUB_CHILD_PIDS[@]}"; do
            kill -KILL "$pid" 2>/dev/null || true
        done
    fi
    _state_unlock 2>/dev/null || true
}

trap cleanup_children INT TERM

# ── Usage ──────────────────────────────────────────────────────────────────────

usage() {
    cat << 'EOF'
Usage: hub.sh <command> [args]

Commands:
  setup                           Create worktrees, state directory, per-lane CLAUDE.md
  dispatch <task-file>            Parse task file and spawn agents into worktrees
  run <agent> [-m model] <prompt> Run a standalone agent task (tracked in status)
  status                          Show all lanes + standalone tasks
  watch [interval]                Auto-refresh status (default: 5s), stops when all done
  logs <lane|task-id> [-f]        Show log for a lane (a/b/c) or task (#1, #2, ...)
  review                          Cross-review completed lanes (ownership + agent review)
  rework <lane>                   Re-dispatch a lane with rejection feedback
  merge                           Merge all approved lanes to main (runs make verify)
  summary                         Show session stats: agent usage, token/cost estimates, events
  teardown                        Remove worktrees and clean up state

Agents: claude, codex, gemini, kimi, opencode

Options:
  -h, --help             Show this help

Examples:
  hub.sh setup
  hub.sh dispatch tasks/my-task.md
  hub.sh run codex "Audit all SQL queries and write a report"
  hub.sh run codex -m o3 "Deep code review of jarvis/"
  hub.sh run kimi "Generate test fixtures for contracts"
  hub.sh status
  hub.sh review
  hub.sh rework b
  hub.sh merge
  hub.sh teardown
EOF
    exit 0
}

# ── Command: setup ─────────────────────────────────────────────────────────────

cmd_setup() {
    hub_log "Setting up hub-spoke worktrees..."

    # Check we're in the main repo
    if [[ ! -d "$REPO_ROOT/.git" ]]; then
        hub_error "Must run from the main repository root"
        exit 1
    fi

    # Initialize state
    init_state
    hub_success "State directory created: $HUB_DIR"

    # Create worktrees for each lane
    for lane in $ALL_LANES; do
        local branch="${LANE_BRANCHES[$lane]}"
        local wt_path
        wt_path=$(get_worktree_path "$lane")
        local label="${LANE_LABELS[$lane]}"
        local agent="${LANE_AGENTS[$lane]}"

        if worktree_exists "$lane"; then
            hub_warn "Worktree already exists: $wt_path (skipping)"
            continue
        fi

        lane_log "$lane" "Creating worktree: $wt_path (branch: $branch)"

        # Create branch from main if it doesn't exist
        if ! git -C "$REPO_ROOT" show-ref --verify --quiet "refs/heads/$branch" 2>/dev/null; then
            git -C "$REPO_ROOT" branch "$branch" main
        fi

        # Create worktree
        git -C "$REPO_ROOT" worktree add "$wt_path" "$branch"

        # Copy lane-specific CLAUDE.md
        local lane_lower=$lane
        cp "$SCRIPT_DIR/templates/lane_${lane_lower}_claude.md" "$wt_path/CLAUDE.md"

        # Create LANE_OWNERSHIP.md
        cat > "$wt_path/LANE_OWNERSHIP.md" << OWNEREOF
# Lane ${lane^^}: $label

Agent: $agent
Branch: $branch

## Owned Paths
$(for p in ${LANE_OWNED_PATHS[$lane]}; do echo "- \`$p\`"; done)

## Shared Paths (require cross-lane approval)
- \`jarvis/contracts/pipeline.py\`

## Do Not Touch
$(for other_lane in $ALL_LANES; do
    if [[ "$other_lane" != "$lane" ]]; then
        for p in ${LANE_OWNED_PATHS[$other_lane]}; do
            echo "- \`$p\` (Lane ${other_lane^^})"
        done
    fi
done)
OWNEREOF

        lane_log "$lane" "Worktree ready with CLAUDE.md and LANE_OWNERSHIP.md"
    done

    echo ""
    hub_success "Setup complete. Worktrees:"
    for lane in $ALL_LANES; do
        local wt_path
        wt_path=$(get_worktree_path "$lane")
        echo "  Lane ${lane^^} (${LANE_AGENTS[$lane]}): $wt_path"
    done
    echo ""
    hub_log "Next: hub.sh dispatch <task-file>"
}

# ── Command: dispatch ──────────────────────────────────────────────────────────

cmd_dispatch() {
    local task_file="${1:-}"

    if [[ -z "$task_file" ]]; then
        hub_error "Usage: hub.sh dispatch <task-file>"
        exit 1
    fi

    if [[ ! -f "$task_file" ]]; then
        hub_error "Task file not found: $task_file"
        exit 1
    fi

    # Use absolute path
    task_file="$(cd "$(dirname "$task_file")" && pwd)/$(basename "$task_file")"

    hub_log "Dispatching tasks from: $task_file"

    local dispatched=0

    for lane in $ALL_LANES; do
        local task_content
        task_content=$(parse_lane_task "$task_file" "$lane")

        if [[ -z "$task_content" ]]; then
            lane_log "$lane" "No task (IDLE)"
            continue
        fi

        local status
        status=$(get_lane_status "$lane")
        if [[ "$status" == "working" ]]; then
            hub_warn "Lane $lane is already working, skipping"
            continue
        fi

        local wt_path
        wt_path=$(get_worktree_path "$lane")
        local agent="${LANE_AGENTS[$lane]}"
        local label="${LANE_LABELS[$lane]}"

        if ! worktree_exists "$lane"; then
            hub_error "Worktree missing for lane $lane. Run 'hub.sh setup' first."
            exit 1
        fi

        # Write task to worktree
        echo "$task_content" > "$wt_path/.hub-task.md"

        # Remove stale done sentinel
        rm -f "$wt_path/.agent-done"

        # Build prompt
        local prompt="You are working on Lane ${lane^^} ($label) of a multi-agent project.

Your task:
$task_content

IMPORTANT:
- Only modify files you own (see CLAUDE.md in this directory for ownership rules)
- When done, commit your changes and run: touch .agent-done
- Your work will be cross-reviewed by other lane agents before merge
- Read .hub-task.md for the full task description"

        # Build agent command
        local log_file="$LOGS_DIR/lane_${lane}_$(date +%Y%m%d_%H%M%S).log"

        lane_log "$lane" "Dispatching to $agent..."

        # Spawn agent in background with full write permissions + auto-update on exit
        (
            cd "$wt_path"
            run_agent_work "$agent" "$prompt" "$log_file" "$DISPATCH_TIMEOUT"
            on_lane_exit "$lane"
        ) &
        local pid=$!
        HUB_CHILD_PIDS+=("$pid")

        set_lane_status "$lane" "working"
        set_lane_pid "$lane" "$pid"

        # Record start time
        python3 -c "
import json, time
with open('$STATE_FILE') as f:
    state = json.load(f)
state['lanes']['$lane']['started_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
state['lanes']['$lane']['finished_at'] = None
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=4)
" 2>/dev/null

        lane_log "$lane" "Agent PID: $pid, log: $log_file"
        dispatched=$((dispatched + 1))
    done

    echo ""
    if [[ $dispatched -eq 0 ]]; then
        hub_warn "No lanes dispatched (all IDLE or already working)"
    else
        hub_success "$dispatched lane(s) dispatched"
        hub_log "Monitor with: hub.sh status"
    fi
}

# ── Command: run ───────────────────────────────────────────────────────────────

cmd_run() {
    local agent="${1:-}"
    shift || true

    if [[ -z "$agent" ]]; then
        hub_error "Usage: hub.sh run <agent> [-m model] <prompt>"
        hub_error "Agents: claude, codex, gemini, kimi, opencode"
        exit 1
    fi

    # Parse optional flags: -m model, -l label
    local model="default"
    local label=""
    while [[ "${1:-}" == -* ]]; do
        case "${1}" in
            -m) shift; model="${1:-default}"; shift || true ;;
            -l) shift; label="${1:-}"; shift || true ;;
            *) break ;;
        esac
    done

    local prompt="${*}"
    if [[ -z "$prompt" ]]; then
        hub_error "Usage: hub.sh run <agent> [-m model] [-l label] <prompt>"
        exit 1
    fi

    # Ensure state exists
    if [[ ! -f "$STATE_FILE" ]]; then
        mkdir -p "$HUB_DIR" "$REVIEWS_DIR" "$LOGS_DIR"
        init_state
    fi

    # Ensure tasks array exists
    python3 -c "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
if 'tasks' not in state:
    state['tasks'] = []
    with open('$STATE_FILE', 'w') as f:
        json.dump(state, f, indent=4)
" 2>/dev/null

    local log_file="$LOGS_DIR/${agent}_task_$(date +%Y%m%d_%H%M%S).log"
    local short_desc="${label:-${prompt:0:60}}"

    hub_log "Running $agent${model:+ ($model)}: $short_desc..."

    # Build model flag
    local model_flag=""
    if [[ "$model" != "default" ]]; then
        case $agent in
            codex)   model_flag="-m $model" ;;
            claude)  model_flag="--model $model" ;;
            gemini)  model_flag="-m $model" ;;
            kimi)    model_flag="-m $model" ;;
            opencode) model_flag="" ;; # opencode doesn't support model flag in run
        esac
    fi

    # Pre-register task to get ID, then spawn with auto-update on exit
    local task_id
    task_id=$(add_task "$agent" "$short_desc" "0" "$log_file" "$model")

    (
        cd "$REPO_ROOT"
        case $agent in
            claude)
                timeout "$DISPATCH_TIMEOUT" claude -p "$prompt" --dangerously-skip-permissions $model_flag > "$log_file" 2>&1 || true
                ;;
            codex)
                timeout "$DISPATCH_TIMEOUT" codex exec --full-auto $model_flag "$prompt" > "$log_file" 2>&1 || true
                ;;
            gemini)
                timeout "$DISPATCH_TIMEOUT" gemini -p "$prompt" --yolo $model_flag > "$log_file" 2>&1 || true
                ;;
            opencode)
                timeout "$DISPATCH_TIMEOUT" opencode run $model_flag "$prompt" > "$log_file" 2>&1 || true
                ;;
            kimi)
                timeout "$DISPATCH_TIMEOUT" kimi --quiet -p "$prompt" --yolo $model_flag > "$log_file" 2>&1 || true
                ;;
            *)
                echo "Unknown agent: $agent" >> "$log_file"
                ;;
        esac
        on_task_exit "$task_id"
    ) &
    local pid=$!
    HUB_CHILD_PIDS+=("$pid")

    # Update the task with the real PID
    python3 -c "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
for t in state.get('tasks', []):
    if t['id'] == $task_id:
        t['pid'] = $pid
        break
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=4)
" 2>/dev/null

    hub_success "Task #$task_id started: $agent${model:+ ($model)} PID $pid"
    hub_log "Log: $log_file"
    hub_log "Monitor with: hub.sh status"
}

# ── Command: status ────────────────────────────────────────────────────────────

cmd_status() {
    if [[ ! -f "$STATE_FILE" ]]; then
        hub_error "No hub state found. Run 'hub.sh setup' first."
        exit 1
    fi

    echo -e "${BOLD}Hub Status${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf "%-6s %-12s %-10s %-8s %-8s %s\n" "LANE" "STATUS" "AGENT" "PID" "TIME" "LAST COMMIT"
    echo "──────────────────────────────────────────────────────────────────────────────"

    for lane in $ALL_LANES; do
        local status
        status=$(get_lane_status "$lane")
        local agent="${LANE_AGENTS[$lane]}"
        local wt_path
        wt_path=$(get_worktree_path "$lane")

        # Check PID
        local pid_info="-"
        local raw_pid
        raw_pid=$(python3 -c "
import json
with open('$STATE_FILE') as f:
    s = json.load(f)
p = s['lanes']['$lane'].get('pid')
print(p if p else '')
" 2>/dev/null)

        if [[ -n "$raw_pid" ]]; then
            if kill -0 "$raw_pid" 2>/dev/null; then
                pid_info="$raw_pid"
            else
                pid_info="done"
            fi
        fi

        # Check for agent-done sentinel
        local done_flag="no"
        if agent_done_exists "$lane"; then
            done_flag="yes"
            # Auto-transition working -> done
            if [[ "$status" == "working" ]]; then
                # Verify there are commits
                local commits
                commits=$(git -C "$wt_path" log main..HEAD --oneline 2>/dev/null | wc -l | tr -d ' ')
                if [[ "$commits" -gt 0 ]]; then
                    set_lane_status "$lane" "done"
                    status="done"
                    local last
                    last=$(git -C "$wt_path" log -1 --format="%h" 2>/dev/null)
                    set_lane_commit "$lane" "$last"
                fi
            fi
        fi

        # Last commit
        local last_commit="-"
        if worktree_exists "$lane"; then
            last_commit=$(worktree_last_commit "$lane")
        fi

        # Color status
        local status_colored="$status"
        case $status in
            idle)            status_colored="${NC}idle${NC}" ;;
            working)         status_colored="${YELLOW}working${NC}" ;;
            done)            status_colored="${BLUE}done${NC}" ;;
            reviewing)       status_colored="${CYAN}reviewing${NC}" ;;
            approved)        status_colored="${GREEN}approved${NC}" ;;
            needs_revision)  status_colored="${RED}needs_rev${NC}" ;;
            merged)          status_colored="${GREEN}merged${NC}" ;;
        esac

        # Duration
        local duration="-"
        local started_at
        started_at=$(python3 -c "
import json
with open('$STATE_FILE') as f:
    s = json.load(f)
print(s['lanes']['$lane'].get('started_at') or '')
" 2>/dev/null)
        if [[ -n "$started_at" ]]; then
            local finished_at
            finished_at=$(python3 -c "
import json
with open('$STATE_FILE') as f:
    s = json.load(f)
print(s['lanes']['$lane'].get('finished_at') or '')
" 2>/dev/null)
            if [[ -n "$finished_at" ]]; then
                duration=$(format_duration "$started_at" "$finished_at")
            else
                duration=$(format_duration "$started_at")
            fi
        fi

        printf "%-6s %-22b %-10s %-8s %-8s %s\n" \
            "${lane^^}" "$status_colored" "$agent" "$pid_info" "$duration" "$last_commit"
    done

    echo ""

    # Show reviews if any
    local has_reviews
    has_reviews=$(python3 -c "
import json
with open('$STATE_FILE') as f:
    s = json.load(f)
reviews = s.get('reviews', {})
if reviews:
    print('yes')
    for k, v in reviews.items():
        src, _, reviewer = k.partition('_by_')
        result = v['result'].upper()
        reason = v.get('reason', '')[:60]
        print(f'  {src.upper()} reviewed by {reviewer.upper()}: {result} - {reason}')
" 2>/dev/null)

    if [[ -n "$has_reviews" ]]; then
        echo -e "${BOLD}Reviews${NC}"
        echo "────────────────────────────────────────────────────────────────"
        echo "$has_reviews" | tail -n +2
        echo ""
    fi

    # Show standalone tasks
    local task_output
    task_output=$(python3 -c "
import json, os, time
from datetime import datetime, timezone

with open('$STATE_FILE') as f:
    state = json.load(f)
tasks = state.get('tasks', [])
if not tasks:
    exit(0)

def fmt_duration(started, finished=None):
    try:
        s = datetime.fromisoformat(started.replace('Z', '+00:00'))
        e = datetime.fromisoformat(finished.replace('Z', '+00:00')) if finished else datetime.now(timezone.utc)
        secs = int((e - s).total_seconds())
        if secs < 60: return f'{secs}s'
        if secs < 3600: return f'{secs // 60}m {secs % 60}s'
        return f'{secs // 3600}h {(secs % 3600) // 60}m'
    except: return '?'

print('TASKS')
for t in tasks:
    pid = t.get('pid')
    status = t.get('status', 'unknown')
    if status == 'working' and pid:
        try:
            os.kill(pid, 0)
            pid_info = str(pid)
        except (OSError, ProcessLookupError):
            pid_info = 'done'
            status = 'finished'
            t['status'] = 'finished'
            t['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    else:
        pid_info = str(pid) if pid else '-'
    agent = t.get('agent', '?')
    model = t.get('model', 'default')
    desc = t.get('description', '')[:45]
    task_id = t.get('id', '?')
    model_str = f' ({model})' if model != 'default' else ''
    agent_str = f'{agent}{model_str}'
    dur = fmt_duration(t.get('started_at', ''), t.get('finished_at'))
    print(f'  #{task_id:<4} {status:<12} {agent_str:<16} {pid_info:<8} {dur:<8} {desc}')
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=4)
" 2>/dev/null)

    if [[ -n "$task_output" ]]; then
        echo -e "${BOLD}Standalone Tasks${NC}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        printf "  %-6s %-12s %-16s %-8s %-8s %s\n" "ID" "STATUS" "AGENT" "PID" "TIME" "DESCRIPTION"
        echo "  ──────────────────────────────────────────────────────────────────────────"
        echo "$task_output" | tail -n +2
        echo ""
    fi
}

# ── Command: watch ─────────────────────────────────────────────────────────────

cmd_watch() {
    local interval="${1:-5}"
    hub_log "Watching hub status (refresh every ${interval}s, Ctrl+C to stop)"
    echo ""

    while true; do
        clear
        cmd_status

        # Count active work
        local active=0
        for lane in $ALL_LANES; do
            local status
            status=$(get_lane_status "$lane")
            if [[ "$status" == "working" ]]; then
                active=$((active + 1))
            fi
        done

        local active_tasks
        active_tasks=$(python3 -c "
import json, os
with open('$STATE_FILE') as f:
    state = json.load(f)
count = 0
for t in state.get('tasks', []):
    if t.get('status') == 'working' and t.get('pid'):
        try:
            os.kill(t['pid'], 0)
            count += 1
        except (OSError, ProcessLookupError):
            pass
print(count)
" 2>/dev/null)
        active=$((active + active_tasks))

        # Show recent events
        if [[ -f "$LOGS_DIR/hub_events.log" ]]; then
            local recent
            recent=$(tail -5 "$LOGS_DIR/hub_events.log" 2>/dev/null)
            if [[ -n "$recent" ]]; then
                echo -e "${BOLD}Recent Events${NC}"
                echo "────────────────────────────────────────────────────────────────"
                echo "$recent"
                echo ""
            fi
        fi

        echo -e "${CYAN}Active: $active | Refreshing every ${interval}s | Ctrl+C to stop${NC}"

        if [[ "$active" -eq 0 ]]; then
            echo ""
            hub_success "All work complete!"
            break
        fi

        sleep "$interval"
    done
}

# ── Command: logs ─────────────────────────────────────────────────────────────

cmd_logs() {
    local target="${1:-}"

    if [[ -z "$target" ]]; then
        hub_error "Usage: hub.sh logs <lane|task-id>"
        hub_error "  hub.sh logs a          Show Lane A log"
        hub_error "  hub.sh logs 3          Show standalone task #3 log"
        hub_error "  hub.sh logs a --tail   Follow Lane A log"
        exit 1
    fi

    local follow=false
    if [[ "${2:-}" == "--tail" ]] || [[ "${2:-}" == "-f" ]]; then
        follow=true
    fi

    local log_file=""

    # Check if it's a lane (a, b, c)
    if [[ "$target" =~ ^[abc]$ ]]; then
        # Find most recent lane log
        log_file=$(ls -t "$LOGS_DIR"/lane_${target}_*.log 2>/dev/null | head -1)
    elif [[ "$target" =~ ^[0-9]+$ ]]; then
        # It's a task ID
        log_file=$(python3 -c "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
for t in state.get('tasks', []):
    if t['id'] == $target:
        print(t.get('log', ''))
        break
" 2>/dev/null)
    fi

    if [[ -z "$log_file" ]] || [[ ! -f "$log_file" ]]; then
        hub_error "No log found for: $target"
        exit 1
    fi

    if $follow; then
        hub_log "Following: $log_file (Ctrl+C to stop)"
        tail -f "$log_file"
    else
        hub_log "Log: $log_file"
        echo "────────────────────────────────────────────────────────────────"
        cat "$log_file"
    fi
}

# ── Command: review ────────────────────────────────────────────────────────────

cmd_review() {
    if [[ ! -f "$STATE_FILE" ]]; then
        hub_error "No hub state found. Run 'hub.sh setup' first."
        exit 1
    fi

    local reviewed=0

    for lane in $ALL_LANES; do
        local status
        status=$(get_lane_status "$lane")

        if [[ "$status" != "done" ]]; then
            continue
        fi

        local wt_path
        wt_path=$(get_worktree_path "$lane")
        local label="${LANE_LABELS[$lane]}"

        lane_log "$lane" "Generating diff for review..."

        # Get diff
        local diff_content
        diff_content=$(git -C "$wt_path" diff main...HEAD 2>/dev/null)

        if [[ -z "$diff_content" ]]; then
            lane_log "$lane" "No changes to review"
            continue
        fi

        # Save raw diff and generate structured summary
        echo "$diff_content" > "$REVIEWS_DIR/lane_${lane}_diff.patch"

        local review_summary
        if [[ -f "$PREDIFF" ]]; then
            review_summary=$(echo "$diff_content" | python3 "$PREDIFF" --lane "$lane" --contracts-file "$CONTRACTS_FILE" 2>/dev/null)
        fi
        if [[ -z "${review_summary:-}" ]]; then
            review_summary="$diff_content"
        fi

        # Layer 1: Ownership auto-check
        local violations
        violations=$(check_ownership_violations "$lane")

        if [[ -n "$violations" ]]; then
            hub_error "Lane ${lane^^} OWNERSHIP VIOLATION - auto-rejected"
            hub_error "Files outside ownership: $violations"

            # Write feedback
            cat > "$REVIEWS_DIR/${lane}_feedback.md" << FEEDBACKEOF
# Auto-Rejection: Ownership Violation

Lane ${lane^^} modified files outside its ownership boundaries.

## Violating Files
$(for f in $violations; do echo "- \`$f\`"; done)

## Required Action
Remove or revert changes to the above files. Only modify files listed in your CLAUDE.md ownership section.
FEEDBACKEOF

            set_lane_status "$lane" "needs_revision"
            set_review_result "$lane" "hub" "reject" "Ownership violation: $violations"
            reviewed=$((reviewed + 1))
            continue
        fi

        lane_log "$lane" "Ownership check passed"

        # Layer 2: Check for shared file modifications
        local shared_changes
        shared_changes=$(git -C "$wt_path" diff main...HEAD --name-only 2>/dev/null | grep "^jarvis/contracts/" || true)

        if [[ -n "$shared_changes" ]]; then
            hub_warn "Lane ${lane^^} modifies shared contracts - flagging for all-lane review"
        fi

        # Layer 3: Cross-review by other lanes
        set_lane_status "$lane" "reviewing"

        local review_template
        review_template=$(cat "$SCRIPT_DIR/templates/review_prompt.md")

        local all_approved=true

        for reviewer_lane in $ALL_LANES; do
            [[ "$reviewer_lane" == "$lane" ]] && continue

            local reviewer_agent="${LANE_AGENTS[$reviewer_lane]}"
            local reviewer_label="${LANE_LABELS[$reviewer_lane]}"
            local review_file="$REVIEWS_DIR/${lane}_reviewed_by_${reviewer_lane}.md"

            # Build review prompt from template
            local review_prompt="$review_template"
            review_prompt="${review_prompt//\{SOURCE_LANE\}/${lane^^}}"
            review_prompt="${review_prompt//\{SOURCE_LABEL\}/$label}"
            review_prompt="${review_prompt//\{REVIEWER_LANE\}/${reviewer_lane^^}}"
            review_prompt="${review_prompt//\{REVIEWER_LABEL\}/$reviewer_label}"
            review_prompt="${review_prompt//\{DIFF_CONTENT\}/$review_summary}"

            lane_log "$lane" "Requesting review from Lane ${reviewer_lane^^} ($reviewer_agent)..."

            # Run reviewer agent
            local reviewer_wt
            reviewer_wt=$(get_worktree_path "$reviewer_lane")

            (
                cd "$reviewer_wt"
                run_agent_review "$reviewer_agent" "$review_prompt" "$review_file" "$REVIEW_TIMEOUT"
            )

            # Parse verdict
            local verdict
            verdict=$(parse_review_verdict "$review_file")
            local reason
            reason=$(parse_review_reason "$review_file")

            if [[ "$verdict" == "APPROVE" ]]; then
                lane_log "$lane" "Lane ${reviewer_lane^^} ($reviewer_agent): ${GREEN}APPROVED${NC} - $reason"
                set_review_result "$lane" "$reviewer_lane" "approve" "$reason"
            else
                lane_log "$lane" "Lane ${reviewer_lane^^} ($reviewer_agent): ${RED}REJECTED${NC} - $reason"
                set_review_result "$lane" "$reviewer_lane" "reject" "$reason"
                all_approved=false

                # Save feedback
                cat > "$REVIEWS_DIR/${lane}_feedback.md" << RFEOF
# Review Feedback for Lane ${lane^^}

## Rejected by Lane ${reviewer_lane^^} ($reviewer_agent)

**Reason:** $reason

## Full Review
$(cat "$review_file")
RFEOF
            fi
        done

        if $all_approved; then
            set_lane_status "$lane" "approved"
            hub_success "Lane ${lane^^}: ALL APPROVED"
        else
            set_lane_status "$lane" "needs_revision"
            hub_warn "Lane ${lane^^}: NEEDS REVISION (see $REVIEWS_DIR/${lane}_feedback.md)"
        fi

        reviewed=$((reviewed + 1))
    done

    if [[ $reviewed -eq 0 ]]; then
        hub_warn "No lanes ready for review (need status: done)"
    fi
}

# ── Command: rework ────────────────────────────────────────────────────────────

cmd_rework() {
    local lane="${1:-}"

    if [[ -z "$lane" ]]; then
        hub_error "Usage: hub.sh rework <lane>"
        hub_error "Lanes: a, b, c"
        exit 1
    fi

    lane=$(echo "$lane" | tr '[:upper:]' '[:lower:]')

    if [[ ! " $ALL_LANES " =~ " $lane " ]]; then
        hub_error "Invalid lane: $lane (use: a, b, c)"
        exit 1
    fi

    local status
    status=$(get_lane_status "$lane")

    if [[ "$status" != "needs_revision" ]]; then
        hub_error "Lane $lane status is '$status', expected 'needs_revision'"
        exit 1
    fi

    local wt_path
    wt_path=$(get_worktree_path "$lane")
    local agent="${LANE_AGENTS[$lane]}"
    local label="${LANE_LABELS[$lane]}"
    local feedback_file="$REVIEWS_DIR/${lane}_feedback.md"

    if [[ ! -f "$feedback_file" ]]; then
        hub_error "No feedback file found: $feedback_file"
        exit 1
    fi

    local feedback
    feedback=$(cat "$feedback_file")

    # Read original task
    local original_task=""
    if [[ -f "$wt_path/.hub-task.md" ]]; then
        original_task=$(cat "$wt_path/.hub-task.md")
    fi

    # Remove done sentinel
    rm -f "$wt_path/.agent-done"

    # Build rework prompt
    local prompt="You are working on Lane ${lane^^} ($label) of a multi-agent project.

Your previous work was REJECTED during cross-review. You need to address the feedback.

## Original Task
$original_task

## Review Feedback
$feedback

## Instructions
1. Read the feedback carefully
2. Fix the issues identified
3. Commit your changes
4. Run: touch .agent-done

Only modify files you own (see CLAUDE.md for ownership rules)."

    local log_file="$LOGS_DIR/lane_${lane}_rework_$(date +%Y%m%d_%H%M%S).log"

    lane_log "$lane" "Re-dispatching $agent with feedback..."

    # Spawn agent with full write permissions + auto-update on exit
    (
        cd "$wt_path"
        run_agent_work "$agent" "$prompt" "$log_file" "$DISPATCH_TIMEOUT"
        on_lane_exit "$lane"
    ) &
    local pid=$!

    set_lane_status "$lane" "working"
    set_lane_pid "$lane" "$pid"

    # Clear old reviews for this lane
    python3 -c "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
state['reviews'] = {k: v for k, v in state.get('reviews', {}).items()
                    if not k.startswith('${lane}_')}
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=4)
"

    lane_log "$lane" "Rework dispatched. PID: $pid, log: $log_file"
    hub_log "Monitor with: hub.sh status"
}

# ── Command: merge ─────────────────────────────────────────────────────────────

cmd_merge() {
    if [[ ! -f "$STATE_FILE" ]]; then
        hub_error "No hub state found. Run 'hub.sh setup' first."
        exit 1
    fi

    # Check all active lanes are approved
    local all_approved=true
    local active_lanes=""

    for lane in $ALL_LANES; do
        local status
        status=$(get_lane_status "$lane")

        case $status in
            idle|merged)
                continue
                ;;
            approved)
                active_lanes="$active_lanes $lane"
                ;;
            *)
                hub_error "Lane ${lane^^} status is '$status', not 'approved'"
                all_approved=false
                ;;
        esac
    done

    if ! $all_approved; then
        hub_error "All active lanes must be 'approved' before merge"
        hub_log "Run 'hub.sh review' for lanes with status 'done'"
        exit 1
    fi

    if [[ -z "$active_lanes" ]]; then
        hub_warn "No lanes to merge (all idle)"
        exit 0
    fi

    hub_log "Merging lanes:$active_lanes"

    # Merge order: contracts first, then B (ML), A (App), C (QA)
    local merge_order=""

    # Check for contract changes first
    for lane in $active_lanes; do
        local wt_path
        wt_path=$(get_worktree_path "$lane")
        local has_contracts
        has_contracts=$(git -C "$wt_path" diff main...HEAD --name-only 2>/dev/null | grep "^jarvis/contracts/" || true)
        if [[ -n "$has_contracts" ]]; then
            merge_order="$lane"
            hub_log "Lane ${lane^^} has contract changes, merging first"
            break
        fi
    done

    # Then standard order: b, a, c
    for lane in b a c; do
        if [[ " $active_lanes " =~ " $lane " ]] && [[ " $merge_order " != *" $lane "* ]] && [[ "$merge_order" != "$lane" ]]; then
            merge_order="$merge_order $lane"
        fi
    done

    merge_order=$(echo "$merge_order" | xargs)  # trim

    hub_log "Merge order: $merge_order"

    # Perform merges
    for lane in $merge_order; do
        local branch="${LANE_BRANCHES[$lane]}"
        local label="${LANE_LABELS[$lane]}"

        hub_log "Merging Lane ${lane^^} ($branch)..."

        if ! git -C "$REPO_ROOT" merge --no-ff "$branch" \
            -m "Merge lane-${lane}: $label (hub-spoke orchestration)"; then
            hub_error "Merge failed for Lane ${lane^^}. Resolve conflicts manually."
            exit 1
        fi

        set_lane_status "$lane" "merged"
        hub_success "Lane ${lane^^} merged"
    done

    # Run verification
    hub_log "Running make verify..."
    if (cd "$REPO_ROOT" && make verify); then
        hub_success "Verification passed"
    else
        hub_error "make verify failed after merge!"
        hub_error "Check test_results.txt for details"
        hub_warn "You may need to revert: git reset --hard HEAD~${#merge_order}"
        exit 1
    fi

    # Reset states
    for lane in $merge_order; do
        set_lane_status "$lane" "idle"
    done

    echo ""
    hub_success "All lanes merged and verified successfully"
}

# ── Command: summary ──────────────────────────────────────────────────────────

cmd_summary() {
    if [[ ! -f "$STATE_FILE" ]]; then
        hub_error "No hub state found."
        exit 1
    fi

    echo -e "${BOLD}Session Summary${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python3 -c "
import json, os
from datetime import datetime, timezone

with open('$STATE_FILE') as f:
    state = json.load(f)

# Count by status
lanes = state.get('lanes', {})
tasks = state.get('tasks', [])

lane_statuses = {}
for l in lanes.values():
    s = l.get('status', 'idle')
    lane_statuses[s] = lane_statuses.get(s, 0) + 1

task_statuses = {}
for t in tasks:
    s = t.get('status', 'unknown')
    task_statuses[s] = task_statuses.get(s, 0) + 1

print(f'Lanes: {len(lanes)} total', end='')
for s, c in sorted(lane_statuses.items()):
    print(f', {c} {s}', end='')
print()

print(f'Tasks: {len(tasks)} total', end='')
for s, c in sorted(task_statuses.items()):
    print(f', {c} {s}', end='')
print()
print()

# Agent usage breakdown
agent_counts = {}
agent_logs_size = {}
for l in lanes.values():
    a = l.get('agent', '?')
    agent_counts[a] = agent_counts.get(a, 0) + 1
for t in tasks:
    a = t.get('agent', '?')
    m = t.get('model', 'default')
    key = f'{a} ({m})' if m != 'default' else a
    agent_counts[key] = agent_counts.get(key, 0) + 1
    log = t.get('log', '')
    if log and os.path.exists(log):
        agent_logs_size[key] = agent_logs_size.get(key, 0) + os.path.getsize(log)

print('Agent Usage:')
for agent, count in sorted(agent_counts.items(), key=lambda x: -x[1]):
    size = agent_logs_size.get(agent, 0)
    size_str = f' ({size // 1024}KB logs)' if size > 0 else ''
    print(f'  {agent:<20} {count} tasks{size_str}')
print()

# Reviews
reviews = state.get('reviews', {})
if reviews:
    approves = sum(1 for v in reviews.values() if v.get('result') == 'approve')
    rejects = sum(1 for v in reviews.values() if v.get('result') == 'reject')
    print(f'Reviews: {len(reviews)} total ({approves} approved, {rejects} rejected)')
    print()

# Log sizes
total_log_size = 0
for f in os.listdir('$LOGS_DIR'):
    fp = os.path.join('$LOGS_DIR', f)
    if os.path.isfile(fp):
        total_log_size += os.path.getsize(fp)
print(f'Total log output: {total_log_size // 1024}KB across {len(os.listdir(\"$LOGS_DIR\"))} files')
" 2>/dev/null

    # Show event log tail
    if [[ -f "$LOGS_DIR/hub_events.log" ]]; then
        echo ""
        echo -e "${BOLD}Recent Events${NC}"
        echo "────────────────────────────────────────────────────────────────"
        tail -10 "$LOGS_DIR/hub_events.log"
    fi
}

# ── Command: teardown ──────────────────────────────────────────────────────────

cmd_teardown() {
    hub_log "Tearing down hub-spoke worktrees..."

    # Check for uncommitted changes
    local has_uncommitted=false
    for lane in $ALL_LANES; do
        if worktree_exists "$lane" && worktree_has_uncommitted "$lane"; then
            local wt_path
            wt_path=$(get_worktree_path "$lane")
            hub_warn "Lane ${lane^^} has uncommitted changes in $wt_path"
            has_uncommitted=true
        fi
    done

    if $has_uncommitted; then
        echo ""
        read -p "Uncommitted changes found. Stash them before removing? [y/N] " -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            for lane in $ALL_LANES; do
                if worktree_exists "$lane" && worktree_has_uncommitted "$lane"; then
                    local wt_path
                    wt_path=$(get_worktree_path "$lane")
                    git -C "$wt_path" stash push -m "hub teardown stash (lane ${lane^^})"
                    lane_log "$lane" "Changes stashed"
                fi
            done
        fi
    fi

    # Remove worktrees
    for lane in $ALL_LANES; do
        local wt_path
        wt_path=$(get_worktree_path "$lane")

        if worktree_exists "$lane"; then
            lane_log "$lane" "Removing worktree: $wt_path"
            git -C "$REPO_ROOT" worktree remove "$wt_path" --force 2>/dev/null || {
                hub_warn "Failed to remove $wt_path via git, removing manually"
                rm -rf "$wt_path"
                git -C "$REPO_ROOT" worktree prune
            }
        fi

        # Clean up branch if it has been merged
        local branch="${LANE_BRANCHES[$lane]}"
        local status
        status=$(get_lane_status "$lane" 2>/dev/null || echo "unknown")
        if [[ "$status" == "merged" ]] || [[ "$status" == "idle" ]]; then
            git -C "$REPO_ROOT" branch -d "$branch" 2>/dev/null && \
                lane_log "$lane" "Deleted branch: $branch" || true
        fi
    done

    # Clean up state
    if [[ -d "$HUB_DIR" ]]; then
        rm -rf "$HUB_DIR"
        hub_success "Removed state directory: $HUB_DIR"
    fi

    hub_success "Teardown complete"
}

# ── Main ───────────────────────────────────────────────────────────────────────

if [[ $# -eq 0 ]]; then
    usage
fi

COMMAND="$1"
shift

case "$COMMAND" in
    setup)
        cmd_setup
        ;;
    dispatch)
        cmd_dispatch "$@"
        ;;
    run)
        cmd_run "$@"
        ;;
    status)
        cmd_status
        ;;
    watch)
        cmd_watch "$@"
        ;;
    logs)
        cmd_logs "$@"
        ;;
    review)
        cmd_review
        ;;
    rework)
        cmd_rework "$@"
        ;;
    merge)
        cmd_merge
        ;;
    summary)
        cmd_summary
        ;;
    teardown)
        cmd_teardown
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        hub_error "Unknown command: $COMMAND"
        usage
        ;;
esac
