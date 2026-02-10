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

# ── Usage ──────────────────────────────────────────────────────────────────────

usage() {
    cat << 'EOF'
Usage: hub.sh <command> [args]

Commands:
  setup                  Create worktrees, state directory, per-lane CLAUDE.md
  dispatch <task-file>   Parse task file and spawn agents into worktrees
  status                 Show lane status, PIDs, last commits, reviews
  review                 Cross-review completed lanes (ownership + agent review)
  rework <lane>          Re-dispatch a lane with rejection feedback
  merge                  Merge all approved lanes to main (runs make verify)
  teardown               Remove worktrees and clean up state

Options:
  -h, --help             Show this help

Examples:
  hub.sh setup
  hub.sh dispatch tasks/my-task.md
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

        # Spawn agent in background
        (
            cd "$wt_path"
            case $agent in
                claude)
                    timeout "$DISPATCH_TIMEOUT" claude -p "$prompt" --print > "$log_file" 2>&1 || true
                    ;;
                codex)
                    timeout "$DISPATCH_TIMEOUT" codex e "$prompt" > "$log_file" 2>&1 || true
                    ;;
                gemini)
                    timeout "$DISPATCH_TIMEOUT" gemini -p "$prompt" > "$log_file" 2>&1 || true
                    ;;
                opencode)
                    timeout "$DISPATCH_TIMEOUT" opencode run "$prompt" > "$log_file" 2>&1 || true
                    ;;
                kimi)
                    timeout "$DISPATCH_TIMEOUT" kimi --quiet -p "$prompt" > "$log_file" 2>&1 || true
                    ;;
            esac
        ) &
        local pid=$!

        set_lane_status "$lane" "working"
        set_lane_pid "$lane" "$pid"

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

# ── Command: status ────────────────────────────────────────────────────────────

cmd_status() {
    if [[ ! -f "$STATE_FILE" ]]; then
        hub_error "No hub state found. Run 'hub.sh setup' first."
        exit 1
    fi

    echo -e "${BOLD}Hub Status${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf "%-8s %-10s %-10s %-8s %-30s %s\n" "LANE" "STATUS" "AGENT" "PID" "LAST COMMIT" "DONE?"
    echo "────────────────────────────────────────────────────────────────"

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

        printf "%-8s %-20b %-10s %-8s %-30s %s\n" \
            "${lane^^}" "$status_colored" "$agent" "$pid_info" "$last_commit" "$done_flag"
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

        # Save diff
        echo "$diff_content" > "$REVIEWS_DIR/lane_${lane}_diff.patch"

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
            review_prompt="${review_prompt//\{DIFF_CONTENT\}/$diff_content}"

            lane_log "$lane" "Requesting review from Lane ${reviewer_lane^^} ($reviewer_agent)..."

            # Run reviewer agent
            local reviewer_wt
            reviewer_wt=$(get_worktree_path "$reviewer_lane")

            (
                cd "$reviewer_wt"
                case $reviewer_agent in
                    claude)
                        timeout "$REVIEW_TIMEOUT" claude -p "$review_prompt" --print > "$review_file" 2>&1 || true
                        ;;
                    codex)
                        timeout "$REVIEW_TIMEOUT" codex e "$review_prompt" > "$review_file" 2>&1 || true
                        ;;
                    gemini)
                        timeout "$REVIEW_TIMEOUT" gemini -p "$review_prompt" > "$review_file" 2>&1 || true
                        ;;
                    opencode)
                        timeout "$REVIEW_TIMEOUT" opencode run "$review_prompt" > "$review_file" 2>&1 || true
                        ;;
                    kimi)
                        timeout "$REVIEW_TIMEOUT" kimi --quiet -p "$review_prompt" > "$review_file" 2>&1 || true
                        ;;
                esac
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

    # Spawn agent
    (
        cd "$wt_path"
        case $agent in
            claude)
                timeout "$DISPATCH_TIMEOUT" claude -p "$prompt" --print > "$log_file" 2>&1 || true
                ;;
            codex)
                timeout "$DISPATCH_TIMEOUT" codex e "$prompt" > "$log_file" 2>&1 || true
                ;;
            gemini)
                timeout "$DISPATCH_TIMEOUT" gemini -p "$prompt" > "$log_file" 2>&1 || true
                ;;
            opencode)
                timeout "$DISPATCH_TIMEOUT" opencode run "$prompt" > "$log_file" 2>&1 || true
                ;;
            kimi)
                timeout "$DISPATCH_TIMEOUT" kimi --quiet -p "$prompt" > "$log_file" 2>&1 || true
                ;;
        esac
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
    status)
        cmd_status
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
