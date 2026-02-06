#!/bin/bash
# Git Worktree Management Script
# Manages multiple worktrees with CLI tool assignments

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="${HOME}/.jarvis-worktrees.json"
WORKTREE_BASE_DIR="$(dirname "$REPO_ROOT")"

# Default CLI tools
VALID_CLIS=("claude" "gemini" "kimi" "opencode" "codex" "agent")

# Initialize config file if it doesn't exist
init_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        cat > "$CONFIG_FILE" <<EOF
{
  "worktrees": {}
}
EOF
    fi
}

# Read config
read_config() {
    if [[ -f "$CONFIG_FILE" ]]; then
        cat "$CONFIG_FILE"
    else
        echo '{"worktrees": {}}'
    fi
}

# Write config
write_config() {
    echo "$1" | jq '.' > "$CONFIG_FILE"
}

# Get worktree name from branch
get_worktree_name() {
    local branch="$1"
    echo "jarvis-${branch#feature/}"
}

# Create a new worktree
create_worktree() {
    local branch="$1"
    local cli="${2:-}"
    
    if [[ -z "$cli" ]]; then
        echo "Error: CLI tool required"
        echo "Usage: $0 create <branch> <cli>"
        echo "Valid CLIs: ${VALID_CLIS[*]}"
        exit 1
    fi
    
    # Validate CLI
    local valid=false
    for valid_cli in "${VALID_CLIS[@]}"; do
        if [[ "$cli" == "$valid_cli" ]]; then
            valid=true
            break
        fi
    done
    
    if [[ "$valid" == false ]]; then
        echo "Error: Invalid CLI '$cli'"
        echo "Valid CLIs: ${VALID_CLIS[*]}"
        exit 1
    fi
    
    local worktree_name=$(get_worktree_name "$branch")
    local worktree_path="${WORKTREE_BASE_DIR}/${worktree_name}"
    
    # Check if worktree already exists
    if [[ -d "$worktree_path" ]]; then
        echo "Error: Worktree already exists at $worktree_path"
        exit 1
    fi
    
    # Check if branch already exists
    if git show-ref --verify --quiet "refs/heads/$branch"; then
        echo "Branch $branch already exists, using existing branch"
        git worktree add "$worktree_path" "$branch"
    else
        echo "Creating new branch $branch"
        git worktree add "$worktree_path" -b "$branch"
    fi
    
    # Update config
    local config=$(read_config)
    local new_config=$(echo "$config" | jq --arg branch "$branch" \
        --arg path "$worktree_path" \
        --arg cli "$cli" \
        '.worktrees[$branch] = {path: $path, cli: $cli, branch: $branch}')
    write_config "$new_config"
    
    echo "✓ Created worktree: $worktree_path"
    echo "  Branch: $branch"
    echo "  CLI: $cli"
    echo ""
    echo "To use this worktree:"
    echo "  cd $worktree_path"
}

# List all worktrees
list_worktrees() {
    echo "Git Worktrees:"
    echo "=============="
    git worktree list
    echo ""
    
    local config=$(read_config)
    local worktree_count=$(echo "$config" | jq '.worktrees | length')
    
    if [[ "$worktree_count" -gt 0 ]]; then
        echo "CLI Assignments:"
        echo "================"
        echo "$config" | jq -r '.worktrees | to_entries[] | "\(.key): \(.value.cli) @ \(.value.path)"'
    else
        echo "No CLI assignments tracked."
    fi
}

# Assign CLI to existing worktree
assign_cli() {
    local branch="$1"
    local cli="${2:-}"
    
    if [[ -z "$cli" ]]; then
        echo "Error: CLI tool required"
        echo "Usage: $0 assign <branch> <cli>"
        exit 1
    fi
    
    # Validate CLI
    local valid=false
    for valid_cli in "${VALID_CLIS[@]}"; do
        if [[ "$cli" == "$valid_cli" ]]; then
            valid=true
            break
        fi
    done
    
    if [[ "$valid" == false ]]; then
        echo "Error: Invalid CLI '$cli'"
        echo "Valid CLIs: ${VALID_CLIS[*]}"
        exit 1
    fi
    
    local config=$(read_config)
    local worktree_path=$(git worktree list | grep "$branch" | awk '{print $1}' || echo "")
    
    if [[ -z "$worktree_path" ]]; then
        echo "Error: No worktree found for branch $branch"
        exit 1
    fi
    
    local new_config=$(echo "$config" | jq --arg branch "$branch" \
        --arg path "$worktree_path" \
        --arg cli "$cli" \
        '.worktrees[$branch] = {path: $path, cli: $cli, branch: $branch}')
    write_config "$new_config"
    
    echo "✓ Assigned CLI '$cli' to branch '$branch'"
}

# Remove a worktree
remove_worktree() {
    local branch="$1"
    local worktree_name=$(get_worktree_name "$branch")
    local worktree_path="${WORKTREE_BASE_DIR}/${worktree_name}"
    
    if [[ ! -d "$worktree_path" ]]; then
        echo "Error: Worktree not found at $worktree_path"
        exit 1
    fi
    
    # Remove from git
    cd "$REPO_ROOT"
    git worktree remove "$worktree_path"
    
    # Remove from config
    local config=$(read_config)
    local new_config=$(echo "$config" | jq --arg branch "$branch" 'del(.worktrees[$branch])')
    write_config "$new_config"
    
    echo "✓ Removed worktree: $worktree_path"
    echo ""
    echo "Note: Branch '$branch' still exists. Delete it with:"
    echo "  git branch -d $branch"
}

# Get CLI for a worktree
get_cli() {
    local branch="$1"
    local config=$(read_config)
    local cli=$(echo "$config" | jq -r --arg branch "$branch" '.worktrees[$branch].cli // empty')
    
    if [[ -z "$cli" || "$cli" == "null" ]]; then
        echo "No CLI assigned to branch $branch"
        exit 1
    else
        echo "$cli"
    fi
}

# Show usage
usage() {
    cat <<EOF
Git Worktree Management Script

Usage: $0 <command> [args...]

Commands:
  create <branch> <cli>    Create a new worktree for branch with CLI assignment
  list                     List all worktrees and CLI assignments
  assign <branch> <cli>    Assign CLI tool to existing worktree branch
  remove <branch>          Remove a worktree
  get-cli <branch>         Get CLI assignment for a branch
  help                     Show this help message

Valid CLI tools: ${VALID_CLIS[*]}

Examples:
  $0 create feature/api-improvements claude
  $0 create feature/ui-redesign gemini
  $0 list
  $0 assign feature/api-improvements gemini
  $0 remove feature/api-improvements
  $0 get-cli feature/api-improvements

Config file: $CONFIG_FILE
Worktree base: $WORKTREE_BASE_DIR
EOF
}

# Main
init_config

case "${1:-help}" in
    create)
        if [[ $# -lt 3 ]]; then
            echo "Error: Branch and CLI required"
            usage
            exit 1
        fi
        create_worktree "$2" "$3"
        ;;
    list)
        list_worktrees
        ;;
    assign)
        if [[ $# -lt 3 ]]; then
            echo "Error: Branch and CLI required"
            usage
            exit 1
        fi
        assign_cli "$2" "$3"
        ;;
    remove)
        if [[ $# -lt 2 ]]; then
            echo "Error: Branch required"
            usage
            exit 1
        fi
        remove_worktree "$2"
        ;;
    get-cli)
        if [[ $# -lt 2 ]]; then
            echo "Error: Branch required"
            usage
            exit 1
        fi
        get_cli "$2"
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Error: Unknown command '$1'"
        usage
        exit 1
        ;;
esac
