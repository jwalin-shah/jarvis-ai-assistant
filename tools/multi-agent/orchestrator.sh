#!/opt/homebrew/bin/bash
# Multi-Agent Orchestrator
# Run prompts across multiple AI CLIs and have them collaborate

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/outputs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Available agents (edit this list based on what you have installed)
ALL_AGENTS="claude codex gemini opencode kimi"

# Default models per agent (override with --models or ORCHESTRATOR_MODELS env var)
# Format: agent=model,agent=model
declare -A AGENT_MODELS=(
    [claude]=""                              # uses default (your subscription)
    [codex]=""                               # uses default (gpt-5.3-codex from config)
    [gemini]=""                              # uses default (gemini-2.5-pro)
    [opencode]="opencode/glm-4.7-free"       # free tier
    [kimi]=""                                # uses default (kimi-k2.5)
)

# Defaults
MODE="parallel"
ROUNDS=2
AGENTS="$ALL_AGENTS"
ORDER=""
QUIET=false
TIMEOUT=120

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] "prompt"

Multi-agent orchestrator for AI CLI tools.

MODES:
  parallel    Run same prompt on all agents, compare outputs (default)
  debate      Agents critique each other's responses over multiple rounds
  relay       Chain agents: each builds on previous output

OPTIONS:
  -m, --mode MODE       Mode: parallel, debate, relay (default: parallel)
  -a, --agents LIST     Comma-separated agents to use (default: all)
  -r, --rounds N        Number of rounds for debate mode (default: 2)
  -o, --order LIST      Agent order for relay mode (default: all agents)
  -t, --timeout SECS    Timeout per agent in seconds (default: 120)
  --models LIST         Override models: agent=model,agent=model
  -q, --quiet           Suppress progress output
  -h, --help            Show this help

EXAMPLES:
  $(basename "$0") "Explain async/await in JavaScript"
  $(basename "$0") -m debate -r 3 "Monolith vs microservices?"
  $(basename "$0") -m relay -o claude,codex,gemini "Write a REST API"
  $(basename "$0") -a claude,gemini "Compare Python and Go"
  $(basename "$0") --models "opencode=anthropic/claude-sonnet-4-5,gemini=gemini-2.0-flash" "Question"

AVAILABLE AGENTS: $ALL_AGENTS

DEFAULT MODELS:
  claude    = (subscription default)
  codex     = (subscription default)
  gemini    = (subscription default)
  opencode  = opencode/glm-4.7-free
  kimi      = (subscription default)
EOF
    exit 0
}

log() {
    if [ "$QUIET" = false ]; then
        echo -e "${CYAN}[orchestrator]${NC} $1"
    fi
}

log_agent() {
    local agent=$1
    local msg=$2
    if [ "$QUIET" = false ]; then
        echo -e "${GREEN}[$agent]${NC} $msg"
    fi
}

log_error() {
    echo -e "${RED}[error]${NC} $1" >&2
}

# Clean agent output (strip noise)
clean_agent_output() {
    local agent=$1
    local raw_file=$2

    case $agent in
        codex)
            # Extract response between "codex" marker and "tokens used"
            sed -n '/^codex$/,/^tokens used$/ p' "$raw_file" | grep -v "^codex$" | grep -v "^tokens used$"
            ;;
        gemini)
            # Strip node/gemini stderr noise and rate limit messages
            cat "$raw_file" | \
                grep -v "DeprecationWarning" | \
                grep -v "punycode" | \
                grep -v "^(node:" | \
                grep -v "trace-deprecation" | \
                grep -v "Loaded cached" | \
                grep -v "Session cleanup" | \
                grep -v "Hook registry" | \
                grep -v "^(Use \`node" | \
                sed 's/Attempt [0-9]* failed:.*//g' | \
                sed 's/Retrying after.*//g' | \
                sed '/^$/d'
            ;;
        *)
            cat "$raw_file"
            ;;
    esac
}

# Run a single agent with a prompt
run_agent() {
    local agent=$1
    local prompt=$2
    local output_file=$3
    local model="${AGENT_MODELS[$agent]:-}"

    local model_info=""
    if [ -n "$model" ]; then
        model_info=" (model: $model)"
    fi
    log_agent "$agent" "Starting...${model_info}"

    # Capture raw output
    local raw_file="${output_file%.md}.raw"

    case $agent in
        claude)
            if [ -n "$model" ]; then
                timeout "$TIMEOUT" claude -p "$prompt" --print --model "$model" > "$raw_file" 2>&1 || true
            else
                timeout "$TIMEOUT" claude -p "$prompt" --print > "$raw_file" 2>&1 || true
            fi
            ;;
        codex)
            if [ -n "$model" ]; then
                timeout "$TIMEOUT" codex e --model "$model" "$prompt" > "$raw_file" 2>&1 || true
            else
                timeout "$TIMEOUT" codex e "$prompt" > "$raw_file" 2>&1 || true
            fi
            ;;
        gemini)
            if [ -n "$model" ]; then
                timeout "$TIMEOUT" gemini -p "$prompt" --model "$model" > "$raw_file" 2>&1 || true
            else
                timeout "$TIMEOUT" gemini -p "$prompt" > "$raw_file" 2>&1 || true
            fi
            ;;
        opencode)
            if [ -n "$model" ]; then
                timeout "$TIMEOUT" opencode run -m "$model" "$prompt" > "$raw_file" 2>&1 || true
            else
                timeout "$TIMEOUT" opencode run "$prompt" > "$raw_file" 2>&1 || true
            fi
            ;;
        kimi)
            if [ -n "$model" ]; then
                timeout "$TIMEOUT" kimi --quiet -p "$prompt" --model "$model" > "$raw_file" 2>&1 || true
            else
                timeout "$TIMEOUT" kimi --quiet -p "$prompt" > "$raw_file" 2>&1 || true
            fi
            ;;
        *)
            log_error "Unknown agent: $agent"
            return 1
            ;;
    esac

    # Clean output
    if [ -s "$raw_file" ]; then
        clean_agent_output "$agent" "$raw_file" > "$output_file"
        log_agent "$agent" "Done ($(wc -l < "$output_file" | tr -d ' ') lines)"
    else
        touch "$output_file"
        log_agent "$agent" "No output or failed"
    fi
}

# Mode: Parallel - run all agents with same prompt
mode_parallel() {
    local prompt=$1
    local run_dir="$OUTPUT_DIR/parallel_$TIMESTAMP"
    mkdir -p "$run_dir"

    log "Mode: PARALLEL"
    log "Prompt: $prompt"
    log "Agents: $AGENTS"
    log "Output: $run_dir"
    echo ""

    # Save prompt
    echo "$prompt" > "$run_dir/prompt.txt"

    # Run all agents in parallel
    local pids=()
    for agent in $AGENTS; do
        run_agent "$agent" "$prompt" "$run_dir/${agent}.md" &
        pids+=($!)
    done

    # Wait for all to complete
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    echo ""
    log "All agents complete. Generating comparison..."

    # Generate combined output
    {
        echo "# Multi-Agent Comparison"
        echo "**Prompt:** $prompt"
        echo "**Timestamp:** $(date)"
        echo ""

        for agent in $AGENTS; do
            echo "---"
            echo "## $agent"
            echo ""
            if [ -s "$run_dir/${agent}.md" ]; then
                cat "$run_dir/${agent}.md"
            else
                echo "*No output*"
            fi
            echo ""
        done
    } > "$run_dir/comparison.md"

    log "Results saved to: $run_dir/comparison.md"
    echo ""
    echo "=== Quick Summary ==="
    for agent in $AGENTS; do
        if [ -s "$run_dir/${agent}.md" ]; then
            echo -e "${GREEN}$agent${NC}: $(wc -c < "$run_dir/${agent}.md" | tr -d ' ') bytes"
        else
            echo -e "${RED}$agent${NC}: no output"
        fi
    done
}

# Mode: Debate - agents critique each other
mode_debate() {
    local prompt=$1
    local run_dir="$OUTPUT_DIR/debate_$TIMESTAMP"
    mkdir -p "$run_dir"

    log "Mode: DEBATE"
    log "Prompt: $prompt"
    log "Agents: $AGENTS"
    log "Rounds: $ROUNDS"
    log "Output: $run_dir"
    echo ""

    echo "$prompt" > "$run_dir/prompt.txt"

    # Round 1: Initial responses
    log "=== Round 1: Initial Responses ==="
    mkdir -p "$run_dir/round1"

    local pids=()
    for agent in $AGENTS; do
        run_agent "$agent" "$prompt" "$run_dir/round1/${agent}.md" &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    # Subsequent rounds: Debate
    for ((round=2; round<=ROUNDS; round++)); do
        echo ""
        log "=== Round $round: Critique & Refine ==="
        mkdir -p "$run_dir/round${round}"

        # Build context from previous round
        local prev_round=$((round - 1))
        local context_file="$run_dir/round${round}/context.txt"

        {
            echo "Original question: $prompt"
            echo ""
            echo "Here's what each AI said in round $prev_round:"
            echo ""

            for agent in $AGENTS; do
                local prev_file="$run_dir/round${prev_round}/${agent}.md"
                if [ -s "$prev_file" ]; then
                    echo "--- $(echo "$agent" | tr '[:lower:]' '[:upper:]') said ---"
                    cat "$prev_file"
                    echo ""
                fi
            done

            echo ""
            echo "Now, critique the other responses. Point out flaws, missing points, or things you disagree with. Then provide your refined answer."
        } > "$context_file"

        local context
        context=$(cat "$context_file")

        # Run debate round in parallel
        pids=()
        for agent in $AGENTS; do
            run_agent "$agent" "$context" "$run_dir/round${round}/${agent}.md" &
            pids+=($!)
        done

        for pid in "${pids[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
    done

    # Generate debate summary
    {
        echo "# Multi-Agent Debate"
        echo "**Prompt:** $prompt"
        echo "**Rounds:** $ROUNDS"
        echo "**Timestamp:** $(date)"
        echo ""

        for ((round=1; round<=ROUNDS; round++)); do
            echo "---"
            echo "# Round $round"
            echo ""
            for agent in $AGENTS; do
                echo "## $agent"
                echo ""
                local file="$run_dir/round${round}/${agent}.md"
                if [ -s "$file" ]; then
                    cat "$file"
                else
                    echo "*No output*"
                fi
                echo ""
            done
        done
    } > "$run_dir/debate.md"

    log "Debate saved to: $run_dir/debate.md"
}

# Mode: Relay - chain agents
mode_relay() {
    local prompt=$1
    local run_dir="$OUTPUT_DIR/relay_$TIMESTAMP"
    mkdir -p "$run_dir"

    local agent_list=${ORDER:-$AGENTS}

    log "Mode: RELAY"
    log "Prompt: $prompt"
    log "Order: $agent_list"
    log "Output: $run_dir"
    echo ""

    echo "$prompt" > "$run_dir/prompt.txt"

    local current_context="$prompt"
    local step=1

    for agent in $agent_list; do
        log "=== Step $step: $agent ==="

        local step_prompt
        if [ $step -eq 1 ]; then
            step_prompt="$current_context"
        else
            step_prompt="Original task: $prompt

Previous work so far:
$current_context

Your job: Review the above, improve it, fix any issues, and continue the work. Build on what's already done."
        fi

        run_agent "$agent" "$step_prompt" "$run_dir/step${step}_${agent}.md"

        if [ -s "$run_dir/step${step}_${agent}.md" ]; then
            current_context=$(cat "$run_dir/step${step}_${agent}.md")
        fi

        ((step++))
        echo ""
    done

    # Save final output
    echo "$current_context" > "$run_dir/final.md"

    # Generate relay summary
    {
        echo "# Multi-Agent Relay"
        echo "**Prompt:** $prompt"
        echo "**Order:** $agent_list"
        echo "**Timestamp:** $(date)"
        echo ""

        step=1
        for agent in $agent_list; do
            echo "---"
            echo "## Step $step: $agent"
            echo ""
            cat "$run_dir/step${step}_${agent}.md" 2>/dev/null || echo "*No output*"
            echo ""
            ((step++))
        done

        echo "---"
        echo "# Final Output"
        echo ""
        cat "$run_dir/final.md"
    } > "$run_dir/relay.md"

    log "Relay saved to: $run_dir/relay.md"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -a|--agents)
            AGENTS="${2//,/ }"
            shift 2
            ;;
        -r|--rounds)
            ROUNDS="$2"
            shift 2
            ;;
        -o|--order)
            ORDER="${2//,/ }"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --models)
            # Parse model overrides: agent=model,agent=model
            IFS=',' read -ra MODEL_OVERRIDES <<< "$2"
            for override in "${MODEL_OVERRIDES[@]}"; do
                agent_name="${override%%=*}"
                model_name="${override#*=}"
                AGENT_MODELS[$agent_name]="$model_name"
            done
            shift 2
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        -*)
            log_error "Unknown option: $1"
            exit 1
            ;;
        *)
            PROMPT="$1"
            shift
            ;;
    esac
done

if [ -z "$PROMPT" ]; then
    log_error "No prompt provided"
    usage
fi

# Validate mode
case $MODE in
    parallel|debate|relay)
        ;;
    *)
        log_error "Invalid mode: $MODE (use: parallel, debate, relay)"
        exit 1
        ;;
esac

# Run selected mode
case $MODE in
    parallel)
        mode_parallel "$PROMPT"
        ;;
    debate)
        mode_debate "$PROMPT"
        ;;
    relay)
        mode_relay "$PROMPT"
        ;;
esac
