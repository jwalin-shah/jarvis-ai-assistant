#!/bin/bash
# Run all classifier optimization experiment phases sequentially.
#
# Usage:
#   ./experiments/run_experiment.sh           # Full experiment
#   ./experiments/run_experiment.sh --quick   # Quick mode (smaller grid)
#
# Phases:
#   1. prepare_data   - Split data, auto-label, compute embeddings
#   2. coarse_search  - Grid search over sizes and hyperparameters
#   3. fine_search    - Fine-grained search around best region
#   4. final_eval     - Evaluate on held-out test set

set -e  # Exit on first error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
    echo -e "${YELLOW}Running in QUICK mode (reduced grid for faster iteration)${NC}"
fi

# Timestamp function
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# Duration function
duration() {
    local start=$1
    local end=$2
    local diff=$((end - start))
    local mins=$((diff / 60))
    local secs=$((diff % 60))
    echo "${mins}m ${secs}s"
}

# Log function
log() {
    echo -e "${BLUE}[$(timestamp)]${NC} $1"
}

# Success function
success() {
    echo -e "${GREEN}[$(timestamp)] ✓ $1${NC}"
}

# Error function
error() {
    echo -e "${RED}[$(timestamp)] ✗ $1${NC}"
}

# Header
echo ""
echo "============================================================"
echo "  CLASSIFIER OPTIMIZATION EXPERIMENT"
echo "  Started: $(timestamp)"
echo "============================================================"
echo ""

EXPERIMENT_START=$(date +%s)

# ============================================================
# Phase 1: Data Preparation
# ============================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
log "PHASE 1: Data Preparation"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

PHASE1_START=$(date +%s)

if $QUICK_MODE; then
    uv run python -m experiments.scripts.prepare_data --confidence-threshold 0.95
else
    uv run python -m experiments.scripts.prepare_data
fi

PHASE1_END=$(date +%s)
success "Phase 1 complete ($(duration $PHASE1_START $PHASE1_END))"
echo ""

# ============================================================
# Phase 2: Coarse Search
# ============================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
log "PHASE 2: Coarse Hyperparameter Search"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

PHASE2_START=$(date +%s)

if $QUICK_MODE; then
    # Reduced grid: 3 sizes, 3 C values, 1 gamma
    uv run python -m experiments.scripts.coarse_search \
        --sizes 3000 10000 20000 \
        --c-values 1 5 20 \
        --gamma-values scale \
        --n-folds 3
else
    uv run python -m experiments.scripts.coarse_search
fi

PHASE2_END=$(date +%s)
success "Phase 2 complete ($(duration $PHASE2_START $PHASE2_END))"
echo ""

# ============================================================
# Phase 3: Fine Search
# ============================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
log "PHASE 3: Fine-Grained Search"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

PHASE3_START=$(date +%s)

if $QUICK_MODE; then
    # Reduced: 2k step, 3 C values, 1 gamma, 3 folds
    uv run python -m experiments.scripts.fine_search \
        --step 2000 \
        --c-values 1 5 20 \
        --gamma-values scale \
        --n-folds 3
else
    uv run python -m experiments.scripts.fine_search
fi

PHASE3_END=$(date +%s)
success "Phase 3 complete ($(duration $PHASE3_START $PHASE3_END))"
echo ""

# ============================================================
# Phase 4: Final Evaluation
# ============================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
log "PHASE 4: Final Evaluation on Test Set"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

PHASE4_START=$(date +%s)

uv run python -m experiments.scripts.final_eval

PHASE4_END=$(date +%s)
success "Phase 4 complete ($(duration $PHASE4_START $PHASE4_END))"
echo ""

# ============================================================
# Summary
# ============================================================
EXPERIMENT_END=$(date +%s)

echo "============================================================"
echo -e "${GREEN}  EXPERIMENT COMPLETE${NC}"
echo "============================================================"
echo ""
echo "  Phase 1 (prepare_data):  $(duration $PHASE1_START $PHASE1_END)"
echo "  Phase 2 (coarse_search): $(duration $PHASE2_START $PHASE2_END)"
echo "  Phase 3 (fine_search):   $(duration $PHASE3_START $PHASE3_END)"
echo "  Phase 4 (final_eval):    $(duration $PHASE4_START $PHASE4_END)"
echo "  ─────────────────────────────────────"
echo "  Total:                   $(duration $EXPERIMENT_START $EXPERIMENT_END)"
echo ""
echo "  Results saved to:"
echo "    - experiments/results/coarse_search.json"
echo "    - experiments/results/fine_search.json"
echo "    - experiments/results/final_evaluation.json"
echo ""
echo "  Model saved to:"
echo "    - experiments/models/response_v2/svm.pkl"
echo "    - experiments/models/response_v2/config.json"
echo ""
echo "============================================================"
