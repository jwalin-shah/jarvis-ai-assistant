#!/bin/bash
# scripts/overnight_eval.sh
# Run all benchmarks sequentially (8GB safe)

set -e  # Exit on error

RESULTS_DIR="results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "🌙 JARVIS Overnight Evaluation Suite"
echo "======================================"
echo "Started: $(date)"
echo "Results: $RESULTS_DIR"
echo ""

# Pre-flight checks
echo "📋 Pre-flight checks..."
python -c "import mlx; print(f'MLX version: {mlx.__version__}')"
python -c "import psutil; m = psutil.virtual_memory(); print(f'Available RAM: {m.available / 1e9:.1f}GB')"
echo ""

# Memory profiling (models loaded/unloaded one at a time)
echo "📊 [1/4] Memory Profiling..."
python -m benchmarks.memory.run \
    --output "$RESULTS_DIR/memory.json" \
    2>&1 | tee "$RESULTS_DIR/memory.log"
echo ""

# HHEM benchmark (batched for efficiency)
echo "🔍 [2/4] HHEM Hallucination Benchmark..."
python -m benchmarks.hallucination.run \
    --output "$RESULTS_DIR/hhem.json" \
    2>&1 | tee "$RESULTS_DIR/hhem.log"
echo ""

# Template coverage (lightweight, embedding-based)
echo "📋 [3/4] Template Coverage Analysis..."
python -m benchmarks.coverage.run \
    --output "$RESULTS_DIR/coverage.json" \
    2>&1 | tee "$RESULTS_DIR/coverage.log"
echo ""

# Latency benchmarks (cold/warm/hot)
echo "⏱️ [4/4] Latency Benchmarks..."
python -m benchmarks.latency.run \
    --output "$RESULTS_DIR/latency.json" \
    2>&1 | tee "$RESULTS_DIR/latency.log"
echo ""

# Generate report
echo "📈 Generating report..."
python scripts/generate_report.py \
    --results-dir "$RESULTS_DIR" \
    --output docs/BENCHMARKS.md
echo ""

echo "======================================"
echo "✅ Completed: $(date)"
echo "Results: $RESULTS_DIR"
echo "Report: docs/BENCHMARKS.md"
