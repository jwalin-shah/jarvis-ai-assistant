#!/bin/bash
# Run model comparison in background while Optuna continues

set -e

cd "$(dirname "$0")/.."

echo "🚀 Starting model comparison in background..."
echo "📝 Output will be in: model_comparison.log"
echo ""

# Check if lightgbm/xgboost are installed
echo "Checking optional dependencies..."
uv pip list | grep -i lightgbm || echo "⚠️  LightGBM not installed (optional)"
uv pip list | grep -i xgboost || echo "⚠️  XGBoost not installed (optional)"
echo ""

# Run in background with output to log file
nohup uv run python scripts/model_comparison.py > model_comparison.log 2>&1 &
PID=$!

echo "✓ Started model comparison (PID: $PID)"
echo ""
echo "Monitor with:"
echo "  tail -f model_comparison.log"
echo ""
echo "Check status:"
echo "  ps aux | grep $PID"
echo ""
echo "Memory usage:"
echo "  top -pid $PID"
echo ""

# Save PID for easy stopping
echo $PID > model_comparison.pid
echo "To stop: kill $(cat model_comparison.pid)"
