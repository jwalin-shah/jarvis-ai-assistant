#!/bin/bash
# Quick status check for running training process

PID=$(ps aux | grep "train_category_svm.py" | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$PID" ]; then
    echo "✓ Training completed or not running"
    exit 0
fi

# Get process info
RUNTIME=$(ps -p $PID -o etime= | tr -d ' ')
CPU=$(ps -p $PID -o %cpu= | tr -d ' ')
RSS_KB=$(ps -p $PID -o rss= | tr -d ' ')
RSS_MB=$((RSS_KB / 1024))

# Get memory pressure
PRESSURE=$(sysctl -n vm.memory_pressure 2>/dev/null || echo "N/A")

echo "Training still running..."
echo "  PID: $PID"
echo "  Runtime: $RUNTIME"
echo "  CPU: ${CPU}%"
echo "  RAM: ${RSS_MB}MB"
echo "  Memory pressure: $PRESSURE (0=good, >50=warning)"
echo ""
echo "Expected total time: 7-10 minutes"
echo "Check again in 1-2 minutes, or run: watch -n 10 ./scripts/check_training_status.sh"
