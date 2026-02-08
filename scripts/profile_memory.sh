#!/bin/bash
# Quick memory profiling for data preparation
# Samples memory usage every 2 seconds

echo "Starting memory profiling for prepare_dailydialog_data.py"
echo "Timestamp,RSS_MB,VSZ_MB" > /tmp/memory_profile.csv

# Start the Python script in background
uv run python scripts/prepare_dailydialog_data.py &
PID=$!

echo "Monitoring PID: $PID"

# Sample memory every 2 seconds
while kill -0 $PID 2>/dev/null; do
    TIMESTAMP=$(date +%s)
    # Get RSS (real memory) and VSZ (virtual memory) in KB
    MEM=$(ps -p $PID -o rss=,vsz= | awk '{print $1/1024 "," $2/1024}')
    echo "$TIMESTAMP,$MEM" >> /tmp/memory_profile.csv
    sleep 2
done

echo ""
echo "Process completed. Memory profile saved to /tmp/memory_profile.csv"
echo ""
echo "Peak memory usage:"
awk -F',' 'NR>1 {if($2>max) max=$2} END {printf "  RSS: %.1f MB\n", max}' /tmp/memory_profile.csv
awk -F',' 'NR>1 {if($3>max) max=$3} END {printf "  VSZ: %.1f MB\n", max}' /tmp/memory_profile.csv
