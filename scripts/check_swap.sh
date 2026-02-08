#!/bin/bash
# Check swap usage during process

PID=$1

if [ -z "$PID" ]; then
    echo "Usage: $0 <pid>"
    exit 1
fi

echo "Monitoring swap for PID $PID"
echo "Press Ctrl+C to stop"
echo ""
echo "Time       RSS(MB)  Swap(MB)  Status"
echo "----------------------------------------"

while kill -0 $PID 2>/dev/null; do
    # Get memory info from ps
    MEM_INFO=$(ps -o pid=,rss=,vsz= -p $PID 2>/dev/null)
    RSS_KB=$(echo $MEM_INFO | awk '{print $2}')
    RSS_MB=$((RSS_KB / 1024))

    # Check system-wide swap
    SWAP_INFO=$(sysctl vm.swapusage 2>/dev/null | grep -o 'used = [0-9.]*[GM]')
    SWAP_USED=$(echo $SWAP_INFO | awk '{print $3}')

    # Status
    if [ $RSS_MB -lt 500 ]; then
        STATUS="✓ Good"
    elif [ $RSS_MB -lt 1000 ]; then
        STATUS="⚠ Moderate"
    else
        STATUS="✗ High"
    fi

    printf "%s  %6d  %8s  %s\n" "$(date +%H:%M:%S)" "$RSS_MB" "$SWAP_USED" "$STATUS"
    sleep 2
done

echo ""
echo "Process finished"
