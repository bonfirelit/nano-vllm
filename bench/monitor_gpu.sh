#!/bin/bash
# GPU monitoring script for model benchmarking
# Usage: ./monitor_gpu.sh <command_to_run>
# Example: ./monitor_gpu.sh python quick_bench.py

if [ $# -eq 0 ]; then
    echo "Usage: $0 <command_to_run>"
    echo "Example: $0 python quick_bench.py"
    exit 1
fi

# Create log directory
LOG_DIR="./benchmark_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/gpu_monitor_${TIMESTAMP}.log"

echo "Starting GPU monitoring..."
echo "Command: $@"
echo "Log file: $LOG_FILE"
echo ""

# Start monitoring in background
(
    while true; do
        echo "=== $(date) ===" >> "$LOG_FILE"
        nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits >> "$LOG_FILE"
        sleep 1
    done
) &
MONITOR_PID=$!

# Trap to kill monitor on exit
trap "kill $MONITOR_PID 2>/dev/null; echo 'GPU monitoring stopped'; echo 'Logs saved to: $LOG_FILE'" EXIT

# Run the command
echo "Running command..."
eval "$@"
EXIT_CODE=$?

echo ""
echo "Command completed with exit code: $EXIT_CODE"
echo "GPU utilization logs saved to: $LOG_FILE"
echo ""
echo "Quick summary:"
tail -20 "$LOG_FILE"

exit $EXIT_CODE
