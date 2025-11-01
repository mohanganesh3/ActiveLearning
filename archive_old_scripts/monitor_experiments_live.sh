#!/bin/bash
# Live monitoring script for experiments
# Usage: ./monitor_experiments_live.sh

echo "Active Learning Experiments Monitor"
echo "===================================="
echo ""

# Find the latest master log
MASTER_LOG=$(ls -t experiments_*.log 2>/dev/null | head -1)

if [ -z "$MASTER_LOG" ]; then
    echo "No master log found. Experiments may not have started yet."
    echo ""
    echo "Looking for individual experiment logs..."
    echo ""
fi

# Function to show GPU usage
show_gpu_status() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "GPU Status:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | \
    while IFS=, read -r gpu name util mem_used mem_total; do
        printf "GPU %s (%s): %s | Mem: %s/%s\n" "$gpu" "$name" "$util" "$mem_used" "$mem_total"
    done
    echo ""
}

# Function to show running Python processes
show_processes() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running Experiment Processes:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ps aux | grep -E "(cifar10|cifar100)_experiment.py" | grep -v grep | \
    awk '{printf "PID %s: %s\n", $2, substr($0, index($0,$11))}'
    echo ""
}

# Function to show latest log entries
show_latest_logs() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Latest Log Updates:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Show CIFAR-10 logs
    for strategy in random greedy leader advanced; do
        LATEST_C10=$(ls -t logs_cifar10/${strategy}_*.log 2>/dev/null | head -1)
        if [ -n "$LATEST_C10" ]; then
            LAST_LINE=$(tail -1 "$LATEST_C10" 2>/dev/null)
            if [ -n "$LAST_LINE" ]; then
                echo "CIFAR-10 $strategy: $LAST_LINE"
            fi
        fi
    done
    
    echo ""
    
    # Show CIFAR-100 logs
    for strategy in random greedy leader advanced; do
        LATEST_C100=$(ls -t logs_cifar100/${strategy}_*.log 2>/dev/null | head -1)
        if [ -n "$LATEST_C100" ]; then
            LAST_LINE=$(tail -1 "$LATEST_C100" 2>/dev/null)
            if [ -n "$LAST_LINE" ]; then
                echo "CIFAR-100 $strategy: $LAST_LINE"
            fi
        fi
    done
    echo ""
}

# Function to show experiment progress
show_progress() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Experiment Progress:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    for dataset in cifar10 cifar100; do
        echo "${dataset^^}:"
        for strategy in random greedy leader advanced; do
            LATEST_LOG=$(ls -t logs_${dataset}/${strategy}_*.log 2>/dev/null | head -1)
            if [ -n "$LATEST_LOG" ]; then
                ROUNDS=$(grep -o "ROUND [0-9]*/9" "$LATEST_LOG" | tail -1)
                TEST_ACC=$(grep "Test Accuracy:" "$LATEST_LOG" | tail -1 | awk '{print $3}')
                if [ -n "$ROUNDS" ]; then
                    printf "  %-12s: %s" "$strategy" "$ROUNDS"
                    if [ -n "$TEST_ACC" ]; then
                        printf " | Latest Acc: %s" "$TEST_ACC"
                    fi
                    echo ""
                else
                    echo "  $strategy: Starting..."
                fi
            else
                echo "  $strategy: Not started"
            fi
        done
        echo ""
    done
}

# Main monitoring loop
while true; do
    clear
    echo "╔════════════════════════════════════════════╗"
    echo "║   Active Learning Experiments Monitor      ║"
    echo "║   $(date +'%Y-%m-%d %H:%M:%S')                   ║"
    echo "╚════════════════════════════════════════════╝"
    echo ""
    
    show_gpu_status
    show_processes
    show_progress
    show_latest_logs
    
    if [ -n "$MASTER_LOG" ]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Master Log: $MASTER_LOG"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        tail -5 "$MASTER_LOG"
        echo ""
    fi
    
    echo "Press Ctrl+C to exit. Refreshing in 10 seconds..."
    sleep 10
done
