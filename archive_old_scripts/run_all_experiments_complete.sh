#!/bin/bash
# Complete Active Learning Experiment Runner
# Runs CIFAR-10 → CIFAR-100 → Visualization with full logging

set -e  # Exit on error

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="experiments_${TIMESTAMP}.log"

echo "========================================" | tee -a "$MAIN_LOG"
echo "Active Learning Experiments Started" | tee -a "$MAIN_LOG"
echo "Timestamp: $(date)" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"

# Create log directories
mkdir -p logs_cifar10
mkdir -p logs_cifar100
mkdir -p cifar10_results
mkdir -p cifar100_results

# Function to run experiment with detailed logging
run_experiment() {
    local dataset=$1
    local strategy=$2
    local gpu=$3
    local script=$4
    local logfile="logs_${dataset}/${strategy}_${TIMESTAMP}.log"
    
    echo "" | tee -a "$MAIN_LOG"
    echo "========================================" | tee -a "$MAIN_LOG"
    echo "Starting: ${dataset^^} - $strategy on GPU $gpu" | tee -a "$MAIN_LOG"
    echo "Time: $(date)" | tee -a "$MAIN_LOG"
    echo "Log: $logfile" | tee -a "$MAIN_LOG"
    echo "========================================" | tee -a "$MAIN_LOG"
    
    # Run with unbuffered output
    PYTHONUNBUFFERED=1 python -u "$script" \
        --strategy "$strategy" \
        --initial_labeled 5000 \
        --budget 2500 \
        --rounds 9 \
        --epochs 50 \
        --gpu "$gpu" 2>&1 | tee "$logfile"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ Completed: ${dataset^^} - $strategy (GPU $gpu)" | tee -a "$MAIN_LOG"
    else
        echo "✗ FAILED: ${dataset^^} - $strategy (GPU $gpu) - Exit code: $exit_code" | tee -a "$MAIN_LOG"
        return $exit_code
    fi
}

# CIFAR-10 Experiments (4 strategies on 4 GPUs in parallel)
echo "" | tee -a "$MAIN_LOG"
echo "########################################" | tee -a "$MAIN_LOG"
echo "PHASE 1: CIFAR-10 Experiments" | tee -a "$MAIN_LOG"
echo "########################################" | tee -a "$MAIN_LOG"

CIFAR10_START=$(date +%s)

# Run all 4 CIFAR-10 experiments in parallel on different GPUs
run_experiment "cifar10" "random" 0 "cifar10_experiment.py" &
PID_C10_RANDOM=$!

run_experiment "cifar10" "greedy" 1 "cifar10_experiment.py" &
PID_C10_GREEDY=$!

run_experiment "cifar10" "leader" 2 "cifar10_experiment.py" &
PID_C10_LEADER=$!

run_experiment "cifar10" "advanced" 3 "cifar10_experiment.py" &
PID_C10_ADVANCED=$!

# Wait for all CIFAR-10 experiments to complete
echo "" | tee -a "$MAIN_LOG"
echo "Waiting for all CIFAR-10 experiments to complete..." | tee -a "$MAIN_LOG"

wait $PID_C10_RANDOM
RESULT_C10_RANDOM=$?

wait $PID_C10_GREEDY
RESULT_C10_GREEDY=$?

wait $PID_C10_LEADER
RESULT_C10_LEADER=$?

wait $PID_C10_ADVANCED
RESULT_C10_ADVANCED=$?

CIFAR10_END=$(date +%s)
CIFAR10_DURATION=$((CIFAR10_END - CIFAR10_START))

echo "" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "CIFAR-10 Results Summary:" | tee -a "$MAIN_LOG"
echo "  Random: $([ $RESULT_C10_RANDOM -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')" | tee -a "$MAIN_LOG"
echo "  Greedy: $([ $RESULT_C10_GREEDY -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')" | tee -a "$MAIN_LOG"
echo "  Leader: $([ $RESULT_C10_LEADER -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')" | tee -a "$MAIN_LOG"
echo "  Advanced: $([ $RESULT_C10_ADVANCED -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')" | tee -a "$MAIN_LOG"
echo "  Total Time: ${CIFAR10_DURATION}s ($(($CIFAR10_DURATION / 60))m)" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"

# Check if any CIFAR-10 experiment failed
if [ $RESULT_C10_RANDOM -ne 0 ] || [ $RESULT_C10_GREEDY -ne 0 ] || [ $RESULT_C10_LEADER -ne 0 ] || [ $RESULT_C10_ADVANCED -ne 0 ]; then
    echo "⚠ WARNING: Some CIFAR-10 experiments failed, continuing anyway..." | tee -a "$MAIN_LOG"
fi

# CIFAR-100 Experiments (4 strategies on 4 GPUs in parallel)
echo "" | tee -a "$MAIN_LOG"
echo "########################################" | tee -a "$MAIN_LOG"
echo "PHASE 2: CIFAR-100 Experiments" | tee -a "$MAIN_LOG"
echo "########################################" | tee -a "$MAIN_LOG"

CIFAR100_START=$(date +%s)

# Run all 4 CIFAR-100 experiments in parallel on different GPUs
run_experiment "cifar100" "random" 0 "cifar100_experiment.py" &
PID_C100_RANDOM=$!

run_experiment "cifar100" "greedy" 1 "cifar100_experiment.py" &
PID_C100_GREEDY=$!

run_experiment "cifar100" "leader" 2 "cifar100_experiment.py" &
PID_C100_LEADER=$!

run_experiment "cifar100" "advanced" 3 "cifar100_experiment.py" &
PID_C100_ADVANCED=$!

# Wait for all CIFAR-100 experiments to complete
echo "" | tee -a "$MAIN_LOG"
echo "Waiting for all CIFAR-100 experiments to complete..." | tee -a "$MAIN_LOG"

wait $PID_C100_RANDOM
RESULT_C100_RANDOM=$?

wait $PID_C100_GREEDY
RESULT_C100_GREEDY=$?

wait $PID_C100_LEADER
RESULT_C100_LEADER=$?

wait $PID_C100_ADVANCED
RESULT_C100_ADVANCED=$?

CIFAR100_END=$(date +%s)
CIFAR100_DURATION=$((CIFAR100_END - CIFAR100_START))

echo "" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "CIFAR-100 Results Summary:" | tee -a "$MAIN_LOG"
echo "  Random: $([ $RESULT_C100_RANDOM -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')" | tee -a "$MAIN_LOG"
echo "  Greedy: $([ $RESULT_C100_GREEDY -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')" | tee -a "$MAIN_LOG"
echo "  Leader: $([ $RESULT_C100_LEADER -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')" | tee -a "$MAIN_LOG"
echo "  Advanced: $([ $RESULT_C100_ADVANCED -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')" | tee -a "$MAIN_LOG"
echo "  Total Time: ${CIFAR100_DURATION}s ($(($CIFAR100_DURATION / 60))m)" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"

# Visualization Phase
echo "" | tee -a "$MAIN_LOG"
echo "########################################" | tee -a "$MAIN_LOG"
echo "PHASE 3: Generating Visualizations" | tee -a "$MAIN_LOG"
echo "########################################" | tee -a "$MAIN_LOG"

VIZ_LOG="logs_visualization_${TIMESTAMP}.log"
echo "Generating plots..." | tee -a "$MAIN_LOG"
echo "Visualization log: $VIZ_LOG" | tee -a "$MAIN_LOG"

PYTHONUNBUFFERED=1 python -u visualize_results.py 2>&1 | tee "$VIZ_LOG"
VIZ_RESULT=$?

if [ $VIZ_RESULT -eq 0 ]; then
    echo "✓ Visualizations generated successfully" | tee -a "$MAIN_LOG"
else
    echo "✗ Visualization failed with exit code: $VIZ_RESULT" | tee -a "$MAIN_LOG"
fi

# Final Summary
TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - CIFAR10_START))

echo "" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "ALL EXPERIMENTS COMPLETE!" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "Finish Time: $(date)" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "Timing Summary:" | tee -a "$MAIN_LOG"
echo "  CIFAR-10:  ${CIFAR10_DURATION}s ($(($CIFAR10_DURATION / 60))m)" | tee -a "$MAIN_LOG"
echo "  CIFAR-100: ${CIFAR100_DURATION}s ($(($CIFAR100_DURATION / 60))m)" | tee -a "$MAIN_LOG"
echo "  Total:     ${TOTAL_DURATION}s ($(($TOTAL_DURATION / 60))m $(($TOTAL_DURATION % 60))s)" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "Results saved in:" | tee -a "$MAIN_LOG"
echo "  - cifar10_results/" | tee -a "$MAIN_LOG"
echo "  - cifar100_results/" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "Logs saved in:" | tee -a "$MAIN_LOG"
echo "  - logs_cifar10/" | tee -a "$MAIN_LOG"
echo "  - logs_cifar100/" | tee -a "$MAIN_LOG"
echo "  - $MAIN_LOG (master log)" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"

# Exit with failure if any critical experiment failed
if [ $RESULT_C10_GREEDY -ne 0 ] || [ $RESULT_C100_GREEDY -ne 0 ]; then
    echo "⚠ WARNING: Critical experiments failed!" | tee -a "$MAIN_LOG"
    exit 1
fi

echo "✓ All experiments completed successfully!" | tee -a "$MAIN_LOG"
exit 0
