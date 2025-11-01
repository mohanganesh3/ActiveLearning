#!/bin/bash
# Simple launcher for nohup experiments
# This script starts experiments in background and survives SSH disconnection

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
NOHUP_LOG="nohup_experiments_${TIMESTAMP}.log"

echo "========================================="
echo "Starting Active Learning Experiments"
echo "========================================="
echo ""
echo "Configuration:"
echo "  - Initial labeled: 5000 samples"
echo "  - Budget per round: 2500 samples"
echo "  - Total rounds: 9"
echo "  - Final labeled: 25000 samples"
echo ""
echo "Datasets: CIFAR-10 and CIFAR-100"
echo "Strategies: Random, Greedy K-Center, Leader, Advanced"
echo "GPUs: 0, 1, 2, 3 (parallel execution)"
echo ""
echo "========================================="
echo ""

# Check if experiments are already running
RUNNING=$(ps aux | grep -E "(cifar10|cifar100)_experiment.py" | grep -v grep | wc -l)
if [ $RUNNING -gt 0 ]; then
    echo "⚠ WARNING: Experiments may already be running!"
    echo ""
    ps aux | grep -E "(cifar10|cifar100)_experiment.py" | grep -v grep
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Make scripts executable
chmod +x run_all_experiments_complete.sh
chmod +x monitor_experiments_live.sh

echo "Starting experiments in background with nohup..."
echo "Log file: $NOHUP_LOG"
echo ""

# Start with nohup (survives SSH disconnect)
nohup ./run_all_experiments_complete.sh > "$NOHUP_LOG" 2>&1 &
MAIN_PID=$!

echo "✓ Experiments started!"
echo ""
echo "Main Process PID: $MAIN_PID"
echo ""
echo "========================================="
echo "How to Monitor:"
echo "========================================="
echo ""
echo "1. Live monitor (auto-refresh):"
echo "   ./monitor_experiments_live.sh"
echo ""
echo "2. Watch main log:"
echo "   tail -f $NOHUP_LOG"
echo ""
echo "3. Watch specific experiment (example):"
echo "   tail -f logs_cifar10/greedy_*.log"
echo ""
echo "4. Check GPU usage:"
echo "   watch -n 2 nvidia-smi"
echo ""
echo "5. Check if still running:"
echo "   ps aux | grep cifar"
echo ""
echo "========================================="
echo "To stop all experiments:"
echo "========================================="
echo "   pkill -f cifar.*_experiment.py"
echo ""
echo "========================================="
echo ""
echo "You can now safely disconnect from SSH."
echo "Experiments will continue running in background."
echo ""
