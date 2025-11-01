#!/bin/bash
# V3 Experiment Runner - October 31, 2025
# Run V3 (late-round selectivity + gentle class diversity) on both CIFAR-10 and CIFAR-100

echo "=========================================="
echo "VERSION 3 EXPERIMENT LAUNCHER"
echo "=========================================="
echo "Changes from V1:"
echo "  1. Late-round selectivity (rounds 7-9)"
echo "  2. Gentle class-diversity bonus"
echo "  3. Threshold validation (no enforcement)"
echo "  4. NO forced constraints (lessons from V2)"
echo ""

# Backup current results
echo "Backing up current results to old_results_V2..."
mkdir -p old_results_V2
if [ -d "cifar10_results" ]; then
    cp -r cifar10_results old_results_V2/
    echo "  ✓ Backed up cifar10_results"
fi
if [ -d "cifar100_results" ]; then
    cp -r cifar100_results old_results_V2/
    echo "  ✓ Backed up cifar100_results"
fi

# Clean and recreate result directories
echo ""
echo "Cleaning result directories..."
rm -rf cifar10_results cifar100_results
mkdir -p cifar10_results cifar100_results
echo "  ✓ Clean directories created"

# Create log directory
echo ""
echo "Creating V3 log directory..."
mkdir -p logs_v3
echo "  ✓ logs_v3 created"

# Get timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo ""
echo "=========================================="
echo "LAUNCHING V3 EXPERIMENTS"
echo "=========================================="
echo "Start time: $(date)"
echo "Timestamp: $TIMESTAMP"
echo ""

# Launch CIFAR-10 on GPU 0
echo "Launching CIFAR-10 experiment (Advanced Leader V3) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar10_experiment.py --strategy advanced --initial_labeled 5000 --budget 2500 --rounds 9 > logs_v3/cifar10_v3_${TIMESTAMP}.log 2>&1 &
CIFAR10_PID=$!
echo "  PID: $CIFAR10_PID"
echo "  Log: logs_v3/cifar10_v3_${TIMESTAMP}.log"
echo $CIFAR10_PID > logs_v3/cifar10_v3.pid

sleep 5

# Launch CIFAR-100 on GPU 1
echo ""
echo "Launching CIFAR-100 experiment (Advanced Leader V3) on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup python -u cifar100_experiment.py --strategy advanced --initial_labeled 5000 --budget 2500 --rounds 9 > logs_v3/cifar100_v3_${TIMESTAMP}.log 2>&1 &
CIFAR100_PID=$!
echo "  PID: $CIFAR100_PID"
echo "  Log: logs_v3/cifar100_v3_${TIMESTAMP}.log"
echo $CIFAR100_PID > logs_v3/cifar100_v3.pid

echo ""
echo "=========================================="
echo "EXPERIMENTS LAUNCHED"
echo "=========================================="
echo "CIFAR-10:  PID $CIFAR10_PID (GPU 0)"
echo "CIFAR-100: PID $CIFAR100_PID (GPU 1)"
echo ""
echo "Monitor progress with:"
echo "  tail -f logs_v3/cifar10_v3_${TIMESTAMP}.log"
echo "  tail -f logs_v3/cifar100_v3_${TIMESTAMP}.log"
echo ""
echo "Or use:"
echo "  watch -n 30 'tail -20 logs_v3/cifar100_v3_${TIMESTAMP}.log | grep \"Round\\|CV=\\|Final\"'"
echo ""
echo "Check processes:"
echo "  ps aux | grep cifar"
echo ""
echo "Estimated completion: ~8 hours (both running in parallel)"
echo "=========================================="
