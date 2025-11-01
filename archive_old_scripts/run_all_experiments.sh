#!/bin/bash

# Master script to run all experiments sequentially
# CIFAR-10 first (4 algorithms in parallel on 4 GPUs), then CIFAR-100

set -e  # Exit on error

WORKSPACE="/home/mohanganesh/active_learning_coreset"
cd "$WORKSPACE"

echo "=========================================="
echo "ACTIVE LEARNING EXPERIMENTS - MASTER CONTROL"
echo "=========================================="
echo ""
echo "Workflow:"
echo "1. Run CIFAR-10 experiments (4 algos parallel on GPUs 0-3)"
echo "2. Generate CIFAR-10 plots"
echo "3. Run CIFAR-100 experiments (4 algos parallel on GPUs 0-3)"
echo "4. Generate CIFAR-100 plots"
echo ""
echo "=========================================="
echo ""

# ==========================================
# STEP 1: CIFAR-10 EXPERIMENTS
# ==========================================
echo "STEP 1: Launching CIFAR-10 experiments..."
echo "=========================================="

# Create results directory
mkdir -p cifar10_results

# Launch 4 experiments in background (using -u for unbuffered output)
echo "Starting Random Sampling (GPU 0)..."
python -u cifar10_experiment.py --strategy random --gpu 0 > cifar10_results/random.log 2>&1 &
PID_RANDOM=$!

echo "Starting Greedy K-Center (GPU 1)..."
python -u cifar10_experiment.py --strategy greedy --gpu 1 > cifar10_results/greedy.log 2>&1 &
PID_GREEDY=$!

echo "Starting Leader Clustering (GPU 2)..."
python -u cifar10_experiment.py --strategy leader --gpu 2 > cifar10_results/leader.log 2>&1 &
PID_LEADER=$!

echo "Starting Advanced Leader (GPU 3)..."
python -u cifar10_experiment.py --strategy advanced --gpu 3 > cifar10_results/advanced.log 2>&1 &
PID_ADVANCED=$!

echo ""
echo "All CIFAR-10 experiments launched!"
echo "  Random:   PID $PID_RANDOM (GPU 0)"
echo "  Greedy:   PID $PID_GREEDY (GPU 1)"
echo "  Leader:   PID $PID_LEADER (GPU 2)"
echo "  Advanced: PID $PID_ADVANCED (GPU 3)"
echo ""

# Monitor progress
echo "Waiting for all CIFAR-10 experiments to complete..."
echo "(You can monitor logs in cifar10_results/*.log)"
echo ""

# Wait for all to complete
wait $PID_RANDOM
echo "✓ Random Sampling completed"

wait $PID_GREEDY
echo "✓ Greedy K-Center completed"

wait $PID_LEADER
echo "✓ Leader Clustering completed"

wait $PID_ADVANCED
echo "✓ Advanced Leader completed"

echo ""
echo "=========================================="
echo "All CIFAR-10 experiments completed!"
echo "=========================================="
echo ""

# ==========================================
# STEP 2: GENERATE CIFAR-10 PLOTS
# ==========================================
echo "STEP 2: Generating CIFAR-10 plots..."
echo "=========================================="

python -u visualize_results.py --dataset cifar10

echo "✓ CIFAR-10 plots generated"
echo ""

# ==========================================
# STEP 3: CIFAR-100 EXPERIMENTS
# ==========================================
echo "STEP 3: Launching CIFAR-100 experiments..."
echo "=========================================="

# Create results directory
mkdir -p cifar100_results

# Launch 4 experiments in background (using -u for unbuffered output)
echo "Starting Random Sampling (GPU 0)..."
python -u cifar100_experiment.py --strategy random --gpu 0 > cifar100_results/random.log 2>&1 &
PID_RANDOM=$!

echo "Starting Greedy K-Center (GPU 1)..."
python -u cifar100_experiment.py --strategy greedy --gpu 1 > cifar100_results/greedy.log 2>&1 &
PID_GREEDY=$!

echo "Starting Leader Clustering (GPU 2)..."
python -u cifar100_experiment.py --strategy leader --gpu 2 > cifar100_results/leader.log 2>&1 &
PID_LEADER=$!

echo "Starting Advanced Leader (GPU 3)..."
python -u cifar100_experiment.py --strategy advanced --gpu 3 > cifar100_results/advanced.log 2>&1 &
PID_ADVANCED=$!

echo ""
echo "All CIFAR-100 experiments launched!"
echo "  Random:   PID $PID_RANDOM (GPU 0)"
echo "  Greedy:   PID $PID_GREEDY (GPU 1)"
echo "  Leader:   PID $PID_LEADER (GPU 2)"
echo "  Advanced: PID $PID_ADVANCED (GPU 3)"
echo ""

# Monitor progress
echo "Waiting for all CIFAR-100 experiments to complete..."
echo "(You can monitor logs in cifar100_results/*.log)"
echo ""

# Wait for all to complete
wait $PID_RANDOM
echo "✓ Random Sampling completed"

wait $PID_GREEDY
echo "✓ Greedy K-Center completed"

wait $PID_LEADER
echo "✓ Leader Clustering completed"

wait $PID_ADVANCED
echo "✓ Advanced Leader completed"

echo ""
echo "=========================================="
echo "All CIFAR-100 experiments completed!"
echo "=========================================="
echo ""

# ==========================================
# STEP 4: GENERATE CIFAR-100 PLOTS
# ==========================================
echo "STEP 4: Generating CIFAR-100 plots..."
echo "=========================================="

python -u visualize_results.py --dataset cifar100

echo "✓ CIFAR-100 plots generated"
echo ""

# ==========================================
# FINAL SUMMARY
# ==========================================
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Results:"
echo "  CIFAR-10 results:  cifar10_results/"
echo "  CIFAR-100 results: cifar100_results/"
echo ""
echo "Plots:"
echo "  cifar10_results/accuracy_comparison.png"
echo "  cifar10_results/sampling_time_comparison.png"
echo "  cifar100_results/accuracy_comparison.png"
echo "  cifar100_results/sampling_time_comparison.png"
echo ""
echo "=========================================="
