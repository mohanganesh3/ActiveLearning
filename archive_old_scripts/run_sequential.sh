#!/bin/bash
# Sequential execution with nohup: CIFAR-10 first (4 GPUs parallel), then CIFAR-100 (4 GPUs parallel)

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         SEQUENTIAL ACTIVE LEARNING EXPERIMENTS               ║"
echo "║  Step 1: CIFAR-10 (4 strategies on GPUs 0-3 in parallel)     ║"
echo "║  Step 2: CIFAR-100 (4 strategies on GPUs 0-3 in parallel)    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Ensure log directories exist
mkdir -p logs_cifar10 logs_cifar100

# ============================================================================
# STEP 1: CIFAR-10 - All 4 strategies in parallel on 4 GPUs
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 STEP 1/2: Starting CIFAR-10 experiments (4 GPUs parallel)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  GPU 0: Random Sampling"
echo "  GPU 1: Greedy K-Center"
echo "  GPU 2: Leader Clustering"
echo "  GPU 3: Advanced Leader (with ADAPTIVE thresholds)"
echo ""

# Launch all 4 CIFAR-10 experiments in parallel with unbuffered output
python3 -u cifar10_experiment.py --strategy random --gpu 0 > logs_cifar10/random.log 2>&1 &
PID_C10_RANDOM=$!

python3 -u cifar10_experiment.py --strategy greedy --gpu 1 > logs_cifar10/greedy.log 2>&1 &
PID_C10_GREEDY=$!

python3 -u cifar10_experiment.py --strategy leader --gpu 2 > logs_cifar10/leader.log 2>&1 &
PID_C10_LEADER=$!

python3 -u cifar10_experiment.py --strategy advanced --gpu 3 > logs_cifar10/advanced.log 2>&1 &
PID_C10_ADVANCED=$!

echo "✅ CIFAR-10 experiments launched:"
echo "   Random (GPU 0): PID $PID_C10_RANDOM"
echo "   Greedy (GPU 1): PID $PID_C10_GREEDY"
echo "   Leader (GPU 2): PID $PID_C10_LEADER"
echo "   Advanced (GPU 3): PID $PID_C10_ADVANCED"
echo ""
echo "⏳ Waiting for all CIFAR-10 experiments to complete..."
echo "   Monitor: tail -f logs_cifar10/advanced.log"
echo ""

# Wait for all CIFAR-10 experiments to finish
wait $PID_C10_RANDOM
echo "   ✓ Random completed"

wait $PID_C10_GREEDY
echo "   ✓ Greedy completed"

wait $PID_C10_LEADER
echo "   ✓ Leader completed"

wait $PID_C10_ADVANCED
echo "   ✓ Advanced completed"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ CIFAR-10 COMPLETE! All 4 strategies finished."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
sleep 3

# ============================================================================
# STEP 2: CIFAR-100 - All 4 strategies in parallel on 4 GPUs
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 STEP 2/2: Starting CIFAR-100 experiments (4 GPUs parallel)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  GPU 0: Random Sampling"
echo "  GPU 1: Greedy K-Center"
echo "  GPU 2: Leader Clustering"
echo "  GPU 3: Advanced Leader (with ADAPTIVE thresholds)"
echo ""

# Launch all 4 CIFAR-100 experiments in parallel with unbuffered output
python3 -u cifar100_experiment.py --strategy random --gpu 0 > logs_cifar100/random.log 2>&1 &
PID_C100_RANDOM=$!

python3 -u cifar100_experiment.py --strategy greedy --gpu 1 > logs_cifar100/greedy.log 2>&1 &
PID_C100_GREEDY=$!

python3 -u cifar100_experiment.py --strategy leader --gpu 2 > logs_cifar100/leader.log 2>&1 &
PID_C100_LEADER=$!

python3 -u cifar100_experiment.py --strategy advanced --gpu 3 > logs_cifar100/advanced.log 2>&1 &
PID_C100_ADVANCED=$!

echo "✅ CIFAR-100 experiments launched:"
echo "   Random (GPU 0): PID $PID_C100_RANDOM"
echo "   Greedy (GPU 1): PID $PID_C100_GREEDY"
echo "   Leader (GPU 2): PID $PID_C100_LEADER"
echo "   Advanced (GPU 3): PID $PID_C100_ADVANCED"
echo ""
echo "⏳ Waiting for all CIFAR-100 experiments to complete..."
echo "   Monitor: tail -f logs_cifar100/advanced.log"
echo ""

# Wait for all CIFAR-100 experiments to finish
wait $PID_C100_RANDOM
echo "   ✓ Random completed"

wait $PID_C100_GREEDY
echo "   ✓ Greedy completed"

wait $PID_C100_LEADER
echo "   ✓ Leader completed"

wait $PID_C100_ADVANCED
echo "   ✓ Advanced completed"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 ALL EXPERIMENTS COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Results saved in:"
echo "   CIFAR-10:  cifar10_results/"
echo "   CIFAR-100: cifar100_results/"
echo ""
echo "📈 View results: python3 visualize_results.py"
echo ""
