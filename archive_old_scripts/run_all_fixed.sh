#!/bin/bash
# Run all CIFAR-10 and CIFAR-100 experiments with fixes
# All bugs have been fixed - this should produce clean results

echo "================================================================================"
echo "RUNNING ALL EXPERIMENTS WITH BUG FIXES"
echo "================================================================================"
echo ""
echo "Bugs Fixed:"
echo "  ✅ Zero-threshold bug (Advanced Leader)"
echo "  ✅ Round logic (select BEFORE train, not after)"
echo "  ✅ Last round wasteful retrain"
echo "  ✅ Proper labeled size progression"
echo ""
echo "Expected Results:"
echo "  - Advanced Leader CIFAR-10: ~120s per round, ~70% final accuracy"
echo "  - Advanced Leader CIFAR-100: ~180s per round, ~44% final accuracy"
echo "  - No catastrophic slowdowns (3000s)"
echo "  - No accuracy drops from bad samples"
echo ""
echo "================================================================================"
echo ""

GPU=3
SEED=42

# Create results directories
mkdir -p cifar10_results
mkdir -p cifar100_results

# CIFAR-10 Experiments
echo ""
echo "================================================================================"
echo "CIFAR-10 EXPERIMENTS"
echo "================================================================================"

echo ""
echo "------------------------"
echo "1/4: Random Sampling"
echo "------------------------"
python3 cifar10_experiment.py \
    --strategy random \
    --initial_labeled 1000 \
    --budget 1000 \
    --rounds 10 \
    --epochs 50 \
    --lr 0.1 \
    --seed $SEED \
    --gpu $GPU \
    2>&1 | tee cifar10_results/random.log

echo ""
echo "------------------------"
echo "2/4: Greedy K-Center"
echo "------------------------"
python3 cifar10_experiment.py \
    --strategy greedy \
    --initial_labeled 1000 \
    --budget 1000 \
    --rounds 10 \
    --epochs 50 \
    --lr 0.1 \
    --seed $SEED \
    --gpu $GPU \
    2>&1 | tee cifar10_results/greedy.log

echo ""
echo "------------------------"
echo "3/4: Leader Clustering"
echo "------------------------"
python3 cifar10_experiment.py \
    --strategy leader \
    --initial_labeled 1000 \
    --budget 1000 \
    --rounds 10 \
    --epochs 50 \
    --lr 0.1 \
    --seed $SEED \
    --gpu $GPU \
    2>&1 | tee cifar10_results/leader.log

echo ""
echo "------------------------"
echo "4/4: Advanced Leader"
echo "------------------------"
python3 cifar10_experiment.py \
    --strategy advanced \
    --initial_labeled 1000 \
    --budget 1000 \
    --rounds 10 \
    --epochs 50 \
    --lr 0.1 \
    --seed $SEED \
    --gpu $GPU \
    2>&1 | tee cifar10_results/advanced.log

# CIFAR-100 Experiments
echo ""
echo ""
echo "================================================================================"
echo "CIFAR-100 EXPERIMENTS"
echo "================================================================================"

echo ""
echo "------------------------"
echo "1/4: Random Sampling"
echo "------------------------"
python3 cifar100_experiment.py \
    --strategy random \
    --initial_labeled 5000 \
    --budget 2500 \
    --rounds 9 \
    --epochs 50 \
    --lr 0.1 \
    --seed $SEED \
    --gpu $GPU \
    2>&1 | tee cifar100_results/random.log

echo ""
echo "------------------------"
echo "2/4: Greedy K-Center"
echo "------------------------"
python3 cifar100_experiment.py \
    --strategy greedy \
    --initial_labeled 5000 \
    --budget 2500 \
    --rounds 9 \
    --epochs 50 \
    --lr 0.1 \
    --seed $SEED \
    --gpu $GPU \
    2>&1 | tee cifar100_results/greedy.log

echo ""
echo "------------------------"
echo "3/4: Leader Clustering"
echo "------------------------"
python3 cifar100_experiment.py \
    --strategy leader \
    --initial_labeled 5000 \
    --budget 2500 \
    --rounds 9 \
    --epochs 50 \
    --lr 0.1 \
    --seed $SEED \
    --gpu $GPU \
    2>&1 | tee cifar100_results/leader.log

echo ""
echo "------------------------"
echo "4/4: Advanced Leader"
echo "------------------------"
python3 cifar100_experiment.py \
    --strategy advanced \
    --initial_labeled 5000 \
    --budget 2500 \
    --rounds 9 \
    --epochs 50 \
    --lr 0.1 \
    --seed $SEED \
    --gpu $GPU \
    2>&1 | tee cifar100_results/advanced.log

echo ""
echo "================================================================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - cifar10_results/"
echo "  - cifar100_results/"
echo ""
echo "View visualizations:"
echo "  python3 visualize_results.py"
echo ""
