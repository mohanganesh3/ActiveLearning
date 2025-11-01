#!/bin/bash
#
# Run CIFAR-10 experiments in parallel across 4 GPUs
#

echo "Starting CIFAR-10 experiments on 4 GPUs..."
echo "================================================"

# Create output directory
mkdir -p cifar10_results
mkdir -p logs_cifar10

# Run all 4 strategies in parallel on different GPUs
python cifar10_experiment.py --strategy random --gpu 0 > logs_cifar10/random.log 2>&1 &
PID1=$!
echo "GPU 0: Random Sampling (PID: $PID1)"

python cifar10_experiment.py --strategy greedy --gpu 1 > logs_cifar10/greedy.log 2>&1 &
PID2=$!
echo "GPU 1: Greedy K-Center (PID: $PID2)"

python cifar10_experiment.py --strategy leader --gpu 2 > logs_cifar10/leader.log 2>&1 &
PID3=$!
echo "GPU 2: Leader Clustering (PID: $PID3)"

python cifar10_experiment.py --strategy advanced --gpu 3 > logs_cifar10/advanced.log 2>&1 &
PID4=$!
echo "GPU 3: Advanced Leader (PID: $PID4)"

echo ""
echo "All experiments launched!"
echo "Monitor progress with:"
echo "  tail -f logs_cifar10/*.log"
echo ""
echo "Waiting for all experiments to complete..."

# Wait for all processes
wait $PID1
echo "✓ Random Sampling complete"

wait $PID2
echo "✓ Greedy K-Center complete"

wait $PID3
echo "✓ Leader Clustering complete"

wait $PID4
echo "✓ Advanced Leader complete"

echo ""
echo "================================================"
echo "All CIFAR-10 experiments complete!"
echo "Results saved in cifar10_results/"
echo "================================================"
