#!/bin/bash
#
# Run CIFAR-10 experiments with nohup (continues even if SSH disconnects)
#

cd /home/mohanganesh/active_learning_coreset

echo "Starting CIFAR-10 experiments with nohup..."
echo "================================================"
echo "Experiments will continue even if you disconnect from SSH"
echo ""

# Create output directories
mkdir -p cifar10_results
mkdir -p logs_cifar10

# Run all 4 strategies in parallel on different GPUs using nohup
nohup python cifar10_experiment.py --strategy random --gpu 0 > logs_cifar10/random.log 2>&1 &
PID1=$!
echo "GPU 0: Random Sampling (PID: $PID1)"

nohup python cifar10_experiment.py --strategy greedy --gpu 1 > logs_cifar10/greedy.log 2>&1 &
PID2=$!
echo "GPU 1: Greedy K-Center (PID: $PID2)"

nohup python cifar10_experiment.py --strategy leader --gpu 2 > logs_cifar10/leader.log 2>&1 &
PID3=$!
echo "GPU 2: Leader Clustering (PID: $PID3)"

nohup python cifar10_experiment.py --strategy advanced --gpu 3 > logs_cifar10/advanced.log 2>&1 &
PID4=$!
echo "GPU 3: Advanced Leader (PID: $PID4)"

# Save PIDs to file
echo $PID1 > logs_cifar10/pids.txt
echo $PID2 >> logs_cifar10/pids.txt
echo $PID3 >> logs_cifar10/pids.txt
echo $PID4 >> logs_cifar10/pids.txt

echo ""
echo "All experiments launched in background!"
echo "PIDs saved to logs_cifar10/pids.txt"
echo ""
echo "Monitor progress with:"
echo "  tail -f logs_cifar10/random.log"
echo "  tail -f logs_cifar10/greedy.log"
echo "  tail -f logs_cifar10/leader.log"
echo "  tail -f logs_cifar10/advanced.log"
echo ""
echo "Check if still running:"
echo "  ps aux | grep cifar10_experiment.py"
echo ""
echo "Results will be saved in cifar10_results/"
echo "================================================"
