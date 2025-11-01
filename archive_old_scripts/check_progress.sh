#!/bin/bash

# Simple monitoring script to check experiment progress

echo "=========================================="
echo "EXPERIMENT STATUS CHECK"
echo "=========================================="
date
echo ""

# Check CIFAR-10 processes
echo "CIFAR-10 Processes:"
echo "-------------------"
cifar10_count=$(ps aux | grep "cifar10_experiment.py" | grep -v grep | wc -l)
if [ $cifar10_count -gt 0 ]; then
    echo "✓ $cifar10_count CIFAR-10 experiments running"
    ps aux | grep "cifar10_experiment.py" | grep -v grep | awk '{print "  [GPU " substr($14,1,1) "] " $13 " - CPU: " $3 "%, MEM: " $6/1024 " MB, Runtime: " $10}'
else
    echo "✗ No CIFAR-10 experiments running"
fi
echo ""

# Check CIFAR-100 processes
echo "CIFAR-100 Processes:"
echo "--------------------"
cifar100_count=$(ps aux | grep "cifar100_experiment.py" | grep -v grep | wc -l)
if [ $cifar100_count -gt 0 ]; then
    echo "✓ $cifar100_count CIFAR-100 experiments running"
    ps aux | grep "cifar100_experiment.py" | grep -v grep | awk '{print "  [GPU " substr($14,1,1) "] " $13 " - CPU: " $3 "%, MEM: " $6/1024 " MB, Runtime: " $10}'
else
    echo "✗ No CIFAR-100 experiments running"
fi
echo ""

# Check for completed results
echo "Completed Results:"
echo "------------------"
c10_results=$(ls cifar10_results/*_results.pkl 2>/dev/null | wc -l)
c100_results=$(ls cifar100_results/*_results.pkl 2>/dev/null | wc -l)
echo "CIFAR-10: $c10_results/4 completed"
echo "CIFAR-100: $c100_results/4 completed"
echo ""

# Show GPU usage
echo "GPU Usage:"
echo "----------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "GPU %s: %s%% utilization, %s/%s MB memory\n", $1, $3, $4, $5}'
else
    echo "(nvidia-smi not available)"
fi

echo ""
echo "=========================================="
echo "To view master log: tail -f master_experiment.log"
echo "To view individual logs: tail -f cifar10_results/random.log"
echo "=========================================="
