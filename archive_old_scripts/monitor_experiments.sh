#!/bin/bash

echo "=========================================="
echo "ACTIVE LEARNING EXPERIMENT STATUS"
echo "=========================================="
echo ""

# Check CIFAR-10 processes
echo "CIFAR-10 Experiments:"
echo "--------------------"
cifar10_count=$(ps aux | grep cifar10_experiment.py | grep -v grep | wc -l)
if [ $cifar10_count -gt 0 ]; then
    echo "✓ $cifar10_count processes running"
    ps aux | grep cifar10_experiment.py | grep -v grep | awk '{print "  - " $12 " " $13 " (PID: " $2 ", CPU: " $3 "%, MEM: " $4 "%)"}'
else
    echo "✗ No processes running"
fi
echo ""

# Check CIFAR-100 processes
echo "CIFAR-100 Experiments:"
echo "--------------------"
cifar100_count=$(ps aux | grep cifar100_experiment.py | grep -v grep | wc -l)
if [ $cifar100_count -gt 0 ]; then
    echo "✓ $cifar100_count processes running"
    ps aux | grep cifar100_experiment.py | grep -v grep | awk '{print "  - " $12 " " $13 " (PID: " $2 ", CPU: " $3 "%, MEM: " $4 "%)"}'
else
    echo "✗ No processes running"
fi
echo ""

# Check latest log updates
echo "Latest Log Updates:"
echo "-------------------"
echo "CIFAR-10:"
for strategy in random greedy leader advanced; do
    if [ -f "logs_cifar10/${strategy}.log" ]; then
        last_line=$(tail -1 logs_cifar10/${strategy}.log 2>/dev/null | cut -c1-100)
        echo "  $strategy: $last_line"
    fi
done

echo ""
echo "CIFAR-100:"
for strategy in random greedy leader advanced; do
    if [ -f "logs_cifar100/${strategy}.log" ]; then
        last_line=$(tail -1 logs_cifar100/${strategy}.log 2>/dev/null | cut -c1-100)
        echo "  $strategy: $last_line"
    fi
done

echo ""
echo "=========================================="
echo "To view detailed logs:"
echo "  tail -f logs_cifar10/<strategy>.log"
echo "  tail -f logs_cifar100/<strategy>.log"
echo "=========================================="
