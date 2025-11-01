#!/bin/bash
# Run all experiments with unbuffered output

echo "Starting CIFAR-10 experiments (unbuffered for progress)..."
mkdir -p logs_cifar10

python3 -u cifar10_experiment.py --strategy random --gpu 0 > logs_cifar10/random.log 2>&1 &
python3 -u cifar10_experiment.py --strategy greedy --gpu 1 > logs_cifar10/greedy.log 2>&1 &
python3 -u cifar10_experiment.py --strategy leader --gpu 2 > logs_cifar10/leader.log 2>&1 &
python3 -u cifar10_experiment.py --strategy advanced --gpu 3 > logs_cifar10/advanced.log 2>&1 &

echo "Starting CIFAR-100 experiments (unbuffered for progress)..."
mkdir -p logs_cifar100

python3 -u cifar100_experiment.py --strategy random --gpu 0 > logs_cifar100/random.log 2>&1 &
python3 -u cifar100_experiment.py --strategy greedy --gpu 1 > logs_cifar100/greedy.log 2>&1 &
python3 -u cifar100_experiment.py --strategy leader --gpu 2 > logs_cifar100/leader.log 2>&1 &
python3 -u cifar100_experiment.py --strategy advanced --gpu 3 > logs_cifar100/advanced.log 2>&1 &

echo "All 8 experiments launched with unbuffered output!"
echo "Monitor progress: tail -f logs_cifar10/advanced.log"
