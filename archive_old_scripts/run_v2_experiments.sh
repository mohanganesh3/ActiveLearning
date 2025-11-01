#!/bin/bash
# Run Advanced Leader V2 experiments on both CIFAR-10 and CIFAR-100
# Uses nohup to survive SSH disconnection

# Record experiment version
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
VERSION="v2"

echo "================================================================================"
echo "Starting Advanced Leader V2 Experiments"
echo "Version: $VERSION"
echo "Timestamp: $TIMESTAMP"
echo "================================================================================"

# Create logs directory
mkdir -p logs_${VERSION}

# Function to run experiment
run_experiment() {
    local DATASET=$1
    local GPU=$2
    
    echo "Launching $DATASET on GPU $GPU..."
    
    nohup python -u cifar${DATASET}_experiment.py \
        --gpu $GPU \
        --initial_labeled 5000 \
        --budget 2500 \
        --rounds 9 \
        --strategy advanced \
        > logs_${VERSION}/advanced_leader_cifar${DATASET}_${TIMESTAMP}.log 2>&1 &
    
    local PID=$!
    echo "$DATASET PID: $PID"
    echo "$PID" > logs_${VERSION}/cifar${DATASET}_${VERSION}.pid
    
    sleep 2
}

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,utilization.gpu --format=csv,noheader

# Run CIFAR-10 on GPU 0
run_experiment "10" "0"

# Wait a bit before starting second experiment
sleep 5

# Run CIFAR-100 on GPU 1
run_experiment "100" "1"

echo ""
echo "================================================================================"
echo "Both experiments launched!"
echo "================================================================================"
echo ""
echo "Monitor progress with:"
echo "  tail -f logs_${VERSION}/advanced_leader_cifar10_${TIMESTAMP}.log"
echo "  tail -f logs_${VERSION}/advanced_leader_cifar100_${TIMESTAMP}.log"
echo ""
echo "Check processes:"
echo "  ps aux | grep python | grep cifar"
echo ""
echo "Kill if needed:"
echo "  kill \$(cat logs_${VERSION}/cifar10_${VERSION}.pid)"
echo "  kill \$(cat logs_${VERSION}/cifar100_${VERSION}.pid)"
echo ""
echo "Compare with V1 results after completion!"
echo "================================================================================"
