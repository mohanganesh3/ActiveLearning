#!/bin/bash
# Live progress monitor for all experiments

clear
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          ACTIVE LEARNING EXPERIMENTS - LIVE PROGRESS         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

while true; do
    echo "─────────────────────────────────────────────────────────────"
    date
    echo "─────────────────────────────────────────────────────────────"
    
    echo ""
    echo "📊 CIFAR-10 Progress:"
    for log in logs_cifar10/*.log; do
        if [ -f "$log" ]; then
            strategy=$(basename "$log" .log)
            round=$(grep -oP "ROUND \K\d+" "$log" | tail -1)
            epoch=$(grep -oP "Epoch \K\d+/\d+" "$log" | tail -1)
            acc=$(grep -oP "Test Acc: \K[\d.]+" "$log" | tail -1)
            echo "  $strategy: Round $round, Epoch $epoch, Acc=$acc%"
        fi
    done
    
    echo ""
    echo "📊 CIFAR-100 Progress:"
    for log in logs_cifar100/*.log; do
        if [ -f "$log" ]; then
            strategy=$(basename "$log" .log)
            round=$(grep -oP "ROUND \K\d+" "$log" | tail -1)
            epoch=$(grep -oP "Epoch \K\d+/\d+" "$log" | tail -1)
            acc=$(grep -oP "Test Acc: \K[\d.]+" "$log" | tail -1)
            echo "  $strategy: Round $round, Epoch $epoch, Acc=$acc%"
        fi
    done
    
    echo ""
    echo "🔍 Advanced Leader Thresholds (CIFAR-10):"
    grep -A 1 "Distance distribution" logs_cifar10/advanced.log | tail -2
    
    echo ""
    echo "Press Ctrl+C to stop monitoring..."
    sleep 10
    clear
done
