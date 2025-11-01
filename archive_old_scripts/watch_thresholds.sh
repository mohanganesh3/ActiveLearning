#!/bin/bash
# Watch for the adaptive thresholds in Round 2

echo "Monitoring Advanced Leader - waiting for Round 2 thresholds..."
echo "This will auto-refresh every 10 seconds"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    echo "════════════════════════════════════════════════════════════"
    echo "  ADVANCED LEADER THRESHOLD MONITORING"
    echo "════════════════════════════════════════════════════════════"
    date
    echo ""
    
    # Check current round
    ROUND=$(grep "ROUND" logs_cifar10/advanced.log 2>/dev/null | tail -1)
    echo "Current: $ROUND"
    echo ""
    
    # Look for threshold computation
    if grep -q "Distance distribution" logs_cifar10/advanced.log 2>/dev/null; then
        echo "✅ ROUND 2 STARTED - Adaptive Thresholds Found!"
        echo ""
        grep -A10 "Distance distribution" logs_cifar10/advanced.log | head -12
        echo ""
        echo "────────────────────────────────────────────────────────────"
        echo "SUCCESS! Fix is working - see scaled thresholds above"
        echo "────────────────────────────────────────────────────────────"
        break
    else
        echo "⏳ Still in Round 1 training..."
        echo ""
        tail -3 logs_cifar10/advanced.log 2>/dev/null
    fi
    
    sleep 10
done

echo ""
echo "Monitor complete. Experiments continue in background."
echo "Use: tail -f logs_cifar10/advanced.log"
