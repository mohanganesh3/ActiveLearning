#!/bin/bash
# Quick test to verify all bugs are fixed before running full experiments
# This runs ONLY 2 rounds of Advanced Leader on CIFAR-10

echo "================================================================================"
echo "QUICK TEST: Advanced Leader (2 rounds only)"
echo "================================================================================"
echo ""
echo "This test verifies:"
echo "  ✅ Zero-threshold bug is fixed (no 3000s delays)"
echo "  ✅ Round logic is correct (select before train)"
echo "  ✅ No catastrophic accuracy drops"
echo ""
echo "Expected results:"
echo "  - Round 1: ~120s sampling, ~15% accuracy"
echo "  - Round 2: ~120s sampling, ~25% accuracy"
echo "  - Total time: ~40 minutes"
echo ""
echo "================================================================================"
echo ""

python3 cifar10_experiment.py \
    --strategy advanced \
    --initial_labeled 1000 \
    --budget 1000 \
    --rounds 2 \
    --epochs 50 \
    --lr 0.1 \
    --seed 42 \
    --gpu 3

echo ""
echo "================================================================================"
echo "QUICK TEST COMPLETE!"
echo "================================================================================"
echo ""
echo "Check results:"
echo "  - If sampling times are ~120s: ✅ Zero-threshold bug is FIXED"
echo "  - If accuracy increases: ✅ Sample selection is working"
echo "  - If no errors: ✅ Ready for full experiments"
echo ""
echo "Next step:"
echo "  ./run_all_fixed.sh    # Run all experiments"
echo ""
