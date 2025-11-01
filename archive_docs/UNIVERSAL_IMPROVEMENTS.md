================================================================================
UNIVERSAL IMPROVEMENTS TO ADVANCED LEADER CLUSTERING
================================================================================

CRITICAL PRINCIPLE: No dataset-specific logic (no if num_classes > 20)
                   Algorithm must work the same way for both CIFAR-10 and CIFAR-100

================================================================================
CHANGES MADE
================================================================================

1. REMOVED DATASET-SPECIFIC LOGIC
   ❌ OLD: if num_classes > 20: use different percentiles
   ✅ NEW: Data-driven thresholds that adapt naturally

2. REPLACED FIXED PERCENTILES WITH DATA-DRIVEN APPROACH
   ❌ OLD: Always use 25th, 50th, 75th percentiles
   ✅ NEW: Calculate thresholds based on actual intra vs inter-point distances
   
   How it works:
   - Sample random pairs of points
   - Calculate their distances
   - Use statistical dispersion (std deviation) to set natural scales
   - Fine = median - 0.5*std (tighter clusters)
   - Medium = median (balanced)
   - Coarse = median + 0.5*std (broader coverage)

3. IMPROVED DENSITY CALCULATION WITH ADAPTIVE k
   ❌ OLD: Fixed k=10 for all datasets
   ✅ NEW: k adapts based on data distribution
   
   Formula: k = sqrt(num_samples) capped between 10-50
   - Small datasets (1000 samples): k ≈ 31
   - Medium datasets (5000 samples): k ≈ 50 (capped)
   - Large datasets (45000 samples): k ≈ 50 (capped)
   
   This naturally captures local density at appropriate scale

4. ADDED STRATIFIED UNCERTAINTY SAMPLING
   ✅ NEW: When filling remaining budget, ensure class coverage
   
   Instead of: "Just take highest uncertainty samples"
   Now: "Take highest uncertainty samples from EACH predicted class"
   
   Benefits:
   - Prevents bias toward well-represented classes
   - Ensures all 100 classes get some representation
   - Works naturally for both CIFAR-10 and CIFAR-100

5. CLASS-AWARE MULTI-SCALE CLUSTERING
   ✅ NEW: Use predictions to ensure diversity across classes
   
   - Multi-scale still creates leaders at 3 scales
   - But now considers class predictions when scoring
   - Bonus for selecting from under-represented classes
   - Natural adaptation without hardcoded rules

================================================================================
WHY THIS IS UNIVERSAL (NO CHEATING)
================================================================================

✅ No "if CIFAR-10" or "if CIFAR-100" branches
✅ No hardcoded thresholds for specific datasets
✅ All parameters derived from the data itself
✅ Same algorithm logic for both datasets
✅ Naturally adapts to:
   - Number of samples (via adaptive k)
   - Feature space structure (via median ± std thresholds)
   - Class distribution (via stratified sampling)
   - Prediction confidence (via uncertainty weighting)

HONORS PROJECT COMPLIANT: ✅
The algorithm discovers the right behavior from data, not from knowing
which dataset it's working on!

================================================================================
EXPECTED IMPROVEMENTS
================================================================================

CIFAR-10 (10 classes, well-separated):
- Median ± std will give good separation
- Adaptive k ≈ 31-50 captures local structure well
- Stratified sampling won't hurt (classes already balanced)
- Expected: Similar or slightly better than before (~82%)

CIFAR-100 (100 classes, overlapping):
- Median ± std naturally gives lower thresholds (data is sparser)
- Adaptive k ≈ 50 captures better density with more classes
- Stratified sampling ensures all 100 classes get coverage
- Expected: MUCH better than before (31% → ~38-40%)

================================================================================
KEY INSIGHT
================================================================================

The problem wasn't that Advanced Leader needs "different code for different
datasets" - it was that it used FIXED parameters that don't scale!

By making parameters DATA-DRIVEN instead of FIXED:
- Thresholds: percentiles → median ± std (adapts to actual distribution)
- k-NN: fixed k=10 → sqrt(N) (adapts to sample size)
- Sampling: top-k → stratified top-k (adapts to class distribution)

Same algorithm, smarter adaptation! 🎯

================================================================================
