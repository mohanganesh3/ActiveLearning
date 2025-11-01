# Version 2 (V2) Experiment Plan

**Date:** October 29, 2025  
**Goal:** Reduce training volatility while maintaining final performance

---

## What Changed from V1 to V2

### V1 Results (Baseline for Comparison)
- **CIFAR-10:** 82.12% (+5.09% vs random) ✅ Excellent
- **CIFAR-100:** 39.61% (+3.79% vs random) ✅ Good, but volatile

### V1 Problems
1. High volatility (Round 6: 24.40%, Round 8: 18.81%)
2. Over-aggressive thresholds (3-4x more leaders than needed)
3. Imbalanced selection (88% uncertainty filling in some rounds)
4. No temporal smoothing between rounds

---

## V2 Improvements

### 1. Smooth CV-Based Threshold Adaptation
**Problem:** V1 had discrete jumps in percentiles  
**Solution:** Continuous interpolation

```
V1: if cv < 0.3: [15, 35, 60]  elif cv > 0.5: [25, 50, 75]  (JUMPS)
V2: smooth_interp(cv): [20, 40, 65] → [30, 55, 75]  (SMOOTH)
```

**Impact:** Fewer leaders, better quality, less noise

### 2. More Conservative Percentiles
**Problem:** V1 used [15, 35, 60] for low CV - too aggressive!  
**Solution:** V2 uses [20, 40, 65] for low CV - more selective

```
V1 Round 3: percentiles [15,35,60] → 322 leaders
V2 Round 3: percentiles [20,40,65] → expect ~150-200 leaders
```

**Impact:** Fewer but higher quality leaders

### 3. Temporal Smoothing (Momentum)
**Problem:** Thresholds varied wildly between rounds  
**Solution:** Exponential moving average (30% momentum)

```python
# V2 only
smoothed_threshold = 0.3 * prev_threshold + 0.7 * new_threshold
```

**Impact:** Smoother transitions, more consistent leader counts

### 4. Minimum Leader Target (50%)
**Problem:** V1 sometimes only got 11% leaders, 89% uncertainty  
**Solution:** Ensure at least 50% of budget from leaders

```
Target: 70% leaders (1750 samples)
Minimum: 50% leaders (1250 samples)
If too few: Relax thresholds iteratively (up to 5 attempts)
```

**Impact:** Maintains diversity benefit, prevents over-reliance on uncertainty

### 5. Controlled 70/30 Balance
**Problem:** Imbalanced selection ratio  
**Solution:** Explicit budget split

```
Leader budget: 70% of 2500 = 1750 samples
Uncertainty budget: 30% of 2500 = 750 samples
```

**Impact:** Consistent algorithm behavior across all rounds

---

## Expected Results

### CIFAR-10
- **V1:** 82.12% with low volatility
- **V2 Goal:** Maintain ≥81% (allow small drop for consistency)
- **V2 Expectation:** Should be very similar to V1 (already stable)

### CIFAR-100
- **V1:** 39.61% final, but volatile (18.81% dip in Round 8)
- **V2 Goal:** Reduce volatility, maintain ≥38%
- **V2 Expectation:** 
  - Less dramatic swings (no 18% dips)
  - More monotonic increase
  - Final accuracy: 38-40%
  - Standard deviation of round-to-round changes: < 5%

---

## Success Criteria

### Primary Goal ✅
**Reduce CIFAR-100 volatility while maintaining performance**
- Standard deviation of accuracy changes: V1 = 8.7%, V2 target < 6%
- No rounds below 25% accuracy
- Final accuracy ≥ 38%

### Secondary Goal ✅
**Verify CIFAR-10 still works well**
- Final accuracy ≥ 80%
- Maintain leader-based advantage

### Bonus 🎯
**Universal improvements benefit both datasets**
- Both datasets show reduced volatility
- Both maintain or improve final accuracy

---

## Metrics to Track

### 1. Accuracy Metrics
- Final accuracy (Round 9)
- Peak accuracy (best round)
- Average accuracy (Rounds 2-9)
- Minimum accuracy (worst round)

### 2. Volatility Metrics
- Standard deviation of round-to-round changes
- Max drop between consecutive rounds
- Number of rounds with negative change

### 3. Leader Metrics
- Average leader count per round
- Leader count stability (std dev)
- Leader/uncertainty ratio per round

### 4. Comparison Metrics
- V2 vs V1 final accuracy
- V2 vs V1 volatility reduction
- V2 vs Random baseline

---

## How to Analyze Results

### After Experiments Complete

1. **Extract Accuracy Curves**
```bash
grep "Test Accuracy:" logs_v2/advanced_leader_cifar10_*.log
grep "Test Accuracy:" logs_v2/advanced_leader_cifar100_*.log
```

2. **Compare Leader Counts**
```bash
grep "Candidate leaders:" logs_v2/advanced_leader_cifar100_*.log
grep "Selected.*leaders" logs_v2/advanced_leader_cifar100_*.log
```

3. **Check Threshold Smoothing**
```bash
grep "Smoothed:" logs_v2/advanced_leader_cifar100_*.log
```

4. **Verify Minimum Leader Target**
```bash
grep "Relaxed thresholds" logs_v2/advanced_leader_cifar100_*.log
```

### Comparison Script
```python
v1_cifar100 = [6.20, 15.58, 18.34, 34.00, 29.18, 24.40, 38.45, 18.81, 39.61]
v2_cifar100 = [...]  # Extract from logs

import numpy as np

# Volatility: std of changes
v1_changes = np.diff(v1_cifar100)
v2_changes = np.diff(v2_cifar100)

print(f"V1 volatility (std of changes): {np.std(v1_changes):.2f}")
print(f"V2 volatility (std of changes): {np.std(v2_changes):.2f}")
print(f"Reduction: {(np.std(v1_changes) - np.std(v2_changes)):.2f}")
```

---

## Timeline

- **Start:** October 29, 2025
- **Duration:** ~4 hours per dataset
- **Completion:** ~8 hours total
- **Analysis:** After completion

---

## Risk Mitigation

### Risk 1: V2 Worse Than V1
**If final accuracy drops significantly (< 37% CIFAR-100):**
- Percentiles may be TOO conservative
- Solution: Adjust to [18, 38, 63] (between V1 and V2)

### Risk 2: Still Volatile
**If volatility not reduced:**
- May need stronger momentum (40-50% instead of 30%)
- May need more aggressive min_leader_target (60% instead of 50%)

### Risk 3: CIFAR-10 Degrades
**If CIFAR-10 drops below 80%:**
- Conservative percentiles may hurt well-separated data
- Solution: Make CV threshold more sensitive (transition at cv=0.6 instead of 0.5)

---

## After Analysis: Next Steps

### If V2 Successful (volatility < 6%, accuracy ≥ 38%)
✅ Document improvements  
✅ Update EMAIL_VERSION_3_MEDIUM.md with V2 results  
✅ Recommend V2 as final implementation  

### If V2 Partially Successful (reduced volatility, but accuracy dropped)
⚙️ Tune percentile ranges  
⚙️ Adjust momentum parameter  
⚙️ Run V2.1 with refined parameters  

### If V2 Unsuccessful (no improvement)
🔄 Return to V1 as best version  
📝 Document why volatility is acceptable trade-off  
💡 Suggest volatility is inherent to adaptive sampling  

---

**Current Status:** Ready to launch  
**Command:** `./run_v2_experiments.sh`
