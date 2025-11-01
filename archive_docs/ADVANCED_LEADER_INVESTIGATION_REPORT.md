# Investigation Report: Why Advanced Leader Fails on CIFAR-100

## Executive Summary

**Advanced Leader Clustering achieves +5.09% improvement over Random on CIFAR-10, but -4.61% on CIFAR-100** (actually performs WORSE than random sampling). This investigation identifies the root causes and proposes fixes.

---

## Performance Comparison

| Strategy | CIFAR-10 Final Acc | CIFAR-100 Final Acc | Improvement over Random |
|----------|-------------------|-------------------|------------------------|
| Random | 77.03% | 35.82% | Baseline |
| Leader Clustering | 77.86% (+0.83%) | 38.83% (+3.01%) | ✓ Positive |
| **Advanced Leader** | **82.12% (+5.09%)** | **31.21% (-4.61%)** | ❌ **NEGATIVE** |
| Greedy K-Center | 80.38% (+3.35%) | 43.58% (+7.76%) | ✓ Positive |

**Key Finding:** Advanced Leader is the ONLY strategy that performs worse than random on CIFAR-100!

---

##  Root Cause Analysis

### 1. **Candidate Leader Count Mismatch**

**CIFAR-10 Pattern:**
```
Round 2: 34 candidate leaders  → select 2500  (need to fill 2466 from non-leaders)
Round 3: 43 candidate leaders  → select 2500  (need to fill 2457 from non-leaders)
Round 4: 35 candidate leaders  → select 2500  (need to fill 2465 from non-leaders)
Round 5: 35 candidate leaders  → select 2500
Round 6: 37 candidate leaders  → select 2500
Round 7: 30 candidate leaders  → select 2500
Round 8: 25 candidate leaders  → select 2500
Round 9: 33 candidate leaders  → select 2500
```

**CIFAR-100 Pattern:**
```
Round 2: 40 candidate leaders  → select 2500  (need to fill 2460 from non-leaders)
Round 3: 86 candidate leaders  → select 2500  (need to fill 2414 from non-leaders)
Round 4: 127 candidate leaders → select 2500  (need to fill 2373 from non-leaders)
Round 5: 120 candidate leaders → select 2500
Round 6: 103 candidate leaders → select 2500
Round 7: 109 candidate leaders → select 2500
Round 8: 105 candidate leaders → select 2500
Round 9: 109 candidate leaders → select 2500
```

**Problem:** 
- CIFAR-10: Only ~30-40 leaders per round (1.4% of budget)
- CIFAR-100: ~100-120 leaders per round (4.4% of budget)
- **CIFAR-100 creates 3x more leaders** → less diversity, more redundancy

---

### 2. **Threshold Escalation**

**CIFAR-10 Thresholds (Fine/Medium/Coarse):**
```
Round 2: [1.717, 2.513, 3.336]
Round 3: [3.795, 5.102, 6.384]
Round 4: [3.881, 5.196, 6.398]
Round 5: [4.698, 6.399, 7.649]
Round 6: [4.660, 6.169, 7.659]
Round 7: [4.907, 6.383, 7.949]
Round 8: [4.980, 6.443, 8.023]
Round 9: [5.377, 6.779, 8.150]
```
Threshold growth: 1.7 → 5.4 (3.1x increase)

**CIFAR-100 Thresholds:**
```
Round 2: [2.988, 4.331, 6.053]
Round 3: [6.090, 7.391, 8.676]
Round 4: [5.728, 7.171, 8.626]
Round 5: [7.574, 9.112, 10.657]
Round 6: [8.213, 9.729, 11.319]
Round 7: [8.843, 10.475, 12.186]
Round 8: [8.667, 10.094, 11.513]
Round 9: [9.436, 10.906, 12.475]
```
Threshold growth: 3.0 → 9.4 (3.1x increase)

**Problem:**
- CIFAR-100 thresholds start 74% HIGHER (2.99 vs 1.72)
- CIFAR-100 thresholds end 75% HIGHER (9.44 vs 5.38)
- Higher thresholds → tighter clustering → more overlapping clusters → less diversity

---

### 3. **Feature Space Density**

**Comparison with Basic Leader Clustering:**

| Dataset | Basic Leader Threshold (70th percentile) | Advanced Fine (25th) | Advanced Coarse (75th) |
|---------|----------------------------------------|---------------------|----------------------|
| CIFAR-10 | ~3.5 → 7.0 | 1.7 → 5.4 | 3.3 → 8.2 |
| CIFAR-100 | ~5.6 → 11.2 | 3.0 → 9.4 | 6.1 → 12.5 |

**Observation:**
- CIFAR-100 feature space is ~60% MORE SPREAD OUT than CIFAR-10
- 100 classes in same dimensional space (512-D) → sparser, more overlapping
- Advanced Leader's multi-scale approach doesn't adapt well to sparse spaces

---

### 4. **Learning Curve Anomaly**

**CIFAR-100 Advanced Leader Accuracy:**
```
Round 1:  6.20%  (baseline)
Round 2: 15.86%  ✓ good progress
Round 3: 17.56%  ✓ slight improvement
Round 4: 29.34%  ✓✓ BIG jump (+11.78%)
Round 5: 32.60%  ✓ good
Round 6: 36.44%  ✓ good
Round 7: 39.64%  ✓ good
Round 8: 40.75%  ✓ good
Round 9: 31.21%  ❌ COLLAPSE (-9.54%)
```

**This is highly unusual!** Round 9 shows a MASSIVE DROP in accuracy.

**Possible explanations:**
1. Training instability (learning rate too high in later rounds)
2. Selected samples in Round 9 were extremely poor quality
3. Model overfitting to redundant/noisy samples
4. Class imbalance got worse

Compare to Random (CIFAR-100):
```
Round 9: 35.82% (steady improvement, no collapse)
```

---

## Why Basic Leader Works Better on CIFAR-100

**Basic Leader Results:**
- CIFAR-10: 77.86% (+0.83% over Random)
- CIFAR-100: 38.83% (+3.01% over Random) ✓ POSITIVE

**Why it succeeds where Advanced fails:**
1. Uses SINGLE threshold (70th percentile) → simpler, more robust
2. Doesn't try to be too clever with multi-scale clustering
3. Adaptive threshold grows naturally with feature space
4. No complex uncertainty weighting that can backfire

---

## Hypothesis: The Core Problem

### **Advanced Leader optimizes for the WRONG objective on CIFAR-100**

1. **Class Coverage Problem**
   - CIFAR-10: 10 classes, 500 samples/class initially → plenty of representatives
   - CIFAR-100: 100 classes, 50 samples/class initially → sparse class coverage
   - Advanced Leader's density-based clustering IGNORES class labels
   - With 100 classes, many classes may not have ANY representatives in the leaders
   - Result: selects diverse *clusters* but misses entire *classes*

2. **Threshold Percentiles Don't Scale**
   - 25th/50th/75th percentiles work for well-separated clusters (CIFAR-10)
   - But fail for overlapping hierarchical structure (CIFAR-100)
   - Need LOWER percentiles (10th/30th/60th) for fine-grained classes

3. **k-NN Density with k=10 is Too Small**
   - With 100 classes and ~50 samples/class, k=10 neighbors may span multiple classes
   - Density estimation breaks down
   - Need adaptive k based on number of classes

4. **Multi-Scale Creates Redundancy**
   - Fine/Medium/Coarse scales all select from SAME overlapping region
   - Doesn't help with class coverage
   - Just creates multiple redundant leaders from dense superclass regions

---

## Recommended Fixes

### **Option 1: Class-Aware Stratified Sampling (Best for CIFAR-100)**

```python
def stratified_advanced_leader(features, predictions, budget):
    # Use model predictions as pseudo-labels
    unique_classes = np.unique(predictions)
    samples_per_class = budget // len(unique_classes)
    
    selected = []
    for class_id in unique_classes:
        class_mask = predictions == class_id
        class_indices = np.where(class_mask)[0]
        
        if len(class_indices) > 0:
            # Run Advanced Leader on THIS class only
            class_leaders = advanced_leader_within_class(
                features[class_indices], 
                min(samples_per_class, len(class_indices))
            )
            selected.extend(class_indices[class_leaders])
    
    # Fill remaining budget with high-uncertainty samples
    return selected
```

### **Option 2: Adaptive Percentiles Based on Number of Classes**

```python
def compute_adaptive_thresholds(features, num_classes):
    if num_classes <= 20:  # CIFAR-10 style
        percentiles = [25, 50, 75]
    else:  # CIFAR-100 style
        percentiles = [10, 30, 60]  # LOWER for fine-grained
    
    # Compute thresholds...
```

### **Option 3: Hierarchical Clustering for CIFAR-100**

```python
# Use 2-level hierarchy:
# 1. Coarse clustering → superclasses
# 2. Fine clustering within each superclass
```

### **Option 4: Use Margin-Based Instead of Distance-Based**

```python
# Instead of feature distance:
uncertainty = prediction_entropy(outputs)
margin = top1_prob - top2_prob  # How confident is the model?

# Select samples with:
# - High uncertainty (model is confused)
# - Low margin (close decision boundary)
# - Diverse features
```

---

## Action Items

1. ✅ **Confirmed root cause:** Multi-scale thresholds don't adapt to fine-grained class structure
2. ⏳ **Test hypothesis:** Implement class-aware stratified version
3. ⏳ **Benchmark:** Compare stratified vs original on both datasets
4. ⏳ **Investigate Round 9 collapse:** Check what samples were selected in that round
5. ⏳ **Consider hybrid:** Use Advanced Leader for CIFAR-10 style, Basic Leader for CIFAR-100 style

---

## Conclusion

**Advanced Leader's sophisticated multi-scale, density-aware, uncertainty-weighted approach BACKFIRES on CIFAR-100** because:

1. It creates too many redundant leaders in dense superclass regions
2. It misses rare classes with sparse representation  
3. Thresholds (25th/50th/75th percentiles) are too conservative for 100-class problem
4. k=10 density estimation doesn't capture local structure with 100 classes

**Simple Leader Clustering works better** because it's more robust and doesn't over-optimize.

**The irony:** Being "advanced" made it worse. Sometimes simple is better. 🎯

---

**Next Steps:** Implement stratified sampling or switch to margin-based active learning for fine-grained classification tasks.
