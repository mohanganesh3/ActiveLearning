# Complete Development Record: Advanced Leader Clustering
## Timeline from Bug Discovery to Universal Solution

**Project:** Honors Project - Active Learning with Advanced Leader Clustering  
**Datasets:** CIFAR-10 (10 classes) and CIFAR-100 (100 classes)  
**Goal:** Create a universal algorithm that works well on both datasets without dataset-specific logic

---

## Phase 1: Bug Discovery and Initial Fix

### Problem Discovered (October 27, 2025)
**Symptom:** Excessive sampling times (1000s+ seconds) on some rounds

**Root Cause Investigation:**
```python
# BUGGY CODE:
def _compute_multi_scale_thresholds(self, features):
    # Only 100 distance pairs computed
    distances = []
    for i in range(min(100, len(sample_features))):
        dists = np.linalg.norm(sample_features[i] - sample_features, axis=1)
        distances.extend(dists[dists > 0])
    
    median = np.median(distances)
    return [median * 0.5, median * 1.0, median * 1.5]  # BUG: median ≈ 0 → all thresholds = 0
```

**Why It Failed:**
- Small sample (100 pairs) gave inaccurate median
- When random sampling picked points from same cluster → median ≈ 0
- Zero thresholds → every point became a leader (45,000 instead of ~100)
- Result: 2847+ seconds per round

**Evidence from Old Logs:**
```
CIFAR-10:
  Round 2: Thresholds: ['0.000', '0.000', '0.000'] → 8,986 leaders → 2888s
  Round 5: Thresholds: ['0.000', '0.000', '0.000'] → 10,481 leaders → 3306s
  
CIFAR-100:
  Never went to zero (overlapping classes prevented it)
  But still had issues with poor accuracy
```

**First Fix (Percentile-Based):**
```python
def _compute_multi_scale_thresholds(self, features):
    # Compute 44,850 pairs from 300 samples
    from sklearn.metrics import pairwise_distances
    pdists = pairwise_distances(sample_features, metric='euclidean')
    triu_idx = np.triu_indices_from(pdists, k=1)
    distances = pdists[triu_idx]
    
    # Direct percentiles - no multiplication
    p25 = float(np.percentile(distances, 25))
    p50 = float(np.percentile(distances, 50))
    p75 = float(np.percentile(distances, 75))
    return [p25, p50, p75]
```

**Result:** Sampling time fixed (2888s → 82s) ✅

---

## Phase 2: Performance Gap Investigation

### Results After Bug Fix (October 27, 2025)

**CIFAR-10:** ✅ Excellent
- Final Accuracy: **82.12%** (vs 77.03% random)
- Improvement: **+5.09%** over random
- Best non-greedy strategy!

**CIFAR-100:** ❌ Catastrophic Failure
- Final Accuracy: **31.21%** (vs 35.82% random)
- Degradation: **-4.61%** WORSE than random
- Only strategy that underperforms random!

### Deep Investigation: Why CIFAR-100 Failed?

**Threshold Analysis:**
```
CIFAR-10: [1.7, 2.5, 3.3] → [5.4, 6.8, 8.2]  (avg ~5.5)
CIFAR-100: [3.0, 4.3, 6.1] → [9.4, 10.9, 12.5] (avg ~8.0)

→ CIFAR-100 thresholds 45% HIGHER (natural, not a bug!)
```

**Leader Count:**
```
CIFAR-10: 30-40 leaders per round (stable)
CIFAR-100: 100-120 leaders per round (3x more!)

→ More leaders = less selectivity = poorer quality
```

**Four Root Causes Identified:**

1. **Threshold Mismatch (60-75% higher on CIFAR-100)**
   - Fixed percentiles (25th, 50th, 75th) don't adapt to data structure
   - CIFAR-100 has naturally larger distances (100 classes, more spread)
   - Tighter clusters needed but thresholds are too coarse

2. **Leader Redundancy (3x more leaders)**
   - Multi-scale clustering at 3 thresholds
   - Each scale captures similar dense regions
   - Result: redundant samples, poor diversity

3. **Class Coverage Failure**
   - CIFAR-10: 10 classes × 500 samples = good coverage naturally
   - CIFAR-100: 100 classes × 50 samples = many classes get ZERO samples
   - Algorithm optimizes cluster diversity, NOT class diversity

4. **Round 9 Catastrophic Collapse**
   ```
   Round 8: 40.75% ✓
   Round 9: 31.21% ✗ (-9.54% DROP!)
   ```
   - Poor sample selection in final round
   - Model learned wrong patterns from biased samples

### Fundamental Issue Discovered

**The algorithm makes 4 implicit assumptions:**
1. Classes form well-separated clusters (TRUE for CIFAR-10, FALSE for CIFAR-100)
2. Fixed percentile thresholds capture structure (TRUE for 10 classes, FALSE for 100)
3. k=10 neighbors captures local density (TRUE for sparse, FALSE for overlapping)
4. Multi-scale adds diversity (TRUE for separated clusters, CREATES REDUNDANCY for overlapping)

**Key Insight:** Algorithm designed for coarse-grained problems, fails on fine-grained problems!

---

## Phase 3: Universal Improvements (Version 1)

### Design Principles (Honors Project Constraint!)
- ❌ NO dataset-specific logic (if CIFAR-10 do X, else do Y)
- ✅ Data-driven adaptation (learn from actual data characteristics)
- ✅ Same algorithm for both datasets
- ✅ Universal applicability

### Three Universal Changes Implemented (October 28, 2025)

#### Change 1: Adaptive Distance-Based Thresholds
**Problem:** Fixed percentiles (25, 50, 75) don't adapt to data spread

**Solution:** Statistical measures + CV-based adaptation
```python
def _compute_adaptive_thresholds(self, distances):
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    median_dist = np.median(distances)
    cv = std_dist / mean_dist  # Coefficient of variation
    
    # CV indicates data spread:
    # Low CV (< 0.3) = well-separated → use wider percentiles
    # High CV (> 0.5) = overlapping → use tighter percentiles
    
    if cv < 0.3:
        percentiles = [25, 50, 75]  # Standard
    elif cv < 0.5:
        percentiles = [22, 46, 71]  # Slightly tighter
    else:
        percentiles = [15, 35, 60]  # Much tighter
    
    fine = np.percentile(distances, percentiles[0])
    medium = np.percentile(distances, percentiles[1])
    coarse = np.percentile(distances, percentiles[2])
    
    return [fine, medium, coarse]
```

**Why It Works:** Automatically detects data structure from CV ratio, adapts thresholds accordingly

#### Change 2: Dynamic k for Density Estimation
**Problem:** Fixed k=10 doesn't scale with dataset complexity

**Solution:** Adaptive k based on data size
```python
def _compute_densities(self, features):
    # Scale k with square root of data size
    k = max(10, int(np.sqrt(len(features)) / 3))
    k = min(k, len(features) - 1)
    
    nbrs = NearestNeighbors(n_neighbors=k).fit(features)
    distances, _ = nbrs.kneighbors(features)
    densities = 1.0 / (distances.mean(axis=1) + 1e-10)
    
    return densities
```

**Why It Works:** 
- Small datasets → k ≈ 10-15 (captures local structure)
- Large datasets → k ≈ 30-50 (averages over more neighbors)
- Scales automatically with √N

#### Change 3: Class-Aware Selection for Coverage
**Problem:** No mechanism to ensure class diversity

**Solution:** Diversity bonus + stratified uncertainty filling
```python
# During leader scoring: Add diversity bonus
class_counts = {}
for idx in selected:
    pred_class = predictions[idx]
    class_counts[pred_class] = class_counts.get(pred_class, 0) + 1

for i in candidates:
    pred_class = predictions[i]
    class_frequency = class_counts.get(pred_class, 0)
    diversity_bonus = 1.0 / (1.0 + class_frequency)  # Inverse frequency
    score = density[i] * uncertainties[i] * (1.0 + diversity_bonus)

# During filling: Stratify by predicted class
def _fill_with_stratified_uncertainty(self, remaining, uncertainties, predictions, needed):
    class_buckets = {}
    for idx in remaining:
        pred_class = predictions[idx]
        if pred_class not in class_buckets:
            class_buckets[pred_class] = []
        class_buckets[pred_class].append((idx, uncertainties[idx]))
    
    samples_per_class = max(1, needed // len(class_buckets))
    selected = []
    for class_id, samples in class_buckets.items():
        samples.sort(key=lambda x: x[1], reverse=True)  # High uncertainty
        selected.extend([idx for idx, _ in samples[:samples_per_class]])
    
    return selected[:needed]
```

**Why It Works:** Uses model's own predictions (pseudo-labels) to ensure all classes represented

---

## Phase 4: Results Analysis (Version 1)

### CIFAR-100 Results (October 29, 2025)

**Comparison:**
```
OLD (Bug Fixed):     31.21% (-4.61% vs random) ❌
NEW (Universal V1):  39.61% (+3.79% vs random) ✅
Improvement:         +8.40% absolute gain
```

**Round-by-Round:**
```
Round   OLD     NEW     Δ
1       6.20%   6.20%   +0.00%  (same start)
2      15.86%  15.58%   -0.28%
3      17.56%  18.34%   +0.78%
4      29.34%  34.00%   +4.66%  ✅
5      32.60%  29.18%   -3.42%
6      36.44%  24.40%  -12.04%  ⚠️ volatility
7      39.64%  38.45%   -1.19%
8      40.75%  18.81%  -21.94%  ⚠️ major dip
9      31.21%  39.61%   +8.40%  ✅ OLD collapsed, NEW stable
```

### Success Metrics ✅

1. **Primary Goal Achieved:** CIFAR-100 now beats random!
2. **Round 9 Collapse Fixed:** No -9.54% drop
3. **Best Non-Greedy:** Beats Basic Leader (38.83%)
4. **Universal Algorithm:** Same code for both datasets
5. **Faster:** 76s vs 91s per round

### New Problem Discovered ⚠️

**Training Volatility:** Round 6 (24.40%) and Round 8 (18.81%) show unexpected drops

**Root Cause Investigation:**

1. **Over-Aggressive Thresholds:**
   ```
   OLD Round 3: [6.090, 7.391, 8.676] → 86 leaders
   NEW Round 3: [4.448, 5.719, 6.994] → 322 leaders (3.7x MORE!)
   ```
   - Lower thresholds → more leaders
   - More leaders → less selectivity → noise

2. **Imbalanced Selection:**
   ```
   Round 8:
   - Leaders selected: 282 (11% of 2500 budget)
   - Uncertainty filled: 2200 (89% of budget!)
   ```
   - Lost the benefit of leader-based diversity
   - Became mostly uncertainty sampling

3. **Inconsistent Leader Counts:**
   ```
   Round 2: 100 leaders
   Round 4: 375 leaders
   Round 8: 282 leaders
   ```
   - Wildly varying leader counts
   - Inconsistent sample quality across rounds

---

## Phase 5: Deep Analysis for Next Improvements

### What Works ✅

1. **Adaptive thresholds concept** - right idea, but needs tuning
2. **Stratified filling** - prevents collapse, ensures coverage
3. **Dynamic k** - scales appropriately with data size
4. **No dataset-specific code** - maintains honors project integrity

### What Needs Fixing ⚠️

1. **CV-based percentile selection is too aggressive**
   - Need smoother adaptation
   - Current: discrete jumps in percentiles
   - Better: continuous function of CV

2. **No minimum leader target**
   - Algorithm should maintain minimum leader ratio (e.g., 50% of budget)
   - If too few leaders, need to relax thresholds further

3. **Leader/Uncertainty balance not controlled**
   - Currently: whatever leaders we get + fill the rest with uncertainty
   - Better: Target ratio like 70% leaders + 30% uncertainty

4. **Threshold stability**
   - Thresholds vary too much between rounds
   - Need some temporal smoothing/momentum

### Hypothesis for Volatility

**Theory:** The volatility isn't necessarily BAD - it might indicate:
1. Algorithm exploring different parts of feature space
2. Some rounds get "lucky" samples, others don't
3. Model's feature representation changes as it trains

**Evidence:**
- Round 8: 18.81% (poor sample quality from 89% uncertainty filling)
- Round 9: 39.61% (better balance recovered performance)
- Final result is GOOD even with volatility!

**Question:** Is volatility a problem if final result is good?
- For research/production: Some volatility acceptable if converges well
- For honors project: Should demonstrate understanding and control

---

## Phase 6: Design for Next Iteration (Version 2)

### Goals for V2

1. ✅ Maintain universality (no dataset-specific code)
2. ✅ Reduce training volatility
3. ✅ Ensure minimum leader contribution (≥ 50% of budget)
4. ✅ Smoother threshold adaptation
5. ✅ Better leader/uncertainty balance

### Proposed Improvements

#### Improvement 1: Smooth CV-Based Threshold Adaptation
**Current Problem:** Discrete jumps in percentiles based on CV ranges

**Proposed Solution:** Continuous interpolation
```python
def _compute_adaptive_thresholds(self, distances):
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    cv = std_dist / mean_dist
    
    # Smooth interpolation instead of discrete buckets
    # cv = 0.2 → percentiles ≈ [30, 50, 70] (wide)
    # cv = 0.5 → percentiles ≈ [20, 40, 60] (medium)
    # cv = 0.8 → percentiles ≈ [15, 30, 50] (tight)
    
    # Linear interpolation based on CV
    cv_clamped = np.clip(cv, 0.2, 0.8)
    
    # Map CV to percentile ranges
    fine_p = 30 - (cv_clamped - 0.2) * 25    # 30 → 15
    medium_p = 50 - (cv_clamped - 0.2) * 33  # 50 → 30
    coarse_p = 70 - (cv_clamped - 0.2) * 33  # 70 → 50
    
    fine = np.percentile(distances, fine_p)
    medium = np.percentile(distances, medium_p)
    coarse = np.percentile(distances, coarse_p)
    
    return [fine, medium, coarse]
```

#### Improvement 2: Minimum Leader Target with Threshold Relaxation
**Current Problem:** Sometimes only 11% of budget comes from leaders

**Proposed Solution:** Iteratively relax thresholds if too few leaders
```python
def _ensure_minimum_leaders(self, features, target_budget):
    min_leader_ratio = 0.5  # Want at least 50% from leaders
    min_leaders = int(target_budget * min_leader_ratio)
    
    # Initial thresholds
    thresholds = self._compute_adaptive_thresholds(distances)
    leaders = self._multi_scale_clustering(features, thresholds, ...)
    
    # If too few leaders, relax thresholds
    attempts = 0
    while len(leaders) < min_leaders and attempts < 5:
        # Increase thresholds by 20% to include more points
        thresholds = [t * 1.2 for t in thresholds]
        leaders = self._multi_scale_clustering(features, thresholds, ...)
        attempts += 1
    
    return leaders, thresholds
```

#### Improvement 3: Controlled Leader/Uncertainty Balance
**Current Problem:** Imbalanced selection (89% uncertainty in some rounds)

**Proposed Solution:** Target fixed ratio
```python
def select(self, budget, features, predictions, uncertainties):
    target_leader_ratio = 0.7  # 70% from leaders
    target_uncertainty_ratio = 0.3  # 30% from uncertainty
    
    leader_budget = int(budget * target_leader_ratio)
    uncertainty_budget = budget - leader_budget
    
    # Select leaders up to leader_budget
    leaders = self._get_leaders_with_target(features, leader_budget, ...)
    
    # Fill remainder with stratified uncertainty
    remaining = [i for i in range(len(features)) if i not in leaders]
    uncertainty_samples = self._stratified_uncertainty(
        remaining, uncertainties, predictions, uncertainty_budget
    )
    
    return leaders + uncertainty_samples
```

#### Improvement 4: Threshold Momentum (Temporal Smoothing)
**Current Problem:** Thresholds can vary dramatically between rounds

**Proposed Solution:** Exponential moving average
```python
def __init__(self):
    self.prev_thresholds = None
    self.momentum = 0.3  # 30% weight to previous
    
def _compute_adaptive_thresholds(self, distances):
    # Compute new thresholds
    new_thresholds = self._compute_from_cv(distances)
    
    # Smooth with previous round
    if self.prev_thresholds is not None:
        smoothed = [
            self.momentum * prev + (1 - self.momentum) * new
            for prev, new in zip(self.prev_thresholds, new_thresholds)
        ]
    else:
        smoothed = new_thresholds
    
    self.prev_thresholds = smoothed
    return smoothed
```

### Expected Impact

**Smooth CV Adaptation:**
- Less aggressive threshold changes → fewer leaders → better selectivity
- Reduces from 322 leaders to ~150-200 leaders

**Minimum Leader Target:**
- Ensures at least 50% of samples come from diversity-based selection
- Prevents over-reliance on uncertainty sampling

**Controlled Balance:**
- Consistent 70/30 split across all rounds
- Maintains algorithm's core strength (diversity) while leveraging uncertainty

**Threshold Momentum:**
- Reduces round-to-round variance
- Smoother training progression

### Risks & Considerations

1. **May reduce peak performance:** Old version hit 40.75% (even if collapsed later)
2. **Might be over-engineered:** Current version already works (39.61%)
3. **Could harm CIFAR-10:** Changes optimized for CIFAR-100 might hurt CIFAR-10

**Mitigation:** Test on BOTH datasets, compare against Version 1

---

## Phase 7: Implementation Plan

### Step 1: Implement V2 Improvements
- Smooth CV-based thresholds
- Minimum leader target (50%)
- 70/30 leader/uncertainty balance
- Threshold momentum (30%)

### Step 2: Run Experiments on Both Datasets
- CIFAR-10: Verify still works well (≥ 80%)
- CIFAR-100: Test volatility reduction + maintain 39%+

### Step 3: Compare Results
- V1 vs V2 on both datasets
- Volatility metrics (std dev of accuracy changes)
- Final accuracy
- Leader count stability

### Step 4: Document Findings
- What improved, what didn't
- Trade-offs discovered
- Recommendations for future work

---

## Next Actions

1. ✅ Create this comprehensive record
2. ⏳ Implement Version 2 improvements
3. ⏳ Run both CIFAR-10 and CIFAR-100 with nohup
4. ⏳ Analyze results and compare V1 vs V2
5. ⏳ Update EMAIL_VERSION_3_MEDIUM.md with latest findings

---

**Record Created:** October 29, 2025  
**Status:** Ready to implement Version 2  
**Next Milestone:** Universal improvements with volatility reduction
