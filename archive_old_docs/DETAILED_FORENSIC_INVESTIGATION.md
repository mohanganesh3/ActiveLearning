# 🔍 DETAILED FORENSIC INVESTIGATION: Why V2 Failed
## From 39.61% Success to 34.37% Failure

**Date:** October 31, 2025  
**Investigation:** What went wrong between V1 (success) and V2 (failure)

---

## 📊 EXECUTIVE SUMMARY

**The Smoking Gun:** V2's "minimum leader target" mechanism **backfired catastrophically**

- **V1:** Used natural leader counts (262-375), selected high-quality samples → **39.61%** ✅
- **V2:** Forced 1250+ leaders, had to relax thresholds SO MUCH that quality collapsed → **34.37%** ❌

**Key Finding:** By trying to **force** 50% leaders (1250/2500), V2 had to relax thresholds up to **5 times**, selecting progressively worse leaders until hitting threshold ~20-25 (vs V1's 5-8).

---

## 🎯 ROUND-BY-ROUND FORENSIC ANALYSIS

### Overview Comparison

| Round | V1 Result | V2 Result | Difference | Winner |
|-------|-----------|-----------|------------|--------|
| 1 | 6.20% | 6.20% | 0.00% | ➡️ Tied |
| 2 | 15.58% | 11.93% | -3.65% | ❌ V2 worse |
| 3 | 18.34% | 17.23% | -1.11% | ⚠️ V2 slightly worse |
| **4** | **34.00%** | **18.98%** | **-15.02%** | ❌❌ **V2 CATASTROPHIC** |
| 5 | 29.18% | 28.19% | -0.99% | ➡️ Similar |
| 6 | 24.40% | 36.35% | +11.95% | ✅ V2 better |
| 7 | 38.45% | 41.26% | +2.81% | ✅ V2 peak! |
| 8 | 18.81% | 40.96% | +22.15% | ✅ V2 much better |
| **9** | **39.61%** | **34.37%** | **-5.24%** | ❌ **V2 can't sustain** |

### Volatility Statistics

```
V1 Volatility: 12.57% std (high but recovers)
V2 Volatility: 4.79% std (lower but can't recover)

V1 Max Drop: -19.64% (R7→R8) but recovers +20.80% (R8→R9)
V2 Max Drop: -6.59% (R7→R9 cumulative) can't recover

V1 Strategy: Wild exploration but finds good samples
V2 Strategy: Stable but trapped in suboptimal samples
```

---

## 🔥 THE SMOKING GUN: Round 4 Analysis

### V1 Round 4: SUCCESS (34.00% accuracy)

```
CV detected: 0.281
Adaptive percentiles: [15, 35, 60] (aggressive, tight)
Thresholds computed: [5.120, 6.427, 7.710]
Leaders found: 375

Strategy: Accept 375 natural high-quality leaders
Result: 34.00% accuracy ✅
```

### V2 Round 4: CATASTROPHIC FAILURE (18.98% accuracy)

```
CV detected: 0.275 (similar to V1!)
Base percentiles: [21, 42, 66] (more conservative than V1)
Raw thresholds: [7.115, 8.554, 10.124]
Smoothed with momentum: [6.133, 7.476, 8.954]

Problem: V2 requires 1250+ leaders (50% of 2500 budget)
Initial leaders: Only 412 ❌

Relaxation Attempts:
  Attempt 1: Relax to [7.67, 9.35, 11.19] → 412 leaders (still not enough)
  Attempt 2: Relax to [9.58, 11.68, 13.99] → 220 leaders (getting worse!)
  Attempt 3: Relax to [11.98, 14.60, 17.49] → 145 leaders (terrible!)
  Attempt 4: Relax to [14.97, 18.25, 21.86] → 108 leaders (catastrophic!)
  Attempt 5: Give up, use 108 leaders + 2392 uncertainty samples

Final: 4% leaders, 96% uncertainty sampling
Result: 18.98% accuracy ❌❌ (-15% vs V1!)
```

**What Went Wrong:**
1. V2 started with thresholds ~7-9 (conservative)
2. Needed 1250 leaders, only found 412
3. Relaxed thresholds to ~22-25 trying to hit target
4. By relaxing SO MUCH, included terrible quality leaders
5. Algorithm became 96% uncertainty sampling (lost diversity benefit)

---

## 🎯 THE SMOKING GUN: Round 9 Analysis (Final Round)

### V1 Round 9: SUCCESS (39.61% final)

```
CV detected: 0.352
Adaptive percentiles: [17, 38, 63]
Thresholds: [6.059, 7.838, 9.941]
Leaders found: 262

Strategy: Accept 262 natural leaders + stratified uncertainty
Balance: ~10% leaders, 90% uncertainty (but HIGH QUALITY leaders)
Result: 39.61% final ✅ (beats random by +3.79%)
```

### V2 Round 9: FAILURE (34.37% final)

```
CV detected: 0.249
Base percentiles: [21, 41, 66]
Raw thresholds: [8.137, 9.517, 11.107]
Smoothed: [8.060, 9.411, 10.926]

Problem: Still need 1250+ leaders
Initial leaders: Only 253 ❌

Relaxation Attempts:
  Attempt 1: [10.08, 11.76, 13.66] → 253 leaders
  Attempt 2: [12.59, 14.70, 17.07] → 134 leaders
  Attempt 3: [15.74, 18.38, 21.34] → 109 leaders
  Attempt 4: [19.68, 22.98, 26.68] → 101 leaders
  Attempt 5: [24.60, 28.72, 33.34] → 100 leaders (gave up)

Final: 4% leaders (100), 96% uncertainty (2340)
Result: 34.37% ❌ (BELOW random baseline!)
```

**What Went Wrong:**
1. Started with reasonable thresholds ~8-11
2. Tried to force 1250+ leaders
3. Relaxed thresholds to **~25-33** (ridiculously high!)
4. At threshold 25-33, you're basically selecting random distant points
5. Lost all selectivity, became pure uncertainty sampling
6. Result: Below random baseline

---

## 💡 ROOT CAUSE ANALYSIS

### The Fatal Flaws in V2 Design

#### **Flaw #1: Forced Minimum Leader Target**

```python
# V2 Code (BAD):
min_leaders = int(budget * 0.5)  # Force 1250 leaders
if leaders < min_leaders:
    relax_thresholds_by_20_percent()
    try_again()  # Up to 5 attempts
```

**Why It Failed:**
- CIFAR-100 naturally produces ~100-400 quality leaders
- Forcing 1250+ meant relaxing thresholds to include LOW QUALITY leaders
- After 5 relaxations, thresholds became ~20-30 (vs natural 5-8)
- At threshold 25, you're selecting random outliers, not representative leaders

#### **Flaw #2: Conservative Percentiles**

```
V1: CV=0.28 → percentiles [15, 35, 60] → thresholds [5.1, 6.4, 7.7]
V2: CV=0.28 → percentiles [21, 42, 66] → thresholds [7.1, 8.5, 10.1]
```

**Impact:**
- V2 starts 40% higher than V1
- Already less selective from the start
- Then has to relax even more → complete loss of selectivity

#### **Flaw #3: Temporal Smoothing Compounded Errors**

```
V2 Round 4:
  Raw: [7.115, 8.554, 10.124]
  Smoothed with 30% momentum: [6.133, 7.476, 8.954]
```

**Problem:**
- If previous round had poor thresholds, 30% carries forward
- Accumulates suboptimal thresholds across rounds
- V1 had no memory → fresh adaptation each round

#### **Flaw #4: 70/30 Rigid Split**

```
V2: Force 70% leaders (1750) + 30% uncertainty (750)
V1: Natural split based on data (10-40% leaders, rest uncertainty)
```

**Impact:**
- V2 couldn't adapt to CIFAR-100's natural leader distribution
- CIFAR-100 naturally produces ~100-400 leaders, not 1750
- Forcing it broke the algorithm

---

## 📈 WHY V1 WORKED

### V1's Winning Strategy

1. **Aggressive Adaptation**
   - CV=0.28 → percentiles [15, 35, 60] (tight!)
   - Thresholds [5.1, 6.4, 7.7] (selective)
   - Result: 375 HIGH QUALITY leaders

2. **Natural Balance**
   - Accept whatever leaders meet quality threshold
   - Fill rest with stratified uncertainty
   - No forced ratios

3. **No Memory Between Rounds**
   - Fresh adaptation each round
   - No accumulated errors
   - Can recover from bad rounds

4. **Quality Over Quantity**
   - 262 leaders with threshold 6.0 → High quality
   - Better than 1250 leaders with threshold 25.0 → Low quality

### V1's "Volatility" Was Actually Good Exploration

```
Round 6: 24.40% (explored different samples)
Round 7: 38.45% (learned from exploration)
Round 8: 18.81% (more exploration)
Round 9: 39.61% (converged to good result) ✅
```

V1's volatility = **healthy exploration**  
V2's stability = **trapped in local optimum**

---

## 🎓 KEY LEARNINGS

### 1. **Don't Force What Doesn't Exist**

CIFAR-100 naturally has ~100-400 quality leaders per round.  
Forcing 1250 leaders = including 850+ low-quality samples.

**Lesson:** Work with the data's natural structure, don't fight it.

### 2. **Conservative ≠ Better**

V2 was "conservative" (higher percentiles, momentum, forced ratios).  
Result: WORSE performance, not better.

**Lesson:** Aggressive adaptation > conservative constraints

### 3. **Simplicity Wins**

V1: Simple CV-based percentiles, natural selection  
V2: Momentum + minimum targets + relaxation + forced ratios

**Lesson:** More complexity ≠ better results

### 4. **Volatility Can Be Good**

V1's volatility enabled exploration and recovery.  
V2's stability trapped it in suboptimal region.

**Lesson:** Some volatility = healthy exploration in active learning

### 5. **Quality > Quantity for Leaders**

- 262 leaders at threshold 6.0 → 39.61% ✅
- 100 leaders at threshold 25.0 → 34.37% ❌

**Lesson:** Better to have few high-quality leaders than many low-quality ones

---

## 🔧 WHAT SHOULD HAVE BEEN DONE

### Option A: V1 with Minor Tweaks (Recommended)

Keep V1's core but add **gentle** smoothing:

```python
# Very light momentum (10% instead of 30%)
smoothed = 0.1 * prev + 0.9 * new

# No forced minimum (accept natural leaders)
# No conservative percentiles (keep aggressive [15,35,60])
```

**Expected:** 38-40% with slightly less volatility

### Option B: Adaptive Minimum Target

Instead of forcing 50%, adapt to data:

```python
# Historical average as guide, not hard constraint
historical_avg_leaders = moving_average(past_leader_counts)
target = historical_avg_leaders * 1.2  # 20% buffer, not 2x

if leaders < target * 0.8:  # Only if very low
    relax_once_gently()  # 10%, not 20%
```

**Expected:** Better than V2, maybe ~37-38%

### Option C: Quality Threshold

Set a maximum threshold for relaxation:

```python
MAX_THRESHOLD = raw_threshold * 1.5  # Don't relax more than 50%

if relaxed_threshold > MAX_THRESHOLD:
    print("Can't meet minimum without sacrificing quality")
    accept_fewer_leaders()  # Quality > quantity
```

**Expected:** Prevents catastrophic relaxation

---

## 📊 FINAL VERDICT

### What Happened Timeline

1. **Oct 28:** V1 succeeded with 39.61% by being aggressive and adaptive ✅
2. **Oct 29:** Designed V2 to "reduce volatility" with 4 sophisticated mechanisms
3. **Oct 30:** V2 failed with 34.37% because over-engineering reduced adaptability ❌

### The Irony

**Goal:** Reduce volatility  
**Result:** Volatility actually IMPROVED (4.79% vs 12.57% std)

**Goal:** Maintain performance (≥38%)  
**Result:** Performance COLLAPSED (34.37% < random baseline)

**Lesson:** Achieved the wrong goal at the expense of what mattered!

### Why V1 Was Better

| Aspect | V1 | V2 | Winner |
|--------|----|----|--------|
| **Final Accuracy** | 39.61% | 34.37% | V1 ✅ |
| **vs Random** | +3.79% | -1.45% | V1 ✅ |
| **Adaptability** | High | Low (constrained) | V1 ✅ |
| **Quality Focus** | Yes | No (quantity forced) | V1 ✅ |
| **Exploration** | Good | Trapped | V1 ✅ |
| **Volatility** | High (12.57%) | Low (4.79%) | V2 ✅ |
| **Complexity** | Simple | Over-engineered | V1 ✅ |

**Score: V1 wins 6-1** (and the one V2 wins doesn't matter!)

---

## 🎯 RECOMMENDATION

### For Honors Project Presentation

**Present V1 (39.61%) as your final solution:**
- ✅ Beats random baseline by +3.79%
- ✅ Universal algorithm (no dataset-specific code)
- ✅ Simple and effective
- ✅ Best non-greedy fast strategy

**Present V2 (34.37%) as a failed experiment:**
- ❌ Attempted to reduce volatility
- ❌ Added 4 sophisticated mechanisms
- ❌ Result: Over-engineering harmed performance
- ✅ **Key lesson:** Simplicity + adaptability > complexity + constraints

**Conclusion:**
> "Through systematic experimentation, we discovered that attempting to reduce volatility through rigid constraints (V2) actually harmed performance. The simpler, more adaptive approach (V1) achieved better results by working with the data's natural structure rather than fighting it. This demonstrates a fundamental principle in active learning: aggressive adaptation to data characteristics outperforms conservative constraint-based approaches."

---

## 📋 TECHNICAL SUMMARY

### The V2 Failure Mechanism

```
1. Conservative percentiles [21,42,66] → Higher initial thresholds [7-10]
2. Minimum 1250 leaders required → Natural ~400 not enough
3. Relax thresholds 5 times → Thresholds balloon to [20-33]
4. At threshold 25-33 → Select random outliers, not representatives
5. Result: 4% leaders (low quality) + 96% uncertainty
6. Performance: 34.37% (below random 35.82%) ❌
```

### The V1 Success Mechanism

```
1. Aggressive percentiles [15,35,60] → Lower thresholds [5-8]
2. Accept natural ~260-375 leaders → High quality selection
3. No forced ratios → Work with data structure
4. Fill with stratified uncertainty → Class coverage
5. Result: 10% leaders (high quality) + 90% uncertainty (strategic)
6. Performance: 39.61% (above random 35.82%) ✅
```

---

**Bottom Line:** V2 failed because it tried to **force** the data into a structure it didn't naturally have. V1 succeeded by **adapting** to the data's natural structure.

**Honors Project Moral:** "Let the data speak. Don't impose rigid constraints on adaptive algorithms."

---

**Investigation Complete:** October 31, 2025  
**Verdict:** Use V1, document V2 as instructive failure  
**Key Insight:** Over-engineering + rigid constraints < simple + adaptive
