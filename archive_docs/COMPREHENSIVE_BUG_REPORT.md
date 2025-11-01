# 🚨 COMPREHENSIVE BUG REPORT - Round 2 Investigation

## CRITICAL ISSUES FOUND

### 🔴 BUG #1: CIFAR-100 Budget Mismatch
**Location**: Run scripts or experiment setup
**Issue**: Labeled size increases by 2500 each round, but then jumps to 25000 (max)
**Evidence**:
```
Expected progression: 5000 → 7500 → 10000 → 12500 → 15000 → 17500 → 20000 → 22500 → 25000
Actual progression:   5000 → 10000 → 12500 → 15000 → 17500 → 20000 → 22500 → 25000 → 25000
                            ^^^^^ WRONG! Should be 7500
```

**Root Cause**: Budget might be 2500 but first round selects 5000 instead, OR there's a cap at 25000

---

### 🔴 BUG #2: CIFAR-10 Labeled Size Cap
**Issue**: Final labeled size is 10000 instead of 11000
**Evidence**: All strategies show `Round 10: Expected 11000, Got 10000`

**Root Cause**: Total dataset size is 50000, but experiments might cap at 10000

---

### 🔴 BUG #3: Random Sampling Time = 0.00s
**Issue**: Random sampling shows 0.00s for all rounds
**Evidence**:
```
Random (CIFAR-10):  Sampling Time (avg): 0.00s
Random (CIFAR-100): Sampling Time (avg): 0.00s
```

**Root Cause**: Random sampling is so fast it's below timing resolution, OR timing isn't being recorded

---

### 🔴 BUG #4: Advanced Leader Accuracy Catastrophe (CIFAR-10)
**Issue**: Final accuracy is 49.96% - WORST of all strategies!
**Evidence**:
```
Random:           65.47% ✅
Leader:           67.17% ✅
Greedy K-Center:  69.17% ✅
Advanced Leader:  49.96% ❌ TERRIBLE!
```

**Multiple Accuracy Drops**:
- Round 4: 40.08% → 12.62% (massive 27% drop!)
- Round 6: 62.38% → 54.67%
- Round 9: 62.67% → 49.96%

**This is CRITICAL** - Advanced Leader should be BEST, not WORST!

---

### 🟡 BUG #5: Leader Clustering Accuracy Drop
**Issue**: Round 2: 20.90% → 10.00% (major drop)

---

### 🟡 BUG #6: Greedy K-Center Accuracy Drops
**Issues**:
- Round 3: 34.04% → 28.66%
- Round 4: 28.66% → 18.12%

---

## HYPOTHESIS: Why Advanced Leader Failed

Looking at the accuracy drops in Round 4 (40% → 12%), this suggests:

1. **Bad sample selection** - Selected 1000 samples that don't help (or hurt)
2. **Duplicate selection** - Might be selecting already-labeled samples
3. **Class imbalance** - Selecting all from one class
4. **Bug in selection logic** - Index mapping error

Let me investigate the selection logic...

