# 🔧 ALL BUGS FOUND - COMPREHENSIVE FIX PLAN

## CRITICAL BUGS

### 🔴 BUG #1: Round 10 Retrains with Same Data (CIFAR-10)
**Location**: `cifar10_experiment.py` line 153
**Issue**: Last round doesn't select new samples BUT still retrains model from scratch
**Impact**: 
- Accuracy drops because model is reinitialized
- Wastes computation time
- Final accuracy is unreliable

**Fix**: Skip retraining in last round OR continue sampling

---

### 🔴 BUG #2: Advanced Leader Zero-Threshold = Bad Samples
**Location**: `active_learning_strategies.py` - `_compute_multi_scale_thresholds`
**Issue**: When thresholds become [0.0, 0.0, 0.0], ALL points become leaders
**Impact**:
- Selects 10,000+ "leaders" instead of budget (1000)
- Then `_score_and_select` picks top 1000 from these
- But the scoring is biased/wrong when all are leaders
- Results in BAD sample selection → accuracy drops 27%

**Evidence**: Round 5 had 3306s sampling (zero threshold bug) → Round 5 accuracy = 12.62% (massive drop from 40%)

**Fix**: Already applied safety checks, but need to verify selection quality

---

### 🔴 BUG #3: CIFAR-100 Budget Inconsistency
**Location**: Unknown - need to find where first round selects wrong amount
**Issue**: Round 1 selects 5000 samples instead of 2500
**Impact**: All subsequent rounds have wrong labeled sizes

**Fix**: Investigate CIFAR-100 experiment setup

---

### 🔴 BUG #4: Labeled Size Cap at Dataset Size
**Issue**: Can't select more than total dataset
**Evidence**: CIFAR-10 caps at 10000, CIFAR-100 caps at 25000
**Fix**: This is actually CORRECT - can't have more labeled than total

---

## MODERATE BUGS

### 🟡 BUG #5: Random Sampling Time Not Recorded
**Issue**: Shows 0.00s (too fast to measure OR not being timed)
**Fix**: Random sampling doesn't use model, so timing might start/stop immediately

---

### 🟡 BUG #6: Multiple Accuracy Drops
**Issue**: Several strategies show accuracy decreases
**Possible Causes**:
1. Model reinitialization each round (correct behavior)
2. Bad samples selected
3. Training not converging
4. Learning rate schedule issues

**Fix**: Need to investigate if drops are expected or bugs

---

## ANALYSIS REQUIRED

### ❓ Question 1: Should Model Persist Across Rounds?
Current: Model is **reinitialized** each round
Alternative: Model **continues training** with new samples

**Impact**: Reinitialization causes some accuracy variance

### ❓ Question 2: Why does Round 10 exist if no sampling?
Current: 10 rounds, but only 9 samplings
Fix options:
A) Skip round 10 entirely (make it 9 rounds)
B) Continue sampling in round 10
C) Don't retrain in round 10 (just use round 9's model)

---

## BUGS TO FIX NOW

1. ✅ **Advanced Leader zero-threshold** - Already fixed with safety checks
2. 🔧 **Round 10 logic** - Fix experiment script to not retrain in last round
3. 🔧 **CIFAR-100 budget** - Investigate why round 1 selects wrong amount
4. ✅ **Labeled size cap** - Not a bug, this is correct

---

## RECOMMENDED FIXES

### Fix #1: Experiment Script Logic
```python
# CURRENT (BUGGY):
for round_num in range(args.rounds):
    # Train model
    if round_num < args.rounds - 1:
        # Select new samples
        
# OPTION A (CLEANER):
for round_num in range(args.rounds):
    # Train model
    # Select new samples (if unlabeled remain)
    if len(unlabeled_indices) >= args.budget:
        # Select samples

# OPTION B (SIMPLEST):
# Just make rounds = actual training rounds
# If you want 10 trained models, set rounds=10 and budget appropriately
```

### Fix #2: Verify Sample Selection Quality
After fixing zero-threshold, verify Advanced Leader selects good samples

### Fix #3: Check CIFAR-100 Initial Setup
Look at round 1 - why does it select 5000 instead of 2500?

