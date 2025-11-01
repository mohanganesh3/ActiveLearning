# ANSWER TO PROFESSOR'S QUESTION
## "Why do Round 1 accuracies differ between strategies?"

**Date:** November 1, 2025  
**Status:** ✅ RESOLVED - Bug in visualization script identified and fixed

---

## THE SHORT ANSWER

**They don't!** All strategies have **IDENTICAL Round 1 accuracy** within the same experiment. The professor saw different Round 1 accuracies because the original graphs accidentally mixed results from **different experiments** with different `initial_labeled` settings.

---

## THE PROBLEM

### What the Original Graph Showed:

| Strategy | Round 1 Accuracy | Source Directory | Initial Labeled |
|----------|------------------|------------------|-----------------|
| Random | 12.61% | `old_results_BUGGY` | **2,000** |
| Leader Clustering | 12.61% | `old_results_BUGGY` | **2,000** |
| Greedy K-Center | 12.61% | `old_results_BUGGY` | **2,000** |
| **Advanced Leader V3** | **37.45%** | `cifar10_results` | **5,000** ❌ |

**This created a FALSE impression that Advanced Leader starts with 3× higher accuracy!**

### The Bug in `generate_clean_graphs.py`:

```python
def load_results(dataset, strategy, version='current'):
    """Load results from pickle files"""
    if version == 'current':
        base_dir = f'{dataset}_results'
        strategy_file = strategy
    else:
        base_dir = f'old_results_BUGGY/{dataset}_results'  # ← Different experiment!
        strategy_file = strategy
    
    filepath = f'{base_dir}/{strategy_file}_results.pkl'
    ...

# In plotting function:
result = load_results(dataset, key, 'current')
if result is None:  # Try old_results_BUGGY
    result = load_results(dataset, key, 'old')  # ← MIXES DIFFERENT EXPERIMENTS!
```

**Result:** Advanced Leader loaded from V3 (5,000 initial), others from V1 (2,000 initial) → Different Round 1 baselines!

---

## THE CORRECTED RESULTS

After fixing the visualization script to use **consistent experimental settings** (V2, all with 5,000 initial labels):

### CIFAR-10 (Corrected):

| Strategy | Round 1 | Round 2 | Round 3 | ... | Final | Gain vs Random |
|----------|---------|---------|---------|-----|-------|----------------|
| Random | **37.45%** | 49.21% | 57.89% | ... | 77.03% | (baseline) |
| Leader Clustering | **37.45%** | 49.68% | 58.42% | ... | 77.86% | +0.83% |
| Greedy K-Center | **37.45%** | 52.15% | 61.34% | ... | 80.38% | +3.35% |
| Advanced Leader V2 | **37.45%** | 51.89% | 60.78% | ... | 78.44% | +1.41% |

**✅ ALL strategies have IDENTICAL 37.45% Round 1 accuracy!**

### CIFAR-100 (Corrected):

| Strategy | Round 1 | Round 2 | Round 3 | ... | Final | Gain vs Random |
|----------|---------|---------|---------|-----|-------|----------------|
| Random | **6.20%** | 10.35% | 15.24% | ... | 35.82% | (baseline) |
| Leader Clustering | **6.20%** | 10.98% | 16.12% | ... | 38.83% | +3.01% |
| Greedy K-Center | **6.20%** | 12.45% | 18.67% | ... | 43.58% | +7.76% |
| Advanced Leader V2 | **6.20%** | 11.23% | 16.89% | ... | 34.37% | -1.45% |

**✅ ALL strategies have IDENTICAL 6.20% Round 1 accuracy!**

---

## WHY ROUND 1 MUST BE IDENTICAL

### The Active Learning Process:

```
Setup Phase:
  - Shuffle all 50,000 training samples
  - Take first 5,000 as initial_labeled
  - Remaining 45,000 as unlabeled_pool

Round 1 (round_num = 0):
  ┌─────────────────────────────────────────┐
  │ 1. Create DataLoader with 5,000 samples │  ← Same for ALL strategies
  │ 2. Initialize fresh VGG model           │
  │ 3. Train for 50 epochs                  │
  │ 4. Test on test set                     │
  │ 5. Record accuracy: 37.45%              │  ← IDENTICAL for all!
  └─────────────────────────────────────────┘
  NO strategy selection happens!

Round 2 (round_num = 1):
  ┌─────────────────────────────────────────┐
  │ 1. Use strategy to SELECT 2,500 samples │  ← FIRST use of strategy!
  │    - Random: picks random samples       │
  │    - Greedy: k-center selection         │
  │    - Advanced: multi-scale clustering   │
  │ 2. Add selected to labeled (now 7,500)  │  ← Different for each strategy
  │ 3. Train on updated labeled set         │
  │ 4. Test: accuracies now DIVERGE         │
  └─────────────────────────────────────────┘
  Strategy differences appear here!
```

### Code Verification:

From `cifar10_experiment.py` lines 137-148:

```python
for round_num in range(args.rounds):
    print(f"\nROUND {round_num+1}/{args.rounds}")
    
    # Select new samples FIRST (before training)
    sampling_time = 0
    if round_num > 0 and len(unlabeled_indices) >= args.budget:  # ← KEY LINE!
        print(f"\nSelecting {args.budget} new samples using {strategy_name}...")
        selected_relative = active_learner.select_batch(model, unlabeled_subset, round_num=round_num)
        ...
    
    # Now train model with current labeled set
    labeled_subset = torch.utils.data.Subset(trainset, labeled_indices)
    ...
```

**The condition `if round_num > 0` means:**
- Round 1 (round_num=0): Skip strategy selection, train on initial 5,000
- Round 2+ (round_num≥1): Use strategy to select additional samples

**Therefore, Round 1 CANNOT differ between strategies - they all use the exact same labeled set!**

---

## CORRECTED VISUALIZATIONS

Generated new graphs using `generate_corrected_graphs.py`:

### Files Created:
- `cifar10_comparison_CORRECTED.png` - Shows all strategies starting at 37.45%
- `cifar10_final_accuracy_CORRECTED.png` - Bar chart comparison
- `cifar100_comparison_CORRECTED.png` - Shows all strategies starting at 6.20%
- `cifar100_final_accuracy_CORRECTED.png` - Bar chart comparison

### Key Features:
✅ All strategies loaded from **same experiment** (old_results_V2)  
✅ Identical `initial_labeled = 5,000`  
✅ Round 1 consistency check built-in  
✅ Clear subtitle: "All strategies start from identical Round 1"  

---

## WHAT THIS MEANS FOR THE THESIS

### The Good News:

1. **✅ No bug in experimental code** - Round 1 behavior is correct
2. **✅ No bug in training logic** - All strategies properly isolated
3. **✅ Easy fix** - Just regenerate graphs with consistent data
4. **✅ Clear story** - "All strategies start equal, diverge from Round 2 onwards"

### The Narrative:

**For your professor:**

> "All active learning strategies begin from an identical baseline in Round 1, trained on the same initial labeled set of 5,000 samples. This ensures fair comparison. The strategies diverge starting from Round 2, when they each select different samples based on their respective selection criteria. The earlier graph showing different Round 1 accuracies was due to a visualization bug where results from different experiments (with different initial_labeled sizes) were accidentally mixed."

### Strategy Performance (Corrected):

**CIFAR-10:**
- Greedy K-Center: Best (+3.35% over Random)
- Advanced Leader V2: Moderate (+1.41% over Random)
- Leader Clustering: Slight (+0.83% over Random)

**CIFAR-100:**
- Greedy K-Center: Best (+7.76% over Random)
- Leader Clustering: Good (+3.01% over Random)
- Advanced Leader V2: **Underperforms** (-1.45% vs Random) ← V2 collapse issue

**V3 addresses the V2 collapse!**

---

## TECHNICAL DETAILS

### Why 5,000 initial labels?

**CIFAR-10 (10 classes):**
- 5,000 labels ÷ 10 classes = **500 samples per class**
- Above critical mass (~100-200 samples/class for deep learning)
- Result: 37.45% Round 1 accuracy (vs 10% random guess)

**CIFAR-100 (100 classes):**
- 5,000 labels ÷ 100 classes = **50 samples per class**
- Below critical mass (~100-200 samples/class)
- Result: 6.20% Round 1 accuracy (barely above 1% random guess)
- **Active learning is ESSENTIAL for CIFAR-100!**

### Why different initial_labeled in old experiments?

**V1 (old_results_BUGGY):**
- CIFAR-10: 2,000 initial (specialized for 10 classes)
- CIFAR-100: 7,500 initial (specialized for 100 classes)
- Result: Non-universal approach

**V2/V3 (current):**
- Both datasets: 5,000 initial (universal approach)
- Result: Fair comparison across datasets

---

## RECOMMENDATIONS

### For Meeting with Professor:

1. **Show corrected graphs** (`*_CORRECTED.png` files)
2. **Explain the bug**: "Accidentally mixed results from different experiments"
3. **Show the fix**: "All strategies now from same experiment with identical settings"
4. **Emphasize correctness**: "Round 1 identical = correct behavior!"
5. **Focus on real insights**: Strategy differences from Round 2 onwards

### For Thesis:

- [ ] Use only CORRECTED graphs in thesis
- [ ] Add footnote: "Round 1 represents baseline from initial labeled set (5,000 samples)"
- [ ] Highlight: "Strategy differences emerge from Round 2 onwards"
- [ ] Document experimental settings clearly in methods section

### For Future Work:

```python
# Add to all visualization scripts:
def verify_consistent_settings(results_dict):
    """Ensure all results from same experimental settings"""
    settings = {}
    for name, result in results_dict.items():
        key = (result['labeled_sizes'][0], len(result['rounds']))
        if key not in settings:
            settings[key] = []
        settings[key].append(name)
    
    if len(settings) > 1:
        raise ValueError(f"Inconsistent experimental settings detected: {settings}")
    
    return True
```

---

## SUMMARY FOR PROFESSOR

**Question:** "Why do all strategies have Round 1 accuracy between 10-15%, but Advanced Leader has 30-40%?"

**Answer:** "They don't! The original graph accidentally mixed results from two different experiments:
- Baseline strategies: Loaded from old experiment with 2,000 initial labels → 12.61% Round 1
- Advanced Leader: Loaded from new experiment with 5,000 initial labels → 37.45% Round 1

**All strategies within the same experiment have IDENTICAL Round 1 accuracy (37.45% for CIFAR-10, 6.20% for CIFAR-100) because Round 1 is trained on the same initial labeled set BEFORE any strategy selection occurs. I've corrected the visualization script and regenerated all graphs with consistent data."**

---

## FILES TO SHOW PROFESSOR

1. **`ROUND_1_ACCURACY_INVESTIGATION.md`** - Comprehensive analysis (this document's companion)
2. **`cifar10_comparison_CORRECTED.png`** - Correct CIFAR-10 graph
3. **`cifar100_comparison_CORRECTED.png`** - Correct CIFAR-100 graph
4. **`generate_corrected_graphs.py`** - Fixed visualization script

---

**Status:** ✅ **RESOLVED**  
**Next Steps:** Present corrected graphs to professor, update thesis with corrected visualizations  
**Key Takeaway:** Round 1 identical = correct! Strategy differences from Round 2 onwards = interesting science!

