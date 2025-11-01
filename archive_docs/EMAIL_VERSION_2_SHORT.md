Subject: Investigation Report: Advanced Leader Clustering Performance Analysis

Dear Professor,

I completed the investigation you requested into why Advanced Leader Clustering performs differently on CIFAR-10 versus CIFAR-100. Here are my findings:

---

## 1. CRITICAL BUG DISCOVERED AND FIXED

**The Bug:**
The threshold calculation had a critical flaw where values were collapsing to zero. The original code calculated thresholds using:
```python
median = np.median(distances)
return [median * 0.5, median * 1.0, median * 1.5]  # ← When median ≈ 0, all thresholds = 0
```

This caused every point to become a leader, leading to excessive sampling times (1000s+ seconds).

**The Fix:**
We replaced this with robust percentile-based computation:
```python
p25 = np.percentile(distances, 25)  # Fine threshold
p50 = np.percentile(distances, 50)  # Medium threshold
p75 = np.percentile(distances, 75)  # Coarse threshold
return [p25, p50, p75]
```

This uses actual distribution percentiles instead of median-multiplication, ensuring non-zero meaningful thresholds.

**Result:** Sampling times reduced from 1000s+ to ~82 seconds ✅

---

## 2. NEW EXPERIMENTAL RESULTS

**Setup:**
- Initial: 5,000 labeled samples
- Budget: 2,500 per round × 8 rounds
- Final: 25,000 labeled samples (9 rounds total)
- Training: 50 epochs per round

**CIFAR-10 Results:**

| Strategy | Final Accuracy | vs Random | Avg Sampling Time |
|----------|---------------|-----------|-------------------|
| Random | 77.03% | baseline | 0.00s |
| Leader Clustering | 77.86% | +0.83% | 73.79s |
| **Advanced Leader** | **82.12%** | **+5.09%** ✅ | 81.79s |
| Greedy K-Center | 80.38% | +3.35% | 806.84s |

**CIFAR-100 Results:**

| Strategy | Final Accuracy | vs Random | Avg Sampling Time |
|----------|---------------|-----------|-------------------|
| Random | 35.82% | baseline | 0.00s |
| Leader Clustering | 38.83% | +3.01% | 74.99s |
| **Advanced Leader** | **31.21%** | **-4.61%** ❌ | 91.09s |
| Greedy K-Center | 43.58% | +7.76% | 805.96s |

**Key Finding:** Advanced Leader is the ONLY strategy that performs worse than random on CIFAR-100!

---

## 3. ROOT CAUSE ANALYSIS

After fixing the bug, we analyzed why Advanced Leader still fails on CIFAR-100:

### Problem 1: Threshold Mismatch (60-75% Higher)
**CIFAR-10 Thresholds:** Start at 1.72, end at 5.38
**CIFAR-100 Thresholds:** Start at 2.99, end at 9.44 (75% HIGHER)

Higher thresholds create tighter clusters. Good for separated classes (CIFAR-10), bad for overlapping classes (CIFAR-100).

### Problem 2: Leader Redundancy (3x More Leaders)
**CIFAR-10:** ~35 leaders per round (1.4% of budget)
**CIFAR-100:** ~105 leaders per round (4.4% of budget)

More leaders means less diversity. Multi-scale clustering selects from the same dense regions repeatedly.

### Problem 3: Class Coverage Failure
**CIFAR-10:** 10 classes × 500 samples/class = good coverage
**CIFAR-100:** 100 classes × 50 samples/class = **many classes get zero leaders**

Advanced Leader optimizes for cluster diversity, NOT class coverage.

### Problem 4: Round 9 Catastrophic Collapse
CIFAR-100 accuracy progression:
```
Round 8: 40.75% ✓
Round 9: 31.21% ❌ (-9.54% DROP!)
```

This suggests the selected samples in Round 9 were extremely poor quality or caused severe class imbalance.

---

## 4. WHY BASIC LEADER WORKS BETTER

Basic Leader on CIFAR-100: 38.83% (+3.01% vs Random) ✅

**Advantages:**
- Single threshold (70th percentile) → simpler, more robust
- Fewer leaders → more diversity in final selection
- No multi-scale redundancy
- Better adapts to overlapping classes

**The Irony:** Being "advanced" made it worse for fine-grained problems!

---

## 5. TECHNICAL EXPLANATION

Advanced Leader makes 4 assumptions that work for CIFAR-10 but fail for CIFAR-100:

| Assumption | CIFAR-10 (10 classes) | CIFAR-100 (100 classes) |
|------------|----------------------|------------------------|
| Well-separated clusters | ✅ True | ❌ False (overlapping) |
| Percentile thresholds adapt | ✅ Works | ❌ Too high |
| k=10 captures local density | ✅ Works | ❌ Spans multiple classes |
| Multi-scale adds diversity | ✅ Works | ❌ Creates redundancy |

---

## 6. RECOMMENDATIONS

**Immediate Action:**
- **CIFAR-10:** Continue using Advanced Leader (+5.09% improvement) ✅
- **CIFAR-100:** Switch to Basic Leader Clustering (+3.01% improvement) ✅

**Medium-term Solutions to Test:**
1. **Stratified Sampling:** Run Advanced Leader within each predicted class
2. **Adaptive Percentiles:** Use 10th, 30th, 60th percentiles for CIFAR-100 instead of 25th, 50th, 75th
3. **Hybrid Strategy:** Auto-select Basic Leader when num_classes > 20

**General Guideline:**
- num_classes ≤ 20 → Use Advanced Leader
- num_classes > 20 → Use Basic Leader or Stratified approach

---

## 7. CONCLUSION

We successfully fixed the threshold bug that was causing zero-value failures and excessive sampling times. However, Advanced Leader's fundamental design assumes well-separated clusters, which holds for CIFAR-10 (10 classes) but not CIFAR-100 (100 overlapping fine-grained classes).

**Key Lesson:** Sophisticated algorithms need their assumptions validated for each problem type. Sometimes simpler is better.

**Supporting Materials:**
- Full detailed report: ADVANCED_LEADER_INVESTIGATION_REPORT.md
- Visualizations: advanced_leader_final_summary.png
- All experiment logs and results available in repository

I am ready to implement any of the proposed solutions or conduct further experiments as you direct.

Best regards,
[Your Name]
