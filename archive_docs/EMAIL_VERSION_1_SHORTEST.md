Subject: Advanced Leader Investigation - Quick Summary

Dear Professor,

**Summary:** I investigated why Advanced Leader works on CIFAR-10 but fails on CIFAR-100.

**Bug Found & Fixed:**
- **Problem:** Thresholds were collapsing to zero, causing excessive sampling times
- **Fix:** Replaced median-based calculation with robust percentile-based approach
- **Result:** Sampling times now normal (82s vs 1000s+ before)

**New Results (After Fix):**

CIFAR-10: Advanced Leader = 82.12% (+5.09% over Random) ✅ **BEST**
CIFAR-100: Advanced Leader = 31.21% (-4.61% vs Random) ❌ **WORSE than random**

**Why It Still Fails on CIFAR-100:**
1. **Too many leaders** (105 vs 35 on CIFAR-10) → less diversity, more redundancy
2. **High thresholds** (75% higher) → creates tight clusters that miss rare classes
3. **Poor class coverage** (100 classes, only 50 samples/class initially) → many classes ignored
4. **Round 9 collapse** (-9.54% accuracy drop) → catastrophic sample selection

**Root Cause:**
Advanced Leader assumes well-separated clusters. This works for 10 classes (CIFAR-10) but fails for 100 overlapping classes (CIFAR-100).

**Recommendation:**
- CIFAR-10: Keep using Advanced Leader ✅
- CIFAR-100: Switch to Basic Leader (+3.01% improvement) ✅

**Supporting Files:**
- EMAIL_REPORT_TO_PROFESSOR.md (full detailed report)
- ADVANCED_LEADER_INVESTIGATION_REPORT.md (technical analysis)
- advanced_leader_final_summary.png (visualization)

Best regards,
[Your Name]
