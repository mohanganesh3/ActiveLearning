# Advanced Leader V2 Experiments - Running

**Started:** October 29, 2025, 21:03:13  
**Status:** ✅ RUNNING  
**Expected Duration:** ~8 hours total

---

## Experiments Running

### CIFAR-10
- **GPU:** 0
- **PID:** Check `logs_v2/cifar10_v2.pid`
- **Log:** `logs_v2/advanced_leader_cifar10_20251029_210313.log`
- **Monitor:** `tail -f logs_v2/advanced_leader_cifar10_20251029_210313.log`

### CIFAR-100
- **GPU:** 1
- **PID:** Check `logs_v2/cifar100_v2.pid`
- **Log:** `logs_v2/advanced_leader_cifar100_20251029_210313.log`
- **Monitor:** `tail -f logs_v2/advanced_leader_cifar100_20251029_210313.log`

---

## What's Being Tested (V2 Improvements)

1. **Smooth CV-based threshold adaptation** - No discrete jumps
2. **More conservative percentiles** - [20,40,65]→[30,55,75] instead of [15,35,60]→[25,50,75]
3. **Temporal smoothing** - 30% momentum between rounds
4. **Minimum leader target** - Ensure at least 50% from leaders
5. **Controlled 70/30 balance** - 70% diversity leaders, 30% uncertainty

---

## Expected Improvements

### CIFAR-100 (Primary Focus)
- **V1 Result:** 39.61% final (but volatile: 18.81% dip in Round 8)
- **V2 Goal:** 
  - Reduced volatility (no dips below 25%)
  - Final accuracy ≥ 38%
  - More monotonic increase
  - Standard deviation of changes < 6%

### CIFAR-10 (Verification)
- **V1 Result:** 82.12% (already good, low volatility)
- **V2 Goal:** 
  - Maintain ≥ 81%
  - Verify improvements don't hurt well-separated data

---

## Monitor Progress

### Quick Check
```bash
# Check if still running
ps aux | grep "[c]ifar.*experiment"

# See latest from both
tail -n 20 logs_v2/advanced_leader_cifar10_20251029_210313.log
tail -n 20 logs_v2/advanced_leader_cifar100_20251029_210313.log
```

### Live Monitoring
```bash
# Watch CIFAR-100 (more interesting)
tail -f logs_v2/advanced_leader_cifar100_20251029_210313.log

# Or both in split screen
tmux
# Split: Ctrl+B then "
# Upper: tail -f logs_v2/advanced_leader_cifar10_20251029_210313.log
# Lower: tail -f logs_v2/advanced_leader_cifar100_20251029_210313.log
```

### Extract Results When Done
```bash
# CIFAR-10 accuracies
grep "Test Accuracy:" logs_v2/advanced_leader_cifar10_20251029_210313.log

# CIFAR-100 accuracies
grep "Test Accuracy:" logs_v2/advanced_leader_cifar100_20251029_210313.log

# Leader counts
grep "Candidate leaders:" logs_v2/advanced_leader_cifar100_20251029_210313.log

# Threshold smoothing
grep "Smoothed:" logs_v2/advanced_leader_cifar100_20251029_210313.log
```

---

## Stop/Kill if Needed

```bash
# Kill both
kill $(cat logs_v2/cifar10_v2.pid) $(cat logs_v2/cifar100_v2.pid)

# Or individually
kill $(cat logs_v2/cifar10_v2.pid)
kill $(cat logs_v2/cifar100_v2.pid)
```

---

## After Completion

### 1. Extract Results
Run the comparison script in `V2_EXPERIMENT_PLAN.md`

### 2. Compare V1 vs V2
- Final accuracy
- Volatility (std of round-to-round changes)
- Leader count stability
- Round-by-round comparison

### 3. Update Documentation
- `COMPLETE_DEVELOPMENT_RECORD.md` - Add V2 results
- `EMAIL_VERSION_3_MEDIUM.md` - Add V2 findings
- `IMPROVED_RESULTS_SUMMARY.md` - Update with best version

### 4. Decide Next Steps
- If V2 better: Use V2 as final
- If V2 worse: Stick with V1
- If mixed: Consider V2.1 with tuning

---

## Notes

- Experiments use **unbuffered output** (`python -u`) for real-time logging
- Both run in **nohup** so they survive SSH disconnection
- Configuration: 5000 initial + 2500×8 rounds = 25000 labeled (50% of data)
- VGG with BatchNorm, 50 epochs per round
- Same config for both datasets (universal algorithm)

---

**Record Created:** October 29, 2025, 21:03  
**Next Check:** In 1-2 hours to see Round 2-3 results  
**Final Check:** ~8 hours from start
