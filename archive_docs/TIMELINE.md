# Experiment Timeline and Status

## FINAL PARAMETERS (UPDATED)

### CIFAR-10:
- **Initial labeled**: 1,000
- **Budget per round**: 1,000  
- **Rounds**: 10
- **Final total**: 1,000 + (9 × 1,000) = **10,000 / 50,000 = 20%**

### CIFAR-100:
- **Initial labeled**: 5,000
- **Budget per round**: 2,500
- **Rounds**: 9 ✅ **(UPDATED from 8)**
- **Final total**: 5,000 + (8 × 2,500) = **25,000 / 50,000 = 50%**
- **Math**: Initial 5k + 8 more rounds with 2.5k each = exactly 25k ✅

## Estimated Completion Time: **~11-12 hours total**

### Breakdown:

#### CIFAR-10 (Running Now)
- **Duration**: ~6-6.5 hours
- **4 Algorithms in Parallel** (one per GPU):
  - Random Sampling (GPU 0)
  - Greedy K-Center (GPU 1) - slowest due to O(N²) sampling
  - Leader Clustering (GPU 2)
  - Advanced Leader (GPU 3)
- **10 rounds** × 50 epochs each
- Training time increases each round as labeled set grows (1K → 10K samples)

#### CIFAR-100 (Auto-starts after CIFAR-10)
- **Duration**: ~5-5.5 hours
- **4 Algorithms in Parallel** (one per GPU)
- **9 rounds** × 50 epochs each **(UPDATED)**
- Larger initial set (5K) and bigger increments (2.5K per round → 25K total)

### Current Status (as of run start):
- ✅ All 4 CIFAR-10 experiments running
- ✅ GPU utilization: 64-98% across all 4 GPUs
- ✅ Each process using ~2GB RAM
- ⏳ Runtime so far: ~2-3 minutes
- ⏳ Estimated remaining for CIFAR-10: ~6 hours
- ⏳ Then CIFAR-100: ~4.5 hours

### Timeline:
```
Now (19:09)        → CIFAR-10 starts (4 algos parallel, 10 rounds)
~01:30 (next day)  → CIFAR-10 completes, plots generated
~01:30             → CIFAR-100 starts (4 algos parallel, 9 rounds)
~06:30 (next day)  → CIFAR-100 completes, plots generated
~06:30             → ALL EXPERIMENTS COMPLETE ✓
```

### What Happens Automatically:
1. ✅ CIFAR-10 experiments run (currently in progress)
2. 🔄 Wait for all 4 to finish
3. 📊 Generate CIFAR-10 accuracy and timing plots
4. 🔄 Launch CIFAR-100 experiments (4 parallel)
5. 🔄 Wait for all 4 to finish
6. 📊 Generate CIFAR-100 accuracy and timing plots
7. ✅ Complete!

### Monitoring:
```bash
# Quick status
bash check_progress.sh

# Master log (shows workflow progress)
tail -f master_experiment.log

# Individual experiment logs (once data flushes from buffer)
tail -f cifar10_results/random.log
tail -f cifar10_results/advanced.log

# Check results as they complete
ls -lh cifar10_results/*_results.pkl
ls -lh cifar100_results/*_results.pkl
```

### Output Files (when complete):
```
cifar10_results/
  ├── Random_results.pkl
  ├── Greedy_K-Center_results.pkl
  ├── Leader_Clustering_results.pkl
  ├── Advanced_Leader_results.pkl
  ├── accuracy_comparison.png
  └── sampling_time_comparison.png

cifar100_results/
  ├── Random_results.pkl
  ├── Greedy_K-Center_results.pkl
  ├── Leader_Clustering_results.pkl
  ├── Advanced_Leader_results.pkl
  ├── accuracy_comparison.png
  └── sampling_time_comparison.png
```

### Notes:
- **SSH-safe**: Running with nohup, will continue if you disconnect
- **Parallel**: All 4 algorithms run simultaneously per dataset
- **Sequential**: CIFAR-10 completes before CIFAR-100 starts
- **Automatic**: No manual intervention needed
- **cuDNN disabled**: For GPU compatibility (slightly slower but stable)
- **No multiprocessing**: DataLoader workers=0 to avoid nohup issues

### Research Goal:
After completion, you'll have evidence for:
- ✅ Advanced Leader performs best on CIFAR-10 (~89-90% accuracy)
- ❌ Advanced Leader fails dramatically on CIFAR-100 
- 📊 Timing comparisons showing Greedy is O(N²) vs Leader O(N)
- 📈 Accuracy curves showing the performance degradation

**Expected completion: Tomorrow morning ~5:30 AM UTC (if started at 19:00)**
