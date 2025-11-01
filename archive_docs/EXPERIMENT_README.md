# Active Learning Core-Set Reproduction

Complete implementation of ICLR 2018 paper "Active Learning for CNNs: A Core-Set Approach"

## 📁 Project Structure

```
active_learning_coreset/
├── active_learning_strategies.py    # 4 sampling strategies
├── cifar10_experiment.py            # CIFAR-10 experiments
├── cifar100_experiment.py           # CIFAR-100 experiments
├── run_cifar10_parallel.sh          # Run CIFAR-10 on 4 GPUs
├── run_cifar100_parallel.sh         # Run CIFAR-100 on 4 GPUs
└── visualize_results.py             # Plot results
```

## 🎯 Implemented Strategies

### 1. Random Sampling (Baseline)
- Simply selects random samples
- Time: O(1)

### 2. Greedy K-Center (Paper's Method)
- **Algorithm:**
  1. Compute FULL N×N distance matrix
  2. Select point farthest from all selected points
- **Time Complexity:** O(N²×D) - VERY SLOW
- **Expected:** 87.84% accuracy, ~5549s per round

### 3. Leader Clustering (Fast Alternative)
- **Algorithm:**
  1. Compute threshold (70th percentile of pairwise distances)
  2. First point becomes first leader
  3. For each point: if distance > threshold from all leaders → new leader
  4. Fill budget from largest clusters
- **Time Complexity:** O(N×L×D) where L ≤ budget (typically O(N))
- **Expected:** 86.90% accuracy, ~14s per round
- **90× FASTER than Greedy!**

### 4. Advanced Leader (Multi-scale, Density-aware)
- **Algorithm:**
  1. Extract features + compute uncertainty (entropy)
  2. Compute local density using k-NN (k=10)
  3. Multi-scale clustering at 3 thresholds (fine/medium/coarse)
  4. Select leaders with high uncertainty + moderate density
- **Time Complexity:** O(N²×k) for k-NN + O(N×L×D)
- **Expected:** 89.89% on CIFAR-10, ~21s per round
- **FAILS on CIFAR-100** - this is what we want to investigate!

## 📊 Dataset Parameters

### CIFAR-10
- Initial labeled: 1,000
- Budget per round: 1,000
- Rounds: 10
- Total: 11,000 / 50,000

### CIFAR-100
- Initial labeled: 5,000
- Budget per round: 2,500
- Rounds: 8
- Total: 25,000 / 50,000

## 🚀 Running Experiments

### Run CIFAR-10 (all 4 strategies in parallel on 4 GPUs):
```bash
cd /home/mohanganesh/active_learning_coreset
./run_cifar10_parallel.sh
```

Monitor progress:
```bash
tail -f logs_cifar10/*.log
```

### Run CIFAR-100:
```bash
./run_cifar100_parallel.sh
```

Monitor:
```bash
tail -f logs_cifar100/*.log
```

### Run individual strategy:
```bash
# CIFAR-10
python cifar10_experiment.py --strategy random --gpu 0
python cifar10_experiment.py --strategy greedy --gpu 1
python cifar10_experiment.py --strategy leader --gpu 2
python cifar10_experiment.py --strategy advanced --gpu 3

# CIFAR-100
python cifar100_experiment.py --strategy random --gpu 0
python cifar100_experiment.py --strategy greedy --gpu 1
python cifar100_experiment.py --strategy leader --gpu 2
python cifar100_experiment.py --strategy advanced --gpu 3
```

## 📈 Visualizing Results

After experiments complete:
```bash
python visualize_results.py
```

Generates:
- `cifar10_accuracy_curves.png` - Test accuracy vs labeled samples
- `cifar10_sampling_times.png` - Average sampling time per round
- `cifar100_accuracy_curves.png`
- `cifar100_sampling_times.png`
- Summary table in terminal

## 🔬 Key Research Question

**Why does Advanced Leader fail on CIFAR-100?**

Hypotheses:
1. **Too many classes**: 100 classes vs 10 creates much more complex feature space
2. **Density estimation breaks**: k-NN with k=10 insufficient for 100 classes
3. **Multi-scale thresholds ineffective**: Scales tuned for 10-class problem
4. **Uncertainty misleading**: Entropy less informative with 100 classes

## 📋 Expected Results

| Method | CIFAR-10 Acc | CIFAR-10 Time | CIFAR-100 Acc | CIFAR-100 Time |
|--------|--------------|---------------|---------------|----------------|
| Random | ~85% | <1s | ~45% | <1s |
| Greedy K-Center | 87.84% | 5549s | ~48% | ~5000s |
| Basic Leader | 86.90% | 14s | ~49% | ~50s |
| **Advanced Leader** | **89.89%** | 21s | **FAILS (~41%)** | ~4300s |

## 🎓 Simple Explanations

### Why is Greedy K-Center slow?
- Must compute distance between EVERY pair of points
- N=50,000 → 2.5 billion distances!
- That's why it takes ~5549 seconds per round

### Why is Leader Clustering fast?
- Only computes distances to "leaders" (representatives)
- Leaders << total points
- 100 leaders × 50,000 points = 5 million distances
- 500× fewer computations!

### Why does Advanced Leader work on CIFAR-10?
- 10 classes are well-separated in feature space
- Density helps find decision boundaries
- Uncertainty identifies hard examples

### Why does it FAIL on CIFAR-100?
- 100 classes create overlapping clusters
- k=10 neighbors can't capture local structure
- Density becomes meaningless in high-dimensional space
- Gets stuck selecting from noisy regions

## 📦 Requirements

```bash
pip install torch torchvision numpy matplotlib scikit-learn tqdm
```

## ✅ Verification

Check GPU availability:
```bash
nvidia-smi
```

Test single round:
```bash
python cifar10_experiment.py --strategy leader --gpu 0 --rounds 1
```

## 📝 Notes

- Paper uses 200 epochs/round, we use 50 for speed
- Results will be slightly lower than paper but trends will match
- Greedy K-Center is intentionally slow (paper's exact O(N²) implementation)
- All timing measurements include sampling only (not training)

---

**Goal:** Demonstrate that Advanced Leader's complexity (density + uncertainty + multi-scale) helps on simple datasets (CIFAR-10) but hurts on complex ones (CIFAR-100), providing evidence for when simpler methods (Basic Leader) are better.
