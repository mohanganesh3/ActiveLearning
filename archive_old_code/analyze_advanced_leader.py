#!/usr/bin/env python3
"""
Deep Analysis: Why Advanced Leader Fails on CIFAR-100 but Works on CIFAR-10

This script investigates the performance difference between datasets.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
def load_results(dataset):
    results_dir = Path(f'{dataset}_results')
    results = {}
    for strategy in ['Random', 'Greedy_K-Center', 'Leader_Clustering', 'Advanced_Leader']:
        pkl_file = results_dir / f'{strategy}_results.pkl'
        if pkl_file.exists():
            with open(pkl_file, 'rb') as f:
                results[strategy] = pickle.load(f)
    return results

print("="*80)
print("INVESTIGATION: Advanced Leader Performance Gap")
print("="*80)
print()

cifar10_results = load_results('cifar10')
cifar100_results = load_results('cifar100')

print("CIFAR-10 Results:")
print("-"*80)
for strategy, data in cifar10_results.items():
    final_acc = data['test_accuracies'][-1]
    avg_sampling = np.mean(data['sampling_times'])
    print(f"{strategy:20s}: Final Acc={final_acc:6.2f}%, Avg Sampling Time={avg_sampling:7.2f}s")

print()
print("CIFAR-100 Results:")
print("-"*80)
for strategy, data in cifar100_results.items():
    final_acc = data['test_accuracies'][-1]
    avg_sampling = np.mean(data['sampling_times'])
    print(f"{strategy:20s}: Final Acc={final_acc:6.2f}%, Avg Sampling Time={avg_sampling:7.2f}s")

print()
print("="*80)
print("PERFORMANCE GAP ANALYSIS")
print("="*80)
print()

# Compare Advanced Leader vs Random on both datasets
adv_c10 = cifar10_results['Advanced_Leader']['test_accuracies'][-1]
rand_c10 = cifar10_results['Random']['test_accuracies'][-1]
adv_c100 = cifar100_results['Advanced_Leader']['test_accuracies'][-1]
rand_c100 = cifar100_results['Random']['test_accuracies'][-1]

print("Advanced Leader vs Random Baseline:")
print(f"  CIFAR-10:  {adv_c10:.2f}% vs {rand_c10:.2f}% (Δ={adv_c10-rand_c10:+.2f}%)")
print(f"  CIFAR-100: {adv_c100:.2f}% vs {rand_c100:.2f}% (Δ={adv_c100-rand_c100:+.2f}%)")
print()

# Compare all strategies relative to Random
print("Performance Improvement over Random:")
print("-"*80)
print(f"{'Strategy':<20s} {'CIFAR-10':>15s} {'CIFAR-100':>15s}")
print("-"*80)
for strategy in ['Greedy_K-Center', 'Leader_Clustering', 'Advanced_Leader']:
    c10_imp = cifar10_results[strategy]['test_accuracies'][-1] - rand_c10
    c100_imp = cifar100_results[strategy]['test_accuracies'][-1] - rand_c100
    print(f"{strategy:<20s} {c10_imp:+7.2f}% {c100_imp:+7.2f}%")

print()
print("="*80)
print("HYPOTHESIS: Why Advanced Leader Fails on CIFAR-100")
print("="*80)
print()

# Analyze accuracy progression
print("Learning Curves (Accuracy by Round):")
print("-"*80)
print(f"{'Round':<8s} {'CIFAR-10 Adv':>15s} {'CIFAR-10 Random':>15s} {'CIFAR-100 Adv':>15s} {'CIFAR-100 Random':>15s}")
print("-"*80)

adv_c10_accs = cifar10_results['Advanced_Leader']['test_accuracies']
rand_c10_accs = cifar10_results['Random']['test_accuracies']
adv_c100_accs = cifar100_results['Advanced_Leader']['test_accuracies']
rand_c100_accs = cifar100_results['Random']['test_accuracies']

for i in range(min(len(adv_c10_accs), len(adv_c100_accs))):
    print(f"Round {i+1:<2d} {adv_c10_accs[i]:>12.2f}% {rand_c10_accs[i]:>15.2f}% {adv_c100_accs[i]:>15.2f}% {rand_c100_accs[i]:>15.2f}%")

print()
print("="*80)
print("DETAILED INVESTIGATION: Sampling Patterns")
print("="*80)
print()

# Check if Advanced Leader is selecting diverse samples
print("Key Questions to Answer:")
print("-"*80)
print()
print("1. DATASET COMPLEXITY")
print("   - CIFAR-10:  10 classes  → ~5000 images/class (500 initial/class)")
print("   - CIFAR-100: 100 classes → ~500 images/class  (50 initial/class)")
print("   → CIFAR-100 has 10x more classes with 10x less data per class")
print()

print("2. THRESHOLD SENSITIVITY")
print("   - Advanced Leader uses adaptive thresholds based on distances")
print("   - With 100 classes, feature space is likely MORE SPARSE")
print("   - Clusters may be TOO TIGHT → selecting redundant samples")
print()

print("3. FEATURE SPACE STRUCTURE")
print("   - CIFAR-10: 10 well-separated clusters in feature space")
print("   - CIFAR-100: 100 classes, likely overlapping/hierarchical")
print("   - Advanced Leader may fail to capture fine-grained distinctions")
print()

# Create detailed comparison plot
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Advanced Leader Performance Analysis: CIFAR-10 vs CIFAR-100', fontsize=16, fontweight='bold')

# Plot 1: CIFAR-10 accuracy curves
ax = axes[0, 0]
for strategy in ['Random', 'Leader_Clustering', 'Advanced_Leader', 'Greedy_K-Center']:
    if strategy in cifar10_results:
        rounds = cifar10_results[strategy]['rounds']
        accs = cifar10_results[strategy]['test_accuracies']
        ax.plot(rounds, accs, marker='o', label=strategy, linewidth=2)
ax.set_title('CIFAR-10: Test Accuracy', fontsize=12, fontweight='bold')
ax.set_xlabel('Round')
ax.set_ylabel('Test Accuracy (%)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: CIFAR-100 accuracy curves
ax = axes[0, 1]
for strategy in ['Random', 'Leader_Clustering', 'Advanced_Leader', 'Greedy_K-Center']:
    if strategy in cifar100_results:
        rounds = cifar100_results[strategy]['rounds']
        accs = cifar100_results[strategy]['test_accuracies']
        ax.plot(rounds, accs, marker='o', label=strategy, linewidth=2)
ax.set_title('CIFAR-100: Test Accuracy', fontsize=12, fontweight='bold')
ax.set_xlabel('Round')
ax.set_ylabel('Test Accuracy (%)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Improvement over Random (CIFAR-10)
ax = axes[1, 0]
rand_baseline_c10 = np.array(cifar10_results['Random']['test_accuracies'])
for strategy in ['Leader_Clustering', 'Advanced_Leader', 'Greedy_K-Center']:
    if strategy in cifar10_results:
        rounds = cifar10_results[strategy]['rounds']
        improvement = np.array(cifar10_results[strategy]['test_accuracies']) - rand_baseline_c10
        ax.plot(rounds, improvement, marker='o', label=strategy, linewidth=2)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.set_title('CIFAR-10: Improvement over Random', fontsize=12, fontweight='bold')
ax.set_xlabel('Round')
ax.set_ylabel('Accuracy Gain (%)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Improvement over Random (CIFAR-100)
ax = axes[1, 1]
rand_baseline_c100 = np.array(cifar100_results['Random']['test_accuracies'])
for strategy in ['Leader_Clustering', 'Advanced_Leader', 'Greedy_K-Center']:
    if strategy in cifar100_results:
        rounds = cifar100_results[strategy]['rounds']
        improvement = np.array(cifar100_results[strategy]['test_accuracies']) - rand_baseline_c100
        ax.plot(rounds, improvement, marker='o', label=strategy, linewidth=2)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.set_title('CIFAR-100: Improvement over Random', fontsize=12, fontweight='bold')
ax.set_xlabel('Round')
ax.set_ylabel('Accuracy Gain (%)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('advanced_leader_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved detailed analysis plot: advanced_leader_analysis.png")
print()

# Create summary comparison
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

strategies = ['Random', 'Leader\nClustering', 'Advanced\nLeader', 'Greedy\nK-Center']
c10_final = [cifar10_results[s.replace('\n', '_')]['test_accuracies'][-1] 
             for s in ['Random', 'Leader_Clustering', 'Advanced_Leader', 'Greedy_K-Center']]
c100_final = [cifar100_results[s.replace('\n', '_')]['test_accuracies'][-1] 
              for s in ['Random', 'Leader_Clustering', 'Advanced_Leader', 'Greedy_K-Center']]

x = np.arange(len(strategies))
width = 0.35

bars1 = ax.bar(x - width/2, c10_final, width, label='CIFAR-10', alpha=0.8)
bars2 = ax.bar(x + width/2, c100_final, width, label='CIFAR-100', alpha=0.8)

# Highlight Advanced Leader
bars1[2].set_color('green')
bars1[2].set_alpha(0.9)
bars2[2].set_color('red')
bars2[2].set_alpha(0.9)

ax.set_ylabel('Final Test Accuracy (%)', fontsize=12)
ax.set_title('Final Accuracy Comparison: Why Advanced Leader Fails on CIFAR-100', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(strategies)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('advanced_leader_summary.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved summary comparison: advanced_leader_summary.png")
print()

print("="*80)
print("CONCLUSIONS")
print("="*80)
print()
print("Possible Reasons for Advanced Leader Failure on CIFAR-100:")
print()
print("1. CLASS IMBALANCE IN INITIAL SET")
print("   - With 100 classes and 5000 initial samples → only ~50 per class")
print("   - Leader clustering may not have enough representatives per class")
print("   - Threshold adaptation may be biased toward over-represented classes")
print()
print("2. FEATURE SPACE SPARSITY")
print("   - 100 classes in same dimensional space as 10 classes")
print("   - Clusters are likely more spread out and overlapping")
print("   - Advanced Leader's threshold may be TOO CONSERVATIVE")
print()
print("3. HIERARCHICAL CLASS STRUCTURE")
print("   - CIFAR-100 has hierarchical superclasses")
print("   - Advanced Leader may cluster by superclass, missing fine details")
print("   - Need class-aware or multi-scale clustering")
print()
print("4. THRESHOLD CALCULATION BUG")
print("   - Check if percentile-based threshold works for sparse spaces")
print("   - May need different percentile for high-dimensional problems")
print()
print("RECOMMENDATIONS:")
print("  → Adjust threshold percentile for CIFAR-100 (lower percentile)")
print("  → Use stratified sampling to ensure class coverage")
print("  → Consider multi-scale or hierarchical clustering")
print("  → Add diversity penalty to prevent over-clustering")
print()
print("="*80)
