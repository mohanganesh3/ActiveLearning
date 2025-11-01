#!/usr/bin/env python3
"""
Final Summary Visualization: Advanced Leader Investigation
"""

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('Advanced Leader Investigation: Why it Fails on CIFAR-100', 
             fontsize=18, fontweight='bold')

# Data from logs
cifar10_thresholds_fine = [1.717, 3.795, 3.881, 4.698, 4.660, 4.907, 4.980, 5.377]
cifar10_thresholds_coarse = [3.336, 6.384, 6.398, 7.649, 7.659, 7.949, 8.023, 8.150]
cifar10_leaders = [34, 43, 35, 35, 37, 30, 25, 33]

cifar100_thresholds_fine = [2.988, 6.090, 5.728, 7.574, 8.213, 8.843, 8.667, 9.436]
cifar100_thresholds_coarse = [6.053, 8.676, 8.626, 10.657, 11.319, 12.186, 11.513, 12.475]
cifar100_leaders = [40, 86, 127, 120, 103, 109, 105, 109]

rounds = list(range(2, 10))

# Plot 1: Threshold Evolution (Top Left - spans 2 columns)
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(rounds, cifar10_thresholds_fine, 'o-', label='CIFAR-10 Fine (25th)', linewidth=2, markersize=8)
ax1.plot(rounds, cifar10_thresholds_coarse, 's-', label='CIFAR-10 Coarse (75th)', linewidth=2, markersize=8)
ax1.plot(rounds, cifar100_thresholds_fine, 'o--', label='CIFAR-100 Fine (25th)', linewidth=2, markersize=8)
ax1.plot(rounds, cifar100_thresholds_coarse, 's--', label='CIFAR-100 Coarse (75th)', linewidth=2, markersize=8)
ax1.set_xlabel('Round', fontsize=12)
ax1.set_ylabel('Threshold Value', fontsize=12)
ax1.set_title('Threshold Evolution: CIFAR-100 is 60-75% Higher', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=6, color='red', linestyle=':', alpha=0.5, label='Typical inter-class distance')

# Plot 2: Candidate Leaders Count (Top Right)
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(rounds, cifar10_leaders, 'o-', label='CIFAR-10', linewidth=2, markersize=8, color='green')
ax2.plot(rounds, cifar100_leaders, 's-', label='CIFAR-100', linewidth=2, markersize=8, color='red')
ax2.set_xlabel('Round', fontsize=12)
ax2.set_ylabel('Number of Leaders', fontsize=12)
ax2.set_title('Candidate Leaders:\nCIFAR-100 has 3x more', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Accuracy Learning Curves (Middle Left)
ax3 = fig.add_subplot(gs[1, 0])
c10_adv = [37.45, 64.20, 72.26, 76.63, 70.32, 77.65, 76.58, 73.16, 82.12]
c10_rand = [37.45, 67.92, 68.41, 73.49, 74.44, 74.56, 73.86, 73.99, 77.03]
rounds_all = list(range(1, 10))
ax3.plot(rounds_all, c10_adv, 'o-', label='Advanced Leader', linewidth=2, markersize=7, color='green')
ax3.plot(rounds_all, c10_rand, 's--', label='Random', linewidth=2, markersize=7, color='gray')
ax3.set_xlabel('Round', fontsize=11)
ax3.set_ylabel('Test Accuracy (%)', fontsize=11)
ax3.set_title('CIFAR-10: Advanced Leader WINS\n(+5.09% improvement)', 
              fontsize=12, fontweight='bold', color='green')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: CIFAR-100 Learning Curves with COLLAPSE (Middle Center)
ax4 = fig.add_subplot(gs[1, 1])
c100_adv = [6.20, 15.86, 17.56, 29.34, 32.60, 36.44, 39.64, 40.75, 31.21]
c100_rand = [6.20, 14.89, 13.87, 29.33, 36.03, 34.75, 40.86, 41.89, 35.82]
ax4.plot(rounds_all, c100_adv, 'o-', label='Advanced Leader', linewidth=2, markersize=7, color='red')
ax4.plot(rounds_all, c100_rand, 's--', label='Random', linewidth=2, markersize=7, color='gray')
# Highlight the collapse
ax4.annotate('COLLAPSE!\n-9.54%', xy=(9, 31.21), xytext=(8, 25),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, fontweight='bold', color='red')
ax4.set_xlabel('Round', fontsize=11)
ax4.set_ylabel('Test Accuracy (%)', fontsize=11)
ax4.set_title('CIFAR-100: Advanced Leader FAILS\n(-4.61% vs Random)', 
              fontsize=12, fontweight='bold', color='red')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# Plot 5: Final Accuracy Comparison (Middle Right)
ax5 = fig.add_subplot(gs[1, 2])
strategies = ['Random', 'Leader', 'Advanced', 'Greedy']
c10_final = [77.03, 77.86, 82.12, 80.38]
c100_final = [35.82, 38.83, 31.21, 43.58]
x = np.arange(len(strategies))
width = 0.35
bars1 = ax5.bar(x - width/2, c10_final, width, label='CIFAR-10', alpha=0.8, color='green')
bars2 = ax5.bar(x + width/2, c100_final, width, label='CIFAR-100', alpha=0.8, color='red')
# Highlight Advanced Leader
bars1[2].set_edgecolor('black')
bars1[2].set_linewidth(3)
bars2[2].set_edgecolor('black')
bars2[2].set_linewidth(3)
ax5.set_ylabel('Final Accuracy (%)', fontsize=11)
ax5.set_title('Final Performance:\nAdvanced = Best/Worst', fontsize=12, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(strategies, fontsize=10)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Problem Diagnosis (Bottom - spans all columns)
ax6 = fig.add_subplot(gs[2, :])
ax6.axis('off')

diagnosis_text = """
ROOT CAUSE ANALYSIS: Why Advanced Leader Fails on CIFAR-100

1. THRESHOLD MISMATCH
   • CIFAR-10: Thresholds 1.7 → 5.4 (well-calibrated for 10 separated classes)
   • CIFAR-100: Thresholds 3.0 → 9.4 (TOO HIGH for 100 overlapping classes)
   • Result: Creates tight clusters that span multiple classes → loses diversity

2. TOO MANY LEADERS (Redundancy Problem)
   • CIFAR-10: ~30-35 leaders per round (1.4% of budget) → fills with diverse non-leaders
   • CIFAR-100: ~100-120 leaders per round (4.4% of budget) → 3x more redundancy
   • Result: Multi-scale clustering selects similar samples multiple times

3. SPARSE CLASS COVERAGE
   • CIFAR-10: 10 classes, 500 samples/class → plenty of representatives
   • CIFAR-100: 100 classes, 50 samples/class → many classes get NO leaders
   • Result: Advanced Leader optimizes for cluster diversity, NOT class coverage

4. k-NN DENSITY BREAKS DOWN
   • k=10 neighbors is fine for 10 well-separated classes
   • But with 100 classes, k=10 spans multiple classes → density estimation fails
   • Result: Uncertainty + density weighting backfires

5. ROUND 9 COLLAPSE (CIFAR-100)
   • Accuracy drops from 40.75% → 31.21% (-9.54%) in final round
   • Suggests catastrophic sample selection or training instability
   • Random sampling doesn't show this collapse → problem is in Advanced Leader's selection

THE IRONY: Being "advanced" made it worse. Multi-scale, density-aware, uncertainty-weighted 
           clustering is OVER-ENGINEERING for fine-grained classification. Simple Leader 
           Clustering (+3.01%) outperforms Advanced Leader (-4.61%) on CIFAR-100.

RECOMMENDATION: Use stratified sampling or margin-based active learning for 100-class problems.
"""

ax6.text(0.05, 0.95, diagnosis_text, transform=ax6.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.savefig('advanced_leader_final_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: advanced_leader_final_summary.png")

# Print summary to console
print("\n" + "="*80)
print("INVESTIGATION COMPLETE")
print("="*80)
print("\nKEY FINDINGS:")
print("  1. Advanced Leader: +5.09% on CIFAR-10, -4.61% on CIFAR-100")
print("  2. CIFAR-100 thresholds are 60-75% HIGHER → less diversity")
print("  3. CIFAR-100 creates 3x MORE leaders → more redundancy")
print("  4. Round 9 shows -9.54% accuracy COLLAPSE on CIFAR-100")
print("  5. Multi-scale clustering fails on fine-grained (100-class) problems")
print("\nRECOMMENDATION:")
print("  → Use class-aware stratified sampling for CIFAR-100")
print("  → Or switch to margin-based / uncertainty sampling")
print("  → Simple Leader Clustering is more robust for fine-grained tasks")
print("\nGenerated files:")
print("  - advanced_leader_analysis.png")
print("  - advanced_leader_summary.png")
print("  - advanced_leader_diagnostic.png")
print("  - advanced_leader_final_summary.png")
print("  - ADVANCED_LEADER_INVESTIGATION_REPORT.md")
print("\n" + "="*80)
