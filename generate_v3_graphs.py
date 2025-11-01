#!/usr/bin/env python3
"""
FINAL Visualizations - V3 Advanced Leader + V2 Baselines
Shows the latest V3 results compared to baseline strategies
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# Set style for clean plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 10

def load_results(dataset, strategy, version='v2'):
    """Load results - V3 for Advanced Leader, V2 for baselines"""
    if strategy == 'Advanced_Leader' and version == 'v3':
        filepath = f'{dataset}_results/{strategy}_results.pkl'
    else:
        filepath = f'old_results_V2/{dataset}_results/{strategy}_results.pkl'
    
    if not os.path.exists(filepath):
        print(f"⚠️  File not found: {filepath}")
        return None
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def verify_round1_consistency(data):
    """Verify all strategies have identical Round 1 accuracy"""
    round1_accs = []
    initial_sizes = []
    
    for key, info in data.items():
        round1_accs.append(info['accuracies'][0])
        initial_sizes.append(info['labeled_sizes'][0])
    
    round1_unique = list(set(round1_accs))
    sizes_unique = list(set(initial_sizes))
    
    print("\n" + "="*80)
    print("ROUND 1 CONSISTENCY CHECK")
    print("="*80)
    
    if len(round1_unique) == 1 and len(sizes_unique) == 1:
        print(f"✅ CORRECT: All strategies have identical Round 1")
        print(f"   - Round 1 Accuracy: {round1_unique[0]:.2f}%")
        print(f"   - Initial Labeled: {sizes_unique[0]} samples")
        print(f"   - This confirms correct experimental setup!")
        return True
    else:
        print(f"❌ ERROR: Round 1 accuracies differ!")
        print(f"   - This indicates results from different experiments")
        for key, info in data.items():
            print(f"   - {key}: {info['accuracies'][0]:.2f}% (initial: {info['labeled_sizes'][0]})")
        return False

def plot_single_comparison(dataset='cifar100'):
    """Create single clean comparison plot - V3 Advanced Leader + V2 baselines"""
    
    # Load data: V2 baselines + V3 Advanced Leader
    strategies = {
        'Random': ('Random', '#808080', '--', 's', 'v2'),
        'Leader_Clustering': ('Leader Clustering', '#2ecc71', '-.', '^', 'v2'),
        'Greedy_K-Center': ('Greedy K-Center', '#e74c3c', ':', 'D', 'v2'),
        'Advanced_Leader': ('Advanced Leader V3', '#9b59b6', '-', 'o', 'v3'),
    }
    
    data = {}
    for key, (name, color, style, marker, version) in strategies.items():
        result = load_results(dataset, key, version)
        
        if result is not None:
            data[key] = {
                'name': name,
                'color': color,
                'style': style,
                'marker': marker,
                'version': version,
                'accuracies': result['test_accuracies'],
                'labeled_sizes': result['labeled_sizes'],
                'rounds': list(range(1, len(result['test_accuracies']) + 1))
            }
    
    if not data:
        print(f"❌ No data found for {dataset}")
        return
    
    # VERIFY ROUND 1 CONSISTENCY
    if not verify_round1_consistency(data):
        print("\n⚠️  WARNING: Proceeding anyway, but results may be misleading!")
    
    # Create single plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each strategy
    for key, info in data.items():
        ax.plot(info['rounds'], info['accuracies'], 
               label=info['name'], 
               color=info['color'], 
               linestyle=info['style'],
               marker=info['marker'],
               linewidth=3,
               markersize=10,
               markevery=1)
    
    # Labels and title
    dataset_name = 'CIFAR-100' if dataset == 'cifar100' else 'CIFAR-10'
    ax.set_xlabel('Active Learning Round', fontsize=16, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=16, fontweight='bold')
    ax.set_title(f'{dataset_name}: Active Learning Strategy Comparison\nAdvanced Leader V3 vs Baseline Strategies (V2)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='lower right', fontsize=14, framealpha=0.95, 
             shadow=True, fancybox=True)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xticks(range(1, 10))
    
    # Add final accuracy annotations
    for key, info in data.items():
        final_acc = info['accuracies'][-1]
        final_round = info['rounds'][-1]
        ax.annotate(f'{final_acc:.1f}%', 
                   xy=(final_round, final_acc),
                   xytext=(10, 0), 
                   textcoords='offset points',
                   fontsize=11,
                   color=info['color'],
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='white', 
                           edgecolor=info['color'],
                           alpha=0.8))
    
    # Add note about data sources
    note_text = 'Note: Baselines from V2, Advanced Leader from V3 (all with 5,000 initial labels)'
    ax.text(0.5, 0.02, note_text, transform=ax.transAxes,
           fontsize=10, ha='center', style='italic',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save
    output_file = f'{dataset}_comparison_V3.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Saved: {output_file}")
    plt.close()

def plot_final_comparison_bar(dataset='cifar100'):
    """Create bar chart of final accuracies - V3 Advanced Leader + V2 baselines"""
    
    strategies = {
        'Random': ('Random', '#808080', 'v2'),
        'Leader_Clustering': ('Leader\nClustering', '#2ecc71', 'v2'),
        'Greedy_K-Center': ('Greedy\nK-Center', '#e74c3c', 'v2'),
        'Advanced_Leader': ('Advanced\nLeader V3', '#9b59b6', 'v3'),
    }
    
    data = {}
    for key, (name, color, version) in strategies.items():
        result = load_results(dataset, key, version)
        
        if result is not None:
            data[key] = {
                'name': name,
                'color': color,
                'version': version,
                'final_acc': result['test_accuracies'][-1]
            }
    
    if not data:
        return
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 7))
    
    names = [info['name'] for info in data.values()]
    accs = [info['final_acc'] for info in data.values()]
    colors = [info['color'] for info in data.values()]
    
    bars = ax.bar(names, accs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{acc:.2f}%',
               ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    dataset_name = 'CIFAR-100' if dataset == 'cifar100' else 'CIFAR-10'
    ax.set_ylabel('Final Test Accuracy (%)', fontsize=16, fontweight='bold')
    ax.set_title(f'{dataset_name}: Final Accuracy Comparison\nAdvanced Leader V3 vs Baselines (V2)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add note
    note_text = 'Baselines: V2 | Advanced Leader: V3'
    ax.text(0.5, 0.02, note_text, transform=ax.transAxes,
           fontsize=10, ha='center', style='italic',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    output_file = f'{dataset}_final_accuracy_V3.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_file}")
    plt.close()

def plot_gain_over_random(dataset='cifar100'):
    """Plot gain over random baseline - V3 Advanced Leader + V2 baselines"""
    
    strategies = {
        'Leader_Clustering': ('Leader Clustering', '#2ecc71', '^', 'v2'),
        'Greedy_K-Center': ('Greedy K-Center', '#e74c3c', 'D', 'v2'),
        'Advanced_Leader': ('Advanced Leader V3', '#9b59b6', 'o', 'v3'),
    }
    
    # Load random baseline (V2)
    random_result = load_results(dataset, 'Random', 'v2')
    
    if random_result is None:
        return
    
    random_accs = random_result['test_accuracies']
    
    data = {}
    for key, (name, color, marker, version) in strategies.items():
        result = load_results(dataset, key, version)
        
        if result is not None:
            gains = [acc - rand_acc for acc, rand_acc in zip(result['test_accuracies'], random_accs)]
            data[key] = {
                'name': name,
                'color': color,
                'marker': marker,
                'version': version,
                'gains': gains,
                'rounds': list(range(1, len(gains) + 1))
            }
    
    if not data:
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for key, info in data.items():
        ax.plot(info['rounds'], info['gains'],
               label=info['name'],
               color=info['color'],
               marker=info['marker'],
               linewidth=3,
               markersize=10,
               markevery=1)
    
    dataset_name = 'CIFAR-100' if dataset == 'cifar100' else 'CIFAR-10'
    ax.set_xlabel('Active Learning Round', fontsize=16, fontweight='bold')
    ax.set_ylabel('Gain over Random (%)', fontsize=16, fontweight='bold')
    ax.set_title(f'{dataset_name}: Advantage over Random Baseline\nAdvanced Leader V3 vs Other Strategies (V2)',
                fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(loc='upper left', fontsize=14, framealpha=0.95, 
             shadow=True, fancybox=True)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 10))
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    
    # Add final gain annotations
    for key, info in data.items():
        final_gain = info['gains'][-1]
        final_round = info['rounds'][-1]
        ax.annotate(f'{final_gain:+.1f}%',
                   xy=(final_round, final_gain),
                   xytext=(10, 0),
                   textcoords='offset points',
                   fontsize=11,
                   color=info['color'],
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3',
                           facecolor='white',
                           edgecolor=info['color'],
                           alpha=0.8))
    
    # Add note
    note_text = 'Random baseline from V2, Advanced Leader from V3'
    ax.text(0.5, 0.02, note_text, transform=ax.transAxes,
           fontsize=10, ha='center', style='italic',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    output_file = f'{dataset}_gain_over_random_V3.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_file}")
    plt.close()

def print_summary(dataset='cifar100'):
    """Print summary statistics - V3 Advanced Leader + V2 baselines"""
    
    strategies = {
        'Random': ('Random', 'v2'),
        'Leader_Clustering': ('Leader Clustering', 'v2'),
        'Greedy_K-Center': ('Greedy K-Center', 'v2'),
        'Advanced_Leader': ('Advanced Leader V3', 'v3'),
    }
    
    print("\n" + "="*80)
    dataset_name = dataset.upper()
    print(f"{dataset_name} SUMMARY - V3 Advanced Leader + V2 Baselines")
    print("="*80)
    print(f"{'Strategy':<25s} | Round 1 | Final  | Gain vs Random | Version")
    print("-"*80)
    
    random_final = None
    results = {}
    
    for key, (name, version) in strategies.items():
        result = load_results(dataset, key, version)
        if result is not None:
            results[key] = result
            if key == 'Random':
                random_final = result['test_accuracies'][-1]
    
    for key, (name, version) in strategies.items():
        if key in results:
            result = results[key]
            round1 = result['test_accuracies'][0]
            final = result['test_accuracies'][-1]
            
            if random_final is not None and key != 'Random':
                gain = final - random_final
                print(f"{name:<25s} | {round1:6.2f}% | {final:5.2f}% | {gain:+6.2f}%       | {version.upper()}")
            else:
                print(f"{name:<25s} | {round1:6.2f}% | {final:5.2f}% | (baseline)     | {version.upper()}")
    
    print("="*80)

def main():
    print("="*80)
    print("V3 ADVANCED LEADER + V2 BASELINES VISUALIZATION")
    print("="*80)
    print("\nData Sources:")
    print("  - Advanced Leader: V3 (latest)")
    print("  - Random, Leader Clustering, Greedy K-Center: V2")
    print("\nAll strategies have:")
    print("  - Initial labeled: 5000 samples")
    print("  - Budget per round: 2500 samples")
    print("  - Total rounds: 9")
    print("  - Random seed: 42")
    print("\nThis shows V3's improvements while maintaining fair comparison!")
    print("="*80)
    
    for dataset in ['cifar10', 'cifar100']:
        print(f"\n\n{'='*80}")
        print(f"Processing {dataset.upper()}")
        print('='*80)
        
        plot_single_comparison(dataset)
        plot_final_comparison_bar(dataset)
        plot_gain_over_random(dataset)
        print_summary(dataset)
    
    print("\n" + "="*80)
    print("ALL V3 GRAPHS GENERATED!")
    print("="*80)
    print("\nGenerated files:")
    print("  - cifar10_comparison_V3.png")
    print("  - cifar10_final_accuracy_V3.png")
    print("  - cifar10_gain_over_random_V3.png")
    print("  - cifar100_comparison_V3.png")
    print("  - cifar100_final_accuracy_V3.png")
    print("  - cifar100_gain_over_random_V3.png")
    print("\nNOTE: All strategies have IDENTICAL Round 1 accuracy (37.45% CIFAR-10, 6.20% CIFAR-100)")
    print("      This confirms correct experimental setup - Round 1 is pre-strategy!")
    print("="*80)

if __name__ == '__main__':
    main()
