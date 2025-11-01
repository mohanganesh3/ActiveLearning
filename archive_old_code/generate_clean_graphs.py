#!/usr/bin/env python3
"""
Simple, Clean Visualizations - One Graph Per Image
Compares V3 (Advanced Leader) with baselines: Random, Leader Clustering, Greedy K-Center
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

def load_results(dataset, strategy, version='current'):
    """Load results from pickle files"""
    if version == 'current':
        base_dir = f'{dataset}_results'
        strategy_file = strategy
    else:
        base_dir = f'old_results_BUGGY/{dataset}_results'
        strategy_file = strategy
    
    filepath = f'{base_dir}/{strategy_file}_results.pkl'
    
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def plot_single_comparison(dataset='cifar100'):
    """Create single clean comparison plot"""
    
    # Load data
    strategies = {
        'Random': ('Random', '#808080', '--', 's'),
        'Leader_Clustering': ('Leader Clustering', '#2ecc71', '-.', '^'),
        'Greedy_K-Center': ('Greedy K-Center', '#e74c3c', ':', 'D'),
        'Advanced_Leader': ('Advanced Leader V3', '#9b59b6', '-', 'o'),
    }
    
    data = {}
    for key, (name, color, style, marker) in strategies.items():
        result = load_results(dataset, key, 'current')
        if result is None:  # Try old_results_BUGGY
            result = load_results(dataset, key, 'old')
        
        if result is not None:
            data[key] = {
                'name': name,
                'color': color,
                'style': style,
                'marker': marker,
                'accuracies': result['test_accuracies'],
                'rounds': list(range(1, len(result['test_accuracies']) + 1))
            }
    
    if not data:
        print(f"No data found for {dataset}")
        return
    
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
    ax.set_title(f'{dataset_name}: Active Learning Strategy Comparison', 
                fontsize=18, fontweight='bold', pad=20)
    
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
    
    plt.tight_layout()
    
    # Save
    output_file = f'{dataset}_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_file}")
    plt.close()

def plot_final_comparison_bar(dataset='cifar100'):
    """Create bar chart of final accuracies"""
    
    strategies = {
        'Random': ('Random', '#808080'),
        'Leader_Clustering': ('Leader\nClustering', '#2ecc71'),
        'Greedy_K-Center': ('Greedy\nK-Center', '#e74c3c'),
        'Advanced_Leader': ('Advanced\nLeader V3', '#9b59b6'),
    }
    
    data = {}
    for key, (name, color) in strategies.items():
        result = load_results(dataset, key, 'current')
        if result is None:
            result = load_results(dataset, key, 'old')
        
        if result is not None:
            data[key] = {
                'name': name,
                'color': color,
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
    ax.set_title(f'{dataset_name}: Final Accuracy Comparison', 
                fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_file = f'{dataset}_final_accuracy.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_file}")
    plt.close()

def plot_gain_over_random(dataset='cifar100'):
    """Plot gain over random baseline"""
    
    strategies = {
        'Leader_Clustering': ('Leader Clustering', '#2ecc71', '^'),
        'Greedy_K-Center': ('Greedy K-Center', '#e74c3c', 'D'),
        'Advanced_Leader': ('Advanced Leader V3', '#9b59b6', 'o'),
    }
    
    # Load random baseline
    random_result = load_results(dataset, 'Random', 'current')
    if random_result is None:
        random_result = load_results(dataset, 'Random', 'old')
    
    if random_result is None:
        return
    
    random_accs = random_result['test_accuracies']
    
    data = {}
    for key, (name, color, marker) in strategies.items():
        result = load_results(dataset, key, 'current')
        if result is None:
            result = load_results(dataset, key, 'old')
        
        if result is not None:
            accs = result['test_accuracies']
            gains = [accs[i] - random_accs[i] for i in range(len(accs))]
            data[key] = {
                'name': name,
                'color': color,
                'marker': marker,
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
    ax.set_ylabel('Accuracy Gain vs Random (%)', fontsize=16, fontweight='bold')
    ax.set_title(f'{dataset_name}: Advantage Over Random Baseline', 
                fontsize=18, fontweight='bold', pad=20)
    
    ax.legend(loc='best', fontsize=14, framealpha=0.95, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.5)
    ax.set_xticks(range(1, 10))
    
    # Annotate final gains
    for key, info in data.items():
        final_gain = info['gains'][-1]
        final_round = info['rounds'][-1]
        ax.annotate(f'{final_gain:+.1f}%', 
                   xy=(final_round, final_gain),
                   xytext=(10, 0), 
                   textcoords='offset points',
                   fontsize=11,
                   color=info['color'],
                   fontweight='bold')
    
    plt.tight_layout()
    
    output_file = f'{dataset}_gain_over_random.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_file}")
    plt.close()

def print_summary(dataset='cifar100'):
    """Print summary table"""
    
    strategies = {
        'Random': 'Random',
        'Leader_Clustering': 'Leader Clustering',
        'Greedy_K-Center': 'Greedy K-Center',
        'Advanced_Leader': 'Advanced Leader V3',
    }
    
    print(f"\n{'='*70}")
    print(f"{dataset.upper()} - RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Strategy':<25} {'Final Accuracy':<20} {'vs Random':<15}")
    print(f"{'-'*70}")
    
    results = []
    random_acc = None
    
    for key, name in strategies.items():
        result = load_results(dataset, key, 'current')
        if result is None:
            result = load_results(dataset, key, 'old')
        
        if result is not None:
            final_acc = result['test_accuracies'][-1]
            results.append((name, final_acc))
            if key == 'Random':
                random_acc = final_acc
    
    # Sort by accuracy
    results.sort(key=lambda x: x[1], reverse=True)
    
    for name, acc in results:
        if random_acc is not None and name != 'Random':
            gain = acc - random_acc
            print(f"{name:<25} {acc:>6.2f}%             {gain:>+6.2f}%")
        else:
            print(f"{name:<25} {acc:>6.2f}%             {'baseline':>8}")
    
    print(f"\n{'='*70}\n")

def main():
    """Generate clean, simple visualizations"""
    
    print("\n" + "="*70)
    print("GENERATING CLEAN COMPARISON GRAPHS")
    print("One graph per image - V3 vs all baselines")
    print("="*70 + "\n")
    
    for dataset in ['cifar100', 'cifar10']:
        print(f"\nProcessing {dataset.upper()}...")
        print("-" * 70)
        
        # 1. Main comparison (line plot)
        plot_single_comparison(dataset)
        
        # 2. Final accuracy bar chart
        plot_final_comparison_bar(dataset)
        
        # 3. Gain over random
        plot_gain_over_random(dataset)
        
        # 4. Print summary
        print_summary(dataset)
    
    print("\n" + "="*70)
    print("✅ ALL GRAPHS GENERATED")
    print("="*70)
    print("\nGenerated files:")
    print("  CIFAR-100:")
    print("    - cifar100_comparison.png (main comparison)")
    print("    - cifar100_final_accuracy.png (bar chart)")
    print("    - cifar100_gain_over_random.png (vs baseline)")
    print("\n  CIFAR-10:")
    print("    - cifar10_comparison.png (main comparison)")
    print("    - cifar10_final_accuracy.png (bar chart)")
    print("    - cifar10_gain_over_random.png (vs baseline)")
    print("\n")

if __name__ == '__main__':
    main()
