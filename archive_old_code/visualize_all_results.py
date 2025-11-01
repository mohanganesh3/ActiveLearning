#!/usr/bin/env python3
"""
Comprehensive Visualization of All Active Learning Results
Compares: Random, Leader Clustering, Greedy K-Center, Advanced Leader (V1, V2, V3)
Datasets: CIFAR-10 and CIFAR-100
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 8

def load_results(dataset, strategy, version='V3'):
    """Load results from pickle files"""
    if version == 'V3':
        base_dir = f'{dataset}_results'
    elif version == 'V2':
        base_dir = f'old_results_V2/{dataset}_results'
    elif version == 'V1':
        base_dir = f'old_results_BUGGY/{dataset}_results'
    else:
        raise ValueError(f"Unknown version: {version}")
    
    filepath = f'{base_dir}/{strategy}_results.pkl'
    
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def plot_comparison(dataset='cifar100'):
    """Create comprehensive comparison plots"""
    
    # Strategy configurations
    strategies = {
        'Random': ('Random', 'gray', '--'),
        'Leader_Clustering': ('Leader Clustering', 'green', '-.'),
        'Greedy_K-Center': ('Greedy K-Center', 'red', ':'),
        'Advanced_Leader_V1': ('Advanced Leader V1', 'blue', '-'),
        'Advanced_Leader_V2': ('Advanced Leader V2', 'orange', '-'),
        'Advanced_Leader_V3': ('Advanced Leader V3', 'purple', '-'),
    }
    
    # Load all data
    data = {}
    for key, (name, color, style) in strategies.items():
        if 'V3' in key:
            result = load_results(dataset, 'Advanced_Leader', 'V3')
        elif 'V2' in key:
            result = load_results(dataset, 'Advanced_Leader', 'V2')
        elif 'V1' in key:
            result = load_results(dataset, 'Advanced_Leader', 'V1')
        else:
            # Try V1 first (old_results_BUGGY has baselines), then V2
            result = load_results(dataset, key, 'V1')
            if result is None:
                result = load_results(dataset, key, 'V2')
        
        if result is not None:
            data[key] = {
                'name': name,
                'color': color,
                'style': style,
                'accuracies': result['test_accuracies'],
                'sampling_times': result.get('sampling_times', [0] * len(result['test_accuracies'])),
                'rounds': list(range(1, len(result['test_accuracies']) + 1))
            }
    
    if not data:
        print(f"No data found for {dataset}")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Main accuracy comparison (large plot)
    ax1 = plt.subplot(2, 3, (1, 4))
    for key, info in data.items():
        ax1.plot(info['rounds'], info['accuracies'], 
                label=info['name'], color=info['color'], 
                linestyle=info['style'], marker='o')
    
    ax1.set_xlabel('Round', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_title(f'{dataset.upper()} - Active Learning Strategy Comparison', 
                  fontsize=16, fontweight='bold')
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, 10))
    
    # Add horizontal line for random baseline
    if 'Random' in data:
        random_final = data['Random']['accuracies'][-1]
        ax1.axhline(y=random_final, color='gray', linestyle='--', 
                   alpha=0.5, label=f'Random Final ({random_final:.2f}%)')
    
    # 2. Final accuracy bar chart
    ax2 = plt.subplot(2, 3, 2)
    names = [info['name'] for info in data.values()]
    final_accs = [info['accuracies'][-1] for info in data.values()]
    colors = [info['color'] for info in data.values()]
    
    bars = ax2.barh(names, final_accs, color=colors, alpha=0.7)
    ax2.set_xlabel('Final Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Final Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, final_accs)):
        ax2.text(acc + 0.5, i, f'{acc:.2f}%', va='center', fontsize=9)
    
    # 3. Round-by-round improvement
    ax3 = plt.subplot(2, 3, 3)
    for key, info in data.items():
        if 'Advanced_Leader' in key:  # Only show advanced leader versions
            improvements = [0] + [info['accuracies'][i] - info['accuracies'][i-1] 
                                 for i in range(1, len(info['accuracies']))]
            ax3.plot(info['rounds'], improvements, 
                    label=info['name'], color=info['color'], 
                    linestyle=info['style'], marker='s', markersize=6)
    
    ax3.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Accuracy Gain (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Round-by-Round Improvement', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', framealpha=0.9, fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xticks(range(1, 10))
    
    # 4. Sampling time comparison
    ax4 = plt.subplot(2, 3, 5)
    for key, info in data.items():
        if len(info['sampling_times']) > 1:  # Skip if no sampling data
            # Skip round 1 (usually 0)
            rounds_with_sampling = info['rounds'][1:]
            sampling_times = info['sampling_times'][1:]
            if sampling_times and any(t > 0 for t in sampling_times):
                ax4.plot(rounds_with_sampling, sampling_times, 
                        label=info['name'], color=info['color'], 
                        linestyle=info['style'], marker='d', markersize=5)
    
    ax4.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Sampling Time (seconds)', fontsize=12, fontweight='bold')
    ax4.set_title('Computational Efficiency', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', framealpha=0.9, fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(range(2, 10))
    
    # 5. Cumulative gain over random
    ax5 = plt.subplot(2, 3, 6)
    if 'Random' in data:
        random_accs = data['Random']['accuracies']
        for key, info in data.items():
            if key != 'Random':
                gains = [info['accuracies'][i] - random_accs[i] 
                        for i in range(len(info['accuracies']))]
                ax5.plot(info['rounds'], gains, 
                        label=info['name'], color=info['color'], 
                        linestyle=info['style'], marker='o', markersize=5)
        
        ax5.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Gain vs Random (%)', fontsize=12, fontweight='bold')
        ax5.set_title('Advantage Over Random Baseline', fontsize=14, fontweight='bold')
        ax5.legend(loc='best', framealpha=0.9, fontsize=8)
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax5.set_xticks(range(1, 10))
    
    plt.tight_layout()
    
    # Save figure
    output_file = f'visualizations_{dataset}_complete.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    
    return fig

def create_summary_table(dataset='cifar100'):
    """Create detailed comparison table"""
    
    strategies = {
        'Random': 'V1',
        'Leader_Clustering': 'V1',
        'Greedy_K-Center': 'V1',
        'Advanced_Leader_V1': 'V1',
        'Advanced_Leader_V2': 'V2',
        'Advanced_Leader_V3': 'V3',
    }
    
    print(f"\n{'='*80}")
    print(f"{dataset.upper()} - COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    # Load all data
    results = {}
    for strategy_key, version in strategies.items():
        if 'Advanced_Leader' in strategy_key:
            result = load_results(dataset, 'Advanced_Leader', version)
            name = f"Advanced Leader {version}"
        else:
            result = load_results(dataset, strategy_key, version)
            name = strategy_key.replace('_', ' ')
        
        if result is not None:
            results[name] = result
    
    # Print table header
    print(f"{'Strategy':<25} {'Final Acc':<12} {'Best Round':<12} {'Worst Round':<12} {'Volatility (σ)':<15} {'Avg Sampling':<15}")
    print(f"{'-'*95}")
    
    # Collect data for sorting
    summary_data = []
    
    for name, result in results.items():
        accs = result['test_accuracies']
        final_acc = accs[-1]
        best_acc = max(accs)
        worst_acc = min(accs)
        volatility = np.std(accs)
        
        sampling_times = result.get('sampling_times', [0] * len(accs))
        avg_sampling = np.mean([t for t in sampling_times[1:] if t > 0]) if len(sampling_times) > 1 else 0
        
        summary_data.append({
            'name': name,
            'final_acc': final_acc,
            'best_acc': best_acc,
            'worst_acc': worst_acc,
            'volatility': volatility,
            'avg_sampling': avg_sampling
        })
    
    # Sort by final accuracy (descending)
    summary_data.sort(key=lambda x: x['final_acc'], reverse=True)
    
    # Print sorted results
    for data in summary_data:
        print(f"{data['name']:<25} {data['final_acc']:>6.2f}%      "
              f"{data['best_acc']:>6.2f}%      "
              f"{data['worst_acc']:>6.2f}%      "
              f"{data['volatility']:>6.2f}%        "
              f"{data['avg_sampling']:>6.1f}s")
    
    print(f"\n{'='*80}")
    
    # Calculate improvements
    if summary_data:
        random_acc = next((d['final_acc'] for d in summary_data if 'Random' in d['name']), None)
        if random_acc:
            print(f"\n{'Strategy':<25} {'vs Random':<15} {'Rank':<10}")
            print(f"{'-'*50}")
            for i, data in enumerate(summary_data, 1):
                if 'Random' not in data['name']:
                    improvement = data['final_acc'] - random_acc
                    print(f"{data['name']:<25} {improvement:>+6.2f}%         #{i-1}")

def create_version_comparison(dataset='cifar100'):
    """Detailed comparison of V1 vs V2 vs V3"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    versions = ['V1', 'V2', 'V3']
    colors = ['blue', 'orange', 'purple']
    markers = ['o', 's', '^']
    
    data = {}
    for version in versions:
        result = load_results(dataset, 'Advanced_Leader', version)
        if result is not None:
            data[version] = result
    
    if not data:
        print(f"No version data found for {dataset}")
        return
    
    # Plot 1: Accuracy comparison
    ax = axes[0, 0]
    for version, color, marker in zip(versions, colors, markers):
        if version in data:
            accs = data[version]['test_accuracies']
            rounds = list(range(1, len(accs) + 1))
            ax.plot(rounds, accs, label=f'V{version[-1]}', 
                   color=color, marker=marker, linewidth=2.5, markersize=8)
    
    ax.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{dataset.upper()} - Version Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 10))
    
    # Plot 2: Round-by-round differences (V2 vs V1, V3 vs V1)
    ax = axes[0, 1]
    if 'V1' in data:
        v1_accs = data['V1']['test_accuracies']
        rounds = list(range(1, len(v1_accs) + 1))
        
        for version, color, marker in zip(['V2', 'V3'], ['orange', 'purple'], ['s', '^']):
            if version in data:
                # Handle different array lengths
                min_len = min(len(v1_accs), len(data[version]['test_accuracies']))
                diffs = [data[version]['test_accuracies'][i] - v1_accs[i] 
                        for i in range(min_len)]
                rounds_diff = list(range(1, min_len + 1))
                ax.plot(rounds_diff, diffs, label=f'{version} - V1', 
                       color=color, marker=marker, linewidth=2.5, markersize=8)
    
    ax.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy Difference (%)', fontsize=12, fontweight='bold')
    ax.set_title('Difference from V1 Baseline', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xticks(range(1, 10))
    
    # Highlight critical failures
    if 'V2' in data and 'V1' in data:
        v1_accs = data['V1']['test_accuracies']
        v2_accs = data['V2']['test_accuracies']
        min_len = min(len(v1_accs), len(v2_accs))
        for i in range(min_len):
            v1, v2 = v1_accs[i], v2_accs[i]
            if v2 < v1 - 5:  # More than 5% drop
                ax.plot(i+1, v2 - v1, 'ro', markersize=12, alpha=0.5)
                ax.annotate(f'R{i+1}\n-{v1-v2:.1f}%', 
                          xy=(i+1, v2-v1), xytext=(i+1+0.3, v2-v1-1),
                          fontsize=9, color='red', fontweight='bold')
    
    # Plot 3: Cumulative performance
    ax = axes[1, 0]
    for version, color, marker in zip(versions, colors, markers):
        if version in data:
            accs = data[version]['test_accuracies']
            cumulative = np.cumsum(accs)
            rounds = list(range(1, len(accs) + 1))
            ax.plot(rounds, cumulative, label=f'V{version[-1]}', 
                   color=color, marker=marker, linewidth=2.5, markersize=8)
    
    ax.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Performance', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 10))
    
    # Plot 4: Statistical summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"{dataset.upper()} - Statistical Summary\n" + "="*40 + "\n\n"
    
    for version in versions:
        if version in data:
            accs = data[version]['test_accuracies']
            summary_text += f"{version}:\n"
            summary_text += f"  Final:     {accs[-1]:6.2f}%\n"
            summary_text += f"  Mean:      {np.mean(accs):6.2f}%\n"
            summary_text += f"  Std Dev:   {np.std(accs):6.2f}%\n"
            summary_text += f"  Max:       {max(accs):6.2f}%\n"
            summary_text += f"  Min:       {min(accs):6.2f}%\n"
            summary_text += f"  Range:     {max(accs) - min(accs):6.2f}%\n\n"
    
    # Add comparison
    if 'V1' in data and 'V2' in data:
        v1_final = data['V1']['test_accuracies'][-1]
        v2_final = data['V2']['test_accuracies'][-1]
        summary_text += f"V2 vs V1: {v2_final - v1_final:+.2f}% "
        summary_text += "❌ FAILURE\n" if v2_final < v1_final else "✅ SUCCESS\n"
    
    if 'V1' in data and 'V3' in data:
        v1_final = data['V1']['test_accuracies'][-1]
        v3_final = data['V3']['test_accuracies'][-1]
        summary_text += f"V3 vs V1: {v3_final - v1_final:+.2f}% "
        summary_text += "✅ MATCH" if abs(v3_final - v1_final) < 0.01 else "📊 CHANGED"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
           fontsize=11, verticalalignment='top', 
           family='monospace', bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_file = f'visualizations_{dataset}_versions.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    
    return fig

def main():
    """Generate all visualizations"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ACTIVE LEARNING RESULTS VISUALIZATION")
    print("="*80 + "\n")
    
    for dataset in ['cifar100', 'cifar10']:
        print(f"\n{'='*80}")
        print(f"Processing {dataset.upper()}...")
        print(f"{'='*80}\n")
        
        # Create comprehensive comparison
        plot_comparison(dataset)
        
        # Create version comparison
        create_version_comparison(dataset)
        
        # Print summary table
        create_summary_table(dataset)
        
        print(f"\n✅ Completed {dataset.upper()} visualizations\n")
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - visualizations_cifar100_complete.png")
    print("  - visualizations_cifar100_versions.png")
    print("  - visualizations_cifar10_complete.png")
    print("  - visualizations_cifar10_versions.png")
    print("\n")

if __name__ == '__main__':
    main()
