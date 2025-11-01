"""
Visualize and Compare Active Learning Results
Generate plots showing accuracy curves and sampling time comparisons
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(dataset='cifar10'):
    """Load all results for a dataset"""
    results_dir = Path(f'{dataset}_results')
    strategies = ['Random', 'Greedy_K-Center', 'Leader_Clustering', 'Advanced_Leader']
    
    results = {}
    for strategy in strategies:
        result_file = results_dir / f'{strategy}_results.pkl'
        if result_file.exists():
            with open(result_file, 'rb') as f:
                results[strategy] = pickle.load(f)
    
    return results

def plot_accuracy_curves(results, dataset='CIFAR-10'):
    """Plot test accuracy curves for all strategies"""
    plt.figure(figsize=(12, 6))
    
    colors = {
        'Random': 'gray',
        'Greedy_K-Center': 'blue',
        'Leader_Clustering': 'green',
        'Advanced_Leader': 'red'
    }
    
    for strategy, data in results.items():
        labeled_sizes = data['labeled_sizes']
        accuracies = data['test_accuracies']
        plt.plot(labeled_sizes, accuracies, 'o-', label=strategy, 
                color=colors.get(strategy, 'black'), linewidth=2, markersize=8)
    
    plt.xlabel('Number of Labeled Samples', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title(f'{dataset} Active Learning - Test Accuracy Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_dir = Path(f'{dataset.lower().replace("-", "")}_results')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'accuracy_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_sampling_times(results, dataset='CIFAR-10'):
    """Plot average sampling times"""
    plt.figure(figsize=(10, 6))
    
    strategies = []
    avg_times = []
    colors_list = []
    
    colors = {
        'Random': 'gray',
        'Greedy_K-Center': 'blue',
        'Leader_Clustering': 'green',
        'Advanced_Leader': 'red'
    }
    
    for strategy, data in results.items():
        sampling_times = data['sampling_times']
        # Exclude first round (no sampling) and last round (sampling=0)
        valid_times = [t for t in sampling_times if t > 0]
        if valid_times:
            strategies.append(strategy.replace('_', ' '))
            avg_times.append(np.mean(valid_times))
            colors_list.append(colors.get(strategy, 'black'))
    
    plt.bar(range(len(strategies)), avg_times, color=colors_list, alpha=0.7, edgecolor='black')
    plt.xticks(range(len(strategies)), strategies, rotation=15, ha='right')
    plt.ylabel('Average Sampling Time (seconds)', fontsize=12)
    plt.title(f'{dataset} - Average Sampling Time per Round', fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(avg_times):
        plt.text(i, v, f'{v:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_dir = Path(f'{dataset.lower().replace("-", "")}_results')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'sampling_time_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def generate_summary_table(results_c10, results_c100):
    """Generate comparison table"""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    for dataset, results in [('CIFAR-10', results_c10), ('CIFAR-100', results_c100)]:
        print(f"\n{dataset}:")
        print("-" * 80)
        print(f"{'Strategy':<25} {'Final Accuracy':<20} {'Avg Sampling Time':<20}")
        print("-" * 80)
        
        for strategy, data in sorted(results.items()):
            final_acc = data['test_accuracies'][-1]
            valid_times = [t for t in data['sampling_times'] if t > 0]
            avg_time = np.mean(valid_times) if valid_times else 0
            
            print(f"{strategy:<25} {final_acc:>6.2f}%              {avg_time:>10.2f}s")
        
        print("-" * 80)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize Active Learning Results')
    parser.add_argument('--dataset', type=str, default='both', choices=['cifar10', 'cifar100', 'both'],
                        help='Which dataset to visualize')
    args = parser.parse_args()
    
    # Load results
    print("Loading results...")
    
    results_c10 = None
    results_c100 = None
    
    if args.dataset in ['cifar10', 'both']:
        results_c10 = load_results('cifar10')
        if not results_c10:
            print("No CIFAR-10 results found!")
        else:
            print(f"Found {len(results_c10)} CIFAR-10 results")
            plot_accuracy_curves(results_c10, 'CIFAR-10')
            plot_sampling_times(results_c10, 'CIFAR-10')
    
    if args.dataset in ['cifar100', 'both']:
        results_c100 = load_results('cifar100')
        if not results_c100:
            print("No CIFAR-100 results found!")
        else:
            print(f"Found {len(results_c100)} CIFAR-100 results")
            plot_accuracy_curves(results_c100, 'CIFAR-100')
            plot_sampling_times(results_c100, 'CIFAR-100')
    
    # Summary table
    if results_c10 and results_c100:
        generate_summary_table(results_c10, results_c100)
    
    print("\n✅ Visualization complete!")

if __name__ == '__main__':
    main()
