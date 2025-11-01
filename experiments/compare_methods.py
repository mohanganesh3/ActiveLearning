"""
Script to run comparison experiments across all sampling strategies.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from src.utils import load_config, merge_configs
from experiments.run_experiment import ActiveLearningExperiment
from src.visualization import Visualizer
import logging

logger = logging.getLogger(__name__)


def run_comparison_experiments(base_config_path: str, strategies: list, 
                              output_dir: str = "./results/comparison"):
    """
    Run experiments for multiple sampling strategies and compare results.
    
    Args:
        base_config_path: Path to base configuration
        strategies: List of strategy names to compare
        output_dir: Directory to save comparison results
    """
    base_config = load_config(base_config_path)
    results = {}
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running comparison of {len(strategies)} strategies")
    
    for strategy in strategies:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running experiment with strategy: {strategy}")
        logger.info(f"{'='*80}\n")
        
        # Create config for this strategy
        strategy_config = merge_configs(base_config, {
            'experiment': {
                'name': f"cifar10_{strategy}",
            },
            'active_learning': {
                'strategy': strategy,
            }
        })
        
        # Save modified config
        from src.utils import save_config
        config_path = output_path / f"config_{strategy}.yaml"
        save_config(strategy_config, str(config_path))
        
        # Run experiment
        try:
            experiment = ActiveLearningExperiment(str(config_path))
            experiment.run()
            
            # Collect results
            metrics_file = experiment.output_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    results[strategy] = json.load(f)
        
        except Exception as e:
            logger.error(f"Error running {strategy}: {e}")
            continue
    
    # Generate comparison visualizations
    logger.info(f"\n{'='*80}")
    logger.info("Generating comparison visualizations")
    logger.info(f"{'='*80}\n")
    
    visualizer = Visualizer(output_path / "comparison_plots")
    
    # Extract test accuracies for each strategy
    test_accuracies = {}
    for strategy, metrics in results.items():
        round_metrics = [m for m in metrics if 'round' in m and 'epoch' not in m]
        test_accs = [m.get('test_accuracy', 0) for m in round_metrics]
        test_accuracies[strategy] = test_accs
    
    # Plot learning curves comparison
    if test_accuracies:
        visualizer.plot_learning_curves(
            test_accuracies,
            metric_name='Test Accuracy',
            title='Comparison of Sampling Strategies',
            filename='strategies_comparison'
        )
    
    # Extract final metrics
    final_metrics = {}
    for strategy, metrics in results.items():
        round_metrics = [m for m in metrics if 'round' in m and 'epoch' not in m]
        if round_metrics:
            final_round = round_metrics[-1]
            final_metrics[strategy] = {
                'test_accuracy': final_round.get('test_accuracy', 0),
                'train_loss': final_round.get('train_loss', 0),
                'num_labeled': final_round.get('num_labeled', 0),
            }
    
    # Plot comparison table
    if final_metrics:
        visualizer.plot_comparison_table(
            final_metrics,
            filename='final_metrics_comparison'
        )
    
    # Calculate sample efficiency (samples to reach 85% accuracy)
    cumulative_samples = {}
    for strategy, metrics in results.items():
        round_metrics = [m for m in metrics if 'round' in m and 'epoch' not in m]
        cum_samples = [m.get('num_labeled', 0) for m in round_metrics]
        cumulative_samples[strategy] = cum_samples
    
    if cumulative_samples and test_accuracies:
        visualizer.plot_sample_efficiency(
            cumulative_samples,
            test_accuracies,
            target_accuracy=0.85,
            filename='sample_efficiency'
        )
    
    # Save comparison summary
    summary = {
        'strategies_compared': strategies,
        'final_metrics': final_metrics,
        'test_accuracies': {k: [float(x) for x in v] for k, v in test_accuracies.items()},
    }
    
    summary_file = output_path / "comparison_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Comparison complete! Results saved to {output_path}")
    logger.info(f"{'='*80}\n")
    
    # Print summary
    logger.info("\nFinal Results Summary:")
    logger.info("-" * 80)
    for strategy, metrics in final_metrics.items():
        logger.info(f"{strategy:20s}: Acc={metrics['test_accuracy']:.4f}, "
                   f"Loss={metrics['train_loss']:.4f}, "
                   f"Samples={metrics['num_labeled']}")
    logger.info("-" * 80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Active Learning Strategies')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to base configuration file'
    )
    parser.add_argument(
        '--strategies',
        nargs='+',
        default=['random', 'uncertainty', 'coreset'],
        help='List of strategies to compare'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./results/comparison',
        help='Output directory for comparison results'
    )
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run comparison
    run_comparison_experiments(args.config, args.strategies, args.output)


if __name__ == '__main__':
    main()
