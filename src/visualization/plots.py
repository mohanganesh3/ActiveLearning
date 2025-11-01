"""
Visualization utilities for active learning experiments.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100


class Visualizer:
    """Create visualizations for active learning experiments."""
    
    def __init__(self, output_dir: Path, dpi: int = 300, format: str = 'png'):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
            dpi: DPI for saved figures
            format: Image format ('png', 'pdf', 'svg')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.format = format
        logger.info(f"Initialized visualizer, saving to: {output_dir}")
    
    def plot_learning_curves(self,
                            experiments: Dict[str, List[float]],
                            metric_name: str = 'Test Accuracy',
                            xlabel: str = 'Round',
                            title: Optional[str] = None,
                            filename: str = 'learning_curves') -> None:
        """
        Plot learning curves for multiple experiments.
        
        Args:
            experiments: Dict mapping experiment name to metric values per round
            metric_name: Name of the metric being plotted
            xlabel: Label for x-axis
            title: Plot title (auto-generated if None)
            filename: Filename for saved plot
        """
        plt.figure(figsize=(10, 6))
        
        for exp_name, values in experiments.items():
            rounds = list(range(len(values)))
            plt.plot(rounds, values, marker='o', label=exp_name, linewidth=2, markersize=6)
        
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.title(title or f'{metric_name} vs {xlabel}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / f"{filename}.{self.format}"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved learning curves to {save_path}")
    
    def plot_sample_efficiency(self,
                              experiments: Dict[str, List[int]],
                              accuracies: Dict[str, List[float]],
                              target_accuracy: float = 0.85,
                              filename: str = 'sample_efficiency') -> None:
        """
        Plot sample efficiency: samples needed to reach target accuracy.
        
        Args:
            experiments: Dict mapping experiment name to cumulative samples per round
            accuracies: Dict mapping experiment name to accuracy per round
            target_accuracy: Target accuracy threshold
            filename: Filename for saved plot
        """
        plt.figure(figsize=(10, 6))
        
        efficiency_data = []
        for exp_name in experiments.keys():
            samples = experiments[exp_name]
            accs = accuracies[exp_name]
            
            # Find first round reaching target
            for i, acc in enumerate(accs):
                if acc >= target_accuracy:
                    efficiency_data.append((exp_name, samples[i]))
                    break
            else:
                # Didn't reach target
                efficiency_data.append((exp_name, samples[-1]))
        
        # Sort by efficiency (fewer samples = better)
        efficiency_data.sort(key=lambda x: x[1])
        
        names = [x[0] for x in efficiency_data]
        values = [x[1] for x in efficiency_data]
        
        plt.barh(names, values, color=sns.color_palette("viridis", len(names)))
        plt.xlabel('Samples Required', fontsize=12)
        plt.title(f'Sample Efficiency (to reach {target_accuracy:.1%} accuracy)', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / f"{filename}.{self.format}"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved sample efficiency plot to {save_path}")
    
    def plot_comparison_table(self,
                             experiments: Dict[str, Dict[str, float]],
                             filename: str = 'comparison_table') -> None:
        """
        Create a comparison table of final metrics.
        
        Args:
            experiments: Dict mapping experiment name to final metrics dict
            filename: Filename for saved plot
        """
        # Extract metrics
        exp_names = list(experiments.keys())
        metric_names = list(experiments[exp_names[0]].keys()) if exp_names else []
        
        # Create table data
        table_data = []
        for metric in metric_names:
            row = [metric]
            for exp_name in exp_names:
                value = experiments[exp_name].get(metric, 0)
                if isinstance(value, float):
                    row.append(f"{value:.4f}")
                else:
                    row.append(str(value))
            table_data.append(row)
        
        fig, ax = plt.subplots(figsize=(12, len(metric_names) * 0.5 + 1))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=table_data,
                        colLabels=['Metric'] + exp_names,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.2] + [0.15] * len(exp_names))
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(exp_names) + 1):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Experiment Comparison', fontsize=14, fontweight='bold', pad=20)
        
        save_path = self.output_dir / f"{filename}.{self.format}"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved comparison table to {save_path}")
    
    def plot_round_breakdown(self,
                            round_metrics: Dict[int, Dict[str, float]],
                            metrics_to_plot: List[str],
                            filename: str = 'round_breakdown') -> None:
        """
        Plot multiple metrics across rounds in subplots.
        
        Args:
            round_metrics: Dict mapping round number to metrics dict
            metrics_to_plot: List of metric names to plot
            filename: Filename for saved plot
        """
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics))
        
        if n_metrics == 1:
            axes = [axes]
        
        rounds = sorted(round_metrics.keys())
        
        for idx, metric_name in enumerate(metrics_to_plot):
            values = [round_metrics[r].get(metric_name, 0) for r in rounds]
            
            axes[idx].plot(rounds, values, marker='o', linewidth=2, markersize=6)
            axes[idx].set_xlabel('Round', fontsize=11)
            axes[idx].set_ylabel(metric_name, fontsize=11)
            axes[idx].set_title(f'{metric_name} per Round', fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / f"{filename}.{self.format}"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved round breakdown to {save_path}")
    
    def generate_experiment_report(self,
                                  metrics_file: Path,
                                  experiment_names: Optional[List[str]] = None) -> None:
        """
        Generate a comprehensive visualization report from metrics files.
        
        Args:
            metrics_file: Path to metrics JSON file
            experiment_names: Optional list of experiment names for comparison
        """
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Extract round-level metrics
        round_metrics = {}
        for entry in metrics:
            if 'round' in entry and 'epoch' not in entry:
                round_num = entry['round']
                round_metrics[round_num] = entry
        
        # Plot key metrics
        if round_metrics:
            test_accs = [m.get('test_accuracy', 0) for m in round_metrics.values()]
            train_losses = [m.get('train_loss', 0) for m in round_metrics.values()]
            
            # Learning curve
            self.plot_learning_curves(
                {'Experiment': test_accs},
                metric_name='Test Accuracy',
                filename='learning_curve'
            )
            
            # Loss curve
            self.plot_learning_curves(
                {'Experiment': train_losses},
                metric_name='Training Loss',
                filename='loss_curve'
            )
            
            # Round breakdown
            self.plot_round_breakdown(
                round_metrics,
                ['test_accuracy', 'train_loss'],
                filename='round_breakdown'
            )
        
        logger.info("Generated comprehensive experiment report")
