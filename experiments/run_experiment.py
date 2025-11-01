"""
Main experiment runner for active learning.
"""
import torch
import logging
from pathlib import Path
from typing import Optional
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, setup_logging, set_seed, get_reproducibility_info
from src.data import ActiveLearningDataset
from src.models import create_model
from src.trainers import StandardTrainer
from src.samplers import RandomSampler, UncertaintySampler, CoreSetSampler
from src.metrics import MetricsTracker
from src.visualization import Visualizer

logger = logging.getLogger(__name__)


class ActiveLearningExperiment:
    """Main experiment class for active learning."""
    
    def __init__(self, config_path: str):
        """
        Initialize experiment.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logging
        setup_logging(self.config)
        logger.info("="*80)
        logger.info("Starting Active Learning Experiment")
        logger.info("="*80)
        
        # Set random seed for reproducibility
        set_seed(
            self.config.experiment.seed,
            deterministic=self.config.reproducibility.deterministic,
            benchmark=self.config.reproducibility.benchmark
        )
        
        # Log reproducibility info
        repro_info = get_reproducibility_info()
        logger.info(f"Reproducibility Info: {repro_info}")
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir = Path(self.config.experiment.output_dir) / self.config.experiment.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
        
        # Initialize components
        self.data = ActiveLearningDataset(self.config)
        self.model = create_model(self.config)
        self.trainer = StandardTrainer(self.model, self.device, self.config)
        self.sampler = self._create_sampler()
        self.metrics = MetricsTracker(self.output_dir, self.config.experiment.name)
        self.visualizer = Visualizer(
            self.output_dir / "plots",
            dpi=self.config.visualization.get('dpi', 300),
            format=self.config.visualization.get('plot_format', 'png')
        )
        
        logger.info(f"Initialized experiment: {self.config.experiment.name}")
        logger.info(f"Sampling strategy: {self.config.active_learning.strategy}")
    
    def _create_sampler(self):
        """Create sampling strategy from config."""
        strategy = self.config.active_learning.strategy.lower()
        
        if strategy == 'random':
            return RandomSampler(self.config)
        elif strategy == 'uncertainty':
            method = self.config.active_learning.uncertainty_config.method
            return UncertaintySampler(self.config, method=method)
        elif strategy == 'coreset':
            metric = self.config.active_learning.diversity_config.distance_metric
            return CoreSetSampler(self.config, distance_metric=metric)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    def run(self) -> None:
        """Run the active learning experiment."""
        logger.info("="*80)
        logger.info("Starting Active Learning Loop")
        logger.info("="*80)
        
        # Initialize labeled pool
        self.data.initialize_labeled_pool(
            self.config.active_learning.initial_samples,
            strategy='random'
        )
        
        # Active learning loop
        for round_num in range(self.config.active_learning.num_rounds):
            logger.info(f"\n{'='*80}")
            logger.info(f"Round {round_num + 1}/{self.config.active_learning.num_rounds}")
            logger.info(f"{'='*80}")
            
            self.metrics.start_round(round_num)
            
            # Train model on current labeled data
            train_metrics = self._train_round(round_num)
            
            # Evaluate on test set
            test_metrics = self._evaluate()
            
            # Get class distribution
            class_dist = self.data.get_labeled_class_distribution()
            
            # Combine metrics
            round_metrics = {
                **train_metrics,
                **test_metrics,
                'num_labeled': len(self.data.labeled_indices),
                'num_unlabeled': len(self.data.unlabeled_indices),
                'class_distribution': class_dist.tolist(),
            }
            
            # Log round metrics
            self.metrics.log_round_metrics(round_num, round_metrics)
            
            # Save checkpoint
            if self.config.experiment.save_checkpoints and \
               round_num % self.config.experiment.checkpoint_frequency == 0:
                checkpoint_path = self.output_dir / f"checkpoint_round_{round_num}.pth"
                self.trainer.save_checkpoint(
                    str(checkpoint_path),
                    round=round_num,
                    metrics=round_metrics
                )
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Sample new data (if not last round)
            if round_num < self.config.active_learning.num_rounds - 1:
                new_indices = self._sample_new_data(round_num)
                round_metrics['selected_indices'] = new_indices.tolist()
        
        # Save final results
        self._save_results()
        
        logger.info("="*80)
        logger.info("Experiment completed successfully!")
        logger.info("="*80)
    
    def _train_round(self, round_num: int) -> dict:
        """Train model for one round."""
        logger.info(f"Training with {len(self.data.labeled_indices)} labeled samples...")
        
        train_loader = self.data.get_labeled_loader()
        epochs = self.config.training.epochs_per_round
        
        best_train_acc = 0.0
        final_metrics = {}
        
        for epoch in range(epochs):
            metrics = self.trainer.train_epoch(train_loader, epoch)
            
            if metrics['train_accuracy'] > best_train_acc:
                best_train_acc = metrics['train_accuracy']
            
            # Log epoch metrics
            if epoch % self.config.metrics.eval_frequency == 0:
                logger.info(
                    f"  Epoch {epoch}/{epochs}: "
                    f"Loss={metrics['train_loss']:.4f}, "
                    f"Acc={metrics['train_accuracy']:.4f}, "
                    f"LR={metrics['learning_rate']:.6f}"
                )
            
            final_metrics = metrics
        
        return final_metrics
    
    def _evaluate(self) -> dict:
        """Evaluate model on test set."""
        test_loader = self.data.get_test_loader()
        metrics = self.trainer.evaluate(test_loader)
        
        logger.info(
            f"Test Results: "
            f"Loss={metrics['test_loss']:.4f}, "
            f"Acc={metrics['test_accuracy']:.4f}"
        )
        
        return metrics
    
    def _sample_new_data(self, round_num: int) -> torch.Tensor:
        """Sample new data points to label."""
        logger.info("Sampling new data points...")
        
        n_samples = self.config.active_learning.samples_per_round
        unlabeled_loader = self.data.get_unlabeled_loader()
        
        selected_indices = self.sampler.select_samples(
            unlabeled_indices=self.data.unlabeled_indices,
            labeled_indices=self.data.labeled_indices,
            model=self.trainer,
            unlabeled_loader=unlabeled_loader,
            n_samples=n_samples
        )
        
        # Add to labeled pool
        self.data.add_to_labeled_pool(selected_indices)
        
        logger.info(f"Selected {len(selected_indices)} new samples")
        
        return selected_indices
    
    def _save_results(self) -> None:
        """Save experiment results and generate visualizations."""
        logger.info("Saving results and generating visualizations...")
        
        # Save metrics
        self.metrics.save()
        
        # Generate summary statistics
        summary = self.metrics.compute_summary_statistics()
        logger.info(f"Summary statistics: {summary}")
        
        # Generate visualizations
        test_accs = self.metrics.get_all_round_metrics('test_accuracy')
        if test_accs:
            self.visualizer.plot_learning_curves(
                {self.config.experiment.name: test_accs},
                metric_name='Test Accuracy',
                filename='learning_curve'
            )
            
            train_losses = self.metrics.get_all_round_metrics('train_loss')
            if train_losses:
                self.visualizer.plot_learning_curves(
                    {self.config.experiment.name: train_losses},
                    metric_name='Training Loss',
                    filename='loss_curve'
                )
        
        logger.info(f"Results saved to {self.output_dir}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Active Learning Experiment')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    
    # Run experiment
    experiment = ActiveLearningExperiment(args.config)
    experiment.run()


if __name__ == '__main__':
    main()
