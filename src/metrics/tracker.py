"""
Metrics tracking and logging system for active learning experiments.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track and log metrics throughout active learning experiments."""
    
    def __init__(self, experiment_dir: Path, experiment_name: str):
        """
        Initialize metrics tracker.
        
        Args:
            experiment_dir: Directory to save metrics
            experiment_name: Name of the experiment
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_name = experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics: List[Dict[str, Any]] = []
        self.round_metrics: Dict[int, Dict[str, Any]] = {}
        self.start_time = time.time()
        self.round_start_times: Dict[int, float] = {}
        
        logger.info(f"Initialized metrics tracker for experiment: {experiment_name}")
        logger.info(f"Saving metrics to: {self.experiment_dir}")
    
    def start_round(self, round_num: int) -> None:
        """
        Mark the start of a training round.
        
        Args:
            round_num: Round number
        """
        self.round_start_times[round_num] = time.time()
        logger.info(f"Started Round {round_num}")
    
    def log_round_metrics(self, round_num: int, metrics: Dict[str, Any]) -> None:
        """
        Log metrics for a specific round.
        
        Args:
            round_num: Round number
            metrics: Dictionary of metrics to log
        """
        if round_num in self.round_start_times:
            round_time = time.time() - self.round_start_times[round_num]
            metrics['round_time_seconds'] = round_time
        
        metrics['round'] = round_num
        metrics['timestamp'] = datetime.now().isoformat()
        
        self.round_metrics[round_num] = metrics
        self.metrics.append(metrics)
        
        # Log key metrics
        logger.info(f"Round {round_num} metrics:")
        for key, value in metrics.items():
            if key not in ['round', 'timestamp', 'selected_indices']:
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")
    
    def log_epoch_metrics(self, round_num: int, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Log metrics for a specific epoch within a round.
        
        Args:
            round_num: Round number
            epoch: Epoch number
            metrics: Dictionary of metrics to log
        """
        entry = {
            'round': round_num,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.metrics.append(entry)
    
    def get_round_metric(self, round_num: int, metric_name: str) -> Optional[Any]:
        """
        Get a specific metric for a round.
        
        Args:
            round_num: Round number
            metric_name: Name of the metric
            
        Returns:
            Metric value or None if not found
        """
        if round_num in self.round_metrics:
            return self.round_metrics[round_num].get(metric_name)
        return None
    
    def get_all_round_metrics(self, metric_name: str) -> List[Any]:
        """
        Get a specific metric across all rounds.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of metric values across rounds
        """
        values = []
        for round_num in sorted(self.round_metrics.keys()):
            if metric_name in self.round_metrics[round_num]:
                values.append(self.round_metrics[round_num][metric_name])
        return values
    
    def save(self) -> None:
        """Save all metrics to disk."""
        # Save detailed metrics
        metrics_file = self.experiment_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        # Save round summary
        round_summary_file = self.experiment_dir / "round_summary.json"
        with open(round_summary_file, 'w') as f:
            json.dump(self.round_metrics, f, indent=2, default=str)
        
        # Save experiment summary
        total_time = time.time() - self.start_time
        summary = {
            'experiment_name': self.experiment_name,
            'total_time_seconds': total_time,
            'total_rounds': len(self.round_metrics),
            'final_metrics': self.round_metrics.get(max(self.round_metrics.keys())) if self.round_metrics else {},
        }
        
        summary_file = self.experiment_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved metrics to {self.experiment_dir}")
    
    def compute_summary_statistics(self) -> Dict[str, Any]:
        """
        Compute summary statistics across all rounds.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.round_metrics:
            return {}
        
        summary = {}
        
        # Get all numeric metrics
        numeric_metrics = set()
        for metrics in self.round_metrics.values():
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and key not in ['round', 'epoch']:
                    numeric_metrics.add(key)
        
        # Compute statistics for each metric
        for metric in numeric_metrics:
            values = self.get_all_round_metrics(metric)
            if values:
                summary[f"{metric}_mean"] = np.mean(values)
                summary[f"{metric}_std"] = np.std(values)
                summary[f"{metric}_min"] = np.min(values)
                summary[f"{metric}_max"] = np.max(values)
                summary[f"{metric}_final"] = values[-1]
        
        return summary
    
    def __repr__(self) -> str:
        return f"MetricsTracker(experiment={self.experiment_name}, rounds={len(self.round_metrics)})"
