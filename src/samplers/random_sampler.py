"""
Random sampling strategy (baseline).
"""
import numpy as np
from typing import Any
from torch.utils.data import DataLoader
from .base import BaseSampler


class RandomSampler(BaseSampler):
    """Random sampling strategy for active learning (baseline)."""
    
    def __init__(self, config: Any):
        """
        Initialize random sampler.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
    
    def select_samples(self,
                      unlabeled_indices: np.ndarray,
                      labeled_indices: np.ndarray,
                      model: Any,
                      unlabeled_loader: DataLoader,
                      n_samples: int) -> np.ndarray:
        """
        Randomly select samples from unlabeled pool.
        
        Args:
            unlabeled_indices: Indices of unlabeled data
            labeled_indices: Indices of currently labeled data
            model: Trained model (not used for random sampling)
            unlabeled_loader: DataLoader for unlabeled data (not used)
            n_samples: Number of samples to select
            
        Returns:
            Array of randomly selected indices
        """
        n_samples = min(n_samples, len(unlabeled_indices))
        selected = np.random.choice(unlabeled_indices, 
                                   size=n_samples, 
                                   replace=False)
        
        return self._validate_selection(selected, unlabeled_indices, n_samples)
    
    def __repr__(self) -> str:
        return "RandomSampler()"
