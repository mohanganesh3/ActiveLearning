"""
Abstract base classes for sampling strategies in active learning.
"""
from abc import ABC, abstractmethod
from typing import List, Any
import numpy as np
from torch.utils.data import DataLoader


class BaseSampler(ABC):
    """Abstract base class for all sampling strategies."""
    
    def __init__(self, config: Any):
        """
        Initialize sampler.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    @abstractmethod
    def select_samples(self,
                      unlabeled_indices: np.ndarray,
                      labeled_indices: np.ndarray,
                      model: Any,
                      unlabeled_loader: DataLoader,
                      n_samples: int) -> np.ndarray:
        """
        Select samples from unlabeled pool.
        
        Args:
            unlabeled_indices: Indices of unlabeled data
            labeled_indices: Indices of currently labeled data
            model: Trained model (BaseTrainer instance)
            unlabeled_loader: DataLoader for unlabeled data
            n_samples: Number of samples to select
            
        Returns:
            Array of selected indices from unlabeled pool
        """
        pass
    
    def _validate_selection(self, 
                           selected_indices: np.ndarray,
                           unlabeled_indices: np.ndarray,
                           n_samples: int) -> np.ndarray:
        """
        Validate and ensure selected indices are valid.
        
        Args:
            selected_indices: Selected indices
            unlabeled_indices: Pool of unlabeled indices
            n_samples: Expected number of samples
            
        Returns:
            Validated selection
        """
        # Ensure we have the right number of samples
        if len(selected_indices) > n_samples:
            selected_indices = selected_indices[:n_samples]
        elif len(selected_indices) < n_samples:
            # If not enough samples selected, add random ones
            remaining = n_samples - len(selected_indices)
            available = np.setdiff1d(unlabeled_indices, selected_indices)
            if len(available) > 0:
                additional = np.random.choice(available, 
                                            size=min(remaining, len(available)),
                                            replace=False)
                selected_indices = np.concatenate([selected_indices, additional])
        
        # Ensure all selected indices are from unlabeled pool
        selected_indices = np.array([idx for idx in selected_indices 
                                    if idx in unlabeled_indices])
        
        return selected_indices
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
