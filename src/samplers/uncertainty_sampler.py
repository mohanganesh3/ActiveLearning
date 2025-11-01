"""
Uncertainty-based sampling strategies.
"""
import numpy as np
from typing import Any
import torch
from torch.utils.data import DataLoader
from .base import BaseSampler
import logging

logger = logging.getLogger(__name__)


class UncertaintySampler(BaseSampler):
    """Uncertainty-based sampling for active learning."""
    
    def __init__(self, config: Any, method: str = 'entropy'):
        """
        Initialize uncertainty sampler.
        
        Args:
            config: Configuration object
            method: Uncertainty measure ('entropy', 'margin', 'least_confidence')
        """
        super().__init__(config)
        self.method = method
        logger.info(f"Initialized UncertaintySampler with method: {method}")
    
    def select_samples(self,
                      unlabeled_indices: np.ndarray,
                      labeled_indices: np.ndarray,
                      model: Any,
                      unlabeled_loader: DataLoader,
                      n_samples: int) -> np.ndarray:
        """
        Select samples with highest uncertainty.
        
        Args:
            unlabeled_indices: Indices of unlabeled data
            labeled_indices: Indices of currently labeled data
            model: Trained model (BaseTrainer instance)
            unlabeled_loader: DataLoader for unlabeled data
            n_samples: Number of samples to select
            
        Returns:
            Array of selected indices with highest uncertainty
        """
        # Get predictions from model
        _, probs = model.get_predictions(unlabeled_loader)
        
        # Compute uncertainty scores
        if self.method == 'entropy':
            uncertainty_scores = self._entropy(probs)
        elif self.method == 'margin':
            uncertainty_scores = self._margin(probs)
        elif self.method == 'least_confidence':
            uncertainty_scores = self._least_confidence(probs)
        else:
            raise ValueError(f"Unknown uncertainty method: {self.method}")
        
        # Select top uncertain samples
        n_samples = min(n_samples, len(unlabeled_indices))
        top_uncertain_idx = np.argsort(uncertainty_scores)[-n_samples:]
        selected = unlabeled_indices[top_uncertain_idx]
        
        return self._validate_selection(selected, unlabeled_indices, n_samples)
    
    @staticmethod
    def _entropy(probs: np.ndarray) -> np.ndarray:
        """
        Compute entropy of predictions.
        
        Args:
            probs: Prediction probabilities [N x num_classes]
            
        Returns:
            Entropy scores [N]
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        probs = np.clip(probs, eps, 1 - eps)
        entropy = -np.sum(probs * np.log(probs), axis=1)
        return entropy
    
    @staticmethod
    def _margin(probs: np.ndarray) -> np.ndarray:
        """
        Compute margin (difference between top two probabilities).
        Smaller margin = higher uncertainty.
        
        Args:
            probs: Prediction probabilities [N x num_classes]
            
        Returns:
            Negative margin scores [N] (so higher = more uncertain)
        """
        # Sort probabilities in descending order
        sorted_probs = np.sort(probs, axis=1)[:, ::-1]
        # Margin is difference between top 2
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        # Return negative so higher values = more uncertain
        return -margin
    
    @staticmethod
    def _least_confidence(probs: np.ndarray) -> np.ndarray:
        """
        Compute least confidence (1 - max probability).
        
        Args:
            probs: Prediction probabilities [N x num_classes]
            
        Returns:
            Least confidence scores [N]
        """
        max_probs = np.max(probs, axis=1)
        return 1 - max_probs
    
    def __repr__(self) -> str:
        return f"UncertaintySampler(method={self.method})"
