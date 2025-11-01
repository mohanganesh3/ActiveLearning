"""
CoreSet / K-Center Greedy sampling for diversity-based selection.
"""
import numpy as np
from typing import Any
from torch.utils.data import DataLoader
from .base import BaseSampler
import logging

logger = logging.getLogger(__name__)


class CoreSetSampler(BaseSampler):
    """
    CoreSet sampling using k-center greedy algorithm.
    Selects samples that maximize diversity in feature space.
    """
    
    def __init__(self, config: Any, distance_metric: str = 'euclidean'):
        """
        Initialize coreset sampler.
        
        Args:
            config: Configuration object
            distance_metric: Distance metric to use ('euclidean', 'cosine')
        """
        super().__init__(config)
        self.distance_metric = distance_metric
        logger.info(f"Initialized CoreSetSampler with metric: {distance_metric}")
    
    def select_samples(self,
                      unlabeled_indices: np.ndarray,
                      labeled_indices: np.ndarray,
                      model: Any,
                      unlabeled_loader: DataLoader,
                      n_samples: int) -> np.ndarray:
        """
        Select samples using k-center greedy algorithm.
        
        Args:
            unlabeled_indices: Indices of unlabeled data
            labeled_indices: Indices of currently labeled data
            model: Trained model (BaseTrainer instance)
            unlabeled_loader: DataLoader for unlabeled data
            n_samples: Number of samples to select
            
        Returns:
            Array of selected diverse samples
        """
        # Extract features for unlabeled data
        unlabeled_features = model.get_features(unlabeled_loader)
        
        # If we have labeled data, get their features too
        if len(labeled_indices) > 0:
            # Create a dataloader for labeled data
            # For now, we'll approximate by assuming labeled features are from the model
            # In a full implementation, we'd need access to labeled data
            labeled_features = None
        else:
            labeled_features = None
        
        # Run k-center greedy selection
        selected_indices = self._greedy_k_center(
            unlabeled_features,
            labeled_features,
            n_samples
        )
        
        # Map back to original indices
        selected = unlabeled_indices[selected_indices]
        
        return self._validate_selection(selected, unlabeled_indices, n_samples)
    
    def _greedy_k_center(self,
                        unlabeled_features: np.ndarray,
                        labeled_features: np.ndarray,
                        n_samples: int) -> np.ndarray:
        """
        Greedy k-center algorithm for selecting diverse samples.
        
        Args:
            unlabeled_features: Features of unlabeled samples [N x D]
            labeled_features: Features of labeled samples [M x D] or None
            n_samples: Number of samples to select
            
        Returns:
            Indices of selected samples from unlabeled_features
        """
        n_unlabeled = unlabeled_features.shape[0]
        n_samples = min(n_samples, n_unlabeled)
        
        # Initialize with labeled features if available
        if labeled_features is not None and len(labeled_features) > 0:
            # Start with labeled samples as centers
            centers = labeled_features
        else:
            # Start with a random sample
            first_idx = np.random.randint(n_unlabeled)
            centers = unlabeled_features[first_idx:first_idx+1]
        
        selected_indices = []
        
        # Compute initial distances
        min_distances = self._compute_min_distances(unlabeled_features, centers)
        
        # Iteratively select samples
        for _ in range(n_samples):
            # Find the sample farthest from current centers
            farthest_idx = np.argmax(min_distances)
            
            # Add to selected
            selected_indices.append(farthest_idx)
            
            # Update centers
            new_center = unlabeled_features[farthest_idx:farthest_idx+1]
            centers = np.vstack([centers, new_center])
            
            # Update minimum distances
            distances_to_new = self._compute_distances(
                unlabeled_features, 
                new_center
            ).flatten()
            min_distances = np.minimum(min_distances, distances_to_new)
            
            # Set distance of selected sample to -inf to avoid reselection
            min_distances[farthest_idx] = -np.inf
        
        return np.array(selected_indices)
    
    def _compute_min_distances(self,
                              points: np.ndarray,
                              centers: np.ndarray) -> np.ndarray:
        """
        Compute minimum distance from each point to any center.
        
        Args:
            points: Points to compute distances for [N x D]
            centers: Center points [M x D]
            
        Returns:
            Minimum distances [N]
        """
        if self.distance_metric == 'euclidean':
            # Compute pairwise euclidean distances
            distances = np.sqrt(((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
        elif self.distance_metric == 'cosine':
            # Normalize vectors
            points_norm = points / (np.linalg.norm(points, axis=1, keepdims=True) + 1e-10)
            centers_norm = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-10)
            # Cosine distance = 1 - cosine similarity
            similarities = np.dot(points_norm, centers_norm.T)
            distances = 1 - similarities
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        # Return minimum distance to any center
        return np.min(distances, axis=1)
    
    def _compute_distances(self,
                          points: np.ndarray,
                          center: np.ndarray) -> np.ndarray:
        """
        Compute distances from points to a single center.
        
        Args:
            points: Points [N x D]
            center: Single center point [1 x D]
            
        Returns:
            Distances [N x 1]
        """
        if self.distance_metric == 'euclidean':
            distances = np.sqrt(((points - center) ** 2).sum(axis=1, keepdims=True))
        elif self.distance_metric == 'cosine':
            points_norm = points / (np.linalg.norm(points, axis=1, keepdims=True) + 1e-10)
            center_norm = center / (np.linalg.norm(center) + 1e-10)
            similarities = np.dot(points_norm, center_norm.T)
            distances = 1 - similarities
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distances
    
    def __repr__(self) -> str:
        return f"CoreSetSampler(metric={self.distance_metric})"
