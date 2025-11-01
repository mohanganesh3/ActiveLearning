"""
CIFAR-10 data loading and management for active learning.
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class ActiveLearningDataset:
    """Manages data for active learning experiments."""
    
    def __init__(self, config):
        """
        Initialize active learning dataset.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.data_dir = config.data.data_dir
        
        # Load full dataset
        self.train_dataset, self.test_dataset = self._load_datasets()
        
        # Initialize active learning splits
        self.total_size = len(self.train_dataset)
        self.labeled_indices = np.array([], dtype=int)
        self.unlabeled_indices = np.arange(self.total_size)
        
        logger.info(f"Initialized CIFAR-10 dataset: {self.total_size} training samples")
    
    def _load_datasets(self) -> Tuple[Dataset, Dataset]:
        """Load CIFAR-10 datasets with augmentation."""
        # Training transforms with augmentation
        if self.config.data.augmentation:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    self.config.data.augmentation_config.normalization_mean,
                    self.config.data.augmentation_config.normalization_std
                ),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    self.config.data.augmentation_config.normalization_mean,
                    self.config.data.augmentation_config.normalization_std
                ),
            ])
        
        # Test transforms (no augmentation)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                self.config.data.augmentation_config.normalization_mean,
                self.config.data.augmentation_config.normalization_std
            ),
        ])
        
        # Download and load datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=train_transform
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=test_transform
        )
        
        return train_dataset, test_dataset
    
    def initialize_labeled_pool(self, n_samples: int, strategy: str = 'random') -> None:
        """
        Initialize the labeled pool with initial samples.
        
        Args:
            n_samples: Number of samples for initial pool
            strategy: Selection strategy ('random', 'balanced')
        """
        if strategy == 'random':
            # Random selection
            selected = np.random.choice(
                self.unlabeled_indices,
                size=n_samples,
                replace=False
            )
        elif strategy == 'balanced':
            # Balanced class selection
            targets = np.array([self.train_dataset[i][1] for i in self.unlabeled_indices])
            selected = self._balanced_sample(self.unlabeled_indices, targets, n_samples)
        else:
            raise ValueError(f"Unknown initialization strategy: {strategy}")
        
        self.add_to_labeled_pool(selected)
        logger.info(f"Initialized labeled pool with {n_samples} samples ({strategy})")
    
    def _balanced_sample(self, indices: np.ndarray, targets: np.ndarray, 
                        n_samples: int) -> np.ndarray:
        """Sample indices to maintain class balance."""
        unique_classes = np.unique(targets)
        samples_per_class = n_samples // len(unique_classes)
        
        selected = []
        for cls in unique_classes:
            cls_indices = indices[targets == cls]
            if len(cls_indices) >= samples_per_class:
                selected.extend(np.random.choice(cls_indices, samples_per_class, replace=False))
            else:
                selected.extend(cls_indices)
        
        # If we need more samples (due to rounding), add random ones
        if len(selected) < n_samples:
            remaining = n_samples - len(selected)
            available = np.setdiff1d(indices, selected)
            selected.extend(np.random.choice(available, remaining, replace=False))
        
        return np.array(selected[:n_samples])
    
    def add_to_labeled_pool(self, indices: np.ndarray) -> None:
        """
        Add samples to labeled pool and remove from unlabeled pool.
        
        Args:
            indices: Indices to add to labeled pool
        """
        # Add to labeled
        self.labeled_indices = np.concatenate([self.labeled_indices, indices])
        
        # Remove from unlabeled
        self.unlabeled_indices = np.setdiff1d(self.unlabeled_indices, indices)
        
        logger.info(f"Labeled: {len(self.labeled_indices)}, Unlabeled: {len(self.unlabeled_indices)}")
    
    def get_labeled_loader(self, batch_size: Optional[int] = None, 
                          shuffle: bool = True) -> DataLoader:
        """Get DataLoader for labeled data."""
        batch_size = batch_size or self.config.data.batch_size
        subset = Subset(self.train_dataset, self.labeled_indices)
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )
    
    def get_unlabeled_loader(self, batch_size: Optional[int] = None,
                            shuffle: bool = False) -> DataLoader:
        """Get DataLoader for unlabeled data."""
        batch_size = batch_size or self.config.data.batch_size
        subset = Subset(self.train_dataset, self.unlabeled_indices)
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )
    
    def get_test_loader(self, batch_size: Optional[int] = None) -> DataLoader:
        """Get DataLoader for test data."""
        batch_size = batch_size or self.config.metrics.test_batch_size
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )
    
    def get_labeled_class_distribution(self) -> np.ndarray:
        """Get class distribution of labeled data."""
        targets = [self.train_dataset[i][1] for i in self.labeled_indices]
        return np.bincount(targets, minlength=self.config.model.num_classes)
