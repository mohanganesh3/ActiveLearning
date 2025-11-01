"""
Abstract base classes for trainers in active learning framework.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""
    
    def __init__(self, model: nn.Module, device: torch.device, config: Any):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to use for training
            config: Configuration object
        """
        self.model = model
        self.device = device
        self.config = config
        self.model.to(device)
    
    @abstractmethod
    def train_epoch(self, data_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            data_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on given data.
        
        Args:
            data_loader: DataLoader for evaluation data
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    @abstractmethod
    def get_features(self, data_loader: DataLoader) -> np.ndarray:
        """
        Extract feature representations from model.
        
        Args:
            data_loader: DataLoader for data to featurize
            
        Returns:
            NumPy array of features [N x feature_dim]
        """
        pass
    
    def get_predictions(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get model predictions and probabilities.
        
        Args:
            data_loader: DataLoader for data to predict
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # In case model returns (logits, features)
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
        
        return np.concatenate(all_preds), np.concatenate(all_probs)
    
    def save_checkpoint(self, filepath: str, **kwargs) -> None:
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            **kwargs: Additional items to save in checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            **kwargs
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Dictionary with checkpoint contents
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint
