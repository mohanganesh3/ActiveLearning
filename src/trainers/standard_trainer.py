"""
Standard trainer implementation for active learning.
"""
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging

from .base import BaseTrainer

logger = logging.getLogger(__name__)


class StandardTrainer(BaseTrainer):
    """Standard supervised trainer for active learning."""
    
    def __init__(self, model: nn.Module, device: torch.device, config: Any):
        """
        Initialize standard trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to use for training
            config: Configuration object
        """
        super().__init__(model, device, config)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Initialized StandardTrainer with {config.training.optimizer} optimizer")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from config."""
        opt_config = self.config.training
        
        if opt_config.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=opt_config.learning_rate,
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=self.config.model.weight_decay,
                nesterov=opt_config.get('nesterov', False)
            )
        elif opt_config.optimizer.lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config.learning_rate,
                weight_decay=self.config.model.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config.optimizer}")
        
        return optimizer
    
    def _create_scheduler(self) -> Any:
        """Create learning rate scheduler from config."""
        opt_config = self.config.training
        
        if opt_config.get('lr_scheduler') == 'multi_step':
            scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=opt_config.get('lr_milestones', [60, 120, 160]),
                gamma=opt_config.get('lr_gamma', 0.2)
            )
        elif opt_config.get('lr_scheduler') == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=opt_config.get('max_epochs', 200)
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, data_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            data_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Handle models that return (logits, features)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
        
        metrics = {
            'train_loss': total_loss / len(data_loader),
            'train_accuracy': correct / total,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return metrics
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on given data.
        
        Args:
            data_loader: DataLoader for evaluation data
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                
                # Handle models that return (logits, features)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        metrics = {
            'test_loss': total_loss / len(data_loader),
            'test_accuracy': correct / total
        }
        
        return metrics
    
    def get_features(self, data_loader: DataLoader) -> np.ndarray:
        """
        Extract feature representations from model.
        
        Args:
            data_loader: DataLoader for data to featurize
            
        Returns:
            NumPy array of features [N x feature_dim]
        """
        self.model.eval()
        all_features = []
        
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                
                # Extract features
                if isinstance(outputs, tuple):
                    # Model returns (logits, features)
                    features = outputs[1]
                else:
                    # Try to get features from a specific layer
                    # This assumes the model has a 'features' attribute or similar
                    # For VGG-like models, we need to access before the classifier
                    if hasattr(self.model, 'features'):
                        features = self.model.features(inputs)
                        features = features.view(features.size(0), -1)
                    else:
                        # Fallback: use the output before softmax as "features"
                        features = outputs
                
                all_features.append(features.cpu().numpy())
        
        return np.concatenate(all_features, axis=0)
