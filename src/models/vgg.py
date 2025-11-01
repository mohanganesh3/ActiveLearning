"""
VGG model for CIFAR-10.
Adapted from PyTorch examples.
"""
import torch
import torch.nn as nn


class VGG(nn.Module):
    """VGG network for CIFAR-10."""
    
    def __init__(self, vgg_name: str = 'VGG16', num_classes: int = 10, 
                 dropout: float = 0.5, return_features: bool = True):
        """
        Initialize VGG model.
        
        Args:
            vgg_name: VGG variant ('VGG11', 'VGG13', 'VGG16', 'VGG19')
            num_classes: Number of output classes
            dropout: Dropout probability
            return_features: If True, return (logits, features)
        """
        super(VGG, self).__init__()
        self.return_features = return_features
        
        cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        
        self.features = self._make_layers(cfg[vgg_name])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        """Forward pass."""
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.classifier(feat)
        
        if self.return_features:
            return out, feat
        return out
    
    def _make_layers(self, cfg):
        """Create VGG layers from config."""
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def create_model(config) -> nn.Module:
    """
    Create model from configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        PyTorch model
    """
    if config.model.architecture.upper().startswith('VGG'):
        model = VGG(
            vgg_name=config.model.architecture.upper(),
            num_classes=config.model.num_classes,
            dropout=config.model.dropout,
            return_features=True
        )
    else:
        raise ValueError(f"Unknown architecture: {config.model.architecture}")
    
    return model
