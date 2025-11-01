"""
Utility functions for loading and managing configurations.
"""
import yaml
from pathlib import Path
from typing import Dict, Any
import logging


class Config:
    """Configuration management class."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
        """
        self._config = config_dict
        
    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to config values."""
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to config values."""
        return self._config[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default."""
        return self._config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self._config.copy()
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self._config.update(updates)
    
    def __repr__(self) -> str:
        return f"Config({self._config})"


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object with loaded parameters
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)


def save_config(config: Config, save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Config object to save
        save_path: Path where to save the configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)


def merge_configs(base_config: Config, override_config: Dict[str, Any]) -> Config:
    """
    Merge two configurations, with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Configuration values to override
        
    Returns:
        New merged Config object
    """
    merged = base_config.to_dict()
    
    def deep_update(base: Dict, override: Dict) -> Dict:
        """Recursively update nested dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = deep_update(base[key], value)
            else:
                base[key] = value
        return base
    
    merged = deep_update(merged, override_config)
    return Config(merged)


def setup_logging(config: Config) -> logging.Logger:
    """
    Setup logging based on configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Configured logger
    """
    log_level = getattr(logging, config.logging.log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger('ActiveLearning')
    return logger
