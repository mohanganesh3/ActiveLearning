"""
Utilities for ensuring reproducibility across experiments.
"""
import random
import numpy as np
import torch
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True, benchmark: bool = False) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (may be slower)
        benchmark: If True, use cudnn benchmarking (faster but non-deterministic)
    """
    logger.info(f"Setting random seed to {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Using deterministic algorithms (may be slower)")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = benchmark
        if benchmark:
            logger.info("Using cudnn benchmark (faster but non-deterministic)")


def get_reproducibility_info() -> dict:
    """
    Get information about current reproducibility settings.
    
    Returns:
        Dictionary with reproducibility information
    """
    info = {
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['num_gpus'] = torch.cuda.device_count()
        info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    
    return info
