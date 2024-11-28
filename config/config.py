"""Configuration for training CIFAR-100 classifier"""

from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any

@dataclass
class TrainingConfig:
    # Model configuration
    model_name: str = "wrn-28-10"  # Wide ResNet 28-10
    num_classes: int = 100
    
    # Training parameters
    batch_size: int = 128
    epochs: int = 200
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    
    # Learning rate schedule
    lr_schedule: List[int] = (60, 120, 160)  # Epochs to reduce LR
    lr_factor: float = 0.2  # Factor to reduce LR by
    
    # Data augmentation
    random_crop: bool = True
    random_crop_padding: int = 4
    random_horizontal_flip: bool = True
    normalize: bool = True
    normalize_mean: List[float] = (0.5071, 0.4867, 0.4408)  # CIFAR-100 means
    normalize_std: List[float] = (0.2675, 0.2565, 0.2761)   # CIFAR-100 stds
    
    # Hardware
    num_workers: int = 4
    pin_memory: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

# Default training configuration
default_config = TrainingConfig()
