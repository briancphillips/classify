"""Configuration for training CIFAR-100 classifier"""

from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any
from .defaults import TRAINING_DEFAULTS as DEFAULTS

@dataclass
class TrainingConfig:
    # Model configuration
    model_name: str = DEFAULTS["model"]
    num_classes: int = DEFAULTS["num_classes"]
    
    # Training parameters
    batch_size: int = DEFAULTS["batch_size"]
    epochs: int = DEFAULTS["epochs"]
    learning_rate: float = DEFAULTS["learning_rate"]
    momentum: float = DEFAULTS["momentum"]
    weight_decay: float = DEFAULTS["weight_decay"]
    
    # Learning rate schedule
    lr_schedule: List[int] = DEFAULTS["lr_schedule"]
    lr_factor: float = DEFAULTS["lr_factor"]
    
    # Data augmentation
    random_crop: bool = DEFAULTS["random_crop"]
    random_crop_padding: int = DEFAULTS["random_crop_padding"]
    random_horizontal_flip: bool = DEFAULTS["random_horizontal_flip"]
    normalize: bool = DEFAULTS["normalize"]
    normalize_mean: List[float] = DEFAULTS["normalize_mean"]
    normalize_std: List[float] = DEFAULTS["normalize_std"]
    
    # Hardware
    num_workers: int = DEFAULTS["num_workers"]
    pin_memory: bool = DEFAULTS["pin_memory"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

# Default training configuration
default_config = TrainingConfig()
