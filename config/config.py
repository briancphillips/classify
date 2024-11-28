"""Configuration for training CIFAR-100 classifier"""

from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any

@dataclass
class TrainingConfig:
    # Model configuration
    model_name: str = "wrn-28-10"  # Wide ResNet 28-10
    pretrained: bool = True
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
    
    # Advanced training features
    use_amp: bool = True  # Automatic Mixed Precision
    use_swa: bool = True  # Stochastic Weight Averaging
    swa_start: int = 160  # When to start SWA
    swa_lr: float = 0.05  # SWA learning rate
    
    use_mixup: bool = True  # Mixup augmentation
    mixup_alpha: float = 1.0  # Mixup interpolation strength
    mixup_epochs: int = 150  # Number of epochs to use mixup
    
    label_smoothing: float = 0.1  # Label smoothing factor
    
    # Early stopping
    patience: int = 20  # Epochs to wait before early stopping
    min_delta: float = 0.001  # Minimum change to qualify as an improvement
    
    # Checkpointing
    save_freq: int = 10  # Save checkpoint every N epochs
    keep_last_n: int = 3  # Number of checkpoints to keep
    
    # Data augmentation
    random_crop: bool = True
    random_horizontal_flip: bool = True
    normalize: bool = True
    cutout: bool = True
    cutout_length: int = 16
    
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
