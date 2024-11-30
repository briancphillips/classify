"""Default configuration values for the entire project.

This module serves as the single source of truth for all default configuration values.
All other modules should import their defaults from here.
"""

from typing import Dict, Any, List, Tuple

# Training defaults
TRAINING_DEFAULTS = {
    # Model configuration
    "model": "wrn-28-10",
    "depth": 28,
    "widen_factor": 10,
    "dropout_rate": 0.3,
    "num_classes": 100,
    
    # Training parameters
    "batch_size": 128,  # Larger batch size for efficient training
    "epochs": 200,  # Training for full 200 epochs as per WideResNet paper
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0005,  # 5e-4
    
    # Learning rate schedule
    "lr_schedule": [60, 120, 160],
    "lr_factor": 0.2,
    
    # Data augmentation
    "random_crop": True,
    "random_crop_padding": 4,
    "random_horizontal_flip": True,
    "normalize": True,
    "normalize_mean": [0.5071, 0.4867, 0.4408],  # CIFAR-100 means
    "normalize_std": [0.2675, 0.2565, 0.2761],   # CIFAR-100 stds
    
    # Hardware
    "num_workers": 4,
    "pin_memory": True,
}

# Dataset-specific defaults that override the training defaults
DATASET_DEFAULTS = {
    "cifar100": {
        "model": "wrn-28-10",
        "num_classes": 100,
        "epochs": 200,
    },
    "gtsrb": {
        "model": "custom-cnn",
        "num_classes": 43,
        "epochs": 5,
    },
    "imagenette": {
        "model": "resnet50",
        "num_classes": 10,
        "epochs": 5,
    }
}

# Poisoning attack defaults
POISON_DEFAULTS = {
    "poison_ratio": 0.1,  # Percentage of dataset to poison
    "batch_size": 32,  # Smaller batch size for more frequent updates during poisoning
    # PGD attack parameters
    "pgd_eps": 0.3,
    "pgd_alpha": 0.01,
    "pgd_steps": 40,
    # Gradient Ascent attack parameters
    "ga_steps": 50,
    "ga_iterations": 100,
    "ga_lr": 0.1,
}

# Output defaults
OUTPUT_DEFAULTS = {
    "base_dir": "results",
    "save_model": True,
    "save_frequency": 10,
    "consolidated_file": "all_results.csv"
}

# Execution defaults
EXECUTION_DEFAULTS = {
    "max_workers": 1,
    "gpu_ids": [0],
}

def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """Get the configuration for a specific dataset.
    
    This merges the dataset-specific defaults with the training defaults.
    Dataset-specific values override training defaults.
    """
    config = TRAINING_DEFAULTS.copy()
    if dataset_name in DATASET_DEFAULTS:
        config.update(DATASET_DEFAULTS[dataset_name])
    return config

def get_poison_config(poison_type: str = None) -> Dict[str, Any]:
    """Get the configuration for poisoning attacks.
    
    If poison_type is specified, only returns relevant parameters for that attack type.
    """
    config = POISON_DEFAULTS.copy()
    if poison_type == "pgd":
        return {k: v for k, v in config.items() if k.startswith("pgd_") or k == "poison_ratio" or k == "batch_size"}
    elif poison_type == "gradient_ascent":
        return {k: v for k, v in config.items() if k.startswith("ga_") or k == "poison_ratio" or k == "batch_size"}
    return config
