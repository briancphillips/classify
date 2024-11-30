"""Configuration management for experiments.

This module provides a clean interface for managing experiment configurations,
with support for defaults and easy overrides.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml
from copy import deepcopy

@dataclass
class ModelConfig:
    """Model configuration"""
    name: str = "wrn-28-10"
    depth: int = 28
    widen_factor: int = 10
    dropout_rate: float = 0.3
    num_classes: int = 100
    pretrained: bool = False

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 128
    epochs: int = 200
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0005
    lr_schedule: List[int] = field(default_factory=lambda: [60, 120, 160])
    lr_factor: float = 0.2
    num_workers: int = 4
    pin_memory: bool = True

@dataclass
class DataConfig:
    """Data augmentation and preprocessing configuration"""
    random_crop: bool = True
    random_crop_padding: int = 4
    random_horizontal_flip: bool = True
    normalize: bool = True
    normalize_mean: List[float] = field(default_factory=lambda: [0.5071, 0.4867, 0.4408])
    normalize_std: List[float] = field(default_factory=lambda: [0.2675, 0.2565, 0.2761])
    subset_size: Optional[int] = None

@dataclass
class CheckpointConfig:
    """Checkpoint configuration"""
    save_dir: str = "checkpoints"
    save_freq: int = 10
    save_best: bool = True
    resume: bool = True

@dataclass
class PoisonConfig:
    """Poisoning attack configuration"""
    poison_type: str = "pgd"  # One of: pgd, gradient_ascent, label_flip
    poison_ratio: float = 0.1  # Percentage of dataset to poison
    batch_size: int = 32  # Batch size for poisoning
    # PGD specific parameters
    pgd_eps: float = 0.3
    pgd_alpha: float = 0.01
    pgd_steps: int = 40
    # Gradient Ascent specific parameters
    ga_steps: int = 50
    ga_iterations: int = 100
    ga_lr: float = 0.1
    # Label flip specific parameters
    source_class: Optional[int] = None
    target_class: Optional[int] = None
    random_seed: int = 42

@dataclass
class ExperimentGroupConfig:
    """Configuration for a group of experiments"""
    description: str
    experiments: List[Dict[str, Any]]

@dataclass
class ExecutionConfig:
    """Execution environment configuration"""
    max_workers: int = 1
    gpu_ids: List[int] = field(default_factory=lambda: [0])

@dataclass
class OutputConfig:
    """Output and logging configuration"""
    base_dir: str = "results"
    save_models: bool = True
    consolidated_file: str = "all_results.csv"
    save_individual_results: bool = True

@dataclass
class ExperimentConfig:
    """Main experiment configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    poison: PoisonConfig = field(default_factory=PoisonConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    experiment_groups: Dict[str, ExperimentGroupConfig] = field(default_factory=dict)
    dataset_name: str = "cifar100"
    experiment_name: str = "default"
    seed: int = 42

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        def _recursive_update(obj, updates):
            for key, value in updates.items():
                if hasattr(obj, key):
                    if isinstance(value, dict) and hasattr(getattr(obj, key), '__dataclass_fields__'):
                        _recursive_update(getattr(obj, key), value)
                    else:
                        setattr(obj, key, value)
        _recursive_update(self, updates)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from dictionary."""
        config = cls()
        config.update(config_dict)
        return config

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def save_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

# Dataset-specific configurations
DATASET_CONFIGS = {
    "cifar100": {
        "model": {
            "name": "wrn-28-10",
            "depth": 28,
            "widen_factor": 10,
            "num_classes": 100
        },
        "training": {"epochs": 200},
        "data": {"subset_size": 50000}
    },
    "gtsrb": {
        "model": {
            "name": "custom-cnn",  # GTSRB uses a custom CNN architecture
            "num_classes": 43
        },
        "training": {"epochs": 100},
        "data": {"subset_size": 39209}
    },
    "imagenette": {
        "model": {
            "name": "resnet50",  # ImageNette uses ResNet50
            "pretrained": True,
            "num_classes": 10
        },
        "training": {"epochs": 100},
        "data": {"subset_size": 9469}
    }
}

def create_config(dataset_name: str = "cifar100", **overrides) -> ExperimentConfig:
    """Create a configuration with dataset-specific defaults and optional overrides.
    
    Args:
        dataset_name: Name of the dataset to use
        **overrides: Additional configuration overrides
        
    Returns:
        ExperimentConfig: The complete configuration
    """
    config = ExperimentConfig(dataset_name=dataset_name)
    
    # Apply dataset-specific configuration
    if dataset_name in DATASET_CONFIGS:
        config.update(DATASET_CONFIGS[dataset_name])
    
    # Apply any additional overrides
    if overrides:
        config.update(overrides)
        
    return config
