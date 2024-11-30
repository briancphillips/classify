from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import json
import os
from .types import PoisonType
from .experiment_config import ExperimentConfig
from utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ['PoisonConfig', 'PoisonResult', 'ExperimentConfig']


@dataclass
class PoisonConfig:
    """Configuration for poisoning attacks"""

    poison_type: PoisonType
    poison_ratio: float  # Percentage of dataset to poison (0.0 to 1.0)
    batch_size: Optional[int] = 32  # Batch size for poisoning attacks
    # PGD attack parameters
    pgd_eps: Optional[float] = 0.3  # Epsilon for PGD attack
    pgd_alpha: Optional[float] = 0.01  # Step size for PGD attack
    pgd_steps: Optional[int] = 40  # Number of steps for PGD attack
    # Gradient Ascent attack parameters
    ga_steps: Optional[int] = 50  # Number of gradient steps per iteration
    ga_iterations: Optional[int] = 100  # Number of outer iterations
    ga_lr: Optional[float] = 0.1  # Learning rate for gradient ascent
    # Attack targeting parameters
    source_class: Optional[int] = None  # Source class for targeted attacks
    target_class: Optional[int] = None  # Target class for targeted attacks
    random_seed: Optional[int] = 42  # Random seed for reproducibility

    def __post_init__(self):
        # Convert string to PoisonType enum if needed
        if isinstance(self.poison_type, str):
            for poison_type in PoisonType:
                if poison_type.value == self.poison_type:
                    self.poison_type = poison_type
                    break
            else:
                raise ValueError(f"Invalid poison type: {self.poison_type}")


@dataclass
class PoisonResult:
    """Results from a poisoning experiment."""

    config: PoisonConfig
    dataset_name: str
    poisoned_indices: List[int] = None
    poison_success_rate: float = 0.0
    original_accuracy: float = 0.0
    poisoned_accuracy: float = 0.0
    timestamp: str = None

    def __post_init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def to_dict(self) -> Dict:
        """Convert result to dictionary for serialization."""
        if isinstance(self.config, ExperimentConfig):
            poison_config = self.config.poison
        else:
            poison_config = self.config

        # Convert PoisonType enum to string if needed
        poison_type_str = poison_config.poison_type.value if isinstance(poison_config.poison_type, PoisonType) else poison_config.poison_type

        return {
            "dataset_name": self.dataset_name,
            "config": {
                "poison_type": poison_type_str,
                "poison_ratio": poison_config.poison_ratio,
                "batch_size": poison_config.batch_size,
                "pgd_eps": poison_config.pgd_eps,
                "pgd_alpha": poison_config.pgd_alpha,
                "pgd_steps": poison_config.pgd_steps,
                "random_seed": poison_config.random_seed,
            },
            "poisoned_indices": self.poisoned_indices,
            "poison_success_rate": self.poison_success_rate,
            "original_accuracy": self.original_accuracy,
            "poisoned_accuracy": self.poisoned_accuracy,
            "timestamp": self.timestamp,
        }

    def save(self, output_dir: str):
        """Save results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        filename = f"poison_results_{self.timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        logger.info(f"Results saved to {filepath}")
