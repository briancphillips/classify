from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import json
import os
from .types import PoisonType
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PoisonConfig:
    """Configuration for poisoning attacks"""

    poison_type: PoisonType
    poison_ratio: float  # Percentage of dataset to poison (0.0 to 1.0)
    # PGD specific parameters
    pgd_eps: Optional[float] = 0.3  # Epsilon for PGD attack
    pgd_alpha: Optional[float] = 0.01  # Step size for PGD
    pgd_steps: Optional[int] = 40  # Number of PGD steps
    # GA specific parameters
    ga_pop_size: Optional[int] = 50  # Population size for GA
    ga_generations: Optional[int] = 100  # Number of generations
    ga_mutation_rate: Optional[float] = 0.1
    # Label flipping specific parameters
    source_class: Optional[int] = None  # Source class for source->target flipping
    target_class: Optional[int] = None  # Target class for targeted flipping
    random_seed: Optional[int] = 42  # Random seed for reproducibility


@dataclass
class PoisonResult:
    """Results from a poisoning experiment."""

    config: PoisonConfig
    poisoned_indices: List[int] = None
    poison_success_rate: float = 0.0
    original_accuracy: float = 0.0
    poisoned_accuracy: float = 0.0
    timestamp: str = None

    def __post_init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def to_dict(self) -> Dict:
        """Convert result to dictionary for serialization."""
        return {
            "config": {
                "poison_type": self.config.poison_type.value,
                "poison_ratio": self.config.poison_ratio,
                "pgd_eps": self.config.pgd_eps,
                "pgd_alpha": self.config.pgd_alpha,
                "pgd_steps": self.config.pgd_steps,
                "random_seed": self.config.random_seed,
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
