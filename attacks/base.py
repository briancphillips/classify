import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset
from typing import Tuple, Optional
import logging
from utils.logging import get_logger

logger = get_logger(__name__)


class PoisonAttack:
    """Base class for poison attacks"""

    def __init__(self, config: "PoisonConfig", device: torch.device):
        self.config = config
        self.device = device
        self.dataset_name = ""  # Initialize with empty string
        self.model = None
        if config.random_seed is not None:
            torch.manual_seed(config.random_seed)
            np.random.seed(config.random_seed)
            random.seed(config.random_seed)

    def poison_dataset(
        self, dataset: Dataset, model: nn.Module
    ) -> Tuple[Dataset, "PoisonResult"]:
        """Poison a dataset according to configuration"""
        raise NotImplementedError

    def validate_image(self, image: torch.Tensor, normalized: bool = True) -> bool:
        """Validate image tensor format and values."""
        if not isinstance(image, torch.Tensor):
            logger.error("Input must be a torch.Tensor")
            return False

        if image.dim() != 4:
            logger.error(f"Expected 4D tensor (B,C,H,W), got {image.dim()}D")
            return False

        # GTSRB normalization parameters
        mean = torch.tensor([0.3337, 0.3064, 0.3171], device=image.device).view(
            1, 3, 1, 1
        )
        std = torch.tensor([0.2672, 0.2564, 0.2629], device=image.device).view(
            1, 3, 1, 1
        )

        if normalized:
            # For normalized images, use a more lenient range check
            # Allow for some numerical error in the normalization process
            min_val = (-mean / std).min().item() - 1.0  # Add buffer for numerical error
            max_val = (
                (1 - mean) / std
            ).max().item() + 1.0  # Add buffer for numerical error

            if not (min_val <= image.min() and image.max() <= max_val):
                logger.error(
                    f"Normalized image values must be in [{min_val:.2f}, {max_val:.2f}], "
                    f"got [{image.min():.2f}, {image.max():.2f}]"
                )
                return False
        else:
            # For unnormalized images, check if values are in [0,1]
            if not (0 <= image.min() and image.max() <= 1):
                logger.error(
                    f"Image values must be in [0,1], got [{image.min():.2f}, {image.max():.2f}]"
                )
                return False

        return True

    def validate_labels(self, labels: torch.Tensor, num_classes: int) -> bool:
        """Validate label tensor format and values."""
        if not isinstance(labels, torch.Tensor):
            logger.error("Labels must be a torch.Tensor")
            return False

        if labels.dim() != 1:
            logger.error(f"Expected 1D tensor, got {labels.dim()}D")
            return False

        if not (0 <= labels.min() and labels.max() < num_classes):
            logger.error(
                f"Labels must be in [0,{num_classes-1}], got [{labels.min()}, {labels.max()}]"
            )
            return False

        return True
