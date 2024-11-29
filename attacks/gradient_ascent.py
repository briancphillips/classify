import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional
import copy
from PIL import Image
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F

from .base import PoisonAttack
from config.dataclasses import PoisonResult
from utils.device import clear_memory, move_to_device
from utils.logging import get_logger

logger = get_logger(__name__)


class GradientAscentAttack(PoisonAttack):
    """Gradient Ascent Attack"""

    def _compute_gradient(
        self, x: torch.Tensor, original: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient for optimization"""
        x.requires_grad = True
        output = self.model(x)
        loss = F.cross_entropy(output, self.model(original).argmax(dim=1))
        loss.backward()
        return x.grad.detach()

    def _step(
        self, x: torch.Tensor, grad: torch.Tensor, learning_rate: float
    ) -> torch.Tensor:
        """Take a gradient step"""
        return torch.clamp(x + learning_rate * grad.sign(), 0, 1).detach()

    def validate_image(self, image: torch.Tensor) -> bool:
        """Validate if the image is within valid range"""
        return bool(torch.all((image >= 0) & (image <= 1)))

    def poison_dataset(
        self, dataset: Dataset, model: nn.Module
    ) -> Tuple[Dataset, PoisonResult]:
        """Poison a dataset using gradient ascent."""
        logger.info(
            f"Starting Gradient Ascent poisoning with ratio={self.config.poison_ratio}, "
            f"steps={self.config.ga_steps}, iterations={self.config.ga_iterations}, "
            f"learning_rate={self.config.ga_lr}"
        )

        # Move model to device and set to eval mode
        self.model = model.to(self.device)
        self.model.eval()

        # Create a copy of the dataset to avoid modifying the original
        poisoned_dataset = copy.deepcopy(dataset)
        
        # Create dataloader for poisoning
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Calculate number of samples to poison
        num_samples = len(dataset)
        num_poison = int(num_samples * self.config.poison_ratio)
        logger.info(f"Poisoning {num_poison} out of {num_samples} samples")
        
        # Randomly select indices to poison
        indices_to_poison = np.random.choice(num_samples, size=num_poison, replace=False)
        
        # Track poisoning success
        poison_success = 0
        total_poisoned = 0
        poisoned_indices = []

        # Perform gradient ascent attack on selected samples
        for idx, (data, target) in enumerate(tqdm(dataloader, desc="Poisoning samples")):
            if idx not in indices_to_poison:
                continue

            # Move data to device and normalize
            data = data.to(self.device)  # Shape: [1, C, H, W]
            target = target.to(self.device)

            # Normalize to [0,1] range
            if data.dtype == torch.uint8:
                data = data.float() / 255.0

            # Initialize perturbed data
            perturbed_data = data.clone().detach()
            
            # Gradient ascent attack loop
            for iteration in range(self.config.ga_iterations):
                # Inner optimization loop
                for step in range(self.config.ga_steps):
                    grad = self._compute_gradient(perturbed_data, data)
                    perturbed_data = self._step(perturbed_data, grad, self.config.ga_lr)

                # Validate and project perturbed data
                if not self.validate_image(perturbed_data):
                    logger.warning(f"Invalid perturbed image at index {idx}, iteration {iteration}")
                    continue

            # Check if poisoning was successful
            with torch.no_grad():
                output = self.model(perturbed_data)
                pred = output.argmax(dim=1)
                if pred != target:
                    poison_success += 1

            # Convert back to uint8 and correct format
            perturbed_data = (perturbed_data * 255).byte()  # Shape: [1, C, H, W]
            perturbed_data = perturbed_data.squeeze(0)  # Remove batch dimension -> [C, H, W]
            perturbed_data = perturbed_data.permute(1, 2, 0)  # Convert to HWC format -> [H, W, C]

            # Update the dataset with poisoned sample
            if isinstance(dataset, torch.utils.data.dataset.Subset):
                poisoned_dataset.dataset.data[dataset.indices[idx]] = perturbed_data.cpu().numpy()
            else:
                poisoned_dataset.data[idx] = perturbed_data.cpu().numpy()

            poisoned_indices.append(idx)
            total_poisoned += 1

            # Clear GPU memory
            clear_memory()

        # Calculate success rate
        poison_success_rate = poison_success / total_poisoned if total_poisoned > 0 else 0.0
        logger.info(f"Poisoning success rate: {poison_success_rate:.2%}")

        # Create and return PoisonResult
        result = PoisonResult(
            config=self.config,
            dataset_name=self.dataset_name,
            poisoned_indices=poisoned_indices,
            poison_success_rate=poison_success_rate,
            original_accuracy=0.0,  # Will be computed later
            poisoned_accuracy=0.0,  # Will be computed later
        )

        return poisoned_dataset, result

    def _evaluate_model(self, dataloader: DataLoader) -> float:
        """Evaluate model accuracy"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = move_to_device(inputs, self.device), move_to_device(
                    targets, self.device
                )
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return 100.0 * correct / total
