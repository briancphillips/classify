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
        x.requires_grad_(True)
        loss = torch.mean((x - original) ** 2)  # L2 loss
        loss.backward()
        grad = x.grad.detach()
        x.requires_grad_(False)
        return grad

    def _step(
        self, x: torch.Tensor, grad: torch.Tensor, learning_rate: float
    ) -> torch.Tensor:
        """Take a gradient step"""
        return torch.clamp(x - learning_rate * grad, 0, 1)

    def validate_image(self, image: torch.Tensor) -> bool:
        """Validate if the image is within valid range"""
        return torch.all(image >= 0) and torch.all(image <= 1)

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

        # Calculate number of samples to poison
        num_samples = len(dataset)
        num_poison = int(num_samples * self.config.poison_ratio)
        logger.info(f"Poisoning {num_poison} out of {num_samples} samples")

        # Store original labels for accuracy evaluation
        original_labels = []
        if isinstance(dataset, torch.utils.data.dataset.Subset):
            base_dataset = dataset.dataset
            if isinstance(base_dataset, datasets.CIFAR100):
                original_labels = [base_dataset.targets[i] for i in dataset.indices]
            elif isinstance(base_dataset, datasets.GTSRB):
                original_labels = [base_dataset._samples[i][1] for i in dataset.indices]
            elif isinstance(base_dataset, datasets.ImageFolder):
                original_labels = [base_dataset.targets[i] for i in dataset.indices]
        else:
            original_labels = [dataset[i][1] for i in range(num_samples)]

        # Create a copy of the dataset to avoid modifying the original
        poisoned_dataset = copy.deepcopy(dataset)
        poisoned_indices = []

        # Randomly select indices to poison
        indices_to_poison = np.random.choice(
            num_samples, size=num_poison, replace=False
        )

        # Create dataloader for poisoning
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False
        )

        # Track poisoning success
        poison_success = 0
        total_poisoned = 0

        # Perform gradient ascent attack on selected samples
        for idx, (data, target) in enumerate(tqdm(dataloader, desc="Poisoning samples")):
            if idx not in indices_to_poison:
                continue

            data = data.to(self.device)
            target = target.to(self.device)

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

            # Update the dataset with poisoned sample
            poisoned_data = (perturbed_data.cpu().numpy() * 255).astype(np.uint8)
            if len(poisoned_data.shape) == 4:  # If batched
                poisoned_data = poisoned_data[0]  # Remove batch dimension
            
            # Convert from CHW to HWC if needed
            if poisoned_data.shape[0] == 3:  # If in CHW format
                poisoned_data = np.transpose(poisoned_data, (1, 2, 0))  # Convert to HWC
                
            if isinstance(dataset, torch.utils.data.dataset.Subset):
                poisoned_dataset.dataset.data[dataset.indices[idx]] = poisoned_data
            else:
                poisoned_dataset.data[idx] = poisoned_data

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
