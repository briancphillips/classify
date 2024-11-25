import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional

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

    def poison_dataset(
        self, dataset: Dataset, model: nn.Module
    ) -> Tuple[Dataset, PoisonResult]:
        """Poison a dataset using gradient ascent."""
        logger.info(
            f"Starting Gradient Ascent poisoning with ratio={self.config.poison_ratio}"
        )

        # Move model to device and set to eval mode
        self.model = model.to(self.device)
        self.model.eval()

        result = PoisonResult(self.config)
        num_poison = int(len(dataset) * self.config.poison_ratio)
        indices = np.random.choice(len(dataset), num_poison, replace=False)

        # Convert dataset to tensor format if needed
        if hasattr(dataset, "data") and not isinstance(dataset.data, torch.Tensor):
            data = torch.tensor(dataset.data).float()
            if len(data.shape) == 3:
                data = data.unsqueeze(1)
            if data.shape[-3] == 3:
                data = data.permute(0, 3, 1, 2)
        else:
            data = dataset.data.clone()

        # Normalize data to [0, 1] if needed
        if data.max() > 1:
            data = data / 255.0

        poisoned_data = data.clone()
        poisoned_data = move_to_device(poisoned_data, self.device)

        # Gradient ascent parameters
        num_steps = self.config.ga_pop_size
        num_iterations = self.config.ga_generations
        learning_rate = self.config.ga_mutation_rate
        eps = 0.1  # Maximum perturbation size

        for idx in tqdm(indices, desc="Applying Gradient Ascent"):
            original = poisoned_data[idx : idx + 1].to(self.device)
            x = original.clone()

            # Gradient ascent loop
            for iteration in range(num_iterations):
                for step in range(num_steps):
                    grad = self._compute_gradient(x, original)
                    x = self._step(x, grad, learning_rate)
                    # Clip perturbation size
                    x = torch.clamp(
                        original + torch.clamp(x - original, -eps, eps), 0, 1
                    )

                if iteration % 10 == 0:
                    clear_memory(self.device)

            poisoned_data[idx] = x

        # Update dataset
        if hasattr(dataset, "data"):
            if isinstance(dataset.data, np.ndarray):
                poisoned_data = (poisoned_data.cpu().numpy() * 255).astype(np.uint8)
            dataset.data = (
                poisoned_data.cpu()
                if isinstance(poisoned_data, torch.Tensor)
                else poisoned_data
            )

        # Create data loaders for evaluation
        clean_loader = DataLoader(dataset, batch_size=128, shuffle=False)
        poisoned_loader = DataLoader(
            dataset, batch_size=128, shuffle=False
        )  # Using updated dataset

        # Evaluate model on clean and poisoned data
        result.original_accuracy = self._evaluate_model(clean_loader)
        result.poisoned_accuracy = self._evaluate_model(poisoned_loader)

        # Calculate attack success rate
        success_count = 0
        total_poison = 0
        self.model.eval()
        with torch.no_grad():
            for idx in indices:
                img = poisoned_data[idx : idx + 1].to(self.device)
                output = self.model(img)
                pred = output.argmax(1).item()
                if pred != dataset[idx][1]:  # Different from original label
                    success_count += 1
                total_poison += 1

        result.poison_success_rate = (
            100.0 * success_count / total_poison if total_poison > 0 else 0.0
        )
        result.poisoned_indices = indices.tolist()

        logger.info(f"Attack success rate: {result.poison_success_rate:.2f}%")
        logger.info(f"Original accuracy: {result.original_accuracy:.2f}%")
        logger.info(f"Poisoned accuracy: {result.poisoned_accuracy:.2f}%")

        return dataset, result

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
