import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import random
import copy
from tqdm import tqdm
from typing import Optional, Tuple, List

from .base import PoisonAttack
from config.dataclasses import PoisonResult
from utils.device import clear_memory, move_to_device
from utils.logging import get_logger

logger = get_logger(__name__)


class PGDPoisonAttack(PoisonAttack):
    """PGD-based poisoning attack."""

    def poison_dataset(
        self, dataset: Dataset, model: nn.Module
    ) -> Tuple[Dataset, PoisonResult]:
        """Poison a dataset using PGD attack."""
        logger.info(f"Starting PGD poisoning with epsilon={self.config.pgd_eps}")

        # Move model to device and set to eval mode
        self.model = model.to(self.device)
        self.model.eval()

        # Create a copy of the dataset to avoid modifying the original
        poisoned_dataset = copy.deepcopy(dataset)
        
        # Calculate number of samples to poison
        num_samples = len(dataset)
        num_poison = int(num_samples * self.config.poison_ratio)
        logger.info(f"Poisoning {num_poison} out of {num_samples} samples")
        
        # Randomly select indices to poison
        indices_to_poison = np.random.choice(num_samples, size=num_poison, replace=False)
        indices_to_poison = sorted(indices_to_poison)  # Sort for sequential access
        
        # Track poisoning success
        poison_success = 0
        total_poisoned = 0
        poisoned_indices = []

        # Create progress bar for poisoned samples only
        pbar = tqdm(indices_to_poison, desc="Poisoning samples", total=num_poison)
        
        # Perform PGD attack on selected samples
        for idx in pbar:
            # Get sample from dataset
            data, target = dataset[idx]

            # Move data to device and ensure proper dimensions
            if not isinstance(data, torch.Tensor):
                data = transforms.ToTensor()(data)
            
            # Ensure we have a 4D tensor [B, C, H, W]
            if len(data.shape) == 3:  # [C, H, W]
                data = data.unsqueeze(0)  # Add batch dimension -> [1, C, H, W]
            elif len(data.shape) == 5:  # [1, 1, C, H, W]
                data = data.squeeze(1)  # Remove extra dimension -> [1, C, H, W]
            
            data = data.to(self.device)
            target = torch.tensor([target]).to(self.device)

            # Normalize to [0,1] range if needed
            if data.dtype == torch.uint8:
                data = data.float() / 255.0

            # Initialize perturbed data
            perturbed_data = data.clone().detach()

            # PGD attack loop
            for step in range(self.config.pgd_steps):
                perturbed_data.requires_grad = True
                
                # Ensure proper dimensions before model forward pass
                if len(perturbed_data.shape) != 4:
                    logger.error(f"Unexpected perturbed data shape: {perturbed_data.shape}")
                    raise ValueError(f"Expected 4D tensor, got shape {perturbed_data.shape}")
                
                output = self.model(perturbed_data)
                loss = F.cross_entropy(output, target)
                loss.backward()

                # Update perturbed data
                grad = perturbed_data.grad.detach()
                perturbed_data = perturbed_data + self.config.pgd_alpha * grad.sign()

                # Project back to epsilon ball
                delta = torch.clamp(perturbed_data - data, -self.config.pgd_eps, self.config.pgd_eps)
                perturbed_data = torch.clamp(data + delta, 0, 1).detach()

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

            # Update progress bar description
            pbar.set_description(f"Poisoning samples (success rate: {100.0 * poison_success / total_poisoned:.1f}%)")

            # Clear GPU memory
            clear_memory()

        # Calculate success rate
        poison_success_rate = poison_success / total_poisoned if total_poisoned > 0 else 0.0
        logger.info(f"Poisoning success rate: {poison_success_rate:.2%}")

        # Create data loaders for evaluation with proper batch handling
        eval_batch_size = min(128, len(dataset))
        dataloader_kwargs = {
            "batch_size": eval_batch_size,
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": True,
            "persistent_workers": False,
        }
        
        clean_loader = DataLoader(dataset, **dataloader_kwargs)
        poisoned_loader = DataLoader(poisoned_dataset, **dataloader_kwargs)

        # Evaluate model on clean and poisoned data
        result = PoisonResult(self.config, dataset_name=self.dataset_name)
        result.original_accuracy = self._evaluate_model(clean_loader)
        result.poisoned_accuracy = self._evaluate_model(poisoned_loader)
        result.poison_success_rate = float(100.0 * poison_success_rate)
        result.poisoned_indices = indices_to_poison.tolist() if isinstance(indices_to_poison, np.ndarray) else list(indices_to_poison)

        logger.info(f"Original accuracy: {result.original_accuracy:.2f}%")
        logger.info(f"Poisoned accuracy: {result.poisoned_accuracy:.2f}%")

        return poisoned_dataset, result

    def _evaluate_model(self, dataloader: DataLoader) -> float:
        """Evaluate model accuracy"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                # Handle tensor conversion and dimensions
                if not isinstance(inputs, torch.Tensor):
                    inputs = transforms.ToTensor()(inputs)
                
                # Ensure proper dimensions [B, C, H, W]
                if len(inputs.shape) == 3:  # [C, H, W]
                    inputs = inputs.unsqueeze(0)  # Add batch dimension
                elif len(inputs.shape) == 5:  # [1, 1, C, H, W]
                    inputs = inputs.squeeze(1)  # Remove extra dimension
                
                # Move to device and normalize if needed
                inputs = inputs.to(self.device)
                if inputs.dtype == torch.uint8:
                    inputs = inputs.float() / 255.0
                targets = targets.to(self.device)

                # Get predictions
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return 100.0 * correct / total

    def pgd_attack(
        self,
        image: torch.Tensor,
        target: torch.Tensor,
        epsilon: float = 0.3,
        alpha: float = 0.01,
        num_steps: int = 40,
        num_classes: int = 1000,
    ) -> Optional[torch.Tensor]:
        """
        Perform PGD attack with validation and proper device management.
        Returns None if validation fails.
        """
        # Validate inputs
        if not self.validate_image(image, normalized=True) or not self.validate_labels(
            target, num_classes
        ):
            return None

        # GTSRB normalization parameters
        mean = torch.tensor([0.3337, 0.3064, 0.3171], device=image.device).view(
            1, 3, 1, 1
        )
        std = torch.tensor([0.2672, 0.2564, 0.2629], device=image.device).view(
            1, 3, 1, 1
        )

        # Calculate valid value ranges for normalized space
        min_val = (-mean / std).min().item() - 1.0  # Add buffer for numerical error
        max_val = (
            (1 - mean) / std
        ).max().item() + 1.0  # Add buffer for numerical error

        # Convert epsilon and alpha to normalized space
        epsilon_norm = epsilon / std.mean().item()
        alpha_norm = alpha / std.mean().item()

        # Initialize perturbation
        delta = torch.zeros_like(image, requires_grad=True)

        # Create a random target different from the original
        random_target = torch.randint(
            0, num_classes - 1, target.shape, device=target.device
        )
        random_target = (
            random_target + (random_target >= target).long()
        )  # Ensure different class

        for step in range(num_steps):
            self.model.zero_grad()

            # Add perturbation to image
            perturbed_image = image + delta

            # Forward pass
            output = self.model(perturbed_image)

            # Maximize loss for correct class (minimize probability of correct class)
            loss = -F.cross_entropy(output, target)  # Negative sign for maximization
            loss.backward()

            # Update perturbation with gradient ascent
            with torch.no_grad():
                # Update delta
                grad_sign = delta.grad.sign()
                delta.data = delta.data + alpha_norm * grad_sign

                # Project to epsilon ball
                delta.data = torch.clamp(delta.data, -epsilon_norm, epsilon_norm)

                # Ensure perturbed image is valid
                perturbed = image + delta.data
                delta.data = torch.clamp(perturbed, min_val, max_val) - image

            # Reset gradients
            delta.grad.zero_()

            if step % 10 == 0:
                clear_memory(self.device)

        # Get final perturbed image
        final_image = torch.clamp(image + delta.detach(), min_val, max_val)

        # Validate final image
        if not self.validate_image(final_image, normalized=True):
            return None

        return final_image

    def _evaluate_model_with_original_labels(
        self, dataloader: DataLoader, original_labels: List[int]
    ) -> float:
        """Evaluate model accuracy using original labels."""
        self.model.eval()
        correct = 0
        total = 0
        idx = 0

        with torch.no_grad():
            for inputs, _ in dataloader:
                # Handle tensor conversion and dimensions
                if not isinstance(inputs, torch.Tensor):
                    inputs = transforms.ToTensor()(inputs)
                
                # Ensure proper dimensions [B, C, H, W]
                if len(inputs.shape) == 3:  # [C, H, W]
                    inputs = inputs.unsqueeze(0)  # Add batch dimension
                elif len(inputs.shape) == 5:  # [1, 1, C, H, W]
                    inputs = inputs.squeeze(1)  # Remove extra dimension
                
                # Move to device and normalize if needed
                inputs = inputs.to(self.device)
                if inputs.dtype == torch.uint8:
                    inputs = inputs.float() / 255.0

                # Get batch size number of original labels
                batch_size = inputs.size(0)
                batch_labels = original_labels[idx : idx + batch_size]
                batch_labels = torch.tensor(batch_labels, device=self.device)

                # Get predictions
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += batch_size
                correct += predicted.eq(batch_labels).sum().item()
                idx += batch_size

        return 100.0 * correct / total
