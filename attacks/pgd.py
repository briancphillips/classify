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
from typing import Optional, Tuple

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

        # Calculate number of samples to poison
        num_samples = len(dataset)
        num_poison = int(num_samples * self.config.poison_ratio)
        logger.info(f"Poisoning {num_poison} out of {num_samples} samples")

        # Create a new dataset with the same transforms
        poisoned_dataset = copy.deepcopy(dataset)

        # Randomly select indices to poison
        all_indices = list(range(num_samples))
        random.shuffle(all_indices)
        poison_indices = all_indices[:num_poison]

        # Apply PGD attack to selected samples
        for idx in tqdm(poison_indices, desc="Applying PGD attack"):
            # Get the image and label
            img, label = dataset[idx]

            # Convert to tensor if needed
            if not isinstance(img, torch.Tensor):
                img = transforms.ToTensor()(img)

            # Add batch dimension and move to device
            img = img.unsqueeze(0).to(self.device)

            # Create target tensor for the attack
            target = torch.tensor([label], device=self.device)

            # Perform PGD attack
            perturbed_img = self.pgd_attack(
                img,
                target,
                epsilon=self.config.pgd_eps,
                alpha=self.config.pgd_alpha,
                num_steps=self.config.pgd_steps,
                num_classes=1000,  # This will be validated inside pgd_attack
            )

            if perturbed_img is None:
                logger.warning(f"PGD attack failed for sample {idx}, skipping")
                continue

            # Remove batch dimension and move back to CPU
            perturbed_img = perturbed_img.squeeze(0).cpu()

            # Update the dataset based on its type
            if isinstance(poisoned_dataset, datasets.ImageFolder):
                # For ImageFolder datasets
                img_path = poisoned_dataset.imgs[idx][0]
                poisoned_dataset.imgs[idx] = (img_path, label)
                poisoned_dataset.samples[idx] = (img_path, label)
                if hasattr(poisoned_dataset, "cache"):
                    poisoned_dataset.cache[img_path] = transforms.ToPILImage()(
                        perturbed_img
                    )
            elif isinstance(poisoned_dataset, datasets.GTSRB):
                # For GTSRB dataset
                if hasattr(poisoned_dataset, "_samples"):
                    # Get the original sample info
                    img_path, target = poisoned_dataset._samples[idx]
                    # Save the perturbed image
                    perturbed_pil = transforms.ToPILImage()(perturbed_img)
                    perturbed_pil.save(img_path)
                    # Update the sample info
                    poisoned_dataset._samples[idx] = (img_path, target)
                else:
                    logger.warning(f"Unexpected GTSRB dataset structure at index {idx}")
            else:
                logger.warning(f"Unsupported dataset type: {type(poisoned_dataset)}")
                continue

        # Create result object
        result = PoisonResult(self.config)
        result.poisoned_indices = poison_indices

        # Create data loaders for evaluation
        clean_loader = DataLoader(dataset, batch_size=128, shuffle=False)
        poisoned_loader = DataLoader(poisoned_dataset, batch_size=128, shuffle=False)

        # Evaluate model on clean and poisoned data
        result.original_accuracy = self._evaluate_model(clean_loader)
        result.poisoned_accuracy = self._evaluate_model(poisoned_loader)

        # Calculate attack success rate
        success_count = 0
        total_poison = 0
        self.model.eval()
        with torch.no_grad():
            for idx in poison_indices:
                img, _ = poisoned_dataset[idx]
                if not isinstance(img, torch.Tensor):
                    img = transforms.ToTensor()(img)
                img = img.unsqueeze(0).to(self.device)
                output = self.model(img)
                pred = output.argmax(1).item()
                if pred != dataset[idx][1]:  # Different from original label
                    success_count += 1
                total_poison += 1

        result.poison_success_rate = (
            100.0 * success_count / total_poison if total_poison > 0 else 0.0
        )
        logger.info(f"Attack success rate: {result.poison_success_rate:.2f}%")
        logger.info(f"Original accuracy: {result.original_accuracy:.2f}%")
        logger.info(f"Poisoned accuracy: {result.poisoned_accuracy:.2f}%")

        return poisoned_dataset, result

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

        for step in range(num_steps):
            self.model.zero_grad()

            # Add perturbation to image
            perturbed_image = image + delta

            # Forward pass
            output = self.model(perturbed_image)
            loss = F.cross_entropy(output, target)
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
