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
                num_classes=100,  # CIFAR100 has 100 classes
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
            elif isinstance(poisoned_dataset, datasets.CIFAR100):
                # For CIFAR100 dataset
                poisoned_dataset.data[idx] = (
                    (perturbed_img.permute(1, 2, 0) * 255).numpy().astype(np.uint8)
                )
                poisoned_dataset.targets[idx] = label
            elif isinstance(poisoned_dataset, torch.utils.data.dataset.Subset):
                # For Subset datasets, modify the underlying dataset
                base_dataset = poisoned_dataset.dataset
                base_idx = poisoned_dataset.indices[idx]

                if isinstance(base_dataset, datasets.CIFAR100):
                    base_dataset.data[base_idx] = (
                        (perturbed_img.permute(1, 2, 0) * 255).numpy().astype(np.uint8)
                    )
                    base_dataset.targets[base_idx] = label
                elif isinstance(base_dataset, datasets.GTSRB):
                    if hasattr(base_dataset, "_samples"):
                        img_path, target = base_dataset._samples[base_idx]
                        perturbed_pil = transforms.ToPILImage()(perturbed_img)
                        perturbed_pil.save(img_path)
                        base_dataset._samples[base_idx] = (img_path, target)
                elif isinstance(base_dataset, datasets.ImageFolder):
                    img_path = base_dataset.imgs[base_idx][0]
                    base_dataset.imgs[base_idx] = (img_path, label)
                    base_dataset.samples[base_idx] = (img_path, label)
                    if hasattr(base_dataset, "cache"):
                        base_dataset.cache[img_path] = transforms.ToPILImage()(
                            perturbed_img
                        )

        # Create result object
        result = PoisonResult(self.config)
        result.poisoned_indices = poison_indices

        # Create data loaders for evaluation
        clean_loader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0,  # Use single process
            pin_memory=True,
            persistent_workers=False,
        )
        poisoned_loader = DataLoader(
            poisoned_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0,  # Use single process
            pin_memory=True,
            persistent_workers=False,
        )

        # Evaluate model on clean and poisoned data using original labels
        result.original_accuracy = self._evaluate_model(clean_loader)
        result.poisoned_accuracy = self._evaluate_model_with_original_labels(
            poisoned_loader, original_labels
        )

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
                if pred != original_labels[idx]:  # Compare with original label
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
                inputs = move_to_device(inputs, self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)

                # Get batch size number of original labels
                batch_size = inputs.size(0)
                batch_labels = original_labels[idx : idx + batch_size]
                batch_labels = torch.tensor(batch_labels, device=self.device)

                total += batch_size
                correct += predicted.eq(batch_labels).sum().item()
                idx += batch_size

        return 100.0 * correct / total
