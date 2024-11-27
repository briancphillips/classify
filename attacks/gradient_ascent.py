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
        if not hasattr(result, 'dataset_name') or not result.dataset_name:
            result.dataset_name = self.dataset_name
        num_poison = int(len(dataset) * self.config.poison_ratio)
        indices = np.random.choice(len(dataset), num_poison, replace=False)

        # Create a copy of the dataset
        poisoned_dataset = copy.deepcopy(dataset)

        # Convert dataset to tensor format
        if isinstance(dataset, datasets.ImageFolder):
            # For ImageFolder datasets, load and resize all images to a consistent size
            all_data = []
            target_size = (224, 224)  # Standard ImageNet size
            transform = transforms.Compose(
                [
                    transforms.Resize(target_size),
                    transforms.ToTensor(),
                ]
            )
            for img_path, _ in dataset.samples:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img)
                all_data.append(img_tensor)
            data = torch.stack(all_data)
        elif isinstance(dataset, torch.utils.data.dataset.Subset):
            # For Subset datasets, get data from the underlying dataset
            base_dataset = dataset.dataset
            if isinstance(base_dataset, datasets.CIFAR100):
                data = torch.tensor(base_dataset.data[dataset.indices]).float()
                # CIFAR100 data is in (N, H, W, C) format, convert to (N, C, H, W)
                data = data.permute(0, 3, 1, 2)
                data = data / 255.0  # Normalize to [0, 1]
            elif isinstance(base_dataset, datasets.GTSRB):
                all_data = []
                transform = transforms.Compose(
                    [
                        transforms.Resize(
                            (32, 32),
                            interpolation=transforms.InterpolationMode.BILINEAR,
                        ),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
                for idx in dataset.indices:
                    img_path, _ = base_dataset._samples[idx]
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = transform(img)
                    all_data.append(img_tensor)
                data = torch.stack(all_data)
            elif isinstance(base_dataset, datasets.ImageFolder):
                all_data = []
                target_size = (224, 224)  # Standard ImageNet size
                transform = transforms.Compose(
                    [
                        transforms.Resize(target_size),
                        transforms.ToTensor(),
                    ]
                )
                for idx in dataset.indices:
                    img_path, _ = base_dataset.samples[idx]
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = transform(img)
                    all_data.append(img_tensor)
                data = torch.stack(all_data)
            else:
                raise ValueError(f"Unsupported base dataset type: {type(base_dataset)}")
        elif hasattr(dataset, "data"):
            # For datasets with .data attribute (e.g., CIFAR100)
            if not isinstance(dataset.data, torch.Tensor):
                data = torch.tensor(dataset.data).float()
                # Convert from (N, H, W, C) to (N, C, H, W)
                data = data.permute(0, 3, 1, 2)
                data = data / 255.0  # Normalize to [0, 1]
            else:
                data = dataset.data.clone()
        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")

        poisoned_data = data.clone()
        poisoned_data = move_to_device(poisoned_data, self.device)

        # Get attack parameters from config
        steps_per_iter = self.config.ga_steps  # Number of gradient steps per iteration
        num_iterations = self.config.ga_iterations  # Number of outer iterations
        learning_rate = self.config.ga_lr  # Learning rate for gradient ascent
        eps = 0.1  # Maximum perturbation size

        for idx in tqdm(indices, desc="Applying Gradient Ascent"):
            original = poisoned_data[idx : idx + 1].to(self.device)
            x = original.clone()

            # Gradient ascent loop
            for iteration in range(num_iterations):
                for step in range(steps_per_iter):
                    grad = self._compute_gradient(x, original)
                    x = self._step(x, grad, learning_rate)
                    # Clip perturbation size
                    x = torch.clamp(
                        original + torch.clamp(x - original, -eps, eps), 0, 1
                    )

                if iteration % 10 == 0:
                    clear_memory(self.device)

            poisoned_data[idx] = x

        # Update dataset based on its type
        if isinstance(poisoned_dataset, datasets.ImageFolder):
            # For ImageFolder datasets, save perturbed images back to files
            for idx in indices:
                img_path = poisoned_dataset.samples[idx][0]
                perturbed_img = transforms.ToPILImage()(poisoned_data[idx].cpu())
                # Resize back to original size if needed
                original_img = Image.open(img_path)
                if perturbed_img.size != original_img.size:
                    perturbed_img = perturbed_img.resize(
                        original_img.size, Image.BILINEAR
                    )
                perturbed_img.save(img_path)
        elif isinstance(poisoned_dataset, torch.utils.data.dataset.Subset):
            # For Subset datasets, update the underlying dataset
            base_dataset = poisoned_dataset.dataset
            for i, idx in enumerate(indices):
                base_idx = poisoned_dataset.indices[idx]
                if isinstance(base_dataset, datasets.CIFAR100):
                    # Convert from (C, H, W) to (H, W, C) for CIFAR100
                    perturbed = poisoned_data[i].permute(1, 2, 0)
                    base_dataset.data[base_idx] = (
                        perturbed.cpu().numpy() * 255
                    ).astype(np.uint8)
                elif isinstance(base_dataset, datasets.GTSRB):
                    if hasattr(base_dataset, "_samples"):
                        img_path, target = base_dataset._samples[base_idx]
                        perturbed_img = transforms.ToPILImage()(poisoned_data[i].cpu())
                        # Resize to match the transform size
                        perturbed_img = perturbed_img.resize((32, 32), Image.BILINEAR)
                        perturbed_img.save(img_path)
                elif isinstance(base_dataset, datasets.ImageFolder):
                    img_path = base_dataset.samples[base_idx][0]
                    perturbed_img = transforms.ToPILImage()(poisoned_data[i].cpu())
                    # Resize back to original size if needed
                    original_img = Image.open(img_path)
                    if perturbed_img.size != original_img.size:
                        perturbed_img = perturbed_img.resize(
                            original_img.size, Image.BILINEAR
                        )
                    perturbed_img.save(img_path)
        elif hasattr(poisoned_dataset, "data"):
            # For datasets with .data attribute
            if isinstance(poisoned_dataset.data, np.ndarray):
                # Convert from (N, C, H, W) to (N, H, W, C) for numpy arrays
                poisoned_data = poisoned_data.permute(0, 2, 3, 1)
                poisoned_data = (poisoned_data.cpu().numpy() * 255).astype(np.uint8)
            poisoned_dataset.data = poisoned_data

        # Create data loaders for evaluation
        clean_loader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )
        poisoned_loader = DataLoader(
            poisoned_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )

        # Evaluate attack effectiveness
        result.original_accuracy = self._evaluate_model(clean_loader)
        result.poisoned_accuracy = self._evaluate_model(poisoned_loader)

        # Calculate attack success rate
        success_count = 0
        total_poison = 0
        self.model.eval()
        with torch.no_grad():
            for idx in indices:
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
