import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List
from torchvision import datasets, transforms
import copy
from tqdm import tqdm

from .base import PoisonAttack
from config.dataclasses import PoisonResult
from config.types import PoisonType
from utils.device import move_to_device
from utils.logging import get_logger

logger = get_logger(__name__)


class LabelFlipAttack(PoisonAttack):
    """Label Flipping Attack"""

    def poison_dataset(
        self, dataset: Dataset, model: nn.Module
    ) -> Tuple[Dataset, PoisonResult]:
        """Poison a dataset by flipping labels according to the specified strategy."""
        # Validate configuration
        if not 0 <= self.config.poison_ratio <= 1:
            raise ValueError("Poison ratio must be between 0 and 1")
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")

        # Store model for evaluation
        self.model = model.to(self.device)
        self.model.eval()

        # Create a copy of the dataset to avoid modifying the original
        poisoned_dataset = copy.deepcopy(dataset)
        
        result = PoisonResult(self.config, dataset_name=self.dataset_name)
        num_samples = len(dataset)
        num_poison = int(num_samples * self.config.poison_ratio)

        # Get all labels
        if isinstance(dataset, torch.utils.data.dataset.Subset):
            base_dataset = dataset.dataset
            if isinstance(base_dataset, datasets.CIFAR100):
                all_labels = [base_dataset.targets[i] for i in dataset.indices]
            elif isinstance(base_dataset, datasets.GTSRB):
                all_labels = [base_dataset._samples[i][1] for i in dataset.indices]
            elif isinstance(base_dataset, datasets.ImageFolder):
                all_labels = [base_dataset.targets[i] for i in dataset.indices]
            else:
                raise ValueError(f"Unsupported base dataset type: {type(base_dataset)}")
        else:
            all_labels = [dataset[i][1] for i in range(num_samples)]

        unique_labels = sorted(list(set(all_labels)))
        logger.info(f"Found {len(unique_labels)} unique labels in dataset")

        # Select indices to poison based on attack type
        if self.config.poison_type == PoisonType.LABEL_FLIP_RANDOM_TO_RANDOM:
            indices = self._random_to_random_flip(
                num_samples, num_poison, all_labels, unique_labels
            )
        elif self.config.poison_type == PoisonType.LABEL_FLIP_RANDOM_TO_TARGET:
            indices = self._random_to_target_flip(num_samples, num_poison, all_labels)
        elif self.config.poison_type == PoisonType.LABEL_FLIP_SOURCE_TO_TARGET:
            indices = self._source_to_target_flip(num_samples, num_poison, all_labels)
        else:
            raise ValueError(f"Unsupported label flip type: {self.config.poison_type}")

        # Apply label flips
        logger.info(f"Applying label flips to {len(indices)} samples")
        pbar = tqdm(indices, desc="Flipping labels", total=len(indices))
        
        if isinstance(dataset, torch.utils.data.dataset.Subset):
            base_dataset = dataset.dataset
            for idx in pbar:
                base_idx = dataset.indices[idx]
                if isinstance(base_dataset, datasets.CIFAR100):
                    if self.config.poison_type == PoisonType.LABEL_FLIP_RANDOM_TO_RANDOM:
                        current_label = base_dataset.targets[base_idx]
                        new_label = np.random.choice([l for l in unique_labels if l != current_label])
                        base_dataset.targets[base_idx] = new_label
                    else:
                        base_dataset.targets[base_idx] = self.config.target_class
                elif isinstance(base_dataset, datasets.GTSRB):
                    if hasattr(base_dataset, "_samples"):
                        img_path, _ = base_dataset._samples[base_idx]
                        if self.config.poison_type == PoisonType.LABEL_FLIP_RANDOM_TO_RANDOM:
                            current_label = base_dataset._samples[base_idx][1]
                            new_label = np.random.choice([l for l in unique_labels if l != current_label])
                        else:
                            new_label = self.config.target_class
                        base_dataset._samples[base_idx] = (img_path, new_label)
                elif isinstance(base_dataset, datasets.ImageFolder):
                    img_path = base_dataset.samples[base_idx][0]
                    if self.config.poison_type == PoisonType.LABEL_FLIP_RANDOM_TO_RANDOM:
                        current_label = base_dataset.targets[base_idx]
                        new_label = np.random.choice([l for l in unique_labels if l != current_label])
                    else:
                        new_label = self.config.target_class
                    # Update all label references consistently
                    base_dataset.targets[base_idx] = new_label
                    base_dataset.samples[base_idx] = (img_path, new_label)
                    base_dataset.imgs[base_idx] = (img_path, new_label)
        else:
            for idx in pbar:
                if hasattr(dataset, "targets"):
                    if self.config.poison_type == PoisonType.LABEL_FLIP_RANDOM_TO_RANDOM:
                        current_label = dataset.targets[idx]
                        new_label = np.random.choice([l for l in unique_labels if l != current_label])
                    else:
                        new_label = self.config.target_class
                    dataset.targets[idx] = new_label

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
        result.original_accuracy = self._evaluate_model(clean_loader)
        result.poisoned_accuracy = self._evaluate_model(poisoned_loader)

        # Calculate attack success rate
        success_count = 0
        total_poison = 0
        self.model.eval()
        
        with torch.no_grad():
            for idx in indices:
                img, _ = dataset[idx]
                # Convert to tensor if needed
                if not isinstance(img, torch.Tensor):
                    img = transforms.ToTensor()(img)
                
                # Ensure proper dimensions (B, C, H, W)
                if len(img.shape) == 3:
                    img = img.unsqueeze(0)
                elif len(img.shape) != 4:
                    raise ValueError(f"Unexpected image shape: {img.shape}")
                
                # Move to device and normalize if needed
                img = move_to_device(img, self.device)
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                
                # Get prediction
                output = self.model(img)
                pred = output.argmax(1).item()
                
                # Check if prediction differs from original label
                if pred != all_labels[idx]:
                    success_count += 1
                total_poison += 1

        # Update results
        result.poison_success_rate = float(100.0 * success_count / total_poison if total_poison > 0 else 0.0)
        result.poisoned_indices = indices.tolist() if isinstance(indices, np.ndarray) else list(indices)

        logger.info(f"Attack success rate: {result.poison_success_rate:.2f}%")
        logger.info(f"Original accuracy: {result.original_accuracy:.2f}%")
        logger.info(f"Poisoned accuracy: {result.poisoned_accuracy:.2f}%")

        return poisoned_dataset, result

    def _random_to_random_flip(
        self,
        num_samples: int,
        num_poison: int,
        all_labels: List[int],
        unique_labels: List[int],
    ) -> List[int]:
        """Select indices for random-to-random label flipping."""
        indices = np.random.choice(num_samples, num_poison, replace=False)
        logger.info(
            f"Selected {len(indices)} samples for random-to-random label flipping"
        )
        return indices

    def _random_to_target_flip(
        self, num_samples: int, num_poison: int, all_labels: List[int]
    ) -> List[int]:
        """Select indices for random-to-target label flipping."""
        if self.config.target_class is None:
            raise ValueError(
                "Target class must be specified for random->target flipping"
            )
        # Exclude samples that already have the target label
        valid_indices = [
            i for i, label in enumerate(all_labels) if label != self.config.target_class
        ]
        if len(valid_indices) < num_poison:
            logger.warning(
                f"Only {len(valid_indices)} samples available for flipping to target class {self.config.target_class}"
            )
            num_poison = len(valid_indices)
        indices = np.random.choice(valid_indices, num_poison, replace=False)
        logger.info(
            f"Selected {len(indices)} samples for random-to-target label flipping"
        )
        return indices

    def _source_to_target_flip(
        self, num_samples: int, num_poison: int, all_labels: List[int]
    ) -> List[int]:
        """Select indices for source-to-target label flipping."""
        if None in (self.config.source_class, self.config.target_class):
            raise ValueError(
                "Source and target classes must be specified for source->target flipping"
            )
        # Get indices of samples from source class
        source_indices = [
            i for i, label in enumerate(all_labels) if label == self.config.source_class
        ]
        if len(source_indices) < num_poison:
            logger.warning(
                f"Only {len(source_indices)} samples available from source class {self.config.source_class}"
            )
            num_poison = len(source_indices)
        indices = np.random.choice(source_indices, num_poison, replace=False)
        logger.info(
            f"Selected {len(indices)} samples for source-to-target label flipping"
        )
        return indices

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

    def _evaluate_model_with_original_labels(
        self, dataloader: DataLoader, original_labels: List[int]
    ) -> float:
        """Evaluate model accuracy using original labels"""
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
                batch_labels = original_labels[idx:idx + batch_size]
                batch_labels = torch.tensor(batch_labels, device=self.device)

                # Get predictions
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += batch_size
                correct += predicted.eq(batch_labels).sum().item()
                idx += batch_size

        return 100.0 * correct / total
