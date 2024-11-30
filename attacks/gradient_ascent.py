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
            f"steps={self.config.ga_steps * self.config.ga_iterations}, "
            f"learning_rate={self.config.ga_lr}"
        )
        
        logger.info(f"Attack configuration: {self.config}")

        # Move model to device and set to eval mode
        self.model = model.to(self.device)
        self.model.eval()

        # Create a copy of the dataset to avoid modifying the original
        poisoned_dataset = copy.deepcopy(dataset)
        
        # Calculate number of samples to poison
        num_samples = len(dataset)
        num_poison = int(num_samples * self.config.poison_ratio)
        logger.info(f"Dataset size={num_samples}, poison_ratio={self.config.poison_ratio}")
        logger.info(f"Poisoning {num_poison} out of {num_samples} samples")
        
        # Randomly select indices to poison
        indices_to_poison = np.random.choice(num_samples, size=num_poison, replace=False)
        indices_to_poison = sorted(indices_to_poison)  # Sort for sequential access

        # Track poisoning success
        poison_success = 0
        total_poisoned = 0
        poisoned_indices = []

        # Process samples in batches using poisoning batch size
        batch_size = self.config.batch_size  # Get poisoning batch size from config
        num_batches = (len(indices_to_poison) + batch_size - 1) // batch_size
        
        # Create progress bar for poisoned samples only
        steps_per_batch = self.config.ga_steps  # Steps per iteration
        total_steps = steps_per_batch * self.config.ga_iterations * num_batches  # Total steps across all batches
        
        logger.info(f"Debug - Poisoning batch size: {batch_size}, Num batches: {num_batches}")
        logger.info(f"Debug - Steps per iteration: {steps_per_batch}, Iterations: {self.config.ga_iterations}")
        logger.info(f"Debug - Total steps: {total_steps}")
        
        pbar = tqdm(total=total_steps, desc="Poisoning steps")
        
        # Process samples in batches
        for batch_start in range(0, len(indices_to_poison), batch_size):
            batch_indices = indices_to_poison[batch_start:batch_start + batch_size]
            batch_data = []
            batch_targets = []
            
            # Collect batch data
            for idx in batch_indices:
                data, target = dataset[idx]
                if not isinstance(data, torch.Tensor):
                    data = transforms.ToTensor()(data)
                
                # Ensure we have a 4D tensor [B, C, H, W]
                if len(data.shape) == 3:  # [C, H, W]
                    data = data.unsqueeze(0)  # Add batch dimension -> [1, C, H, W]
                elif len(data.shape) == 5:  # [1, 1, C, H, W]
                    data = data.squeeze(1)  # Remove extra dimension -> [1, C, H, W]
                
                # Normalize to [0,1] range if needed
                if data.dtype == torch.uint8:
                    data = data.float() / 255.0
                
                batch_data.append(data)
                batch_targets.append(target)
            
            # Stack batch data
            batch_data = torch.cat(batch_data, dim=0).to(self.device)
            batch_targets = torch.tensor(batch_targets, device=self.device)
            
            # Initialize perturbed data
            perturbed_data = batch_data.clone().detach()
            
            # Gradient ascent iterations for the batch
            for iteration in range(self.config.ga_iterations):
                # Inner optimization steps
                for step in range(self.config.ga_steps):
                    perturbed_data.requires_grad = True
                    output = self.model(perturbed_data)
                    original_preds = self.model(batch_data).argmax(dim=1)
                    loss = F.cross_entropy(output, original_preds)
                    loss.backward()
                    
                    # Update perturbed data
                    grad = perturbed_data.grad.detach()
                    perturbed_data = torch.clamp(
                        perturbed_data + self.config.ga_lr * grad.sign(), 
                        0, 
                        1
                    ).detach()
                    
                    pbar.update(1)

            # Check poisoning success for each sample in batch
            with torch.no_grad():
                output = self.model(perturbed_data)
                preds = output.argmax(dim=1)
                poison_success += (preds != batch_targets).sum().item()

            # Convert back to uint8 and correct format for each sample
            perturbed_data = (perturbed_data * 255).byte()  # [B, C, H, W]
            
            # Update dataset with poisoned samples
            for i, idx in enumerate(batch_indices):
                # Get individual sample and reshape
                sample = perturbed_data[i].squeeze(0)  # [C, H, W]
                sample = sample.permute(1, 2, 0)  # [H, W, C]
                
                if isinstance(dataset, torch.utils.data.dataset.Subset):
                    poisoned_dataset.dataset.data[dataset.indices[idx]] = sample.cpu().numpy()
                else:
                    poisoned_dataset.data[idx] = sample.cpu().numpy()
                
                poisoned_indices.append(idx)
                total_poisoned += 1

        # Clear GPU memory once after all batches
        clear_memory()

        # Calculate success rate
        poison_success_rate = poison_success / total_poisoned if total_poisoned > 0 else 0.0
        logger.info(f"Poisoning success rate: {poison_success_rate:.2%}")

        # Create data loaders for evaluation with proper batch handling
        eval_batch_size = min(128, len(dataset))  # Use larger batch size for evaluation
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
                targets = torch.tensor(targets, device=self.device)  # Convert target to tensor and move to device

                # Get predictions
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return 100.0 * correct / total
