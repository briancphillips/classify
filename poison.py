import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import random
import copy
from tqdm import tqdm
import os
import json
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import torch.optim as optim
from utils import get_device, clear_memory, logger
from models import (
    get_model,
    save_model,
    load_model,
    get_dataset_loaders,
    CIFAR100_TRANSFORM_TRAIN,
    CIFAR100_TRANSFORM_TEST,
    GTSRB_TRANSFORM_TRAIN,
    GTSRB_TRANSFORM_TEST,
    IMAGENETTE_TRANSFORM_TRAIN,
    IMAGENETTE_TRANSFORM_TEST,
)


def setup_logging():
    """Configure logging settings."""
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    )

    # Set PIL logging to INFO to suppress debug messages
    logging.getLogger("PIL").setLevel(logging.INFO)

    # Get our logger
    return logging.getLogger(__name__)


# Initialize logger
logger = setup_logging()


class PoisonType(Enum):
    """Types of poisoning attacks"""

    PGD = "pgd"  # Projected Gradient Descent
    GA = "ga"  # Genetic Algorithm
    LABEL_FLIP_RANDOM_TO_RANDOM = "label_flip_random_random"
    LABEL_FLIP_RANDOM_TO_TARGET = "label_flip_random_target"
    LABEL_FLIP_SOURCE_TO_TARGET = "label_flip_source_target"


@dataclass
class PoisonConfig:
    """Configuration for poisoning attacks"""

    poison_type: PoisonType
    poison_ratio: float  # Percentage of dataset to poison (0.0 to 1.0)
    # PGD specific parameters
    pgd_eps: Optional[float] = 0.3  # Epsilon for PGD attack
    pgd_alpha: Optional[float] = 0.01  # Step size for PGD
    pgd_steps: Optional[int] = 40  # Number of PGD steps
    # GA specific parameters
    ga_pop_size: Optional[int] = 50  # Population size for GA
    ga_generations: Optional[int] = 100  # Number of generations
    ga_mutation_rate: Optional[float] = 0.1
    # Label flipping specific parameters
    source_class: Optional[int] = None  # Source class for source->target flipping
    target_class: Optional[int] = None  # Target class for targeted flipping
    random_seed: Optional[int] = 42  # Random seed for reproducibility


@dataclass
class PoisonResult:
    """Results from a poisoning experiment."""

    config: "PoisonConfig"
    poisoned_indices: List[int] = None
    poison_success_rate: float = 0.0
    original_accuracy: float = 0.0
    poisoned_accuracy: float = 0.0
    timestamp: str = None

    def __post_init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def to_dict(self) -> Dict:
        """Convert result to dictionary for serialization."""
        return {
            "config": {
                "poison_type": self.config.poison_type.value,
                "poison_ratio": self.config.poison_ratio,
                "pgd_eps": self.config.pgd_eps,
                "pgd_alpha": self.config.pgd_alpha,
                "pgd_steps": self.config.pgd_steps,
                "random_seed": self.config.random_seed,
            },
            "poisoned_indices": self.poisoned_indices,
            "poison_success_rate": self.poison_success_rate,
            "original_accuracy": self.original_accuracy,
            "poisoned_accuracy": self.poisoned_accuracy,
            "timestamp": self.timestamp,
        }

    def save(self, output_dir: str):
        """Save results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        filename = f"poison_results_{self.timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        logger.info(f"Results saved to {filepath}")


class PoisonAttack:
    """Base class for poison attacks"""

    def __init__(self, config: PoisonConfig, device: torch.device):
        self.config = config
        self.device = device
        if config.random_seed is not None:
            torch.manual_seed(config.random_seed)
            np.random.seed(config.random_seed)
            random.seed(config.random_seed)

    def poison_dataset(
        self, dataset: Dataset, model: nn.Module
    ) -> Tuple[Dataset, PoisonResult]:
        """Poison a dataset according to configuration"""
        raise NotImplementedError

    def validate_image(self, image: torch.Tensor) -> bool:
        """Validate image tensor format and values."""
        if not isinstance(image, torch.Tensor):
            logger.error("Input must be a torch.Tensor")
            return False

        if image.dim() != 4:
            logger.error(f"Expected 4D tensor (B,C,H,W), got {image.dim()}D")
            return False

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
        if not self.validate_image(image) or not self.validate_labels(
            target, num_classes
        ):
            return None

        # Move data to device
        image = move_to_device(image, self.device)
        target = move_to_device(target, self.device)

        # Initialize perturbation
        delta = torch.zeros_like(image, requires_grad=True)

        for step in range(num_steps):
            self.model.zero_grad()

            perturbed_image = image + delta
            perturbed_image = torch.clamp(perturbed_image, 0, 1)

            output = self.model(perturbed_image)
            loss = F.cross_entropy(output, target)
            loss.backward()

            # Update perturbation
            grad_sign = delta.grad.detach().sign()
            delta.data = delta.data + alpha * grad_sign
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            delta.data = torch.clamp(image + delta.data, 0, 1) - image

            delta.grad.zero_()

            if step % 10 == 0:
                clear_memory(self.device)

        final_image = torch.clamp(image + delta.detach(), 0, 1)
        return final_image

    def genetic_attack(
        self,
        image: torch.Tensor,
        target: torch.Tensor,
        population_size: int = 50,
        num_generations: int = 100,
        mutation_rate: float = 0.1,
        num_classes: int = 1000,
    ) -> Optional[torch.Tensor]:
        """
        Perform genetic algorithm-based attack with validation and proper cleanup.
        Returns None if validation fails.
        """
        # Validate inputs
        if not self.validate_image(image) or not self.validate_labels(
            target, num_classes
        ):
            return None

        image = move_to_device(image, self.device)
        target = move_to_device(target, self.device)

        # Initialize population
        population = []
        for _ in range(population_size):
            noise = torch.randn_like(image) * 0.1
            perturbed = torch.clamp(image + noise, 0, 1)
            population.append(perturbed)

        best_fitness = float("-inf")
        best_individual = None

        try:
            for generation in range(num_generations):
                # Evaluate fitness
                fitness_scores = []
                for individual in population:
                    with torch.no_grad():
                        output = self.model(individual)
                        # Negative cross entropy as fitness (higher is better)
                        fitness = -F.cross_entropy(output, target).item()
                        fitness_scores.append(fitness)

                # Select best individuals
                indices = torch.tensor(fitness_scores).argsort(descending=True)
                population = [population[i] for i in indices[: population_size // 2]]

                # Track best individual
                if fitness_scores[indices[0]] > best_fitness:
                    best_fitness = fitness_scores[indices[0]]
                    best_individual = population[0].clone()

                # Create new generation
                while len(population) < population_size:
                    # Crossover
                    parent1, parent2 = random.sample(population, 2)
                    child = (parent1 + parent2) / 2

                    # Mutation
                    if random.random() < mutation_rate:
                        noise = torch.randn_like(child) * 0.1
                        child = torch.clamp(child + noise, 0, 1)

                    population.append(child)

                if generation % 10 == 0:
                    clear_memory(self.device)
                    logger.info(
                        f"Generation {generation}: Best fitness = {best_fitness:.4f}"
                    )

        finally:
            # Cleanup
            clear_memory(self.device)

        return best_individual if best_individual is not None else image


def get_device(device_str: Optional[str] = None) -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU)"""
    if device_str:
        if device_str == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif device_str == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device")
        elif device_str == "cpu":
            device = torch.device("cpu")
            logger.info("Using CPU device")
        else:
            logger.warning(
                f"Requested device '{device_str}' not available, falling back to best available device"
            )
            return get_device()
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device


class PGDPoisonAttack(PoisonAttack):
    """PGD-based poisoning attack."""

    def __init__(self, config: PoisonConfig, device: torch.device):
        self.config = config
        self.device = device
        if config.random_seed is not None:
            torch.manual_seed(config.random_seed)
            np.random.seed(config.random_seed)
            random.seed(config.random_seed)

    def poison_dataset(
        self, dataset: Dataset, model: nn.Module
    ) -> Tuple[Dataset, PoisonResult]:
        """Poison a dataset using PGD attack."""
        logger.info(f"Starting PGD poisoning with epsilon={self.config.pgd_eps}")

        # Move model to device and set to eval mode
        model = model.to(self.device)
        model.eval()

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

            # Perform PGD attack
            perturbed_img = self._pgd_attack(img, model)

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
        result.poison_success_rate = 1.0  # Will be updated after evaluation

        return poisoned_dataset, result

    def _pgd_attack(self, image: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Perform PGD attack on a single image."""
        # Initialize perturbation
        delta = torch.zeros_like(image, requires_grad=True)
        original_label = torch.argmax(model(image), dim=1)

        for step in range(self.config.pgd_steps):
            # Forward pass
            perturbed_image = image + delta
            perturbed_image = torch.clamp(perturbed_image, 0, 1)

            output = model(perturbed_image)
            # Try to maximize loss for the original label
            loss = -F.cross_entropy(output, original_label)

            # Backward pass
            loss.backward()

            # Update perturbation
            grad_sign = delta.grad.detach().sign()
            delta.data = delta.data + self.config.pgd_alpha * grad_sign
            delta.data = torch.clamp(
                delta.data, -self.config.pgd_eps, self.config.pgd_eps
            )
            delta.data = torch.clamp(image + delta.data, 0, 1) - image

            delta.grad.zero_()

            if step % 10 == 0:
                clear_memory(self.device)

        # Return perturbed image
        return torch.clamp(image + delta.detach(), 0, 1)


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

    def poison_dataset(self, dataset: Dataset) -> Tuple[Dataset, PoisonResult]:
        result = PoisonResult(self.config)
        device = get_device()
        num_poison = int(len(dataset) * self.config.poison_ratio)
        indices = np.random.choice(len(dataset), num_poison, replace=False)

        # Convert dataset to tensor format
        if not isinstance(dataset.data, torch.Tensor):
            data = torch.tensor(dataset.data).float()
            if len(data.shape) == 3:
                data = data.unsqueeze(1)
            if data.shape[-3] == 3:
                data = data.permute(0, 3, 1, 2)
        else:
            data = dataset.data.clone()

        # Normalize data to [0, 1]
        if data.max() > 1:
            data = data / 255.0

        poisoned_data = data.clone()
        poisoned_data = poisoned_data.to(device)

        # Gradient ascent parameters
        num_steps = self.config.ga_pop_size
        num_iterations = self.config.ga_generations
        learning_rate = self.config.ga_mutation_rate
        eps = 0.1  # Maximum perturbation size

        for idx in indices:
            original = poisoned_data[idx : idx + 1].to(device)
            x = original.clone()

            # Gradient ascent loop
            for _ in range(num_iterations):
                for _ in range(num_steps):
                    grad = self._compute_gradient(x, original)
                    x = self._step(x, grad, learning_rate)
                    # Clip perturbation size
                    x = torch.clamp(
                        original + torch.clamp(x - original, -eps, eps), 0, 1
                    )

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

        result.poisoned_indices = indices.tolist()
        return dataset, result


class LabelFlipAttack(PoisonAttack):
    """Label Flipping Attack"""

    def poison_dataset(self, dataset: Dataset) -> Tuple[Dataset, PoisonResult]:
        result = PoisonResult(self.config)
        num_samples = len(dataset)
        num_poison = int(num_samples * self.config.poison_ratio)

        # Get all labels
        all_labels = [dataset[i][1] for i in range(num_samples)]
        unique_labels = list(set(all_labels))

        if self.config.poison_type == PoisonType.LABEL_FLIP_RANDOM_TO_RANDOM:
            indices = np.random.choice(num_samples, num_poison, replace=False)
            for idx in indices:
                current_label = dataset[idx][1]
                new_label = np.random.choice(
                    [l for l in unique_labels if l != current_label]
                )
                dataset.targets[idx] = new_label

        elif self.config.poison_type == PoisonType.LABEL_FLIP_RANDOM_TO_TARGET:
            if self.config.target_class is None:
                raise ValueError(
                    "Target class must be specified for random->target flipping"
                )
            indices = np.random.choice(num_samples, num_poison, replace=False)
            for idx in indices:
                dataset.targets[idx] = self.config.target_class

        elif self.config.poison_type == PoisonType.LABEL_FLIP_SOURCE_TO_TARGET:
            if None in (self.config.source_class, self.config.target_class):
                raise ValueError(
                    "Source and target classes must be specified for source->target flipping"
                )
            source_indices = [
                i
                for i, label in enumerate(all_labels)
                if label == self.config.source_class
            ]
            num_poison = min(num_poison, len(source_indices))
            indices = np.random.choice(source_indices, num_poison, replace=False)
            for idx in indices:
                dataset.targets[idx] = self.config.target_class

        result.poisoned_indices = indices.tolist()
        return dataset, result


def create_poison_attack(config: PoisonConfig, device: torch.device) -> PoisonAttack:
    """Create appropriate poison attack based on config."""
    if config.poison_type == PoisonType.PGD:
        return PGDPoisonAttack(config, device)
    elif config.poison_type == PoisonType.GA:
        return GradientAscentAttack(config)
    elif config.poison_type in [
        PoisonType.LABEL_FLIP_RANDOM_TO_RANDOM,
        PoisonType.LABEL_FLIP_RANDOM_TO_TARGET,
        PoisonType.LABEL_FLIP_SOURCE_TO_TARGET,
    ]:
        return LabelFlipAttack(config)
    else:
        raise ValueError(f"Unknown poison type: {config.poison_type}")


class PoisonExperiment:
    """Manages poisoning experiments"""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        test_dataset: Dataset,
        configs: List[PoisonConfig],
        output_dir: str = "poison_results",
        checkpoint_dir: str = "checkpoints",
        device: Optional[torch.device] = None,
        epochs: int = 30,
        learning_rate: float = 0.001,
        batch_size: int = 128,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.configs = configs
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.device = device if device is not None else get_device()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Move model to device
        self.model = self.model.to(self.device)

    def train_model(
        self,
        train_loader: DataLoader,
        epochs: int = 30,
        learning_rate: float = 0.001,
        checkpoint_name: Optional[str] = None,
    ) -> None:
        """Train model on data."""
        logger.info(
            f"Starting training for {epochs} epochs with learning rate {learning_rate}"
        )

        if self.device.type == "cuda":
            # Log initial GPU state
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / 1024**2
            logger.info(f"Initial GPU Memory: {initial_memory:.1f}MB")

            # Set memory limits to 90% of available memory
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
            memory_limit = int(total_memory * 0.9)  # 90% of total memory
            torch.cuda.set_per_process_memory_fraction(0.9, device=0)
            logger.info(
                f"Setting GPU memory limit to {memory_limit:.1f}MB out of {total_memory:.1f}MB"
            )

        # Initialize optimizer and criterion
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Try to load latest checkpoint if it exists
        start_epoch, best_loss = 0, float("inf")
        if checkpoint_name:
            start_epoch, best_loss = self.load_checkpoint(checkpoint_name, optimizer)

        patience = 5  # Number of epochs to wait before early stopping
        patience_counter = 0

        try:
            for epoch in range(start_epoch, epochs):
                self.model.train()
                running_loss = 0.0
                total_batches = len(train_loader)

                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    try:
                        inputs, targets = inputs.to(self.device), targets.to(
                            self.device
                        )

                        optimizer.zero_grad()
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()

                        if batch_idx % 10 == 0:
                            avg_loss = running_loss / (batch_idx + 1)
                            progress = (batch_idx + 1) / total_batches * 100
                            logger.debug(
                                f"Epoch {epoch+1}/{epochs} - {progress:.1f}% - Loss: {avg_loss:.4f}"
                            )

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            if checkpoint_name:
                                self.save_checkpoint(
                                    checkpoint_name,
                                    optimizer=optimizer,
                                    epoch=epoch,
                                    loss=running_loss / (batch_idx + 1),
                                    is_emergency=True,
                                )
                            if self.device.type == "cuda":
                                torch.cuda.empty_cache()
                            raise

                # Calculate average loss for the epoch
                epoch_loss = running_loss / len(train_loader)
                logger.info(
                    f"Epoch {epoch+1}/{epochs} complete - Avg Loss: {epoch_loss:.4f}"
                )

                # Save checkpoint
                if checkpoint_name:
                    is_best = epoch_loss < best_loss
                    self.save_checkpoint(
                        checkpoint_name,
                        optimizer=optimizer,
                        epoch=epoch,
                        loss=epoch_loss,
                        is_best=is_best,
                    )

                    if is_best:
                        best_loss = epoch_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                # Early stopping check
                if patience_counter >= patience:
                    logger.info(
                        f"Early stopping triggered after {patience} epochs without improvement"
                    )
                    break

                # Clear GPU cache at end of epoch
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            if checkpoint_name:
                self.save_checkpoint(
                    checkpoint_name,
                    optimizer=optimizer,
                    epoch=epoch if "epoch" in locals() else None,
                    loss=(
                        running_loss / len(train_loader)
                        if "running_loss" in locals()
                        else None
                    ),
                    is_emergency=True,
                )
            raise

    def save_checkpoint(
        self,
        name: str,
        optimizer=None,
        epoch: int = None,
        loss: float = None,
        is_best: bool = False,
        is_emergency: bool = False,
    ) -> None:
        """Save a model checkpoint.

        Args:
            name: Base name for the checkpoint
            optimizer: Optional optimizer state to save
            epoch: Optional epoch number
            loss: Optional loss value
            is_best: If True, also save as best checkpoint
            is_emergency: If True, save as emergency checkpoint
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Prepare checkpoint data
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "epoch": epoch,
            "loss": loss,
        }

        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        # Determine checkpoint path based on type
        if is_emergency:
            path = os.path.join(self.checkpoint_dir, f"{name}_emergency.pt")
        else:
            path = os.path.join(self.checkpoint_dir, f"{name}_latest.pt")

        # Save checkpoint
        torch.save(checkpoint, path)
        logger.info(
            f"Saved {'emergency' if is_emergency else 'latest'} checkpoint to {path}"
        )

        # Optionally save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f"{name}_best.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")

    def load_checkpoint(
        self, checkpoint_name: str, load_best: bool = False
    ) -> Tuple[int, float]:
        """Load a model checkpoint.

        Args:
            checkpoint_name: Base name of the checkpoint
            load_best: If True, load the best checkpoint instead of latest

        Returns:
            tuple: (start_epoch, best_loss)
                - start_epoch: Epoch to resume from (0 if not found)
                - best_loss: Best validation loss (inf if not found)

        Raises:
            ValueError: If checkpoint format is invalid
        """
        suffix = "_best.pt" if load_best else "_latest.pt"
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"{checkpoint_name}{suffix}"
        )

        if not os.path.exists(checkpoint_path):
            logger.warning(f"No checkpoint found at {checkpoint_path}")
            return 0, float("inf")

        try:
            checkpoint = torch.load(checkpoint_path)
            if not isinstance(checkpoint, dict):
                raise ValueError("Invalid checkpoint format")

            if "model_state_dict" not in checkpoint:
                raise ValueError("Checkpoint missing model state dict")

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model = self.model.to(self.device)

            epoch = checkpoint.get("epoch", -1)
            loss = checkpoint.get("loss", float("inf"))

            logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")
            return epoch + 1, loss

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return 0, float("inf")

    def evaluate_attack(
        self, poisoned_loader: DataLoader, clean_loader: DataLoader
    ) -> Tuple[float, float]:
        """Evaluate model on poisoned and clean data"""
        poisoned_acc = evaluate_model(self.model, poisoned_loader, self.device)
        clean_acc = evaluate_model(self.model, clean_loader, self.device)
        return poisoned_acc, clean_acc

    def extract_features(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from dataset using the model's feature extractor."""
        logger.debug(f"Starting feature extraction for dataset of size {len(dataset)}")
        self.model.eval()
        features = []
        labels = []
        loader = DataLoader(dataset, batch_size=128, pin_memory=True)
        total_batches = len(loader)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs = inputs.to(self.device)
                logger.debug(
                    f"Processing batch {batch_idx + 1}/{total_batches}, input shape: {inputs.shape}"
                )

                batch_features = self.model.extract_features(inputs).cpu().numpy()
                features.append(batch_features)
                labels.append(targets.numpy())

                if (batch_idx + 1) % 10 == 0:
                    logger.debug(
                        f"Processed {batch_idx + 1} batches, current features shape: {batch_features.shape}"
                    )

        features_array = np.vstack(features)
        labels_array = np.concatenate(labels)
        logger.debug(
            f"Feature extraction complete. Features shape: {features_array.shape}, Labels shape: {labels_array.shape}"
        )
        return features_array, labels_array

    def evaluate_classifiers(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        test_features: np.ndarray,
        test_labels: np.ndarray,
    ) -> Dict[str, float]:
        """Train and evaluate traditional classifiers."""
        logger.debug(
            f"Starting classifier evaluation with shapes - Train: {train_features.shape}, Test: {test_features.shape}"
        )

        # Normalize features
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)
        logger.debug("Features normalized")

        # Add PCA
        pca = PCA(n_components=0.95)
        train_features = pca.fit_transform(train_features)
        test_features = pca.transform(test_features)
        logger.debug(
            f"PCA applied. New feature dimensions - Train: {train_features.shape}, Test: {test_features.shape}"
        )
        logger.info(
            f"Reduced feature dimension to {train_features.shape[1]} components"
        )

        classifiers = {
            "knn": KNeighborsClassifier(
                n_neighbors=5, weights="distance", metric="cosine", n_jobs=-1
            ),
            "rf": RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            ),
            "svm": SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                class_weight="balanced",
                cache_size=1000,
                random_state=42,
            ),
            "lr": LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                C=1.0,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            ),
        }

        results = {}

        for name, clf in classifiers.items():
            logger.debug(f"Training {name.upper()} classifier...")
            try:
                clf.fit(train_features, train_labels)
                acc = clf.score(test_features, test_labels) * 100
                logger.info(f"{name.upper()} Accuracy: {acc:.2f}%")
                results[name] = acc
            except Exception as e:
                logger.error(f"Error training {name.upper()} classifier: {str(e)}")
                results[name] = 0.0

        return results

    def plot_combined_classifier_comparison(
        self, results: List[PoisonResult], output_dir: str
    ):
        """Plot combined classifier performance comparison across all datasets."""
        # Prepare data for plotting
        data = []
        for result in results:
            attack_type = result.config.poison_type.value
            poison_ratio = result.config.poison_ratio

            # Add clean dataset results
            for clf_name, acc in result.classifier_results_clean.items():
                data.append(
                    {
                        "Classifier": clf_name.upper(),
                        "Accuracy": acc,
                        "Dataset": f"{attack_type}_{poison_ratio}_clean",
                    }
                )

            # Add poisoned dataset results
            for clf_name, acc in result.classifier_results_poisoned.items():
                data.append(
                    {
                        "Classifier": clf_name.upper(),
                        "Accuracy": acc,
                        "Dataset": f"{attack_type}_{poison_ratio}_poisoned",
                    }
                )

        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(data)

        # Create the plot
        plt.figure(figsize=(15, 8))
        sns.barplot(data=df, x="Dataset", y="Accuracy", hue="Classifier")

        # Customize the plot
        plt.title("Classifier Performance Across All Datasets", fontsize=14, pad=20)
        plt.xlabel("Dataset (Attack_Type_Ratio_Status)", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Classifier", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(output_dir, "combined_classifier_comparison.png")
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()
        logger.info(f"Saved combined classifier comparison plot to {plot_path}")

    def plot_classifier_comparison(self, results: List[PoisonResult], output_dir: str):
        """Plot classifier performance comparison."""
        # Prepare data for plotting
        data = []
        for result in results:
            # Add both clean and poisoned results for each classifier
            for clf_name in ["knn", "rf", "svm", "lr"]:
                # Add clean results (default to 0 if not present)
                clean_acc = result.classifier_results_clean.get(clf_name, 0.0)
                data.append(
                    {
                        "Classifier": clf_name.upper(),
                        "Accuracy": clean_acc,
                        "Dataset": "Clean",
                        "Attack": result.config.poison_type.value,
                        "Poison Ratio": result.config.poison_ratio,
                    }
                )

                # Add poisoned results (default to 0 if not present)
                poisoned_acc = result.classifier_results_poisoned.get(clf_name, 0.0)
                data.append(
                    {
                        "Classifier": clf_name.upper(),
                        "Accuracy": poisoned_acc,
                        "Dataset": "Poisoned",
                        "Attack": result.config.poison_type.value,
                        "Poison Ratio": result.config.poison_ratio,
                    }
                )

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Create plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x="Classifier", y="Accuracy", hue="Dataset")
        plt.title("Classifier Performance Comparison")
        plt.ylabel("Accuracy (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(
            output_dir,
            f'classifier_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
        )
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Classifier comparison plot saved to {plot_path}")

        # Add the combined plot
        self.plot_combined_classifier_comparison(results, output_dir)

    def run_experiments(self):
        """Run all configured poisoning experiments."""
        for config in self.configs:
            logger.info(f"Running experiment with config: {config}")

            # Create attack instance
            attack = create_poison_attack(config, self.device)

            # Run poisoning attack
            poisoned_dataset, result = attack.poison_dataset(
                self.train_dataset, self.model
            )

            # Train model on poisoned data
            poisoned_loader = DataLoader(
                poisoned_dataset, batch_size=self.batch_size, shuffle=True
            )

            # Train with checkpoint support
            self.train_model(
                poisoned_loader,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                checkpoint_name=f"poisoned_{config.poison_type.value}",
            )

            # Save results
            result.save(self.output_dir)


def evaluate_model(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> float:
    """Evaluate model accuracy"""
    logger.debug("Starting model evaluation")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    logger.debug(
        f"Evaluation complete. Total samples: {total}, Correct predictions: {correct}"
    )
    logger.debug(f"Accuracy: {accuracy:.2f}%")
    return accuracy


def run_example():
    import argparse

    parser = argparse.ArgumentParser(description="Run poisoning experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar100", "gtsrb", "imagenette"],
    )
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Number of samples per class to use (default: None, use full dataset)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of worker processes for data loading (default: 2)",
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs (default: 30)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size (default: 128)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Device to use for training (default: best available)",
    )

    # Output parameters
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory (default: checkpoints)",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="clean_model",
        help="Name for model checkpoint (default: clean_model)",
    )

    args = parser.parse_args()

    # Create output directory
    dataset_output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(dataset_output_dir, exist_ok=True)

    # Get model and dataset
    model = get_model(args.dataset)
    train_loader, test_loader, train_dataset, test_dataset = get_dataset_loaders(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_size_per_class=args.subset_size,
    )

    logger.info(
        f"Dataset sizes - Train: {len(train_dataset)}, Test: {len(test_dataset)}"
    )

    # Define poisoning configurations for testing
    configs = [
        PoisonConfig(
            poison_type=PoisonType.PGD,
            poison_ratio=0.1,
            pgd_eps=0.3,
            pgd_alpha=0.01,
            pgd_steps=40,
            random_seed=42,
        )
    ]

    # Create and run experiment
    experiment = PoisonExperiment(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        configs=configs,
        output_dir=dataset_output_dir,
        checkpoint_dir=os.path.join(args.checkpoint_dir, args.dataset),
        device=get_device(args.device),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )

    experiment.run_experiments()


if __name__ == "__main__":
    run_example()
