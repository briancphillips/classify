import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable
import logging
import os
import random
import json
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from enum import Enum
from dataclasses import dataclass
import torch.optim as optim
from tqdm import tqdm
import argparse
from models import get_model, save_model, load_model, get_dataset_loaders
import torchvision
from torchvision import datasets, transforms
import copy
from PIL import Image

def setup_logging():
    """Configure logging settings."""
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # Set PIL logging to INFO to suppress debug messages
    logging.getLogger('PIL').setLevel(logging.INFO)
    
    # Get our logger
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

class PoisonType(Enum):
    """Types of poisoning attacks"""
    PGD = "pgd"  # Projected Gradient Descent
    GA = "ga"    # Genetic Algorithm
    LABEL_FLIP_RANDOM_TO_RANDOM = "label_flip_random_random"
    LABEL_FLIP_RANDOM_TO_TARGET = "label_flip_random_target"
    LABEL_FLIP_SOURCE_TO_TARGET = "label_flip_source_target"

@dataclass
class PoisonConfig:
    """Configuration for poisoning attacks"""
    poison_type: PoisonType
    poison_ratio: float  # Percentage of dataset to poison (0.0 to 1.0)
    # PGD specific parameters
    pgd_eps: Optional[float] = 0.3        # Epsilon for PGD attack
    pgd_alpha: Optional[float] = 0.01     # Step size for PGD
    pgd_steps: Optional[int] = 40         # Number of PGD steps
    # GA specific parameters
    ga_pop_size: Optional[int] = 50       # Population size for GA
    ga_generations: Optional[int] = 100    # Number of generations
    ga_mutation_rate: Optional[float] = 0.1
    # Label flipping specific parameters
    source_class: Optional[int] = None     # Source class for source->target flipping
    target_class: Optional[int] = None     # Target class for targeted flipping
    random_seed: Optional[int] = 42        # Random seed for reproducibility

class PoisonResult:
    """Stores results of poisoning attack"""
    def __init__(self, config: PoisonConfig):
        self.config = config
        self.original_accuracy: float = 0.0
        self.poisoned_accuracy: float = 0.0
        self.poison_success_rate: float = 0.0
        self.poisoned_indices: List[int] = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.classifier_results_clean: Dict[str, float] = {}
        self.classifier_results_poisoned: Dict[str, float] = {}

    def to_dict(self) -> Dict:
        """Convert results to dictionary for logging"""
        # Convert config to dict and handle PoisonType enum
        config_dict = {}
        for key, value in self.config.__dict__.items():
            if isinstance(value, PoisonType):
                config_dict[key] = value.value
            else:
                config_dict[key] = value
                
        return {
            "poison_type": self.config.poison_type.value,
            "poison_ratio": self.config.poison_ratio,
            "original_accuracy": self.original_accuracy,
            "poisoned_accuracy": self.poisoned_accuracy,
            "poison_success_rate": self.poison_success_rate,
            "timestamp": self.timestamp,
            "config": config_dict,
            "classifier_results_clean": self.classifier_results_clean,
            "classifier_results_poisoned": self.classifier_results_poisoned
        }
    
    def save(self, output_dir: str):
        """Save results to JSON file"""
        os.makedirs(output_dir, exist_ok=True)
        filename = f"poison_results_{self.timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
        logger.info(f"Results saved to {filepath}")

class PoisonAttack:
    """Base class for poison attacks"""
    def __init__(self, config: PoisonConfig):
        self.config = config
        if config.random_seed is not None:
            torch.manual_seed(config.random_seed)
            np.random.seed(config.random_seed)
            random.seed(config.random_seed)
    
    def poison_dataset(self, dataset: Dataset) -> Tuple[Dataset, PoisonResult]:
        """Poison a dataset according to configuration"""
        raise NotImplementedError

def get_device(device_str: Optional[str] = None) -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU)"""
    if device_str:
        if device_str == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif device_str == 'mps' and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using MPS device")
        elif device_str == 'cpu':
            device = torch.device('cpu')
            logger.info("Using CPU device")
        else:
            logger.warning(f"Requested device '{device_str}' not available, falling back to best available device")
            return get_device()
        return device

    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS device")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    return device

class PGDPoisonAttack(PoisonAttack):
    """Projected Gradient Descent Attack"""
    
    def __init__(self, config: PoisonConfig, device: torch.device):
        super().__init__(config)
        self.device = device

    def poison_dataset(self, original_dataset: datasets.ImageFolder) -> Tuple[datasets.ImageFolder, PoisonResult]:
        """Poison a dataset using PGD attack."""
        logger.info("Starting PGD poisoning attack")
        
        # Calculate number of samples to poison
        num_samples = len(original_dataset)
        num_poison = int(num_samples * self.config.poison_ratio)
        logger.info(f"Poisoning {num_poison} out of {num_samples} samples")
        
        # Create a new dataset with the same transforms
        poisoned_dataset = copy.deepcopy(original_dataset)
        
        # Randomly select indices to poison
        all_indices = list(range(num_samples))
        random.shuffle(all_indices)
        poison_indices = all_indices[:num_poison]
        
        # Apply PGD attack to selected samples
        for idx in poison_indices:
            # Get the image path and label
            if hasattr(dataset, 'imgs'):  # ImageFolder dataset
                img_path = dataset.imgs[idx][0]
                label = dataset.imgs[idx][1]
            else:  # GTSRB or similar dataset
                img_path = dataset.samples[idx][0]
                label = dataset.samples[idx][1]
            
            # Load the image
            img = Image.open(img_path)
            
            # Convert to tensor and add batch dimension
            img = transforms.ToTensor()(img)
            img = img.unsqueeze(0).to(self.device)
            
            # Perform PGD attack
            perturbed_img = self._pgd_attack(img)
            
            # Convert back to PIL image
            perturbed_img = transforms.ToPILImage()(perturbed_img.squeeze().cpu())
            
            # Replace the image in the dataset
            # Store the path and update the image
            if hasattr(poisoned_dataset, 'imgs'):
                poisoned_dataset.imgs[idx] = (img_path, label)
                poisoned_dataset.samples[idx] = (img_path, label)
            else:
                poisoned_dataset.samples[idx] = (img_path, label)
            
            # Update the cached image if it exists
            if hasattr(poisoned_dataset, 'cache'):
                poisoned_dataset.cache[img_path] = perturbed_img
        
        result = PoisonResult(self.config)
        result.poisoned_indices = poison_indices
        result.poison_success_rate = 1.0  # We'll update this after evaluation
        
        return poisoned_dataset, result

    def _pgd_attack(self, image: torch.Tensor) -> torch.Tensor:
        """Perform PGD attack on a single image."""
        # Clone the image and initialize perturbation
        perturbed = image.clone().detach().requires_grad_(True)
        
        for step in range(self.config.pgd_steps):
            # Forward pass
            loss = -torch.norm(perturbed - image)  # Maximize L2 distance
            
            # Backward pass
            loss.backward()
            
            # Update perturbed image
            with torch.no_grad():
                grad_sign = perturbed.grad.sign()
                perturbed.data = perturbed.data + self.config.pgd_alpha * grad_sign
                
                # Project back to epsilon ball
                delta = perturbed.data - image
                delta = torch.clamp(delta, -self.config.pgd_eps, self.config.pgd_eps)
                perturbed.data = torch.clamp(image + delta, 0, 1)
                
                # Reset gradients
                perturbed.grad.zero_()
        
        return perturbed.detach()

class GeneticAttack(PoisonAttack):
    """Genetic Algorithm Attack"""
    def _fitness_function(self, x: torch.Tensor, original: torch.Tensor) -> float:
        """Compute fitness score for a candidate solution"""
        # Example fitness: balance between perturbation size and visual similarity
        perturbation_size = torch.norm(x - original)
        visual_similarity = -torch.mean((x - original) ** 2)
        return float(-perturbation_size + 0.5 * visual_similarity)
    
    def _crossover(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Perform crossover between two parents"""
        mask = torch.rand_like(x1) < 0.5
        return torch.where(mask, x1, x2)
    
    def _mutate(self, x: torch.Tensor, mutation_rate: float, eps: float) -> torch.Tensor:
        """Apply mutation to an individual"""
        mask = torch.rand_like(x) < mutation_rate
        noise = torch.randn_like(x) * eps
        return torch.clamp(x + mask.float() * noise, 0, 1)
    
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
        
        # GA parameters
        pop_size = self.config.ga_pop_size
        num_generations = self.config.ga_generations
        mutation_rate = self.config.ga_mutation_rate
        eps = 0.1  # Mutation strength
        
        for idx in indices:
            original = poisoned_data[idx:idx+1].to(device)
            
            # Initialize population
            population = [original + torch.randn_like(original) * eps for _ in range(pop_size)]
            population = [torch.clamp(p, 0, 1) for p in population]
            
            # Evolution loop
            for _ in range(num_generations):
                # Evaluate fitness
                fitness_scores = [self._fitness_function(p, original) for p in population]
                
                # Selection (tournament selection)
                new_population = []
                for _ in range(pop_size):
                    idx1, idx2 = np.random.choice(pop_size, 2, replace=False)
                    winner = population[idx1] if fitness_scores[idx1] > fitness_scores[idx2] else population[idx2]
                    new_population.append(winner)
                
                # Crossover and mutation
                offspring = []
                for i in range(0, pop_size, 2):
                    p1, p2 = new_population[i], new_population[min(i+1, pop_size-1)]
                    c1 = self._crossover(p1, p2)
                    c2 = self._crossover(p2, p1)
                    c1 = self._mutate(c1, mutation_rate, eps)
                    c2 = self._mutate(c2, mutation_rate, eps)
                    offspring.extend([c1, c2])
                
                population = offspring[:pop_size]
            
            # Select best individual
            fitness_scores = [self._fitness_function(p, original) for p in population]
            best_idx = np.argmax(fitness_scores)
            poisoned_data[idx] = population[best_idx]
        
        # Update dataset
        if hasattr(dataset, 'data'):
            if isinstance(dataset.data, np.ndarray):
                poisoned_data = (poisoned_data.cpu().numpy() * 255).astype(np.uint8)
            dataset.data = poisoned_data.cpu() if isinstance(poisoned_data, torch.Tensor) else poisoned_data
        
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
                new_label = np.random.choice([l for l in unique_labels if l != current_label])
                dataset.targets[idx] = new_label
        
        elif self.config.poison_type == PoisonType.LABEL_FLIP_RANDOM_TO_TARGET:
            if self.config.target_class is None:
                raise ValueError("Target class must be specified for random->target flipping")
            indices = np.random.choice(num_samples, num_poison, replace=False)
            for idx in indices:
                dataset.targets[idx] = self.config.target_class
        
        elif self.config.poison_type == PoisonType.LABEL_FLIP_SOURCE_TO_TARGET:
            if None in (self.config.source_class, self.config.target_class):
                raise ValueError("Source and target classes must be specified for source->target flipping")
            source_indices = [i for i, label in enumerate(all_labels) if label == self.config.source_class]
            num_poison = min(num_poison, len(source_indices))
            indices = np.random.choice(source_indices, num_poison, replace=False)
            for idx in indices:
                dataset.targets[idx] = self.config.target_class
        
        result.poisoned_indices = indices.tolist()
        return dataset, result

def create_poison_attack(config: PoisonConfig, device: torch.device) -> PoisonAttack:
    """Factory function to create poison attacks"""
    if config.poison_type == PoisonType.PGD:
        return PGDPoisonAttack(config, device)
    elif config.poison_type == PoisonType.GA:
        return GeneticAttack(config)
    elif config.poison_type.value.startswith("label_flip"):
        return LabelFlipAttack(config)
    else:
        raise ValueError(f"Unknown poison type: {config.poison_type}")

class PoisonExperiment:
    """Manages poisoning experiments"""
    def __init__(self, 
                 model: nn.Module,
                 train_dataset: Dataset,
                 test_dataset: Dataset,
                 configs: List[PoisonConfig],
                 output_dir: str = "poison_results",
                 checkpoint_dir: str = "checkpoints",
                 device: Optional[torch.device] = None,
                 epochs: int = 30,
                 learning_rate: float = 0.001,
                 batch_size: int = 128):
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
    
    def train_model(self, 
                   train_loader: DataLoader,
                   epochs: int = 30,
                   learning_rate: float = 0.001,
                   checkpoint_name: Optional[str] = None) -> None:
        """Train model on data."""
        logger.info(f"Starting training for {epochs} epochs with learning rate {learning_rate}")
        
        if self.device.type == 'cuda':
            # Log initial GPU state
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / 1024**2
            logger.info(f"Initial GPU Memory: {initial_memory:.1f}MB")
            
            # Set memory limits to 90% of available memory
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
            memory_limit = int(total_memory * 0.9)  # 90% of total memory
            torch.cuda.set_per_process_memory_fraction(0.9, device=0)
            logger.info(f"Setting GPU memory limit to {memory_limit:.1f}MB out of {total_memory:.1f}MB")
        
        # Initialize optimizer and criterion
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Try to load latest checkpoint if it exists
        start_epoch = 0
        if checkpoint_name:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}_latest.pth")
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    logger.info(f"Resuming from epoch {start_epoch}")
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint: {e}")

        best_loss = float('inf')
        patience = 5  # Number of epochs to wait before early stopping
        patience_counter = 0
        
        try:
            for epoch in range(start_epoch, epochs):
                self.model.train()
                running_loss = 0.0
                total_batches = len(train_loader)
                
                # Log GPU memory usage at start of epoch
                if self.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                    memory_allocated = torch.cuda.memory_allocated() / 1024**2
                    memory_reserved = torch.cuda.memory_reserved() / 1024**2
                    logger.info(f"Epoch {epoch+1} Start - GPU Memory: Allocated={memory_allocated:.1f}MB, Reserved={memory_reserved:.1f}MB")
                
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    try:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                        
                        running_loss += loss.item()
                        
                        if batch_idx % 10 == 0:
                            avg_loss = running_loss / (batch_idx + 1)
                            progress = (batch_idx + 1) / total_batches * 100
                            
                            if self.device.type == 'cuda':
                                peak_memory = torch.cuda.max_memory_allocated() / 1024**2
                                current_memory = torch.cuda.memory_allocated() / 1024**2
                                logger.debug(f"Epoch {epoch+1}/{epochs} - {progress:.1f}% - Loss: {avg_loss:.4f} - Current GPU Memory: {current_memory:.1f}MB, Peak: {peak_memory:.1f}MB")
                            else:
                                logger.debug(f"Epoch {epoch+1}/{epochs} - {progress:.1f}% - Loss: {avg_loss:.4f}")
                        
                        # Clear GPU cache periodically
                        if self.device.type == 'cuda' and batch_idx % 50 == 0:
                            torch.cuda.empty_cache()
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            # Save checkpoint before clearing memory
                            if checkpoint_name:
                                emergency_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}_emergency.pth")
                                torch.save({
                                    'epoch': epoch,
                                    'batch_idx': batch_idx,
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'loss': running_loss / (batch_idx + 1),
                                }, emergency_path)
                                logger.warning(f"Saved emergency checkpoint to {emergency_path}")
                            
                            if self.device.type == 'cuda':
                                torch.cuda.empty_cache()
                                current_memory = torch.cuda.memory_allocated() / 1024**2
                                peak_memory = torch.cuda.max_memory_allocated() / 1024**2
                                logger.error(f"GPU OOM: Current={current_memory:.1f}MB, Peak={peak_memory:.1f}MB")
                            raise
                
                # Calculate average loss for the epoch
                epoch_loss = running_loss / len(train_loader)
                logger.info(f"Epoch {epoch+1}/{epochs} complete - Avg Loss: {epoch_loss:.4f}")
                
                # Log GPU stats at end of epoch
                if self.device.type == 'cuda':
                    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
                    current_memory = torch.cuda.memory_allocated() / 1024**2
                    logger.info(f"Epoch {epoch+1} End - GPU Memory: Current={current_memory:.1f}MB, Peak={peak_memory:.1f}MB")
                    torch.cuda.reset_peak_memory_stats()
                
                # Save checkpoint
                if checkpoint_name:
                    # Save latest checkpoint
                    latest_checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss,
                    }
                    latest_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}_latest.pth")
                    torch.save(latest_checkpoint, latest_path)
                    
                    # Save best checkpoint if this is the best loss
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}_best.pth")
                        torch.save(latest_checkpoint, best_path)
                        patience_counter = 0
                    else:
                        patience_counter += 1
                
                # Early stopping check
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                    break
                
                # Force GPU cache clear at end of epoch
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            if checkpoint_name:
                # Save emergency checkpoint
                emergency_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}_emergency.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': running_loss / len(train_loader) if 'running_loss' in locals() else None,
                }, emergency_path)
                logger.warning(f"Saved emergency checkpoint to {emergency_path}")
            raise
    
    def load_checkpoint(self, checkpoint_name: str) -> dict:
        """Load a model checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pt")
        self.model, metadata = load_model(checkpoint_path)
        self.model = self.model.to(self.device)
        return metadata

    def evaluate_attack(self, 
                       poisoned_loader: DataLoader,
                       clean_loader: DataLoader) -> Tuple[float, float]:
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
                logger.debug(f"Processing batch {batch_idx + 1}/{total_batches}, input shape: {inputs.shape}")
                
                batch_features = self.model.extract_features(inputs).cpu().numpy()
                features.append(batch_features)
                labels.append(targets.numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    logger.debug(f"Processed {batch_idx + 1} batches, current features shape: {batch_features.shape}")

        features_array = np.vstack(features)
        labels_array = np.concatenate(labels)
        logger.debug(f"Feature extraction complete. Features shape: {features_array.shape}, Labels shape: {labels_array.shape}")
        return features_array, labels_array

    def evaluate_classifiers(self, 
                           train_features: np.ndarray, 
                           train_labels: np.ndarray,
                           test_features: np.ndarray, 
                           test_labels: np.ndarray) -> Dict[str, float]:
        """Train and evaluate traditional classifiers."""
        logger.debug(f"Starting classifier evaluation with shapes - Train: {train_features.shape}, Test: {test_features.shape}")
        
        # Normalize features
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)
        logger.debug("Features normalized")
        
        # Add PCA
        pca = PCA(n_components=0.95)
        train_features = pca.fit_transform(train_features)
        test_features = pca.transform(test_features)
        logger.debug(f"PCA applied. New feature dimensions - Train: {train_features.shape}, Test: {test_features.shape}")
        logger.info(f"Reduced feature dimension to {train_features.shape[1]} components")
        
        classifiers = {
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='cosine',
                n_jobs=-1
            ),
            'rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                class_weight='balanced',
                cache_size=1000,
                random_state=42
            ),
            'lr': LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                C=1.0,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            )
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

    def plot_combined_classifier_comparison(self, results: List[PoisonResult], output_dir: str):
        """Plot combined classifier performance comparison across all datasets."""
        # Prepare data for plotting
        data = []
        for result in results:
            attack_type = result.config.poison_type.value
            poison_ratio = result.config.poison_ratio
            
            # Add clean dataset results
            for clf_name, acc in result.classifier_results_clean.items():
                data.append({
                    'Classifier': clf_name.upper(),
                    'Accuracy': acc,
                    'Dataset': f"{attack_type}_{poison_ratio}_clean"
                })
            
            # Add poisoned dataset results
            for clf_name, acc in result.classifier_results_poisoned.items():
                data.append({
                    'Classifier': clf_name.upper(),
                    'Accuracy': acc,
                    'Dataset': f"{attack_type}_{poison_ratio}_poisoned"
                })
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(data)
        
        # Create the plot
        plt.figure(figsize=(15, 8))
        sns.barplot(data=df, x='Dataset', y='Accuracy', hue='Classifier')
        
        # Customize the plot
        plt.title('Classifier Performance Across All Datasets', fontsize=14, pad=20)
        plt.xlabel('Dataset (Attack_Type_Ratio_Status)', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Classifier', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(output_dir, 'combined_classifier_comparison.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Saved combined classifier comparison plot to {plot_path}")

    def plot_classifier_comparison(self, results: List[PoisonResult], output_dir: str):
        """Plot classifier performance comparison."""
        # Prepare data for plotting
        data = []
        for result in results:
            # Add both clean and poisoned results for each classifier
            for clf_name in ['knn', 'rf', 'svm', 'lr']:
                # Add clean results (default to 0 if not present)
                clean_acc = result.classifier_results_clean.get(clf_name, 0.0)
                data.append({
                    'Classifier': clf_name.upper(),
                    'Accuracy': clean_acc,
                    'Dataset': 'Clean',
                    'Attack': result.config.poison_type.value,
                    'Poison Ratio': result.config.poison_ratio
                })
                
                # Add poisoned results (default to 0 if not present)
                poisoned_acc = result.classifier_results_poisoned.get(clf_name, 0.0)
                data.append({
                    'Classifier': clf_name.upper(),
                    'Accuracy': poisoned_acc,
                    'Dataset': 'Poisoned',
                    'Attack': result.config.poison_type.value,
                    'Poison Ratio': result.config.poison_ratio
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Classifier', y='Accuracy', hue='Dataset')
        plt.title('Classifier Performance Comparison')
        plt.ylabel('Accuracy (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'classifier_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Classifier comparison plot saved to {plot_path}")
        
        # Add the combined plot
        self.plot_combined_classifier_comparison(results, output_dir)
    
    def run_experiments(self) -> List[PoisonResult]:
        """Run all poisoning experiments."""
        results = []
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, pin_memory=True)
        clean_train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        logger.debug(f"Dataset sizes - Train: {len(self.train_dataset)}, Test: {len(self.test_dataset)}")
        
        # Train and save clean model first
        logger.info("Training clean model")
        clean_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_checkpoint_name = f"clean_model_{clean_timestamp}"
        self.train_model(
            clean_train_loader,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            checkpoint_name=clean_checkpoint_name
        )
        
        # Get clean model accuracy
        clean_acc = evaluate_model(self.model, test_loader, self.device)
        logger.info(f"Clean model accuracy: {clean_acc:.2f}%")
        
        # Extract features from clean training and test data
        logger.info("Extracting features from clean data")
        train_features, train_labels = self.extract_features(self.train_dataset)
        test_features, test_labels = self.extract_features(self.test_dataset)
        
        for config in self.configs:
            logger.info(f"Running experiment with config: {config}")
            
            # Create and apply poison attack
            attack = create_poison_attack(config, self.device)
            poisoned_dataset, result = attack.poison_dataset(self.train_dataset)
            logger.debug(f"Created poisoned training dataset with {len(poisoned_dataset)} samples")
            
            # Create poisoned test dataset
            poisoned_test_dataset, _ = attack.poison_dataset(self.test_dataset)
            poisoned_test_loader = DataLoader(poisoned_test_dataset, batch_size=self.batch_size, pin_memory=True)
            logger.debug(f"Created poisoned test dataset with {len(poisoned_test_dataset)} samples")
            
            # Create poisoned dataloader
            poisoned_loader = DataLoader(poisoned_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
            
            # Train model on poisoned data
            checkpoint_name = f"poisoned_model_{config.poison_type.value}_{config.poison_ratio}_{result.timestamp}"
            logger.info(f"Training model with checkpoint name: {checkpoint_name}")
            self.train_model(
                poisoned_loader,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                checkpoint_name=checkpoint_name
            )
            
            # Evaluate neural network results
            logger.info("Evaluating model on poisoned and clean data")
            poisoned_acc, clean_acc = self.evaluate_attack(poisoned_test_loader, test_loader)
            result.original_accuracy = clean_acc
            result.poisoned_accuracy = poisoned_acc
            result.poison_success_rate = 1.0 - (poisoned_acc / clean_acc)
            
            # Extract features from poisoned test data
            logger.info("Extracting features from poisoned test data")
            poisoned_test_features, poisoned_test_labels = self.extract_features(poisoned_test_dataset)
            
            # Evaluate traditional classifiers on clean data
            logger.info("Evaluating classifiers on clean data")
            result.classifier_results_clean = self.evaluate_classifiers(
                train_features, train_labels,
                test_features, test_labels
            )
            
            # Evaluate traditional classifiers on poisoned data
            logger.info("Evaluating classifiers on poisoned data")
            result.classifier_results_poisoned = self.evaluate_classifiers(
                train_features, train_labels,
                poisoned_test_features, poisoned_test_labels
            )
            
            results.append(result)
            
            # Save results
            try:
                result.save(self.output_dir)
                logger.debug(f"Results saved to {self.output_dir}")
            except Exception as e:
                logger.error(f"Error saving results: {str(e)}")
            
            logger.info("Attack Results Summary:")
            logger.info(f"Original Accuracy: {clean_acc:.2f}%")
            logger.info(f"Poisoned Accuracy: {poisoned_acc:.2f}%")
            logger.info(f"Attack Success Rate: {result.poison_success_rate:.2f}")
            logger.info("Traditional Classifier Results (Clean):")
            for clf_name, acc in result.classifier_results_clean.items():
                logger.info(f"  {clf_name.upper()}: {acc:.2f}%")
            logger.info("Traditional Classifier Results (Poisoned):")
            for clf_name, acc in result.classifier_results_poisoned.items():
                logger.info(f"  {clf_name.upper()}: {acc:.2f}%")
        
        # Plot classifier comparison
        try:
            self.plot_classifier_comparison(results, self.output_dir)
            logger.debug("Classifier comparison plot saved")
        except Exception as e:
            logger.error(f"Error creating classifier comparison plot: {str(e)}")
        
        return results

def evaluate_model(model: nn.Module, 
                  dataloader: DataLoader,
                  device: torch.device) -> float:
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
    logger.debug(f"Evaluation complete. Total samples: {total}, Correct predictions: {correct}")
    logger.debug(f"Accuracy: {accuracy:.2f}%")
    return accuracy

def run_example():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run poisoning experiments')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'gtsrb', 'imagenette'])
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--subset-size', type=int, default=None, 
                      help='Number of samples per class to use (default: None, use full dataset)')
    parser.add_argument('--num-workers', type=int, default=2,
                      help='Number of worker processes for data loading (default: 2)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
                      help='Number of epochs (default: 30)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=128,
                      help='Batch size (default: 128)')
    parser.add_argument('--device', type=str, choices=['cuda', 'mps', 'cpu'], default=None,
                      help='Device to use for training (default: best available)')
    
    # Output parameters
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                      help='Checkpoint directory (default: checkpoints)')
    parser.add_argument('--checkpoint-name', type=str, default='clean_model',
                      help='Name for model checkpoint (default: clean_model)')
    
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
        subset_size_per_class=args.subset_size
    )
    
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Define poisoning configurations for testing
    configs = [
        PoisonConfig(
            poison_type=PoisonType.PGD,
            poison_ratio=0.1,
            pgd_eps=0.3,
            pgd_alpha=0.01,
            pgd_steps=40,
            random_seed=42
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
        batch_size=args.batch_size
    )
    
    experiment.run_experiments()

if __name__ == "__main__":
    run_example()
