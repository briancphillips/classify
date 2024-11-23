import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable
import logging
from enum import Enum
import json
from datetime import datetime
import os
from dataclasses import dataclass
import random
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import argparse
from models import get_model, save_model, load_model
import torchvision
from torchvision import datasets, transforms
import copy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
            "config": config_dict
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

def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

class PGDAttack(PoisonAttack):
    """Projected Gradient Descent Attack"""
    
    def poison_dataset(self, dataset: Dataset) -> Tuple[Dataset, PoisonResult]:
        """Poison a dataset using PGD attack"""
        result = PoisonResult(self.config)
        
        # Create a copy of the dataset
        poisoned_dataset = copy.deepcopy(dataset)
        num_samples = len(dataset)
        num_poison = int(num_samples * self.config.poison_ratio)
        indices = np.random.choice(num_samples, num_poison, replace=False)
        
        # Get device
        device = get_device()
        
        # Create a simple model for generating adversarial examples
        temp_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*32*32, 100)  # Assuming CIFAR100 dimensions
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        
        for idx in tqdm(indices, desc="Generating poisoned samples"):
            x, y = dataset[idx]
            x = x.unsqueeze(0).to(device)  # Add batch dimension
            y = torch.tensor([y]).to(device)
            
            # Initialize delta randomly
            delta = torch.zeros_like(x, requires_grad=True)
            
            # PGD attack loop
            for _ in range(self.config.pgd_steps):
                # Forward pass
                x_adv = x + delta
                output = temp_model(x_adv)
                loss = criterion(output, y)
                
                # Backward pass
                loss.backward()
                
                # Update delta
                grad = delta.grad.sign()
                delta.data = delta.data + self.config.pgd_alpha * grad
                delta.data = torch.clamp(delta.data, -self.config.pgd_eps, self.config.pgd_eps)
                delta.data = torch.clamp(x + delta.data, 0, 1) - x
                
                # Reset gradients
                delta.grad.zero_()
            
            # Apply the perturbation
            x_poisoned = torch.clamp(x + delta.detach(), 0, 1)
            
            # Convert from (C,H,W) to (H,W,C) for dataset storage
            x_poisoned = x_poisoned.squeeze().permute(1, 2, 0).cpu().numpy()
            poisoned_dataset.data[idx] = x_poisoned
            result.poisoned_indices.append(idx)
        
        return poisoned_dataset, result

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

def create_poison_attack(config: PoisonConfig) -> PoisonAttack:
    """Factory function to create poison attacks"""
    if config.poison_type == PoisonType.PGD:
        return PGDAttack(config)
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
                 device: Optional[torch.device] = None):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.configs = configs
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.device = device if device is not None else get_device()
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
    
    def train_model(self, 
                   train_loader: DataLoader,
                   epochs: int = 30,
                   learning_rate: float = 0.001,
                   checkpoint_name: Optional[str] = None):
        """Train model on poisoned data"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in tqdm(range(epochs), desc="Training"):
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Save checkpoint if name provided
        if checkpoint_name:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pt")
            save_model(self.model, checkpoint_path)
    
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
    
    def run_experiments(self) -> List[PoisonResult]:
        """Run all poisoning experiments"""
        results = []
        
        # Create dataloaders
        train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=128)
        
        # Get clean model accuracy
        clean_acc = evaluate_model(self.model, test_loader, self.device)
        logger.info(f"Clean model accuracy: {clean_acc:.2f}%")
        
        for config in self.configs:
            logger.info(f"Running experiment with config: {config}")
            
            # Create and apply poison attack
            attack = create_poison_attack(config)
            poisoned_dataset, result = attack.poison_dataset(self.train_dataset)
            
            # Create poisoned dataloader
            poisoned_loader = DataLoader(poisoned_dataset, batch_size=128, shuffle=True)
            
            # Train model on poisoned data
            checkpoint_name = f"poisoned_model_{config.poison_type.value}_{config.poison_ratio}_{result.timestamp}"
            self.train_model(poisoned_loader, checkpoint_name=checkpoint_name)
            
            # Evaluate results
            poisoned_acc, clean_acc = self.evaluate_attack(test_loader, test_loader)
            result.original_accuracy = clean_acc
            result.poisoned_accuracy = poisoned_acc
            result.poison_success_rate = 1.0 - (poisoned_acc / clean_acc)
            
            results.append(result)
            result.save(self.output_dir)
            
            logger.info(f"Attack Results:")
            logger.info(f"Original Accuracy: {clean_acc:.2f}%")
            logger.info(f"Poisoned Accuracy: {poisoned_acc:.2f}%")
            logger.info(f"Attack Success Rate: {result.poison_success_rate:.2f}")
        
        return results

def evaluate_model(model: nn.Module, 
                  dataloader: DataLoader,
                  device: torch.device) -> float:
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total

def run_example():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run poisoning experiments')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'gtsrb', 'imagenette'],
                      help='Dataset to use (default: cifar100)')
    parser.add_argument('--attack', type=str, default='pgd', 
                      choices=['pgd', 'ga', 'label_flip_random_random', 'label_flip_random_target', 'label_flip_source_target'],
                      help='Attack type (default: pgd)')
    parser.add_argument('--poison-ratio', type=float, default=0.1,
                      help='Poison ratio (default: 0.1)')
    parser.add_argument('--random-seed', type=int, default=42,
                      help='Random seed (default: 42)')
    
    # PGD specific args
    parser.add_argument('--pgd-eps', type=float, default=0.3,
                      help='PGD epsilon (default: 0.3)')
    parser.add_argument('--pgd-alpha', type=float, default=0.01,
                      help='PGD step size (default: 0.01)')
    parser.add_argument('--pgd-steps', type=int, default=40,
                      help='PGD number of steps (default: 40)')
    
    # GA specific args
    parser.add_argument('--ga-pop-size', type=int, default=50,
                      help='GA population size (default: 50)')
    parser.add_argument('--ga-generations', type=int, default=100,
                      help='GA number of generations (default: 100)')
    parser.add_argument('--ga-mutation-rate', type=float, default=0.1,
                      help='GA mutation rate (default: 0.1)')
    
    # Label flipping specific args
    parser.add_argument('--source-class', type=int, default=None,
                      help='Source class for label flipping (default: None)')
    parser.add_argument('--target-class', type=int, default=None,
                      help='Target class for label flipping (default: None)')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=30,
                      help='Number of epochs (default: 30)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=128,
                      help='Batch size (default: 128)')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of data loading workers (default: 4)')
    
    # Output args
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Output directory (default: results/[dataset])')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                      help='Checkpoint directory (default: checkpoints/[dataset])')
    parser.add_argument('--checkpoint-name', type=str, default=None,
                      help='Name of checkpoint to save/load')
    parser.add_argument('--load-checkpoint', action='store_true',
                      help='Load model from checkpoint instead of training')
    
    args = parser.parse_args()
    
    # Set default directories if not specified
    if args.output_dir is None:
        args.output_dir = f"results/{args.dataset}"
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"checkpoints/{args.dataset}"
    if args.checkpoint_name is None:
        args.checkpoint_name = "clean_model"
    
    # Create attack config based on args
    if args.attack == 'pgd':
        config = PoisonConfig(
            poison_type=PoisonType.PGD,
            poison_ratio=args.poison_ratio,
            pgd_eps=args.pgd_eps,
            pgd_alpha=args.pgd_alpha,
            pgd_steps=args.pgd_steps,
            random_seed=args.random_seed
        )
    elif args.attack == 'ga':
        config = PoisonConfig(
            poison_type=PoisonType.GA,
            poison_ratio=args.poison_ratio,
            ga_pop_size=args.ga_pop_size,
            ga_generations=args.ga_generations,
            ga_mutation_rate=args.ga_mutation_rate,
            random_seed=args.random_seed
        )
    elif args.attack == 'label_flip_random_random':
        config = PoisonConfig(
            poison_type=PoisonType.LABEL_FLIP_RANDOM_TO_RANDOM,
            poison_ratio=args.poison_ratio,
            random_seed=args.random_seed
        )
    elif args.attack == 'label_flip_random_target':
        if args.target_class is None:
            raise ValueError("Must specify --target-class for label_flip_random_target attack")
        config = PoisonConfig(
            poison_type=PoisonType.LABEL_FLIP_RANDOM_TO_TARGET,
            poison_ratio=args.poison_ratio,
            target_class=args.target_class,
            random_seed=args.random_seed
        )
    elif args.attack == 'label_flip_source_target':
        if args.source_class is None or args.target_class is None:
            raise ValueError("Must specify both --source-class and --target-class for label_flip_source_target attack")
        config = PoisonConfig(
            poison_type=PoisonType.LABEL_FLIP_SOURCE_TO_TARGET,
            poison_ratio=args.poison_ratio,
            source_class=args.source_class,
            target_class=args.target_class,
            random_seed=args.random_seed
        )
    
    # Create model and load dataset
    model = get_model(args.dataset)
    
    # Get appropriate transforms
    if args.dataset == 'cifar100':
        from models import CIFAR100_TRANSFORM_TRAIN, CIFAR100_TRANSFORM_TEST
        train_transform = CIFAR100_TRANSFORM_TRAIN
        test_transform = CIFAR100_TRANSFORM_TEST
        dataset_class = torchvision.datasets.CIFAR100
    elif args.dataset == 'gtsrb':
        from models import get_gtsrb_transforms
        train_transform, test_transform = get_gtsrb_transforms()
        dataset_class = torchvision.datasets.GTSRB
    else:  # imagenette
        from models import get_imagenette_transforms
        train_transform, test_transform = get_imagenette_transforms()
        dataset_class = torchvision.datasets.ImageFolder
    
    # Load datasets
    train_dataset = dataset_class(
        root='./data', train=True, download=True,
        transform=train_transform
    )
    test_dataset = dataset_class(
        root='./data', train=False, download=True,
        transform=test_transform
    )
    
    # Create experiment
    experiment = PoisonExperiment(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        configs=[config],
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Either load checkpoint or train clean model
    if args.load_checkpoint:
        logging.info(f"Loading checkpoint: {args.checkpoint_name}")
        experiment.load_checkpoint(args.checkpoint_name)
    else:
        logging.info("Training clean model...")
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers
        )
        experiment.train_model(
            train_loader,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            checkpoint_name=args.checkpoint_name
        )
    
    # Run poisoning experiments
    experiment.run_experiments()

if __name__ == "__main__":
    run_example()
