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
        return {
            "poison_type": self.config.poison_type.value,
            "poison_ratio": self.config.poison_ratio,
            "original_accuracy": self.original_accuracy,
            "poisoned_accuracy": self.poisoned_accuracy,
            "poison_success_rate": self.poison_success_rate,
            "timestamp": self.timestamp,
            "config": self.config.__dict__
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
        result = PoisonResult(self.config)
        device = get_device()
        num_poison = int(len(dataset) * self.config.poison_ratio)
        indices = np.random.choice(len(dataset), num_poison, replace=False)
        
        # Convert dataset to tensor format if not already
        if not isinstance(dataset.data, torch.Tensor):
            data = torch.tensor(dataset.data).float()
            if len(data.shape) == 3:
                data = data.unsqueeze(1)  # Add channel dimension if needed
            if data.shape[-3] == 3:  # If channels are last, transpose
                data = data.permute(0, 3, 1, 2)
        else:
            data = dataset.data.clone()
        
        # Normalize data to [0, 1]
        if data.max() > 1:
            data = data / 255.0
        
        poisoned_data = data.clone()
        poisoned_data = poisoned_data.to(device)
        
        # PGD attack
        for idx in indices:
            x = poisoned_data[idx:idx+1].clone().requires_grad_(True)
            
            # Random initialization
            x = x + torch.zeros_like(x).uniform_(-self.config.pgd_eps, self.config.pgd_eps)
            x = torch.clamp(x, 0, 1)
            
            for _ in range(self.config.pgd_steps):
                x.requires_grad_(True)
                loss = torch.norm(x)  # Simple L2 norm loss (can be modified)
                loss.backward()
                
                # PGD step
                grad = x.grad.sign()
                x = x + self.config.pgd_alpha * grad
                
                # Project back to epsilon ball
                diff = x - data[idx:idx+1].to(device)
                diff = torch.clamp(diff, -self.config.pgd_eps, self.config.pgd_eps)
                x = torch.clamp(data[idx:idx+1].to(device) + diff, 0, 1)
                x = x.detach()
            
            poisoned_data[idx] = x.squeeze()
        
        # Update dataset with poisoned samples
        if hasattr(dataset, 'data'):
            if isinstance(dataset.data, np.ndarray):
                poisoned_data = (poisoned_data.cpu().numpy() * 255).astype(np.uint8)
            dataset.data = poisoned_data.cpu() if isinstance(poisoned_data, torch.Tensor) else poisoned_data
        
        result.poisoned_indices = indices.tolist()
        return dataset, result

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
                 device: Optional[torch.device] = None):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.configs = configs
        self.output_dir = output_dir
        self.device = device if device is not None else get_device()
        os.makedirs(output_dir, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
    
    def train_model(self, 
                   train_loader: DataLoader,
                   epochs: int = 30,
                   learning_rate: float = 0.001) -> None:
        """Train model on poisoned data"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            accuracy = 100. * correct / total
            logger.info(f'Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, Acc={accuracy:.2f}%')
    
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
            self.train_model(poisoned_loader)
            
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
    """Example of running poisoning experiments"""
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from models import create_model, CIFAR100_TRANSFORM_TRAIN, CIFAR100_TRANSFORM_TEST
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Load CIFAR100 dataset
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True,
        transform=CIFAR100_TRANSFORM_TRAIN
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True,
        transform=CIFAR100_TRANSFORM_TEST
    )
    
    # Create model
    model = create_model()
    
    # Create poisoning configurations
    configs = [
        # PGD Attack config
        PoisonConfig(
            poison_type=PoisonType.PGD,
            poison_ratio=0.1,  # Poison 10% of the dataset
            pgd_eps=0.3,       # Maximum perturbation
            pgd_alpha=0.01,    # Step size
            pgd_steps=40       # Number of PGD steps
        ),
        
        # Genetic Algorithm config
        PoisonConfig(
            poison_type=PoisonType.GA,
            poison_ratio=0.1,      # Poison 10% of the dataset
            ga_pop_size=50,        # Population size
            ga_generations=100,     # Number of generations
            ga_mutation_rate=0.1    # Mutation rate
        ),
        
        # Label Flipping config
        PoisonConfig(
            poison_type=PoisonType.LABEL_FLIP_RANDOM_TO_RANDOM,
            poison_ratio=0.1,           # Poison 10% of the dataset
            source_class=None,         # No specific source class
            target_class=None          # No specific target class
        )
    ]
    
    # Create experiment
    experiment = PoisonExperiment(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        configs=configs,
        output_dir="poison_results"
    )
    
    # Run experiments
    results = experiment.run_experiments()
    
    # Print summary
    print("\nExperiment Summary:")
    for result in results:
        print(f"\nAttack Type: {result.config.poison_type}")
        print(f"Original Accuracy: {result.original_accuracy:.2f}%")
        print(f"Poisoned Accuracy: {result.poisoned_accuracy:.2f}%")
        print(f"Attack Success Rate: {result.poison_success_rate:.2f}")

if __name__ == "__main__":
    run_example()
