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

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# Set matplotlib logger to WARNING level to suppress font debug messages
logging.getLogger('matplotlib').setLevel(logging.WARNING)

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
        
        # If dataset is a Subset, get the original dataset
        if isinstance(dataset, torch.utils.data.Subset):
            original_dataset = dataset.dataset
            # Map subset indices to original dataset indices
            indices = [dataset.indices[i] for i in indices]
        else:
            original_dataset = dataset
        
        # Create a new poisoned dataset
        poisoned_data = copy.deepcopy(original_dataset.data)
        poisoned_targets = copy.deepcopy(original_dataset.targets)
        
        for idx in tqdm(indices, desc="Generating poisoned samples"):
            x, y = dataset[idx if not isinstance(dataset, torch.utils.data.Subset) else indices.index(idx)]
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
            poisoned_data[idx] = x_poisoned
            result.poisoned_indices.append(idx)
        
        # Create a new dataset with poisoned data
        poisoned_dataset = copy.deepcopy(original_dataset)
        poisoned_dataset.data = poisoned_data
        poisoned_dataset.targets = poisoned_targets
        
        # If original dataset was a subset, create a new subset with poisoned data
        if isinstance(dataset, torch.utils.data.Subset):
            poisoned_dataset = torch.utils.data.Subset(poisoned_dataset, dataset.indices)
        
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
    
    def extract_features(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from dataset using the model's feature extractor."""
        logger.debug(f"Starting feature extraction for dataset of size {len(dataset)}")
        self.model.eval()
        features = []
        labels = []
        loader = DataLoader(dataset, batch_size=128)
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
    
    def run_experiments(self) -> List[PoisonResult]:
        """Run all poisoning experiments."""
        results = []
        test_loader = DataLoader(self.test_dataset, batch_size=128)
        clean_train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True)

        logger.debug(f"Dataset sizes - Train: {len(self.train_dataset)}, Test: {len(self.test_dataset)}")
        
        # Train and save clean model first
        logger.info("Training clean model")
        clean_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_checkpoint_name = f"clean_model_{clean_timestamp}"
        self.train_model(clean_train_loader, checkpoint_name=clean_checkpoint_name)
        
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
            attack = create_poison_attack(config)
            poisoned_dataset, result = attack.poison_dataset(self.train_dataset)
            logger.debug(f"Created poisoned training dataset with {len(poisoned_dataset)} samples")
            
            # Create poisoned test dataset
            poisoned_test_dataset, _ = attack.poison_dataset(self.test_dataset)
            poisoned_test_loader = DataLoader(poisoned_test_dataset, batch_size=128)
            logger.debug(f"Created poisoned test dataset with {len(poisoned_test_dataset)} samples")
            
            # Create poisoned dataloader
            poisoned_loader = DataLoader(poisoned_dataset, batch_size=128, shuffle=True)
            
            # Train model on poisoned data
            checkpoint_name = f"poisoned_model_{config.poison_type.value}_{config.poison_ratio}_{result.timestamp}"
            logger.info(f"Training model with checkpoint name: {checkpoint_name}")
            self.train_model(poisoned_loader, checkpoint_name=checkpoint_name)
            
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
    args = parser.parse_args()
    
    # Create output directory
    dataset_output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Get model and dataset
    model = get_model(args.dataset)
    train_loader, test_loader, train_dataset, test_dataset = get_dataset_loaders(
        args.dataset, 
        batch_size=128, 
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
        output_dir=dataset_output_dir
    )
    
    experiment.run_experiments()

if __name__ == "__main__":
    run_example()
