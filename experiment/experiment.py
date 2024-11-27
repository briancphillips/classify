import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from typing import List, Optional
from tqdm import tqdm
import numpy as np
import time
from torch.optim import Adam
from config.dataclasses import PoisonConfig, PoisonResult
from attacks import create_poison_attack
from models import train_model, get_model, get_dataset
from utils.device import get_device, clear_memory
from utils.logging import get_logger
from utils.error_logging import get_error_logger
from .evaluation import evaluate_model, evaluate_attack
from .visualization import plot_results, plot_attack_comparison
from traditional_classifiers import evaluate_traditional_classifiers_on_poisoned

logger = get_logger(__name__)
error_logger = get_error_logger()

class Trainer:
    def __init__(self, model, criterion, optimizer, device, config):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.metrics = {
            'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': [],
            'training_time': 0,
            'inference_time': 0,
        }

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0
        total_correct = 0
        for batch in loader:
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
        accuracy = total_correct / len(loader.dataset)
        self.metrics['train_losses'].append(total_loss / len(loader))
        self.metrics['train_accs'].append(accuracy)
        return {'train_loss': total_loss / len(loader), 'train_acc': accuracy}

    def evaluate(self, loader, epoch):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        with torch.no_grad():
            for batch in loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
        accuracy = total_correct / len(loader.dataset)
        self.metrics['val_losses'].append(total_loss / len(loader))
        self.metrics['val_accs'].append(accuracy)
        return {'val_loss': total_loss / len(loader), 'val_acc': accuracy}

    def get_metrics(self):
        self.metrics['final_train_loss'] = self.metrics['train_losses'][-1]
        self.metrics['final_test_loss'] = self.metrics['val_losses'][-1]
        self.metrics['best_train_loss'] = min(self.metrics['train_losses'])
        self.metrics['best_test_loss'] = min(self.metrics['val_losses'])
        return self.metrics

class PoisonExperiment:
    """Class to manage poisoning experiments."""

    def __init__(
        self,
        dataset_name: str,
        configs: List[PoisonConfig],
        batch_size: int = 128,
        epochs: int = 30,
        learning_rate: float = 0.001,
        num_workers: int = 2,
        subset_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        output_dir: str = "results",
        checkpoint_dir: str = "checkpoints",
    ):
        """Initialize experiment.

        Args:
            dataset_name: Name of dataset to use
            configs: List of poisoning configurations to run
            batch_size: Batch size for training
            epochs: Number of epochs to train
            learning_rate: Learning rate for optimizer
            num_workers: Number of workers for data loading
            subset_size: Optional number of samples per class
            device: Optional device to use
            output_dir: Directory to save results
            checkpoint_dir: Directory to save checkpoints
        """
        self.dataset_name = dataset_name
        self.configs = configs
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.subset_size = subset_size
        self.device = device if device is not None else get_device()
        self.output_dir = os.path.join(output_dir, dataset_name)
        self.checkpoint_dir = os.path.join(checkpoint_dir, dataset_name)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Get datasets
        self.train_dataset = get_dataset(
            dataset_name, train=True, subset_size=subset_size
        )
        self.test_dataset = get_dataset(
            dataset_name, train=False, subset_size=subset_size
        )

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False if num_workers == 0 else True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False if num_workers == 0 else True,
        )

        # Create model
        self.model = get_model(dataset_name).to(self.device)

        logger.info(f"Initialized experiment for {dataset_name} dataset")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Test samples: {len(self.test_dataset)}")
        logger.info(f"Running {len(self.configs)} poisoning configurations")

    def run(self) -> List[PoisonResult]:
        """Run all configured poisoning experiments.

        Returns:
            list: Results from all experiments
        """
        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        results = []
        experiment_start_time = time.time()

        # First evaluate clean data with traditional classifiers
        logger.info("Evaluating traditional classifiers on clean data...")
        traditional_results = evaluate_traditional_classifiers_on_poisoned(
            self.train_dataset,
            self.test_dataset,
            self.dataset_name
        )
        results.extend(traditional_results)

        # Then train clean neural network model
        logger.info("Training clean neural network model...")
        clean_checkpoint_path = os.path.join(self.checkpoint_dir, "clean_model")

        # Create trainer with advanced configuration
        trainer_config = {
            'use_amp': True,
            'use_swa': True,
            'use_mixup': True,
            'label_smoothing': 0.1,
            'swa_start': self.epochs // 2,
            'swa_lr': 0.05,
            'mixup_epochs': 150,
            'model_type': self.model.__class__.__name__,
            'model_architecture': str(self.model),
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': 0.0001,
            'optimizer': 'Adam',
        }
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.0001
        )
        
        trainer = Trainer(
            model=self.model,
            criterion=criterion,
            optimizer=optimizer,
            device=self.device,
            config=trainer_config
        )

        # Train clean model
        for epoch in range(self.epochs):
            train_metrics = trainer.train_epoch(self.train_loader, epoch)
            val_metrics = trainer.evaluate(self.test_loader, epoch)
            logger.info(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f} - "
                f"Val Loss: {val_metrics['val_loss']:.4f} - "
                f"Val Acc: {val_metrics['val_acc']:.2f}%"
            )

        # Get final training metrics
        training_metrics = trainer.get_metrics()

        # Run each poisoning configuration
        for config in self.configs:
            logger.info(f"\nRunning experiment with config: {config}")
            checkpoint_name = f"poisoned_{config.poison_type.value}"
            poison_start_time = time.time()

            try:
                # Create attack instance
                attack = create_poison_attack(config, self.device)
                attack.dataset_name = self.dataset_name  # Set dataset name before running attack

                # Run poisoning attack
                poisoned_dataset, result = attack.poison_dataset(
                    self.train_dataset, self.model
                )
                result.dataset_name = self.dataset_name

                # Create poisoned data loader
                poisoned_loader = DataLoader(
                    poisoned_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True,
                    persistent_workers=False,
                )

                # Train model on poisoned data
                poisoned_trainer = Trainer(
                    model=self.model,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=self.device,
                    config=trainer_config
                )

                for epoch in range(self.epochs):
                    train_metrics = poisoned_trainer.train_epoch(poisoned_loader, epoch)
                    val_metrics = poisoned_trainer.evaluate(self.test_loader, epoch)
                    logger.info(
                        f"Epoch {epoch+1}/{self.epochs} - "
                        f"Train Loss: {train_metrics['train_loss']:.4f} - "
                        f"Val Loss: {val_metrics['val_loss']:.4f} - "
                        f"Val Acc: {val_metrics['val_acc']:.2f}%"
                    )

                # Get final poisoned training metrics
                poisoned_metrics = poisoned_trainer.get_metrics()

                # Evaluate attack with neural network
                attack_results = evaluate_attack(
                    self.model,
                    poisoned_loader,
                    self.test_loader,
                    self.device,
                )

                # Evaluate traditional classifiers on poisoned data
                logger.info("Evaluating traditional classifiers on poisoned data...")
                traditional_results = evaluate_traditional_classifiers_on_poisoned(
                    poisoned_dataset,
                    self.test_dataset,
                    self.dataset_name,
                    config
                )
                results.extend(traditional_results)

                # Collect neural network metrics
                experiment_metrics = {
                    'dataset_name': self.dataset_name,
                    
                    # Configuration
                    'config': {
                        'model_type': trainer_config['model_type'],
                        'model_architecture': trainer_config['model_architecture'],
                        'poison_type': config.poison_type.value,
                        'epochs': self.epochs,
                        'batch_size': self.batch_size,
                        'learning_rate': self.learning_rate,
                        'weight_decay': trainer_config['weight_decay'],
                        'optimizer': trainer_config['optimizer'],
                        'num_classes': 100 if self.dataset_name.lower() == 'cifar100' else 43 if self.dataset_name.lower() == 'gtsrb' else 10,
                    },
                    
                    # Dataset info
                    'dataset': {
                        'name': self.dataset_name,
                        'train_size': len(self.train_dataset),
                        'test_size': len(self.test_dataset),
                        'subset_size': self.subset_size,
                    },
                    
                    # Training metrics
                    'clean_training': training_metrics,
                    'poisoned_training': poisoned_metrics,
                    
                    # Attack metrics
                    'original_accuracy': attack_results['clean_accuracy'],
                    'poisoned_accuracy': attack_results['poisoned_accuracy'],
                    'poison_success_rate': attack_results['attack_success_rate'],
                    'relative_success_rate': attack_results['relative_success_rate'],
                    
                    # Loss metrics
                    'final_train_loss': poisoned_metrics['final_train_loss'],
                    'final_test_loss': poisoned_metrics['final_test_loss'],
                    'best_train_loss': poisoned_metrics['best_train_loss'],
                    'best_test_loss': poisoned_metrics['best_test_loss'],
                    
                    # Timing metrics
                    'training_time': poisoned_metrics['training_time'],
                    'inference_time': poisoned_metrics['inference_time'],
                    'total_time': time.time() - poison_start_time,
                    
                    # Per-class metrics
                    'clean_per_class_accuracies': attack_results['clean_per_class_accuracies'],
                    'poisoned_per_class_accuracies': attack_results['poisoned_per_class_accuracies'],
                }

                # Update result object with metrics
                result.metrics = experiment_metrics
                result.save(self.output_dir)
                results.append(result)

            except Exception as e:
                error_msg = f"Error during poisoning experiment: {str(e)}"
                error_logger.log_error(e, f"Poisoning experiment failed with config: {config}")
                logger.error(error_msg)
                continue

            finally:
                # Clear memory
                clear_memory(self.device)

        # Plot results
        if results:
            plot_results(results, self.output_dir)
            plot_attack_comparison(results, self.output_dir)

        logger.info(f"Total experiment time: {time.time() - experiment_start_time:.2f}s")
        return results
