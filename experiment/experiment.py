import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from typing import List, Optional
from tqdm import tqdm
import numpy as np
import time
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
from config.dataclasses import PoisonConfig, PoisonResult
from attacks import create_poison_attack
from models import train_model, get_model, get_dataset
from utils.device import get_device, clear_memory
from utils.logging import get_logger
from utils.error_logging import get_error_logger
from utils.checkpoints import save_checkpoint, load_checkpoint  # Added import
from .evaluation import evaluate_model, evaluate_attack
from .visualization import plot_results, plot_attack_comparison
from traditional_classifiers import evaluate_traditional_classifiers_on_poisoned
import yaml
from torchvision import transforms

logger = get_logger(__name__)
error_logger = get_error_logger()

class Trainer:
    def __init__(self, model, criterion, optimizer, device, config):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        # Initialize training components based on config
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        self.use_swa = config.get('use_swa', True)
        self.use_mixup = config.get('use_mixup', True)
        self.label_smoothing = config.get('label_smoothing', 0.1)
        
        # Setup mixed precision training
        self.scaler = GradScaler() if self.use_amp else None
        
        # Setup SWA if enabled
        if self.use_swa:
            self.swa_model = AveragedModel(model)
            self.swa_scheduler = SWALR(
                optimizer,
                swa_lr=config.get('swa_lr', 0.05)
            )
            self.swa_start = config.get('swa_start', 160)
        else:
            self.swa_model = None
            self.swa_scheduler = None
        
        self.metrics = {
            'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': [],
            'training_time': 0,
            'inference_time': 0,
        }
        
        logger.info(f"Initialized trainer with config: {config}")

    def mixup_data(self, x, y, alpha=1.0):
        """Performs mixup on the input and target."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, pred, y_a, y_b, lam):
        """Mixup loss function."""
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total = 0
        
        for batch in tqdm(loader, desc=f'Epoch {epoch+1}', leave=False):
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Apply mixup if enabled and within mixup epochs
            if self.use_mixup and epoch < self.config.get('mixup_epochs', 150):
                inputs, labels_a, labels_b, lam = self.mixup_data(inputs, labels)
                
            self.optimizer.zero_grad()
            
            # Mixed precision training
            with autocast() if self.use_amp else nullcontext():
                outputs = self.model(inputs)
                if self.use_mixup and epoch < self.config.get('mixup_epochs', 150):
                    loss = self.mixup_criterion(outputs, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(outputs, labels)
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            if not self.use_mixup or epoch >= self.config.get('mixup_epochs', 150):
                _, predicted = outputs.max(1)
                total += labels.size(0)
                total_correct += predicted.eq(labels).sum().item()
            
        # Update SWA model if enabled
        if self.use_swa and epoch >= self.swa_start:
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()
        
        accuracy = total_correct / total if total > 0 else 0
        avg_loss = total_loss / len(loader)
        
        self.metrics['train_losses'].append(avg_loss)
        self.metrics['train_accs'].append(accuracy)
        
        return {'train_loss': avg_loss, 'train_acc': accuracy}

    def evaluate(self, loader, epoch):
        # Use SWA model for evaluation if enabled and after SWA start epoch
        model_to_eval = self.swa_model if self.use_swa and epoch >= self.swa_start else self.model
        model_to_eval.eval()
        
        total_loss = 0
        total_correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc='Evaluating', leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model_to_eval(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                total_correct += predicted.eq(labels).sum().item()
        
        accuracy = total_correct / total
        avg_loss = total_loss / len(loader)
        
        self.metrics['val_losses'].append(avg_loss)
        self.metrics['val_accs'].append(accuracy)
        
        return {'val_loss': avg_loss, 'val_acc': accuracy}

    def get_metrics(self):
        self.metrics['final_train_loss'] = self.metrics['train_losses'][-1]
        self.metrics['final_test_loss'] = self.metrics['val_losses'][-1]
        self.metrics['best_train_loss'] = min(self.metrics['train_losses'])
        self.metrics['best_test_loss'] = min(self.metrics['val_losses'])
        return self.metrics

class PoisonExperiment:
    def __init__(
            self,
            dataset_name: str,
            configs: List[PoisonConfig],
            config_path: str = "experiments/config.yaml",
            device: Optional[torch.device] = None,
            output_dir: str = "results",
            checkpoint_dir: str = "checkpoints",
        ):
        """Initialize experiment.

        Args:
            dataset_name: Name of dataset to use
            configs: List of poisoning configurations to run
            config_path: Path to config.yaml
            device: Optional device to use
            output_dir: Directory to save results
            checkpoint_dir: Directory to save checkpoints
        """
        # Load config
        with open(config_path) as f:
            full_config = yaml.safe_load(f)
        
        # Merge configs
        self.config = {**full_config['defaults']}
        if dataset_name in full_config['dataset_defaults']:
            self.config.update(full_config['dataset_defaults'][dataset_name])
            
        self.dataset_name = dataset_name
        self.configs = configs
        self.device = device if device is not None else get_device()
        self.output_dir = os.path.join(output_dir, dataset_name)
        self.checkpoint_dir = os.path.join(checkpoint_dir, dataset_name)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Get datasets with transforms
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4) if self.config.get('random_crop', True) else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip() if self.config.get('random_horizontal_flip', True) else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) if self.config.get('normalize', True) else transforms.Lambda(lambda x: x),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0) if self.config.get('cutout', True) else transforms.Lambda(lambda x: x)
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) if self.config.get('normalize', True) else transforms.Lambda(lambda x: x)
        ])

        self.train_dataset = get_dataset(
            dataset_name, 
            train=True,
            transform=train_transform
        )
        self.test_dataset = get_dataset(
            dataset_name,
            train=False,
            transform=test_transform
        )

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=self.config.get('pin_memory', True),
            persistent_workers=True if self.config['num_workers'] > 0 else False,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=self.config.get('pin_memory', True),
            persistent_workers=True if self.config['num_workers'] > 0 else False,
        )

        # Create model
        self.model = get_model(dataset_name).to(self.device)

        logger.info(f"Initialized experiment for {dataset_name} dataset")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Test samples: {len(self.test_dataset)}")
        logger.info(f"Running {len(self.configs)} poisoning configurations")

    def _train_model(self):
        """Train clean neural network model."""
        logger.info("Training clean neural network model...")
        
        # Setup criterion and optimizer
        criterion = nn.CrossEntropyLoss(
            label_smoothing=float(self.config.get('label_smoothing', 0.1))
        )
        
        optimizer = SGD(
            self.model.parameters(),
            lr=float(self.config['learning_rate']),
            momentum=float(self.config.get('momentum', 0.9)),
            weight_decay=float(self.config.get('weight_decay', 5e-4))
        )
        
        # Setup scheduler
        scheduler = MultiStepLR(
            optimizer,
            milestones=[int(x) for x in self.config.get('lr_schedule', [60, 120, 160])],
            gamma=float(self.config.get('lr_factor', 0.2))
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            criterion=criterion,
            optimizer=optimizer,
            device=self.device,
            config=self.config
        )

        # Train clean model
        patience = self.config.get('early_stopping_patience', 10)
        min_delta = self.config.get('early_stopping_min_delta', 0.001)
        patience_counter = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            train_metrics = trainer.train_epoch(self.train_loader, epoch)
            val_metrics = trainer.evaluate(self.test_loader, epoch)
            
            # Step the learning rate scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"Learning rate: {current_lr:.6f}")
            
            # Early stopping check
            val_loss = val_metrics['val_loss']
            if val_loss < (best_val_loss - min_delta):
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            logger.info(
                f"Epoch {epoch+1}/{self.config['epochs']} - "
                f"Train Loss: {train_metrics['train_loss']:.4f} - "
                f"Val Loss: {val_metrics['val_loss']:.4f} - "
                f"Val Acc: {val_metrics['val_acc']:.2f}% - "
                f"Early Stop Counter: {patience_counter}/{patience}"
            )
            
            # Save checkpoint
            is_best = val_metrics['val_acc'] > best_val_loss
            if is_best:
                best_val_loss = val_metrics['val_acc']
            
            save_checkpoint(
                state={
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'best_acc': best_val_loss,
                    'optimizer': optimizer.state_dict(),
                    'swa_state_dict': trainer.swa_model.state_dict() if hasattr(trainer, 'swa_model') and trainer.swa_model is not None else None,
                    'metrics': trainer.get_metrics(),
                    'early_stopping_state': {
                        'counter': patience_counter,
                        'best_val_loss': best_val_loss,
                        'min_delta': min_delta
                    }
                },
                is_best=is_best,
                checkpoint_dir=self.checkpoint_dir,
                filename=f"clean_model_latest.pth.tar"
            )
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Get final training metrics
        training_metrics = trainer.get_metrics()

        # Now evaluate clean data with traditional classifiers using trained CNN features
        logger.info("Evaluating traditional classifiers on clean data...")
        traditional_results = evaluate_traditional_classifiers_on_poisoned(
            self.train_dataset,
            self.test_dataset,
            self.dataset_name
        )
        results = traditional_results

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
                    batch_size=self.config['batch_size'],
                    shuffle=True,
                    num_workers=0,
                    pin_memory=False,
                    persistent_workers=False,
                )

                # Train model on poisoned data
                poisoned_trainer = Trainer(
                    model=self.model,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=self.device,
                    config=self.config
                )

                for epoch in range(self.config['epochs']):
                    train_metrics = poisoned_trainer.train_epoch(poisoned_loader, epoch)
                    val_metrics = poisoned_trainer.evaluate(self.test_loader, epoch)
                    logger.info(
                        f"Epoch {epoch+1}/{self.config['epochs']} - "
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
                        'model_type': self.config.get('model_type', 'ResNet18'),
                        'model_architecture': str(self.model),
                        'poison_type': config.poison_type.value,
                        'epochs': self.config['epochs'],
                        'batch_size': self.config['batch_size'],
                        'learning_rate': self.config['learning_rate'],
                        'weight_decay': self.config.get('weight_decay', 5e-4),
                        'optimizer': 'SGD',
                        'early_stopping_patience': self.config.get('early_stopping_patience', 10),
                        'early_stopping_min_delta': self.config.get('early_stopping_min_delta', 0.001),
                        'lr_milestones': self.config.get('lr_schedule', [60, 120, 160]),
                        'lr_gamma': self.config.get('lr_factor', 0.2),
                    },
                    
                    # Dataset info
                    'dataset': {
                        'name': self.dataset_name,
                        'train_size': len(self.train_dataset),
                        'test_size': len(self.test_dataset),
                        'subset_size': self.config.get('subset_size', None),
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

        logger.info(f"Total experiment time: {time.time() - time.time():.2f}s")
        return results

    def run(self):
        """Run the poisoning experiment."""
        logger.info("Starting poisoning experiment")
        
        # Train model on clean data first
        logger.info("Training model on clean data")
        self._train_model()
        clean_accuracy = evaluate_model(
            self.model,
            self.test_loader,
            self.device
        )
        
        # Apply poisoning
        logger.info(f"Applying {self.configs[0].poison_type} poisoning")
        if self.configs[0].poison_type == PoisonType.PGD:
            self._apply_pgd_attack()
        elif self.configs[0].poison_type == PoisonType.GRADIENT_ASCENT:
            self._apply_gradient_ascent()
        else:
            self._apply_label_flip()
            
        # Train model on poisoned data
        logger.info("Training model on poisoned data")
        self._train_model()
        poisoned_accuracy = evaluate_model(
            self.model,
            self.test_loader,
            self.device
        )
        
        # Save results
        result = PoisonResult(
            config=self.configs[0],
            dataset_name=self.dataset_name,
            poisoned_indices=self.poisoned_indices,
            poison_success_rate=self.poison_success_rate,
            original_accuracy=clean_accuracy,
            poisoned_accuracy=poisoned_accuracy
        )
        result.save(self.output_dir)
        logger.info(f"Experiment complete. Results saved to {self.output_dir}")
