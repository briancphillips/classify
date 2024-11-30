import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from typing import List, Optional, Any, Dict
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.swa_utils import AveragedModel, SWALR
from config.dataclasses import PoisonConfig, PoisonResult
from attacks import create_poison_attack
from models import train_model, get_model, get_dataset
from utils.device import get_device, clear_memory
from utils.logging import get_logger
from utils.error_logging import get_error_logger
from utils.checkpoints import save_checkpoint, load_checkpoint
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
        # Disable label smoothing if using mixup to avoid instability
        self.label_smoothing = config.get('label_smoothing', 0.1) if not self.use_mixup else 0.0
        
        # Setup mixed precision training
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient clipping value
        self.grad_clip = config.get('grad_clip', 1.0)
        
        # Setup SWA if enabled
        if self.use_swa:
            self.swa_model = AveragedModel(model)
            self.swa_scheduler = SWALR(
                optimizer,
                swa_lr=float(config.get('swa_lr', 0.05))
            )
            self.swa_start = int(config.get('swa_start', 160))
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
        
    def track_memory_usage(self):
        """Log GPU memory usage."""
        if not torch.cuda.is_available():
            return
            
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        
        logger.info(f"GPU Memory: {allocated:.1f}MB allocated, {cached:.1f}MB cached, "
                   f"{total_memory:.1f}MB total")

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
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        batch_count = len(loader)
        
        # Track initial memory usage
        self.track_memory_usage()
        
        for batch_idx, (inputs, targets) in enumerate(loader, 1):
            try:
                # Move to device and get batch size
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                batch_size = inputs.size(0)
                
                # Normalize input range if needed
                if inputs.abs().max() > 100:
                    inputs = inputs.clamp(-100, 100)
                
                # Zero gradients
                self.optimizer.zero_grad(set_to_none=True)
                
                # Forward pass with autocast
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    # Mixup
                    if self.use_mixup:
                        inputs, targets_a, targets_b, lam = self.mixup_data(inputs, targets)
                        outputs = self.model(inputs)
                        loss = self.mixup_criterion(outputs, targets_a, targets_b, lam)
                    else:
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                
                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf detected in loss at batch {batch_idx}")
                    continue
                
                # Backward pass with gradient scaling
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item() * batch_size
                _, predicted = outputs.max(1)
                total += batch_size
                if self.use_mixup:
                    correct += (lam * predicted.eq(targets_a).sum().item()
                            + (1 - lam) * predicted.eq(targets_b).sum().item())
                else:
                    correct += predicted.eq(targets).sum().item()
                
                # Log progress
                if batch_idx % max(1, batch_count // 5) == 0:
                    logger.info(f'Epoch: {epoch} [{batch_idx}/{batch_count}] '
                            f'Loss: {loss.item():.4f} '
                            f'Acc: {100. * correct/total:.2f}% '
                            f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
                    self.track_memory_usage()
                
                # Clear GPU cache periodically
                if batch_idx % 50 == 0:
                    del outputs, loss
                    if self.use_mixup:
                        del targets_a, targets_b
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        # Calculate epoch metrics
        if total > 0:  # Ensure we processed at least one batch successfully
            avg_loss = total_loss / total
            accuracy = 100. * correct / total
        else:
            logger.error("No batches were processed successfully in this epoch")
            avg_loss = float('inf')
            accuracy = 0.0
        
        # Update metrics history
        self.metrics['train_losses'].append(avg_loss)
        self.metrics['train_accs'].append(accuracy)
        
        return avg_loss, accuracy

    def evaluate(self, loader, epoch):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        batch_count = len(loader)
        
        # Track initial memory usage
        self.track_memory_usage()
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_amp):
            for batch_idx, (inputs, targets) in enumerate(loader, 1):
                # Move to device and get batch size
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                batch_size = inputs.size(0)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                total_loss += loss.item() * batch_size
                _, predicted = outputs.max(1)
                total += batch_size
                correct += predicted.eq(targets).sum().item()
                
                # Log progress and memory usage
                if batch_idx % max(1, batch_count // 3) == 0:
                    logger.info(f'Eval: [{batch_idx}/{batch_count}] '
                              f'Loss: {loss.item():.4f} '
                              f'Acc: {100. * correct/total:.2f}%')
                    self.track_memory_usage()
                
                # Clear intermediate tensors
                if batch_idx % 50 == 0:
                    del outputs, loss
                    torch.cuda.empty_cache()
        
        # Calculate epoch metrics
        avg_loss = total_loss / total
        accuracy = 100. * correct / total
        
        # Update metrics history
        self.metrics['val_losses'].append(avg_loss)
        self.metrics['val_accs'].append(accuracy)
        
        # Final memory check
        self.track_memory_usage()
        
        return avg_loss, accuracy

    def get_metrics(self):
        self.metrics['final_train_loss'] = self.metrics['train_losses'][-1]
        self.metrics['final_test_loss'] = self.metrics['val_losses'][-1]
        self.metrics['best_train_loss'] = min(self.metrics['train_losses'])
        self.metrics['best_test_loss'] = min(self.metrics['val_losses'])
        return self.metrics

    def should_stop(self):
        patience = self.config.get('early_stopping_patience', 10)
        min_delta = self.config.get('early_stopping_min_delta', 0.001)
        patience_counter = 0
        best_val_loss = float('inf')
        
        for val_loss in self.metrics['val_losses']:
            if val_loss < (best_val_loss - min_delta):
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                return True
        
        return False

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
        
        # Initialize with default config values
        self.config = {
            'batch_size': 128,
            'epochs': 200,
            'learning_rate': 0.1,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'lr_schedule': [60, 120, 160],
            'lr_factor': 0.2,
            'num_workers': 4,
            'pin_memory': True,
            'random_crop': True,
            'random_horizontal_flip': True,
            'normalize': True,
            'cutout': True
        }
        
        # Update with config file values if they exist
        if full_config and isinstance(full_config, dict):
            if 'defaults' in full_config:
                self.config.update(full_config['defaults'])
            if 'dataset_defaults' in full_config and dataset_name in full_config['dataset_defaults']:
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

    def _save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Any,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        is_best: bool = False,
        is_final: bool = False
    ) -> str:
        """Save a model checkpoint."""
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
            },
            'config': self.config,
        }
        
        if is_final:
            filename = "final"
        else:
            filename = f"epoch_{epoch}"
        
        save_checkpoint(
            state=state,
            checkpoint_dir=self.checkpoint_dir,
            filename=filename,
            is_best=is_best
        )
        
        return os.path.join(self.checkpoint_dir, f"{filename}.pt")

    def _load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load a model checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        # Create optimizer
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            momentum=self.config['momentum'],
            weight_decay=self.config['weight_decay']
        )
        
        # Create scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.config.get('lr_schedule', [60, 120, 160]),
            gamma=self.config.get('lr_factor', 0.2)
        )
        
        checkpoint = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device
        )
        
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            logger.info(f"Checkpoint metrics:")
            logger.info(f"Training - Loss: {metrics['train_loss']:.4f}, Acc: {metrics['train_acc']:.2f}%")
            logger.info(f"Validation - Loss: {metrics['val_loss']:.4f}, Acc: {metrics['val_acc']:.2f}%")
        
        return checkpoint

    def _train_model(self):
        """Train clean neural network model."""
        logger.info("Training clean neural network model...")
        start_time = time.time()

        # Create criterion, optimizer and scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            momentum=self.config['momentum'],
            weight_decay=self.config['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.config.get('lr_schedule', [60, 120, 160]),
            gamma=self.config.get('lr_factor', 0.2)
        )

        # Check if we should resume from a checkpoint
        start_epoch = 1
        best_val_acc = 0.0
        checkpoint_path = os.path.join(self.checkpoint_dir, "model_best.pt")
        if os.path.exists(checkpoint_path) and self.config.get('resume_training', False):
            try:
                checkpoint_data = self._load_checkpoint(checkpoint_path)
                start_epoch = checkpoint_data['epoch'] + 1
                optimizer = checkpoint_data['optimizer']
                scheduler = checkpoint_data['scheduler']
                best_val_acc = checkpoint_data['val_acc']
                logger.info(f"Resuming training from epoch {start_epoch}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                logger.info("Starting training from scratch")
        
        trainer = Trainer(
            model=self.model,
            criterion=criterion,
            optimizer=optimizer,
            device=self.device,
            config=self.config
        )
        
        # Training loop
        epochs = int(self.config.get('epochs', 5))  # Default to 5 epochs if not specified
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Batch size: {self.config['batch_size']}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        
        for epoch in range(start_epoch, epochs + 1):
            epoch_start = time.time()
            logger.info(f"Starting epoch {epoch}/{epochs}")
            
            # Train one epoch
            train_loss, train_acc = trainer.train_epoch(self.train_loader, epoch)
            logger.info(f"Epoch {epoch} - Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
            
            # Evaluate
            val_loss, val_acc = trainer.evaluate(self.test_loader, epoch)
            logger.info(f"Epoch {epoch} - Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch} - Learning rate: {current_lr}")
            
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            
            # Early stopping check
            if trainer.should_stop():
                logger.info("Early stopping triggered")
                break
            
            # Save periodic checkpoint if configured
            if self.config.get('save_model', True) and epoch % self.config.get('save_frequency', 10) == 0:
                self._save_checkpoint(
                    epoch, self.model, optimizer, scheduler,
                    train_loss, train_acc, val_loss, val_acc
                )
            
            # Save best model if validation accuracy improved
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
                self._save_checkpoint(
                    epoch, self.model, optimizer, scheduler,
                    train_loss, train_acc, val_loss, val_acc,
                    is_best=True
                )
                
        # Save final checkpoint
        self._save_checkpoint(
            epoch, self.model, optimizer, scheduler,
            train_loss, train_acc, val_loss, val_acc,
            is_final=True
        )
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        return trainer

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
                        'model_type': self.config.get('model_type', 'ResNet50'),
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
