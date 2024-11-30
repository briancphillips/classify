import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from typing import List, Optional, Any, Dict, Union
from tqdm import tqdm
import numpy as np
import time
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.swa_utils import AveragedModel, SWALR
from config.dataclasses import PoisonConfig, PoisonResult, ExperimentConfig
from config.types import PoisonType
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
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
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
                with autocast('cuda', enabled=self.use_amp):
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
        
        with torch.no_grad(), autocast('cuda', enabled=self.use_amp):
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
    """Class for running poisoning experiments."""

    def __init__(
            self,
            dataset_name: str,
            configs: List[Union[ExperimentConfig, PoisonConfig]],
            device: torch.device,
            output_dir: str = "results",
            checkpoint_dir: str = "checkpoints",
    ):
        """Initialize experiment."""
        self.dataset_name = dataset_name
        self.configs = configs
        self.device = device
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        
        # Set up transforms based on dataset
        if self.dataset_name == 'gtsrb':
            transform = transforms.Compose([
                transforms.Resize((32, 32)),  # Resize all images to 32x32
                transforms.ToTensor(),
                transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
        # Load dataset
        self.train_dataset = get_dataset(
            dataset_name,
            train=True,
            transform=transform
        )
        self.test_dataset = get_dataset(
            dataset_name,
            train=False,
            transform=transform
        )

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
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
            'config': {},
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
            lr=0.1,
            momentum=0.9,
            weight_decay=0.0005
        )
        
        # Create scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[60, 120, 160],
            gamma=0.2
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
        
        # Get model-specific checkpoint path
        if self.dataset_name == 'cifar100':
            workspace_checkpoint = "/workspace/classify/checkpoints/wideresnet/wideresnet_best.pt"
        elif self.dataset_name == 'gtsrb':
            workspace_checkpoint = "/workspace/classify/checkpoints/gtsrb/gtsrb_best.pt"
        elif self.dataset_name == 'imagenette':
            workspace_checkpoint = "/workspace/classify/checkpoints/resnet50/resnet50_best.pt"
        else:
            workspace_checkpoint = None
            
        if workspace_checkpoint and os.path.exists(workspace_checkpoint):
            logger.info(f"Loading existing best model from {workspace_checkpoint}")
            try:
                checkpoint = self._load_checkpoint(workspace_checkpoint)
                logger.info("Successfully loaded pre-trained model")
                return
            except Exception as e:
                logger.warning(f"Failed to load workspace checkpoint: {e}")
                logger.info("Will train model from scratch")
            
        # Fallback to local checkpoint directory
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.dataset_name}_best.pt")
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading existing best model from {checkpoint_path}")
            try:
                checkpoint = self._load_checkpoint(checkpoint_path)
                logger.info("Successfully loaded pre-trained model")
                return
            except Exception as e:
                logger.warning(f"Failed to load local checkpoint: {e}")
                logger.info("Will train model from scratch")

        # If no checkpoint was loaded, continue with original training code
        start_time = time.time()

    def _apply_gradient_ascent_attack(self, poison_config: PoisonConfig):
        """Apply gradient ascent attack to create poisoned dataset."""
        logger.info("Applying gradient ascent attack...")
        attack = create_poison_attack(
            config=poison_config,
            device=self.device
        )
        self.poisoned_dataset, self.poisoned_indices, self.poison_success_rate = attack.poison_dataset()

    def _apply_pgd_attack(self, poison_config: PoisonConfig):
        pass

    def _apply_label_flip_attack(self, poison_config: PoisonConfig):
        pass

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
        
        # Get subset size from config if available
        subset_size = None
        if self.configs and isinstance(self.configs[0], ExperimentConfig):
            if hasattr(self.configs[0], 'data') and hasattr(self.configs[0].data, 'subset_size'):
                subset_size = self.configs[0].data.subset_size
                logger.info(f"Using subset size of {subset_size} for traditional classifiers")

        # Now evaluate clean data with traditional classifiers using trained CNN features
        logger.info("Evaluating traditional classifiers on clean data...")
        traditional_results = evaluate_traditional_classifiers_on_poisoned(
            self.train_dataset,
            self.test_dataset,
            self.dataset_name,
            poison_config=None,
            subset_size=subset_size
        )
        results = traditional_results

        # Apply poisoning
        poison_config = self.configs[0].poison if isinstance(self.configs[0], ExperimentConfig) else self.configs[0]
        logger.info(f"Applying {poison_config.poison_type} poisoning")
        if poison_config.poison_type == PoisonType.PGD:
            self._apply_pgd_attack(poison_config)
        elif poison_config.poison_type == PoisonType.GRADIENT_ASCENT:
            self._apply_gradient_ascent_attack(poison_config)
        elif poison_config.poison_type in [PoisonType.LABEL_FLIP_RANDOM_TO_RANDOM, 
                                         PoisonType.LABEL_FLIP_RANDOM_TO_TARGET,
                                         PoisonType.LABEL_FLIP_SOURCE_TO_TARGET]:
            self._apply_label_flip_attack(poison_config)
            
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
            poisoned_indices=[],
            poison_success_rate=0.0,
            original_accuracy=clean_accuracy,
            poisoned_accuracy=poisoned_accuracy
        )
        result.save(self.output_dir)
        logger.info(f"Experiment complete. Results saved to {self.output_dir}")

        # Run each poisoning configuration
        for config in self.configs:
            logger.info(f"\nRunning experiment with config: {config}")
            poison_start_time = time.time()

            try:
                # Create attack instance
                attack = create_poison_attack(config.poison, self.device)
                attack.dataset_name = self.dataset_name  # Set dataset name before running attack

                # Run poisoning attack
                poisoned_dataset, result = attack.poison_dataset(
                    self.train_dataset, self.model
                )
                result.dataset_name = self.dataset_name

                # Evaluate traditional classifiers on poisoned data
                logger.info("Evaluating traditional classifiers on poisoned data...")
                traditional_results = evaluate_traditional_classifiers_on_poisoned(
                    poisoned_dataset,
                    self.test_dataset,
                    self.dataset_name,
                    config
                )
                results.extend(traditional_results)

            except Exception as e:
                error_msg = f"Error during poisoning experiment: {str(e)}"
                logger.error(error_msg)
                continue

            finally:
                # Clear memory
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

        # Plot results if available
        if results:
            plot_results(results, self.output_dir)

        logger.info(f"Total experiment time: {time.time() - time.time():.2f}s")
        return results
