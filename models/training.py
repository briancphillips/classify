"""
Training module with advanced training techniques for neural networks.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple
from contextlib import nullcontext
import torch.nn.functional as F
from collections import defaultdict
import time

from utils.logging import get_logger
from utils.checkpoints import save_checkpoint, load_checkpoint, cleanup_old_checkpoints, get_latest_checkpoint
from utils.results import ResultsManager
from utils.device import clear_memory

logger = get_logger(__name__)

class Trainer:
    """
    Advanced neural network trainer with support for:
    - Mixed precision training
    - Stochastic Weight Averaging (SWA)
    - Label smoothing
    - Mixup augmentation
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        config: Dict[str, Any],
        results_manager: Optional[ResultsManager] = None
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.results_manager = results_manager
        
        # Initialize training components based on config
        self.use_amp = config.get('use_amp', True) and self.device.type != 'cpu'
        self.use_swa = config.get('use_swa', True)
        self.use_mixup = config.get('use_mixup', True)
        self.label_smoothing = config.get('label_smoothing', 0.1)
        
        # Track metrics
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_train_loss': float('inf'),
            'best_test_loss': float('inf'),
            'per_class_accuracies': {},
            'training_time': 0,
            'inference_time': 0,
            'total_time': 0,
        }
        
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
        
        logger.info(f"Initialized trainer with config: {config}")
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        start_time = time.time()
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        per_class_correct = defaultdict(int)
        per_class_total = defaultdict(int)
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Apply mixup if enabled
            if self.use_mixup and epoch < self.config.get('mixup_epochs', 150):
                inputs, targets_a, targets_b, lam = self._mixup_data(inputs, targets)
                
            self.optimizer.zero_grad()
            
            # Mixed precision training
            with autocast() if self.use_amp else nullcontext():
                outputs = self.model(inputs)
                if self.use_mixup and epoch < self.config.get('mixup_epochs', 150):
                    loss = self._mixup_criterion(outputs, targets_a, targets_b, lam)
                else:
                    loss = self.criterion(outputs, targets)
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            
            if self.use_mixup and epoch < self.config.get('mixup_epochs', 150):
                correct += (lam * predicted.eq(targets_a).sum().float()
                          + (1 - lam) * predicted.eq(targets_b).sum().float())
            else:
                correct += predicted.eq(targets).sum().item()
                # Update per-class accuracies
                for target, pred in zip(targets, predicted):
                    target_class = target.item()
                    per_class_total[target_class] += 1
                    if pred == target:
                        per_class_correct[target_class] += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        # Calculate per-class accuracies
        per_class_accuracies = {
            cls: 100. * per_class_correct[cls] / per_class_total[cls]
            for cls in per_class_total.keys()
        }
        
        epoch_time = time.time() - start_time
        self.metrics['training_time'] += epoch_time
        self.metrics['total_time'] += epoch_time
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Update best metrics
        if train_loss < self.metrics['best_train_loss']:
            self.metrics['best_train_loss'] = train_loss
        
        metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'per_class_accuracies': per_class_accuracies,
            'epoch_time': epoch_time
        }
        
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        
        # Update SWA if enabled
        if self.use_swa and epoch >= self.swa_start:
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()
        
        return metrics
    
    def evaluate(
        self,
        val_loader: torch.utils.data.DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Evaluate the model."""
        start_time = time.time()
        model_to_eval = self.swa_model if self.use_swa and epoch >= self.swa_start else self.model
        model_to_eval.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        per_class_correct = defaultdict(int)
        per_class_total = defaultdict(int)
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model_to_eval(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update per-class accuracies
                for target, pred in zip(targets, predicted):
                    target_class = target.item()
                    per_class_total[target_class] += 1
                    if pred == target:
                        per_class_correct[target_class] += 1
        
        # Calculate per-class accuracies
        per_class_accuracies = {
            cls: 100. * per_class_correct[cls] / per_class_total[cls]
            for cls in per_class_total.keys()
        }
        
        eval_time = time.time() - start_time
        self.metrics['inference_time'] += eval_time
        self.metrics['total_time'] += eval_time
        
        val_loss = total_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        # Update best metrics
        if val_loss < self.metrics['best_test_loss']:
            self.metrics['best_test_loss'] = val_loss
        
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_acc'].append(val_acc)
        
        return {
            'val_loss': val_loss,
            'val_acc': val_acc,
            'per_class_accuracies': per_class_accuracies,
            'eval_time': eval_time
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return {
            'best_train_loss': self.metrics['best_train_loss'],
            'best_test_loss': self.metrics['best_test_loss'],
            'final_train_loss': self.metrics['train_loss'][-1] if self.metrics['train_loss'] else None,
            'final_test_loss': self.metrics['val_loss'][-1] if self.metrics['val_loss'] else None,
            'per_class_accuracies': self.metrics['per_class_accuracies'],
            'training_time': self.metrics['training_time'],
            'inference_time': self.metrics['inference_time'],
            'total_time': self.metrics['total_time'],
            'train_history': {
                'loss': self.metrics['train_loss'],
                'acc': self.metrics['train_acc']
            },
            'val_history': {
                'loss': self.metrics['val_loss'],
                'acc': self.metrics['val_acc']
            }
        }

    def save_state(
        self,
        epoch: int,
        best_acc: float,
        is_best: bool = False
    ) -> None:
        """Save training state."""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics,
            'config': self.config,
        }
        
        if self.use_swa and epoch >= self.swa_start:
            state['swa_state_dict'] = self.swa_model.state_dict()
            state['swa_n'] = self.swa_model.n_averaged
        
        checkpoint_dir = (self.results_manager.get_checkpoint_dir() 
                         if self.results_manager else 'checkpoints')
        
        save_checkpoint(
            state=state,
            checkpoint_dir=checkpoint_dir,
            filename=f"checkpoint_epoch_{epoch}",
            is_best=is_best
        )
        
        # Clean up old checkpoints
        cleanup_old_checkpoints(checkpoint_dir)

    def load_state(
        self,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load training state."""
        checkpoint_dir = (self.results_manager.get_checkpoint_dir() 
                         if self.results_manager else 'checkpoints')
        
        if checkpoint_path is None:
            # Try to load latest checkpoint
            checkpoint_path = get_latest_checkpoint(checkpoint_dir)
            if checkpoint_path is None:
                logger.warning("No checkpoints found")
                return {}
        
        checkpoint = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            device=self.device
        )
        
        # Restore metrics
        if 'metrics' in checkpoint:
            self.metrics = checkpoint['metrics']
        
        # Restore SWA state if available
        if self.use_swa and 'swa_state_dict' in checkpoint:
            self.swa_model.load_state_dict(checkpoint['swa_state_dict'])
            if 'swa_n' in checkpoint:
                self.swa_model.n_averaged = checkpoint['swa_n']
        
        return checkpoint
    
    def _mixup_data(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply mixup augmentation to inputs and targets."""
        lam = np.random.beta(1.0, 1.0)
        batch_size = inputs.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
        targets_a, targets_b = targets, targets[index]
        
        return mixed_inputs, targets_a, targets_b, lam
    
    def _mixup_criterion(
        self,
        outputs: torch.Tensor,
        targets_a: torch.Tensor,
        targets_b: torch.Tensor,
        lam: float
    ) -> torch.Tensor:
        """Compute mixup criterion."""
        return lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    epochs: int = 100,
    device: Optional[torch.device] = None,
    early_stopping_patience: int = 10,
    gradient_clip_val: float = 0.5,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    checkpoint_dir: Optional[str] = None,
    checkpoint_name: Optional[str] = None,
    resume_training: bool = False,
    use_advanced_training: bool = False,
    num_classes: Optional[int] = None,
) -> Tuple[int, float]:
    """Train model with improved stability and monitoring."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    
    if use_advanced_training:
        if num_classes is None:
            raise ValueError("num_classes must be specified when using advanced training")
        
        # Use advanced training components
        criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True
        )
        
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            config={
                'use_amp': True,
                'use_swa': True,
                'use_mixup': True,
                'label_smoothing': 0.1,
                'swa_start': epochs // 2,
                'swa_lr': 0.05,
                'mixup_epochs': 150
            }
        )
        
        best_acc = 0
        for epoch in range(epochs):
            train_metrics = trainer.train_epoch(train_loader, epoch)
            val_metrics = trainer.evaluate(val_loader, epoch)
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_metrics['train_loss']:.4f} - Val Loss: {val_metrics['val_loss']:.4f} - Val Acc: {val_metrics['val_acc']:.2f}%")
            
            if val_metrics['val_acc'] > best_acc:
                best_acc = val_metrics['val_acc']
                trainer.save_state(epoch, best_acc, is_best=True)
        
        return epochs, best_acc
        
    else:
        # Original training logic
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=5, verbose=True
        )

        # Check if training was already completed in either checkpoint
        if resume_training and checkpoint_dir and checkpoint_name:
            for checkpoint_type in ["best", "latest"]:
                checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_{checkpoint_type}.pt")
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    if "early_stopping_state" in checkpoint:
                        if checkpoint["early_stopping_state"].get("training_completed", False):
                            logger.info(f"Training was already completed (found in {checkpoint_type} checkpoint). Not resuming.")
                            return checkpoint.get("epoch", 0) + 1, checkpoint.get(
                                "loss", float("inf")
                            )

        start_epoch = 0
        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}
        last_epoch = 0
        training_completed = False

        # Try to load checkpoint if resuming
        if resume_training and checkpoint_dir and checkpoint_name:
            try:
                latest_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_latest.pt")
                if os.path.exists(latest_path):
                    checkpoint = torch.load(latest_path, map_location=device)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    start_epoch = checkpoint.get("epoch", -1) + 1
                    last_epoch = start_epoch - 1
                    if "early_stopping_state" in checkpoint:
                        state = checkpoint["early_stopping_state"]
                        patience_counter = state.get("patience_counter", 0)
                        history = state.get("history", {"train_loss": [], "val_loss": []})
                        best_val_loss = state.get("best_val_loss", float("inf"))
                        training_completed = state.get("training_completed", False)
                    logger.info(f"Resumed training from epoch {start_epoch}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {str(e)}")
                start_epoch = 0

        try:
            for epoch in range(start_epoch, epochs):
                if training_completed:
                    break

                last_epoch = epoch
                model.train()
                train_loss = 0

                for batch_idx, (data, target) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
                ):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                    optimizer.step()
                    train_loss += loss.item()

                    if (batch_idx + 1) % 100 == 0 and checkpoint_dir and checkpoint_name:
                        save_checkpoint(
                            model,
                            checkpoint_dir,
                            checkpoint_name,
                            optimizer=optimizer,
                            epoch=epoch,
                            loss=train_loss / (batch_idx + 1),
                            early_stopping_state={
                                "patience_counter": patience_counter,
                                "history": history,
                                "best_val_loss": best_val_loss,
                                "training_completed": training_completed,
                            },
                        )

                    if (batch_idx + 1) % 10 == 0:
                        clear_memory(device)

                train_loss /= len(train_loader)
                history["train_loss"].append(train_loss)

                if val_loader is not None:
                    val_loss = validate_model(model, val_loader, device)
                    history["val_loss"].append(val_loss)
                    scheduler.step(val_loss)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        if checkpoint_dir and checkpoint_name:
                            save_checkpoint(
                                model,
                                checkpoint_dir,
                                checkpoint_name,
                                optimizer=optimizer,
                                epoch=epoch,
                                loss=val_loss,
                                is_best=True,
                                early_stopping_state={
                                    "patience_counter": patience_counter,
                                    "history": history,
                                    "best_val_loss": best_val_loss,
                                    "training_completed": training_completed,
                                },
                            )
                    else:
                        patience_counter += 1

                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        training_completed = True
                        # Save both latest and best checkpoints with training_completed=True
                        for is_best in [True, False]:
                            save_checkpoint(
                                model,
                                checkpoint_dir,
                                checkpoint_name,
                                optimizer=optimizer,
                                epoch=epoch,
                                loss=val_loss,
                                is_best=is_best,
                                early_stopping_state={
                                    "patience_counter": patience_counter,
                                    "history": history,
                                    "best_val_loss": best_val_loss,
                                    "training_completed": training_completed,
                                },
                            )
                        break

                    logger.info(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"Train Loss: {train_loss:.4f} - "
                        f"Val Loss: {val_loss:.4f} - "
                        f"Best Val Loss: {best_val_loss:.4f}"
                    )
                else:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")

                if checkpoint_dir and checkpoint_name:
                    save_checkpoint(
                        model,
                        checkpoint_dir,
                        checkpoint_name,
                        optimizer=optimizer,
                        epoch=epoch,
                        loss=train_loss,
                        early_stopping_state={
                            "patience_counter": patience_counter,
                            "history": history,
                            "best_val_loss": best_val_loss,
                            "training_completed": training_completed,
                        },
                    )

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            if checkpoint_dir and checkpoint_name:
                save_checkpoint(
                    model,
                    checkpoint_dir,
                    checkpoint_name,
                    optimizer=optimizer,
                    epoch=last_epoch,
                    loss=float("inf"),
                    early_stopping_state={
                        "patience_counter": patience_counter,
                        "history": history,
                        "best_val_loss": best_val_loss,
                        "training_completed": training_completed,
                    },
                )
            raise

        return last_epoch + 1, best_val_loss


def validate_model(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Validate model performance.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        device: Device to validate on

    Returns:
        float: Validation loss
    """
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.cross_entropy(output, target).item()

    return val_loss / len(val_loader)


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str,
    is_best: bool = False,
) -> None:
    """Save a model checkpoint.

    Args:
        state: The state to save
        checkpoint_dir: Directory to save checkpoint
        filename: Base name for the checkpoint
        is_best: If True, also save as best checkpoint
    """
    import os

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save latest checkpoint
    path = os.path.join(checkpoint_dir, f"{filename}.pt")
    torch.save(state, path)
    logger.info(f"Saved checkpoint to {path}")

    # Optionally save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, f"best_{filename}.pt")
        torch.save(state, best_path)
        logger.info(f"Saved best checkpoint to {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, Any]:
    """Load a model checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint
        model: The model to load weights into
        optimizer: The optimizer to load state into
        device: The device to load model to

    Returns:
        dict: The loaded checkpoint state
    """
    import os

    if not os.path.exists(checkpoint_path):
        logger.warning(f"No checkpoint found at {checkpoint_path}")
        return {}

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if device is not None:
            model = model.to(device)

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return {}


def cleanup_old_checkpoints(checkpoint_dir: str) -> None:
    """Clean up old checkpoints."""
    import os
    import glob

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    checkpoint_files.sort(key=os.path.getmtime)

    # Keep only the 5 most recent checkpoints
    for file in checkpoint_files[:-5]:
        os.remove(file)


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Get the path to the latest checkpoint."""
    import os
    import glob

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    if not checkpoint_files:
        return None

    checkpoint_files.sort(key=os.path.getmtime)
    return checkpoint_files[-1]
