import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import os
from contextlib import nullcontext

from utils.device import clear_memory, move_to_device
from utils.logging import get_logger
from .advanced_training import (
    AdvancedTrainer, LabelSmoothingLoss, RandAugment,
    Cutout, mixup_data, mixup_criterion
)

logger = get_logger(__name__)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
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
        criterion = LabelSmoothingLoss(num_classes, smoothing=0.15)
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True
        )
        
        trainer = AdvancedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
            swa_start=epochs // 2,  # Start SWA halfway through
            grad_clip=gradient_clip_val,
            mixup_alpha=0.4
        )
        
        history, best_acc, swa_model = trainer.train()
        
        # If SWA model is better, use it
        if swa_model is not None:
            model.load_state_dict(swa_model.state_dict())
        
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
                    data, target = move_to_device(data, device), move_to_device(
                        target, device
                    )
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
    val_loader: DataLoader,
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
            data, target = move_to_device(data, device), move_to_device(target, device)
            output = model(data)
            val_loss += F.cross_entropy(output, target).item()

    return val_loss / len(val_loader)


def save_checkpoint(
    model: nn.Module,
    checkpoint_dir: str,
    name: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
    is_best: bool = False,
    early_stopping_state: Optional[Dict] = None,
) -> None:
    """Save a model checkpoint.

    Args:
        model: The model to save
        checkpoint_dir: Directory to save checkpoint
        name: Base name for the checkpoint
        optimizer: Optional optimizer state to save
        epoch: Optional epoch number
        loss: Optional loss value
        is_best: If True, also save as best checkpoint
        early_stopping_state: Optional dictionary containing early stopping state
    """
    import os

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Prepare checkpoint data
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }

    if optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if early_stopping_state:
        checkpoint["early_stopping_state"] = early_stopping_state

    # Save latest checkpoint
    path = os.path.join(checkpoint_dir, f"{name}_latest.pt")
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")

    # Optionally save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, f"{name}_best.pt")
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best checkpoint to {best_path}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_dir: str,
    name: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    load_best: bool = False,
) -> Tuple[int, float, Optional[Dict]]:
    """Load a model checkpoint.

    Args:
        model: The model to load weights into
        checkpoint_dir: Directory containing checkpoint
        name: Base name of the checkpoint
        optimizer: Optional optimizer to load state into
        device: Optional device to load model to
        load_best: If True, load the best checkpoint instead of latest

    Returns:
        tuple: (epoch, loss, early_stopping_state) loaded from checkpoint
    """
    import os

    suffix = "_best.pt" if load_best else "_latest.pt"
    path = os.path.join(checkpoint_dir, f"{name}{suffix}")

    if not os.path.exists(path):
        logger.warning(f"No checkpoint found at {path}")
        return 0, float("inf"), None

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        if device is not None:
            model = model.to(device)

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        epoch = checkpoint.get("epoch", -1)
        loss = checkpoint.get("loss", float("inf"))
        early_stopping_state = checkpoint.get("early_stopping_state", None)

        logger.info(f"Loaded checkpoint from {path} (epoch {epoch + 1})")
        return epoch, loss, early_stopping_state

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return 0, float("inf"), None
