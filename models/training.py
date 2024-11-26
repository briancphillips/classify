import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from utils.device import clear_memory, move_to_device
from utils.logging import get_logger

logger = get_logger(__name__)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 100,
    device: Optional[torch.device] = None,
    early_stopping_patience: int = 10,
    gradient_clip_val: float = 1.0,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    checkpoint_dir: Optional[str] = None,
    checkpoint_name: Optional[str] = None,
    resume_training: bool = False,
) -> Tuple[int, float]:
    """Train model with improved stability and monitoring.

    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Optional validation data loader
        epochs: Number of epochs to train
        device: Device to train on
        early_stopping_patience: Number of epochs to wait before early stopping
        gradient_clip_val: Maximum gradient norm
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization factor
        checkpoint_dir: Optional directory to save checkpoints
        checkpoint_name: Optional name for checkpoint files
        resume_training: Whether to try to resume from checkpoint

    Returns:
        tuple: (last_epoch, best_loss)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=5, verbose=True
    )

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}
    last_epoch = 0

    # Try to load checkpoint if resuming
    if resume_training and checkpoint_dir and checkpoint_name:
        try:
            # First try to load the best checkpoint
            epoch, loss, early_stopping_state = load_checkpoint(
                model,
                checkpoint_dir,
                checkpoint_name,
                optimizer=optimizer,
                device=device,
                load_best=True,
            )

            # If no best checkpoint, try latest
            if epoch == 0:
                epoch, loss, early_stopping_state = load_checkpoint(
                    model,
                    checkpoint_dir,
                    checkpoint_name,
                    optimizer=optimizer,
                    device=device,
                    load_best=False,
                )

            if early_stopping_state:
                patience_counter = early_stopping_state["patience_counter"]
                history = early_stopping_state["history"]
                best_val_loss = early_stopping_state["best_val_loss"]
                start_epoch = epoch + 1  # Start from next epoch
                last_epoch = epoch

                # Check if we should stop immediately
                if patience_counter >= early_stopping_patience:
                    logger.info(
                        f"Early stopping condition already met (patience={patience_counter}, "
                        f"best_val_loss={best_val_loss:.4f}). Training complete."
                    )
                    return last_epoch + 1, best_val_loss

                logger.info(
                    f"Resumed training from epoch {start_epoch} "
                    f"(patience={patience_counter}, best_val_loss={best_val_loss:.4f})"
                )
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {str(e)}")
            start_epoch = 0

    try:
        for epoch in range(start_epoch, epochs):
            last_epoch = epoch
            model.train()
            train_loss = 0
            total_batches = len(train_loader)

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

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

                optimizer.step()
                train_loss += loss.item()

                # Save checkpoint every 100 batches
                if (batch_idx + 1) % 100 == 0:
                    if checkpoint_dir and checkpoint_name:
                        early_stopping_state = {
                            "patience_counter": patience_counter,
                            "history": history,
                            "best_val_loss": best_val_loss,
                        }
                        save_checkpoint(
                            model,
                            checkpoint_dir,
                            checkpoint_name,
                            optimizer=optimizer,
                            epoch=epoch,
                            loss=train_loss / (batch_idx + 1),
                            early_stopping_state=early_stopping_state,
                        )

                if (batch_idx + 1) % 10 == 0:
                    clear_memory(device)

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            # Validation phase
            if val_loader is not None:
                val_loss = validate_model(model, val_loader, device)
                history["val_loss"].append(val_loss)

                scheduler.step(val_loss)

                # Early stopping and checkpointing
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if checkpoint_dir and checkpoint_name:
                        early_stopping_state = {
                            "patience_counter": patience_counter,
                            "history": history,
                            "best_val_loss": best_val_loss,
                        }
                        save_checkpoint(
                            model,
                            checkpoint_dir,
                            checkpoint_name,
                            optimizer=optimizer,
                            epoch=epoch,
                            loss=val_loss,
                            is_best=True,
                            early_stopping_state=early_stopping_state,
                        )
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch+1} "
                        f"(patience={patience_counter}, best_val_loss={best_val_loss:.4f})"
                    )
                    break

                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Val Loss: {val_loss:.4f} - "
                    f"Best Val Loss: {best_val_loss:.4f} - "
                    f"Patience: {patience_counter}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - " f"Train Loss: {train_loss:.4f}"
                )

            # Save regular checkpoint at end of epoch
            if checkpoint_dir and checkpoint_name:
                early_stopping_state = {
                    "patience_counter": patience_counter,
                    "history": history,
                    "best_val_loss": best_val_loss,
                }
                save_checkpoint(
                    model,
                    checkpoint_dir,
                    checkpoint_name,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=train_loss,
                    early_stopping_state=early_stopping_state,
                )

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        # Save checkpoint on error
        if checkpoint_dir and checkpoint_name:
            early_stopping_state = {
                "patience_counter": patience_counter,
                "history": history,
                "best_val_loss": best_val_loss,
            }
            save_checkpoint(
                model,
                checkpoint_dir,
                checkpoint_name,
                optimizer=optimizer,
                epoch=last_epoch,
                loss=train_loss if "train_loss" in locals() else float("inf"),
                early_stopping_state=early_stopping_state,
            )
        raise e

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
