"""
Checkpoint management utilities for model training.
"""

import os
import shutil
from pathlib import Path
import torch
from typing import Dict, Any, Tuple, Optional
from utils.logging import get_logger

logger = get_logger(__name__)

def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: str = "checkpoints",
    filename: str = "checkpoint.pth.tar"
) -> None:
    """
    Save checkpoint to disk.
    
    Args:
        state: Dictionary containing checkpoint data
        is_best: If True, also save as best model
        checkpoint_dir: Directory to save checkpoints
        filename: Name of checkpoint file
    """
    checkpoint_dir = os.path.join(os.getcwd(), checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        shutil.copyfile(filepath, best_filepath)
        logger.info(f"Saved best model checkpoint to {best_filepath}")
    logger.info(f"Saved checkpoint to {filepath}")

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str = "checkpoints",
    filename: str = "checkpoint.pth.tar",
    swa_model: Optional[torch.nn.Module] = None
) -> Tuple[int, float]:
    """
    Load checkpoint from disk.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        checkpoint_dir: Directory containing checkpoints
        filename: Name of checkpoint file
        swa_model: Optional SWA model to load state into
    
    Returns:
        Tuple of (start_epoch, best_accuracy)
    """
    filepath = os.path.join(checkpoint_dir, filename)
    if not os.path.isfile(filepath):
        logger.warning(f"No checkpoint found at {filepath}")
        return 0, 0.0

    logger.info(f"Loading checkpoint from {filepath}")
    checkpoint = torch.load(filepath)
    
    start_epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)
    
    # Load model state
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    
    # Load optimizer state
    if 'optimizer' in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Load SWA state if available
    if swa_model is not None and 'swa_state_dict' in checkpoint:
        swa_model.load_state_dict(checkpoint['swa_state_dict'])
        logger.info("Loaded SWA model state")
    
    logger.info(f"Loaded checkpoint from epoch {start_epoch} with best accuracy {best_acc:.2f}%")
    return start_epoch, best_acc

def cleanup_checkpoints(checkpoint_dir: str = "checkpoints") -> None:
    """
    Clean up checkpoint directory, keeping only the best model and latest checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
    """
    if not os.path.exists(checkpoint_dir):
        return
        
    keep_files = {'checkpoint.pth.tar', 'model_best.pth.tar'}
    for file in os.listdir(checkpoint_dir):
        if file not in keep_files:
            os.remove(os.path.join(checkpoint_dir, file))
            logger.info(f"Removed old checkpoint: {file}")
