"""Checkpoint management utilities."""

import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Union
from pathlib import Path
from utils.logging import get_logger

logger = get_logger(__name__)

def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: Union[str, Path],
    filename: str,
    is_best: bool = False
) -> None:
    """Save checkpoint with standardized structure.
    
    Args:
        state: Must contain:
            - epoch: Current epoch
            - model_state_dict: Model state
            - optimizer_state_dict: Optimizer state
            - scheduler_state_dict: Scheduler state (if exists)
            - metrics: Dict of current metrics
            - config: Training configuration
        checkpoint_dir: Directory to save checkpoint
        filename: Name of checkpoint file (without extension)
        is_best: Whether to save as best model
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Always use .pt extension
    filepath = checkpoint_dir / f"{filename}.pt"
    torch.save(state, filepath)
    logger.info(f"Saved checkpoint to {filepath}")
    
    if is_best:
        best_path = checkpoint_dir / "best.pt"
        shutil.copyfile(filepath, best_path)
        logger.info(f"Saved best model checkpoint to {best_path}")

def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load checkpoint with standardized structure.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Optional device to load model to
        
    Returns:
        Dictionary containing all checkpoint data
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        if device is not None:
            model = model.to(device)
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        
        return checkpoint
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

def get_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """Get the path to the latest checkpoint in directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
        
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        return None
        
    # Sort by modification time, newest first
    return max(checkpoints, key=lambda p: p.stat().st_mtime)

def cleanup_old_checkpoints(
    checkpoint_dir: Union[str, Path],
    keep_last_n: int = 5,
    keep_best: bool = True
) -> None:
    """Clean up old checkpoints, keeping only the N most recent ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        keep_best: Whether to always keep best.pt
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return
        
    # Get all checkpoints except best.pt
    checkpoints = [
        p for p in checkpoint_dir.glob("*.pt")
        if p.name != "best.pt"
    ]
    
    # Sort by modification time, oldest first
    checkpoints.sort(key=lambda p: p.stat().st_mtime)
    
    # Remove old checkpoints
    for checkpoint in checkpoints[:-keep_last_n]:
        checkpoint.unlink()
        logger.info(f"Removed old checkpoint: {checkpoint}")
