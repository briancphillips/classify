import torch
import torch.nn as nn
from pathlib import Path
import shutil
import time
from utils.checkpoints import save_checkpoint, load_checkpoint, cleanup_checkpoints
from utils.logging import get_logger

logger = get_logger(__name__)

def test_checkpoint_functionality():
    """Test the checkpoint save/load functionality"""
    logger.info("Testing checkpoint functionality...")
    
    # Create a simple model and optimizer for testing
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Save initial weights
    initial_weights = model.state_dict()['weight'].clone()
    
    # Create some dummy state
    test_state = {
        'epoch': 5,
        'state_dict': model.state_dict(),
        'best_acc': 85.5,
        'optimizer': optimizer.state_dict(),
        'swa_n': 3
    }
    
    # Save checkpoint
    logger.info("Saving checkpoint...")
    save_checkpoint(test_state, is_best=True)
    
    # Modify model weights to simulate training
    with torch.no_grad():
        model.weight.fill_(99.9)
    
    # Load checkpoint
    logger.info("Loading checkpoint...")
    start_epoch, best_acc = load_checkpoint(model, optimizer)
    
    # Verify loaded state
    loaded_weights = model.state_dict()['weight']
    weights_match = torch.allclose(loaded_weights, initial_weights)
    
    logger.info("\nTest Results:")
    logger.info(f"Checkpoint Loading: {'✓' if start_epoch == 5 else '✗'}")
    logger.info(f"Best Accuracy Loading: {'✓' if best_acc == 85.5 else '✗'}")
    logger.info(f"Model Weights Restored: {'✓' if weights_match else '✗'}")
    
    # Clean up
    logger.info("\nCleaning up test files...")
    cleanup_checkpoints()
    
    success = all([
        start_epoch == 5,
        best_acc == 85.5,
        weights_match
    ])
    return success

if __name__ == "__main__":
    success = test_checkpoint_functionality()
    logger.info(f"\nOverall Test Status: {'✓ PASSED' if success else '✗ FAILED'}")
