import torch
import torch.nn as nn
from pathlib import Path
import shutil
import time
from test_cifar100_enhanced import save_checkpoint, load_checkpoint

def test_checkpoint_functionality():
    print("Testing checkpoint functionality...")
    
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
    print("Saving checkpoint...")
    save_checkpoint(test_state, is_best=True)
    
    # Modify model weights to simulate training
    with torch.no_grad():
        model.weight.fill_(99.9)
    
    # Load checkpoint
    print("Loading checkpoint...")
    start_epoch, best_acc = load_checkpoint(model, optimizer)
    
    # Verify loaded state
    loaded_weights = model.state_dict()['weight']
    weights_match = torch.allclose(loaded_weights, initial_weights)
    
    print("\nTest Results:")
    print(f"Checkpoint Loading: {'✓' if start_epoch == 5 else '✗'}")
    print(f"Best Accuracy Loading: {'✓' if best_acc == 85.5 else '✗'}")
    print(f"Model Weights Restored: {'✓' if weights_match else '✗'}")
    
    # Clean up
    print("\nCleaning up test files...")
    shutil.rmtree(Path("checkpoints"), ignore_errors=True)
    
    return all([
        start_epoch == 5,
        best_acc == 85.5,
        weights_match
    ])

if __name__ == "__main__":
    success = test_checkpoint_functionality()
    print(f"\nOverall Test Status: {'✓ PASSED' if success else '✗ FAILED'}")
