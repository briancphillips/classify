"""
Tests for the training module.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytest
from pathlib import Path
import shutil

from models.training import Trainer
from utils.logging import get_logger
from utils.checkpoints import cleanup_checkpoints

logger = get_logger(__name__)

@pytest.fixture
def simple_model():
    return nn.Linear(10, 2)

@pytest.fixture
def optimizer(simple_model):
    return optim.SGD(simple_model.parameters(), lr=0.01)

@pytest.fixture
def dummy_data():
    # Create dummy data
    x = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=16)

def test_trainer_initialization(simple_model, optimizer):
    """Test trainer initialization with various configurations."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'use_amp': True,
        'use_swa': True,
        'use_mixup': True,
        'label_smoothing': 0.1
    }
    
    trainer = Trainer(
        model=simple_model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=device,
        config=config
    )
    
    assert trainer.use_amp == (True if device.type != 'cpu' else False)
    assert trainer.use_swa == True
    assert trainer.use_mixup == True
    assert trainer.label_smoothing == 0.1
    
    logger.info("Trainer initialization test passed")

def test_training_loop(simple_model, optimizer, dummy_data):
    """Test the training loop functionality."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'use_amp': False,  # Disable for testing
        'use_swa': False,  # Disable for testing
        'use_mixup': False,  # Disable for testing
        'label_smoothing': 0.0
    }
    
    trainer = Trainer(
        model=simple_model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=device,
        config=config
    )
    
    # Train for one epoch
    metrics = trainer.train_epoch(dummy_data, epoch=0)
    
    assert 'train_loss' in metrics
    assert 'train_acc' in metrics
    assert isinstance(metrics['train_loss'], float)
    assert isinstance(metrics['train_acc'], float)
    
    logger.info("Training loop test passed")

def test_checkpoint_functionality(simple_model, optimizer):
    """Test checkpoint save/load functionality."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {'use_amp': False, 'use_swa': False}
    
    trainer = Trainer(
        model=simple_model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=device,
        config=config
    )
    
    # Save initial weights
    initial_weights = simple_model.state_dict()['weight'].clone()
    
    # Save checkpoint
    trainer.save_state(epoch=5, best_acc=85.5, is_best=True)
    
    # Modify weights
    with torch.no_grad():
        simple_model.weight.fill_(99.9)
    
    # Load checkpoint
    start_epoch, best_acc = trainer.load_state()
    
    # Verify loaded state
    loaded_weights = simple_model.state_dict()['weight']
    weights_match = torch.allclose(loaded_weights, initial_weights)
    
    assert start_epoch == 5
    assert best_acc == 85.5
    assert weights_match
    
    # Clean up
    cleanup_checkpoints()
    
    logger.info("Checkpoint functionality test passed")

if __name__ == "__main__":
    # Run tests
    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    dummy_data = torch.utils.data.DataLoader(
        TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,))),
        batch_size=16
    )
    
    test_trainer_initialization(model, optimizer)
    test_training_loop(model, optimizer, dummy_data)
    test_checkpoint_functionality(model, optimizer)
    
    logger.info("All tests passed successfully!")
