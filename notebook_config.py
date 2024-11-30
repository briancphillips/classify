import yaml
import torch
from pathlib import Path
from copy import deepcopy
from run_experiments import ExperimentManager
from utils.logging import setup_logging, get_logger
from utils.error_logging import get_error_logger
from config.defaults import TRAINING_DEFAULTS, DATASET_DEFAULTS, POISON_DEFAULTS, OUTPUT_DEFAULTS, EXECUTION_DEFAULTS

def deep_update(d, u):
    """Recursively update nested dictionary d with values from u."""
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            d[k] = deep_update(d[k], v)
        else:
            d[k] = v
    return d

# Initialize logging
setup_logging()
logger = get_logger(__name__)
error_logger = get_error_logger()

# Create experiment manager with base config
config_path = 'experiments/config.yaml'
manager = ExperimentManager(config_path)

# Create debug configuration that properly merges with defaults
debug_config = {
    'dataset_overrides': {
        'cifar100': {
            'epochs': 200,
            'batch_size': 128
        },
        'gtsrb': {
            'epochs': 10,
            'batch_size': 128
        },
        'imagenette': {
            'epochs': 10,
            'batch_size': 64
        }
    },
    'execution': {
        'max_workers': 1,  # Run sequentially
        'gpu_ids': [0] if torch.cuda.is_available() else []
    },
    'output': {
        'base_dir': 'results',
        'save_model': True,
        'save_frequency': 10,
        'consolidated_file': 'debug_results.csv'
    },
    'experiment_groups': {
        'basic_comparison': {
            'description': 'Basic comparison of attacks across different datasets',
            'experiments': [{
                'name': 'cifar100_debug',
                'dataset': 'cifar100',
                'attacks': ['ga', 'label_flip'],
                'poison_config': deepcopy(POISON_DEFAULTS)  # Use default poison settings
            },{
                'name': 'gtsrb_debug',
                'dataset': 'gtsrb',
                'attacks': ['pgd', 'ga', 'label_flip'],
                'poison_config': deepcopy(POISON_DEFAULTS)
            },{
                'name': 'imagenette_debug',
                'dataset': 'imagenette',
                'attacks': ['pgd', 'ga', 'label_flip'],
                'poison_config': deepcopy(POISON_DEFAULTS)
            }]
        }   
    }
}

# Helper function to run experiments with proper config handling
def run_debug_experiments():
    # Create a deep copy of the original config
    original_config = deepcopy(manager.config)
    
    try:
        # Properly merge configurations
        merged_config = deepcopy(original_config)
        merged_config = deep_update(merged_config, debug_config)
        
        # Update manager's config
        manager.config = merged_config
        
        # Log device information
        device_info = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Running experiments on: {device_info}")
        logger.info(f"Total experiments to run: {manager.total_experiments}")
        
        # Run experiments with merged config
        manager.run_experiments()
        
    finally:
        # Always restore original config
        manager.config = original_config

if __name__ == "__main__":
    run_debug_experiments()
