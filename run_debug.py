import yaml
import torch
from pathlib import Path
from run_experiments import ExperimentManager
from utils.logging import setup_logging, get_logger
from utils.error_logging import get_error_logger
from multiprocessing import freeze_support
from notebook_config import deep_update

# Initialize logging
setup_logging()
logger = get_logger(__name__)
error_logger = get_error_logger()

def main():
    # Create experiment manager
    config_path = 'experiments/config.yaml'
    manager = ExperimentManager(config_path)

    # Override configs for debugging
    debug_config = {
        'dataset_overrides': {
            'cifar100': {
                'epochs': 2,  # Run fewer epochs for debugging
                'batch_size': 128,  # Training batch size
                'subset_size': 100  # Use only 100 samples
            },
            'gtsrb': {
                'epochs': 2,  # Run fewer epochs for debugging
                'batch_size': 128,  # Training batch size
                'subset_size': 100  # Use only 100 samples
            },
            'imagenette': {
                'epochs': 2,  # Run fewer epochs for debugging
                'batch_size': 128,  # Training batch size
                'subset_size': 100  # Use only 100 samples
            }
        },
        'output': {
            'base_dir': 'results',
            'save_models': False,
            'consolidated_file': 'all_results.csv',
            'save_individual_results': True  # Make sure individual results are saved
        },
        'execution': {
            'max_workers': 1,  # Run sequentially
            'device': 'gpu'
        },
        'experiment_groups': {
            'basic_comparison': {
                'description': 'Basic comparison of attacks across different datasets',
                'experiments': [{
                    'name': 'cifar100_debug',
                    'dataset': 'cifar100',
                    'attacks': ['ga','label_flip'],
                    'subset_size': 100,  # Debug with small subset
                    'poison_config': {
                        'batch_size': 32,  # Poisoning batch size
                        'poison_ratio': 0.005  # Small poison ratio for debugging
                    }
                },{
                    'name': 'gtsrb_debug',
                    'dataset': 'gtsrb',
                    'attacks': ['pgd','ga','label_flip'],
                    'subset_size': 100,  # Debug with small subset
                    'poison_config': {
                        'batch_size': 32,  # Poisoning batch size
                        'poison_ratio': 0.005  # Small poison ratio for debugging
                    }
                },{
                    'name': 'imagenette_debug',
                    'dataset': 'imagenette',
                    'attacks': ['pgd','ga','label_flip'],
                    'subset_size': 100,  # Debug with small subset
                    'poison_config': {
                        'batch_size': 32,  # Poisoning batch size
                        'poison_ratio': 0.005  # Small poison ratio for debugging
                    }
                }]
            }
        }
    }

    # Use the context manager for temporary overrides
    with manager.override_config(**debug_config):
        # Log device information
        device_info = manager._get_device_info()
        logger.info(f"Running experiments on: {device_info}")
        logger.info(f"Total experiments to run: {manager.total_experiments}")
        
        # Run experiments with temporary config
        manager.run_experiments()

if __name__ == '__main__':
    freeze_support()
    main()
