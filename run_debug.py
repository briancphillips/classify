import yaml
import torch
from pathlib import Path
from run_experiments import ExperimentManager
from utils.logging import setup_logging, get_logger
from utils.error_logging import get_error_logger
from multiprocessing import freeze_support
from config.experiment_config import create_config

# Initialize logging
setup_logging()
logger = get_logger(__name__)
error_logger = get_error_logger()

def main():
    # Create base configuration
    config = create_config(
        'cifar100',
        training={
            'epochs': 5,  # Reduced epochs for debugging
            'batch_size': 32  # Smaller batch size for debugging
        },
        model={
            'dropout_rate': 0.5  # Increased dropout for regularization
        },
        experiment_name='debug_run'
    )
    
    # Save the debug configuration
    debug_config_path = 'experiments/debug_config.yaml'
    config.save_yaml(debug_config_path)
    
    # Create experiment manager with the debug configuration
    manager = ExperimentManager(debug_config_path)
    
    try:
        # Run the experiment
        manager.run()
    except Exception as e:
        error_logger.exception(f"Error during experiment execution: {str(e)}")
        raise

if __name__ == '__main__':
    freeze_support()
    main()
