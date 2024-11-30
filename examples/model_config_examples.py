"""Examples of dataset-specific model configurations."""

import torch
from pathlib import Path
from config.experiment_config import create_config
from utils.logging import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)

def find_checkpoint(model_type):
    """Find the best or latest checkpoint for a model."""
    checkpoint_dir = Path('~/Notebooks/classify/checkpoints') / model_type
    checkpoint_dir = checkpoint_dir.expanduser()
    
    if not checkpoint_dir.exists():
        logger.warning(f"Checkpoint directory {checkpoint_dir} does not exist")
        return None
        
    # First try to find best checkpoint
    best_checkpoint = checkpoint_dir / f'{model_type}_best.pt'
    if best_checkpoint.exists():
        logger.info(f"Found best checkpoint: {best_checkpoint}")
        return best_checkpoint
        
    # Otherwise get latest checkpoint
    latest_checkpoint = checkpoint_dir / f'{model_type}_latest.pt'
    if latest_checkpoint.exists():
        logger.info(f"Found latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint
        
    logger.warning(f"No checkpoints found in {checkpoint_dir}")
    return None

def setup_hardware_config():
    """Setup hardware-specific configuration."""
    if torch.cuda.is_available():
        device_info = f"CUDA (GPU: {torch.cuda.get_device_name(0)})"
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        gpu_ids = [0]
    elif torch.backends.mps.is_available():
        device_info = "MPS (Apple Silicon)"
        gpu_ids = []
    else:
        device_info = "CPU"
        gpu_ids = []

    print(f"Running on: {device_info}")
    
    return {
        'execution': {
            'max_workers': 1,
            'gpu_ids': gpu_ids
        },
        'training': {
            'num_workers': 4 if not torch.backends.mps.is_available() else 0,
            'pin_memory': True
        }
    }

def create_cifar_config():
    """Create CIFAR-100 configuration with WideResNet."""
    checkpoint_path = find_checkpoint('wideresnet')
    hardware_config = setup_hardware_config()
    
    return create_config(
        'cifar100',
        **hardware_config,
        checkpoint={
            'save_dir': str(checkpoint_path.parent) if checkpoint_path else 'checkpoints',
            'resume': True if checkpoint_path else False
        },
        poison={
            'poison_type': 'ga',
            'poison_ratio': 0.1,
            'batch_size': 32,
            'ga_steps': 50,
            'ga_iterations': 100,
            'ga_lr': 0.1
        }
    )

def create_gtsrb_config():
    """Create GTSRB configuration with custom CNN."""
    checkpoint_path = find_checkpoint('custom-cnn')
    hardware_config = setup_hardware_config()
    
    return create_config(
        'gtsrb',
        **hardware_config,
        checkpoint={
            'save_dir': str(checkpoint_path.parent) if checkpoint_path else 'checkpoints',
            'resume': True if checkpoint_path else False
        },
        poison={
            'poison_type': 'pgd',
            'poison_ratio': 0.1,
            'batch_size': 32,
            'pgd_eps': 0.3,
            'pgd_alpha': 0.01,
            'pgd_steps': 40
        }
    )

def create_imagenette_config():
    """Create ImageNette configuration with ResNet50."""
    checkpoint_path = find_checkpoint('resnet50')
    hardware_config = setup_hardware_config()
    
    return create_config(
        'imagenette',
        **hardware_config,
        training={
            'batch_size': 64  # Smaller batch size for ImageNette
        },
        checkpoint={
            'save_dir': str(checkpoint_path.parent) if checkpoint_path else 'checkpoints',
            'resume': True if checkpoint_path else False
        },
        poison={
            'poison_type': 'pgd',
            'poison_ratio': 0.1,
            'batch_size': 32,
            'pgd_eps': 0.3,
            'pgd_alpha': 0.01,
            'pgd_steps': 40
        }
    )

def main():
    """Create and display configurations for all datasets."""
    print("\nCIFAR-100 Configuration (WideResNet):")
    cifar_config = create_cifar_config()
    print(cifar_config.model)
    
    print("\nGTSRB Configuration (Custom CNN):")
    gtsrb_config = create_gtsrb_config()
    print(gtsrb_config.model)
    
    print("\nImageNette Configuration (ResNet50):")
    imagenette_config = create_imagenette_config()
    print(imagenette_config.model)
    
    # Create experiment groups
    experiment_groups = {
        'basic_comparison': {
            'description': 'Basic comparison of attacks across different datasets',
            'experiments': [
                {
                    'name': 'cifar100_debug',
                    'dataset': 'cifar100',
                    'attacks': ['ga', 'label_flip']
                },
                {
                    'name': 'gtsrb_debug',
                    'dataset': 'gtsrb',
                    'attacks': ['pgd', 'ga', 'label_flip']
                },
                {
                    'name': 'imagenette_debug',
                    'dataset': 'imagenette',
                    'attacks': ['pgd', 'ga', 'label_flip']
                }
            ]
        }
    }
    
    # Create final configuration
    final_config = create_config(
        'cifar100',
        experiment_groups=experiment_groups,
        output={
            'base_dir': 'results',
            'save_models': True,
            'save_frequency': 10,
            'consolidated_file': 'debug_results.csv',
            'save_individual_results': True
        }
    )
    
    # Save the configuration
    config_path = 'experiments/debug_config.yaml'
    final_config.save_yaml(config_path)
    print(f"\nSaved configuration to {config_path}")

if __name__ == '__main__':
    main()
