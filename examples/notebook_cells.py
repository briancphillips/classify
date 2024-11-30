# Cell 1: Basic Configuration
from config.experiment_config import create_config

# Setup hardware configuration
import torch
num_workers = 4  # Adjust based on your CPU cores
hardware_config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': num_workers,
    'pin_memory': True if torch.cuda.is_available() else False
}

# Create a simple configuration for CIFAR-100
config = create_config('cifar100')

# View the complete configuration
print("Default configuration:")
print(config.to_dict())

# Cell 2: Training Configuration
# Override training parameters for faster experimentation
config = create_config(
    'cifar100',
    training={
        'epochs': 5,  # Reduced epochs for testing
        'batch_size': 32,  # Smaller batch size
        'learning_rate': 0.01  # Lower learning rate
    },
    model={
        'dropout_rate': 0.5  # Increased dropout for regularization
    }
)

print("\nTraining configuration:")
print(config.training)
print("\nModel configuration:")
print(config.model)

# Cell 3: Poisoning Configuration
# Setup a poisoning experiment with PGD attack
config = create_config(
    'cifar100',
    poison={
        'poison_type': 'pgd',
        'poison_ratio': 0.05,  # 5% poison ratio
        'pgd_eps': 0.2,  # Perturbation size
        'pgd_steps': 50  # Number of PGD steps
    },
    training={
        'epochs': 100,
        'batch_size': 64
    }
)

print("\nPoisoning configuration:")
print(config.poison)

# Cell 4: Multi-GPU Configuration
# Setup multiple experiments to run in parallel
config = create_config(
    'cifar100',
    execution={
        'max_workers': 2,  # Run two experiments in parallel
        'gpu_ids': [0, 1]  # Use two GPUs if available
    },
    experiment_groups={
        'poison_comparison': {
            'description': 'Compare different poisoning methods',
            'experiments': [
                {
                    'name': 'pgd_poison',
                    'dataset': 'cifar100',
                    'poison': {
                        'poison_type': 'pgd',
                        'poison_ratio': 0.05
                    }
                },
                {
                    'name': 'gradient_ascent',
                    'dataset': 'cifar100',
                    'poison': {
                        'poison_type': 'gradient_ascent',
                        'poison_ratio': 0.05
                    }
                }
            ]
        }
    }
)

print("\nMulti-experiment configuration:")
print(config.experiment_groups)
print("\nExecution configuration:")
print(config.execution)

# Cell 5: Save and Load Configuration
# Save configuration to YAML
config_path = 'experiments/poison_experiment.yaml'
config.save_yaml(config_path)
print(f"\nSaved configuration to {config_path}")

# Load configuration from YAML
loaded_config = create_config.from_yaml(config_path)
print("\nLoaded configuration matches original:", loaded_config.to_dict() == config.to_dict())

# Cell 6: Running an Experiment
from run_experiments import ExperimentManager

# Create experiment manager with our configuration
manager = ExperimentManager(config_path)

# Run the experiment (commented out for safety)
# manager.run()

print("\nExperiment manager initialized with configuration:")
print(f"Dataset: {manager.config.dataset_name}")
print(f"Model: {manager.config.model.name}")
print(f"Training epochs: {manager.config.training.epochs}")
print(f"Poison type: {manager.config.poison.poison_type}")
