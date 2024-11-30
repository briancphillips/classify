"""Examples of using the new configuration system."""

from config.experiment_config import create_config

def basic_config_example():
    """Basic configuration example."""
    # Create a default configuration for CIFAR-100
    config = create_config('cifar100')
    print("\nDefault configuration:")
    print(config.to_dict())

def training_override_example():
    """Example of overriding training parameters."""
    config = create_config(
        'cifar100',
        model={
            'dropout_rate': 0.5,  # Increase dropout
            'depth': 40  # Deeper network
        },
        training={
            'batch_size': 64,  # Smaller batch size
            'learning_rate': 0.01  # Lower learning rate
        },
        experiment_name='high_dropout_test'
    )
    print("\nModified training configuration:")
    print(config.to_dict())

def poison_experiment_example():
    """Example of setting up a poisoning experiment."""
    config = create_config(
        'cifar100',
        poison={
            'poison_type': 'pgd',
            'poison_ratio': 0.05,  # 5% poison ratio
            'batch_size': 32,
            'pgd_eps': 0.2,  # Smaller epsilon for more subtle perturbations
            'pgd_steps': 50  # More PGD steps
        },
        training={
            'epochs': 100,  # Fewer epochs for poisoning
            'batch_size': 64
        },
        experiment_name='pgd_poison_test'
    )
    print("\nPoisoning experiment configuration:")
    print(config.to_dict())

def multi_experiment_example():
    """Example of setting up multiple experiments."""
    config = create_config(
        'cifar100',
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
        },
        execution={
            'max_workers': 2,  # Run two experiments in parallel
            'gpu_ids': [0, 1]  # Use two GPUs
        }
    )
    print("\nMulti-experiment configuration:")
    print(config.to_dict())

if __name__ == '__main__':
    print("Configuration System Examples")
    print("=" * 50)
    
    basic_config_example()
    training_override_example()
    poison_experiment_example()
    multi_experiment_example()
