from experiment.experiment import PoisonExperiment
from config.dataclasses import PoisonConfig
from config.types import PoisonType
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_base_configs(poison_ratio=0.05):
    """Create base configurations for all attack types."""
    configs = []
    
    # PGD Attack
    configs.append(
        PoisonConfig(
            poison_type=PoisonType.PGD,
            poison_ratio=poison_ratio,
            pgd_eps=0.3,
            pgd_alpha=0.01,
            pgd_steps=40,
            random_seed=42
        )
    )
    
    # Gradient Ascent Attack
    configs.append(
        PoisonConfig(
            poison_type=PoisonType.GRADIENT_ASCENT,
            poison_ratio=poison_ratio,
            ga_steps=50,
            ga_iterations=100,
            ga_lr=0.1,
            random_seed=42
        )
    )
    
    # Label Flip Variants
    configs.extend([
        # Random to Random
        PoisonConfig(
            poison_type=PoisonType.LABEL_FLIP_RANDOM_TO_RANDOM,
            poison_ratio=poison_ratio,
            random_seed=42
        ),
        # Random to Target (target class 0)
        PoisonConfig(
            poison_type=PoisonType.LABEL_FLIP_RANDOM_TO_TARGET,
            poison_ratio=poison_ratio,
            target_class=0,
            random_seed=42
        ),
        # Source to Target (source 1 to target 0)
        PoisonConfig(
            poison_type=PoisonType.LABEL_FLIP_SOURCE_TO_TARGET,
            poison_ratio=poison_ratio,
            source_class=1,
            target_class=0,
            random_seed=42
        )
    ])
    
    return configs

def run_subset_experiments():
    # Common experiment parameters
    batch_size = 32
    epochs = 2      # Same as your command line examples
    learning_rate = 0.001
    num_workers = 0  # Same as your command line examples
    subset_size = 10  # Same as your command line examples
    poison_ratio = 0.05  # Same as your command line examples

    # Datasets to test
    datasets = ['cifar100', 'gtsrb', 'imagenette']
    
    # Get all configurations
    base_configs = create_base_configs(poison_ratio)
    results = []
    
    # Run experiments for each dataset
    for dataset_name in datasets:
        logger.info(f"\nRunning experiments for {dataset_name}")
        
        experiment = PoisonExperiment(
            dataset_name=dataset_name,
            configs=base_configs,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            num_workers=num_workers,
            subset_size=subset_size,
            output_dir="results/subset_experiments",
            checkpoint_dir="checkpoints/subset_experiments"
        )
        
        dataset_results = experiment.run()
        results.extend(dataset_results)
        
        logger.info(f"Completed experiments for {dataset_name}")

    logger.info("All subset experiments completed!")
    return results

if __name__ == "__main__":
    run_subset_experiments()
