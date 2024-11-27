from experiment.experiment import PoisonExperiment
from config.dataclasses import PoisonConfig, PoisonType
import torch
from multiprocessing import freeze_support
from utils.export import export_results
import os
from traditional_classifiers import run_traditional_classifiers

def main():
    # Configure experiment
    dataset_name = "gtsrb"  # Using GTSRB dataset
    subset_size = 100  # Small subset for testing
    batch_size = 32
    epochs = 2  # Small number of epochs for testing
    learning_rate = 0.001

    # Create poison config
    poison_config = PoisonConfig(
        poison_type=PoisonType.LABEL_FLIP_SOURCE_TO_TARGET,
        poison_ratio=0.1,  # 10% of data poisoned
        target_class=1,  # Target class to flip to
        source_class=0,  # Source class to flip from
    )

    # Initialize experiment
    experiment = PoisonExperiment(
        dataset_name=dataset_name,
        configs=[poison_config],
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        subset_size=subset_size,
    )

    # Run neural network experiment
    nn_results = experiment.run()

    # Run traditional classifiers
    trad_results = run_traditional_classifiers(
        dataset_name=dataset_name,
        poison_config=poison_config,
        subset_size=subset_size,
    )

    # Combine results
    all_results = [nn_results] + trad_results

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Export results to CSV
    output_file = os.path.join("results", f"{dataset_name}_experiment_results.csv")
    export_results(all_results, output_file)

    print("Experiment completed successfully!")
    print(f"Results exported to {output_file}")

if __name__ == '__main__':
    freeze_support()
    main()
