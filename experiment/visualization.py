import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from typing import List, Dict
from datetime import datetime
import numpy as np

from config.dataclasses import PoisonResult
from utils.logging import get_logger

logger = get_logger(__name__)


def plot_results(results, output_dir):
    """Plot results from poisoning experiments."""
    # Convert results to pandas DataFrame
    data = []
    for result in results:
        # Handle both PoisonResult objects and dictionaries (from traditional classifiers)
        if isinstance(result, dict):
            row = {
                "Dataset": result.get('dataset_name', ''),
                "Attack": result['config']['poison_type'],
                "Model": result.get('model_architecture', 'Traditional'),  # Default to 'Traditional' for traditional classifiers
                "Accuracy": result['metrics']['accuracy'] * 100,
                "Training Time": result['metrics'].get('training_time', 0),
                "Inference Time": result['metrics'].get('inference_time', 0),
            }
        else:
            # For PoisonResult objects, use the original_accuracy as the accuracy metric
            row = {
                "Dataset": result.dataset_name,
                "Attack": result.config.poison_type.value,
                "Model": "Neural Network",  # Default to 'Neural Network' for PoisonResult
                "Accuracy": result.original_accuracy * 100,
                "Training Time": result.metrics.get('training_time', 0) if hasattr(result, 'metrics') else 0,
                "Inference Time": result.metrics.get('inference_time', 0) if hasattr(result, 'metrics') else 0,
            }
        data.append(row)

    df = pd.DataFrame(data)

    # Plot accuracy comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Model", y="Accuracy", hue="Attack")
    plt.title("Model Accuracy by Attack Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"))
    plt.close()

    # Plot timing comparison
    plt.figure(figsize=(12, 6))
    timing_df = df.melt(id_vars=['Model', 'Attack'], 
                       value_vars=['Training Time', 'Inference Time'],
                       var_name='Metric', value_name='Time (seconds)')
    sns.barplot(data=timing_df, x="Model", y="Time (seconds)", hue="Metric")
    plt.title("Model Timing Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "timing_comparison.png"))
    plt.close()


def plot_attack_comparison(results: List[PoisonResult], output_dir: str) -> None:
    """Plot detailed attack comparison.

    Args:
        results: List of experiment results (PoisonResult objects or dictionaries)
        output_dir: Directory to save plots
    """
    # Prepare data for plotting
    data = []
    metrics = ["Original Accuracy", "Poisoned Accuracy", "Attack Success Rate"]

    for result in results:
        if isinstance(result, dict):
            values = [
                result['metrics'].get('original_accuracy', 0) * 100,
                result['metrics'].get('poisoned_accuracy', 0) * 100,
                result['metrics'].get('poison_success_rate', 0) * 100,
            ]
            attack_type = result['config']['poison_type']
            poison_ratio = result['config'].get('poison_ratio', 0)
        else:
            values = [
                result.original_accuracy * 100,
                result.poisoned_accuracy * 100,
                result.poison_success_rate * 100,
            ]
            attack_type = result.config.poison_type.value
            poison_ratio = result.config.poison_ratio

        for metric, value in zip(metrics, values):
            data.append({
                "Attack": attack_type,
                "Metric": metric,
                "Value": value,
                "Poison Ratio": poison_ratio,
            })

    df = pd.DataFrame(data)

    # Create faceted plot
    g = sns.FacetGrid(
        df,
        col="Metric",
        height=5,
        aspect=0.8,
    )
    g.map_dataframe(
        sns.barplot,
        x="Attack",
        y="Value",
        hue="Poison Ratio",
    )
    g.set_xticklabels(rotation=45)
    g.set_titles("{col_name}")
    g.add_legend()
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(
        output_dir, f"attack_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved attack comparison plot to {plot_path}")


def plot_training_history(
    history: Dict[str, List[float]], output_dir: str, name: str
) -> None:
    """Plot training history.

    Args:
        history: Dictionary containing training metrics
        output_dir: Directory to save plot
        name: Name for the plot file
    """
    plt.figure(figsize=(10, 6))
    for metric, values in history.items():
        plt.plot(values, label=metric)

    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(
        output_dir, f"{name}_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved training history plot to {plot_path}")


def plot_robustness_results(results: Dict[str, float], output_dir: str) -> None:
    """Plot robustness evaluation results.

    Args:
        results: Dictionary containing robustness metrics
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(8, 6))
    metrics = list(results.keys())
    values = [results[m] for m in metrics]

    sns.barplot(x=metrics, y=values)
    plt.title("Model Robustness")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(
        output_dir, f"robustness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved robustness plot to {plot_path}")


def plot_classifier_comparison(
    results: Dict[str, Dict[str, float]], output_dir: str
) -> None:
    """Plot classifier performance comparison across datasets.

    Args:
        results: Dictionary with format {dataset: {classifier: accuracy}}
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(12, 6))

    # Set style
    plt.style.use("seaborn-v0_8-darkgrid")  # Use a valid matplotlib style

    # Prepare data
    datasets = list(results.keys())
    classifiers = ["KNN", "LR", "RF", "SVM"]
    x = np.arange(len(datasets))
    width = 0.2  # Width of bars

    # Plot bars for each classifier
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Set custom colors
    for i, (classifier, color) in enumerate(zip(classifiers, colors)):
        accuracies = [results[dataset][classifier.lower()] for dataset in datasets]
        plt.bar(
            x + i * width - width * 1.5,
            accuracies,
            width,
            label=classifier,
            color=color,
        )

    # Customize plot
    plt.title("Traditional Classifier Performance Across Datasets", pad=20, fontsize=14)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0, 1.0)
    plt.xticks(x, [d.upper() for d in datasets], fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(title="Classifiers", title_fontsize=12, fontsize=11)
    plt.grid(True, axis="y", alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()
    plot_path = os.path.join(
        output_dir,
        f"classifier_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved classifier comparison plot to {plot_path}")


def plot_poisoned_classifier_comparison(
    results: Dict[str, Dict[str, Dict[str, float]]], output_dir: str
) -> None:
    """Plot poisoned classifier performance comparison across datasets.

    Args:
        results: Dictionary with format {dataset: {attack: {metric: value}}}
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(12, 6))

    # Set style
    plt.style.use("seaborn-v0_8-darkgrid")

    # Prepare data
    datasets = list(results.keys())
    attacks = ["PGD", "GA", "Label Flip"]  # Common attack types
    x = np.arange(len(datasets))
    width = 0.2

    # Plot bars for each attack
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Set custom colors
    for i, (attack, color) in enumerate(zip(attacks, colors)):
        accuracies = []
        for dataset in datasets:
            # Get accuracy for this attack type (if available)
            attack_key = attack.lower().replace(" ", "_")
            if attack_key in results[dataset]:
                # Convert percentage to fraction
                accuracies.append(
                    results[dataset][attack_key]["poisoned_accuracy"] / 100.0
                )
            else:
                accuracies.append(0)  # Default if attack not present

        plt.bar(
            x + i * width - width * 1.5,
            accuracies,
            width,
            label=attack,
            color=color,
        )

    # Customize plot
    plt.title("Poisoned Model Performance Across Datasets", pad=20, fontsize=14)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0, 1.0)
    plt.xticks(x, [d.upper() for d in datasets], fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(title="Attack Types", title_fontsize=12, fontsize=11)
    plt.grid(True, axis="y", alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()
    plot_path = os.path.join(
        output_dir,
        f"poisoned_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved poisoned comparison plot to {plot_path}")


def plot_clean_vs_poisoned(
    results: Dict[str, Dict[str, Dict[str, float]]], output_dir: str
) -> None:
    """Plot clean vs poisoned accuracy comparison for each dataset.

    Args:
        results: Dictionary with format {dataset: {attack: {metric: value}}}
        output_dir: Directory to save plot
    """
    # Create a subplot for each dataset
    datasets = list(results.keys())
    fig, axes = plt.subplots(1, len(datasets), figsize=(15, 6))
    plt.style.use("seaborn-v0_8-darkgrid")

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        attacks = ["PGD", "GA", "Label Flip"]
        x = np.arange(len(attacks))
        width = 0.35

        clean_acc = []
        poisoned_acc = []
        for attack in attacks:
            attack_key = attack.lower().replace(" ", "_")
            if attack_key in results[dataset]:
                # Convert percentages to fractions
                clean_acc.append(
                    results[dataset][attack_key]["original_accuracy"] / 100.0
                )
                poisoned_acc.append(
                    results[dataset][attack_key]["poisoned_accuracy"] / 100.0
                )
            else:
                clean_acc.append(0)
                poisoned_acc.append(0)

        # Plot bars
        ax.bar(x - width / 2, clean_acc, width, label="Clean", color="#2ecc71")
        ax.bar(x + width / 2, poisoned_acc, width, label="Poisoned", color="#e74c3c")

        # Customize subplot
        ax.set_title(dataset.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(attacks, rotation=45)
        ax.set_ylim(0, 1.0)
        if idx == 0:
            ax.set_ylabel("Accuracy")
            ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(
        output_dir, f"clean_vs_poisoned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved clean vs poisoned comparison plot to {plot_path}")


def plot_per_dataset_performance(
    results: Dict[str, Dict[str, Dict[str, float]]], output_dir: str
) -> None:
    """Plot detailed performance metrics for each dataset.

    Args:
        results: Dictionary with format {dataset: {attack: {metric: value}}}
        output_dir: Directory to save plot
    """
    datasets = list(results.keys())
    metrics = ["original_accuracy", "poisoned_accuracy", "poison_success_rate"]
    metric_names = ["Clean Accuracy", "Poisoned Accuracy", "Attack Success"]

    for dataset in datasets:
        plt.figure(figsize=(10, 6))
        plt.style.use("seaborn-v0_8-darkgrid")

        attacks = ["PGD", "GA", "Label Flip"]
        x = np.arange(len(attacks))
        width = 0.25
        colors = ["#2ecc71", "#e74c3c", "#3498db"]

        for i, (metric, name, color) in enumerate(zip(metrics, metric_names, colors)):
            values = []
            for attack in attacks:
                attack_key = attack.lower().replace(" ", "_")
                if attack_key in results[dataset]:
                    # Convert percentages to fractions for accuracies
                    value = results[dataset][attack_key][metric]
                    if metric != "poison_success_rate":  # Don't convert success rate
                        value = value / 100.0
                    values.append(value)
                else:
                    values.append(0)

            plt.bar(x + (i - 1) * width, values, width, label=name, color=color)

        plt.title(f"{dataset.upper()} Performance Metrics")
        plt.xlabel("Attack Type")
        plt.ylabel("Score")
        plt.ylim(0, 1.0)
        plt.xticks(x, attacks)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(
            output_dir,
            f"{dataset}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {dataset} performance plot to {plot_path}")
