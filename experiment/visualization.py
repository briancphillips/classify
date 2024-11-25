import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from typing import List, Dict
from datetime import datetime

from config.dataclasses import PoisonResult
from utils.logging import get_logger

logger = get_logger(__name__)


def plot_results(results: List[PoisonResult], output_dir: str) -> None:
    """Plot experiment results.

    Args:
        results: List of experiment results
        output_dir: Directory to save plots
    """
    # Prepare data for plotting
    data = []
    for result in results:
        data.append(
            {
                "Attack": result.config.poison_type.value,
                "Metric": "Original Accuracy",
                "Value": result.original_accuracy,
                "Poison Ratio": result.config.poison_ratio,
            }
        )
        data.append(
            {
                "Attack": result.config.poison_type.value,
                "Metric": "Poisoned Accuracy",
                "Value": result.poisoned_accuracy,
                "Poison Ratio": result.config.poison_ratio,
            }
        )
        data.append(
            {
                "Attack": result.config.poison_type.value,
                "Metric": "Attack Success Rate",
                "Value": result.poison_success_rate,
                "Poison Ratio": result.config.poison_ratio,
            }
        )

    df = pd.DataFrame(data)

    # Create accuracy comparison plot
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df[df["Metric"].isin(["Original Accuracy", "Poisoned Accuracy"])],
        x="Attack",
        y="Value",
        hue="Metric",
    )
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(
        output_dir,
        f"accuracy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
    )
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved accuracy comparison plot to {plot_path}")

    # Create attack success rate plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df[df["Metric"] == "Attack Success Rate"],
        x="Attack",
        y="Value",
        hue="Poison Ratio",
    )
    plt.title("Attack Success Rate Comparison")
    plt.ylabel("Success Rate (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(
        output_dir, f"attack_success_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved attack success rate plot to {plot_path}")


def plot_attack_comparison(results: List[PoisonResult], output_dir: str) -> None:
    """Plot detailed attack comparison.

    Args:
        results: List of experiment results
        output_dir: Directory to save plots
    """
    # Prepare data for plotting
    data = []
    metrics = ["Original Accuracy", "Poisoned Accuracy", "Attack Success Rate"]

    for result in results:
        values = [
            result.original_accuracy,
            result.poisoned_accuracy,
            result.poison_success_rate,
        ]
        for metric, value in zip(metrics, values):
            data.append(
                {
                    "Attack": result.config.poison_type.value,
                    "Metric": metric,
                    "Value": value,
                    "Poison Ratio": result.config.poison_ratio,
                }
            )

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
