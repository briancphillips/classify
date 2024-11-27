"""
Test plotting functionality.
"""

import sys
from pathlib import Path
import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from visualization.plotting import (
    plot_training_history,
    plot_poisoned_results,
    plot_classifier_comparison,
    load_and_plot_results
)

def test_load_and_plot_cifar100_results():
    """Test plotting CIFAR-100 results."""
    results_dir = project_root / "results" / "cifar100"
    
    # Test poisoning results plot
    load_and_plot_results(
        results_dir=results_dir,
        plot_type='poisoning',
        save_dir=project_root / "plots",
        show=False
    )

def test_load_and_plot_gtsrb_results():
    """Test plotting GTSRB results."""
    results_dir = project_root / "results" / "gtsrb"
    
    # Test poisoning results plot
    load_and_plot_results(
        results_dir=results_dir,
        plot_type='poisoning',
        save_dir=project_root / "plots",
        show=False
    )

if __name__ == "__main__":
    test_load_and_plot_cifar100_results()
    test_load_and_plot_gtsrb_results()
