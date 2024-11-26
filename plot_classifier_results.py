import json
from experiment.visualization import plot_classifier_comparison

# Load results
with open("results/classifier_results.json", "r") as f:
    results = json.load(f)

# Generate plot
plot_classifier_comparison(results, "results")
