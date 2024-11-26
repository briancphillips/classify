import json
import os
from experiment.visualization import plot_poisoned_classifier_comparison


def collect_poisoned_results():
    """Collect poisoned results from all datasets."""
    results = {}
    datasets = ["cifar100", "gtsrb", "imagenette"]

    # Map between config poison types and display names
    attack_type_map = {
        "pgd": "pgd",
        "gradient_ascent": "ga",
        "label_flip_random_target": "label_flip",
        "label_flip_random_random": "label_flip",
        "label_flip_source_target": "label_flip",
    }

    for dataset in datasets:
        results[dataset] = {}
        results_dir = os.path.join("results", dataset)

        # Look for poison result files
        if os.path.exists(results_dir):
            for file in os.listdir(results_dir):
                if file.startswith("poison_results_") and file.endswith(".json"):
                    with open(os.path.join(results_dir, file), "r") as f:
                        result = json.load(f)
                        attack_type = result["config"]["poison_type"]

                        # Map the attack type to standard name
                        if attack_type in attack_type_map:
                            std_attack_type = attack_type_map[attack_type]

                            # For label flip variants, combine their results
                            if std_attack_type in results[dataset]:
                                # Take the best result if we already have one
                                if (
                                    result["poisoned_accuracy"]
                                    > results[dataset][std_attack_type][
                                        "poisoned_accuracy"
                                    ]
                                ):
                                    results[dataset][std_attack_type] = {
                                        "original_accuracy": result[
                                            "original_accuracy"
                                        ],
                                        "poisoned_accuracy": result[
                                            "poisoned_accuracy"
                                        ],
                                        "poison_success_rate": result[
                                            "poison_success_rate"
                                        ],
                                    }
                            else:
                                results[dataset][std_attack_type] = {
                                    "original_accuracy": result["original_accuracy"],
                                    "poisoned_accuracy": result["poisoned_accuracy"],
                                    "poison_success_rate": result[
                                        "poison_success_rate"
                                    ],
                                }

    return results


# Collect and plot results
results = collect_poisoned_results()

# Save collected results
with open("results/poisoned_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Collected results:")
print(json.dumps(results, indent=2))

# Generate plot
plot_poisoned_classifier_comparison(results, "results")
