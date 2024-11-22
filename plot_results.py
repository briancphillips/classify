import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
results = {
    "cifar100": {"knn": 0.6481, "rf": 0.6378, "svm": 0.6857, "lr": 0.6563},
    "gtsrb": {"knn": 0.988756927949327, "rf": 0.988756927949327, "svm": 0.9877276326207443, "lr": 0.9884402216943785},
    "imagenette": {"knn": 0.9895541401273885, "rf": 0.987515923566879, "svm": 0.9892993630573248, "lr": 0.9859872611464968}
}

# Create plot
plt.figure(figsize=(12, 6))

# Setup
datasets = list(results.keys())
classifiers = ['knn', 'rf', 'svm', 'lr']
x = np.arange(len(datasets))
width = 0.2

# Plot bars for each classifier
for i, clf in enumerate(classifiers):
    accuracies = [results[dataset][clf] for dataset in datasets]
    plt.bar(x + i*width, accuracies, width, label=clf.upper())

plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.title('Traditional Classifier Performance Across Datasets')
plt.xticks(x + width*1.5, datasets)
plt.legend()
plt.grid(True, alpha=0.3)

# Add value labels on top of each bar
for i, clf in enumerate(classifiers):
    accuracies = [results[dataset][clf] for dataset in datasets]
    for j, v in enumerate(accuracies):
        plt.text(j + i*width, v, f'{v:.2%}', ha='center', va='bottom', rotation=90)

plt.tight_layout()
plt.savefig('classifier_comparison.png')
plt.close()
