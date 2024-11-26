import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm import tqdm

from models.data import get_dataset
from experiment.visualization import plot_classifier_comparison
from utils.logging import get_logger

logger = get_logger(__name__)


def extract_features_and_labels(dataset, batch_size=128):
    """Extract features and labels from a PyTorch dataset."""
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    features = []
    labels = []

    for batch, targets in tqdm(dataloader, desc="Extracting features"):
        if isinstance(batch, torch.Tensor):
            batch = batch.numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.numpy()
        features.append(batch.reshape(batch.shape[0], -1))
        labels.append(targets)

    return np.vstack(features), np.concatenate(labels)


def evaluate_traditional_classifiers(dataset_name, subset_size=10):
    """Evaluate traditional classifiers on a dataset."""
    # Get dataset
    train_dataset = get_dataset(dataset_name, train=True, subset_size=subset_size)
    test_dataset = get_dataset(dataset_name, train=False, subset_size=subset_size)

    # Extract features and labels
    X_train, y_train = extract_features_and_labels(train_dataset)
    X_test, y_test = extract_features_and_labels(test_dataset)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize classifiers
    classifiers = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LR": LogisticRegression(max_iter=1000),
        "RF": RandomForestClassifier(n_estimators=100),
        "SVM": SVC(kernel="rbf"),
    }

    results = {}
    for name, clf in classifiers.items():
        logger.info(f"Training {name} on {dataset_name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        logger.info(f"{name} accuracy: {accuracy:.4f}")

    return results


def main():
    # Create results directory
    output_dir = "results/traditional"
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate on each dataset
    datasets = ["cifar100", "gtsrb", "imagenette"]
    results = {}

    for dataset in datasets:
        logger.info(f"\nEvaluating classifiers on {dataset}...")
        results[dataset] = evaluate_traditional_classifiers(dataset)

    # Plot results
    plot_classifier_comparison(results, output_dir)


if __name__ == "__main__":
    main()
