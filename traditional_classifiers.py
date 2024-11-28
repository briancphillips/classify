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
import time

from models.data import get_dataset
from models.architectures import get_model
from experiment.visualization import plot_classifier_comparison
from utils.logging import get_logger

logger = get_logger(__name__)


def extract_features_and_labels(dataset, model, device, batch_size=128):
    """Extract CNN features and labels from a PyTorch dataset."""
    model.eval()  # Set model to evaluation mode
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    features = []
    labels = []

    with torch.no_grad():  # Disable gradient computation
        for batch, targets in tqdm(dataloader, desc="Extracting CNN features"):
            # Move batch to device
            if isinstance(batch, torch.Tensor):
                batch = batch.to(device)
            # Extract features using CNN
            batch_features = model.extract_features(batch)
            # Move features back to CPU and convert to numpy
            batch_features = batch_features.cpu().numpy()
            if isinstance(targets, torch.Tensor):
                targets = targets.numpy()
            features.append(batch_features)
            labels.append(targets)

    return np.vstack(features), np.concatenate(labels)


def evaluate_traditional_classifiers(dataset_name, subset_size=None):
    """Evaluate traditional classifiers on a dataset."""
    # Get dataset and model
    train_dataset = get_dataset(dataset_name, train=True, subset_size=subset_size)
    test_dataset = get_dataset(dataset_name, train=False, subset_size=subset_size)
    
    # Initialize model and move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = get_model(dataset_name).to(device)
    
    # Load pretrained weights if available
    checkpoint_path = os.path.join('checkpoints', f'clean_model_{dataset_name}_best.pth.tar')
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.warning("No pretrained weights found. Feature extraction may be suboptimal.")
    
    model.eval()
    
    # Extract CNN features and labels with larger batch size for efficiency
    X_train, y_train = extract_features_and_labels(train_dataset, model, device, batch_size=256)
    X_test, y_test = extract_features_and_labels(test_dataset, model, device, batch_size=256)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize classifiers with better hyperparameters
    classifiers = {
        "KNN": KNeighborsClassifier(
            n_neighbors=min(20, len(y_train) // 100),  # Scale with dataset size
            weights='distance',  # Weight by distance
            n_jobs=-1  # Use all CPU cores
        ),
        "LR": LogisticRegression(
            max_iter=2000,
            multi_class='multinomial',
            class_weight='balanced',
            n_jobs=-1
        ),
        "RF": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            n_jobs=-1,
            class_weight='balanced'
        ),
        "SVM": SVC(
            kernel="rbf",
            C=10.0,  # Increased from default 1.0
            gamma='scale',  # Use scaled gamma
            class_weight='balanced',
            cache_size=2000  # Increase cache size for faster training
        ),
    }

    all_results = []
    for name, clf in classifiers.items():
        logger.info(f"Training {name} on {dataset_name}...")
        start_time = time.time()
        
        # Train classifier
        clf.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate
        start_time = time.time()
        y_pred = clf.predict(X_test)
        inference_time = time.time() - start_time
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate per-class accuracies
        num_classes = 100 if dataset_name.lower() == 'cifar100' else 43 if dataset_name.lower() == 'gtsrb' else 10
        class_accuracies = [0.0] * num_classes
        for cls in range(num_classes):
            mask = y_test == cls
            if np.any(mask):
                cls_acc = accuracy_score(y_test[mask], y_pred[mask])
                class_accuracies[cls] = cls_acc
        
        # Create result dict
        result = {
            'dataset_name': dataset_name,
            'metrics': {
                'dataset_name': dataset_name,
                'accuracy': accuracy,
                'class_accuracies': class_accuracies,
                'training_time': training_time,
                'inference_time': inference_time
            },
            'config': {
                'poison_type': 'none',
                'poison_ratio': 0.0,
                'source_class': 'all',
            },
            'train_size': len(train_dataset),
            'epochs': 1,
            'classifier': name.lower(),
            'model_architecture': name.lower(),
        }
        
        all_results.append(result)
        logger.info(f"{name} accuracy: {accuracy:.4f} (train time: {training_time:.2f}s, inference time: {inference_time:.2f}s)")

    return all_results


def evaluate_traditional_classifiers_on_poisoned(train_dataset, test_dataset, dataset_name, poison_config=None):
    """Evaluate traditional classifiers on clean or poisoned data."""
    # Initialize model and move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = get_model(dataset_name).to(device)
    
    # Extract CNN features and labels
    X_train, y_train = extract_features_and_labels(train_dataset, model, device)
    X_test, y_test = extract_features_and_labels(test_dataset, model, device)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize classifiers
    classifiers = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LR": LogisticRegression(max_iter=1000),
        "RF": RandomForestClassifier(n_estimators=100),
        "SVM": SVC(kernel="rbf", probability=True),
    }

    # Get number of classes for the dataset
    num_classes = 100 if dataset_name.lower() == 'cifar100' else 43 if dataset_name.lower() == 'gtsrb' else 10

    all_results = []
    for name, clf in classifiers.items():
        logger.info(f"Training {name}...")
        start_time = time.time()
        
        # Train classifier
        clf.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Get predictions
        start_time = time.time()
        y_pred = clf.predict(X_test)
        inference_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get per-class accuracies
        unique_classes = np.unique(y_test)
        class_accuracies = [0.0] * num_classes  # Initialize with correct number of classes
        for cls in unique_classes:
            mask = y_test == cls
            if np.any(mask):
                cls_acc = accuracy_score(y_test[mask], y_pred[mask])
                class_accuracies[cls] = cls_acc

        # Create result dict in format compatible with CSV export
        result = {
            'dataset_name': dataset_name,
            'metrics': {
                'dataset_name': dataset_name,
                'accuracy': accuracy,
                'class_accuracies': class_accuracies,
                'training_time': training_time,
                'inference_time': inference_time,
            },
            'config': {
                'poison_type': poison_config.poison_type.value if poison_config else 'none',
                'poison_ratio': poison_config.poison_ratio if poison_config else 0.0,
                'source_class': poison_config.source_class if poison_config else 'all',
            },
            'classifier': name.lower(),
            'model_architecture': name.lower(),
        }
        
        all_results.append(result)
        logger.info(f"{name} accuracy: {accuracy:.4f}")

    return all_results


def run_traditional_classifiers(dataset_name, poison_config=None, subset_size=100):
    """Run traditional classifiers and return results in the correct format."""
    # Get dataset
    train_dataset = get_dataset(dataset_name, train=True, subset_size=subset_size)
    test_dataset = get_dataset(dataset_name, train=False, subset_size=subset_size)

    # Extract features and labels
    X_train, y_train = extract_features_and_labels(train_dataset, get_model(dataset_name), torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    X_test, y_test = extract_features_and_labels(test_dataset, get_model(dataset_name), torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize classifiers
    classifiers = {
        'knn': KNeighborsClassifier(n_neighbors=3),
        'logistic_regression': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=100),
        'svm': SVC(kernel='rbf')
    }

    results = []
    for name, clf in classifiers.items():
        start_time = time.time()
        
        # Train classifier
        clf.fit(X_train_scaled, y_train)
        
        # Test classifier
        y_pred = clf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Create result dictionary
        result = {
            'dataset_name': dataset_name,
            'classifier': name,
            'model_architecture': name,
            'config': {
                'poison_type': poison_config.poison_type if poison_config else None,
                'poison_ratio': poison_config.poison_ratio if poison_config else 0,
                'source_class': poison_config.source_class if poison_config else None,
            },
            'metrics': {
                'accuracy': accuracy * 100,  # Convert to percentage
                'training_time': time.time() - start_time,
            }
        }
        results.append(result)
        
    return results


def main():
    # Create results directory
    output_dir = "results/traditional"
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate on each dataset
    datasets = ["cifar100", "gtsrb", "imagenette"]
    all_results = []

    for dataset in datasets:
        logger.info(f"\nEvaluating classifiers on {dataset}...")
        dataset_results = evaluate_traditional_classifiers(dataset)
        all_results.extend(dataset_results)

    # Export results to CSV
    from utils.export import export_results
    export_results(all_results, os.path.join(output_dir, "traditional_results.csv"))

    # Plot results for visualization
    plot_results = {}
    for result in all_results:
        dataset = result['dataset_name']
        if dataset not in plot_results:
            plot_results[dataset] = {}
        plot_results[dataset][result['classifier']] = result['metrics']['accuracy']
    
    plot_classifier_comparison(plot_results, output_dir)


if __name__ == "__main__":
    main()
