import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any
from tqdm import tqdm
import time
from collections import defaultdict

from utils.device import move_to_device
from utils.logging import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate model accuracy and collect detailed metrics.

    Args:
        model: Model to evaluate
        dataloader: Data loader for evaluation
        device: Device to evaluate on

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    start_time = time.time()
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    criterion = nn.CrossEntropyLoss()

    batch_times = []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            batch_start = time.time()
            inputs, targets = move_to_device(inputs, device), move_to_device(
                targets, device
            )
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Track per-class accuracies
            for target, pred in zip(targets, predicted):
                target_class = target.item()
                per_class_total[target_class] += 1
                if pred == target:
                    per_class_correct[target_class] += 1
                    
            batch_times.append(time.time() - batch_start)

    # Calculate metrics
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(dataloader)
    per_class_accuracies = {
        cls: 100.0 * per_class_correct[cls] / per_class_total[cls]
        for cls in per_class_total.keys()
    }
    
    eval_time = time.time() - start_time
    avg_batch_time = sum(batch_times) / len(batch_times)
    
    metrics = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'per_class_accuracies': per_class_accuracies,
        'total_samples': total,
        'correct_samples': correct,
        'evaluation_time': eval_time,
        'average_batch_time': avg_batch_time,
        'batch_times': batch_times,
    }

    logger.debug(f"Evaluation complete - Accuracy: {accuracy:.2f}%")
    return metrics


def evaluate_attack(
    model: nn.Module,
    poisoned_loader: DataLoader,
    clean_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate attack effectiveness with detailed metrics.

    Args:
        model: Model to evaluate
        poisoned_loader: Data loader for poisoned data
        clean_loader: Data loader for clean data
        device: Device to evaluate on

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    start_time = time.time()
    model.eval()
    
    # Evaluate on clean and poisoned data
    clean_metrics = evaluate_model(model, clean_loader, device)
    poisoned_metrics = evaluate_model(model, poisoned_loader, device)
    
    # Calculate attack success metrics
    attack_success_rate = clean_metrics['accuracy'] - poisoned_metrics['accuracy']
    relative_success_rate = (attack_success_rate / clean_metrics['accuracy']) * 100 if clean_metrics['accuracy'] > 0 else 0
    
    results = {
        'clean_accuracy': clean_metrics['accuracy'],
        'poisoned_accuracy': poisoned_metrics['accuracy'],
        'attack_success_rate': attack_success_rate,
        'relative_success_rate': relative_success_rate,
        'clean_loss': clean_metrics['loss'],
        'poisoned_loss': poisoned_metrics['loss'],
        'clean_per_class_accuracies': clean_metrics['per_class_accuracies'],
        'poisoned_per_class_accuracies': poisoned_metrics['per_class_accuracies'],
        'evaluation_time': time.time() - start_time,
        'clean_eval_time': clean_metrics['evaluation_time'],
        'poisoned_eval_time': poisoned_metrics['evaluation_time'],
        'clean_batch_stats': {
            'avg_time': clean_metrics['average_batch_time'],
            'total_samples': clean_metrics['total_samples'],
        },
        'poisoned_batch_stats': {
            'avg_time': poisoned_metrics['average_batch_time'],
            'total_samples': poisoned_metrics['total_samples'],
        }
    }

    logger.info(f"Clean accuracy: {results['clean_accuracy']:.2f}%")
    logger.info(f"Poisoned accuracy: {results['poisoned_accuracy']:.2f}%")
    logger.info(f"Attack success rate: {results['attack_success_rate']:.2f}%")
    logger.info(f"Relative success rate: {results['relative_success_rate']:.2f}%")

    return results


def evaluate_robustness(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epsilon: float = 0.3,
    alpha: float = 0.01,
    num_steps: int = 40,
) -> Dict[str, float]:
    """Evaluate model robustness using PGD attack.

    Args:
        model: Model to evaluate
        dataloader: Data loader for evaluation
        device: Device to evaluate on
        epsilon: Maximum perturbation size
        alpha: Step size for PGD
        num_steps: Number of PGD steps

    Returns:
        dict: Dictionary containing robustness metrics
    """
    model.eval()
    total = 0
    clean_correct = 0
    robust_correct = 0

    for inputs, targets in tqdm(dataloader, desc="Evaluating robustness"):
        inputs, targets = move_to_device(inputs, device), move_to_device(
            targets, device
        )
        batch_size = inputs.size(0)
        total += batch_size

        # Clean accuracy
        with torch.no_grad():
            clean_outputs = model(inputs)
            clean_predictions = clean_outputs.max(1)[1]
            clean_correct += clean_predictions.eq(targets).sum().item()

        # PGD attack
        perturbed_inputs = inputs.clone().detach()
        perturbed_inputs.requires_grad = True

        for _ in range(num_steps):
            outputs = model(perturbed_inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()

            with torch.no_grad():
                grad_sign = perturbed_inputs.grad.sign()
                perturbed_inputs.data = perturbed_inputs.data + alpha * grad_sign
                eta = torch.clamp(
                    perturbed_inputs.data - inputs.data, -epsilon, epsilon
                )
                perturbed_inputs.data = torch.clamp(inputs.data + eta, 0, 1)
                perturbed_inputs.grad.zero_()

        # Robust accuracy
        with torch.no_grad():
            robust_outputs = model(perturbed_inputs)
            robust_predictions = robust_outputs.max(1)[1]
            robust_correct += robust_predictions.eq(targets).sum().item()

    results = {
        "clean_accuracy": 100.0 * clean_correct / total,
        "robust_accuracy": 100.0 * robust_correct / total,
    }

    logger.info(f"Clean accuracy: {results['clean_accuracy']:.2f}%")
    logger.info(f"Robust accuracy: {results['robust_accuracy']:.2f}%")

    return results
