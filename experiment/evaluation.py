import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm

from utils.device import move_to_device
from utils.logging import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate model accuracy.

    Args:
        model: Model to evaluate
        dataloader: Data loader for evaluation
        device: Device to evaluate on

    Returns:
        float: Model accuracy as percentage
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = move_to_device(inputs, device), move_to_device(
                targets, device
            )
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    logger.debug(f"Evaluation complete - Accuracy: {accuracy:.2f}%")
    return accuracy


def evaluate_attack(
    model: nn.Module,
    poisoned_loader: DataLoader,
    clean_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate attack effectiveness.

    Args:
        model: Model to evaluate
        poisoned_loader: Data loader for poisoned data
        clean_loader: Data loader for clean data
        device: Device to evaluate on

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    results = {
        "clean_accuracy": evaluate_model(model, clean_loader, device),
        "poisoned_accuracy": evaluate_model(model, poisoned_loader, device),
    }

    # Calculate attack success rate (difference in accuracy)
    results["attack_success_rate"] = (
        results["clean_accuracy"] - results["poisoned_accuracy"]
    )

    logger.info(f"Clean accuracy: {results['clean_accuracy']:.2f}%")
    logger.info(f"Poisoned accuracy: {results['poisoned_accuracy']:.2f}%")
    logger.info(f"Attack success rate: {results['attack_success_rate']:.2f}%")

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
