"""
utils.py – Small shared utilities.

Exports:
  - set_seed             : set random seeds for reproducibility
  - cleanup_previous_results : clean old output directories before a new run
  - install_package      : conditionally pip-install a package (used on Kaggle)
  - ensemble_predict     : average softmax outputs from multiple models
  - evaluate_ensemble    : accuracy of an ensemble on a DataLoader
"""

import os
import random
import shutil
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def set_seed(seed=42):
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup_previous_results(output_dir='project_outputs'):
    """Clean up previous training results before a new run."""
    print("Cleaning previous results...")

    directories_to_clean = [
        os.path.join(output_dir, 'plots'),
        os.path.join(output_dir, 'results'),
        os.path.join(output_dir, 'models'),
    ]

    cleaned_count = 0
    for dir_path in directories_to_clean:
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                cleaned_count += 1
                print(f"   Cleaned: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            print(f"   WARNING: Failed to clean {dir_path}: {e}")

    print(f"Cleanup complete! Cleaned {cleaned_count} directories")
    return cleaned_count


def install_package(package):
    """Conditionally pip-install a package (useful on Kaggle where packages may be missing)."""
    try:
        __import__(package.split('[')[0])
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


def ensemble_predict(models, image_tensor):
    """Ensemble prediction from multiple models (average softmax outputs)."""
    predictions = []
    device = next(iter(models[0].parameters())).device

    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(image_tensor)
            # Handle tuple outputs (logits, embeddings)
            if isinstance(out, tuple):
                out = out[0]
            pred = F.softmax(out, dim=1)
            predictions.append(pred)

    ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
    return ensemble_pred


def evaluate_ensemble(models, test_loader, class_names):
    """Evaluate an ensemble of models on a DataLoader. Returns accuracy (float 0–100)."""
    device = next(iter(models[0].parameters())).device
    correct = 0
    total = 0

    for images, labels in tqdm(test_loader, desc="Ensemble Evaluation"):
        images, labels = images.to(device), labels.to(device)

        ensemble_pred = ensemble_predict(models, images)
        _, predicted = torch.max(ensemble_pred, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy
