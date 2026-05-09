"""
evaluate.py – Evaluation utilities.

Exports:
  - show_misclassified  : display a grid of misclassified test images
  - evaluate_model      : full evaluation: accuracy, report, confusion matrix, optional mislabeled audit
"""

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm

from training.config import config
from training.datasets import get_transforms
from training.models import EfficientNetEmbeddingModel, FaceEmbeddingModel, get_model


def show_misclassified(samples, class_names, max_items=5):
    """Display a few misclassified test images for quick inspection."""
    if not samples:
        print("\nNo misclassified samples to display.")
        return

    n = min(max_items, len(samples))
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (path, true_label, pred_label) in zip(axes, samples[:n]):
        try:
            img = Image.open(path).convert('RGB')
            ax.imshow(img)
        except Exception as exc:
            print(f"Warning: failed to load image '{path}': {exc}")
            ax.text(0.5, 0.5, "Image load failed", ha="center", va="center")
        ax.set_title(
            f"T: {class_names[true_label]}\nP: {class_names[pred_label]}",
            fontsize=9
        )
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def evaluate_model(model, model_name, test_loader):
    """Evaluate model on test set.

    Handles:
      - TTA batches (5-D tensors)
      - ArcFace embedding-only models (uses model.arcface_eval for logits)
      - Standard classification-head models
      - Optional mislabeled audit (config.FLAG_MISLABELED)

    Returns:
        dict with keys: model, test_acc, precision, recall, f1
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING: {model_name.upper()}")
    print(f"{'='*70}")

    _device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(_device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    def embeddings_to_logits(embeddings):
        """Reconstruct logits from embeddings using the stored ArcFace weights."""
        # Preferred path for ArcFace-trained models: use learned ArcFace weights.
        if hasattr(model, 'arcface_eval') and isinstance(model.arcface_eval, dict):
            weight = model.arcface_eval.get('weight', None)
            scale = float(model.arcface_eval.get('scale', config.ARCFACE_SCALE))
            if weight is not None:
                weight = weight.to(embeddings.device)
                embeddings_norm = F.normalize(embeddings, dim=1)
                W_norm = F.normalize(weight, dim=1)
                return scale * (embeddings_norm @ W_norm.t())

        # Fallback to model head when available.
        if hasattr(model, 'head'):
            try:
                logits = model.head(embeddings)
                if logits.dim() == 2 and logits.size(1) == config.NUM_CLASSES:
                    return logits
            except Exception:
                pass

        # Final fallback: raise error if no evaluation method found
        raise RuntimeError(
            "Cannot reconstruct logits: model has no 'arcface_eval' dict or 'head' attribute. "
            "Ensure the model was trained with ArcFace and arcface_eval was saved."
        )

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if len(batch) == 3:
                images, labels, paths = batch
            else:
                images, labels = batch
                paths = None

            images = images.to(_device)

            # Handle TTA batches (B, N, C, H, W)
            if images.dim() == 5:
                b, n, c, h, w = images.shape
                flat_imgs = images.view(-1, c, h, w)
                outputs = model(flat_imgs)

                if isinstance(outputs, tuple):
                    logits_or_none = (
                        outputs[0] if outputs and isinstance(outputs[0], torch.Tensor) else None
                    )
                    embeddings = outputs[1] if len(outputs) > 1 else None
                    if logits_or_none is None and embeddings is not None:
                        logits = embeddings_to_logits(embeddings)
                    else:
                        logits = logits_or_none
                else:
                    # Embedding-only mode returns (B*N, embedding_dim).
                    if outputs.dim() == 2 and outputs.size(1) != config.NUM_CLASSES:
                        logits = embeddings_to_logits(outputs)
                    else:
                        logits = outputs

                if logits is None:
                    raise RuntimeError("Could not build logits from TTA outputs in evaluate_model")

                logits = logits.view(b, n, -1).mean(dim=1)

            else:
                outputs = model(images)
                if isinstance(outputs, tuple):
                    logits_candidate = (
                        outputs[0] if isinstance(outputs[0], torch.Tensor) else None
                    )
                    embeddings = outputs[1] if len(outputs) > 1 else None
                    if logits_candidate is None and embeddings is not None:
                        logits = embeddings_to_logits(embeddings)
                    else:
                        logits = logits_candidate
                else:
                    if outputs.dim() == 2 and outputs.size(1) != config.NUM_CLASSES:
                        logits = embeddings_to_logits(outputs)
                    else:
                        logits = outputs

                if logits is None:
                    raise RuntimeError("Could not build logits from model outputs in evaluate_model")

            probs = torch.softmax(logits, dim=1)
            _, predicted = logits.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Accuracy
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")

    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=config.CLASS_NAMES,
        digits=4
    )
    print(f"\n{report}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs(f'{config.OUTPUT_DIR}/plots', exist_ok=True)
    plt.savefig(f'{config.OUTPUT_DIR}/plots/{model_name}_confusion_matrix.png', dpi=150)
    plt.show()

    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    # Optional misclassified display
    if config.SHOW_MISCLASSIFIED:
        mis_idx = np.where(all_preds != all_labels)[0]

        if len(mis_idx) == 0:
            print("\nNo misclassified samples to display.")
        else:
            num_to_show = min(5, len(mis_idx))
            selected_indices = mis_idx[:num_to_show]

            misclassified_samples = []
            test_dataset = test_loader.dataset

            for idx in selected_indices:
                try:
                    if hasattr(test_dataset, 'samples'):
                        img_path, _ = test_dataset.samples[idx]
                    elif hasattr(test_dataset, 'image_paths'):
                        img_path = test_dataset.image_paths[idx]
                    else:
                        sample = test_dataset[idx]
                        if len(sample) >= 3:
                            img_path = sample[2]
                        else:
                            continue

                    true_label = int(all_labels[idx])
                    pred_label = int(all_preds[idx])
                    misclassified_samples.append((img_path, true_label, pred_label))
                except Exception:
                    continue

            if misclassified_samples:
                show_misclassified(misclassified_samples, config.CLASS_NAMES, max_items=5)
            else:
                print("\nCould not load misclassified sample images for display.")

    # -----------------------------------------------------------------------
    # Optional mislabeled audit
    # -----------------------------------------------------------------------
    if config.FLAG_MISLABELED:
        _run_mislabeled_audit(model, model_name)

    return {
        'model': model_name,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def _run_mislabeled_audit(model, model_name):
    """Flag likely mislabeled training images (optional, controlled by config.FLAG_MISLABELED)."""
    MOVE_FLAGGED = False  # Set True to move flagged files to review folder.
    CONF_THRESHOLD = 0.80
    _device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    review_dir = Path(config.OUTPUT_DIR) / "mislabeled_review"
    review_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(config.OUTPUT_DIR) / "models" / f"{model_name}_best.pth"

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Run training first or update model_name/checkpoint path.")
        return

    # Build the same model family used for training.
    if model_name and 'inception_resnet_v1' in model_name:
        if 'casia' in model_name:
            review_model = FaceEmbeddingModel(
                num_classes=config.NUM_CLASSES, pretrained='casia-webface'
            )
        else:
            review_model = FaceEmbeddingModel(num_classes=config.NUM_CLASSES, pretrained='vggface2')
    elif model_name == 'efficientnet_b3':
        review_model = EfficientNetEmbeddingModel(num_classes=config.NUM_CLASSES)
    else:
        review_model = get_model(model_name, num_classes=config.NUM_CLASSES)

    checkpoint = torch.load(checkpoint_path, map_location=_device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    review_model.load_state_dict(state_dict, strict=False)

    if isinstance(checkpoint, dict) and 'arcface_eval' in checkpoint:
        review_model.arcface_eval = checkpoint['arcface_eval']

    review_model = review_model.to(_device)
    review_model.eval()

    def to_logits(outputs, model_obj):
        if isinstance(outputs, tuple):
            logits_candidate = (
                outputs[0] if len(outputs) > 0 and isinstance(outputs[0], torch.Tensor) else None
            )
            embeddings = outputs[1] if len(outputs) > 1 else None
        else:
            logits_candidate = None
            embeddings = outputs

        if (isinstance(logits_candidate, torch.Tensor)
                and logits_candidate.dim() == 2
                and logits_candidate.size(1) == config.NUM_CLASSES):
            return logits_candidate

        if not isinstance(embeddings, torch.Tensor):
            raise RuntimeError("Model did not return tensor outputs for mislabeled audit.")

        if hasattr(model_obj, 'arcface_eval') and isinstance(model_obj.arcface_eval, dict):
            weight = model_obj.arcface_eval.get('weight', None)
            scale = float(model_obj.arcface_eval.get('scale', config.ARCFACE_SCALE))
            if weight is not None:
                weight = weight.to(embeddings.device)
                emb_norm = F.normalize(embeddings, dim=1)
                w_norm = F.normalize(weight, dim=1)
                return scale * (emb_norm @ w_norm.t())

        if hasattr(model_obj, 'head'):
            logits = model_obj.head(embeddings)
            if logits.dim() == 2 and logits.size(1) == config.NUM_CLASSES:
                return logits

        raise RuntimeError("Could not convert outputs to class logits for mislabeled audit.")

    class_to_idx = {name: idx for idx, name in enumerate(config.CLASS_NAMES)}
    train_root = Path(config.DATA_DIR) / "train"
    valid_ext = {'.jpg', '.jpeg', '.png'}
    flagged = []

    if not train_root.exists():
        print(f"Training directory not found: {train_root}")
        return

    transform = get_transforms('val', model_name=model_name)
    with torch.no_grad():
        for class_dir in sorted(train_root.iterdir()):
            if not class_dir.is_dir() or class_dir.name not in class_to_idx:
                continue

            true_idx = class_to_idx[class_dir.name]

            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in valid_ext or not img_path.is_file():
                    continue
                try:
                    image = Image.open(img_path).convert('RGB')
                    x = transform(image).unsqueeze(0).to(_device)
                    logits = to_logits(review_model(x), review_model)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred_idx = torch.max(probs, dim=1)

                    conf_val = float(conf.item())
                    pred_val = int(pred_idx.item())
                    if pred_val != true_idx and conf_val > CONF_THRESHOLD:
                        flagged.append(
                            (str(img_path), class_dir.name,
                             config.CLASS_NAMES[pred_val], conf_val)
                        )

                        if MOVE_FLAGGED:
                            target_dir = review_dir / class_dir.name
                            target_dir.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(img_path), str(target_dir / img_path.name))
                except Exception as e:
                    print(f"    Failed on {img_path}: {e}")

    print(
        f"\nFlagged images (pred != folder label and confidence > {CONF_THRESHOLD:.0%}): {len(flagged)}"
    )
    if flagged:
        for path, true_label, pred_label, conf_score in flagged:
            print(f"- {path} | true={true_label} | pred={pred_label} | conf={conf_score:.2%}")
        if MOVE_FLAGGED:
            print(f"\nMoved flagged files to: {review_dir}")
    else:
        print("No high-confidence mismatches found.")
