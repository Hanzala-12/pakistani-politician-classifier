"""
training.py – Training loops for ArcFace and generic (CrossEntropy) models.

Exports:
  - mixup_data          : MixUp data augmentation helper
  - mixup_criterion     : MixUp loss combiner
  - train_one_epoch     : single training epoch (generic CE path)
  - validate            : single validation epoch (generic CE path)
  - plot_training_curves: save loss/accuracy curves
  - train_arcface       : full ArcFace training loop with early stopping
  - train_model         : high-level dispatcher (ArcFace or generic CE)

ResNet-50 device bug fix
------------------------
The original notebook created the CrossEntropyLoss criterion with class_weights
while the weights were still on the CPU, causing a device mismatch when the
model was on the GPU.  In this module, class_weights are explicitly moved to the
correct device *inside* train_model before the criterion is created, so the fix
applies to every model that uses class weights (not just ResNet-50).
"""

import os
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from training.arcface import ArcFaceLoss
from training.config import config
from training.models import (
    EfficientNetEmbeddingModel,
    FaceEmbeddingModel,
    freeze_backbone,
    get_model,
)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# MixUp helpers
# ---------------------------------------------------------------------------

def mixup_data(x, y, alpha=0.2):
    """Apply MixUp to a batch. Returns (mixed_x, y_a, y_b, lam)."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, preds, y_a, y_b, lam):
    """Combined MixUp loss."""
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)


# ---------------------------------------------------------------------------
# Generic CE training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch):
    """Train for one epoch (generic CE path)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        use_mixup = config.USE_MIXUP and (random.random() < config.MIXUP_PROB)
        if use_mixup:
            images, targets_a, targets_b, lam = mixup_data(images, labels, config.MIXUP_ALPHA)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return running_loss / total, correct / total


def validate(model, val_loader, criterion, epoch):
    """Validate the model (generic CE path)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    return running_loss / total, correct / total


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(history, model_name):
    """Plot and save training curves."""
    # Support both history formats:
    # 1) list of dicts (generic CE flow)
    # 2) dict of lists (ArcFace flow)
    if isinstance(history, dict):
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])
        val_acc = history.get('val_acc', [])
        # ArcFace branch logs train accuracy in print statements but does not store it.
        train_acc = [None] * len(val_acc)
    else:
        train_loss = [h['train_loss'] for h in history]
        val_loss = [h['val_loss'] for h in history]
        train_acc = [h['train_acc'] for h in history]
        val_acc = [h['val_acc'] for h in history]

    epochs = range(1, len(val_loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, train_loss, 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Val', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{model_name} - Loss Curve', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    if len(train_acc) == len(val_acc) and train_acc and train_acc[0] is not None:
        ax2.plot(epochs, [a * 100 for a in train_acc], 'b-', label='Train', linewidth=2)
    ax2.plot(epochs, [a * 100 for a in val_acc], 'r-', label='Val', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title(f'{model_name} - Accuracy Curve', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(f'{config.OUTPUT_DIR}/plots', exist_ok=True)
    plt.savefig(f'{config.OUTPUT_DIR}/plots/{model_name}_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# ArcFace training loop
# ---------------------------------------------------------------------------

def train_arcface(model, train_loader, val_loader, cfg=None, _device=None):
    """Full ArcFace training loop with early stopping.

    Args:
        model   : FaceEmbeddingModel or EfficientNetEmbeddingModel
        train_loader, val_loader: DataLoaders
        cfg     : config object (defaults to global config)
        _device : torch.device (defaults to module-level device)

    Returns:
        model   : best-checkpoint model
        history : dict of lists
    """
    cfg = cfg or config
    _device = _device or device
    model.to(_device)

    criterion = ArcFaceLoss(
        num_classes=cfg.NUM_CLASSES,
        embedding_size=512,
        s=cfg.ARCFACE_SCALE,
        m=cfg.ARCFACE_MARGIN
    )
    criterion.to(_device)

    model_params = [p for p in model.parameters() if p.requires_grad]
    arcface_params = [p for p in criterion.margin_product.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {'params': model_params, 'lr': getattr(cfg, 'HEAD_LR', 1e-4)},
            {'params': arcface_params, 'lr': getattr(cfg, 'HEAD_LR', 1e-4)},
        ],
        weight_decay=getattr(cfg, 'WEIGHT_DECAY', 0.0)
    )

    best_val_acc = -1.0
    patience_counter = 0
    best_state = None

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for batch in train_loader:
            imgs, labels = batch[0].to(_device), batch[1].to(_device)
            labels = labels.long()
            optimizer.zero_grad()

            out = model(imgs)
            if isinstance(out, tuple):
                logits_train, embeddings = out
            else:
                embeddings = out

            logits = criterion.margin_product(embeddings, labels)
            loss = criterion(embeddings, labels)

            loss.backward()
            # Use the correct config name for gradient clipping.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=getattr(cfg, 'GRADIENT_CLIP_MAX_NORM', 1.0)
            )
            optimizer.step()

            preds = logits.argmax(dim=1)
            running_loss += loss.item() * imgs.size(0)
            running_corrects += (preds == labels).sum().item()
            total += imgs.size(0)

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = running_corrects / max(total, 1)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        model.eval()
        all_embeddings = []
        all_labels = []
        val_loss_accum = 0.0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs, labels = batch[0].to(_device), batch[1].to(_device)
                labels = labels.long()
                out = model(imgs)
                if isinstance(out, tuple):
                    logits_val_candidate, embeddings = out
                else:
                    embeddings = out

                val_logits = criterion.margin_product(embeddings, label=None)
                if epoch == 1 and val_total == 0:
                    print("Validation logits computed WITHOUT margin")

                try:
                    loss_val = criterion(embeddings, labels).item()
                except Exception:
                    loss_val = 0.0

                preds = val_logits.argmax(dim=1)
                all_embeddings.append(embeddings.detach().cpu())
                all_labels.append(labels.detach().cpu())
                val_loss_accum += loss_val * imgs.size(0)
                val_total += imgs.size(0)

        if val_total > 0:
            all_labels_np = torch.cat(all_labels).numpy()
            all_preds = torch.cat([
                F.softmax(criterion.margin_product(e.to(_device), label=None), dim=1)
                .argmax(dim=1).cpu()
                for e in all_embeddings
            ]).numpy()
            val_acc = (all_preds == all_labels_np).mean()
            val_loss = val_loss_accum / val_total
        else:
            val_acc = 0.0
            val_loss = 0.0

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(
            f"Epoch {epoch}/{cfg.EPOCHS} - "
            f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%"
        )

        if epoch == 6:
            try:
                model_params = [p for p in model.parameters() if p.requires_grad]
                arcface_params = [p for p in criterion.margin_product.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(
                    [
                        {
                            'params': model_params,
                            'lr': getattr(cfg, 'BACKBONE_UNFREEZE_LR',
                                          getattr(cfg, 'HEAD_LR', 1e-4))
                        },
                        {'params': arcface_params, 'lr': getattr(cfg, 'HEAD_LR', 1e-4)},
                    ],
                    weight_decay=getattr(cfg, 'WEIGHT_DECAY', 0.0)
                )
                print("Optimizer rebuilt at epoch 6 and ArcFace params retained in optimizer groups.")
            except Exception:
                print("Warning: Failed to rebuild optimizer at epoch 6 while preserving ArcFace params")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = {
                'model': deepcopy(model.state_dict()),
                'criterion': deepcopy(criterion.state_dict()),
                'optimizer': deepcopy(optimizer.state_dict())
            }
        else:
            patience_counter += 1

        if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch} (no improvement for {cfg.EARLY_STOPPING_PATIENCE} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state['model'])
        try:
            criterion.load_state_dict(best_state['criterion'])
        except Exception:
            pass

    try:
        model.arcface_eval = {
            'weight': criterion.margin_product.weight.detach().cpu().clone(),
            'scale': float(criterion.margin_product.s),
            'margin': float(criterion.margin_product.m)
        }
    except Exception:
        model.arcface_eval = None

    return model, history


# ---------------------------------------------------------------------------
# High-level dispatcher
# ---------------------------------------------------------------------------

def train_model(model_name, train_loader, val_loader, class_weights=None):
    """Train a single model.

    Dispatches to train_arcface for inception_resnet_v1 / efficientnet_b3 when
    config.USE_ARCFACE is True, otherwise uses the generic CE loop.

    ResNet-50 bug fix:
        class_weights (CPU tensor) is moved to the correct device before being
        passed to nn.CrossEntropyLoss, preventing the device mismatch that
        occurred in the original notebook.
    """
    print(f"\n{'='*70}")
    print(f"TRAINING: {model_name.upper()}")
    print(f"{'='*70}")

    # ArcFace flow for supported backbones
    if config.USE_ARCFACE and (
        (model_name and 'inception_resnet_v1' in model_name) or model_name == 'efficientnet_b3'
    ):
        print(f"Config.USE_ARCFACE detected -> running ArcFace flow for {model_name}")
        if model_name and 'inception_resnet_v1' in model_name:
            if 'casia' in model_name:
                model = FaceEmbeddingModel(num_classes=config.NUM_CLASSES, pretrained='casia-webface')
            else:
                model = FaceEmbeddingModel(num_classes=config.NUM_CLASSES, pretrained='vggface2')
        else:  # efficientnet_b3
            model = EfficientNetEmbeddingModel(num_classes=config.NUM_CLASSES)

        model, history = train_arcface(model, train_loader, val_loader, config)
        best_val_acc = max(history.get('val_acc', [0.0])) if isinstance(history, dict) else 0.0
        os.makedirs(f"{config.OUTPUT_DIR}/models", exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'arcface_eval': model.arcface_eval,
            'class_names': config.CLASS_NAMES
        }, f"{config.OUTPUT_DIR}/models/{model_name}_best.pth")
        print(f"ArcFace training complete. Best Val Acc: {best_val_acc*100:.2f}%")
        plot_training_curves(history, model_name)
        return model, best_val_acc, history

    # -----------------------------------------------------------------------
    # Generic CE flow (ResNet-50, ResNet-152, VGG-16, ConvNeXt, …)
    # -----------------------------------------------------------------------
    model = get_model(model_name, num_classes=config.NUM_CLASSES)
    freeze_backbone(model, model_name)

    # ----- ResNet-50 BUG FIX -----
    # Move class_weights to the same device as the model BEFORE creating the
    # criterion.  The original notebook left class_weights on the CPU while the
    # model was on the GPU, causing a RuntimeError during the first forward pass.
    if config.USE_CLASS_WEIGHTS and class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # ---- end fix ----

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, config.EPOCHS + 1):
        # Reduce overfitting: only unfreeze head + last ResNet block with discriminative LR.
        if epoch == 6 and "resnet" in model_name:
            print("\n>>> Unfreezing classifier head + layer4 only")
            for param in model.parameters():
                param.requires_grad = False

            head_params = list(model.fc.parameters())
            for param in head_params:
                param.requires_grad = True

            layer4_params = list(model.layer4.parameters())
            for param in layer4_params:
                param.requires_grad = True

            optimizer = optim.AdamW(
                [
                    {"params": head_params, "lr": 1e-4},
                    {"params": layer4_params, "lr": 1e-5},
                ],
                weight_decay=config.WEIGHT_DECAY,
            )
            # Keep LR decay continuous after optimizer swap.
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.EPOCHS - epoch + 1
            )

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, epoch)
        scheduler.step()

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        print(
            f"Epoch {epoch}/{config.EPOCHS} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            os.makedirs(f"{config.OUTPUT_DIR}/models", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_names': config.CLASS_NAMES
            }, f'{config.OUTPUT_DIR}/models/{model_name}_best.pth')
            print(f"Best model saved! Val Acc: {val_acc*100:.2f}%")
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    plot_training_curves(history, model_name)

    return model, best_val_acc, history
