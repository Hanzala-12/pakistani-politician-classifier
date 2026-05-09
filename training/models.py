"""
models.py – Model definitions for the Pakistani Politician Classifier pipeline.

Includes:
  - FaceEmbeddingModel   : InceptionResnetV1 wrapper for ArcFace training
  - EfficientNetEmbeddingModel : EfficientNet-B3 wrapper for ArcFace training
  - get_model            : factory for ResNet-50, ResNet-152, VGG-16, ConvNeXt, etc.
  - freeze_backbone      : freeze all but the head
  - unfreeze_all         : unfreeze everything
  - save_checkpoint / load_checkpoint : training recovery helpers
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import timm
from facenet_pytorch import InceptionResnetV1

from training.config import config


# ---------------------------------------------------------------------------
# ArcFace-compatible embedding wrappers
# ---------------------------------------------------------------------------

class FaceEmbeddingModel(nn.Module):
    """Wraps InceptionResnetV1 to expose embeddings and a simple head for ArcFace training."""

    def __init__(self, num_classes, pretrained='vggface2', dropout=0.5):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained=pretrained, classify=False, device=None)
        self.embedding_size = 512
        self.head = nn.Identity() if getattr(config, 'USE_ARCFACE', False) else nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embedding_size, num_classes),
        )

    def forward(self, x, return_embeddings=False):
        embeddings = self.backbone(x)

        if getattr(config, 'USE_ARCFACE', False):
            return embeddings  # always return only embeddings when ArcFace is on

        if return_embeddings:
            return embeddings

        logits = self.head(embeddings)
        return logits, embeddings


class EfficientNetEmbeddingModel(nn.Module):
    """Wraps EfficientNet-B3 and projects features to 512-d embeddings for ArcFace."""

    def __init__(self, num_classes, pretrained=True, embedding_size=512, dropout=0.5):
        super().__init__()
        # Use torchvision EfficientNet-B3
        try:
            backbone = tv_models.efficientnet_b3(
                weights=tv_models.EfficientNet_B3_Weights.IMAGENET1K_V1
            )
        except Exception:
            # Fallback for older torchvision versions
            backbone = tv_models.efficientnet_b3(pretrained=pretrained)
        # Remove classifier
        self.backbone = backbone
        if hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        self.embedding_size = embedding_size
        # EfficientNet-B3 feature dim = 1536
        feat_dim = 1536
        self.project = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ReLU()
        )
        self.head = nn.Linear(self.embedding_size, num_classes)

    def forward(self, x, return_embeddings=False):
        # pass through backbone features -> pool -> flatten
        features = self.backbone(x)  # (B, feat_dim) after classifier removed
        embeddings = self.project(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        if return_embeddings or getattr(config, 'USE_ARCFACE', False):
            return embeddings

        logits = self.head(embeddings)
        return logits, embeddings


# ---------------------------------------------------------------------------
# Classic classification-head factory
# ---------------------------------------------------------------------------

def get_model(model_name, num_classes=16):
    """Get pretrained model"""
    print(f"\nLoading model: {model_name}")

    if model_name == 'resnet50':
        model = tv_models.resnet50(pretrained=True)
        # Add dropout to reduce overfitting on small datasets.
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )

    elif model_name == 'resnet152':
        model = tv_models.resnet152(pretrained=True)
        # Add dropout to reduce overfitting on small datasets.
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )

    elif model_name == 'efficientnet_b3':
        model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=num_classes)

    elif model_name == 'vgg16':
        model = tv_models.vgg16(pretrained=True)
        model.classifier[-1] = nn.Linear(4096, num_classes)

    elif model_name == 'convnext_base':
        model = timm.create_model('convnext_base', pretrained=True, num_classes=num_classes)

    elif model_name and 'inception_resnet_v1' in model_name:
        # Face-pretrained backbone (VGGFace2).
        backbone = InceptionResnetV1(pretrained='vggface2')
        model = nn.Sequential(
            backbone,
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model.to(device)


def freeze_backbone(model, model_name):
    """Freeze backbone layers"""
    if 'resnet' in model_name:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    elif 'vgg' in model_name:
        for param in model.features.parameters():
            param.requires_grad = False
    elif 'efficientnet' in model_name or 'convnext' in model_name:
        for name, param in model.named_parameters():
            if 'classifier' not in name and 'head' not in name:
                param.requires_grad = False
    elif 'inception_resnet_v1' in model_name:
        for param in model[0].parameters():
            param.requires_grad = False
        for param in model[1:].parameters():
            param.requires_grad = True


def unfreeze_all(model):
    """Unfreeze all layers"""
    for param in model.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, checkpoint_path):
    """Save training checkpoint for recovery"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_acc': val_acc
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load training checkpoint for recovery"""
    if not os.path.exists(checkpoint_path):
        return 0, 0.0

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"Resumed from epoch {checkpoint['epoch']} with val_acc={checkpoint['val_acc']:.4f}")
    return checkpoint['epoch'], checkpoint['val_acc']
