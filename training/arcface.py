"""
arcface.py – ArcMarginProduct and ArcFaceLoss.

These classes implement the ArcFace angular margin loss used for face-recognition
style training of the InceptionResnetV1 and EfficientNet backbones.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """ArcFace margin product for angular softmax."""

    def __init__(self, in_features, out_features, s=64.0, m=0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_normal_(self.weight)
        self.s = s
        self.m = m
        print(
            f"ArcMarginProduct initialized: in_features={in_features}, "
            f"out_features={out_features}, s={s}, m={m}"
        )

    def forward(self, embeddings, labels=None, label=None):
        """Compute ArcFace logits with optional angular margin.

        If labels is None, return no-margin logits for validation/evaluation.
        """
        if labels is None:
            labels = label

        embeddings = F.normalize(embeddings, dim=1)
        W_norm = F.normalize(self.weight, dim=1)
        cosine = embeddings @ W_norm.t()

        if labels is None:
            return self.s * cosine

        labels = labels.long()
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        one_hot = F.one_hot(labels, num_classes=self.weight.size(0)).bool()
        theta_m = theta + self.m
        theta_m = torch.where(one_hot, theta_m, theta)
        logits = self.s * torch.cos(theta_m)
        return logits


class ArcFaceLoss(nn.Module):
    """ArcFace loss with label smoothing."""

    def __init__(self, num_classes, embedding_size=512, s=64.0, m=0.3):
        super().__init__()
        self.margin_product = ArcMarginProduct(embedding_size, num_classes, s, m)
        self.num_classes = num_classes
        self.embedding_size = embedding_size

    def forward(self, embeddings, labels):
        """Compute ArcFace loss.

        Args:
            embeddings: (batch_size, embedding_size) – embeddings from backbone
            labels: (batch_size,) – ground truth class labels

        Returns:
            loss: scalar cross-entropy loss with ArcFace margin logits
        """
        labels = labels.long()
        logits = self.margin_product(embeddings, labels)
        return F.cross_entropy(logits, labels, label_smoothing=0.1)
