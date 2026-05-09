"""
datasets.py – PoliticianDataset, transforms, and DataLoader factory.

Exports:
  - PoliticianDataset   : custom Dataset wrapping the train/val/test folder tree
  - get_transforms      : returns torchvision transforms with model-specific normalisation
  - create_dataloaders  : convenience factory that creates all three DataLoaders + class weights
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from training.config import config


class PoliticianDataset(Dataset):
    """Custom Dataset for Pakistani Politician Images."""

    def __init__(self, root_dir, transform=None, return_path=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        # Return file paths when needed for inspection.
        self.return_path = return_path
        self.samples = []
        self.class_to_idx = {name: idx for idx, name in enumerate(config.CLASS_NAMES)}

        # Load all images
        for class_name in config.CLASS_NAMES:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            for img_file in class_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_file), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')

            # Validate image size (skip corrupted/tiny images)
            if min(image.size) < config.MIN_IMAGE_SIZE:
                # Image too small, return a random valid image from same class
                class_samples = [i for i, (_, lbl) in enumerate(self.samples) if lbl == label]
                if class_samples:
                    new_idx = np.random.choice(class_samples)
                    return self.__getitem__(new_idx)
                else:
                    # Fallback: return first sample
                    return self.__getitem__(0)

        except Exception as e:
            # Image failed to load, return a random valid image from same class
            print(f"Warning: Failed to load {img_path}: {e}")
            class_samples = [i for i, (_, lbl) in enumerate(self.samples) if lbl == label]
            if class_samples:
                new_idx = np.random.choice(class_samples)
                return self.__getitem__(new_idx)
            else:
                # Fallback: return first sample
                return self.__getitem__(0)

        if self.transform:
            image = self.transform(image)

        if self.return_path:
            return image, label, img_path

        return image, label


def get_transforms(split='train', model_name=None):
    """Get transforms for train/val/test with model-specific normalization."""
    model_name = model_name or getattr(config, 'MODEL_BACKBONE', None)

    if model_name and 'inception_resnet_v1' in model_name:
        # facenet-pytorch InceptionResnetV1 expects [-1, 1] style normalization.
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        # Keep ImageNet normalization for EfficientNet and other backbones.
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15))
        ])

    if split == 'test' and config.USE_TTA:
        def tta_stack(crops):
            stacked = []
            for c in crops:
                t = transforms.ToTensor()(c)
                t = normalize(t)
                stacked.append(t)
                flipped = transforms.functional.hflip(c)
                tf = transforms.ToTensor()(flipped)
                tf = normalize(tf)
                stacked.append(tf)
            return torch.stack(stacked)

        return transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.FiveCrop(config.IMAGE_SIZE),
            transforms.Lambda(lambda crops: tta_stack(crops))
        ])

    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        normalize
    ])


def create_dataloaders(model_name=None):
    """Create train/val/test dataloaders.

    Returns:
        train_loader, val_loader, test_loader, class_weights
        class_weights is a CPU tensor (or None). Move to device before use.
    """
    train_dataset = PoliticianDataset(
        f"{config.DATA_DIR}/train",
        transform=get_transforms('train', model_name=model_name)
    )
    val_dataset = PoliticianDataset(
        f"{config.DATA_DIR}/val",
        transform=get_transforms('val', model_name=model_name)
    )
    test_dataset = PoliticianDataset(
        f"{config.DATA_DIR}/test",
        transform=get_transforms('test', model_name=model_name)
    )

    # Compute class weights from folder counts if requested
    class_weights = None
    if config.USE_CLASS_WEIGHTS:
        try:
            counts = []
            train_root = Path(f"{config.DATA_DIR}/train")
            for cls in sorted(os.listdir(train_root)):
                p = train_root / cls
                if p.is_dir():
                    cnt = len([f for f in p.iterdir()
                                if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
                    counts.append(max(1, cnt))
            total = sum(counts) if counts else 1
            weights = [total / (len(counts) * c) for c in counts]
            class_weights = torch.tensor(weights, dtype=torch.float)
        except Exception:
            class_weights = None

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, config.BATCH_SIZE // 2),
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=max(1, config.BATCH_SIZE // 2),
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, class_weights
