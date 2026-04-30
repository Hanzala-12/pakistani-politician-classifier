"""
PAKISTANI POLITICIAN IMAGE CLASSIFIER - COMPLETE TRAINING PIPELINE
===================================================================
Standalone script for Kaggle - 100% self-contained, no external .py files needed

Upload this to Kaggle, enable GPU, and run all cells.
All trained models will be saved to /kaggle/working/

Author: ML Team
Version: 1.0
"""

# ============================================================================
# SECTION 1: SETUP & IMPORTS
# ============================================================================

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import timm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support
)

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# SECTION 2: CONFIGURATION
# ============================================================================

class Config:
    """Training configuration"""
    # Paths (adjust for Kaggle or local)
    DATA_DIR = "/kaggle/input/politician-dataset" if os.path.exists("/kaggle") else "dataset"
    OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle") else "output"
    
    # Dataset
    NUM_CLASSES = 16
    CLASS_NAMES = sorted([
        "ahmed_sharif_chaudhry", "ahsan_iqbal", "altaf_hussain", "asfandyar_wali",
        "asif_ali_zardari", "barrister_gohar", "bilawal_bhutto", "chaudhry_shujaat",
        "fazlur_rehman", "imran_khan", "khawaja_asif", "maryam_nawaz",
        "nawaz_sharif", "pervez_musharraf", "shahbaz_sharif", "shehryar_afridi"
    ])
    
    # Training hyperparameters
    EPOCHS = 30
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001
    EARLY_STOPPING_PATIENCE = 7
    
    # Models to train
    MODELS_TO_TRAIN = ["resnet50", "efficientnet_b3"]  # Add more: resnet152, vgg16, convnext_base
    
    # Image settings
    IMG_SIZE = 224
    
    # Augmentation
    NUM_AUGMENTATIONS = 3

config = Config()

# Create output directories
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{config.OUTPUT_DIR}/models", exist_ok=True)
os.makedirs(f"{config.OUTPUT_DIR}/plots", exist_ok=True)
os.makedirs(f"{config.OUTPUT_DIR}/results", exist_ok=True)

print(f"\n📁 Configuration:")
print(f"   Data: {config.DATA_DIR}")
print(f"   Output: {config.OUTPUT_DIR}")
print(f"   Models: {config.MODELS_TO_TRAIN}")
print(f"   Epochs: {config.EPOCHS}")
print(f"   Batch Size: {config.BATCH_SIZE}")

# ============================================================================
# SECTION 3: DATASET CLASS & DATALOADERS
# ============================================================================

class PoliticianDataset(Dataset):
    """Custom Dataset for Pakistani Politician Images"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
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
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(split='train'):
    """Get transforms for train/val/test"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if split == 'train':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])


def create_dataloaders():
    """Create train/val/test dataloaders"""
    train_dataset = PoliticianDataset(
        f"{config.DATA_DIR}/train",
        transform=get_transforms('train')
    )
    val_dataset = PoliticianDataset(
        f"{config.DATA_DIR}/val",
        transform=get_transforms('val')
    )
    test_dataset = PoliticianDataset(
        f"{config.DATA_DIR}/test",
        transform=get_transforms('test')
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    print(f"\n📊 Dataset sizes:")
    print(f"   Train: {len(train_dataset)}")
    print(f"   Val: {len(val_dataset)}")
    print(f"   Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

# ============================================================================
# SECTION 4: MODEL DEFINITIONS
# ============================================================================

def get_model(model_name, num_classes=16):
    """Get pretrained model"""
    print(f"\n🔧 Loading model: {model_name}")
    
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, num_classes)
    
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=True)
        model.fc = nn.Linear(2048, num_classes)
    
    elif model_name == 'efficientnet_b3':
        model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=num_classes)
    
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier[-1] = nn.Linear(4096, num_classes)
    
    elif model_name == 'convnext_base':
        model = timm.create_model('convnext_base', pretrained=True, num_classes=num_classes)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
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


def unfreeze_all(model):
    """Unfreeze all layers"""
    for param in model.parameters():
        param.requires_grad = True

# ============================================================================
# SECTION 5: TRAINING FUNCTIONS
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
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
    """Validate the model"""
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


def plot_training_curves(history, model_name):
    """Plot and save training curves"""
    epochs = range(1, len(history) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(epochs, [h['train_loss'] for h in history], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, [h['val_loss'] for h in history], 'r-', label='Val', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{model_name} - Loss Curve', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, [h['train_acc']*100 for h in history], 'b-', label='Train', linewidth=2)
    ax2.plot(epochs, [h['val_acc']*100 for h in history], 'r-', label='Val', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title(f'{model_name} - Accuracy Curve', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{config.OUTPUT_DIR}/plots/{model_name}_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# SECTION 6: MAIN TRAINING LOOP
# ============================================================================

def train_model(model_name, train_loader, val_loader):
    """Train a single model"""
    print(f"\n{'='*70}")
    print(f"🎯 TRAINING: {model_name.upper()}")
    print(f"{'='*70}")
    
    model = get_model(model_name, num_classes=config.NUM_CLASSES)
    freeze_backbone(model, model_name)
    
    criterion = nn.CrossEntropyLoss()
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
        # Unfreeze after epoch 5
        if epoch == 6:
            print("\n>>> Unfreezing all layers")
            unfreeze_all(model)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.LEARNING_RATE * 0.1,
                weight_decay=config.WEIGHT_DECAY
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
        
        print(f"Epoch {epoch}/{config.EPOCHS} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_names': config.CLASS_NAMES
            }, f'{config.OUTPUT_DIR}/models/{model_name}_best.pth')
            print(f"✅ Best model saved! Val Acc: {val_acc*100:.2f}%")
        else:
            patience_counter += 1
        
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n⏹️  Early stopping at epoch {epoch}")
            break
    
    plot_training_curves(history, model_name)
    
    return model, best_val_acc, history

# ============================================================================
# SECTION 7: EVALUATION
# ============================================================================

def evaluate_model(model, model_name, test_loader):
    """Evaluate model on test set"""
    print(f"\n{'='*70}")
    print(f"📊 EVALUATING: {model_name.upper()}")
    print(f"{'='*70}")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Accuracy
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
    
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
    plt.savefig(f'{config.OUTPUT_DIR}/plots/{model_name}_confusion_matrix.png', dpi=150)
    plt.show()
    
    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    return {
        'model': model_name,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# ============================================================================
# SECTION 8: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("🇵🇰 PAKISTANI POLITICIAN IMAGE CLASSIFIER")
    print("="*70)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders()
    
    # Train all models
    results = []
    trained_models = {}
    
    for model_name in config.MODELS_TO_TRAIN:
        try:
            model, best_val_acc, history = train_model(model_name, train_loader, val_loader)
            trained_models[model_name] = model
            
            # Evaluate
            eval_results = evaluate_model(model, model_name, test_loader)
            results.append(eval_results)
            
        except Exception as e:
            print(f"\n❌ Error training {model_name}: {e}")
            continue
    
    # Final comparison
    if results:
        print("\n" + "="*70)
        print("📈 FINAL RESULTS")
        print("="*70)
        
        df = pd.DataFrame(results)
        df['test_acc'] = df['test_acc'].apply(lambda x: f"{x*100:.2f}%")
        df['precision'] = df['precision'].apply(lambda x: f"{x:.4f}")
        df['recall'] = df['recall'].apply(lambda x: f"{x:.4f}")
        df['f1'] = df['f1'].apply(lambda x: f"{x:.4f}")
        
        print(df.to_string(index=False))
        df.to_csv(f'{config.OUTPUT_DIR}/results/model_comparison.csv', index=False)
        
        print(f"\n✅ All results saved to: {config.OUTPUT_DIR}")
        print(f"   - Models: {config.OUTPUT_DIR}/models/")
        print(f"   - Plots: {config.OUTPUT_DIR}/plots/")
        print(f"   - Results: {config.OUTPUT_DIR}/results/")
    
    print("\n🎉 TRAINING COMPLETE!")

# ============================================================================
# RUN TRAINING
# ============================================================================

if __name__ == "__main__":
    main()
