"""
Training Script for Pakistani Politician Image Classification
Trains multiple CNN models with MLflow tracking
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import timm
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from torch.cuda.amp import GradScaler, autocast

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Class names (16 Pakistani politicians)
CLASS_NAMES = sorted([
    "ahmed_sharif_chaudhry", "ahsan_iqbal", "altaf_hussain", "asfandyar_wali",
    "asif_ali_zardari", "barrister_gohar", "bilawal_bhutto", "chaudhry_shujaat",
    "fazlur_rehman", "imran_khan", "khawaja_asif", "maryam_nawaz",
    "nawaz_sharif", "pervez_musharraf", "shahbaz_sharif", "shehryar_afridi"
])

class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
idx_to_class = {idx: name for name, idx in class_to_idx.items()}


class PoliticianDataset(Dataset):
    """
    Custom Dataset for Pakistani Politician Images
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Root directory with class subfolders
            transform: Optional transform to be applied
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        # Load all images
        for class_name in CLASS_NAMES:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            
            for img_file in class_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_file), class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path


def get_transforms(split='train'):
    """
    Get transforms for train/val/test splits
    
    Args:
        split: 'train', 'val', or 'test'
    
    Returns:
        Transform pipeline
    """
    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if split == 'train':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])


def get_dataloaders(batch_size=32, num_workers=4):
    """
    Create train/val/test dataloaders
    
    Args:
        batch_size: Batch size
        num_workers: Number of worker threads
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = PoliticianDataset('dataset/train', transform=get_transforms('train'))
    val_dataset = PoliticianDataset('dataset/val', transform=get_transforms('val'))
    test_dataset = PoliticianDataset('dataset/test', transform=get_transforms('test'))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def get_model(model_name, num_classes=16):
    """
    Get pretrained model
    
    Args:
        model_name: Name of the model
        num_classes: Number of output classes
    
    Returns:
        Model
    """
    print(f"\nLoading model: {model_name}")
    
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
    
    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
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
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif 'efficientnet' in model_name or 'convnext' in model_name:
        for name, param in model.named_parameters():
            if 'classifier' not in name and 'head' not in name:
                param.requires_grad = False


def unfreeze_all(model):
    """Unfreeze all layers"""
    for param in model.parameters():
        param.requires_grad = True


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for images, labels, _ in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, epoch):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for images, labels, _ in pbar:
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
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def plot_training_curves(history, model_name):
    """Plot and save training curves"""
    epochs = range(1, len(history) + 1)
    
    # Loss curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [h['train_loss'] for h in history], 'b-', label='Train Loss')
    plt.plot(epochs, [h['val_loss'] for h in history], 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss Curve')
    plt.legend()
    plt.grid(True)
    
    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [h['train_acc']*100 for h in history], 'b-', label='Train Acc')
    plt.plot(epochs, [h['val_acc']*100 for h in history], 'r-', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model_name} - Accuracy Curve')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_training_curves.png', dpi=150)
    plt.close()


def train_model(model_name, config):
    """
    Train a single model
    
    Args:
        model_name: Name of the model
        config: Training configuration
    """
    print("\n" + "="*60)
    print(f"TRAINING: {model_name.upper()}")
    print("="*60)
    
    # Create directories
    os.makedirs('models/saved', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Get dataloaders
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=config['batch_size'],
        num_workers=4
    )
    
    # Initialize model
    model = get_model(model_name, num_classes=16)
    
    # Freeze backbone initially
    freeze_backbone(model, model_name)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs']
    )
    scaler = GradScaler()
    
    # MLflow tracking
    mlflow.set_experiment("Pakistani-Politician-Classifier")
    
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_params({
            "model": model_name,
            "epochs": config['epochs'],
            "batch_size": config['batch_size'],
            "lr": config['learning_rate'],
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealing",
            "num_classes": 16,
            "augmentation": True
        })
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        history = []
        
        for epoch in range(1, config['epochs'] + 1):
            # Unfreeze all layers after epoch 5
            if epoch == 6:
                print("\n>>> Unfreezing all layers for full fine-tuning")
                unfreeze_all(model)
                # Reset optimizer with new parameters
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=config['learning_rate'] * 0.1,  # Lower LR for fine-tuning
                    weight_decay=config['weight_decay']
                )
            
            # Train and validate
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, epoch
            )
            val_loss, val_acc = validate(model, val_loader, criterion, epoch)
            
            # Step scheduler
            scheduler.step()
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            # Save history
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'class_names': CLASS_NAMES
                }, f'models/saved/{model_name}_best.pth')
                print(f"✓ Best model saved! Val Acc: {val_acc*100:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= config['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        # Plot training curves
        plot_training_curves(history, model_name)
        
        # Log artifacts
        mlflow.log_artifact(f'plots/{model_name}_training_curves.png')
        mlflow.log_artifact(f'models/saved/{model_name}_best.pth')
        
        # Log model
        mlflow.pytorch.log_model(model, artifact_path="model")
        
        print(f"\n✓ Training completed for {model_name}")
        print(f"  Best Val Accuracy: {best_val_acc*100:.2f}%")


def main():
    """Main training function"""
    # Load configuration
    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)['training']
    
    print("\n" + "="*60)
    print("PAKISTANI POLITICIAN IMAGE CLASSIFIER - TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Models to train: {config['models_to_train']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print("="*60)
    
    # Train each model
    for model_name in config['models_to_train']:
        try:
            train_model(model_name, config)
        except Exception as e:
            print(f"\n✗ Error training {model_name}: {e}")
            continue
    
    print("\n" + "="*60)
    print("ALL TRAINING COMPLETED")
    print("="*60)
    print("\nTo view results, run: mlflow ui --port 5000")


if __name__ == "__main__":
    main()
