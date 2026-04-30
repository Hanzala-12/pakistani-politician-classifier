"""
PAKISTANI POLITICIAN IMAGE CLASSIFIER - COMPLETE TRAINING PIPELINE
===================================================================
🚀 100% STANDALONE - INCLUDES EVERYTHING:
   ✅ Data Collection (Web Scraping + Face Detection)
   ✅ Data Splitting (Train/Val/Test)
   ✅ Data Augmentation
   ✅ Model Training (5 CNN Models)
   ✅ Evaluation & Results

📋 INSTRUCTIONS:
   1. Upload this file to Kaggle
   2. Enable GPU (Settings → Accelerator → GPU)
   3. Run all cells (or run as script: python kaggle_training_complete.py)
   4. Download results from /kaggle/working/

⏱️ ESTIMATED TIME: 4-6 hours (with GPU)

📦 OUTPUT:
   - /kaggle/working/models/*.pth (trained models)
   - /kaggle/working/plots/*.png (training curves, confusion matrices)
   - /kaggle/working/results/*.csv (evaluation results)

Author: ML Team
Version: 2.0 - COMPLETE STANDALONE
"""

# ============================================================================
# SECTION 1: SETUP & IMPORTS
# ============================================================================

print("="*70)
print("🇵🇰 PAKISTANI POLITICIAN IMAGE CLASSIFIER")
print("="*70)
print("📦 Installing required packages...")

# Install packages if needed (for Kaggle)
import subprocess
import sys

def install_package(package):
    try:
        __import__(package.split('[')[0])
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

# Install required packages
packages = [
    'icrawler',
    'albumentations',
    'timm',
]

for pkg in packages:
    install_package(pkg)

print("✅ All packages ready!")

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
import shutil
from collections import defaultdict

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

# For data collection
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler

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
print(f"\n🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# SECTION 2: CONFIGURATION
# ============================================================================

class Config:
    """Training configuration"""
    # Paths - use local dataset after collection
    DATA_DIR = "dataset"  # Local dataset after splitting
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
    EPOCHS = 20  # Reduced for faster training
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001
    EARLY_STOPPING_PATIENCE = 5
    
    # Models to train (start with 2 for speed)
    MODELS_TO_TRAIN = ["resnet50", "efficientnet_b3"]  
    # Add more if needed: "resnet152", "vgg16", "convnext_base"
    
    # Image settings
    IMG_SIZE = 224
    
    # Augmentation
    NUM_AUGMENTATIONS = 2  # 2x augmentation for speed

config = Config()

# Create output directories
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{config.OUTPUT_DIR}/models", exist_ok=True)
os.makedirs(f"{config.OUTPUT_DIR}/plots", exist_ok=True)
os.makedirs(f"{config.OUTPUT_DIR}/results", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)
os.makedirs("dataset/train", exist_ok=True)
os.makedirs("dataset/val", exist_ok=True)
os.makedirs("dataset/test", exist_ok=True)

print(f"\n📁 Configuration:")
print(f"   Data: {config.DATA_DIR}")
print(f"   Output: {config.OUTPUT_DIR}")
print(f"   Models: {config.MODELS_TO_TRAIN}")
print(f"   Epochs: {config.EPOCHS}")
print(f"   Batch Size: {config.BATCH_SIZE}")

# ============================================================================
# SECTION 2.5: DATA COLLECTION
# ============================================================================

print("\n" + "="*70)
print("📥 PHASE 1: DATA COLLECTION")
print("="*70)

# 16 Pakistani Politicians with search queries
POLITICIANS = {
    "imran_khan": ["Imran Khan Pakistan PM face", "Imran Khan PTI"],
    "nawaz_sharif": ["Nawaz Sharif Pakistan PM", "Nawaz Sharif PML-N"],
    "asif_ali_zardari": ["Asif Ali Zardari Pakistan President", "Zardari PPP"],
    "bilawal_bhutto": ["Bilawal Bhutto Zardari Pakistan", "Bilawal PPP Chairman"],
    "shahbaz_sharif": ["Shahbaz Sharif Pakistan PM", "Shehbaz Sharif PML-N"],
    "maryam_nawaz": ["Maryam Nawaz Pakistan", "Maryam Nawaz PML-N"],
    "fazlur_rehman": ["Fazlur Rehman Pakistan", "Maulana Fazlur Rehman JUI"],
    "asfandyar_wali": ["Asfandyar Wali Khan Pakistan", "Asfandyar Wali ANP"],
    "altaf_hussain": ["Altaf Hussain MQM Pakistan", "Altaf Hussain London"],
    "chaudhry_shujaat": ["Chaudhry Shujaat Hussain Pakistan", "Shujaat Hussain PML-Q"],
    "pervez_musharraf": ["Pervez Musharraf Pakistan President", "General Musharraf"],
    "shehryar_afridi": ["Shehryar Afridi Pakistan PTI", "Shehryar Khan Afridi"],
    "khawaja_asif": ["Khawaja Asif Pakistan PML-N", "Khawaja Muhammad Asif"],
    "ahsan_iqbal": ["Ahsan Iqbal Pakistan PML-N", "Ahsan Iqbal Minister"],
    "barrister_gohar": ["Barrister Gohar Ali Khan PTI", "Gohar Ali Khan Pakistan"],
    "ahmed_sharif_chaudhry": ["Ahmed Sharif Chaudhry ISPR Pakistan", "DG ISPR Ahmed Sharif"]
}

def crawl_images(name, queries, max_per_query=40):
    """Crawl images for a politician"""
    out_dir = f"data/raw/{name}"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n📸 Collecting: {name}")
    
    for idx, query in enumerate(queries):
        try:
            # Bing crawler
            bing_crawler = BingImageCrawler(
                storage={"root_dir": out_dir},
                downloader_threads=2
            )
            bing_crawler.crawl(keyword=query, max_num=max_per_query)
            print(f"  ✓ Bing: {query}")
        except Exception as e:
            print(f"  ✗ Bing failed: {e}")
        
        try:
            # Google crawler
            google_crawler = GoogleImageCrawler(
                storage={"root_dir": out_dir},
                downloader_threads=2
            )
            google_crawler.crawl(keyword=query, max_num=max_per_query//2)
            print(f"  ✓ Google: {query}")
        except Exception as e:
            print(f"  ✗ Google failed: {e}")

def filter_images_with_faces(data_dir="data/raw", min_face_ratio=0.15):
    """Filter images to keep only those with detectable faces"""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    summary = {}
    
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        print(f"\n🔍 Filtering: {class_name}")
        
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        kept = 0
        removed = 0
        
        for img_file in tqdm(images, desc=f"  Processing"):
            img_path = os.path.join(class_path, img_file)
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    os.remove(img_path)
                    removed += 1
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                
                if len(faces) == 0:
                    os.remove(img_path)
                    removed += 1
                    continue
                
                img_area = img.shape[0] * img.shape[1]
                face_area = sum([w * h for (x, y, w, h) in faces])
                face_ratio = face_area / img_area
                
                if face_ratio < min_face_ratio:
                    os.remove(img_path)
                    removed += 1
                else:
                    kept += 1
                    
            except Exception as e:
                try:
                    os.remove(img_path)
                    removed += 1
                except:
                    pass
        
        summary[class_name] = kept
        print(f"  ✓ Kept: {kept} | Removed: {removed}")
    
    return summary

# Run data collection
print("\n🌐 Starting web scraping...")
for politician_name, queries in POLITICIANS.items():
    crawl_images(politician_name, queries, max_per_query=40)

print("\n🔍 Filtering images with face detection...")
collection_summary = filter_images_with_faces()

# Print summary
print("\n" + "="*70)
print("📊 COLLECTION SUMMARY")
print("="*70)
print(f"{'Class':<30} {'Images':>10}")
print("-"*70)
total = 0
for name, count in sorted(collection_summary.items()):
    print(f"{name:<30} {count:>10}")
    total += count
print("-"*70)
print(f"{'TOTAL':<30} {total:>10}")
print(f"{'AVERAGE':<30} {total/len(collection_summary):>10.1f}")

# ============================================================================
# SECTION 2.6: DATA SPLITTING
# ============================================================================

print("\n" + "="*70)
print("✂️  PHASE 2: DATASET SPLITTING")
print("="*70)

def split_dataset(raw_dir="data/raw", output_dir="dataset", 
                  train_ratio=0.75, val_ratio=0.15, test_ratio=0.10):
    """Split dataset into train/val/test"""
    
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{output_dir}/{split}", exist_ok=True)
    
    split_summary = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    
    for class_name in sorted(os.listdir(raw_dir)):
        class_path = Path(raw_dir) / class_name
        if not class_path.is_dir():
            continue
        
        print(f"\n📂 Splitting: {class_name}")
        
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(images) == 0:
            continue
        
        # Create class subdirectories
        for split in ['train', 'val', 'test']:
            os.makedirs(f"{output_dir}/{split}/{class_name}", exist_ok=True)
        
        # Split
        train_imgs, temp_imgs = train_test_split(
            images, test_size=(val_ratio + test_ratio), random_state=42
        )
        val_imgs, test_imgs = train_test_split(
            temp_imgs, test_size=test_ratio/(val_ratio + test_ratio), random_state=42
        )
        
        # Copy files
        for img in train_imgs:
            shutil.copy2(class_path / img, f"{output_dir}/train/{class_name}/{img}")
        for img in val_imgs:
            shutil.copy2(class_path / img, f"{output_dir}/val/{class_name}/{img}")
        for img in test_imgs:
            shutil.copy2(class_path / img, f"{output_dir}/test/{class_name}/{img}")
        
        split_summary[class_name]['train'] = len(train_imgs)
        split_summary[class_name]['val'] = len(val_imgs)
        split_summary[class_name]['test'] = len(test_imgs)
        
        print(f"  ✓ Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")
    
    return split_summary

split_summary = split_dataset()

# Print split summary
print("\n" + "="*70)
print("📊 SPLIT SUMMARY")
print("="*70)
print(f"{'Class':<30} {'Train':>10} {'Val':>10} {'Test':>10}")
print("-"*70)
for name in sorted(split_summary.keys()):
    print(f"{name:<30} {split_summary[name]['train']:>10} "
          f"{split_summary[name]['val']:>10} {split_summary[name]['test']:>10}")

# ============================================================================
# SECTION 2.7: DATA AUGMENTATION
# ============================================================================

print("\n" + "="*70)
print("🎨 PHASE 3: DATA AUGMENTATION")
print("="*70)

try:
    import albumentations as A
    
    def create_augmentation_pipeline():
        """Create augmentation pipeline"""
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        ])
    
    def augment_dataset(train_dir="dataset/train", num_augmentations=2):
        """Apply augmentation to training images"""
        aug_pipeline = create_augmentation_pipeline()
        aug_summary = {}
        
        for class_name in sorted(os.listdir(train_dir)):
            class_path = Path(train_dir) / class_name
            if not class_path.is_dir():
                continue
            
            print(f"\n🎨 Augmenting: {class_name}")
            
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png')) and '_aug_' not in f]
            
            original_count = len(images)
            augmented_count = 0
            
            for img_file in tqdm(images, desc="  Processing"):
                img_path = class_path / img_file
                
                try:
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue
                    
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    base_name = img_path.stem
                    ext = img_path.suffix
                    
                    for aug_idx in range(1, num_augmentations + 1):
                        augmented = aug_pipeline(image=image)
                        aug_image = augmented['image']
                        
                        aug_filename = f"{base_name}_aug_{aug_idx}{ext}"
                        aug_path = class_path / aug_filename
                        
                        aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(aug_path), aug_image_bgr)
                        augmented_count += 1
                        
                except Exception as e:
                    continue
            
            aug_summary[class_name] = {
                'original': original_count,
                'augmented': augmented_count,
                'total': original_count + augmented_count
            }
            
            print(f"  ✓ Original: {original_count} | Augmented: {augmented_count}")
        
        return aug_summary
    
    aug_summary = augment_dataset(num_augmentations=2)
    
    print("\n✅ Augmentation complete!")
    
except Exception as e:
    print(f"\n⚠️  Augmentation skipped: {e}")
    print("   Continuing with original images only...")

print("\n✅ DATA PREPARATION COMPLETE!")
print("="*70)

# ============================================================================
# SECTION 3: DATASET CLASS & DATALOADERS
# ============================================================================

print("\n" + "="*70)
print("📊 PHASE 4: CREATING DATALOADERS")
print("="*70)

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

print("\n" + "="*70)
print("🧠 PHASE 5: MODEL DEFINITIONS")
print("="*70)

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

print("\n" + "="*70)
print("🏋️ PHASE 6: MODEL TRAINING")
print("="*70)

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

print("\n" + "="*70)
print("📊 PHASE 7: MODEL EVALUATION")
print("="*70)

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
    print("� STARTING TRAINING PIPELINE")
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
            import traceback
            traceback.print_exc()
            continue
    
    # Final comparison
    if results:
        print("\n" + "="*70)
        print("🏆 FINAL RESULTS")
        print("="*70)
        
        df = pd.DataFrame(results)
        df['test_acc'] = df['test_acc'].apply(lambda x: f"{x*100:.2f}%")
        df['precision'] = df['precision'].apply(lambda x: f"{x:.4f}")
        df['recall'] = df['recall'].apply(lambda x: f"{x:.4f}")
        df['f1'] = df['f1'].apply(lambda x: f"{x:.4f}")
        
        print("\n" + df.to_string(index=False))
        df.to_csv(f'{config.OUTPUT_DIR}/results/model_comparison.csv', index=False)
        
        print(f"\n✅ All results saved to: {config.OUTPUT_DIR}")
        print(f"   📁 Models: {config.OUTPUT_DIR}/models/")
        print(f"   📊 Plots: {config.OUTPUT_DIR}/plots/")
        print(f"   📈 Results: {config.OUTPUT_DIR}/results/")
        
        print("\n" + "="*70)
        print("🎉 TRAINING PIPELINE COMPLETE!")
        print("="*70)
        print("\n📦 DOWNLOAD THESE FOLDERS:")
        print(f"   1. {config.OUTPUT_DIR}/models/  (trained model weights)")
        print(f"   2. {config.OUTPUT_DIR}/plots/   (training curves, confusion matrices)")
        print(f"   3. {config.OUTPUT_DIR}/results/ (evaluation reports)")
        print("\n💡 TIP: In Kaggle, these are in /kaggle/working/")
        print("="*70)
    else:
        print("\n❌ No models were successfully trained!")

# ============================================================================
# RUN TRAINING
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Check the error above and try again")
