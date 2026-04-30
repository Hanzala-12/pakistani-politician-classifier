"""
Dataset Splitting Script
Splits raw data into train/val/test sets with stratification
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def split_dataset(
    raw_dir="data/raw",
    output_dir="dataset",
    train_ratio=0.75,
    val_ratio=0.15,
    test_ratio=0.10,
    random_state=42
):
    """
    Split dataset into train/val/test with stratification
    
    Args:
        raw_dir: Source directory with class folders
        output_dir: Destination directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random seed for reproducibility
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Create output directories
    train_dir = Path(output_dir) / "train"
    val_dir = Path(output_dir) / "val"
    test_dir = Path(output_dir) / "test"
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("DATASET SPLITTING")
    print("="*60)
    print(f"Train: {train_ratio*100:.0f}% | Val: {val_ratio*100:.0f}% | Test: {test_ratio*100:.0f}%")
    print("="*60)
    
    summary = {
        "train": {},
        "val": {},
        "test": {}
    }
    
    # Process each class
    for class_name in sorted(os.listdir(raw_dir)):
        class_path = Path(raw_dir) / class_name
        
        if not class_path.is_dir():
            continue
        
        print(f"\nProcessing: {class_name}")
        
        # Get all image files
        images = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        if len(images) == 0:
            print(f"  ⚠️  No images found, skipping...")
            continue
        
        # Create class subdirectories
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)
        (test_dir / class_name).mkdir(exist_ok=True)
        
        # First split: train vs (val+test)
        train_images, temp_images = train_test_split(
            images,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            shuffle=True
        )
        
        # Second split: val vs test
        val_images, test_images = train_test_split(
            temp_images,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=random_state,
            shuffle=True
        )
        
        # Copy files to respective directories
        for img in tqdm(train_images, desc=f"  Copying train", leave=False):
            src = class_path / img
            dst = train_dir / class_name / img
            shutil.copy2(src, dst)
        
        for img in tqdm(val_images, desc=f"  Copying val", leave=False):
            src = class_path / img
            dst = val_dir / class_name / img
            shutil.copy2(src, dst)
        
        for img in tqdm(test_images, desc=f"  Copying test", leave=False):
            src = class_path / img
            dst = test_dir / class_name / img
            shutil.copy2(src, dst)
        
        # Store counts
        summary["train"][class_name] = len(train_images)
        summary["val"][class_name] = len(val_images)
        summary["test"][class_name] = len(test_images)
        
        print(f"  ✓ Train: {len(train_images)} | Val: {len(val_images)} | Test: {len(test_images)}")
    
    # Print summary table
    print("\n" + "="*60)
    print("SPLIT SUMMARY")
    print("="*60)
    print(f"{'Class Name':<30} {'Train':>10} {'Val':>10} {'Test':>10} {'Total':>10}")
    print("-"*60)
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    for class_name in sorted(summary["train"].keys()):
        train_count = summary["train"][class_name]
        val_count = summary["val"][class_name]
        test_count = summary["test"][class_name]
        total = train_count + val_count + test_count
        
        print(f"{class_name:<30} {train_count:>10} {val_count:>10} {test_count:>10} {total:>10}")
        
        total_train += train_count
        total_val += val_count
        total_test += test_count
    
    print("-"*60)
    total_all = total_train + total_val + total_test
    print(f"{'TOTAL':<30} {total_train:>10} {total_val:>10} {total_test:>10} {total_all:>10}")
    print(f"{'PERCENTAGE':<30} {total_train/total_all*100:>9.1f}% {total_val/total_all*100:>9.1f}% {total_test/total_all*100:>9.1f}% {100.0:>9.1f}%")
    print("="*60)
    
    print("\n✓ Dataset split completed successfully!")


def main():
    """Main execution function"""
    split_dataset()


if __name__ == "__main__":
    main()
