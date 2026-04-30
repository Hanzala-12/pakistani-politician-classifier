"""
Data Augmentation Script
Applies augmentation to training images using albumentations
"""

import os
import cv2
from pathlib import Path
from tqdm import tqdm
import albumentations as A


def create_augmentation_pipeline():
    """
    Create albumentations augmentation pipeline
    
    Returns:
        Augmentation pipeline
    """
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=30, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.7
        ),
        A.RandomScale(scale_limit=0.3, p=0.5),
        A.RandomResizedCrop(
            height=224,
            width=224,
            scale=(0.7, 1.0),
            p=0.8
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5
        ),
        A.GridDistortion(p=0.3),
    ])


def augment_dataset(train_dir="dataset/train", num_augmentations=3):
    """
    Apply augmentation to training images
    
    Args:
        train_dir: Training directory path
        num_augmentations: Number of augmented copies per image
    """
    print("\n" + "="*60)
    print("DATA AUGMENTATION")
    print("="*60)
    print(f"Generating {num_augmentations} augmented copies per image")
    print("="*60)
    
    augmentation_pipeline = create_augmentation_pipeline()
    summary = {}
    
    train_path = Path(train_dir)
    
    # Process each class
    for class_name in sorted(os.listdir(train_path)):
        class_path = train_path / class_name
        
        if not class_path.is_dir():
            continue
        
        print(f"\nAugmenting: {class_name}")
        
        # Get original images (not augmented ones)
        images = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            and '_aug_' not in f
        ]
        
        original_count = len(images)
        augmented_count = 0
        
        for img_file in tqdm(images, desc=f"  Processing"):
            img_path = class_path / img_file
            
            try:
                # Read image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Generate augmented copies
                base_name = img_path.stem
                ext = img_path.suffix
                
                for aug_idx in range(1, num_augmentations + 1):
                    # Apply augmentation
                    augmented = augmentation_pipeline(image=image)
                    aug_image = augmented['image']
                    
                    # Save augmented image
                    aug_filename = f"{base_name}_aug_{aug_idx}{ext}"
                    aug_path = class_path / aug_filename
                    
                    aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(aug_path), aug_image_bgr)
                    augmented_count += 1
                    
            except Exception as e:
                print(f"    Error augmenting {img_file}: {e}")
                continue
        
        total_count = original_count + augmented_count
        summary[class_name] = {
            'original': original_count,
            'augmented': augmented_count,
            'total': total_count
        }
        
        print(f"  ✓ Original: {original_count} | Augmented: {augmented_count} | Total: {total_count}")
    
    # Print summary
    print("\n" + "="*60)
    print("AUGMENTATION SUMMARY")
    print("="*60)
    print(f"{'Class Name':<30} {'Original':>12} {'Augmented':>12} {'Total':>12}")
    print("-"*60)
    
    total_original = 0
    total_augmented = 0
    total_all = 0
    
    for class_name in sorted(summary.keys()):
        orig = summary[class_name]['original']
        aug = summary[class_name]['augmented']
        total = summary[class_name]['total']
        
        print(f"{class_name:<30} {orig:>12} {aug:>12} {total:>12}")
        
        total_original += orig
        total_augmented += aug
        total_all += total
    
    print("-"*60)
    print(f"{'TOTAL':<30} {total_original:>12} {total_augmented:>12} {total_all:>12}")
    print("="*60)
    
    print(f"\n✓ Augmentation completed!")
    print(f"  Training set expanded from {total_original} to {total_all} images")
    print(f"  Expansion factor: {total_all/total_original:.2f}x")


def main():
    """Main execution function"""
    augment_dataset()


if __name__ == "__main__":
    main()
