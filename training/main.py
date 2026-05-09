"""
main.py – Orchestrates the full Pakistani Politician Classifier training pipeline.

Usage (from project root):
    python training/main.py

Or from a notebook:
    from training.main import main
    main()

Pipeline stages:
  1. Install required packages (Kaggle-friendly)
  2. Load Kaggle datasets into local data/ directories
  3. Merge data/raw + data/raw2 → data/raw_merged
  4. MTCNN face alignment + pHash deduplication (or Haar-cascade filter)
  5. Stratified train/val/test split
  6. Offline augmentation for under-represented classes
  7. Train all models listed in config.MODELS_TO_TRAIN
  8. Evaluate each model and print a final comparison table
  9. Export results CSV
"""

import os
import shutil
import warnings
from copy import deepcopy

import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from training.config import config
from training.data_prep import (
    align_faces_mtcnn,
    deduplicate_phash,
    filter_images_with_faces,
    load_kaggle_data,
    merge_raw_folders,
    run_offline_augmentation,
    split_dataset,
)
from training.datasets import create_dataloaders
from training.evaluate import evaluate_model
from training.training import train_model
from training.utils import cleanup_previous_results, install_package, set_seed


def _prepare_output_dirs():
    """Create required output directories."""
    for sub in ('models', 'plots', 'results'):
        os.makedirs(os.path.join(config.OUTPUT_DIR, sub), exist_ok=True)


def _install_dependencies():
    """Install any missing packages (needed on fresh Kaggle kernels)."""
    packages = ['albumentations', 'timm', 'facenet-pytorch', 'imagehash']
    for pkg in packages:
        install_package(pkg)


def main():
    """Main execution function – runs the complete training pipeline."""
    print("\n" + "="*70)
    print("STARTING TRAINING PIPELINE")
    print("="*70)

    # -----------------------------------------------------------------------
    # 0. Setup
    # -----------------------------------------------------------------------
    set_seed(42)
    _install_dependencies()
    _prepare_output_dirs()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # -----------------------------------------------------------------------
    # 1. Load data from Kaggle (or confirm local data exists)
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("PHASE 1: LOADING DATA")
    print("="*70)
    load_kaggle_data()

    # -----------------------------------------------------------------------
    # 2. Merge raw folders
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("PHASE 2.0: MERGING RAW IMAGE FOLDERS")
    print("="*70)
    merge_raw_folders(
        raw_dir_1="data/raw",
        raw_dir_2="data/raw2",
        raw_merged_dir="data/raw_merged"
    )

    # -----------------------------------------------------------------------
    # 3. Face alignment / filtering
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("PHASE 2.1: FACE ALIGNMENT / FILTERING")
    print("="*70)

    if config.USE_FACE_ALIGNMENT:
        # Fresh aligned directory
        if os.path.exists(config.ALIGNED_DIR):
            print(f"Removing existing aligned directory: {config.ALIGNED_DIR}")
            shutil.rmtree(config.ALIGNED_DIR)
        os.makedirs(config.ALIGNED_DIR, exist_ok=True)

        print("\nAligning faces with MTCNN...")
        collection_summary = align_faces_mtcnn(
            raw_dir=config.RAW_DIR,
            aligned_dir=config.ALIGNED_DIR,
            image_size=config.IMAGE_SIZE,
            margin_ratio=config.ALIGN_MARGIN,
            min_images_per_class=config.MIN_IMAGES_FOR_OFFLINE_AUG,
        )
        if config.REMOVE_DUPLICATES:
            print("\nRemoving near-duplicates with pHash...")
            dedup_report = deduplicate_phash(
                aligned_dir=config.ALIGNED_DIR,
                max_distance=config.DEDUP_PHASH_DISTANCE
            )
    else:
        print("\nFiltering images with Haar cascade...")
        collection_summary = filter_images_with_faces(
            data_dir=config.RAW_DIR,
            min_face_ratio=config.MIN_FACE_RATIO,
            min_images_per_class=config.MIN_IMAGES_FOR_OFFLINE_AUG
        )

    # Update SOURCE_DIR based on alignment choice
    config.SOURCE_DIR = config.ALIGNED_DIR if config.USE_FACE_ALIGNMENT else config.RAW_DIR

    # -----------------------------------------------------------------------
    # 4. Train/val/test split
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("PHASE 2.2: DATASET SPLITTING")
    print("="*70)
    split_summary = split_dataset(
        raw_dir=config.SOURCE_DIR,
        min_images=config.MIN_IMAGES_FOR_SPLIT
    )

    # -----------------------------------------------------------------------
    # 5. Offline augmentation
    # -----------------------------------------------------------------------
    if config.USE_OFFLINE_AUGMENTATION:
        print("\n" + "="*70)
        print("PHASE 2.3: OFFLINE AUGMENTATION")
        print("="*70)
        run_offline_augmentation(
            data_dir=config.DATA_DIR,
            min_images=config.MIN_IMAGES_FOR_OFFLINE_AUG,
            num_augmentations=config.NUM_AUGMENTATIONS
        )

    # -----------------------------------------------------------------------
    # 6. Train all models
    # -----------------------------------------------------------------------
    results = []
    trained_models = {}
    model_names = (
        config.MODELS_TO_TRAIN
        if hasattr(config, "MODELS_TO_TRAIN")
        else [config.MODEL_BACKBONE]
    )

    for model_name in model_names:
        try:
            # Build model-specific dataloaders so normalization matches backbone expectations.
            train_loader, val_loader, test_loader, class_weights = create_dataloaders(
                model_name=model_name
            )

            model, best_val_acc, history = train_model(
                model_name, train_loader, val_loader, class_weights
            )
            trained_models[model_name] = model

            # Evaluate
            eval_results = evaluate_model(model, model_name, test_loader)
            results.append(eval_results)

        except Exception as e:
            print(f"\nERROR training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # -----------------------------------------------------------------------
    # 7. Final comparison table
    # -----------------------------------------------------------------------
    if results:
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)

        df = pd.DataFrame(results)
        accuracy_col = next(
            (col for col in ('test_acc', 'test_accuracy', 'accuracy') if col in df.columns),
            None
        )
        if accuracy_col is None:
            raise KeyError('No accuracy column found in results DataFrame')
        df[accuracy_col] = df[accuracy_col].apply(lambda x: f"{x * 100:.2f}%")
        df['precision'] = df['precision'].apply(lambda x: f"{x:.4f}")
        df['recall'] = df['recall'].apply(lambda x: f"{x:.4f}")
        df['f1'] = df['f1'].apply(lambda x: f"{x:.4f}")

        print("\n" + df.to_string(index=False))
        os.makedirs(f'{config.OUTPUT_DIR}/results', exist_ok=True)
        df.to_csv(f'{config.OUTPUT_DIR}/results/model_comparison.csv', index=False)

        print(f"\nAll results saved to: {config.OUTPUT_DIR}")
        print(f"   Models: {config.OUTPUT_DIR}/models/")
        print(f"   Plots: {config.OUTPUT_DIR}/plots/")
        print(f"   Results: {config.OUTPUT_DIR}/results/")

        print("\n" + "="*70)
        print("TRAINING PIPELINE COMPLETE!")
        print("="*70)
        print("\nDOWNLOAD THESE FOLDERS:")
        print(f"   1. {config.OUTPUT_DIR}/models/  (trained model weights)")
        print(f"   2. {config.OUTPUT_DIR}/plots/   (training curves, confusion matrices)")
        print(f"   3. {config.OUTPUT_DIR}/results/ (evaluation reports)")
        print("\nTIP: In Kaggle, these are in /kaggle/working/")
    else:
        print("\nNo models were successfully trained!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nCheck the error above and try again")
