"""
data_prep.py – Data preparation utilities.

Functions:
  - load_kaggle_data        : copy Kaggle input datasets into local data/ directories
  - merge_raw_folders       : merge data/raw + data/raw2 into data/raw_merged with sequential naming
  - filter_images_with_faces: legacy Haar-cascade face filter (USE_FACE_ALIGNMENT=False path)
  - rotate_point            : geometric helper used by align_face_with_landmarks
  - align_face_with_landmarks: crop + rotate a face using MTCNN eye landmarks
  - align_faces_mtcnn       : MTCNN-based alignment pipeline over a full directory tree
  - deduplicate_phash       : per-class pHash deduplication
  - split_dataset           : stratified train/val/test split
  - run_offline_augmentation: albumentations-based offline augmentation for small classes
"""

import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import imagehash
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from training.config import config


# ---------------------------------------------------------------------------
# Kaggle data loading
# ---------------------------------------------------------------------------

def load_kaggle_data(kaggle_input="/kaggle/input",
                     local_raw="data/raw",
                     local_raw2="data/raw2"):
    """Copy raw images from Kaggle input datasets into local directories.

    When running locally (no /kaggle mount), this is a no-op.
    """
    os.makedirs(local_raw, exist_ok=True)
    os.makedirs(local_raw2, exist_ok=True)

    if not os.path.isdir(kaggle_input):
        print("\nRunning locally (not on Kaggle)")
        print(f"   Using local dataset from: {local_raw}")
        if os.path.isdir(local_raw):
            classes = [d for d in os.listdir(local_raw) if os.path.isdir(os.path.join(local_raw, d))]
            if classes:
                print(f"   Found {len(classes)} classes in {local_raw}")
        return

    print("\nRunning on Kaggle – searching for datasets...")
    source_raw = None
    source_raw2 = None

    for ds in os.listdir(kaggle_input):
        ds_path = os.path.join(kaggle_input, ds)
        if not os.path.isdir(ds_path):
            continue
        for root, dirs, files in os.walk(ds_path):
            if 'raw' in dirs:
                source_raw = os.path.join(root, 'raw')
            if 'raw 2' in dirs:
                source_raw2 = os.path.join(root, 'raw 2')
            if 'raw2' in dirs:
                source_raw2 = source_raw2 or os.path.join(root, 'raw2')
        if source_raw and source_raw2:
            break

    def _copy_dir(src_dir, dst_dir):
        for cls in os.listdir(src_dir):
            src_cls = os.path.join(src_dir, cls)
            dst_cls = os.path.join(dst_dir, cls)
            if os.path.isdir(src_cls):
                os.makedirs(dst_cls, exist_ok=True)
                for f in os.listdir(src_cls):
                    src_file = os.path.join(src_cls, f)
                    dst_file = os.path.join(dst_cls, f)
                    if not os.path.exists(dst_file):
                        shutil.copy2(src_file, dst_file)
                print(f"     {cls}: {len(os.listdir(dst_cls))} images")

    if source_raw and os.path.isdir(source_raw):
        print(f"   Found raw images: {source_raw}")
        _copy_dir(source_raw, local_raw)
    else:
        print("   No 'raw' folder found inside Kaggle input – will check local data/raw")

    if source_raw2 and os.path.isdir(source_raw2):
        print(f"   Found raw 2 images: {source_raw2}")
        _copy_dir(source_raw2, local_raw2)
    else:
        print("   No 'raw 2' folder found inside Kaggle input – will check local data/raw2")

    print("\nData loaded into local directories.")


# ---------------------------------------------------------------------------
# Merge raw folders
# ---------------------------------------------------------------------------

def copy_with_rename(src, dst_folder, counter):
    """Copy a file to destination folder with sequential numeric naming.

    Returns the next available counter.
    """
    src_path = Path(src)
    ext = src_path.suffix.lower()

    while True:
        new_name = f"{counter:04d}{ext}"
        dst_path = Path(dst_folder) / new_name

        if not dst_path.exists():
            try:
                shutil.copy2(src, dst_path)
                return counter + 1
            except Exception as e:
                print(f"    Failed to copy {src_path.name} to {new_name}: {e}")
                return counter + 1
        else:
            counter += 1


def merge_raw_folders(raw_dir_1="data/raw",
                      raw_dir_2="data/raw2",
                      raw_merged_dir="data/raw_merged"):
    """Merge data/raw and data/raw2 into data/raw_merged with sequential numeric naming."""
    os.makedirs(raw_merged_dir, exist_ok=True)
    merge_summary = {}

    for class_name in config.CLASS_NAMES:
        merged_class_dir = Path(raw_merged_dir) / class_name
        merged_class_dir.mkdir(parents=True, exist_ok=True)

        # Find highest existing numeric counter to avoid collisions
        existing_files = os.listdir(merged_class_dir) if merged_class_dir.exists() else []
        # NOTE: f[:-4] assumes a 3-letter extension; .jpeg files might be skipped in max-counter
        numeric_files = [f for f in existing_files if f[0].isdigit() and f[:-4].isdigit()]

        if numeric_files:
            max_counter = max(int(Path(f).stem) for f in numeric_files)
            counter = max_counter + 1
        else:
            counter = 1

        images_from_raw = 0
        images_from_raw2 = 0

        # Copy from RAW_DIR_1 (data/raw)
        source_1 = Path(raw_dir_1) / class_name
        if source_1.is_dir():
            for img_file in sorted(os.listdir(source_1)):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_path = source_1 / img_file
                    if src_path.is_file():
                        counter = copy_with_rename(src_path, merged_class_dir, counter)
                        images_from_raw += 1

        # Copy from RAW_DIR_2 (data/raw2)
        source_2 = Path(raw_dir_2) / class_name
        if source_2.is_dir():
            for img_file in sorted(os.listdir(source_2)):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_path = source_2 / img_file
                    if src_path.is_file():
                        counter = copy_with_rename(src_path, merged_class_dir, counter)
                        images_from_raw2 += 1

        total_for_class = images_from_raw + images_from_raw2
        merge_summary[class_name] = {
            'from_raw': images_from_raw,
            'from_raw2': images_from_raw2,
            'total': total_for_class
        }

    print("\n" + "="*70)
    print("MERGE SUMMARY")
    print("="*70)
    print(f"{'Class':<30} {'data/raw':>12} {'data/raw2':>12} {'Total':>10}")
    print("-"*70)
    total_all = 0
    for name in sorted(merge_summary.keys()):
        stats = merge_summary[name]
        print(f"{name:<30} {stats['from_raw']:>12} {stats['from_raw2']:>12} {stats['total']:>10}")
        total_all += stats['total']
    print("-"*70)
    print(f"{'TOTAL':<30} "
          f"{sum(s['from_raw'] for s in merge_summary.values()):>12} "
          f"{sum(s['from_raw2'] for s in merge_summary.values()):>12} "
          f"{total_all:>10}")
    print(f"\nRaw folders merged into: {raw_merged_dir}")
    print(f"  Naming convention: Sequential numeric (0001.jpg, 0002.jpg, ...)")
    print(f"  Total images preserved: {total_all}")
    return merge_summary


# ---------------------------------------------------------------------------
# Legacy Haar-cascade face filter
# ---------------------------------------------------------------------------

def filter_images_with_faces(data_dir="data/raw_merged",
                             min_face_ratio=0.02,
                             min_images_per_class=80):
    """Filter images to keep only those with detectable faces (Haar cascade)."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    summary = {}
    warnings = []

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        print(f"\nFiltering: {class_name}")

        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        kept = 0
        removed = 0

        for img_file in tqdm(images, desc="  Processing"):
            img_path = os.path.join(class_path, img_file)

            try:
                img = cv2.imread(img_path)
                if img is None:
                    os.remove(img_path)
                    removed += 1
                    continue

                # Resize large images for faster processing
                h, w = img.shape[:2]
                if max(h, w) > 1000:
                    scale = 1000 / max(h, w)
                    img = cv2.resize(img, None, fx=scale, fy=scale)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.03,
                    minNeighbors=2,
                    minSize=(15, 15)
                )

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

            except Exception:
                try:
                    os.remove(img_path)
                    removed += 1
                except Exception:
                    pass

        summary[class_name] = kept
        status = "OK" if kept >= min_images_per_class else "LOW"
        print(f"  {status} Kept: {kept} | Removed: {removed}")

        if kept < min_images_per_class:
            warnings.append(f"{class_name}: Only {kept} images (need {min_images_per_class})")

    if warnings:
        print("\n" + "="*70)
        print("WARNING: Some classes have insufficient images:")
        print("="*70)
        for warning in warnings:
            print(f"   {warning}")
        print("\nRecommendation: Collect more images for these classes")

    return summary


# ---------------------------------------------------------------------------
# MTCNN face alignment helpers
# ---------------------------------------------------------------------------

def rotate_point(x, y, cx, cy, angle_rad):
    """Rotate a point (x, y) around center (cx, cy) by angle_rad."""
    x -= cx
    y -= cy
    xr = x * math.cos(angle_rad) - y * math.sin(angle_rad)
    yr = x * math.sin(angle_rad) + y * math.cos(angle_rad)
    return xr + cx, yr + cy


def align_face_with_landmarks(img, box, landmarks, image_size=336, margin_ratio=0.2):
    """Align a face using eye landmarks and crop to a square with margin."""
    left_eye = landmarks[0]
    right_eye = landmarks[1]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))
    eye_center = (
        (left_eye[0] + right_eye[0]) / 2.0,
        (left_eye[1] + right_eye[1]) / 2.0
    )

    rotated = img.rotate(-angle, resample=Image.BICUBIC, center=eye_center, expand=False)

    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    angle_rad = -math.radians(angle)
    rotated_corners = [
        rotate_point(x, y, eye_center[0], eye_center[1], angle_rad)
        for x, y in corners
    ]

    xs = [p[0] for p in rotated_corners]
    ys = [p[1] for p in rotated_corners]
    x1r, x2r = min(xs), max(xs)
    y1r, y2r = min(ys), max(ys)

    w = x2r - x1r
    h = y2r - y1r
    if w <= 0 or h <= 0:
        return None

    side = max(w, h)
    pad = side * margin_ratio
    side = side + 2 * pad

    cx = (x1r + x2r) / 2.0
    cy = (y1r + y2r) / 2.0

    x1c = cx - side / 2.0
    y1c = cy - side / 2.0
    x2c = cx + side / 2.0
    y2c = cy + side / 2.0

    x1c = max(0, x1c)
    y1c = max(0, y1c)
    x2c = min(rotated.width, x2c)
    y2c = min(rotated.height, y2c)

    if x2c <= x1c or y2c <= y1c:
        return None

    face = rotated.crop((x1c, y1c, x2c, y2c)).resize((image_size, image_size), Image.BICUBIC)
    return face


def align_faces_mtcnn(raw_dir="data/raw",
                      aligned_dir="data/aligned",
                      image_size=336,
                      margin_ratio=0.2,
                      min_images_per_class=80):
    """Align and filter faces with MTCNN, saving to aligned_dir."""
    from facenet_pytorch import MTCNN as _MTCNN

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = _MTCNN(keep_all=True, device=device)
    summary = {}
    warnings = []

    for class_name in os.listdir(raw_dir):
        class_path = Path(raw_dir) / class_name
        if not class_path.is_dir():
            continue

        print(f"\nAligning: {class_name}")
        out_dir = Path(aligned_dir) / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        kept = 0
        removed = 0

        for img_file in tqdm(images, desc="  Processing"):
            img_path = class_path / img_file
            try:
                img = Image.open(img_path).convert('RGB')
                # Convert PIL image to uint8 NumPy array (works with both NumPy 1.x and 2.x)
                img_np = np.array(img.convert('RGB'), dtype=np.uint8)

                with torch.no_grad():
                    boxes, probs, landmarks = mtcnn.detect(img_np, landmarks=True)

                if boxes is None or landmarks is None:
                    removed += 1
                    continue

                # Keep only single-face images; skip group photos and no-face cases.
                if len(boxes) != 1:
                    removed += 1
                    continue

                idx = 0
                aligned = align_face_with_landmarks(
                    img, boxes[idx], landmarks[idx],
                    image_size=image_size, margin_ratio=margin_ratio
                )
                if aligned is None:
                    removed += 1
                    continue

                aligned.save(out_dir / img_file)
                kept += 1

            except Exception as e:
                print(f"    Failed to process {img_file}: {e}")
                removed += 1

        # Clear GPU cache after each class
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        summary[class_name] = kept
        status = "OK" if kept >= min_images_per_class else "LOW"
        print(f"  {status} Kept: {kept} | Removed: {removed}")

        if kept < min_images_per_class:
            warnings.append(f"{class_name}: Only {kept} images (need {min_images_per_class})")

    # Clean up MTCNN model
    del mtcnn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if warnings:
        print("\n" + "="*70)
        print("WARNING: Some classes have insufficient images:")
        print("="*70)
        for warning in warnings:
            print(f"   {warning}")
        print("\nRecommendation: Collect more images for these classes")

    return summary


# ---------------------------------------------------------------------------
# pHash deduplication
# ---------------------------------------------------------------------------

def deduplicate_phash(aligned_dir="data/aligned", max_distance=5):
    """Remove near-duplicates using perceptual hash per class (O(n) with hash bucketing)."""
    report = {}

    for class_name in os.listdir(aligned_dir):
        class_path = Path(aligned_dir) / class_name
        if not class_path.is_dir():
            continue

        print(f"\nDeduplicating: {class_name}")
        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        hashes = []
        kept = 0
        removed = 0

        for img_file in tqdm(images, desc="  Checking duplicates"):
            img_path = class_path / img_file
            try:
                img = Image.open(img_path).convert('RGB')
                h = imagehash.phash(img)
            except Exception as e:
                print(f"    Failed to hash {img_file}: {e}")
                continue

            # Check if duplicate (O(n) per class, acceptable for ~100 images)
            is_dup = any((h - existing) <= max_distance for existing in hashes)
            if is_dup:
                try:
                    os.remove(img_path)
                    removed += 1
                except Exception as e:
                    print(f"    Failed to remove {img_file}: {e}")
            else:
                hashes.append(h)
                kept += 1

        report[class_name] = {"kept": kept, "removed": removed}
        print(f"  Kept: {kept} | Removed: {removed}")

    return report


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def split_dataset(raw_dir=None, output_dir="dataset",
                  train_ratio=0.75, val_ratio=0.15, test_ratio=0.10, min_images=5):
    """Split dataset into train/val/test with stratification and per-class minimums."""
    if raw_dir is None:
        raw_dir = config.SOURCE_DIR

    for split in ['train', 'val', 'test']:
        os.makedirs(f"{output_dir}/{split}", exist_ok=True)

    split_summary = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    skipped_classes = []

    class_to_idx = {name: idx for idx, name in enumerate(config.CLASS_NAMES)}
    paths = []
    labels = []

    for class_name in config.CLASS_NAMES:
        class_path = Path(raw_dir) / class_name
        if not class_path.is_dir():
            continue

        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(images) == 0:
            skipped_classes.append(f"{class_name}: 0 images")
            continue

        for img in images:
            paths.append(str(class_path / img))
            labels.append(class_to_idx[class_name])

    if len(paths) == 0:
        return split_summary

    counts_total = Counter(labels)
    small_classes = {cls for cls, cnt in counts_total.items() if cnt < min_images}

    # Stratified split for classes with enough samples.
    paths_large = [p for p, l in zip(paths, labels) if l not in small_classes]
    labels_large = [l for l in labels if l not in small_classes]

    X_train, X_val, X_test, y_train, y_val, y_test = [], [], [], [], [], []

    if len(paths_large) > 0 and len(set(labels_large)) >= 2:
        X_train, X_temp, y_train, y_temp = train_test_split(
            paths_large, labels_large,
            test_size=(val_ratio + test_ratio),
            stratify=labels_large,
            random_state=42
        )

        if len(X_temp) >= 2 and len(set(y_temp)) >= 2:
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=test_ratio / (val_ratio + test_ratio),
                stratify=y_temp,
                random_state=42
            )
        else:
            X_val, y_val = X_temp, y_temp
            X_test, y_test = [], []
    else:
        X_train, y_train = paths_large, labels_large

    # Add small-class samples to train only.
    for p, l in zip(paths, labels):
        if l in small_classes:
            X_train.append(p)
            y_train.append(l)

    # Enforce at least one sample in val/test for classes with enough images.
    def move_one(src_paths, src_labels, dst_paths, dst_labels, class_id):
        for i, lab in enumerate(src_labels):
            if lab == class_id:
                dst_paths.append(src_paths.pop(i))
                dst_labels.append(src_labels.pop(i))
                return True
        return False

    warnings = []
    counts_val = Counter(y_val)
    counts_test = Counter(y_test)
    for class_id, total in counts_total.items():
        if total >= min_images:
            if counts_val.get(class_id, 0) == 0:
                if not move_one(X_train, y_train, X_val, y_val, class_id):
                    warnings.append(f"{config.CLASS_NAMES[class_id]} missing from val")
            if counts_test.get(class_id, 0) == 0:
                if not move_one(X_train, y_train, X_test, y_test, class_id):
                    warnings.append(f"{config.CLASS_NAMES[class_id]} missing from test")

    # Copy files and build summary.
    def copy_split(files, labels_list, split_name):
        for src, label in zip(files, labels_list):
            class_name = config.CLASS_NAMES[label]
            dst_dir = Path(output_dir) / split_name / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / Path(src).name)
            split_summary[class_name][split_name] += 1

    copy_split(X_train, y_train, 'train')
    copy_split(X_val, y_val, 'val')
    copy_split(X_test, y_test, 'test')

    if skipped_classes:
        print("\n" + "="*70)
        print("WARNING: Some classes were skipped or have issues:")
        print("="*70)
        for msg in skipped_classes:
            print(f"   {msg}")

    if warnings:
        print("\n" + "="*70)
        print("WARNING: Some classes could not be placed in val/test:")
        print("="*70)
        for msg in warnings:
            print(f"   {msg}")

    return split_summary


# ---------------------------------------------------------------------------
# Offline augmentation
# ---------------------------------------------------------------------------

def run_offline_augmentation(data_dir="dataset",
                             min_images=None,
                             num_augmentations=None):
    """Albumentations-based offline augmentation for under-represented classes in train set."""
    import albumentations as A

    if min_images is None:
        min_images = config.MIN_IMAGES_FOR_OFFLINE_AUG
    if num_augmentations is None:
        num_augmentations = config.NUM_AUGMENTATIONS

    train_root = Path(data_dir) / "train"

    if not train_root.exists():
        print(f"WARNING: {train_root} does not exist; skipping offline augmentation.")
        return {}

    aug_pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=25, p=0.7),
        A.RandomBrightnessContrast(p=0.5),
    ])

    added_per_class = {}

    for class_dir in sorted([d for d in train_root.iterdir() if d.is_dir()]):
        image_files = [
            p for p in class_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            and not p.stem.startswith("aug_")
        ]

        if len(image_files) >= min_images:
            added_per_class[class_dir.name] = 0
            continue

        counter = 1
        while (class_dir / f"aug_{counter:05d}.jpg").exists():
            counter += 1

        added = 0
        for img_path in image_files:
            try:
                image = Image.open(img_path).convert("RGB")
                image_np = np.array(image)
            except Exception as exc:
                print(f"  WARNING: Failed to read {img_path.name}: {exc}")
                continue

            for _ in range(num_augmentations):
                augmented = aug_pipeline(image=image_np)
                aug_image = augmented["image"]
                out_path = class_dir / f"aug_{counter:05d}.jpg"
                Image.fromarray(aug_image).save(out_path)
                counter += 1
                added += 1

        added_per_class[class_dir.name] = added

    print("\n" + "="*70)
    print("OFFLINE AUGMENTATION SUMMARY")
    print("="*70)
    print(f"{'Class':<30} {'Added':>10}")
    print("-"*42)
    for name in sorted(added_per_class.keys()):
        print(f"{name:<30} {added_per_class[name]:>10}")

    return added_per_class
