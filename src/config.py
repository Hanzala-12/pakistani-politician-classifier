"""
src/config.py – Legacy configuration for the MLflow / DVC local pipeline (src/ package).

CANONICAL CONFIG → training/config.py
======================================
The `training/` package is the authoritative source for the Kaggle production pipeline
(ArcFace, MTCNN, full dedup/augment flow).  This file (src/config.py) is kept for
backward compatibility with `src/train.py`, `src/evaluate.py`, and the DVC pipeline.

When changing a hyperparameter, update BOTH files to keep them in sync:
  • training/config.py  ← production / Kaggle
  • src/config.py       ← legacy MLflow / DVC (this file)
"""
import os


class Config:
    """Training configuration"""
    # Paths
    RAW_DIR = "data/raw_merged"
    ALIGNED_DIR = "data/aligned"
    DATA_DIR = "dataset"

    # Environment-aware output directory
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or os.path.exists('/kaggle'):
        OUTPUT_DIR = "/kaggle/working"
    elif 'COLAB_GPU' in os.environ:
        OUTPUT_DIR = "/content/output"
    else:
        OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'project_outputs')

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
    EARLY_STOPPING_PATIENCE = 10

    # Models to train
    MODELS_TO_TRAIN = ["inception_resnet_v1", "inception_resnet_v1_casia", "resnet50"]
    MODEL_BACKBONE = "inception_resnet_v1"

    # Image settings
    IMAGE_SIZE = 336
    IMG_SIZE = IMAGE_SIZE  # backward compat

    # Augmentation
    USE_OFFLINE_AUGMENTATION = True
    NUM_AUGMENTATIONS = 3
    MIN_IMAGES_FOR_OFFLINE_AUG = 120

    # Face alignment + dedup
    USE_FACE_ALIGNMENT = True
    ALIGN_MARGIN = 0.2
    REMOVE_DUPLICATES = True
    DEDUP_PHASH_DISTANCE = 5
    MIN_IMAGES_FOR_SPLIT = 5

    # Legacy Haar parameters (USE_FACE_ALIGNMENT=False path)
    MIN_FACE_RATIO = 0.02
    FACE_SCALE_FACTOR = 1.03
    FACE_MIN_NEIGHBORS = 2
    FACE_MIN_SIZE = (15, 15)

    # Loss / class balancing
    USE_FOCAL_LOSS = False
    FOCAL_ALPHA = 1
    FOCAL_GAMMA = 2
    USE_CLASS_WEIGHTS = True

    # MixUp
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2
    MIXUP_PROB = 0.5

    # Evaluation
    USE_TTA = True
    TTA_NUM_AUGMENTATIONS = 5
    SHOW_MISCLASSIFIED = True
    USE_ENSEMBLE = True

    # ArcFace
    USE_ARCFACE = True
    ARCFACE_MARGIN = 0.3
    ARCFACE_SCALE = 64.0
    HEAD_LR = 1e-4
    BACKBONE_UNFREEZE_LR = 3e-6
    GRADIENT_CLIP_MAX_NORM = 1.0

    # Disabled loss variants
    USE_ADACOS = False
    USE_CURRICULARFACE = False

    # Image validation
    MIN_IMAGE_SIZE = 60

    # Checkpoint recovery
    RESUME_FROM_CHECKPOINT = False

    # Optional mislabeled audit
    FLAG_MISLABELED = False

    # DataLoader workers (0 = main process, safest for Kaggle/Windows)
    NUM_WORKERS = 0


# Singleton used everywhere — import this, not the class.
config = Config()
config.SOURCE_DIR = config.ALIGNED_DIR if config.USE_FACE_ALIGNMENT else config.RAW_DIR
