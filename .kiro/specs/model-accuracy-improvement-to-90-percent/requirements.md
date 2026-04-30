# Requirements Document: Model Accuracy Improvement to 90%

## Introduction

This document specifies the requirements for improving the Pakistani Politician Image Classification model from its current 76.03% test accuracy to the target 90% accuracy. The system currently uses CNN architectures (ResNet50, EfficientNet-B3) trained on 16 politician classes with approximately 60 images per class after face detection filtering. The improvement strategy focuses on data quality enhancement, advanced augmentation techniques, and model optimization.

## Glossary

- **Training_Pipeline**: The Jupyter notebook (COMPLETE_TRAINING_PIPELINE.ipynb) containing all data collection, preprocessing, training, and evaluation code
- **Face_Detector**: OpenCV Haar Cascade classifier used to filter images containing faces
- **Augmentation_Pipeline**: Albumentations-based image transformation system that generates synthetic training variations
- **Data_Collector**: Web scraping system that gathers politician images from Bing and Google
- **Model_Trainer**: Training system that fine-tunes pre-trained CNN models on politician dataset
- **Test_Accuracy**: Classification accuracy measured on held-out test set (10% of total data)
- **Class_Balance**: Distribution of images across the 16 politician classes
- **Face_Ratio**: Proportion of image area occupied by detected faces (currently 5% minimum threshold)
- **Augmentation_Multiplier**: Number of synthetic variations generated per original image (currently 2x)
- **Minimum_Images_Per_Class**: Target number of images required per politician class after filtering (currently 80)

## Requirements

### Requirement 1: Relaxed Face Detection Filtering

**User Story:** As a data scientist, I want more lenient face detection filtering, so that fewer valid politician images are discarded during preprocessing.

#### Acceptance Criteria

1. WHEN the Face_Detector processes raw images, THE Face_Detector SHALL use a minimum face ratio threshold of 2% instead of 5%
2. WHEN the Face_Detector applies Haar Cascade detection, THE Face_Detector SHALL use scaleFactor of 1.03 and minNeighbors of 2 for increased sensitivity
3. WHEN the Face_Detector evaluates face size, THE Face_Detector SHALL accept faces as small as 15×15 pixels
4. WHEN face detection completes for all classes, THE Training_Pipeline SHALL display a summary showing kept vs removed image counts per class
5. FOR ALL classes with fewer than Minimum_Images_Per_Class images after filtering, THE Training_Pipeline SHALL display a warning message listing those classes

### Requirement 2: Enhanced Data Augmentation Pipeline

**User Story:** As a machine learning engineer, I want more sophisticated augmentation techniques, so that the model learns robust features from limited training data.

#### Acceptance Criteria

1. THE Augmentation_Pipeline SHALL include geometric transformations: RandomRotate90, Rotate (±30°), HorizontalFlip, ShiftScaleRotate, and Perspective
2. THE Augmentation_Pipeline SHALL include color transformations: RandomBrightnessContrast, HueSaturationValue, and RandomGamma
3. THE Augmentation_Pipeline SHALL include noise and blur: GaussianBlur, MotionBlur, and GaussNoise
4. THE Augmentation_Pipeline SHALL include CoarseDropout with max 8 holes of 20×20 pixels to simulate partial occlusion
5. THE Augmentation_Pipeline SHALL include RandomScale with scale limit of 0.2
6. WHEN augmentation is applied to training images, THE Augmentation_Pipeline SHALL apply each transformation with its specified probability independently
7. FOR ALL augmentation transformations, THE Augmentation_Pipeline SHALL preserve the face region visibility and image dimensions

### Requirement 3: Increased Augmentation Multiplier

**User Story:** As a data scientist, I want to generate more synthetic training variations, so that the model has sufficient diverse examples to learn from.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL set the Augmentation_Multiplier to 5 instead of 2
2. WHEN augmentation is performed on the training set, THE Training_Pipeline SHALL generate 5 augmented versions for each original training image
3. WHEN augmentation completes, THE Training_Pipeline SHALL display the total training set size after augmentation
4. THE Training_Pipeline SHALL ensure that validation and test sets remain unaugmented

### Requirement 4: Smart Data Collection with Minimum Thresholds

**User Story:** As a data engineer, I want automated collection to continue until minimum image counts are met, so that all classes have sufficient training data.

#### Acceptance Criteria

1. WHEN the Data_Collector starts collection for a politician class, THE Data_Collector SHALL continue searching until at least Minimum_Images_Per_Class images are collected or all backup queries are exhausted
2. WHEN a primary search query yields insufficient results, THE Data_Collector SHALL automatically try alternate name spellings and backup queries
3. WHEN the Data_Collector completes collection for all classes, THE Data_Collector SHALL display a summary showing final image counts per class
4. IF any class has fewer than Minimum_Images_Per_Class images after all queries are exhausted, THEN THE Data_Collector SHALL display a warning listing those classes
5. THE Data_Collector SHALL attempt collection from both Bing and Google sources for each query

### Requirement 5: Automatic Results Cleanup

**User Story:** As a developer, I want old training results automatically removed before new runs, so that disk space is managed and results are not confused between runs.

#### Acceptance Criteria

1. WHEN the Training_Pipeline starts a new training run, THE Training_Pipeline SHALL check for existing results directories
2. IF existing results directories are found, THEN THE Training_Pipeline SHALL remove all files in those directories before proceeding
3. THE Training_Pipeline SHALL clean up the following directories: plots/, results/, models/saved/, and notebooks/project_outputs/
4. WHEN cleanup completes, THE Training_Pipeline SHALL display a confirmation message indicating which directories were cleaned
5. IF cleanup fails for any directory, THEN THE Training_Pipeline SHALL display an error message but continue execution

### Requirement 6: Extended Training Duration

**User Story:** As a machine learning engineer, I want longer training with more patience, so that models have sufficient opportunity to converge to optimal accuracy.

#### Acceptance Criteria

1. THE Model_Trainer SHALL train for a maximum of 30 epochs instead of 20 epochs
2. THE Model_Trainer SHALL use an early stopping patience of 7 epochs instead of 5 epochs
3. WHEN validation loss does not improve for 7 consecutive epochs, THE Model_Trainer SHALL stop training and restore the best model weights
4. WHEN training completes, THE Model_Trainer SHALL display the epoch at which the best validation accuracy was achieved

### Requirement 7: Advanced Model Architecture Support

**User Story:** As a researcher, I want to experiment with more powerful model architectures, so that I can identify the best performing model for this task.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL support training EfficientNet-B4 in addition to existing models
2. WHEN a model is selected for training, THE Training_Pipeline SHALL load the appropriate pre-trained weights from ImageNet
3. WHEN EfficientNet-B4 is trained, THE Model_Trainer SHALL use the same two-phase training strategy as other models (frozen backbone then full fine-tuning)
4. THE Training_Pipeline SHALL allow configuration of which models to train via a models_to_train list

### Requirement 8: Ensemble Prediction Capability

**User Story:** As a machine learning engineer, I want to combine predictions from multiple models, so that I can achieve higher accuracy through ensemble methods.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL provide an ensemble_predict function that accepts multiple trained models
2. WHEN ensemble_predict is called with multiple models and an input image, THE Training_Pipeline SHALL compute predictions from each model independently
3. WHEN all model predictions are obtained, THE Training_Pipeline SHALL average the prediction probabilities across all models
4. THE Training_Pipeline SHALL return the class with the highest averaged probability as the ensemble prediction
5. WHEN ensemble evaluation is performed on the test set, THE Training_Pipeline SHALL display the ensemble test accuracy alongside individual model accuracies

### Requirement 9: Focal Loss for Class Imbalance

**User Story:** As a data scientist, I want a loss function that handles class imbalance, so that the model learns equally well from classes with fewer examples.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL implement a FocalLoss class with configurable alpha and gamma parameters
2. THE FocalLoss SHALL compute cross-entropy loss and apply the focal term (1-pt)^gamma to down-weight easy examples
3. THE Training_Pipeline SHALL provide an option to use FocalLoss instead of standard CrossEntropyLoss
4. WHEN FocalLoss is used, THE Model_Trainer SHALL use alpha=1 and gamma=2 as default parameters
5. THE Training_Pipeline SHALL allow switching between FocalLoss and CrossEntropyLoss via configuration

### Requirement 10: Test-Time Augmentation

**User Story:** As a machine learning engineer, I want to apply augmentation during inference, so that predictions are more robust to image variations.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL provide a predict_with_tta function that accepts a model, image, and number of augmentations
2. WHEN predict_with_tta is called, THE Training_Pipeline SHALL generate the specified number of augmented versions of the input image
3. WHEN augmented versions are generated, THE Training_Pipeline SHALL compute predictions for each augmented version
4. THE Training_Pipeline SHALL average the predictions across all augmented versions and return the final prediction
5. WHEN test-time augmentation is used during evaluation, THE Training_Pipeline SHALL display both standard accuracy and TTA accuracy

### Requirement 11: Comprehensive Results Display

**User Story:** As a researcher, I want all training results displayed inline in the notebook, so that I can review model performance without accessing external files.

#### Acceptance Criteria

1. WHEN training completes for all models, THE Training_Pipeline SHALL display a comparison table showing test accuracy, precision, recall, and F1-score for each model
2. WHEN evaluation completes, THE Training_Pipeline SHALL display confusion matrices as inline plots for each model
3. WHEN evaluation completes, THE Training_Pipeline SHALL display training curves (loss and accuracy) as inline plots for each model
4. THE Training_Pipeline SHALL display per-class classification reports showing precision, recall, and F1-score for each politician class
5. WHEN the 90% accuracy target is achieved, THE Training_Pipeline SHALL display a success message highlighting the achievement

### Requirement 12: Data Leakage Prevention

**User Story:** As a data scientist, I want strict separation between train/val/test sets, so that model performance metrics are trustworthy and not inflated by data leakage.

#### Acceptance Criteria

1. WHEN the Training_Pipeline splits the dataset, THE Training_Pipeline SHALL perform the split before any augmentation is applied
2. THE Training_Pipeline SHALL ensure that no image from the validation or test sets appears in the training set
3. THE Training_Pipeline SHALL apply augmentation only to the training set and not to validation or test sets
4. WHEN dataset splitting completes, THE Training_Pipeline SHALL display the number of unique images in each split to confirm no overlap
5. THE Training_Pipeline SHALL use a fixed random seed for dataset splitting to ensure reproducibility

