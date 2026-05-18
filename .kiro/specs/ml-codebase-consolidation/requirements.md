# Requirements Document

## Introduction

This document specifies the requirements for consolidating, fixing, and production-hardening the Pakistani Politician Face Recognition ML codebase. The system currently has duplicate training pipelines, duplicate inference services, inconsistent preprocessing, and hardcoded class labels. The consolidation must preserve all trained model checkpoints and working functionality while eliminating redundancy and ensuring consistency across training, evaluation, and inference.

## Glossary

- **Training_Pipeline**: The canonical training codebase located in `training/` that implements ArcFace metric learning, MixUp augmentation, and TTA evaluation
- **Legacy_Pipeline**: The older training codebase located in `src/` that uses MLflow and DVC orchestration
- **FastAPI_Service**: The inference API service located in `api/` that serves predictions via REST endpoints
- **Flask_Service**: The legacy inference service located in `backend/` that provides similar prediction functionality
- **Frontend**: The React-based web UI that displays politician classifications
- **Class_Label**: The canonical ordered list of 16 Pakistani politician names used for model training and inference
- **MTCNN**: Multi-task Cascaded Convolutional Networks face detector used for face alignment in ArcFace models
- **ArcFace**: Additive Angular Margin Loss metric learning approach used for face recognition
- **TTA**: Test-Time Augmentation using 5-crop + flip (10 views total) for improved evaluation accuracy
- **Model_Checkpoint**: Trained model weights stored as `.pth`, `.pt`, or `.onnx` files
- **Preprocessing_Pipeline**: The sequence of image transformations applied before model inference (MTCNN alignment, resizing, normalization)
- **DVC**: Data Version Control tool used for dataset versioning
- **Airflow**: Workflow orchestration tool used to automate the training pipeline
- **Embedding**: 512-dimensional feature vector output from ArcFace models before classification
- **Logit_Reconstruction**: Process of converting ArcFace embeddings to class logits using stored weights and scale parameters

## Requirements

### Requirement 1: Repository Audit and Documentation

**User Story:** As a developer, I want a complete audit of the codebase structure, so that I can understand all redundancies and inconsistencies before making changes.

#### Acceptance Criteria

1. THE Audit_System SHALL document all class label definitions and their locations (training/config.py, API endpoints, frontend hardcoded mappings)
2. THE Audit_System SHALL identify class name mismatches between `training/config.py CLASS_NAMES` and `frontend/src/App.tsx CLASS_MAPPINGS`
3. THE Audit_System SHALL identify all preprocessing differences between training transforms, evaluation TTA, and inference pipelines
4. THE Audit_System SHALL document all model loading logic including checkpoint discovery, ArcFace embedding-to-logit reconstruction, and classifier head implementations
5. THE Audit_System SHALL trace all DVC and Airflow dependencies to determine which stages reference `src/` versus `training/`
6. THE Audit_System SHALL identify all redundant or duplicate code across training pipelines and inference services
7. THE Audit_System SHALL identify all imports from `src/` modules in `api/main.py`, `dvc.yaml`, and `airflow/dags/`
8. THE Audit_System SHALL produce a concise audit report listing all mismatches before any code modifications

### Requirement 2: Canonical Class Label Definition

**User Story:** As a system maintainer, I want a single source of truth for class labels, so that training, evaluation, and inference always use consistent politician names and indices.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL define the canonical class list in `training/config.py` with fixed alphabetical ordering
2. THE Class_Label list SHALL contain exactly 16 politician names in sorted order matching folder names in the dataset
3. WHEN the class list is modified, THE Training_Pipeline SHALL maintain index-to-name consistency with existing model checkpoints
4. THE Training_Pipeline SHALL validate that class indices match model output dimensions during checkpoint loading
5. THE Consolidation_System SHALL verify that `training/config.py CLASS_NAMES` matches the actual folder names in `data/raw_merged/`
6. THE Consolidation_System SHALL identify and fix any mismatches between frontend `CLASS_MAPPINGS` and canonical `CLASS_NAMES`

### Requirement 3: Dynamic Class Label API

**User Story:** As a frontend developer, I want to fetch class labels dynamically from the API, so that I don't need to hardcode politician names in the UI.

#### Acceptance Criteria

1. THE FastAPI_Service SHALL expose a `/classes` endpoint that returns the canonical class list from `training.config`
2. WHEN the `/classes` endpoint is called, THE FastAPI_Service SHALL return class names in the same order as model output indices
3. THE `/classes` endpoint SHALL return a JSON response with structure `{"classes": [...], "count": 16}`
4. THE `/classes` endpoint SHALL return both machine-readable IDs (folder names) and human-readable display names
5. THE Frontend SHALL fetch class names from the `/classes` endpoint on initialization
6. THE Frontend SHALL remove the hardcoded `CLASS_MAPPINGS` dictionary from `frontend/src/App.tsx`
7. THE Frontend SHALL use the API-provided class names for all display purposes

### Requirement 4: Unified Inference Service

**User Story:** As a deployment engineer, I want a single canonical inference service, so that I can maintain one codebase instead of two duplicate services.

#### Acceptance Criteria

1. THE FastAPI_Service SHALL support all three model architectures (inception_resnet_v1, inception_resnet_v1_casia, resnet50)
2. THE FastAPI_Service SHALL implement MTCNN face alignment for ArcFace models with the same parameters as training
3. THE FastAPI_Service SHALL implement standard torchvision transforms for ResNet-50 models
4. THE FastAPI_Service SHALL provide an optional TTA parameter for `/predict` endpoint to enable 5-crop + flip augmentation
5. WHEN TTA is enabled, THE FastAPI_Service SHALL average predictions across all 10 augmented views
6. THE FastAPI_Service SHALL handle multi-face images by selecting the largest detected face
7. THE Flask_Service SHALL be marked as deprecated after FastAPI_Service verification
8. THE FastAPI_Service SHALL return predictions in a format compatible with the existing frontend expectations
9. THE FastAPI_Service SHALL support both `image` and `file` form field names for backward compatibility with Flask API

### Requirement 5: Preprocessing Consistency

**User Story:** As an ML engineer, I want identical preprocessing across training, evaluation, and inference, so that model performance is consistent and reproducible.

#### Acceptance Criteria

1. WHEN an ArcFace model is used, THE Preprocessing_Pipeline SHALL apply MTCNN face alignment with margin_ratio=0.2
2. WHEN an ArcFace model is used, THE Preprocessing_Pipeline SHALL resize aligned faces to 336×336 pixels
3. WHEN an ArcFace model is used, THE Preprocessing_Pipeline SHALL normalize with mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5]
4. WHEN a ResNet-50 model is used, THE Preprocessing_Pipeline SHALL resize images to 224×224 pixels
5. WHEN a ResNet-50 model is used, THE Preprocessing_Pipeline SHALL normalize with ImageNet mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
6. THE Preprocessing_Pipeline SHALL apply identical transforms in training (without augmentation), evaluation, and inference
7. WHEN TTA is enabled, THE Preprocessing_Pipeline SHALL apply 5-crop (4 corners + center) and horizontal flip to create 10 views

### Requirement 6: ArcFace Logit Reconstruction

**User Story:** As an ML engineer, I want consistent ArcFace embedding-to-logit conversion, so that evaluation and inference produce identical predictions for the same input.

#### Acceptance Criteria

1. WHEN an ArcFace model checkpoint is saved, THE Training_Pipeline SHALL store the ArcFace weight matrix and scale parameter in `arcface_eval` dictionary
2. WHEN an ArcFace model is loaded for evaluation, THE Evaluation_System SHALL reconstruct logits using stored ArcFace weights and scale
3. WHEN an ArcFace model is loaded for inference, THE FastAPI_Service SHALL reconstruct logits using stored ArcFace weights and scale
4. THE Logit_Reconstruction process SHALL normalize embeddings to unit length before computing cosine similarity
5. THE Logit_Reconstruction process SHALL normalize weight matrix to unit length before computing cosine similarity
6. THE Logit_Reconstruction process SHALL multiply cosine similarities by the stored scale parameter (default 64.0)

### Requirement 7: Model Checkpoint Preservation

**User Story:** As an ML engineer, I want all trained model checkpoints preserved, so that I can compare model performance and avoid retraining.

#### Acceptance Criteria

1. THE Consolidation_System SHALL NOT modify any `.pth`, `.pt`, or `.onnx` model checkpoint files
2. THE Consolidation_System SHALL NOT delete any model checkpoint files
3. THE Consolidation_System SHALL verify that all existing checkpoints can be loaded after code changes
4. WHEN checkpoint loading fails, THE Consolidation_System SHALL report the specific checkpoint path and error message

### Requirement 8: Legacy Code Deprecation

**User Story:** As a maintainer, I want clear identification of deprecated code, so that I can safely remove unused components after verification.

#### Acceptance Criteria

1. THE Consolidation_System SHALL trace all references to `src/` modules from `dvc.yaml`, `airflow/dags/`, and `api/main.py`
2. THE Consolidation_System SHALL identify that `api/main.py` imports `get_transforms` and `CLASS_NAMES` from `src/train.py` instead of `training/`
3. WHEN `src/` modules are still referenced, THE Consolidation_System SHALL migrate all imports to use `training/` equivalents
4. WHEN `src/` modules are no longer referenced, THE Consolidation_System SHALL mark the `src/` directory as deprecated with a README
5. THE Consolidation_System SHALL document which `src/` modules duplicate `training/` functionality
6. THE Consolidation_System SHALL provide a safe deletion checklist for deprecated components
7. THE Consolidation_System SHALL verify that no Docker, deployment, or CI/CD scripts reference `src/` modules

### Requirement 8.1: FastAPI Service Import Migration

**User Story:** As a developer, I want the FastAPI service to import from the canonical training pipeline, so that preprocessing and class labels remain consistent.

#### Acceptance Criteria

1. THE FastAPI_Service SHALL import `get_transforms` from `training.datasets` instead of `src.train`
2. THE FastAPI_Service SHALL import `CLASS_NAMES` from `training.config` instead of `src.train`
3. THE FastAPI_Service SHALL remove all imports from `src/` modules
4. WHEN imports are migrated, THE FastAPI_Service SHALL verify that `/predict` endpoint continues to function correctly
5. THE FastAPI_Service SHALL use the same preprocessing transforms as `training/datasets.py` for consistency

### Requirement 9: MTCNN Face Detection Robustness

**User Story:** As a user, I want reliable face detection, so that inference succeeds on a wide variety of input images.

#### Acceptance Criteria

1. WHEN MTCNN detects zero faces, THE FastAPI_Service SHALL return an error message "No face detected"
2. WHEN MTCNN detects multiple faces, THE FastAPI_Service SHALL select the largest face by bounding box area
3. WHEN MTCNN fails on the original image, THE FastAPI_Service SHALL retry detection on a 2× upscaled version (capped at 2000px)
4. WHEN upscaled detection succeeds, THE FastAPI_Service SHALL scale bounding boxes and landmarks back to original image coordinates
5. THE FastAPI_Service SHALL handle EXIF orientation metadata before face detection using `ImageOps.exif_transpose`
6. THE FastAPI_Service SHALL use MTCNN thresholds [0.5, 0.6, 0.7] and min_face_size=20 for improved small face detection
7. THE Backend model_loader SHALL implement the same MTCNN robustness features for consistency

### Requirement 10: Test-Time Augmentation Parity

**User Story:** As an ML engineer, I want TTA to work identically in evaluation and inference, so that reported test accuracy matches production performance.

#### Acceptance Criteria

1. WHEN TTA is enabled in evaluation, THE Evaluation_System SHALL apply 5-crop (4 corners + center) and horizontal flip
2. WHEN TTA is enabled in inference, THE FastAPI_Service SHALL apply 5-crop (4 corners + center) and horizontal flip
3. THE TTA implementation SHALL produce exactly 10 augmented views per input image
4. THE TTA implementation SHALL average softmax probabilities across all 10 views before selecting the predicted class
5. WHEN TTA is disabled, THE Evaluation_System and FastAPI_Service SHALL use only center crop

### Requirement 11: DVC Pipeline Migration

**User Story:** As a data scientist, I want DVC stages to use the canonical training pipeline, so that `dvc repro` runs the correct code.

#### Acceptance Criteria

1. THE DVC stage `collect_data` SHALL call `training/data_prep.py` functions instead of `src/collect_data.py`
2. THE DVC stage `split_data` SHALL call `training/data_prep.py::split_dataset` instead of `src/split_dataset.py`
3. THE DVC stage `augment` SHALL call `training/data_prep.py::run_offline_augmentation` instead of `src/augment.py`
4. THE DVC stage `train` SHALL call `training/main.py` instead of `src/train.py`
5. THE DVC stage `evaluate` SHALL call `training/evaluate.py` instead of `src/evaluate.py`
6. WHEN DVC stages are migrated, THE Consolidation_System SHALL update `dvc.yaml` to reference `training/` modules
7. WHEN DVC stages are migrated, THE Consolidation_System SHALL verify that `dvc repro` executes without errors
8. THE Consolidation_System SHALL verify that all DVC dependencies and outputs remain valid after migration

### Requirement 12: Airflow DAG Migration

**User Story:** As a data engineer, I want Airflow DAGs to use the canonical training pipeline, so that scheduled training runs use the correct code.

#### Acceptance Criteria

1. THE Airflow task `collect_data` SHALL import from `training.data_prep` instead of `src.collect_data`
2. THE Airflow task `split_dataset` SHALL import from `training.data_prep` instead of `src.split_dataset`
3. THE Airflow task `augment_data` SHALL import from `training.data_prep` instead of `src.augment`
4. THE Airflow task `train_models` SHALL import from `training.main` instead of `src.train`
5. THE Airflow task `evaluate_models` SHALL import from `training.evaluate` instead of `src.evaluate`
6. WHEN Airflow tasks are migrated, THE Consolidation_System SHALL verify that the DAG parses without errors

### Requirement 13: Inference Accuracy Verification

**User Story:** As an ML engineer, I want to verify that inference predictions match evaluation predictions, so that I can trust production results.

#### Acceptance Criteria

1. THE Verification_System SHALL select 10 random test images from the test set
2. FOR ALL selected test images, THE Verification_System SHALL run evaluation using `training/evaluate.py`
3. FOR ALL selected test images, THE Verification_System SHALL run inference using FastAPI_Service `/predict` endpoint
4. FOR ALL selected test images, THE Verification_System SHALL compare predicted class and confidence scores
5. WHEN predictions differ, THE Verification_System SHALL report the image path, evaluation result, and inference result
6. THE Verification_System SHALL verify that prediction differences are within 0.01 confidence tolerance

### Requirement 14: Model Loading Consistency

**User Story:** As a developer, I want consistent model loading across evaluation and inference, so that both systems use the same checkpoint format.

#### Acceptance Criteria

1. WHEN a checkpoint contains `model_state_dict`, THE Model_Loader SHALL load weights from `model_state_dict`
2. WHEN a checkpoint contains `arcface_eval`, THE Model_Loader SHALL store the ArcFace evaluation dictionary for logit reconstruction
3. WHEN a checkpoint contains `class_names`, THE Model_Loader SHALL validate that class count matches model output dimension
4. THE Model_Loader SHALL support loading checkpoints from `project_outputs/models/`, `models/saved/`, and `models/` directories
5. WHEN a checkpoint is missing required keys, THE Model_Loader SHALL raise a descriptive error message
6. THE Model_Loader SHALL handle both torch.Tensor and numpy.ndarray formats for `arcface_eval['weight']`
7. THE Model_Loader SHALL convert `arcface_eval['weight']` to torch.Tensor on the correct device before logit reconstruction

### Requirement 15: Frontend Configuration

**User Story:** As a frontend developer, I want configurable API endpoints, so that I can switch between FastAPI and Flask services during migration.

#### Acceptance Criteria

1. THE Frontend SHALL read the API base URL from the `VITE_API_BASE_URL` environment variable with fallback to `http://127.0.0.1:5000`
2. THE Frontend SHALL support both `/predict` endpoints (FastAPI and Flask) with the same request format
3. THE Frontend SHALL handle error responses from both API services consistently
4. THE Frontend SHALL display politician names fetched from `/classes` endpoint instead of hardcoded `CLASS_MAPPINGS`
5. WHEN the API endpoint is changed via environment variable, THE Frontend SHALL continue to function without code modifications
6. THE Frontend SHALL gracefully handle cases where `/classes` endpoint is unavailable by falling back to predictions without display name mapping

### Requirement 16: Incremental Verification

**User Story:** As a developer, I want to verify changes incrementally, so that I can catch errors early and avoid breaking working functionality.

#### Acceptance Criteria

1. WHEN a code change is made, THE Verification_System SHALL run relevant tests before proceeding to the next change
2. THE Verification_System SHALL verify that `/classes` endpoint returns correct class names after API changes
3. THE Verification_System SHALL verify that `/predict` endpoint produces valid predictions after model loading changes
4. THE Verification_System SHALL verify that TTA produces different results than non-TTA inference
5. THE Verification_System SHALL verify that class indices align between training config and API responses

### Requirement 17: Production Readiness Assessment

**User Story:** As a project manager, I want a production readiness score, so that I can understand the system's deployment maturity.

#### Acceptance Criteria

1. THE Assessment_System SHALL evaluate code consolidation completeness (0-10 scale)
2. THE Assessment_System SHALL evaluate preprocessing consistency (0-10 scale)
3. THE Assessment_System SHALL evaluate test coverage (0-10 scale)
4. THE Assessment_System SHALL evaluate documentation quality (0-10 scale)
5. THE Assessment_System SHALL evaluate error handling robustness (0-10 scale)
6. THE Assessment_System SHALL compute an overall production readiness score as the average of all subscores
7. THE Assessment_System SHALL provide specific recommendations for improving each subscore

### Requirement 18: Maintainability Assessment

**User Story:** As a technical lead, I want a maintainability score, so that I can understand the long-term maintenance burden.

#### Acceptance Criteria

1. THE Assessment_System SHALL evaluate code duplication (0-10 scale, 10 = no duplication)
2. THE Assessment_System SHALL evaluate configuration centralization (0-10 scale, 10 = single config file)
3. THE Assessment_System SHALL evaluate dependency management (0-10 scale, 10 = minimal dependencies)
4. THE Assessment_System SHALL evaluate code modularity (0-10 scale, 10 = highly modular)
5. THE Assessment_System SHALL evaluate documentation coverage (0-10 scale, 10 = comprehensive docs)
6. THE Assessment_System SHALL compute an overall maintainability score as the average of all subscores
7. THE Assessment_System SHALL identify specific technical debt items that reduce maintainability

### Requirement 19: Safe Deletion Verification

**User Story:** As a developer, I want to verify that code is unused before deletion, so that I don't break hidden dependencies.

#### Acceptance Criteria

1. WHEN a file is marked for deletion, THE Verification_System SHALL search for all imports of that file across the codebase
2. WHEN a file is marked for deletion, THE Verification_System SHALL search for all string references to that file path
3. WHEN a file is marked for deletion, THE Verification_System SHALL check DVC and Airflow configurations for references
4. WHEN a file is marked for deletion, THE Verification_System SHALL check Docker and deployment scripts for references
5. WHEN no references are found, THE Verification_System SHALL mark the file as safe to delete
6. WHEN references are found, THE Verification_System SHALL report all reference locations and block deletion

### Requirement 20: Checkpoint Compatibility Testing

**User Story:** As an ML engineer, I want to verify that all existing checkpoints remain loadable, so that I don't lose access to trained models.

#### Acceptance Criteria

1. THE Verification_System SHALL discover all `.pth` checkpoint files in `project_outputs/models/`, `models/saved/`, and `models/`
2. FOR ALL discovered checkpoints, THE Verification_System SHALL attempt to load the checkpoint using the updated model loader
3. FOR ALL discovered checkpoints, THE Verification_System SHALL verify that `model_state_dict` can be loaded into the model architecture
4. FOR ALL ArcFace checkpoints, THE Verification_System SHALL verify that `arcface_eval` dictionary is present and valid
5. WHEN a checkpoint fails to load, THE Verification_System SHALL report the checkpoint path and specific error
6. THE Verification_System SHALL produce a compatibility report listing all checkpoints and their load status

### Requirement 21: Configuration Centralization

**User Story:** As a developer, I want all configuration in a single file, so that I can change hyperparameters without searching through multiple files.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL define all hyperparameters in `training/config.py`
2. THE Training_Pipeline SHALL define all file paths in `training/config.py`
3. THE Training_Pipeline SHALL define all model architectures in `training/config.py`
4. THE Training_Pipeline SHALL define all preprocessing parameters in `training/config.py`
5. WHEN a configuration value is needed, THE Training_Pipeline SHALL import from `training.config` instead of hardcoding values
6. THE FastAPI_Service SHALL import preprocessing parameters from `training.config` to ensure consistency
7. THE Configuration SHALL support environment variable overrides for deployment-specific settings (MODEL_DIR, MODEL_PATH, OUTPUT_DIR)
8. THE Configuration SHALL document all environment variables with their default values and purposes

### Requirement 22: Error Handling Consistency

**User Story:** As a user, I want consistent error messages, so that I can understand what went wrong regardless of which service I use.

#### Acceptance Criteria

1. WHEN no face is detected, THE FastAPI_Service SHALL return HTTP 200 with JSON `{"error": "No face detected"}`
2. WHEN an invalid model name is provided, THE FastAPI_Service SHALL return HTTP 400 with JSON `{"error": "Model 'X' not found. Available: [...]"}`
3. WHEN an invalid image file is uploaded, THE FastAPI_Service SHALL return HTTP 400 with JSON `{"error": "File must be an image"}`
4. WHEN a checkpoint is missing, THE FastAPI_Service SHALL return HTTP 500 with JSON `{"error": "Model checkpoint not found: <path>"}`
5. WHEN an internal error occurs, THE FastAPI_Service SHALL return HTTP 500 with JSON `{"error": "<descriptive message>"}`
6. THE FastAPI_Service SHALL log all errors with stack traces for debugging

### Requirement 23: Documentation Completeness

**User Story:** As a new developer, I want comprehensive documentation, so that I can understand the system without reading all the code.

#### Acceptance Criteria

1. THE Documentation SHALL explain the difference between ArcFace and classifier models
2. THE Documentation SHALL explain the MTCNN face alignment process with margin_ratio parameter
3. THE Documentation SHALL explain the TTA augmentation strategy (5-crop + flip)
4. THE Documentation SHALL explain the ArcFace logit reconstruction process
5. THE Documentation SHALL provide a decision tree for when to use each model architecture
6. THE Documentation SHALL document all API endpoints with request/response examples
7. THE Documentation SHALL document the DVC and Airflow pipeline stages

### Requirement 24: Regression Testing

**User Story:** As a developer, I want automated regression tests, so that I can verify that consolidation doesn't break existing functionality.

#### Acceptance Criteria

1. THE Test_Suite SHALL include a test that verifies `/classes` endpoint returns 16 class names
2. THE Test_Suite SHALL include a test that verifies `/predict` endpoint returns valid predictions for a sample image
3. THE Test_Suite SHALL include a test that verifies TTA produces different results than non-TTA
4. THE Test_Suite SHALL include a test that verifies all three model architectures can be loaded
5. THE Test_Suite SHALL include a test that verifies MTCNN face alignment produces 336×336 images
6. THE Test_Suite SHALL include a test that verifies class indices match between config and model outputs
7. WHEN any test fails, THE Test_Suite SHALL report the specific failure with expected and actual values

### Requirement 25: Deployment Verification

**User Story:** As a DevOps engineer, I want to verify that the consolidated system works in production, so that I can deploy with confidence.

#### Acceptance Criteria

1. THE Deployment_System SHALL verify that Docker containers build successfully with the consolidated code
2. THE Deployment_System SHALL verify that the FastAPI service starts and responds to health checks
3. THE Deployment_System SHALL verify that all model checkpoints are accessible from the Docker container
4. THE Deployment_System SHALL verify that the frontend can connect to the FastAPI service
5. THE Deployment_System SHALL verify that end-to-end prediction works (upload image → receive prediction)
6. WHEN deployment verification fails, THE Deployment_System SHALL report the specific failure and rollback instructions

