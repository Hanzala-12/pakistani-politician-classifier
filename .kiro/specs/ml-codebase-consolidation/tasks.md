# Implementation Plan: ML Codebase Consolidation

## Overview

This implementation plan consolidates the Pakistani Politician Face Recognition ML codebase by eliminating duplicate training pipelines (`training/` vs `src/`), duplicate inference services (FastAPI vs Flask), inconsistent preprocessing, and hardcoded class labels. The consolidation follows a 14-phase incremental migration approach with verification at each step to preserve all trained model checkpoints and ensure no functionality is broken.

## Tasks

- [ ] 1. Phase 1: Audit and Documentation (No Code Changes)
  - [ ] 1.1 Document all class label definitions and their locations
    - Identify all places where class labels are defined: `training/config.py`, `src/train.py`, `frontend/src/App.tsx`
    - Document the canonical class list and any mismatches
    - _Requirements: 1.1, 1.2, 2.1, 2.5_
  
  - [ ] 1.2 Trace all preprocessing pipelines
    - Document preprocessing in training (`training/datasets.py`)
    - Document preprocessing in evaluation (`training/evaluate.py`)
    - Document preprocessing in inference (`api/main.py`, `backend/model_loader.py`)
    - Identify any differences in transforms, normalization, or image sizes
    - _Requirements: 1.3, 5.1, 5.2, 5.3, 5.4, 5.5_
  
  - [ ] 1.3 Document model loading logic and checkpoint formats
    - Document checkpoint structure in `training/models.py`
    - Document model loading in `backend/model_loader.py`
    - Document ArcFace logit reconstruction logic
    - Identify all checkpoint directories and file formats
    - _Requirements: 1.4, 6.1, 6.2, 6.3, 14.1, 14.2, 14.3_
  
  - [ ] 1.4 Trace DVC and Airflow dependencies
    - Identify all stages in `dvc.yaml` and which modules they reference
    - Identify all tasks in `airflow/dags/training_pipeline.py` and their imports
    - Document which stages/tasks use `src/` vs `training/`
    - _Requirements: 1.5, 11.1, 11.2, 11.3, 11.4, 11.5, 12.1, 12.2, 12.3, 12.4, 12.5_
  
  - [ ] 1.5 Identify all imports from `src/` modules
    - Search for `from src` and `import src` in `api/main.py`
    - Search for `src/` references in `dvc.yaml`
    - Search for `src.*` imports in `airflow/dags/`
    - Search for `src/` references in Docker files and deployment scripts
    - _Requirements: 1.7, 8.1, 8.2, 8.3, 8.7_
  
  - [ ] 1.6 Identify redundant code across pipelines
    - Compare `training/` and `src/` modules for duplicate functionality
    - Compare `api/` (FastAPI) and `backend/` (Flask) for duplicate inference logic
    - Document which code is canonical and which is legacy
    - _Requirements: 1.6, 4.1, 4.2, 4.3, 4.7_
  
  - [ ] 1.7 Produce audit report
    - Create `AUDIT_REPORT.md` with all findings
    - Include class label mismatches, preprocessing differences, import dependencies
    - Include recommendations for migration order
    - _Requirements: 1.8_

- [ ] 2. Checkpoint - Review audit report
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 3. Phase 2: Configuration Centralization
  - [ ] 3.1 Verify and validate `training/config.py`
    - Verify `CLASS_NAMES` is sorted alphabetically
    - Verify `CLASS_NAMES` matches folder names in `data/raw_merged/`
    - Verify `len(CLASS_NAMES) == 16`
    - Add validation logic to raise errors if invariants are violated
    - _Requirements: 2.1, 2.2, 2.5, 21.1, 21.2_
  
  - [ ] 3.2 Add environment variable support
    - Add support for `MODEL_DIR`, `MODEL_PATH`, `OUTPUT_DIR` environment variables
    - Document all environment variables with default values
    - Test environment variable overrides
    - _Requirements: 21.7, 21.8_
  
  - [ ]* 3.3 Write unit tests for configuration
    - Test `CLASS_NAMES` is sorted
    - Test `CLASS_NAMES` length is 16
    - Test environment variable overrides work
    - Test model-specific normalization parameters
    - _Requirements: 2.3, 2.4, 21.3, 21.4, 21.5_

- [ ] 4. Phase 3: API Import Migration
  - [ ] 4.1 Update `api/main.py` imports
    - Change `from src.train import get_transforms, CLASS_NAMES` to `from training.datasets import get_transforms` and `from training.config import config`
    - Update all `CLASS_NAMES` references to `config.CLASS_NAMES`
    - Verify no `from src` or `import src` remains in the file
    - _Requirements: 8.1, 8.2, 8.3_
  
  - [ ] 4.2 Verify `api/model_loader.py` uses correct model loading
    - Ensure it uses `backend.model_loader.get_predictor()` for model loading
    - Verify it doesn't import from `src/` modules
    - _Requirements: 4.1, 4.2, 8.5_
  
  - [ ]* 4.3 Write integration tests for API imports
    - Test `/predict` endpoint works after import changes
    - Test predictions are valid and match expected format
    - Test no import errors occur
    - _Requirements: 8.4, 16.3_

- [ ] 5. Checkpoint - Verify API still works
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Phase 4: Add `/classes` Endpoint
  - [ ] 6.1 Implement `/classes` endpoint in `api/main.py`
    - Add `@app.get("/classes")` endpoint that returns `config.CLASS_NAMES`
    - Return JSON with structure `{"classes": [...], "count": 16}`
    - _Requirements: 3.1, 3.2, 3.3_
  
  - [ ] 6.2 Add `ClassesResponse` schema to `api/schemas.py`
    - Define Pydantic model with `classes: List[str]` and `count: int`
    - _Requirements: 3.3_
  
  - [ ]* 6.3 Write integration tests for `/classes` endpoint
    - Test endpoint returns 200 status code
    - Test response has correct structure
    - Test `count` equals 16
    - Test `classes` list is sorted
    - _Requirements: 3.4, 16.2, 24.1_

- [ ] 7. Phase 5: Frontend Migration
  - [ ] 7.1 Update `frontend/src/App.tsx` to fetch classes dynamically
    - Remove hardcoded `CLASS_MAPPINGS` dictionary
    - Add API call to `/classes` on component mount
    - Store class names in component state
    - _Requirements: 3.5, 3.6, 3.7_
  
  - [ ] 7.2 Add display name formatting for class names
    - Convert underscore-separated names (e.g., `ahmed_sharif_chaudhry`) to title case (e.g., `Ahmed Sharif Chaudhry`)
    - Apply formatting function to all class names received from API
    - Ensure proper capitalization for display in UI
    - _Requirements: 3.7, 15.2_
  
  - [ ] 7.3 Add fallback for when `/classes` is unavailable
    - Handle API errors gracefully
    - Display predictions without class name mapping if endpoint fails
    - Show user-friendly error message when class fetch fails
    - _Requirements: 15.6_
  
  - [ ]* 7.4 Write frontend test for `/classes` fetch failure
    - Mock failed API response for `/classes` endpoint
    - Verify app doesn't crash when fetch fails
    - Verify fallback behavior displays correctly
    - _Requirements: 15.6_
  
  - [ ] 7.5 Configure API base URL via environment variable
    - Read API base URL from `VITE_API_BASE_URL` environment variable
    - Fallback to `http://127.0.0.1:5000` if not set
    - _Requirements: 15.1, 15.5_

- [ ] 8. Phase 6: Model Loading Consistency
  - [ ] 8.1 Verify `backend/model_loader.py` handles all checkpoint formats
    - Verify it loads `model_state_dict` correctly
    - Verify it handles `arcface_eval` dict with `weight` and `scale`
    - Verify it validates `class_names` list
    - _Requirements: 14.1, 14.2, 14.3_
  
  - [ ] 8.2 Add support for both Tensor and ndarray formats
    - Handle `arcface_eval['weight']` as both `torch.Tensor` and `numpy.ndarray`
    - Convert to Tensor on correct device before logit reconstruction
    - _Requirements: 14.6, 14.7_
  
  - [ ] 8.3 Add checkpoint discovery across multiple directories
    - Search `MODEL_DIR` env var, then `project_outputs/models/`, `models/saved/`, `models/`
    - _Requirements: 14.4_
  
  - [ ] 8.4 Add descriptive error messages for checkpoint loading failures
    - Raise clear errors when required keys are missing
    - Include checkpoint path in error messages
    - _Requirements: 14.5, 22.4_
  
  - [ ]* 8.5 Write unit tests for model loading
    - Test loading checkpoints with all required keys
    - Test loading ArcFace checkpoints with `arcface_eval`
    - Test loading classifier checkpoints without `arcface_eval`
    - Test error handling for missing keys
    - Test Tensor and ndarray format handling
    - _Requirements: 7.3, 7.4, 20.1, 20.2, 20.3, 20.4, 20.5_

- [ ] 9. Checkpoint - Verify model loading works
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Phase 7: MTCNN Robustness
  - [ ] 10.1 Add EXIF orientation handling
    - Apply `ImageOps.exif_transpose()` before face detection
    - Handle EXIF parsing failures gracefully
    - _Requirements: 9.5_
  
  - [ ] 10.2 Add upscale fallback for face detection
    - If MTCNN detects zero faces, upscale image 2× (capped at 2000px)
    - Retry detection on upscaled image
    - Scale bounding boxes and landmarks back to original coordinates
    - _Requirements: 9.3, 9.4_
  
  - [ ] 10.3 Update MTCNN parameters for better detection
    - Set `min_face_size=20` for small face detection
    - Set `thresholds=[0.5, 0.6, 0.7]` for improved detection
    - _Requirements: 9.6_
  
  - [ ] 10.4 Add multi-face handling
    - When multiple faces detected, select largest by bounding box area
    - _Requirements: 9.2_
  
  - [ ] 10.5 Add error response for no face detected
    - Return HTTP 200 with `{"error": "No face detected"}` when no face found
    - _Requirements: 9.1, 22.1_
  
  - [ ]* 10.6 Write integration tests for MTCNN robustness
    - Test no face detected returns error message
    - Test multiple faces selects largest
    - Test EXIF orientation handling
    - Test small face detection
    - _Requirements: 9.7_

- [ ] 11. Phase 8: TTA Implementation
  - [ ] 11.1 Verify TTA implementation in `training/datasets.py`
    - Verify 5-crop (4 corners + center) implementation
    - Verify horizontal flip for each crop (total 10 views)
    - _Requirements: 10.1, 10.3_
  
  - [ ] 11.2 Add TTA support to `/predict` endpoint
    - Add optional `tta` boolean parameter to request (default: `false`)
    - Apply TTA transforms when enabled
    - Average softmax probabilities across all 10 views
    - _Requirements: 4.4, 4.5, 10.2, 10.4_
  
  - [ ] 11.3 Ensure TTA uses center crop when disabled
    - When `tta=false`, use only center crop (no augmentation)
    - Document that API defaults to `tta=false` for speed (user can opt-in for accuracy)
    - Note: This differs from `training/config.py` where `USE_TTA=True` for evaluation
    - Add comment explaining the deliberate accuracy vs speed tradeoff
    - _Requirements: 10.5_
  
  - [ ]* 11.4 Write integration tests for TTA
    - Test TTA produces 10 views
    - Test TTA averages probabilities correctly
    - Test TTA produces different results than non-TTA
    - Test TTA works with all model architectures
    - Test default `tta=false` behavior
    - _Requirements: 16.4_

- [ ] 12. Phase 9: DVC Pipeline Migration
  - [ ] 12.1 Update `dvc.yaml` to reference `training/` modules
    - Change `collect_data` stage to call `training.data_prep`
    - Change `split_data` stage to call `training.data_prep.split_dataset`
    - Change `augment` stage to call `training.data_prep.run_offline_augmentation`
    - Change `train` stage to call `training.main`
    - Change `evaluate` stage to call `training.evaluate`
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_
  
  - [ ] 12.2 Verify DVC dependencies and outputs remain valid
    - Ensure all file paths in dependencies and outputs are correct
    - _Requirements: 11.8_
  
  - [ ]* 12.3 Test DVC pipeline execution
    - Run `dvc repro` in dry-run mode to verify stages parse correctly
    - Verify no import errors occur
    - _Requirements: 11.7_

- [ ] 13. Phase 10: Airflow DAG Migration
  - [ ] 13.1 Update `airflow/dags/training_pipeline.py` imports
    - Change imports from `src.*` to `training.*`
    - Update `collect_data` task to import from `training.data_prep`
    - Update `split_dataset` task to import from `training.data_prep`
    - Update `augment_data` task to import from `training.data_prep`
    - Update `train_models` task to import from `training.main`
    - Update `evaluate_models` task to import from `training.evaluate`
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_
  
  - [ ]* 13.2 Test Airflow DAG parsing
    - Run `airflow dags list` to verify DAG parses without errors
    - Verify no import errors occur
    - _Requirements: 12.6_

- [ ] 14. Checkpoint - Verify orchestration works
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 15. Phase 11: Inference Accuracy Verification
  - [ ] 15.1 Create verification script
    - Select 10 random test images from `dataset/test/`
    - Run evaluation using `training/evaluate.py` for each image
    - Run inference using `/predict` endpoint for each image
    - Compare predicted class and confidence scores
    - Report differences > 0.01 confidence tolerance
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6_
  
  - [ ] 15.2 Run verification and document results
    - Execute verification script
    - Document any discrepancies found
    - Investigate and fix any prediction mismatches
    - _Requirements: 16.1_

- [ ] 16. Phase 12: Deprecation and Documentation
  - [ ] 16.1 Create `src/README.md` with deprecation notice
    - Add "DEPRECATED" header
    - Document migration path from `src/` to `training/`
    - Include safe deletion checklist
    - Note: Actual deletion will occur in a future cleanup PR after grace period
    - _Requirements: 8.4, 8.6_
  
  - [ ] 16.2 Create `backend/README.md` with Flask deprecation notice
    - Add "DEPRECATED" header
    - Document that FastAPI service is now canonical
    - Note: Actual deletion will occur in a future cleanup PR after grace period
    - _Requirements: 4.7_
  
  - [ ] 16.3 Update main `README.md` with consolidation summary
    - Document the consolidation changes
    - Update architecture diagrams if present
    - Update setup instructions to reference `training/` instead of `src/`
    - Document TTA default differences (training eval vs API inference)
    - Note that `src/` and `backend/` are deprecated but not yet deleted
    - _Requirements: 23.1, 23.2, 23.3, 23.4, 23.5, 23.6, 23.7_
  
  - [ ] 16.4 Verify no active references to deprecated code
    - Search for `from src` in all active code
    - Search for `src/` references in Docker files
    - Search for `src/` references in deployment scripts
    - Search for `src/` references in CI/CD pipelines
    - _Requirements: 8.7, 19.1, 19.2, 19.3, 19.4_

- [ ] 17. Phase 13: Regression Testing
  - [ ]* 17.1 Write comprehensive regression test suite
    - Test all API endpoints (`/classes`, `/predict`, `/predict/batch`, `/health`)
    - Test all three model architectures load successfully
    - Test preprocessing consistency across models
    - Test TTA produces valid predictions
    - Test error handling for edge cases
    - _Requirements: 24.2, 24.3, 24.4, 24.5, 24.6, 24.7_
  
  - [ ] 17.2 Run full test suite and document results
    - Execute `pytest tests/ -v`
    - Test all API endpoints manually
    - Test frontend with API backend
    - Test model loading for all checkpoints
    - Document any failures and fix them
    - _Requirements: 16.5_

- [ ] 18. Checkpoint - Verify all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 19. Phase 14: Deployment Verification
  - [ ] 19.1 Build and test Docker image
    - Build Docker image with consolidated code
    - Start Docker container
    - Verify container starts successfully
    - _Requirements: 25.1, 25.2_
  
  - [ ] 19.2 Test deployed service endpoints
    - Test `/health` endpoint returns status and loaded models
    - Test `/classes` endpoint returns correct class list
    - Test `/predict` endpoint with sample images
    - Verify all model checkpoints are accessible
    - _Requirements: 25.3_
  
  - [ ] 19.3 Test frontend integration with deployed service
    - Verify frontend can connect to deployed API
    - Test end-to-end prediction flow (upload image → receive prediction)
    - Verify predictions display correctly
    - _Requirements: 25.4, 25.5_
  
  - [ ] 19.4 Monitor logs for errors
    - Check Docker logs for any errors or warnings
    - Verify no import errors or missing dependencies
    - _Requirements: 25.6_

- [ ] 20. Final checkpoint - Consolidation complete
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional testing tasks and can be skipped for faster consolidation
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation and provide opportunities to address issues
- The consolidation preserves all model checkpoints and ensures no functionality is broken
- All changes are verified before proceeding to the next phase
- The migration follows a low-risk incremental approach with rollback plans at each phase
