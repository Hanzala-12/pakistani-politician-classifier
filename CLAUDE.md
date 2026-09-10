# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

End-to-end facial-recognition system that classifies images of 16 Pakistani politicians. A model is trained on Kaggle/Colab/local GPU, the `.pth` checkpoints are copied into `project_outputs/models/`, and a Flask (or FastAPI) service plus a React UI serve predictions.

Datasets and model weights are **gitignored** and not in the repo. Code that touches `data/`, `dataset/`, `models/`, or `project_outputs/models/` will not run without those artifacts (obtained separately — see README "Data and Models").

## Two parallel stacks — know which one you're in

| Concern | Current / preferred | Legacy (still present, still referenced) |
|---|---|---|
| Training | `training/` package — ArcFace metric learning, `training/config.py` is the single config source | `src/` package — CrossEntropy only, MLflow + DVC, driven by `params.yaml` |
| Serving | `backend/` — Flask, port 5000, what the React app talks to | `api/` — FastAPI, port 8000, imports `from src.train import ...` |
| Pipeline entry | `python training/main.py` (8 stages, one process) | `dvc repro` / individual `python src/*.py` scripts |

`notebooks/COMPLETE_TRAINING_PIPELINE.ipynb` is a thin wrapper that calls `training.main.main()`. Edit the package, not the notebook.

## Commands

```bash
# --- Training (modern pipeline) ---
python training/main.py                 # full pipeline: merge -> MTCNN align -> pHash dedup -> split -> augment -> train -> eval
                                        # edit training/config.py first (EPOCHS, MODELS_TO_TRAIN, USE_ARCFACE, ...)

# Single-image inference
python training/predict.py --image face.jpg --model inception_resnet_v1 --top-k 3   # ArcFace model
python training/predict.py --image face.jpg --model resnet50                        # CE model

# --- Serving ---
bash start.sh                          # pip install backend deps + run Flask on :5000
python backend/app.py                  # Flask directly (:5000)
uvicorn api.main:app --reload --port 8000   # FastAPI alternative

# Frontend (from frontend/)
npm install
npm run dev                            # Vite dev server on :3000, needs VITE_API_BASE_URL (default http://127.0.0.1:5000)
npm run build
npm run lint                           # = tsc --noEmit

# --- Tests / lint (Python) ---
pytest tests/ -v                       # test_model.py + test_dataset.py run standalone; test_api.py needs trained models
pytest tests/test_model.py -v          # single file
pytest tests/test_model.py::test_name -v
flake8 src/ api/ --max-line-length=120 --exclude=__pycache__

# --- Legacy DVC pipeline ---
dvc repro                              # collect_data -> split_data -> augment -> train -> evaluate
mlflow ui --port 5000                  # inspect legacy runs
```

There is no top-level Python dependency manager beyond `requirements.txt` (training/legacy) and `backend/requirements.txt` (serving). On Kaggle the pipeline auto-`pip install`s missing packages.

## Architecture notes that span files

**ArcFace checkpoints have no classifier head.** `FaceEmbeddingModel` (in both `training/models.py` and `backend/model_loader.py`) wraps `InceptionResnetV1` and, when `USE_ARCFACE` is on, returns raw 512-d embeddings only. Class logits are reconstructed at eval/inference time as `scale * normalize(embeddings) @ normalize(W).T`, where `W`/`scale`/`margin` are saved in the checkpoint under the `arcface_eval` key alongside `class_names`. Any code that loads these `.pth` files must go through that path — calling `.head(...)` gives you `nn.Identity`.

**Validation/eval logits are computed WITHOUT the angular margin.** `ArcMarginProduct.forward` returns `s * cosine` when `labels is None`. The margin is only applied to the true class during training.

**The ResNet-50 head layout must match the checkpoint.** `get_model('resnet50')` and `build_classifier_model` both set `model.fc = Sequential(Dropout(0.5), Linear(2048, 16))`. Checkpoints store `fc.1.weight` etc.; a plain `Linear` head will fail to load (this was a real bug — commit `02218b1`).

**Model keys are the contract between UI, backend, and loader:** `inception_resnet_v1` (VGGFace2), `inception_resnet_v1_casia` (CASIA-WebFace), `resnet50`. Defined in `backend/model_loader.py::MODEL_CHOICES` and `frontend/src/App.tsx::AVAILABLE_MODELS`. Checkpoint filenames are `<key>_best.pth`.

**Image size / normalization differ by backbone and by stack.** Modern pipeline: 336×336, mean/std `[0.5]*3` for InceptionResnet, ImageNet stats otherwise (`training/datasets.py::get_transforms`). `tests/conftest.py` and the legacy FastAPI path assume 224×224. Don't "unify" these without checking the checkpoint that will be loaded.

**Two epoch-6 tricks in the training loops:** `train_arcface` rebuilds the AdamW optimizer at epoch 6 to start updating the backbone at `BACKBONE_UNFREEZE_LR` (3e-6); the generic CE loop unfreezes `fc` + `layer4` only at epoch 6 with discriminative LRs. Early-stopping patience is 10 in `training/config.py` but 7 in the legacy `params.yaml`.

**Backend request shape differs from FastAPI.** Flask `/predict` reads form fields `image` (or legacy `file`) and `model`; returns a normalized dict with both `predictions` and `top3`, always HTTP 200 with an `error` field on failure. FastAPI `/predict` reads `file` and `model_name` and raises HTTP 4xx/5xx. The React app targets the Flask shape.

**`frontend/src/App.tsx::CLASS_MAPPINGS` is stale** (contains politicians outside the 16). It's display-only; unknown ids fall back to title-casing. The authoritative class list is `training/config.py::Config.CLASS_NAMES` (16, sorted).

**Data prep is destructive and deletes source images in place** (`filter_images_with_faces`, `deduplicate_phash` call `os.remove`). MTCNN alignment keeps only single-face images, so the aligned set is much smaller than the scraped set — this is why the legacy DVC `split_data` stage can fail with "n_samples=1" on a thin scrape.

**Determinism:** everything seeds through `training/utils.set_seed(42)`.

## Environment variables

- `OUTPUT_DIR` — where training writes `models/`, `plots/`, `results/` (auto-set to `/kaggle/working` or `/content/output` when detected, else `project_outputs`)
- `MODEL_DIR` / `MODEL_PATH` — override checkpoint discovery for the Flask/FastAPI loaders
- `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, `MLFLOW_REGISTER_MODEL` — legacy `src/` training
- `VITE_API_BASE_URL` — frontend → backend base URL
- `GOOGLE_API_KEY`, `GOOGLE_CX` — only for `scrapper/google_scrapper.py`; never hard-code (see `scrapper/_secrets.example.py`)

## CI/CD

`.github/workflows/ci-cd.yml` runs on `main`/`develop`: pytest + flake8, then (on `main`) Docker build → push to Docker Hub → SSH deploy to EC2. Needs repo secrets `DOCKER_USERNAME`/`DOCKER_PASSWORD`, `EC2_HOST`/`EC2_USER`/`EC2_PRIVATE_KEY`. Docker image expects model weights volume-mounted at `/app/models` (not baked in).
