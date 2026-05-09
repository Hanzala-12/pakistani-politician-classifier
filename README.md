# Pakistani Politician Image Classifier

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?style=flat-square&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=flat-square&logo=flask)
![React](https://img.shields.io/badge/React-19-61dafb?style=flat-square&logo=react)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

An end-to-end facial recognition system that identifies 16 Pakistani politicians from images. Built with ArcFace metric learning and MTCNN face alignment, served through a Flask/FastAPI backend and a React frontend.

---

## Table of Contents

- [Results](#results)
- [Classified Politicians](#classified-politicians)
- [Data Pipeline](#data-pipeline)
- [Training Configuration](#training-configuration)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [API Documentation](#api-documentation)
- [MLOps Pipeline](#mlops-pipeline)
- [Docker Deployment](#docker-deployment)
- [Known Limitations](#known-limitations)
- [Responsible Use](#responsible-use)
- [Contributing](#contributing)
- [Citation](#citation)

---

## Results

Three models were trained and evaluated on a held-out test set of 136 images.

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| InceptionResNetV1 + ArcFace (CASIA-WebFace) | **96.32%** | 0.9645 | 0.9608 | 0.9607 |
| InceptionResNetV1 + ArcFace (VGGFace2) | **95.59%** | 0.9689 | 0.9504 | 0.9573 |
| ResNet-50 (ImageNet transfer) | **94.85%** | 0.9596 | 0.9534 | 0.9550 |

All three models exceed the 90% target accuracy. ArcFace models train with `s=64.0`, `m=0.3`; validation uses no-margin logits; early stopping (patience 10) selects the best checkpoint. ResNet-50 uses a standard classification head with label smoothing and class weights. All results were obtained with seed 42 and are fully reproducible.

Training curves and confusion matrices are saved to `project_outputs/plots/` after a run.

---

## Classified Politicians

| # | Name | # | Name |
|---|---|---|---|
| 1 | Ahmed Sharif Chaudhry | 9 | Fazlur Rehman |
| 2 | Ahsan Iqbal | 10 | Imran Khan |
| 3 | Altaf Hussain | 11 | Khawaja Asif |
| 4 | Asfandyar Wali | 12 | Maryam Nawaz |
| 5 | Asif Ali Zardari | 13 | Nawaz Sharif |
| 6 | Barrister Gohar | 14 | Pervez Musharraf |
| 7 | Bilawal Bhutto | 15 | Shahbaz Sharif |
| 8 | Chaudhry Shujaat | 16 | Shehryar Afridi |

---

## Data Pipeline

| Stage | Detail |
|---|---|
| Raw images | 3,870 images merged from Bing and Google/DuckDuckGo (1,583 + 2,287) |
| After MTCNN alignment | 1,687 clean, aligned, single-face images |
| Train / Val / Test split | 75% / 15% / 10% stratified |
| Offline augmentation | 3× per class (skips classes with 120+ originals) |
| Training samples (post-augmentation) | ~140–290 images per politician |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Image size | 336 × 336 |
| Batch size | 32 |
| Optimizer | AdamW — `lr=1e-4` (head), `3e-6` (backbone) |
| ArcFace margin | 0.3 |
| ArcFace scale | 64.0 |
| Label smoothing | 0.1 |
| MixUp | α = 0.2, prob = 0.5 |
| Early stopping | patience = 10 epochs |
| TTA | 5-crop + flip (10 views) |

---

## Project Structure

```
pakistani-politician-classifier/
├── notebooks/
│   └── COMPLETE_TRAINING_PIPELINE.ipynb    # Reference notebook (Kaggle)
│
├── training/                               # Production training package
│   ├── config.py                           # Central config — edit here first
│   ├── arcface.py                          # ArcMarginProduct + ArcFaceLoss
│   ├── models.py                           # Model factories + wrappers
│   ├── data_prep.py                        # MTCNN align, pHash dedup, split, augment
│   ├── datasets.py                         # PoliticianDataset + dataloader factory
│   ├── training.py                         # ArcFace + CE training loops
│   ├── evaluate.py                         # Evaluation, confusion matrix, audit
│   ├── predict.py                          # Single-image inference CLI
│   ├── utils.py                            # set_seed, install_package, ensemble
│   └── main.py                             # Pipeline entry point
│
├── src/                                    # Legacy package (MLflow / DVC pipeline)
│   ├── config.py
│   ├── train.py                            # MLflow-tracked training
│   ├── evaluate.py
│   ├── predict.py                          # CLI inference (CE models)
│   ├── augment.py
│   ├── split_dataset.py
│   └── collect_data.py                     # Web scraping
│
├── backend/                                # Flask inference API
├── frontend/                               # React glassmorphism web UI
├── api/                                    # FastAPI REST API
├── docker/                                 # Docker configuration
├── tests/                                  # Unit tests
├── project_outputs/
│   ├── models/                             # Trained .pth checkpoints
│   ├── plots/                              # Training curves + confusion matrices
│   └── results/                            # model_comparison.csv
├── requirements.txt
├── QUICKSTART.md
└── start.sh
```

---

## Installation

**Prerequisites:** Python 3.10+, CUDA 11.7+ (optional), Docker (optional)

```bash
git clone https://github.com/Hanzala-12/pakistani-politician-classifier.git
cd pakistani-politician-classifier

python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

---

## Validation Status

✅ End-to-end validation is documented in [VALIDATION.md](./VALIDATION.md).

- DVC pipeline is structurally sound and executes through the early stages; the remaining failure is a real data constraint, not a tooling issue.
- Docker is production-ready and expects trained artifacts to be mounted or supplied at runtime.
- FastAPI health checks and model discovery are working, with three trained models loaded successfully.
- MLflow, Airflow, and GitHub Actions are present and wired into the project.

The project is deployment-ready, with the remaining caveats coming from data volume and infrastructure setup rather than broken code.

---

## Quick Start

```bash
bash start.sh
# Frontend: http://localhost:5173
# API:      http://localhost:8000
```

Upload an image through the drag-and-drop UI, or call the API directly:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@politician.jpg" \
  -F "model_name=inception_resnet_v1"
```

---

## Training

### Option 1 — Kaggle (recommended)

1. Upload `notebooks/COMPLETE_TRAINING_PIPELINE.ipynb` to a Kaggle notebook.
2. Enable GPU T4 x2 and attach the dataset.
3. Click Run All — outputs land in `/kaggle/working/`.

Alternatively, use the modular package directly in a Kaggle cell:

```python
import sys
sys.path.insert(0, '/kaggle/input/<your-dataset-slug>')
from training.main import main
main()
```

### Option 2 — Local

```bash
# Edit training/config.py first, then:
python training/main.py
```

This runs all 8 stages: install deps → merge raw data → MTCNN align → pHash dedup → split → augment → train → evaluate.

Outputs go to `project_outputs/models/`, `project_outputs/plots/`, and `project_outputs/results/`.

### Option 3 — Legacy MLflow pipeline

```bash
python src/collect_data.py
python src/split_dataset.py
python src/augment.py
python src/train.py
mlflow ui --port 5000
python src/evaluate.py
```

### Single-image prediction

```bash
# ArcFace model
python training/predict.py --image face.jpg --model inception_resnet_v1 --top-k 3

# CE model
python training/predict.py --image face.jpg --model resnet50
```

---

## API Documentation

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server status and loaded models |
| `GET` | `/classes` | List of 16 politician classes |
| `POST` | `/predict` | Single-image prediction (top-3) |
| `POST` | `/predict/batch` | Multi-image batch prediction |

Full request/response formats are documented in `FINAL_IMPLEMENTATION_REPORT.md`.

### API Keys (scraper)

The scraping helper (`scrapper/google_scrapper.py`) uses the Google Custom Search API. Provide keys via environment variables — do not hard-code them into source files.

```bash
# Linux / macOS
export GOOGLE_API_KEY="your_api_key_here"
export GOOGLE_CX="your_custom_search_engine_id_here"
```

```powershell
# Windows PowerShell
$env:GOOGLE_API_KEY = 'your_api_key_here'
$env:GOOGLE_CX = 'your_custom_search_engine_id_here'
```

Alternatively, copy `scrapper/_secrets.example.py` to `scrapper/_secrets.py` and fill in the values. This file is listed in `.gitignore` and will not be committed.

---

## MLOps Pipeline

| Tool | Role |
|---|---|
| DVC | Data versioning for raw images, splits, and preprocessed datasets |
| MLflow | Experiment tracking — logs per-epoch metrics, hyperparameters, and artifacts |
| Airflow | Orchestration DAG: collect → split → augment → train → evaluate |
| GitHub Actions | CI/CD: tests → Docker build & push → deploy to EC2 |

The pipeline honours these environment variables:

- `MLFLOW_TRACKING_URI` — points training and evaluation to a remote MLflow server
- `MLFLOW_EXPERIMENT_NAME` — overrides the default experiment name
- `MLFLOW_REGISTER_MODEL=true` — registers models in MLflow Model Registry after training

To configure a DVC remote and run the reproducible pipeline:

```bash
dvc remote add -f default s3://your-bucket/path/to/dvc-cache
dvc repro
dvc push
```

---

## Docker Deployment

```bash
docker build -f docker/Dockerfile -t politician-classifier:latest .

docker run -d --name politician-api -p 8000:8000 \
  -v $(pwd)/project_outputs/models:/app/models \
  politician-classifier:latest
```

For EC2 deployment:

```bash
bash deploy/ec2_deploy.sh ubuntu@<EC2_HOST> /path/to/key.pem yourdockerhubuser/politician-classifier:latest
```

---

## Data and Models

The `data/`, `dataset/`, `models/`, and `project_outputs/models/` directories contain large artifacts that are excluded from the repository. To obtain the dataset or trained checkpoints, contact:

- Email: yaqoobhanzala@gmail.com
- GitHub: [Hanzala-12](https://github.com/Hanzala-12)

If large files were accidentally committed, remove them from tracking:

```bash
git rm -r --cached data dataset models project_outputs/models
git commit -m "Remove large datasets and model artifacts from repo"
git push origin <branch>
```

To purge from history using BFG Repo-Cleaner:

```bash
bfg --delete-folders data --delete-folders project_outputs/models --delete-folders models
git reflog expire --expire=now --all && git gc --prune=now --aggressive
git push --force
```

Force-pushing rewrites history and affects all collaborators — coordinate before proceeding.

---

## Known Limitations

**Dataset size.** After MTCNN filtering, the dataset contains 1,687 aligned images across 16 identities. Performance on photographs with substantially different lighting, angles, or era may degrade compared to the test set results.

**Small test set.** The held-out set contains 136 images (4–19 samples per class). Per-class F1 scores can shift with a single misclassification and should be read as indicative rather than definitive. Overall accuracy is stable.

**Face-detection dependency.** All models require a detectable, reasonably frontal face. Extreme poses, heavy occlusion, or very low-resolution inputs will cause inference to fail.

**No Pakistani-specific pretraining.** Backbones were pretrained on VGGFace2, CASIA-WebFace, and ImageNet — none include a significant proportion of Pakistani identities. Domain-specific pretraining could further improve accuracy.

---

## Responsible Use

This project uses facial recognition technology applied to images of real public figures. The code, models, and data are provided strictly for educational, research, and demonstration purposes and are not intended for surveillance, profiling, automated decision-making, or any commercial deployment.

Images were sourced from publicly available web pages; no consent was obtained from the individuals in the dataset. Users are responsible for ensuring any use complies with applicable data protection law, platform terms of service, and jurisdictional restrictions on biometric data.

Models trained from this data may exhibit biases due to imbalanced or non-representative training samples. Do not use model outputs to make decisions that materially affect individuals — including decisions related to employment, credit, housing, immigration, or law enforcement.

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes and push to the branch.
4. Open a pull request.

---

## Citation

```bibtex
@misc{pakistani_politician_classifier,
  title  = {Pakistani Politician Image Classifier},
  author = {Hanzala},
  year   = {2026},
  url    = {https://github.com/Hanzala-12/pakistani-politician-classifier}
}
```

---

## Acknowledgements

- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) — MTCNN and InceptionResNetV1 implementations
- PyTorch and torchvision
- Image data collected from publicly available web sources (Bing, Google, DuckDuckGo)
