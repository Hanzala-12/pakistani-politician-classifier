# Pakistani Politician Image Classifier

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?style=flat-square&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=flat-square&logo=flask)
![React](https://img.shields.io/badge/React-19-61dafb?style=flat-square&logo=react)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

An end-to-end facial recognition system that identifies 16 Pakistani politicians using ArcFace metric learning, MTCNN face alignment, and a full MLOps pipeline.

---

## Table of Contents

- [Overview](#overview)
- [Results](#models--results)
- [Data Pipeline](#data-pipeline)
- [Classified Politicians](#classified-politicians)
- [Training Configuration](#training-configuration)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [API Documentation](#api-documentation)
- [MLOps Pipeline](#mlops-pipeline)
- [Docker Deployment](#docker-deployment)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)
- [Citation](#citation)

---

## Overview

This project classifies 16 Pakistani politicians from facial images using three deep-learning approaches:

| Model | Test Accuracy |
|---|---|
| InceptionResNetV1 + ArcFace (VGGFace2) | **97.06%** |
| InceptionResNetV1 + ArcFace (CASIA-WebFace) | **96.32%** |
| ResNet-50 (ImageNet transfer learning) | **96.32%** |

All three models exceed the 90% target accuracy. The pipeline includes MTCNN face detection, landmark-based alignment, perceptual-hash deduplication, stratified splitting, offline augmentation, and a complete evaluation suite — including TTA, per-class metrics, confusion matrices, and top-5 misclassified samples.

A React-based glassmorphism frontend and a Flask inference API make the system production-ready.

---

## Data Pipeline

| Stage | Detail |
|---|---|
| Raw images merged | 3,870 images from Bing and Google/DuckDuckGo (1,583 + 2,287) |
| After MTCNN alignment | 1,687 clean, aligned, single-face images |
| Training / Validation / Test | 75% / 15% / 10% stratified split |
| Offline augmentation | 3x per class (skips classes with 120+ originals) |
| Training samples (post-augmentation) | ~140–290 images per politician |

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

## Models & Results

### Test-Set Evaluation (held-out, 136 images)

| Model | Test Acc | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| InceptionResNetV1 (VGGFace2 + ArcFace) | **97.06%** | 0.9804 | 0.9730 | 0.9748 |
| InceptionResNetV1 (CASIA-WebFace + ArcFace) | **96.32%** | 0.9655 | 0.9697 | 0.9653 |
| ResNet-50 (ImageNet transfer) | **96.32%** | 0.9687 | 0.9673 | 0.9662 |

ArcFace models train with `s=64.0`, `m=0.3`; validation uses no-margin logits; early stopping (patience 10) selects the best checkpoint. ResNet-50 uses a standard classification head with label smoothing and class weights.

All results were obtained from a notebook run with seed 42. The notebook is fully reproducible.

## Plots

The repository includes training curves and confusion matrices in `project_outputs/plots/`.
Representative visuals from the experiments are shown below.

- InceptionResNetV1 (VGGFace2) — training curves

  ![InceptionResNetV1 curves](project_outputs/plots/inception_resnet_v1_curves.png)

- InceptionResNetV1 (VGGFace2) — confusion matrix

  ![InceptionResNetV1 confusion](project_outputs/plots/inception_resnet_v1_confusion_matrix.png)

- InceptionResNetV1 (CASIA) — training curves

  ![InceptionResNetV1 CASIA curves](project_outputs/plots/inception_resnet_v1_casia_curves.png)

- InceptionResNetV1 (CASIA) — confusion matrix

  ![InceptionResNetV1 CASIA confusion](project_outputs/plots/inception_resnet_v1_casia_confusion_matrix.png)

- ResNet-50 — training curves

  ![ResNet50 curves](project_outputs/plots/resnet50_curves.png)

- ResNet-50 — confusion matrix

  ![ResNet50 confusion](project_outputs/plots/resnet50_confusion_matrix.png)


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
│   └── COMPLETE_TRAINING_PIPELINE.ipynb   # Full training notebook
├── backend/                                # Flask inference API
├── frontend/                               # React glassmorphism web UI
├── src/                                    # Source modules: split, augment, train, eval, predict
├── api/                                    # FastAPI REST API
├── docker/                                 # Docker configuration
├── tests/                                  # Unit tests
├── project_outputs/
│   ├── models/                             # Trained model checkpoints (.pth)
│   ├── plots/                              # Training curves, confusion matrices
│   └── results/                            # model_comparison.csv
├── requirements.txt
├── start.sh
└── FINAL_IMPLEMENTATION_REPORT.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.7+ (optional, GPU acceleration)
- Docker (optional)

### Setup

```bash
git clone https://github.com/Hanzala-12/pakistani-politician-classifier.git
cd pakistani-politician-classifier

python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### Quick Start

```bash
bash start.sh
# Frontend: http://localhost:5173
# API:      http://localhost:8000
```

Upload an image through the drag-and-drop UI or call the API directly:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@politician.jpg" \
  -F "model_name=inception_resnet_v1"
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

### API & Search Engine Keys

The project includes a scraping helper (`scrapper/google_scrapper.py`) that can use the
Google Custom Search API. Do NOT hard-code API keys or engine IDs into source files.

Recommended ways to provide keys:

- Environment variables (preferred):

  - Linux / macOS:

    ```bash
    export GOOGLE_API_KEY="your_api_key_here"
    export GOOGLE_CX="your_custom_search_engine_id_here"
    ```

  - Windows PowerShell:

    ```powershell
    $env:GOOGLE_API_KEY = 'your_api_key_here'
    $env:GOOGLE_CX = 'your_custom_search_engine_id_here'
    ```

- Local, git-ignored secrets file (alternative):

  Copy `scrapper/_secrets.example.py` to `scrapper/_secrets.py` and fill in your keys.
  The repository already lists `scrapper/_secrets.py` in `.gitignore`, so the real file
  will not be committed.

If you discover API keys or other secrets already present in the repo, remove them and
rotate the keys immediately.

---

## MLOps Pipeline

| Tool | Role |
|---|---|
| **DVC** | Data versioning for raw images, splits, and preprocessed datasets |
| **MLflow** | Experiment tracking — logs per-epoch metrics, hyperparameters, and artifacts |
| **Airflow** | Orchestration DAG: collect → split → augment → train → evaluate |
| **GitHub Actions** | CI/CD: tests → Docker build & push → deploy to EC2 |

---

## Docker Deployment

```bash
docker build -f docker/Dockerfile -t politician-classifier:latest .

docker run -d --name politician-api -p 8000:8000 \
  -v $(pwd)/project_outputs/models:/app/models \
  politician-classifier:latest
```

---

## Data & Models (Private)

The `data/`, `dataset/`, `models/`, and `project_outputs/models/` directories contain large or sensitive
artifacts (raw datasets and trained model checkpoints). These directories are intentionally excluded from
the repository and should not be pushed to GitHub.

If you need the dataset or trained models, contact the project owner:

- Email: yaqoobhanzala@gmail.com
- GitHub: https://github.com/Hanzala-12

To ensure these are not accidentally pushed, `.gitignore` contains entries for the directories listed above.
If you have already committed large files, remove them from tracking with:

```bash
git rm -r --cached data dataset models project_outputs/models
git commit -m "Remove large datasets and model artifacts from repo"
git push origin <branch>
```

To purge large files from repository history, use `git filter-repo` or the BFG Repo-Cleaner. Example (BFG):

```bash
bfg --delete-folders data --delete-folders project_outputs/models --delete-folders models
git reflog expire --expire=now --all && git gc --prune=now --aggressive
git push --force
```

Warning: Rewriting history with force-push affects all collaborators — coordinate before proceeding.

## Known Limitations

**Dataset size.** After MTCNN filtering, the dataset contains 1,687 aligned images across 16 identities. The model generalises well on the held-out test set, but real-world performance on photographs with substantially different lighting, angles, or era may degrade. Expanding the dataset with additional verified images would further improve robustness.

**Small test set.** The held-out set contains 136 images (4–19 samples per class). Individual per-class F1 scores can shift with a single misclassification; they should be read as indicative rather than definitive. Overall accuracy is stable.

**Face-detection dependency.** All models require a detectable, reasonably frontal face. Extreme poses, heavy occlusion, or very low-resolution inputs will cause inference to fail.

**No Pakistani-specific pretraining.** Backbones were pretrained on VGGFace2, CASIA-WebFace, and ImageNet — none include a significant proportion of Pakistani identities. Domain-specific pretraining could further improve accuracy.

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes and push to the branch.
4. Open a Pull Request.

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

## ‼️ Responsible Use

This project uses facial recognition technology applied to images of real public figures. The code, models, and data in this repository are provided strictly for educational, research, and demonstration purposes only. They are not intended for operational surveillance, profiling, or commercial deployment.

This system must not be used for surveillance, profiling, automated decision-making that affects people's rights, or any activity that violates privacy, civil liberties, or applicable law. No consent was obtained from the individuals whose images were collected for the dataset; however, the images were sourced from publicly available web pages.

Users are responsible for ensuring that any use of the code, models, or data complies with applicable laws, regulations, and institutional policies governing biometric and personal data. This includes data protection legislation, platform terms of service, and any jurisdictional restrictions on collecting or processing biometric identifiers.

Models trained from this data may exhibit biases due to imbalanced, non-representative, or noisy training samples. Do not use the outputs of these models to make decisions that materially affect individuals (for example, related to employment, credit, housing, immigration, or law enforcement).

Before deploying or sharing systems derived from this repository, perform an ethical and legal review, obtain necessary approvals and consents where required, and implement appropriate safeguards (including human oversight, transparency about limitations, and opt-out mechanisms). If you are unsure whether a proposed use is appropriate, consult legal counsel, institutional review boards, or privacy experts.

Consider the ethical implications carefully and document any additional safeguards you apply.

## Acknowledgements

- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) — MTCNN and InceptionResNetV1 implementations
- PyTorch and torchvision
- Image data collected from publicly available web sources (Bing, Google, DuckDuckGo)
