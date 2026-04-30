# Tasks

You are an expert ML engineer. Build a complete end-to-end Pakistani Politician 
Image Classification system. Follow every step exactly as specified below.

=======================================================================
PROJECT: Pakistani Politician Face Classification (16 Classes)
STACK: Python, PyTorch, DVC, MLflow, Docker, FastAPI, AWS EC2
=======================================================================

PHASE 0: PROJECT SCAFFOLDING
STEP 0.1 — Create the following exact directory structure:

politician-classifier/
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── airflow/
│   └── dags/
│       └── training_pipeline.py
├── data/
│   └── raw/                     # placeholder only, tracked by DVC
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
├── mlruns/                      # MLflow tracking dir
├── models/
│   └── saved/
├── notebooks/
│   ├── eda.ipynb
│   └── complete_training_pipeline.ipynb  # Standalone notebook for Kaggle
├── src/
│   ├── __init__.py
│   ├── collect_data.py
│   ├── split_dataset.py
│   ├── augment.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── api/
│   ├── main.py
│   ├── model_loader.py
│   └── schemas.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── .dvcignore
├── .gitignore
└── README.md

STEP 0.2 — Create requirements.txt with:
torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
timm
opencv-python
Pillow
scikit-learn
matplotlib
seaborn
pandas
numpy
mlflow
dvc[s3]
fastapi
uvicorn
python-multipart
boto3
apache-airflow
pytest
albumentations
icrawler
tqdm

PHASE 1: DATA COLLECTION
STEP 1.1 — Create src/collect_data.py

The 16 classes to collect are:
  imran_khan, nawaz_sharif, asif_ali_zardari, bilawal_bhutto,
  shahbaz_sharif, maryam_nawaz, fazlur_rehman, asfandyar_wali,
  altaf_hussain, chaudhry_shujaat, pervez_musharraf, 
  shehryar_afridi, khawaja_asif, ahsan_iqbal, 
  barrister_gohar, ahmed_sharif_chaudhry

Use icrawler (BingImageCrawler + GoogleImageCrawler) to download:
  - 120 images per class (buffer for bad images)
  - Save to: data/raw/{class_name}/
  - Use these search queries per politician (their full name + "Pakistan politician face")
  - For ahmed_sharif_chaudhry use query: "Ahmed Sharif Chaudhry ISPR Pakistan"

Code structure:
  from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
  
  POLITICIANS = {
      "imran_khan": ["Imran Khan Pakistan PM face", "Imran Khan PTI"],
      "nawaz_sharif": ["Nawaz Sharif Pakistan PM", "Nawaz Sharif PML-N"],
      ... (build full dict for all 16)
  }
  
  def crawl_images(name, queries, max_per_query=60):
      out_dir = f"data/raw/{name}"
      os.makedirs(out_dir, exist_ok=True)
      for query in queries:
          crawler = BingImageCrawler(storage={"root_dir": out_dir})
          crawler.crawl(keyword=query, max_num=max_per_query)

  Run crawl for all 16 classes.
  
  After crawling, run a face detection filter:
  - Use OpenCV haarcascade or MTCNN to verify each image contains at least 1 face
  - Delete images with no detected face
  - Keep only images where the face region is > 15% of image area
  - Ensure minimum 80 images remain per class after filtering
  - Print a summary report: class name → count

STEP 1.2 — Create src/split_dataset.py

Split data/raw/ into dataset/train/, dataset/val/, dataset/test/
Ratios: 75% train, 15% val, 10% test
Use stratified split (sklearn train_test_split twice)
Preserve class subfolder structure
Print split summary table

PHASE 2: DATA AUGMENTATION
STEP 2.1 — Create src/augment.py

Apply augmentation ONLY to dataset/train/ images.
Use albumentations library.

Required augmentations (combine into a pipeline):
  1. Rotation: RandomRotate90 + Rotate(limit=30)
  2. Flipping: HorizontalFlip(p=0.5)
  3. Brightness: RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3)
  4. Zoom: RandomScale(scale_limit=0.3)
  5. Cropping: RandomResizedCrop(height=224, width=224, scale=(0.7, 1.0))
  6. Also add: GaussianBlur, HueSaturationValue, GridDistortion

Strategy: For each original training image, generate 3 augmented copies.
Save augmented images alongside originals in the same folder.
Name convention: {original_name}_aug_{1,2,3}.jpg

After augmentation, print per-class image counts in train/.

PHASE 3: PYTORCH DATASET & DATALOADER
STEP 3.1 — Inside src/train.py, create:

class PoliticianDataset(Dataset):
    - Load images from folder structure
    - Apply transforms (different for train vs val/test)
    - Return (image_tensor, label_index, image_path)

Transforms:
  TRAIN: Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet mean/std)
  VAL/TEST: Resize(224) → CenterCrop(224) → ToTensor → Normalize

CLASS_NAMES = sorted list of 16 folder names (alphabetical)
Build label→index and index→label mappings.

Create DataLoaders:
  - train: batch_size=32, shuffle=True, num_workers=4
  - val: batch_size=32, shuffle=False
  - test: batch_size=32, shuffle=False

PHASE 4: MODEL DEFINITIONS
STEP 4.1 — Inside src/train.py, create get_model(model_name, num_classes=16):

Support these models using torchvision + timm:

  "resnet50":
    torchvision.models.resnet50(pretrained=True)
    Replace fc: nn.Linear(2048, 16)

  "resnet152":
    torchvision.models.resnet152(pretrained=True)
    Replace fc: nn.Linear(2048, 16)

  "efficientnet_b3":
    timm.create_model('efficientnet_b3', pretrained=True, num_classes=16)

  "vgg16":
    torchvision.models.vgg16(pretrained=True)
    Replace classifier[-1]: nn.Linear(4096, 16)

  "vgg19":
    torchvision.models.vgg19(pretrained=True)
    Replace classifier[-1]: nn.Linear(4096, 16)

  "convnext_base":  ← 5th bonus model
    timm.create_model('convnext_base', pretrained=True, num_classes=16)

For all models:
  - Freeze backbone initially for first 5 epochs (fine-tune only head)
  - Unfreeze all layers after epoch 5 (full fine-tuning)

PHASE 5: TRAINING LOOP
STEP 4.2 — Training configuration (read from params.yaml):

params.yaml contents:
  training:
    epochs: 30
    batch_size: 32
    learning_rate: 0.001
    weight_decay: 0.0001
    scheduler: cosine
    early_stopping_patience: 7
    models_to_train:
      - resnet50
      - resnet152
      - efficientnet_b3
      - vgg16
      - convnext_base

STEP 4.3 — Training loop per model:

For each model in models_to_train:
  1. Initialize model with get_model()
  2. Optimizer: AdamW(lr=0.001, weight_decay=0.0001)
  3. Scheduler: CosineAnnealingLR(T_max=epochs)
  4. Loss: CrossEntropyLoss()
  5. Mixed precision: torch.cuda.amp.GradScaler()

  Per epoch:
    - Train pass: forward, loss, backward, step, scaler
    - Val pass: no grad, compute val_loss and val_accuracy
    - Step scheduler
    - Log to MLflow: train_loss, val_loss, train_acc, val_acc
    - Save best model (by val_accuracy) to models/saved/{model_name}_best.pth
    - Early stopping if val_acc doesn't improve for patience epochs
    - Save training curve data (list of dicts per epoch)

  After training:
    - Plot training vs validation loss curve → save as plots/{model_name}_loss_curve.png
    - Plot training vs validation accuracy curve → save as plots/{model_name}_acc_curve.png

PHASE 6: MLFLOW EXPERIMENT TRACKING
STEP 5.1 — Wrap ALL training in MLflow:

import mlflow
import mlflow.pytorch

mlflow.set_experiment("Pakistani-Politician-Classifier")

with mlflow.start_run(run_name=model_name):
    # Log params
    mlflow.log_params({
        "model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": learning_rate,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealing",
        "num_classes": 16,
        "augmentation": True
    })
    
    # Log metrics each epoch
    mlflow.log_metrics({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc
    }, step=epoch)
    
    # Log artifacts at end
    mlflow.log_artifact(f"plots/{model_name}_loss_curve.png")
    mlflow.log_artifact(f"plots/{model_name}_acc_curve.png")
    mlflow.log_artifact(f"models/saved/{model_name}_best.pth")
    
    # Log model
    mlflow.pytorch.log_model(model, artifact_path="model")

Print MLflow UI command at end: "Run: mlflow ui --port 5000"

PHASE 7: EVALUATION
STEP 6.1 — Create src/evaluate.py

For EACH trained model, run evaluation on test set:

Load best checkpoint → model.eval() → run inference on test DataLoader

Collect: all_preds (list), all_labels (list), all_probs (list), all_paths (list)

Compute and print/save:

  1. Overall Accuracy: 
     accuracy_score(all_labels, all_preds)

  2. Per-class Precision, Recall, F1:
     classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
     Save as: results/{model_name}_classification_report.txt

  3. Confusion Matrix Heatmap:
     Use seaborn.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
     figsize=(16, 14), colormap='Blues'
     Save as: plots/{model_name}_confusion_matrix.png

  4. Top-5 Misclassified Samples:
     Find 5 images where predicted != true with highest confidence
     Create a matplotlib figure with 5 subplots showing:
       - The image
       - Title: "True: X | Pred: Y | Conf: Z%"
     Save as: plots/{model_name}_top5_misclassified.png

  5. Final comparison table:
     Print a pandas DataFrame with columns:
     Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1
     Save as: results/model_comparison.csv

  6. Log all metrics and artifacts to MLflow under the same run.

PHASE 8: DVC SETUP
STEP 7.1 — Initialize DVC and create pipeline:

Run in shell:
  git init (if not done)
  dvc init
  dvc add data/raw/
  dvc add dataset/

Create dvc.yaml with stages:

stages:
  collect_data:
    cmd: python src/collect_data.py
    deps:
      - src/collect_data.py
    outs:
      - data/raw/

  split_data:
    cmd: python src/split_dataset.py
    deps:
      - src/split_dataset.py
      - data/raw/
    outs:
      - dataset/

  augment:
    cmd: python src/augment.py
    deps:
      - src/augment.py
      - dataset/train/
    outs:
      - dataset/train/  (append mode)

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - dataset/
      - params.yaml
    params:
      - training
    outs:
      - models/saved/
      - plots/

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - models/saved/
      - dataset/test/
    outs:
      - results/

Add .dvc files to git, add data/ to .gitignore
Configure DVC remote (S3 or local):
  dvc remote add -d myremote s3://your-bucket/politician-classifier
  OR
  dvc remote add -d localremote /tmp/dvc-storage

PHASE 9: FASTAPI SERVING
STEP 8.1 — Create api/main.py

FastAPI app with these endpoints:

  GET /health
    Returns: {"status": "ok", "models_loaded": [...]}

  GET /classes
    Returns: {"classes": CLASS_NAMES, "count": 16}

  POST /predict
    Input: multipart/form-data with image file + model_name (optional, default: best model)
    Processing:
      1. Read image bytes → PIL Image
      2. Apply val/test transforms
      3. Run inference with loaded model
      4. Return top-3 predictions with confidence scores
    Returns:
    {
      "predicted_class": "imran_khan",
      "confidence": 0.94,
      "top3": [
        {"class": "imran_khan", "confidence": 0.94},
        {"class": "shahbaz_sharif", "confidence": 0.04},
        {"class": "nawaz_sharif", "confidence": 0.02}
      ],
      "model_used": "resnet50",
      "inference_time_ms": 45
    }

  POST /predict/batch
    Input: list of images
    Returns: list of predictions

STEP 8.2 — Create api/model_loader.py

  - Load ALL trained models at startup
  - Cache in dict: {"resnet50": model, "efficientnet_b3": model, ...}
  - Auto-select best model based on results/model_comparison.csv
  - Handle GPU/CPU device automatically

STEP 8.3 — Create api/schemas.py
  Pydantic models for request/response validation

PHASE 10: DOCKER
STEP 9.1 — Create docker/Dockerfile:

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY models/saved/ ./models/saved/
COPY results/ ./results/
COPY params.yaml .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

STEP 9.2 — Create docker/docker-compose.yml:

version: '3.8'
services:
  politician-api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ../models:/app/models
    environment:
      - PYTHONPATH=/app
    restart: unless-stopped

  mlflow:
    image: python:3.10-slim
    command: mlflow server --host 0.0.0.0 --port 5000
    ports:
      - "5000:5000"
    volumes:
      - ../mlruns:/mlruns

PHASE 11: CI/CD PIPELINE
STEP 10.1 — Create .github/workflows/ci-cd.yml:

name: Politician Classifier CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run unit tests
        run: pytest tests/ -v
      
      - name: Lint check
        run: |
          pip install flake8
          flake8 src/ api/ --max-line-length=120

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: docker build -f docker/Dockerfile -t politician-classifier:latest .
      
      - name: Push to Docker Hub
        env:
          DOCKER_USERNAME: ${{ secrets.DOCKER_USERNAME }}
          DOCKER_PASSWORD: ${{ secrets.DOCKER_PASSWORD }}
        run: |
          echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin
          docker tag politician-classifier:latest $DOCKER_USERNAME/politician-classifier:latest
          docker push $DOCKER_USERNAME/politician-classifier:latest

  deploy-to-ec2:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: SSH Deploy to EC2
        env:
          EC2_HOST: ${{ secrets.EC2_HOST }}
          EC2_KEY: ${{ secrets.EC2_PRIVATE_KEY }}
        run: |
          echo "$EC2_KEY" > key.pem && chmod 600 key.pem
          ssh -o StrictHostKeyChecking=no -i key.pem ubuntu@$EC2_HOST "
            docker pull ${{ secrets.DOCKER_USERNAME }}/politician-classifier:latest
            docker stop politician-api || true
            docker rm politician-api || true
            docker run -d --name politician-api \
              -p 8000:8000 \
              --restart unless-stopped \
              ${{ secrets.DOCKER_USERNAME }}/politician-classifier:latest
          "

PHASE 12: TESTS
STEP 11.1 — Create tests/ directory with:

tests/
├── test_dataset.py       # test image loading, transforms, class count
├── test_model.py         # test model output shape = (batch, 16)
├── test_api.py           # test /health, /classes, /predict endpoints
└── conftest.py           # shared fixtures

test_model.py example:
  def test_resnet50_output_shape():
      model = get_model("resnet50", num_classes=16)
      dummy = torch.randn(4, 3, 224, 224)
      out = model(dummy)
      assert out.shape == (4, 16)

test_api.py example:
  from fastapi.testclient import TestClient
  from api.main import app
  client = TestClient(app)
  
  def test_health():
      r = client.get("/health")
      assert r.status_code == 200
      assert r.json()["status"] == "ok"

PHASE 13: STANDALONE KAGGLE NOTEBOOK
STEP 12.1 — Create notebooks/complete_training_pipeline.ipynb

This is a CRITICAL requirement: Create a single, self-contained Jupyter notebook that includes ALL code needed for training, independent of any .py files.

The notebook must be modular and runnable on Kaggle with these sections:

  SECTION 1: Setup & Imports
    - All necessary imports (torch, timm, albumentations, etc.)
    - Device configuration (GPU/CPU detection)
    - Set random seeds for reproducibility

  SECTION 2: Configuration
    - All hyperparameters in one cell (epochs, batch_size, lr, etc.)
    - Class names list (16 politicians)
    - Model selection (which models to train)

  SECTION 3: Data Collection (Optional - can be skipped if data already uploaded)
    - Image crawling code (icrawler)
    - Face detection filtering
    - Can be commented out if using pre-uploaded Kaggle dataset

  SECTION 4: Dataset Class & DataLoaders
    - Complete PoliticianDataset class definition
    - Transform definitions (train/val/test)
    - DataLoader creation
    - Data visualization (show sample images)

  SECTION 5: Model Definitions
    - get_model() function with all 5 models
    - Model architecture summary

  SECTION 6: Training Functions
    - train_one_epoch() function
    - validate() function
    - save_checkpoint() function
    - Includes mixed precision training

  SECTION 7: Training Loop
    - Main training loop for each model
    - Progress bars with tqdm
    - Loss/accuracy tracking
    - Best model saving
    - Training curves plotting (inline)

  SECTION 8: Evaluation
    - Load best checkpoint
    - Test set evaluation
    - Confusion matrix generation
    - Classification report
    - Top-5 misclassified samples visualization

  SECTION 9: Model Export
    - Save final models
    - Export to ONNX (optional)
    - Create model metadata file

  SECTION 10: Results Summary
    - Model comparison table
    - Final metrics display
    - Download links for trained models

Key Requirements:
  ✓ 100% self-contained - no imports from local .py files
  ✓ Can run on Kaggle with GPU
  ✓ All functions defined within notebook
  ✓ Clear markdown explanations between sections
  ✓ Modular cells - can run sections independently
  ✓ Outputs saved to /kaggle/working/ (Kaggle default)
  ✓ Can handle both local paths and Kaggle dataset paths
  ✓ Includes data visualization and progress tracking
  ✓ Model checkpoints saved with timestamps
  ✓ Results exportable as CSV/images

PHASE 14: AIRFLOW DAG (Optional)
STEP 13.1 — Create airflow/dags/training_pipeline.py:

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

dag = DAG(
    'politician_classifier_pipeline',
    schedule_interval='@weekly',
    start_date=datetime(2024, 1, 1),
    catchup=False
)

tasks in order:
  t1: collect_data      → runs src/collect_data.py
  t2: split_dataset     → runs src/split_dataset.py  
  t3: augment           → runs src/augment.py
  t4: train_models      → runs src/train.py
  t5: evaluate_models   → runs src/evaluate.py

Dependencies: t1 >> t2 >> t3 >> t4 >> t5

PHASE 15: README & FINAL REPORT
STEP 14.1 — Create README.md with:

  # Pakistani Politician Image Classifier
  
  Sections:
  1. Project Overview (16 classes, CNN models used)
  2. Dataset Info (sources, counts per class, augmentation)
  3. Models Trained (table: model | params | val_acc)
  4. Results (copy from model_comparison.csv)
  5. How to Run:
     - **Option A: Kaggle Notebook (Recommended for Training)**
       1. Upload notebooks/complete_training_pipeline.ipynb to Kaggle
       2. Enable GPU accelerator
       3. Run all cells
       4. Download trained models from output
     - **Option B: Local Python Scripts**
       - Data collection: python src/collect_data.py
       - Training: python src/train.py
       - Evaluate: python src/evaluate.py
     - API: uvicorn api.main:app --reload
     - Docker: docker-compose up
     - DVC: dvc repro
     - MLflow UI: mlflow ui
  6. API Documentation (endpoint examples with curl)
  7. CI/CD Pipeline diagram description
  8. EC2 Deployment URL

EXECUTION ORDER FOR AGENT
Execute phases in this exact order:

1.  Create all directories and files (Phase 0)
2.  Create standalone Kaggle notebook (Phase 13) - PRIORITY
3.  Install requirements.txt
4.  git init + dvc init
5.  Run collect_data.py → verify ≥80 images per class (OR skip if using Kaggle dataset)
6.  Run split_dataset.py → verify 75/15/10 split
7.  Run augment.py → verify 3x augmentation on train
8.  Run train.py → trains all 5 models with MLflow logging (OR use Kaggle notebook)
9.  Run evaluate.py → generates all plots, reports, confusion matrices
10. Launch FastAPI: uvicorn api.main:app --reload → test all endpoints
11. Build Docker image → docker-compose up → verify
12. Push to GitHub → CI/CD triggers → deploy to EC2
13. Run: mlflow ui → verify all runs visible
14. Run: dvc repro → verify full pipeline reproducible
15. Run: pytest tests/ → all tests pass
16. Generate final README with results table

VALIDATION CHECKPOINTS:
□ All 16 classes have ≥80 images in train/
□ At least 2 models trained (minimum requirement)
□ At least 1 model achieves ≥90% test accuracy
□ MLflow shows experiment runs for all models
□ DVC pipeline stages all complete without error  
□ /predict endpoint returns valid JSON in <2 seconds
□ Docker container runs without error on port 8000
□ GitHub Actions pipeline: all 3 jobs pass (test → build → deploy)
□ Confusion matrix heatmaps saved for all models
□ Top-5 misclassified samples visualized for all models


Tip: Give this prompt in phases — one phase at a time — to your AI agent. Start with Phase 0–2 first, verify the data is good, then proceed to training. Don't attempt all phases in one shot, as data collection quality determines everything downstream.
You said: ok also wrtie a   paragerkap tht mlops working if doxcker is said it should be last i dnt wana buile rebuild image write ill paste it my selfok also wrtie a   paragerkap tht mlops working if doxcker is said it should be last i dnt wana buile rebuild image write ill paste it my self8:19 PMClaude responded: Here's the paragraph you can paste into your report/documentation:Here's the paragraph you can paste into your report/documentation:

MLOps Pipeline Overview
The MLOps pipeline for this project is structured around four core tools: DVC, MLflow, Airflow, and Docker. DVC handles dataset versioning by tracking all raw and split data files, ensuring every experiment is reproducible from the exact same data snapshot without committing large files to Git. MLflow is integrated directly into the training loop to automatically log all hyperparameters, per-epoch metrics (train/val loss and accuracy), model artifacts, and evaluation plots for every model run, making it easy to compare ResNet, EfficientNet, VGG, and ConvNeXt experiments side by side through the MLflow UI. Airflow orchestrates the entire pipeline as a DAG with five sequential tasks — data collection, splitting, augmentation, training, and evaluation — allowing the full workflow to be triggered, monitored, and scheduled automatically without manual intervention. Docker is used purely for model serving, packaging only the trained model weights and the FastAPI inference API into a container that exposes the /predict endpoint; it is intentionally kept separate from the training pipeline so that the image does not need to be rebuilt whenever training is re-run — the container is built once, deployed to AWS EC2, and remains stable while models are updated by mounting the weights directory as a volume.
