# 🚀 Quick Start Guide

Get up and running with the Pakistani Politician Classifier in minutes!

---

## 📋 Prerequisites

```bash
# Check Python version (3.10+ required)
python --version

# Check if CUDA is available (optional, for GPU training)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## ⚡ Quick Setup (5 minutes)

### 1. Clone Repository
```bash
git clone https://github.com/Hanzala-12/pakistani-politician-classifier.git
cd pakistani-politician-classifier
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🎯 Choose Your Path

### Path A: Kaggle Training (Easiest - Recommended)

**Best for:** Quick results, no local GPU needed

#### Option 1 — Notebook (COMPLETE_TRAINING_PIPELINE.ipynb)
1. Upload `notebooks/COMPLETE_TRAINING_PIPELINE.ipynb` to Kaggle
2. Enable GPU: Settings → Accelerator → GPU T4 x2
3. Click **Run All**

#### Option 2 — Modular package (recommended for reproducibility)
```python
# In a Kaggle notebook cell:
import sys, subprocess
# Mount your dataset, then:
sys.path.insert(0, '/kaggle/input/<your-dataset-slug>')

from training.main import main
main()
```
Or run from the terminal accelerator:
```bash
cd /kaggle/working && python training/main.py
```

**Download results from:**
- `/kaggle/working/models/`  — trained `.pth` checkpoints
- `/kaggle/working/plots/`   — training curves + confusion matrices
- `/kaggle/working/results/` — `model_comparison.csv`

---

### Path B: Local Training — New Pipeline (Recommended)

**Best for:** Full pipeline with ArcFace, dedup, and offline augmentation

#### One command runs everything:
```bash
# From project root
python training/main.py
```

This executes all 8 stages automatically:
1. Install missing packages
2. Load data from Kaggle datasets (or confirm `data/` exists)
3. Merge `data/raw` + `data/raw2` → `data/raw_merged`
4. MTCNN face alignment + pHash deduplication
5. Stratified train/val/test split
6. Offline augmentation for under-represented classes
7. Train all models in `config.MODELS_TO_TRAIN`
8. Evaluate + export `model_comparison.csv`

**Output:** `project_outputs/models/`, `project_outputs/plots/`, `project_outputs/results/`

#### Configure before running:
Edit `training/config.py`:
```python
EPOCHS = 10                  # Quick test
MODELS_TO_TRAIN = ["resnet50"]  # Single model
USE_ARCFACE = False          # Disable ArcFace for speed
```

---

### Path B (Legacy): Local Training via src/

**Best for:** MLflow experiment tracking, DVC pipeline

```bash
python src/collect_data.py   # ~30-60 min → data/raw/
python src/split_dataset.py  # ~2-5 min  → dataset/
python src/augment.py        # ~10-20 min
python src/train.py          # ~4-8 hours
mlflow ui --port 5000        # Monitor training
python src/evaluate.py       # Evaluate + reports
```

---

### Path C: Use Pre-trained Models (Fastest)

**Best for:** Testing API, deployment

1. **Download pre-trained models** (if available)
2. **Place in:** `models/saved/`
3. **Start API:**
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
4. **Test:**
   ```bash
   curl http://localhost:8000/health
   ```

---

## 🧪 Quick Test

### Test Model Architecture
```bash
pytest tests/test_model.py -v
```

### Test Dataset Loading
```bash
pytest tests/test_dataset.py -v
```

### Test API (requires trained models)
```bash
# Start API
uvicorn api.main:app --reload --port 8000

# In another terminal
pytest tests/test_api.py -v
```

---

## 🐳 Docker Quick Start

### Build and Run
```bash
# Build image
docker build -f docker/Dockerfile -t politician-classifier:latest .

# Run container
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  politician-classifier:latest

# Test
curl http://localhost:8000/health
```

### Using Docker Compose
```bash
docker-compose -f docker/docker-compose.yml up -d

# Services:
# - API: http://localhost:8000
# - MLflow: http://localhost:5000
```

---

## 📊 Quick Prediction

### New Pipeline CLI (supports ArcFace models)
```bash
# ArcFace model (inception_resnet_v1)
python training/predict.py \
  --image path/to/politician.jpg \
  --model inception_resnet_v1 \
  --top-k 3

# Generic CE model (resnet50)
python training/predict.py \
  --image path/to/politician.jpg \
  --model resnet50 \
  --checkpoint project_outputs/models/resnet50_best.pth
```

### Python (new training/ package)
```python
from training.predict import load_model, predict_image

model = load_model("inception_resnet_v1", "project_outputs/models/inception_resnet_v1_best.pth")
results = predict_image(model, "politician.jpg", top_k=3)
for name, conf in results:
    print(f"{name}: {conf*100:.1f}%")
```

### API Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@politician.jpg"
```

### Legacy CLI (src/)
```bash
python src/predict.py \
  --image path/to/politician.jpg \
  --model resnet50 \
  --top-k 3
```

---

## 🔧 Configuration

### Edit Training Parameters
Edit `params.yaml`:
```yaml
training:
  epochs: 30              # Reduce for faster training
  batch_size: 32          # Reduce if out of memory
  learning_rate: 0.001
  models_to_train:
    - resnet50            # Start with one model
    # - efficientnet_b3   # Add more later
```

### Edit Kaggle Script
Edit `notebooks/kaggle_training_complete.py`:
```python
class Config:
    EPOCHS = 10  # Quick test
    MODELS_TO_TRAIN = ["resnet50"]  # Single model
```

---

## 📈 Monitor Training

### MLflow UI
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

**View:**
- Training/validation loss curves
- Accuracy metrics
- Model parameters
- Artifacts (plots, models)

### TensorBoard (Alternative)
```bash
# If you add TensorBoard logging
tensorboard --logdir=runs
```

---

## 🐛 Troubleshooting

### Out of Memory Error
```python
# Reduce batch size in params.yaml
batch_size: 16  # or 8
```

### CUDA Not Available
```bash
# Check PyTorch installation
python -c "import torch; print(torch.__version__)"

# Reinstall with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Data Collection Fails
```bash
# Check internet connection
# Try reducing max_per_query in src/collect_data.py
max_per_query = 30  # Instead of 60
```

---

## 📚 Next Steps

After quick start:

1. **Read full documentation:** `README.md`
2. **Configure the pipeline:** edit `training/config.py`
3. **Run the full pipeline:** `python training/main.py`
4. **Predict on new images:** `python training/predict.py --image face.jpg`
5. **Run tests:** `pytest tests/`
6. **Deploy:** follow Docker or EC2 deployment guide
7. **Legacy src/:** still fully functional for MLflow/DVC workflows

---

## 🎯 Common Workflows

### Workflow 1: Train Single Model Quickly (new pipeline)
```bash
# Edit training/config.py:
#   EPOCHS = 5
#   MODELS_TO_TRAIN = ["resnet50"]
#   USE_ARCFACE = False
python training/main.py
```

### Workflow 2: Full ArcFace Pipeline
```bash
# Default config already enables ArcFace + inception_resnet_v1
python training/main.py
```

### Workflow 3: Legacy Pipeline with DVC
```bash
dvc repro
```

### Workflow 4: API Development
```bash
# Terminal 1: Start API
uvicorn api.main:app --reload

# Terminal 2: Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@test.jpg"
```

### Workflow 5: Continuous Retraining
```bash
python src/collect_data.py   # Collect new images
python training/main.py      # Full pipeline (align, split, train, eval)
```

---

## 💡 Tips

- **Start small:** Train 1 model with 5 epochs first
- **Use Kaggle:** Free GPU, no setup needed
- **Monitor MLflow:** Track all experiments
- **Save checkpoints:** Don't lose training progress
- **Test API early:** Verify deployment works
- **Use DVC:** Version your data and models
- **Read logs:** Check console output for errors

---

## 🆘 Get Help

- **GitHub Issues:** https://github.com/Hanzala-12/pakistani-politician-classifier/issues
- **Documentation:** `README.md`
- **Code Comments:** Check source files
- **Stack Overflow:** Tag with `pytorch`, `mlops`

---

## ✅ Checklist

- [ ] Repository cloned
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Data collected (or dataset ready)
- [ ] First model trained
- [ ] API tested
- [ ] Results reviewed

---

**Ready to start? Pick your path above and begin! 🚀**
