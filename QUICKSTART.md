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

1. **Upload to Kaggle:**
   - Go to https://www.kaggle.com/code
   - Click "New Notebook"
   - Upload `notebooks/kaggle_training_complete.py`
   - Or copy-paste the code

2. **Configure:**
   - Enable GPU: Settings → Accelerator → GPU
   - Upload your dataset or use data collection code

3. **Run:**
   - Click "Run All"
   - Wait for training to complete (~2-4 hours for 2 models)

4. **Download Results:**
   - Models: `/kaggle/working/models/`
   - Plots: `/kaggle/working/plots/`
   - Results: `/kaggle/working/results/`

---

### Path B: Local Training (Full Control)

**Best for:** Local development, custom modifications

#### Step 1: Collect Data
```bash
python src/collect_data.py
```
**Time:** ~30-60 minutes  
**Output:** `data/raw/` with 16 politician folders

#### Step 2: Split Dataset
```bash
python src/split_dataset.py
```
**Time:** ~2-5 minutes  
**Output:** `dataset/train/`, `dataset/val/`, `dataset/test/`

#### Step 3: Augment Training Data
```bash
python src/augment.py
```
**Time:** ~10-20 minutes  
**Output:** Augmented images in `dataset/train/`

#### Step 4: Train Models
```bash
python src/train.py
```
**Time:** ~4-8 hours (depends on GPU)  
**Output:** `models/saved/*.pth`, `plots/*.png`

#### Step 5: View Training Progress
```bash
# In another terminal
mlflow ui --port 5000
# Open http://localhost:5000
```

#### Step 6: Evaluate Models
```bash
python src/evaluate.py
```
**Time:** ~5-10 minutes  
**Output:** `results/model_comparison.csv`, confusion matrices

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

### CLI Prediction
```bash
python src/predict.py \
  --image path/to/politician.jpg \
  --model resnet50 \
  --top-k 3
```

### API Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@politician.jpg"
```

### Python Script
```python
import torch
from PIL import Image
from src.train import get_model, get_transforms

# Load model
model = get_model("resnet50", num_classes=16)
checkpoint = torch.load("models/saved/resnet50_best.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load and transform image
transform = get_transforms('test')
image = Image.open("politician.jpg").convert('RGB')
image_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    outputs = model(image_tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    top_prob, top_idx = torch.topk(probs, 1)
    
print(f"Predicted: {checkpoint['class_names'][top_idx]}")
print(f"Confidence: {top_prob.item()*100:.2f}%")
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
2. **Check project status:** `PROJECT_STATUS.md`
3. **Review implementation details:** `antasks.md`
4. **Explore code:** `src/` directory
5. **Run tests:** `pytest tests/`
6. **Deploy:** Follow Docker or EC2 deployment guide

---

## 🎯 Common Workflows

### Workflow 1: Train Single Model Quickly
```bash
# Edit params.yaml - set epochs: 5, models: [resnet50]
python src/train.py
python src/evaluate.py
```

### Workflow 2: Full Pipeline with DVC
```bash
dvc repro
```

### Workflow 3: API Development
```bash
# Terminal 1: Start API
uvicorn api.main:app --reload

# Terminal 2: Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@test.jpg"
```

### Workflow 4: Continuous Training
```bash
# Collect new data
python src/collect_data.py

# Retrain
python src/train.py

# Evaluate
python src/evaluate.py

# Update API (models auto-reload)
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
