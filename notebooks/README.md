# 📓 Kaggle Training Notebook

## 🎯 THE ONLY FILE YOU NEED

**File:** `COMPLETE_TRAINING_PIPELINE.ipynb`

This is a **complete, standalone Jupyter notebook** that does EVERYTHING:

### ✅ What It Does:
1. **Data Collection** - Scrapes 16 politician images from web
2. **Face Detection** - Filters images with OpenCV
3. **Data Splitting** - 75% train, 15% val, 10% test
4. **Data Augmentation** - 2x multiplication with transformations
5. **Model Training** - Trains ResNet50 + EfficientNet-B3
6. **Evaluation** - Creates confusion matrices, reports, metrics

### 📦 What You Get:
After running (4-6 hours), download these from `/kaggle/working/`:

```
models/
├── resnet50_best.pth           (~100 MB)
└── efficientnet_b3_best.pth    (~50 MB)

plots/
├── resnet50_curves.png
├── resnet50_confusion_matrix.png
├── efficientnet_b3_curves.png
└── efficientnet_b3_confusion_matrix.png

results/
├── model_comparison.csv
├── resnet50_classification_report.txt
└── efficientnet_b3_classification_report.txt
```

---

## 🚀 How to Use

### Step 1: Upload to Kaggle
1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Upload `COMPLETE_TRAINING_PIPELINE.ipynb`

### Step 2: Enable GPU
1. Settings → Accelerator → GPU T4 x2
2. Save

### Step 3: Run All Cells
1. Click "Run All"
2. Wait 4-6 hours
3. Download results from Output tab

---

## 📋 Full Instructions

See: `../UPLOAD_TO_KAGGLE.md` for detailed step-by-step guide

---

## ✅ Success

You'll know it worked when you see:
```
🎉 TRAINING PIPELINE COMPLETE!
```

Then download the 3 folders (models, plots, results) and bring them back!

---

**Repository:** https://github.com/Hanzala-12/pakistani-politician-classifier
