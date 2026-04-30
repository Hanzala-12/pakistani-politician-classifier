# 🚀 UPLOAD THIS NOTEBOOK TO KAGGLE

## 📁 The File You Need

**File:** `notebooks/COMPLETE_TRAINING_PIPELINE.ipynb`

This is a **complete Jupyter notebook** with ALL code in cells:
- ✅ Data collection (web scraping + face detection)
- ✅ Data splitting (train/val/test)
- ✅ Data augmentation
- ✅ Model training (ResNet50 + EfficientNet-B3)
- ✅ Evaluation with **visual outputs**
- ✅ Everything in ONE notebook!

**📊 The notebook SHOWS results in Kaggle:**
- Training curves (loss & accuracy plots)
- Confusion matrices (heatmaps)
- Classification reports (precision, recall, F1)
- Final comparison table
- **AND saves model files for download**

---

## 📋 Step-by-Step Instructions

### Step 1: Download the Notebook
From your local project:
```
notebooks/COMPLETE_TRAINING_PIPELINE.ipynb
```

Or download from GitHub:
https://github.com/Hanzala-12/pakistani-politician-classifier/blob/master/notebooks/COMPLETE_TRAINING_PIPELINE.ipynb

### Step 2: Go to Kaggle
1. Open https://www.kaggle.com/code
2. Click **"New Notebook"**

### Step 3: Upload the Notebook
1. Click **"File"** → **"Upload Notebook"**
2. Select `COMPLETE_TRAINING_PIPELINE.ipynb`
3. Wait for upload to complete

### Step 4: Enable GPU
1. Click **"Settings"** (right sidebar)
2. Under **"Accelerator"**, select:
   - **"GPU T4 x2"** (recommended)
   - OR **"GPU P100"**
3. Click **"Save"**

### Step 5: Run All Cells
1. Click **"Run All"** button at the top
2. OR use keyboard shortcut: **Shift + Enter** for each cell
3. OR from menu: **Run → Run All Cells**

### Step 6: Monitor Progress
Watch the output for each phase:

```
==================================================================
🇵🇰 PAKISTANI POLITICIAN IMAGE CLASSIFIER
==================================================================
📦 Installing required packages...
✅ All packages ready!

🚀 Using device: cuda
   GPU: Tesla T4
   Memory: 15.00 GB

📁 Configuration:
   Data: dataset
   Output: /kaggle/working
   Models: ['resnet50', 'efficientnet_b3']
   Epochs: 20
   Batch Size: 32

==================================================================
📥 PHASE 1: DATA COLLECTION
==================================================================
🌐 Starting web scraping...
📸 Collecting: imran_khan
  ✓ Bing: Imran Khan Pakistan PM face
  ✓ Google: Imran Khan Pakistan PM face
...
```

**Expected Timeline:**
- Phase 1 (Data Collection): 30-60 minutes
- Phase 2 (Splitting): 2-5 minutes
- Phase 3 (Augmentation): 10-20 minutes
- Phase 4 (Dataloaders): 1-2 minutes
- Phase 5 (Model Definitions): 1 minute
- Phase 6 (Training): 3-4 hours ⏱️
- Phase 7 (Evaluation): 5-10 minutes

**Total: 4-6 hours**

### Step 7: Download Results
After you see "TRAINING PIPELINE COMPLETE!", download these folders:

**In Kaggle:**
1. Click **"Output"** tab (right sidebar)
2. You'll see folders:
   - `models/`
   - `plots/`
   - `results/`
3. Click **"Download All"** or download each folder individually

**Files you'll get:**
```
/kaggle/working/
├── models/
│   ├── resnet50_best.pth           (~100 MB)
│   └── efficientnet_b3_best.pth    (~50 MB)
├── plots/
│   ├── resnet50_curves.png
│   ├── resnet50_confusion_matrix.png
│   ├── efficientnet_b3_curves.png
│   └── efficientnet_b3_confusion_matrix.png
└── results/
    ├── model_comparison.csv
    ├── resnet50_classification_report.txt
    └── efficientnet_b3_classification_report.txt
```

---

## 🎯 What Each Cell Does

The notebook has these sections:

1. **Setup & Imports** - Install packages, import libraries
2. **Configuration** - Set hyperparameters
3. **Data Collection** - Scrape images from web
4. **Data Splitting** - Split into train/val/test
5. **Data Augmentation** - Apply transformations
6. **Dataset Class** - Create PyTorch datasets
7. **Model Definitions** - Define CNN architectures
8. **Training Functions** - Training and validation loops
9. **Main Training** - Train all models
10. **Evaluation** - Test and create reports
11. **Main Execution** - Run everything

---

## ⚙️ Optional: Customize Before Running

If you want to change settings, edit these cells:

### Change Number of Epochs
Find the `Config` class cell and change:
```python
EPOCHS = 20  # Change to 30 for better accuracy, or 5 for quick test
```

### Change Models to Train
```python
MODELS_TO_TRAIN = ["resnet50", "efficientnet_b3"]
# Add more: "resnet152", "vgg16", "convnext_base"
```

### Reduce Batch Size (if out of memory)
```python
BATCH_SIZE = 32  # Change to 16 or 8 if GPU runs out of memory
```

---

## 🐛 Troubleshooting

### "Out of Memory" Error
**Solution:** Reduce batch size
1. Find the `Config` class cell
2. Change `BATCH_SIZE = 32` to `BATCH_SIZE = 16`
3. Restart kernel and run again

### Data Collection Fails
**Solution:** This is normal
- Some images will fail to download
- Script continues with available images
- Minimum 60-70 images per class is OK

### Training is Too Slow
**Solution:** Quick test mode
1. Change `EPOCHS = 20` to `EPOCHS = 5`
2. Change `MODELS_TO_TRAIN = ["resnet50"]` (just one model)
3. This will finish in ~1 hour

### Kernel Crashes
**Solution:** 
1. Click "Restart Kernel"
2. Run all cells again
3. Or reduce batch size and try again

---

## ✅ Success Indicators

You'll know it worked when you see:

```
==================================================================
🏆 FINAL RESULTS
==================================================================

Model              Test Accuracy  Macro Precision  Macro Recall  Macro F1
resnet50           92.50%         0.9234           0.9187        0.9210
efficientnet_b3    94.20%         0.9401           0.9378        0.9389

✅ All results saved to: /kaggle/working
   📁 Models: /kaggle/working/models/
   📊 Plots: /kaggle/working/plots/
   📈 Results: /kaggle/working/results/

==================================================================
🎉 TRAINING PIPELINE COMPLETE!
==================================================================

📦 DOWNLOAD THESE FOLDERS:
   1. /kaggle/working/models/  (trained model weights)
   2. /kaggle/working/plots/   (training curves, confusion matrices)
   3. /kaggle/working/results/ (evaluation reports)

💡 TIP: In Kaggle, these are in /kaggle/working/
==================================================================
```

---

## 📊 Expected Results

After training, you should see:
- ✅ Test accuracy > 85% (good)
- ✅ Test accuracy > 90% (excellent)
- ✅ Confusion matrices showing correct predictions
- ✅ Training curves showing convergence

---

## 💡 Tips

1. **Keep browser tab open** - Don't close Kaggle while training
2. **Save version** - Click "Save Version" periodically
3. **Check GPU usage** - Monitor in Kaggle's resource panel
4. **Download immediately** - Download results as soon as done
5. **Read console output** - Check for any errors or warnings

---

## 🆘 Need Help?

If something goes wrong:
1. Read the error message carefully
2. Check which phase failed
3. Try the troubleshooting solutions above
4. Restart kernel and try again
5. Reduce epochs/batch size for testing

---

## 📝 Checklist

Before running:
- [ ] Uploaded `COMPLETE_TRAINING_PIPELINE.ipynb` to Kaggle
- [ ] Enabled GPU accelerator (T4 or P100)
- [ ] Verified internet connection (for data collection)
- [ ] Ready to wait 4-6 hours

After running:
- [ ] Saw "TRAINING PIPELINE COMPLETE!" message
- [ ] Downloaded `models/` folder
- [ ] Downloaded `plots/` folder
- [ ] Downloaded `results/` folder
- [ ] Ready to bring back to local project

---

## 🎉 You're Ready!

**Upload the notebook and click "Run All"!**

Come back in 4-6 hours with the downloaded folders! 🚀

---

**Repository:** https://github.com/Hanzala-12/pakistani-politician-classifier
**Notebook:** `notebooks/COMPLETE_TRAINING_PIPELINE.ipynb`
