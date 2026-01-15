# 🚀 Run FarmFederate Training on Google Colab

**Quick Start:** Open the notebook directly in Colab and run all cells!

---

## Option 1: Comprehensive Training Notebook (17 Models)

### ✨ One-Click Launch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Solventerritory/FarmFederate-Advisor/blob/feature/multimodal-work/backend/Federated_LLM_ViT_VLM_Comprehensive_Training.ipynb)

**What it does:**
- Trains 9 LLM + 4 ViT + 4 VLM models
- Federated learning with 5 clients
- Non-IID data split
- Generates 20+ plots
- Exports comprehensive report

**Runtime:** 4-6 hours with T4 GPU

---

## Option 2: Zero-Error Edition (Python Script in Colab)

### 🔧 Manual Setup in Colab

Create a new notebook in Colab and run these cells:

#### Cell 1: Clone Repository
```python
!git clone -b feature/multimodal-work https://github.com/Solventerritory/FarmFederate-Advisor.git
%cd FarmFederate-Advisor/backend
```

#### Cell 2: Install Dependencies
```python
!pip install -q transformers>=4.40 datasets peft torch torchvision scikit-learn seaborn matplotlib
```

#### Cell 3: Run Zero-Error Edition
```python
# Run the zero-error edition script
%run farm_advisor_multimodal_zero_error.py
```

**Runtime:** 1-2 hours with T4 GPU

---

## Option 3: Run Comparison Framework Only

If you already have training results and just want to generate comparison plots:

#### Cell 1: Setup
```python
!git clone -b feature/multimodal-work https://github.com/Solventerritory/FarmFederate-Advisor.git
%cd FarmFederate-Advisor/backend
!pip install -q numpy pandas matplotlib seaborn scikit-learn scipy
```

#### Cell 2: Run Comparison
```python
!python run_comparison.py
```

#### Cell 3: Display Results
```python
from IPython.display import Image, display
import os

# Show all comparison plots
plot_dir = "plots/comparison"
for fname in sorted(os.listdir(plot_dir)):
    if fname.endswith('.png'):
        print(f"\n{'='*60}\n{fname}\n{'='*60}")
        display(Image(os.path.join(plot_dir, fname)))
```

**Runtime:** 2-3 minutes

---

## 📋 Step-by-Step: Comprehensive Training on Colab

### Step 1: Enable GPU
1. Go to `Runtime` → `Change runtime type`
2. Select `T4 GPU` (free tier)
3. Click `Save`

### Step 2: Open Notebook
Click the Colab badge above or manually navigate to:
```
https://colab.research.google.com/github/Solventerritory/FarmFederate-Advisor/blob/feature/multimodal-work/backend/Federated_LLM_ViT_VLM_Comprehensive_Training.ipynb
```

### Step 3: Run Training
1. Click `Runtime` → `Run all`
2. Wait for dependencies to install (~2-3 minutes)
3. Training begins automatically
4. Monitor progress in output cells

### Step 4: Save Results
After training completes, download results:

```python
from google.colab import files
import shutil

# Download results JSON
files.download('federated_training_results.json')

# Download comprehensive report
files.download('COMPREHENSIVE_REPORT.md')

# Zip and download all plots
shutil.make_archive('plots', 'zip', 'plots')
files.download('plots.zip')
```

---

## 🎯 Quick Test Run (5 minutes)

Want to test the system quickly? Modify the notebook to use minimal settings:

```python
# Add this cell at the beginning (after Section 2)
FEDERATED_CONFIG = {
    'num_rounds': 2,        # Reduced from 10
    'num_clients': 3,       # Reduced from 5
    'local_epochs': 1,      # Reduced from 3
    'batch_size': 4,        # Reduced from 8
    'max_samples': 500,     # Cap dataset size
}

# Then modify Section 7 to only train 1 model per type:
LLM_MODELS = ["distilbert-base-uncased"]  # Just 1 LLM
VIT_MODELS = ["google/vit-base-patch16-224"]  # Just 1 ViT
VLM_MODELS = ["openai/clip-vit-base-patch32"]  # Just 1 VLM
```

---

## 📊 Expected Outputs

### Training Results
- `federated_training_results.json` - All model metrics
- `COMPREHENSIVE_REPORT.md` - Complete analysis

### Plots (20+)
- `plot_01_overall_f1_comparison.png`
- `plot_02_overall_accuracy_comparison.png`
- `plot_03_llm_convergence.png`
- `plot_04_vit_convergence.png`
- `plot_05_vlm_convergence.png`
- `plot_06_per_class_f1.png`
- `plot_07_confusion_matrix_llm_best.png`
- `plot_08_confusion_matrix_vit_best.png`
- `plot_09_confusion_matrix_vlm_best.png`
- ... and 11+ more plots

### Comparison Framework (8 plots)
- `01_inter_category_comparison.png`
- `02_intra_category_llm.png`
- `03_intra_category_vit.png`
- `04_intra_category_vlm.png`
- `05_centralized_vs_federated_detailed.png`
- `06_per_class_comparison.png`
- `07_statistical_analysis.png`
- `08_comparison_table.png`

---

## 🔧 Troubleshooting

### Issue 1: Out of Memory
**Solution:** Reduce batch size
```python
FEDERATED_CONFIG['batch_size'] = 4  # or even 2
```

### Issue 2: Session Disconnected
**Solution:** Add keep-alive script
```python
from IPython.display import Javascript
display(Javascript('''
function KeepAlive() {
    console.log("Keeping session alive...");
}
setInterval(KeepAlive, 60000);
'''))
```

### Issue 3: Dataset Download Fails
**Solution:** The notebook automatically falls back to synthetic data, so training will continue regardless.

### Issue 4: GPU Not Available
**Solution:** Check runtime type
```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

## 💡 Pro Tips

### 1. Save to Google Drive
Mount Drive to persist results:
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy results to Drive
!cp -r plots /content/drive/MyDrive/farmfederate_results/
!cp federated_training_results.json /content/drive/MyDrive/
!cp COMPREHENSIVE_REPORT.md /content/drive/MyDrive/
```

### 2. Monitor GPU Usage
```python
!nvidia-smi -l 5  # Update every 5 seconds
```

### 3. Run in Background
Start training and close the browser (reconnect later):
```python
# Training will continue even if disconnected
# Just reopen the notebook to check progress
```

### 4. Speed Up Training
Train fewer models for faster results:
```python
# Only train best models from each category
LLM_MODELS = ["roberta-base"]
VIT_MODELS = ["google/vit-large-patch16-224"]
VLM_MODELS = ["Salesforce/blip2-opt-2.7b"]
```

---

## 📚 Documentation References

- [COLAB_QUICK_START.md](COLAB_QUICK_START.md) - Detailed Colab guide
- [COMPREHENSIVE_TRAINING_README.md](backend/COMPREHENSIVE_TRAINING_README.md) - Training documentation
- [ZERO_ERROR_EDITION_README.md](backend/ZERO_ERROR_EDITION_README.md) - Zero-error edition guide
- [COMPARISON_FRAMEWORK_README.md](backend/COMPARISON_FRAMEWORK_README.md) - Comparison methodology

---

## 🎓 For Research Papers

After training completes, you'll have:
- ✅ Results from 17 models (or fewer if customized)
- ✅ 20+ publication-quality plots (300 DPI)
- ✅ Comparison with 10 baseline papers
- ✅ Statistical significance analysis
- ✅ Complete performance metrics (F1, accuracy, per-class)
- ✅ Federated learning convergence analysis

Perfect for:
- Conference papers
- Journal submissions
- Thesis/dissertation chapters
- Technical reports

---

## ⚡ Quick Commands Cheat Sheet

```bash
# Clone repo
!git clone -b feature/multimodal-work https://github.com/Solventerritory/FarmFederate-Advisor.git

# Install deps
!pip install -q transformers datasets peft torch torchvision scikit-learn seaborn

# Run comprehensive training (notebook)
# → Just click "Run all" in the notebook UI

# Run zero-error edition (script)
%cd FarmFederate-Advisor/backend
%run farm_advisor_multimodal_zero_error.py

# Run comparison only
!python run_comparison.py

# Download results
from google.colab import files
files.download('federated_training_results.json')
```

---

## 🚀 Ready to Start?

**Click here to launch:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Solventerritory/FarmFederate-Advisor/blob/feature/multimodal-work/backend/Federated_LLM_ViT_VLM_Comprehensive_Training.ipynb)

**Estimated Time:** 4-6 hours for full training (17 models)

**Cost:** FREE (using Colab's free T4 GPU)

---

**Last Updated:** 2026-01-15
**Status:** Production-ready, tested on Colab

