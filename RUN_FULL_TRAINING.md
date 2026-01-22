# 🚀 Run FULL Comprehensive Training Pipeline

**Train all 17 models (9 LLM + 4 ViT + 4 VLM) with complete analysis**

---

## 📊 What is the Full Pipeline?

The comprehensive training pipeline trains and compares:

### 9 LLM Models (Text-based)
1. `t5-small`
2. `t5-base`
3. `gpt2`
4. `gpt2-medium`
5. `roberta-base`
6. `roberta-large`
7. `bert-base-uncased`
8. `bert-large-uncased`
9. `distilbert-base-uncased`

### 4 ViT Models (Image-based)
1. `google/vit-base-patch16-224`
2. `google/vit-large-patch16-224`
3. `facebook/deit-base-patch16-224`
4. `facebook/deit-tiny-patch16-224`

### 4 VLM Models (Multimodal)
1. `openai/clip-vit-base-patch32`
2. `openai/clip-vit-large-patch14`
3. `Salesforce/blip-image-captioning-base`
4. `Salesforce/blip2-opt-2.7b`

**Total:** 17 models across 3 modalities

---

## 🎯 Option 1: Google Colab (RECOMMENDED)

### Step 1: Open the Comprehensive Training Notebook

Click this badge to launch directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Solventerritory/FarmFederate-Advisor/blob/feature/multimodal-work/backend/Federated_LLM_ViT_VLM_Comprehensive_Training.ipynb)

Or manually navigate to:
```
https://colab.research.google.com/github/Solventerritory/FarmFederate-Advisor/blob/feature/multimodal-work/backend/Federated_LLM_ViT_VLM_Comprehensive_Training.ipynb
```

### Step 2: Enable GPU

1. Click `Runtime` → `Change runtime type`
2. Select **Accelerator**: `T4 GPU` (or `V100` if available)
3. Click `Save`

### Step 3: Configure Training (Optional)

The notebook comes with default settings, but you can customize in **Section 2**:

```python
FEDERATED_CONFIG = {
    'num_rounds': 10,         # Federated learning rounds
    'num_clients': 5,         # Number of distributed clients
    'local_epochs': 3,        # Epochs per client per round
    'batch_size': 8,          # Batch size
    'learning_rate': 3e-4,    # Learning rate
    'max_length': 128,        # Max text sequence length
}
```

**For faster testing (2-3 hours):**
```python
FEDERATED_CONFIG = {
    'num_rounds': 5,          # Reduced rounds
    'num_clients': 3,         # Fewer clients
    'local_epochs': 2,        # Fewer epochs
    'batch_size': 8,
    'learning_rate': 3e-4,
    'max_length': 128,
}
```

### Step 4: Run All Cells

1. Click `Runtime` → `Run all`
2. Wait for dependencies to install (~2-3 minutes)
3. Training begins automatically
4. Monitor progress in output cells

**Estimated Runtime:**
- **Full training (17 models, 10 rounds):** 4-6 hours with T4 GPU
- **Reduced (5 rounds):** 2-3 hours with T4 GPU
- **With V100 GPU:** ~2-3 hours for full training

### Step 5: Keep Session Alive

Colab may disconnect after 90 minutes of inactivity. To prevent this:

**Option A: Browser Console (F12)**
```javascript
function KeepAlive() {
    console.log("Session alive");
    document.querySelector("colab-connect-button")?.click();
}
setInterval(KeepAlive, 60000); // Every 60 seconds
```

**Option B: Add this cell to notebook:**
```python
from IPython.display import Javascript
display(Javascript('''
function KeepAlive() {
    console.log("Keeping session alive...");
}
setInterval(KeepAlive, 60000);
'''))
```

### Step 6: Save Results

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

**Or save to Google Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')

# Create directory
!mkdir -p /content/drive/MyDrive/FarmFederate_Full_Results

# Copy all results
!cp federated_training_results.json /content/drive/MyDrive/FarmFederate_Full_Results/
!cp COMPREHENSIVE_REPORT.md /content/drive/MyDrive/FarmFederate_Full_Results/
!cp -r plots /content/drive/MyDrive/FarmFederate_Full_Results/

print("✅ All results saved to Google Drive!")
```

---

## 💻 Option 2: Local Machine (GPU Required)

### Requirements
- **GPU:** NVIDIA GPU with 8GB+ VRAM (12GB+ recommended)
- **RAM:** 16GB+ system RAM
- **Storage:** 20GB+ free space
- **CUDA:** CUDA 11.7+ installed

### Step 1: Clone Repository

```bash
git clone -b feature/multimodal-work https://github.com/Solventerritory/FarmFederate-Advisor.git
cd FarmFederate-Advisor/backend
```

### Step 2: Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install transformers>=4.40 datasets peft torch torchvision scikit-learn seaborn matplotlib numpy pandas pillow requests

# If GPU not working, install PyTorch with CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Verify GPU

```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Step 5: Run Training

**Option A: Open Jupyter Notebook**
```bash
jupyter notebook Federated_LLM_ViT_VLM_Comprehensive_Training.ipynb
```
Then run all cells.

**Option B: Convert to Python and Run**
```bash
# Convert notebook to Python script
jupyter nbconvert --to script Federated_LLM_ViT_VLM_Comprehensive_Training.ipynb

# Run the script
python Federated_LLM_ViT_VLM_Comprehensive_Training.py
```

### Step 6: Monitor Progress

Training logs will appear in the console. You can monitor:
- Current model being trained
- Round number and client progress
- Loss and metrics per round
- Plot generation status

---

## 📊 Expected Outputs

After training completes, you'll have:

### 1. Results JSON
**File:** `federated_training_results.json`

Contains metrics for all 17 models:
```json
{
  "llm": {
    "t5-small": {
      "history": [...],
      "final_f1": 0.72,
      "final_acc": 0.74
    },
    "roberta-base": {
      "history": [...],
      "final_f1": 0.75,
      "final_acc": 0.77
    },
    ...
  },
  "vit": {
    "google/vit-base-patch16-224": {...},
    ...
  },
  "vlm": {
    "openai/clip-vit-base-patch32": {...},
    ...
  }
}
```

### 2. Comprehensive Report
**File:** `COMPREHENSIVE_REPORT.md`

Includes:
- Executive summary
- Model performance table (all 17 models)
- Baseline comparison (10 papers)
- Key findings and conclusions
- Recommendations

### 3. Plots Directory
**Directory:** `plots/`

Contains 20+ publication-quality plots (300 DPI):

**Overall Comparisons:**
- `plot_01_overall_f1_comparison.png` - F1 scores for all models
- `plot_02_overall_accuracy_comparison.png` - Accuracy for all models
- `plot_03_overall_training_time.png` - Training time comparison

**Convergence Analysis:**
- `plot_04_llm_convergence.png` - LLM models convergence
- `plot_05_vit_convergence.png` - ViT models convergence
- `plot_06_vlm_convergence.png` - VLM models convergence

**Per-Class Analysis:**
- `plot_07_per_class_f1.png` - F1 per stress type
- `plot_08_per_class_precision.png` - Precision per stress type
- `plot_09_per_class_recall.png` - Recall per stress type

**Confusion Matrices:**
- `plot_10_confusion_matrix_llm_best.png` - Best LLM model
- `plot_11_confusion_matrix_vit_best.png` - Best ViT model
- `plot_12_confusion_matrix_vlm_best.png` - Best VLM model

**Statistical Analysis:**
- `plot_13_model_significance_tests.png` - T-tests between models
- `plot_14_variance_analysis.png` - Performance variance
- `plot_15_client_heterogeneity.png` - Non-IID data effects

**Baseline Comparisons:**
- `plot_16_vs_centralized.png` - Federated vs Centralized
- `plot_17_vs_sota_papers.png` - Comparison with 10 baseline papers

**Additional Plots:**
- Learning rate schedules
- Loss curves
- Communication costs
- Privacy-utility trade-offs

### 4. Comparison Framework (Optional)
Run `python run_comparison.py` to generate 8 additional plots:
- Inter-category comparison (LLM vs ViT vs VLM)
- Intra-category analysis (within each type)
- Centralized vs Federated detailed
- Statistical significance tests

---

## 🎛️ Advanced Configuration

### Reduce Memory Usage

If you encounter OOM errors:

```python
# In Section 2, modify:
FEDERATED_CONFIG = {
    'batch_size': 4,          # Reduced from 8
    'max_length': 96,         # Reduced from 128
    'local_epochs': 2,        # Reduced from 3
}

# In Section 6 (LoRA config):
LORA_CONFIG = {
    'r': 4,                   # Reduced from 8
    'lora_alpha': 16,         # Reduced from 32
}
```

### Train Subset of Models

To train only specific models (faster):

```python
# In Section 3, modify the model lists:

# Only train 2 LLMs
LLM_MODELS = [
    "distilbert-base-uncased",
    "roberta-base"
]

# Only train 2 ViTs
VIT_MODELS = [
    "google/vit-base-patch16-224",
    "facebook/deit-base-patch16-224"
]

# Only train 2 VLMs
VLM_MODELS = [
    "openai/clip-vit-base-patch32",
    "Salesforce/blip-image-captioning-base"
]
```

### Use Different Datasets

The notebook supports multiple dataset sources:

```python
# In data loading section:
USE_AG_NEWS = True           # AG News text dataset
USE_PLANTVILLAGE = True      # PlantVillage image dataset
USE_SYNTHETIC = True         # Synthetic fallback data
MAX_SAMPLES_PER_SOURCE = 1000  # Cap per dataset
```

---

## 🔧 Troubleshooting

### Issue 1: CUDA Out of Memory

**Solution 1:** Reduce batch size
```python
FEDERATED_CONFIG['batch_size'] = 4  # or even 2
```

**Solution 2:** Reduce sequence length
```python
FEDERATED_CONFIG['max_length'] = 96  # or 64
```

**Solution 3:** Train fewer models
```python
# Only train best models
LLM_MODELS = ["roberta-base"]
VIT_MODELS = ["google/vit-base-patch16-224"]
VLM_MODELS = ["Salesforce/blip2-opt-2.7b"]
```

### Issue 2: Colab Session Disconnected

**Solution:** Use the keep-alive script (see Step 5 above)

### Issue 3: Dataset Download Fails

**Solution:** The notebook automatically falls back to synthetic data. Training will continue regardless.

### Issue 4: Slow Training

**Causes:**
- CPU mode (no GPU detected)
- Large batch size
- Too many models

**Solutions:**
- Verify GPU is enabled (see Step 2)
- Reduce batch size and models
- Use V100 instead of T4 on Colab

### Issue 5: LoRA Target Module Error

**Fixed!** The latest version auto-detects target modules for different model architectures.

If you still encounter this, the fallback is to disable LoRA:
```python
# In the model initialization:
USE_LORA = False
```

---

## 📈 Performance Expectations

With default settings (10 rounds, 5 clients):

### Best Expected Results:
- **Best VLM:** BLIP-2 (~0.82 F1, ~0.84 accuracy)
- **Best ViT:** ViT-Large (~0.79 F1, ~0.81 accuracy)
- **Best LLM:** RoBERTa-Base (~0.75 F1, ~0.77 accuracy)

### Per Stress Type (VLM):
- **Disease Risk:** ~0.82 F1 (easiest)
- **Pest Risk:** ~0.78 F1
- **Heat Stress:** ~0.76 F1
- **Nutrient Deficiency:** ~0.70 F1
- **Water Stress:** ~0.68 F1 (hardest)

### Training Time:
- **T4 GPU (Colab free):** 4-6 hours
- **V100 GPU (Colab Pro):** 2-3 hours
- **Local RTX 3090:** 2-3 hours
- **Local RTX 4090:** 1-2 hours

---

## 📚 After Training

### 1. Analyze Results

Open `COMPREHENSIVE_REPORT.md` to see:
- Best performing models
- Model rankings
- Statistical comparisons
- Recommendations

### 2. Generate Additional Plots

```bash
cd backend
python run_comparison.py
```

This creates 8 more comparison plots in `plots/comparison/`

### 3. Use for Research

You now have everything needed for a research paper:
- ✅ Results from 17 models
- ✅ 20+ publication-quality plots (300 DPI)
- ✅ Comparison with 10 baseline papers
- ✅ Statistical significance analysis
- ✅ Complete performance metrics

### 4. Deploy Best Model

Extract the best model weights:
```python
import json

with open('federated_training_results.json', 'r') as f:
    results = json.load(f)

# Find best model
best_model = max(results['vlm'].items(), key=lambda x: x[1]['final_f1'])
print(f"Best model: {best_model[0]} with F1={best_model[1]['final_f1']:.3f}")
```

---

## 🎯 Quick Command Summary

```bash
# Colab: Just click the badge and run all cells
# ↓

# Local:
git clone -b feature/multimodal-work https://github.com/Solventerritory/FarmFederate-Advisor.git
cd FarmFederate-Advisor/backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install transformers datasets peft torch torchvision scikit-learn seaborn
jupyter notebook Federated_LLM_ViT_VLM_Comprehensive_Training.ipynb
# → Run all cells

# Generate comparison plots
python run_comparison.py

# Download results (Colab)
from google.colab import files
files.download('federated_training_results.json')
files.download('COMPREHENSIVE_REPORT.md')
```

---

## 🚀 Ready to Start?

**Click here to launch full training in Colab:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Solventerritory/FarmFederate-Advisor/blob/feature/multimodal-work/backend/Federated_LLM_ViT_VLM_Comprehensive_Training.ipynb)

**Or clone locally:**
```bash
git clone -b feature/multimodal-work https://github.com/Solventerritory/FarmFederate-Advisor.git
cd FarmFederate-Advisor/backend
```

---

**Estimated Time:** 4-6 hours (T4 GPU)
**Cost:** FREE on Google Colab
**Output:** Complete analysis of 17 models with 20+ plots

**Last Updated:** 2026-01-15
**Status:** Production-ready, LoRA issues fixed

