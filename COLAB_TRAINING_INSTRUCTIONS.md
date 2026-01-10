# 🚀 Running FarmFederate Training on Google Colab GPU

## Quick Start

1. **Open Colab Notebook:**
   - Upload `FarmFederate_Training_Colab.ipynb` to Google Colab
   - Or create a new notebook and paste the cells below

2. **Enable GPU:**
   - Runtime → Change runtime type → GPU (T4, V100, or A100)
   - Click "Save"

3. **Run Training:**
   - Execute all cells in order
   - Training will take 6-12 hours depending on GPU

---

## 📋 Colab Setup Cells

### Cell 1: Check GPU & Install Dependencies
```python
# Check GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Install dependencies
!pip install -q torch torchvision transformers datasets scikit-learn
!pip install -q matplotlib seaborn pandas pillow timm accelerate
print("✅ Dependencies installed!")
```

### Cell 2: Clone Repository
```python
import os
if not os.path.exists('/content/FarmFederate-Advisor'):
    !git clone https://github.com/Solventerritory/FarmFederate-Advisor.git
    print("✅ Repository cloned")
else:
    print("✅ Repository exists")

os.chdir('/content/FarmFederate-Advisor/backend')
print(f"Working directory: {os.getcwd()}")
```

### Cell 3: Mount Google Drive (Optional - for saving results)
```python
from google.colab import drive
drive.mount('/content/drive')

# Create output directory in Drive
checkpoint_dir = '/content/drive/MyDrive/FarmFederate_Results'
!mkdir -p {checkpoint_dir}
print(f"✅ Results will save to: {checkpoint_dir}")
```

### Cell 4: Run Complete Training
```python
# Run the complete training script
!python federated_complete_training.py

# Copy results to Google Drive (if mounted)
try:
    !cp -r ../results/* /content/drive/MyDrive/FarmFederate_Results/
    !cp -r ../plots/* /content/drive/MyDrive/FarmFederate_Results/plots/
    print("✅ Results copied to Google Drive")
except:
    print("⚠️ Could not copy to Drive (not mounted)")
```

### Cell 5: View Results
```python
# Display final summary
import json
import matplotlib.pyplot as plt
from IPlib.display import Image, display

# Load results
with open('../results/all_results.json', 'r') as f:
    results = json.load(f)

print("="*80)
print("TRAINING COMPLETE - FINAL RESULTS")
print("="*80)

# Show top 5 models
sorted_results = sorted(results, key=lambda x: x['final_metrics']['f1_macro'], reverse=True)
for i, r in enumerate(sorted_results[:5], 1):
    print(f"\n{i}. {r['config']['name']} ({r['config']['model_type'].upper()})")
    print(f"   F1: {r['final_metrics']['f1_macro']:.4f}")
    print(f"   Accuracy: {r['final_metrics']['accuracy']:.4f}")

# Display plots
print("\n📊 Displaying comparison plots...\n")
plot_files = !ls ../plots/*.png
for plot in plot_files[:5]:  # Show first 5 plots
    display(Image(filename=plot))
```

---

## ⏱️ Expected Training Time

| GPU Type | Expected Time | Notes |
|----------|--------------|-------|
| T4 (Free Colab) | 10-12 hours | May need to reconnect |
| V100 (Colab Pro) | 6-8 hours | More stable |
| A100 (Colab Pro+) | 4-6 hours | Fastest option |

## 💾 Training Configuration

- **28 Models**: 13 LLM + 13 ViT + 2 VLM (CLIP)
- **56 Total Runs**: Each model trains in both federated and centralized modes
- **Datasets**: 8 datasets (4 text + 4 image) auto-downloaded
- **Checkpoints**: Saved after each model completes
- **Resume**: Automatically resumes if interrupted

## 📊 Output Files

After training completes:
- `results/all_results.json` - Complete metrics for all models
- `results/{model_name}_results.json` - Individual model results  
- `plots/` - 28+ comparison plots
- `checkpoints/` - Model weights for resuming

## 🔄 If Training is Interrupted

If Colab disconnects (free tier has 12hr limit):

```python
# Just re-run the training cell - it will resume automatically
!python federated_complete_training.py
```

The system automatically:
- Detects completed models
- Skips them
- Continues from where it stopped

## 📤 Download Results

```python
# Zip results for download
!zip -r FarmFederate_Results.zip ../results ../plots ../checkpoints
from google.colab import files
files.download('FarmFederate_Results.zip')
```

---

## 🎯 What You Get

1. **Comparison**: Federated vs Centralized performance for all 28 models
2. **Plots**: 28+ visualizations showing:
   - Model performance comparison
   - Convergence curves
   - Cross-category comparisons (LLM vs ViT vs VLM)
   - Statistical analysis
3. **Best Model**: Identifies which model + paradigm works best
4. **Research Data**: Complete metrics for academic paper

---

## 🆘 Troubleshooting

**Out of Memory Error:**
```python
# Reduce batch size in federated_complete_training.py
# Change lines with batch_size=16 to batch_size=8
```

**Dataset Download Stuck:**
```python
# Clear cache and retry
!rm -rf ~/.cache/huggingface/datasets
!python federated_complete_training.py
```

**GPU Not Available:**
- Runtime → Change runtime type → T4 GPU → Save
- Restart runtime and re-run from Cell 1

---

## 📧 Support

If you encounter issues:
1. Check Colab runtime is GPU-enabled
2. Verify all dependencies installed
3. Check `training_log.txt` for detailed errors
