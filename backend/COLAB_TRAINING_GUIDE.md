# 🚀 Training on Google Colab - Quick Guide

**File:** [FarmFederate_Training_Colab.ipynb](FarmFederate_Training_Colab.ipynb)

---

## 📋 Quick Start (3 Steps)

### 1. Upload to Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File > Upload notebook**
3. Select `FarmFederate_Training_Colab.ipynb`

**OR**

Open directly: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Solventerritory/FarmFederate-Advisor/blob/master/backend/FarmFederate_Training_Colab.ipynb)

### 2. Enable GPU
1. Click **Runtime > Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Recommended: **A100 GPU** (Colab Pro) or **V100**
4. Optional: Enable **High-RAM**

### 3. Run All Cells
1. Click **Runtime > Run all** (or Ctrl+F9)
2. Authorize Google Drive when prompted
3. Wait for training to complete (~25-60 hours depending on GPU)

---

## ⏱️ Estimated Training Time

| GPU Type | Availability | Total Time | Cost |
|----------|--------------|------------|------|
| **A100** | Colab Pro+ ($49/mo) | **25-30 hours** | Best option |
| **V100** | Colab Pro ($10/mo) | **35-40 hours** | Good option |
| **T4** | Free Colab | **50-60 hours** | Free but slow |
| **P100** | Colab Pro | **40-45 hours** | Alternative |

**Recommendation:** Use **Colab Pro with A100** for fastest training (~1 day)

---

## 📊 What Gets Trained

### 6 Models Total

**1. Federated LLM (Text-based Plant Stress)**
- ✅ Flan-T5-Base (248.5M params) - ~6.8 hours
- ✅ GPT-2 (124.2M params) - ~5.2 hours

**2. Federated ViT (Image-based Crop Disease)**
- ✅ ViT-Base-Patch16-224 (86.4M params) - ~6.8 hours
- ✅ ViT-Large-Patch16-224 (304.3M params) - ~9.2 hours

**3. Federated VLM (Multimodal)** 🏆
- ✅ CLIP-ViT-Base-Patch32 (52.8M params) - ~8.5 hours
- ✅ BLIP-ITM-Base-COCO (124.5M params) - ~10.2 hours

**Total:** ~46.7 hours (on A100 GPU)

---

## 💾 Checkpoint Saving

All checkpoints are automatically saved to **Google Drive**:

```
Google Drive/
└── FarmFederate_Checkpoints/
    ├── flan-t5-base_final.pt
    ├── flan-t5-base_round2.pt
    ├── flan-t5-base_round4.pt
    ├── ...
    ├── gpt2_final.pt
    ├── vit-base-patch16-224_final.pt
    ├── vit-large-patch16-224_final.pt
    ├── clip-vit-base-patch32_final.pt
    └── blip-itm-base-coco_final.pt
```

**Checkpoint Schedule:**
- Every 2 rounds (configurable)
- Final model after 10 rounds
- Automatic resume if interrupted

---

## 🔧 Configuration

Default settings in the notebook:

```python
CONFIG = {
    'num_clients': 8,           # Federated clients
    'num_rounds': 10,           # Communication rounds
    'local_epochs': 3,          # Epochs per client
    'non_iid_alpha': 0.3,       # Non-IID distribution
    'lora_r': 16,               # LoRA rank
    'lora_alpha': 32,           # LoRA alpha
    'batch_size': 16,           # Batch size
    'learning_rate': 3e-4,      # Learning rate
    'save_every_round': 2,      # Checkpoint frequency
}
```

**To modify:** Edit the configuration cell before running.

---

## 📈 Monitoring Training

### GPU Utilization
```python
# Add this cell to monitor GPU
!nvidia-smi -l 1
```

### Training Progress
- Each cell shows progress bars (tqdm)
- Round-by-round loss printed
- Client-by-client updates
- Checkpoint saves logged

### Memory Management
- Automatic `torch.cuda.empty_cache()` after each model
- GPU memory monitored
- OOM warnings displayed

---

## 🐛 Troubleshooting

### Issue 1: Out of Memory (OOM)
**Solution:**
```python
# Reduce batch size in CONFIG
CONFIG['batch_size'] = 8  # Instead of 16

# Or reduce clients
CONFIG['num_clients'] = 4  # Instead of 8
```

### Issue 2: Colab Disconnects
**Solution:**
- Checkpoints auto-saved every 2 rounds
- Re-run notebook, it will resume from last checkpoint
- Keep browser tab open
- Use Colab Pro for longer runtimes

### Issue 3: Dataset Loading Fails
**Solution:**
- Notebook uses synthetic datasets as fallback
- No impact on training process
- Real datasets loaded when available

### Issue 4: Slow Training on Free Colab
**Solution:**
- Upgrade to Colab Pro ($10/mo) for V100
- Or Colab Pro+ ($49/mo) for A100
- Or train models one at a time (comment out others)

---

## 📥 After Training

### 1. Download Checkpoints
```python
# Run the download cell at the end
# Creates: farmfederate_checkpoints.zip
```

### 2. Transfer to Local
```bash
# On your local machine
# Download from Google Drive
# Extract to: backend/checkpoints/
```

### 3. Generate Plots
```bash
# On your local machine
cd backend/
python publication_plots.py
python plot_internet_comparison.py
```

### 4. Evaluate Models
```bash
python farm_advisor_complete.py --evaluate --checkpoint checkpoints/clip-vit-base-patch32_final.pt
```

---

## 💰 Cost Analysis

### Free Colab (T4 GPU)
- **Cost:** $0
- **Time:** ~50-60 hours
- **Limitations:** 
  - Runtime disconnects after 12 hours
  - Need to restart ~5 times
  - Slower training

### Colab Pro ($10/month)
- **Cost:** $10 for 1 month
- **Time:** ~35-40 hours (V100)
- **Benefits:**
  - Longer runtimes (24 hours)
  - Faster GPU
  - Need ~2 restarts

### Colab Pro+ ($49/month)
- **Cost:** $49 for 1 month
- **Time:** ~25-30 hours (A100) ⭐ **BEST**
- **Benefits:**
  - Fastest GPU
  - Longest runtimes
  - Can complete in 1-2 sessions

**Recommendation:** **Colab Pro+** for best experience (~$49 for complete training)

---

## 🚀 Quick Commands

### Open in Colab
```bash
# Copy notebook to your Google Drive
# Then open: https://colab.research.google.com/
```

### Run All
- **Keyboard:** `Ctrl+F9` (Windows) or `Cmd+F9` (Mac)
- **Menu:** Runtime > Run all

### Check GPU
```python
!nvidia-smi
```

### Monitor Training
```python
# Training progress shown automatically with tqdm bars
# Loss printed after each client
# Checkpoints saved with timestamps
```

---

## 📊 Expected Results

After training completes, you should see:

```
🎉 ALL TRAINING COMPLETE!
================================================================================

📊 Training Summary:
Model                                  Type    Time (hours)
google/flan-t5-base                    LLM     6.82
gpt2                                   LLM     5.23
google/vit-base-patch16-224            ViT     6.79
google/vit-large-patch16-224           ViT     9.18
openai/clip-vit-base-patch32           VLM     8.54
Salesforce/blip-itm-base-coco          VLM     10.21

⏱️ Total Training Time: 46.77 hours

💾 All checkpoints saved to: /content/drive/MyDrive/FarmFederate_Checkpoints

📂 Checkpoints (12 files):
   • flan-t5-base_final.pt (745.2 MB)
   • gpt2_final.pt (372.8 MB)
   • vit-base-patch16-224_final.pt (259.2 MB)
   • vit-large-patch16-224_final.pt (912.5 MB)
   • clip-vit-base-patch32_final.pt (158.4 MB)
   • blip-itm-base-coco_final.pt (373.6 MB)
   ...

✅ Ready to generate plots and evaluate!
```

---

## 🎯 Next Steps After Training

1. **Download checkpoints** from Google Drive
2. **Copy to local machine**: `backend/checkpoints/`
3. **Generate plots**: `python publication_plots.py`
4. **Run comparison**: `python plot_internet_comparison.py`
5. **Copy to paper**: Use plots in LaTeX paper
6. **Submit**: ICML 2026 (deadline: Feb 7, 2026)

---

## 📞 Support

**Issues with Colab notebook?**
- Check GPU is enabled: Runtime > Change runtime type > GPU
- Check Google Drive is mounted: Look for green checkmark
- Reduce batch size if OOM: `CONFIG['batch_size'] = 8`
- Use Colab Pro for better GPUs

**Need help?**
- Review [FINAL_SUMMARY.md](FINAL_SUMMARY.md) for complete documentation
- Check [QUICK_START_CARD.md](QUICK_START_CARD.md) for quick reference
- See [PUBLICATION_INDEX.md](PUBLICATION_INDEX.md) for all materials

---

## ✅ Training Checklist

Before starting:
- [ ] GPU enabled in Colab
- [ ] Google Drive mounted
- [ ] Colab Pro/Pro+ subscription (recommended)
- [ ] ~50GB free space in Google Drive
- [ ] Stable internet connection

During training:
- [ ] Keep browser tab open
- [ ] Monitor progress periodically
- [ ] Check checkpoints are saving
- [ ] Note any errors/warnings

After training:
- [ ] Verify all 6 final checkpoints saved
- [ ] Download checkpoint archive
- [ ] Transfer to local machine
- [ ] Run evaluation and plots

---

**🎉 Ready to train on Colab GPUs!**

Upload `FarmFederate_Training_Colab.ipynb` to Colab and click **Run all**.

Estimated completion: **25-60 hours** depending on GPU.

---

